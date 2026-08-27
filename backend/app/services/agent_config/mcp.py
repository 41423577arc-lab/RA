import ipaddress
import re
import socket
from datetime import datetime, timezone
from urllib.parse import urlsplit
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models.database import (
    ConfigSecret,
    McpServerDefinition,
    McpServerRevision,
    ToolMappingDefinition,
    ToolMappingRevision,
)
from app.services.agent_config.secrets import SecretStore
from app.services.agent_config.snapshot import canonical_hash


DEFAULT_MCP_SERVER_ID = "00000000-0000-0000-0000-000000000101"
DEFAULT_MCP_SERVER_REVISION_ID = "00000000-0000-0000-0000-000000000102"
DEFAULT_TOOL_MAPPINGS = {
    "identity.find_candidates": {
        "name": "Identity candidate lookup",
        "remote_tool_name": "find_entity_candidates",
        "allowed_nodes": ["intake_agent", "intake_identity_update"],
    },
    "projects.search": {
        "name": "Internal project search",
        "remote_tool_name": "search_projects",
        "allowed_nodes": ["research_pipeline"],
    },
}
LOGICAL_TOOL_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,99}$")


class McpConfigService:
    def __init__(self, session: Session, settings: Settings):
        self.session = session
        self.settings = settings
        self.secrets = SecretStore(session, settings)

    def ensure_defaults(
        self, tenant_id: str
    ) -> tuple[McpServerRevision, dict[str, tuple[ToolMappingRevision, list[str]]]]:
        definition = self.session.get(McpServerDefinition, DEFAULT_MCP_SERVER_ID)
        if definition is None:
            definition = McpServerDefinition(
                id=DEFAULT_MCP_SERVER_ID,
                tenant_id=tenant_id,
                name="Default project MCP",
                slug="default-project-mcp",
            )
            self.session.add(definition)
            self.session.flush()
        revision = self.session.get(McpServerRevision, DEFAULT_MCP_SERVER_REVISION_ID)
        payload = {
            "transport": "streamable_http",
            "url": self.settings.mcp_server_url.rstrip("/"),
            "authentication_type": "none",
            "secret_ref": None,
            "timeout_seconds": 10,
        }
        if revision is None:
            revision = McpServerRevision(
                id=DEFAULT_MCP_SERVER_REVISION_ID,
                mcp_server_definition_id=definition.id,
                version=1,
                **payload,
                config_hash=canonical_hash(payload),
                published_at=datetime.now(timezone.utc),
            )
            self.session.add(revision)
            self.session.flush()
            definition.active_revision_id = revision.id

        mappings: dict[str, tuple[ToolMappingRevision, list[str]]] = {}
        for logical_key, item in DEFAULT_TOOL_MAPPINGS.items():
            mapping_definition = self.session.scalar(
                select(ToolMappingDefinition).where(
                    ToolMappingDefinition.tenant_id == tenant_id,
                    ToolMappingDefinition.logical_tool_key == logical_key,
                )
            )
            if mapping_definition is None:
                mapping_definition = ToolMappingDefinition(
                    tenant_id=tenant_id,
                    name=item["name"],
                    logical_tool_key=logical_key,
                )
                self.session.add(mapping_definition)
                self.session.flush()
            mapping_revision = (
                self.session.get(
                    ToolMappingRevision, mapping_definition.active_revision_id
                )
                if mapping_definition.active_revision_id
                else None
            )
            if mapping_revision is None:
                mapping_payload = {
                    "mcp_server_revision_id": revision.id,
                    "remote_tool_name": item["remote_tool_name"],
                    "adapter_key": "declarative",
                    "input_mapping": {},
                    "output_mapping": {},
                    "timeout_seconds": 10,
                }
                mapping_revision = ToolMappingRevision(
                    tool_mapping_definition_id=mapping_definition.id,
                    version=1,
                    **mapping_payload,
                    config_hash=canonical_hash(mapping_payload),
                    published_at=datetime.now(timezone.utc),
                )
                self.session.add(mapping_revision)
                self.session.flush()
                mapping_definition.active_revision_id = mapping_revision.id
            mappings[logical_key] = (mapping_revision, list(item["allowed_nodes"]))
        self.session.flush()
        return revision, mappings

    def create_server(
        self,
        *,
        tenant_id: str,
        name: str,
        slug: str,
        url: str,
        authentication_type: str,
        api_token: str | None,
        secret_ref: str | None,
        timeout_seconds: int,
    ) -> tuple[McpServerDefinition, McpServerRevision]:
        if self.session.scalar(
            select(McpServerDefinition).where(
                McpServerDefinition.tenant_id == tenant_id,
                McpServerDefinition.slug == slug,
            )
        ):
            raise ValueError(f"MCP server slug already exists: {slug}")
        if not re.fullmatch(r"[a-z][a-z0-9-]{1,62}[a-z0-9]", slug):
            raise ValueError("Slug must contain 3-64 lowercase letters, numbers, or hyphens")
        definition = McpServerDefinition(
            tenant_id=tenant_id, name=name.strip(), slug=slug
        )
        self.session.add(definition)
        self.session.flush()
        revision = self._new_server_revision(
            definition,
            version=1,
            url=url,
            authentication_type=authentication_type,
            api_token=api_token,
            secret_ref=secret_ref,
            timeout_seconds=timeout_seconds,
        )
        definition.active_revision_id = revision.id
        self.session.commit()
        return definition, revision

    def revise_server(self, definition_id: str, **values) -> McpServerRevision:
        definition = self._server_definition(definition_id)
        version = self._next_version(
            McpServerRevision, "mcp_server_definition_id", definition.id
        )
        revision = self._new_server_revision(definition, version=version, **values)
        definition.active_revision_id = revision.id
        self.session.commit()
        return revision

    def create_mapping(
        self,
        *,
        tenant_id: str,
        name: str,
        logical_tool_key: str,
        **values,
    ) -> tuple[ToolMappingDefinition, ToolMappingRevision]:
        self._validate_logical_key(logical_tool_key)
        if self.session.scalar(
            select(ToolMappingDefinition).where(
                ToolMappingDefinition.tenant_id == tenant_id,
                ToolMappingDefinition.logical_tool_key == logical_tool_key,
            )
        ):
            raise ValueError(f"Logical tool already exists: {logical_tool_key}")
        definition = ToolMappingDefinition(
            tenant_id=tenant_id,
            name=name.strip(),
            logical_tool_key=logical_tool_key,
        )
        self.session.add(definition)
        self.session.flush()
        revision = self._new_mapping_revision(definition, version=1, **values)
        definition.active_revision_id = revision.id
        self.session.commit()
        return definition, revision

    def revise_mapping(self, definition_id: str, **values) -> ToolMappingRevision:
        definition = self._mapping_definition(definition_id)
        revision = self._new_mapping_revision(
            definition,
            version=self._next_version(
                ToolMappingRevision, "tool_mapping_definition_id", definition.id
            ),
            **values,
        )
        definition.active_revision_id = revision.id
        self.session.commit()
        return revision

    def resolve_mapping_revision(self, revision_id: str) -> dict:
        revision = self.session.get(ToolMappingRevision, revision_id)
        if revision is None or revision.status != "PUBLISHED":
            raise KeyError(f"Tool mapping revision not found: {revision_id}")
        definition = self._mapping_definition(revision.tool_mapping_definition_id)
        payload = self._mapping_payload(revision)
        if canonical_hash(payload) != revision.config_hash:
            raise ValueError(f"Tool mapping revision failed integrity validation: {revision.id}")
        server = self.resolve_server_revision(revision.mcp_server_revision_id)
        return {
            "logical_tool_key": definition.logical_tool_key,
            "mapping_revision_id": revision.id,
            "provider": "mcp",
            "server_revision_id": revision.mcp_server_revision_id,
            "remote_tool_name": revision.remote_tool_name,
            "adapter_key": revision.adapter_key,
            "timeout_seconds": revision.timeout_seconds,
            "input_mapping": revision.input_mapping,
            "output_mapping": revision.output_mapping,
            "server": server,
        }

    def resolve_server_revision(self, revision_id: str) -> dict:
        revision = self.session.get(McpServerRevision, revision_id)
        if revision is None or revision.status != "PUBLISHED":
            raise KeyError(f"MCP server revision not found: {revision_id}")
        payload = self._server_payload(revision)
        if canonical_hash(payload) != revision.config_hash:
            raise ValueError(f"MCP server revision failed integrity validation: {revision.id}")
        return {"revision_id": revision.id, **payload}

    def _new_server_revision(
        self,
        definition: McpServerDefinition,
        *,
        version: int,
        url: str,
        authentication_type: str,
        api_token: str | None,
        secret_ref: str | None,
        timeout_seconds: int,
    ) -> McpServerRevision:
        normalized_url = validate_mcp_url(
            url,
            allow_private=self.settings.agent_allow_private_mcp_urls,
            trusted_hosts=self.settings.agent_trusted_mcp_hosts,
        )
        if authentication_type not in {"none", "bearer"}:
            raise ValueError(f"Unsupported MCP authentication type: {authentication_type}")
        if not 1 <= timeout_seconds <= 120:
            raise ValueError("MCP timeout must be between 1 and 120 seconds")
        reference = None
        if authentication_type == "none":
            if api_token or secret_ref:
                raise ValueError("Unauthenticated MCP servers cannot have a secret")
        else:
            if bool(api_token) == bool(secret_ref):
                raise ValueError("Provide exactly one of api_token or secret_ref")
            if secret_ref:
                self._validate_secret_ref(secret_ref, definition.tenant_id)
                reference = secret_ref
            else:
                secret = self.secrets.create(
                    definition.tenant_id,
                    f"mcp-{definition.slug}-{uuid4().hex[:8]}",
                    api_token or "",
                )
                reference = f"db:{secret.id}"
        payload = {
            "transport": "streamable_http",
            "url": normalized_url,
            "authentication_type": authentication_type,
            "secret_ref": reference,
            "timeout_seconds": timeout_seconds,
        }
        revision = McpServerRevision(
            mcp_server_definition_id=definition.id,
            version=version,
            **payload,
            config_hash=canonical_hash(payload),
            published_at=datetime.now(timezone.utc),
        )
        self.session.add(revision)
        self.session.flush()
        return revision

    def _new_mapping_revision(
        self,
        definition: ToolMappingDefinition,
        *,
        version: int,
        mcp_server_revision_id: str,
        remote_tool_name: str,
        adapter_key: str,
        input_mapping: dict,
        output_mapping: dict,
        timeout_seconds: int,
    ) -> ToolMappingRevision:
        self.resolve_server_revision(mcp_server_revision_id)
        if not NAME_PATTERN.fullmatch(remote_tool_name):
            raise ValueError("Remote MCP tool name is invalid")
        if adapter_key != "declarative":
            raise ValueError("Unknown code adapter; deploy and register it before configuration")
        self._validate_mapping(input_mapping)
        self._validate_mapping(output_mapping)
        if not 1 <= timeout_seconds <= 120:
            raise ValueError("Tool timeout must be between 1 and 120 seconds")
        payload = {
            "mcp_server_revision_id": mcp_server_revision_id,
            "remote_tool_name": remote_tool_name,
            "adapter_key": adapter_key,
            "input_mapping": input_mapping,
            "output_mapping": output_mapping,
            "timeout_seconds": timeout_seconds,
        }
        revision = ToolMappingRevision(
            tool_mapping_definition_id=definition.id,
            version=version,
            **payload,
            config_hash=canonical_hash(payload),
            published_at=datetime.now(timezone.utc),
        )
        self.session.add(revision)
        self.session.flush()
        return revision

    @staticmethod
    def _validate_mapping(mapping: dict) -> None:
        unknown = set(mapping) - {"rename", "constants"}
        if unknown:
            raise ValueError(f"Unsupported declarative mapping fields: {', '.join(sorted(unknown))}")
        rename = mapping.get("rename", {})
        constants = mapping.get("constants", {})
        if not isinstance(rename, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            and NAME_PATTERN.fullmatch(key) and NAME_PATTERN.fullmatch(value)
            for key, value in rename.items()
        ):
            raise ValueError("Mapping rename must contain simple field-name pairs")
        if not isinstance(constants, dict) or not all(
            isinstance(key, str) and NAME_PATTERN.fullmatch(key) for key in constants
        ):
            raise ValueError("Mapping constants must use simple field names")

    def _validate_secret_ref(self, secret_ref: str, tenant_id: str) -> None:
        if not secret_ref.startswith("db:"):
            raise ValueError("MCP secrets must use encrypted db: references")
        secret = self.session.get(ConfigSecret, secret_ref.removeprefix("db:"))
        if (
            secret is None
            or secret.tenant_id != tenant_id
            or secret.status != "ACTIVE"
        ):
            raise ValueError(f"Secret reference is not active: {secret_ref}")

    @staticmethod
    def _validate_logical_key(key: str) -> None:
        if not LOGICAL_TOOL_PATTERN.fullmatch(key):
            raise ValueError("Logical tool key must use dot-separated lowercase names")

    def _server_definition(self, definition_id: str) -> McpServerDefinition:
        definition = self.session.get(McpServerDefinition, definition_id)
        if definition is None or definition.status != "ACTIVE":
            raise KeyError(f"MCP server not found: {definition_id}")
        return definition

    def _mapping_definition(self, definition_id: str) -> ToolMappingDefinition:
        definition = self.session.get(ToolMappingDefinition, definition_id)
        if definition is None or definition.status != "ACTIVE":
            raise KeyError(f"Tool mapping not found: {definition_id}")
        return definition

    @staticmethod
    def _server_payload(revision: McpServerRevision) -> dict:
        return {
            "transport": revision.transport,
            "url": revision.url,
            "authentication_type": revision.authentication_type,
            "secret_ref": revision.secret_ref,
            "timeout_seconds": revision.timeout_seconds,
        }

    @staticmethod
    def _mapping_payload(revision: ToolMappingRevision) -> dict:
        return {
            "mcp_server_revision_id": revision.mcp_server_revision_id,
            "remote_tool_name": revision.remote_tool_name,
            "adapter_key": revision.adapter_key,
            "input_mapping": revision.input_mapping,
            "output_mapping": revision.output_mapping,
            "timeout_seconds": revision.timeout_seconds,
        }

    def _next_version(self, model, field: str, parent_id: str) -> int:
        return (
            self.session.scalar(
                select(func.max(model.version)).where(getattr(model, field) == parent_id)
            )
            or 0
        ) + 1


def validate_mcp_url(url: str, *, allow_private: bool, trusted_hosts: str = "") -> str:
    value = url.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in ({"http", "https"} if allow_private else {"https"}):
        raise ValueError("MCP URL must use HTTPS unless private MCP URLs are enabled")
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("MCP URL has an invalid authority or suffix")
    hostname = parsed.hostname.lower().rstrip(".")
    trusted = {item.strip().lower().rstrip(".") for item in trusted_hosts.split(",") if item.strip()}
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if not allow_private and address is not None and not address.is_global:
        raise ValueError("Private or non-global MCP addresses are disabled")
    if not allow_private and hostname not in trusted:
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)
            }
        except socket.gaierror as exc:
            raise ValueError("MCP hostname cannot be resolved") from exc
        if not addresses or any(not ipaddress.ip_address(item).is_global for item in addresses):
            raise ValueError("MCP URL resolves to a non-global address")
    return value
