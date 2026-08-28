from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models.database import (
    AgentDefinition,
    AgentNodeBinding,
    AgentRun,
    AgentToolBinding,
    AgentVersion,
    Tenant,
)
from app.services.agent_config.mcp import DEFAULT_TOOL_MAPPINGS, McpConfigService
from app.services.agent_config.snapshot import (
    CONFIG_SCHEMA_VERSION,
    build_legacy_behavior_config,
    canonical_hash,
    resolved_snapshot,
)
from app.services.agent_config.registry import NODE_REGISTRY
from app.services.agent_config.prompts import PromptConfigService
from app.services.auth import SYSTEM_USER_ID


SYSTEM_TENANT_ID = "00000000-0000-0000-0000-000000000001"
DEFAULT_AGENT_DEFINITION_ID = "00000000-0000-0000-0000-000000000002"


class AgentConfigService:
    def __init__(self, session: Session, settings: Settings):
        self.session = session
        self.settings = settings

    def ensure_default_agent(self) -> AgentVersion:
        tenant = self.session.get(Tenant, SYSTEM_TENANT_ID)
        if tenant is None:
            tenant = Tenant(
                id=SYSTEM_TENANT_ID,
                name="System Tenant",
                slug="system-tenant",
            )
            self.session.add(tenant)

        definition = self.session.get(AgentDefinition, DEFAULT_AGENT_DEFINITION_ID)
        if definition is None:
            definition = AgentDefinition(
                id=DEFAULT_AGENT_DEFINITION_ID,
                tenant_id=SYSTEM_TENANT_ID,
                name="Default Agent",
                slug="default-agent",
            )
            self.session.add(definition)
        self.session.flush()

        prompt_service = PromptConfigService(self.session, self.settings)
        default_prompts = prompt_service.ensure_defaults(SYSTEM_TENANT_ID)
        prompt_configs = {
            node_key: prompt_service.resolve_revision(
                revision.id,
                expected_node_key=node_key,
            )
            for node_key, revision in default_prompts.items()
        }
        mcp_service = McpConfigService(self.session, self.settings)
        default_server, default_tools = mcp_service.ensure_defaults(SYSTEM_TENANT_ID)
        mcp_server_configs = [mcp_service.resolve_server_revision(default_server.id)]
        mcp_tool_configs = []
        for logical_key, (revision, allowed_nodes) in default_tools.items():
            item = mcp_service.resolve_mapping_revision(revision.id)
            item.pop("server", None)
            item["allowed_nodes"] = allowed_nodes
            mcp_tool_configs.append(item)
        behavior = build_legacy_behavior_config(
            self.settings,
            prompt_configs=prompt_configs,
            mcp_server_configs=mcp_server_configs,
            mcp_tool_configs=mcp_tool_configs,
        )
        behavior_hash = canonical_hash(behavior)
        published = (
            self.session.get(AgentVersion, definition.published_version_id)
            if definition.published_version_id
            else None
        )
        if (
            published is not None
            and (published.config or {}).get("management", {}).get("source")
            == "admin"
        ):
            self.session.commit()
            return published
        if published is not None and published.config_hash == behavior_hash:
            self.session.commit()
            return published

        next_version = (
            self.session.scalar(
                select(func.max(AgentVersion.version)).where(
                    AgentVersion.agent_definition_id == definition.id
                )
            )
            or 0
        ) + 1
        version = AgentVersion(
            id=str(uuid4()),
            agent_definition_id=definition.id,
            version=next_version,
            status="PUBLISHED",
            config_schema_version=CONFIG_SCHEMA_VERSION,
            config={key: value for key, value in behavior.items() if key != "nodes"},
            config_hash=behavior_hash,
            published_at=datetime.now(timezone.utc),
        )
        self.session.add(version)
        self.session.flush()
        for node_key, node in behavior["nodes"].items():
            self.session.add(
                AgentNodeBinding(
                    agent_version_id=version.id,
                    node_key=node_key,
                    prompt_revision_id=default_prompts[node_key].id,
                    model_config=node["model"],
                    prompt_config=node["prompt"],
                    allowed_tools=node["allowed_tools"],
                )
            )
        for logical_key, (mapping_revision, allowed_nodes) in default_tools.items():
            self.session.add(
                AgentToolBinding(
                    agent_version_id=version.id,
                    logical_tool_key=logical_key,
                    tool_mapping_revision_id=mapping_revision.id,
                    allowed_nodes=allowed_nodes,
                )
            )
        definition.published_version_id = version.id
        self.session.commit()
        self.session.refresh(version)
        return version

    def resolve_published(self, agent_definition_id: str) -> tuple[dict, str]:
        definition = self.session.get(AgentDefinition, agent_definition_id)
        if definition is None or definition.status != "ACTIVE":
            raise KeyError(f"Agent definition {agent_definition_id} not found or inactive")
        if not definition.published_version_id:
            raise ValueError(f"Agent definition {agent_definition_id} has no published version")
        version = self.session.get(AgentVersion, definition.published_version_id)
        if version is None or version.status != "PUBLISHED":
            raise ValueError(f"Agent definition {agent_definition_id} has an invalid published version")
        if version.config_schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported agent config schema version: {version.config_schema_version}"
            )
        behavior = self._behavior_for_version(version)
        if canonical_hash(behavior) != version.config_hash:
            raise ValueError(f"Published AgentVersion {version.id} failed integrity validation")
        snapshot = resolved_snapshot(
            behavior,
            agent_definition_id=definition.id,
            agent_version_id=version.id,
        )
        return snapshot, canonical_hash(snapshot)

    def create_draft(self, agent_definition_id: str) -> AgentVersion:
        definition = self.session.get(AgentDefinition, agent_definition_id)
        if definition is None or not definition.published_version_id:
            raise KeyError(f"Agent definition has no published version: {agent_definition_id}")
        source = self.session.get(AgentVersion, definition.published_version_id)
        if source is None:
            raise ValueError("Published AgentVersion does not exist")
        next_version = (
            self.session.scalar(
                select(func.max(AgentVersion.version)).where(
                    AgentVersion.agent_definition_id == definition.id
                )
            )
            or 0
        ) + 1
        draft_config = {
            **source.config,
            "management": {"source": "admin"},
        }
        draft = AgentVersion(
            agent_definition_id=definition.id,
            version=next_version,
            status="DRAFT",
            config_schema_version=source.config_schema_version,
            config=draft_config,
            config_hash=source.config_hash,
        )
        self.session.add(draft)
        self.session.flush()
        prompt_service = PromptConfigService(self.session, self.settings)
        default_prompts = prompt_service.ensure_defaults(SYSTEM_TENANT_ID)
        for binding in self._bindings(source.id):
            prompt_revision_id = (
                binding.prompt_revision_id or default_prompts[binding.node_key].id
            )
            prompt_config = prompt_service.resolve_revision(
                prompt_revision_id,
                expected_node_key=binding.node_key,
            )
            self.session.add(
                AgentNodeBinding(
                    agent_version_id=draft.id,
                    node_key=binding.node_key,
                    model_profile_revision_id=binding.model_profile_revision_id,
                    prompt_revision_id=prompt_revision_id,
                    model_config=binding.model_config,
                    prompt_config=prompt_config,
                    allowed_tools=binding.allowed_tools,
                )
            )
        for binding in self._tool_bindings(source.id):
            self.session.add(
                AgentToolBinding(
                    agent_version_id=draft.id,
                    logical_tool_key=binding.logical_tool_key,
                    tool_mapping_revision_id=binding.tool_mapping_revision_id,
                    allowed_nodes=binding.allowed_nodes,
                )
            )
        self.session.commit()
        self.session.refresh(draft)
        return draft

    def set_draft_node_model(
        self,
        agent_version_id: str,
        node_key: str,
        model_profile_revision_id: str,
    ) -> AgentVersion:
        if node_key not in NODE_REGISTRY:
            raise ValueError(f"Unknown Agent node: {node_key}")
        version = self.session.get(AgentVersion, agent_version_id)
        if version is None or version.status != "DRAFT":
            raise ValueError("Only draft Agent versions can be edited")
        from app.services.agent_config.models import ModelConfigService

        model_config = ModelConfigService(
            self.session, self.settings
        ).resolve_profile_revision(model_profile_revision_id)
        binding = self.session.scalar(
            select(AgentNodeBinding).where(
                AgentNodeBinding.agent_version_id == version.id,
                AgentNodeBinding.node_key == node_key,
            )
        )
        if binding is None:
            raise ValueError(f"Draft is missing node binding: {node_key}")
        binding.model_profile_revision_id = model_profile_revision_id
        binding.model_config = model_config
        self.session.flush()
        version.config_hash = canonical_hash(self._behavior_for_version(version))
        self.session.commit()
        self.session.refresh(version)
        return version

    def set_draft_node_prompt(
        self,
        agent_version_id: str,
        node_key: str,
        prompt_revision_id: str,
    ) -> AgentVersion:
        if node_key not in NODE_REGISTRY:
            raise ValueError(f"Unknown Agent node: {node_key}")
        version = self.session.get(AgentVersion, agent_version_id)
        if version is None or version.status != "DRAFT":
            raise ValueError("Only draft Agent versions can be edited")
        prompt_config = PromptConfigService(
            self.session, self.settings
        ).resolve_revision(prompt_revision_id, expected_node_key=node_key)
        binding = self.session.scalar(
            select(AgentNodeBinding).where(
                AgentNodeBinding.agent_version_id == version.id,
                AgentNodeBinding.node_key == node_key,
            )
        )
        if binding is None:
            raise ValueError(f"Draft is missing node binding: {node_key}")
        binding.prompt_revision_id = prompt_revision_id
        binding.prompt_config = prompt_config
        self.session.flush()
        version.config_hash = canonical_hash(self._behavior_for_version(version))
        self.session.commit()
        self.session.refresh(version)
        return version

    def set_draft_tool_mapping(
        self,
        agent_version_id: str,
        logical_tool_key: str,
        tool_mapping_revision_id: str,
        allowed_nodes: list[str],
    ) -> AgentVersion:
        version = self.session.get(AgentVersion, agent_version_id)
        if version is None or version.status != "DRAFT":
            raise ValueError("Only draft Agent versions can be edited")
        allowed_callers = {*NODE_REGISTRY, "research_pipeline"}
        normalized_nodes = sorted(set(allowed_nodes))
        if not normalized_nodes or set(normalized_nodes) - allowed_callers:
            raise ValueError("Tool mapping contains an unknown or empty allowed-nodes list")
        resolved = McpConfigService(self.session, self.settings).resolve_mapping_revision(
            tool_mapping_revision_id
        )
        if resolved["logical_tool_key"] != logical_tool_key:
            raise ValueError("Tool mapping revision does not match the logical tool key")
        binding = self.session.scalar(
            select(AgentToolBinding).where(
                AgentToolBinding.agent_version_id == version.id,
                AgentToolBinding.logical_tool_key == logical_tool_key,
            )
        )
        if binding is None:
            binding = AgentToolBinding(
                agent_version_id=version.id,
                logical_tool_key=logical_tool_key,
                tool_mapping_revision_id=tool_mapping_revision_id,
                allowed_nodes=normalized_nodes,
            )
            self.session.add(binding)
        else:
            binding.tool_mapping_revision_id = tool_mapping_revision_id
            binding.allowed_nodes = normalized_nodes
        self.session.flush()
        version.config_hash = canonical_hash(self._behavior_for_version(version))
        self.session.commit()
        self.session.refresh(version)
        return version

    def set_draft_runtime_config(
        self,
        agent_version_id: str,
        *,
        loop: dict,
        output: dict,
    ) -> AgentVersion:
        version = self.session.get(AgentVersion, agent_version_id)
        if version is None or version.status != "DRAFT":
            raise ValueError("Only draft Agent versions can be edited")
        current_output = dict((version.config or {}).get("output", {}))
        version.config = {
            **version.config,
            "management": {"source": "admin"},
            "loop": dict(loop),
            "output": {
                **current_output,
                "formats": sorted(set(output["formats"])),
                "evidence_validation_required": bool(
                    output["evidence_validation_required"]
                ),
            },
        }
        self.session.flush()
        version.config_hash = canonical_hash(self._behavior_for_version(version))
        self.session.commit()
        self.session.refresh(version)
        return version

    def behavior_for_version(self, agent_version_id: str) -> dict:
        version = self.session.get(AgentVersion, agent_version_id)
        if version is None:
            raise KeyError(f"Agent version not found: {agent_version_id}")
        return self._behavior_for_version(version)

    def publish_draft(self, agent_version_id: str) -> AgentVersion:
        version = self.session.get(AgentVersion, agent_version_id)
        if version is None or version.status != "DRAFT":
            raise ValueError("Only draft Agent versions can be published")
        definition = self.session.get(AgentDefinition, version.agent_definition_id)
        if definition is None:
            raise ValueError("Agent definition does not exist")
        behavior = self._behavior_for_version(version)
        if set(behavior["nodes"]) != set(NODE_REGISTRY):
            raise ValueError("Draft does not contain the complete Node Registry")
        bindings = self._bindings(version.id)
        if any(binding.prompt_revision_id is None for binding in bindings):
            raise ValueError("Draft contains a node without a Prompt revision")
        tool_keys = {binding.logical_tool_key for binding in self._tool_bindings(version.id)}
        missing_tools = set(DEFAULT_TOOL_MAPPINGS) - tool_keys
        if missing_tools:
            raise ValueError(
                f"Draft is missing required logical tools: {', '.join(sorted(missing_tools))}"
            )
        version.config_hash = canonical_hash(behavior)
        version.status = "PUBLISHED"
        version.published_at = datetime.now(timezone.utc)
        definition.published_version_id = version.id
        self.session.commit()
        self.session.refresh(version)
        return version

    def ensure_intake_run(
        self,
        intake_session_id: str,
        *,
        owner_id: str = SYSTEM_USER_ID,
        tenant_id: str = SYSTEM_TENANT_ID,
        conversation_id: str | None = None,
    ) -> AgentRun:
        existing = self.session.scalar(
            select(AgentRun).where(AgentRun.intake_session_id == intake_session_id)
        )
        if existing is not None:
            return existing
        self.ensure_default_agent()
        snapshot, config_hash = self.resolve_published(DEFAULT_AGENT_DEFINITION_ID)
        run = AgentRun(
            tenant_id=tenant_id,
            owner_id=owner_id,
            started_by=owner_id,
            conversation_id=conversation_id,
            agent_definition_id=DEFAULT_AGENT_DEFINITION_ID,
            agent_version_id=snapshot["agent_version_id"],
            config_schema_version=snapshot["config_schema_version"],
            resolved_config_snapshot=snapshot,
            config_hash=config_hash,
            status="COLLECTING",
            intake_session_id=intake_session_id,
        )
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def ensure_task_run(
        self,
        research_task_id: str,
        *,
        owner_id: str = SYSTEM_USER_ID,
        tenant_id: str = SYSTEM_TENANT_ID,
        conversation_id: str | None = None,
    ) -> AgentRun:
        existing = self.session.scalar(
            select(AgentRun).where(AgentRun.research_task_id == research_task_id)
        )
        if existing is not None:
            return existing
        self.ensure_default_agent()
        snapshot, config_hash = self.resolve_published(DEFAULT_AGENT_DEFINITION_ID)
        run = AgentRun(
            tenant_id=tenant_id,
            owner_id=owner_id,
            started_by=owner_id,
            conversation_id=conversation_id,
            agent_definition_id=DEFAULT_AGENT_DEFINITION_ID,
            agent_version_id=snapshot["agent_version_id"],
            config_schema_version=snapshot["config_schema_version"],
            resolved_config_snapshot=snapshot,
            config_hash=config_hash,
            status="PENDING",
            research_task_id=research_task_id,
        )
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def link_research_task(
        self,
        intake_session_id: str,
        research_task_id: str,
        *,
        owner_id: str = SYSTEM_USER_ID,
        tenant_id: str = SYSTEM_TENANT_ID,
        conversation_id: str | None = None,
    ) -> AgentRun:
        run = self.ensure_intake_run(
            intake_session_id,
            owner_id=owner_id,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )
        if run.research_task_id and run.research_task_id != research_task_id:
            raise ValueError("Agent run is already linked to another research task")
        run.research_task_id = research_task_id
        run.owner_id = owner_id
        run.started_by = owner_id
        run.conversation_id = conversation_id or run.conversation_id
        run.status = "PENDING"
        self.session.commit()
        self.session.refresh(run)
        return run

    def get_for_task(self, research_task_id: str) -> AgentRun | None:
        return self.session.scalar(
            select(AgentRun).where(AgentRun.research_task_id == research_task_id)
        )

    def get_for_intake(self, intake_session_id: str) -> AgentRun | None:
        return self.session.scalar(
            select(AgentRun).where(AgentRun.intake_session_id == intake_session_id)
        )

    def update_run_status(self, run: AgentRun, status: str) -> AgentRun:
        run.status = status
        self.session.commit()
        self.session.refresh(run)
        return run

    def _behavior_for_version(self, version: AgentVersion) -> dict:
        prompt_service = PromptConfigService(self.session, self.settings)
        nodes = {}
        for binding in self._bindings(version.id):
            prompt_config = binding.prompt_config
            if binding.prompt_revision_id:
                resolved_prompt = prompt_service.resolve_revision(
                    binding.prompt_revision_id,
                    expected_node_key=binding.node_key,
                )
                if canonical_hash(prompt_config) != canonical_hash(resolved_prompt):
                    raise ValueError(
                        f"Prompt binding failed integrity validation: {binding.node_key}"
                    )
                prompt_config = resolved_prompt
            nodes[binding.node_key] = {
                "model": binding.model_config,
                "prompt": prompt_config,
                "allowed_tools": binding.allowed_tools,
            }
        config = {**version.config, "nodes": nodes}
        tool_bindings = self._tool_bindings(version.id)
        if not tool_bindings:
            return config
        mcp_service = McpConfigService(self.session, self.settings)
        tool_mappings = [
            item
            for item in config.get("tool_mappings", [])
            if item.get("provider") != "mcp"
        ]
        servers: dict[str, dict] = {}
        for binding in tool_bindings:
            mapping = mcp_service.resolve_mapping_revision(
                binding.tool_mapping_revision_id
            )
            server = mapping.pop("server")
            mapping["allowed_nodes"] = sorted(set(binding.allowed_nodes))
            tool_mappings.append(mapping)
            servers[server["revision_id"]] = server
        config["mcp_server_revisions"] = [servers[key] for key in sorted(servers)]
        config["tool_mappings"] = sorted(
            tool_mappings, key=lambda item: item["logical_tool_key"]
        )
        return config

    def _bindings(self, agent_version_id: str) -> list[AgentNodeBinding]:
        return list(
            self.session.scalars(
                select(AgentNodeBinding)
                .where(AgentNodeBinding.agent_version_id == agent_version_id)
                .order_by(AgentNodeBinding.node_key)
            )
        )

    def _tool_bindings(self, agent_version_id: str) -> list[AgentToolBinding]:
        return list(
            self.session.scalars(
                select(AgentToolBinding)
                .where(AgentToolBinding.agent_version_id == agent_version_id)
                .order_by(AgentToolBinding.logical_tool_key)
            )
        )
