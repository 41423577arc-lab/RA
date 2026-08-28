import re
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models.database import (
    ConfigSecret,
    ModelConnection,
    ModelConnectionRevision,
    ModelProfile,
    ModelProfileRevision,
)
from app.services.agent_config.provider_adapters import (
    ProviderAdapterRegistry,
    validate_model_base_url,
)
from app.services.agent_config.model_imports import (
    ModelImportPreview,
    parse_model_configuration,
)
from app.services.agent_config.registry import NODE_REGISTRY
from app.services.agent_config.secrets import ALLOWED_ENV_SECRET_REFS, SecretStore
from app.services.agent_config.service import SYSTEM_TENANT_ID
from app.services.agent_config.snapshot import LONG_NODES, REVIEW_NODES, canonical_hash


SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9-]{1,62}[a-z0-9]$")
ALLOWED_MODEL_PARAMETERS = {
    "enabled",
    "max_output_tokens",
    "max_retries",
    "reasoning_effort",
    "response_storage_disabled",
    "store",
    "temperature",
    "timeout_seconds",
    "top_p",
}


class ModelConfigService:
    def __init__(
        self,
        session: Session,
        settings: Settings,
        adapters: ProviderAdapterRegistry | None = None,
    ):
        self.session = session
        self.settings = settings
        self.adapters = adapters or ProviderAdapterRegistry()
        self.secrets = SecretStore(session, settings)

    def ensure_defaults(self) -> dict[str, ModelProfileRevision]:
        connection = self.session.scalar(
            select(ModelConnection).where(
                ModelConnection.tenant_id == SYSTEM_TENANT_ID,
                ModelConnection.slug == "default-model-connection",
            )
        )
        if connection is None:
            connection = ModelConnection(
                tenant_id=SYSTEM_TENANT_ID,
                name="Default Model Connection",
                slug="default-model-connection",
            )
            self.session.add(connection)
            self.session.flush()
        connection_revision = (
            self.session.get(
                ModelConnectionRevision, connection.active_revision_id
            )
            if connection.active_revision_id
            else None
        )
        if connection_revision is None:
            adapter = self.adapters.get(self.settings.model_provider)
            payload = {
                "provider": adapter.provider_key,
                "base_url": validate_model_base_url(
                    self.settings.openai_base_url,
                    allow_private=self.settings.agent_allow_private_model_urls,
                ),
                "authentication_type": "api_key",
                "secret_ref": self._secret_reference(
                    connection_id=connection.id,
                    name=connection.slug,
                    api_key=(
                        self.settings.openai_api_key
                        if self.settings.agent_secret_key
                        and self.settings.openai_api_key
                        else None
                    ),
                    secret_ref=(
                        None
                        if self.settings.agent_secret_key
                        and self.settings.openai_api_key
                        else "env:OPENAI_API_KEY"
                    ),
                ),
            }
            connection_revision = ModelConnectionRevision(
                model_connection_id=connection.id,
                version=1,
                **payload,
                config_hash=canonical_hash(payload),
                published_at=datetime.now(timezone.utc),
            )
            self.session.add(connection_revision)
            self.session.flush()
            connection.active_revision_id = connection_revision.id

        common_parameters = {
            "reasoning_effort": self.settings.llm_reasoning_effort,
            "timeout_seconds": self.settings.llm_timeout_seconds,
            "max_retries": self.settings.llm_max_retries,
            "store": False,
            "enabled": self.settings.llm_enabled,
            "response_storage_disabled": self.settings.llm_disable_response_storage,
        }
        specs = {
            "default": (
                "Default Model",
                self.settings.llm_model,
                {**common_parameters, "max_output_tokens": 8000},
            ),
            "long": (
                "Default Long Output Model",
                self.settings.llm_model,
                {**common_parameters, "max_output_tokens": 16000},
            ),
        }
        if self.settings.llm_review_model != self.settings.llm_model:
            specs["review"] = (
                "Default Review Model",
                self.settings.llm_review_model,
                {**common_parameters, "max_output_tokens": 8000},
            )
        revisions: dict[str, ModelProfileRevision] = {}
        for key, (name, model_id, parameters) in specs.items():
            slug = f"default-{key}-model"
            profile = self.session.scalar(
                select(ModelProfile).where(
                    ModelProfile.tenant_id == SYSTEM_TENANT_ID,
                    ModelProfile.slug == slug,
                )
            )
            if profile is None:
                profile = ModelProfile(
                    tenant_id=SYSTEM_TENANT_ID,
                    name=name,
                    slug=slug,
                )
                self.session.add(profile)
                self.session.flush()
            revision = (
                self.session.get(ModelProfileRevision, profile.active_revision_id)
                if profile.active_revision_id
                else None
            )
            if revision is None:
                payload = {
                    "connection_revision_id": connection_revision.id,
                    "model_id": model_id,
                    "api_mode": self.settings.llm_api_mode,
                    "parameters": self._validate_parameters(parameters),
                }
                revision = ModelProfileRevision(
                    model_profile_id=profile.id,
                    version=1,
                    **payload,
                    config_hash=canonical_hash(payload),
                    published_at=datetime.now(timezone.utc),
                )
                self.session.add(revision)
                self.session.flush()
                profile.active_revision_id = revision.id
            revisions[key] = revision

        return {
            node_key: revisions[
                "long"
                if node_key in LONG_NODES
                else "review"
                if node_key in REVIEW_NODES and "review" in revisions
                else "default"
            ]
            for node_key in NODE_REGISTRY
        }

    def preview_import(self, content: str) -> ModelImportPreview:
        preview = parse_model_configuration(content)
        self.adapters.get(preview.provider)
        validate_model_base_url(
            preview.base_url,
            allow_private=self.settings.agent_allow_private_model_urls,
        )
        self._validate_parameters(preview.parameters)
        return preview

    def import_configuration(
        self,
        content: str,
        *,
        api_key: str,
        connection_name: str,
        connection_slug: str,
        profiles: list[dict],
    ) -> tuple[
        ModelConnection,
        ModelConnectionRevision,
        list[tuple[ModelProfile, ModelProfileRevision]],
    ]:
        try:
            preview = self.preview_import(content)
            self._validate_slug(connection_slug)
            if self.session.scalar(
                select(ModelConnection).where(
                    ModelConnection.tenant_id == SYSTEM_TENANT_ID,
                    ModelConnection.slug == connection_slug,
                )
            ):
                raise ValueError(
                    f"Model connection slug already exists: {connection_slug}"
                )
            expected_roles = {item.role for item in preview.profiles}
            overrides = {str(item["role"]): item for item in profiles}
            if set(overrides) != expected_roles or len(overrides) != len(profiles):
                raise ValueError("Imported model profile overrides do not match the preview")
            for item in overrides.values():
                self._validate_slug(str(item["slug"]))
            slugs = [str(item["slug"]) for item in overrides.values()]
            if len(slugs) != len(set(slugs)):
                raise ValueError("Imported model profile slugs must be unique")
            if self.session.scalar(
                select(ModelProfile).where(
                    ModelProfile.tenant_id == SYSTEM_TENANT_ID,
                    ModelProfile.slug.in_(slugs),
                )
            ):
                raise ValueError("An imported model profile slug already exists")

            adapter = self.adapters.get(preview.provider)
            normalized_url = validate_model_base_url(
                preview.base_url,
                allow_private=self.settings.agent_allow_private_model_urls,
            )
            # Secret、Connection 和全部 Profile 由当前方法统一提交，失败时整体回滚。
            connection_id = str(uuid4())
            secret_ref = self._secret_reference(
                connection_id=connection_id,
                name=connection_slug,
                api_key=api_key,
                secret_ref=None,
            )
            connection = ModelConnection(
                id=connection_id,
                tenant_id=SYSTEM_TENANT_ID,
                name=connection_name.strip(),
                slug=connection_slug,
            )
            connection_payload = {
                "provider": adapter.provider_key,
                "base_url": normalized_url,
                "authentication_type": "api_key",
                "secret_ref": secret_ref,
            }
            connection_revision = ModelConnectionRevision(
                model_connection_id=connection.id,
                version=1,
                **connection_payload,
                config_hash=canonical_hash(connection_payload),
                published_at=datetime.now(timezone.utc),
            )
            self.session.add_all([connection, connection_revision])
            self.session.flush()
            connection.active_revision_id = connection_revision.id

            parameters = self._validate_parameters(preview.parameters)
            created_profiles: list[
                tuple[ModelProfile, ModelProfileRevision]
            ] = []
            for imported in preview.profiles:
                override = overrides[imported.role]
                profile = ModelProfile(
                    id=str(uuid4()),
                    tenant_id=SYSTEM_TENANT_ID,
                    name=str(override["name"]).strip(),
                    slug=str(override["slug"]),
                )
                profile_payload = {
                    "connection_revision_id": connection_revision.id,
                    "model_id": imported.model_id,
                    "api_mode": preview.api_mode,
                    "parameters": parameters,
                }
                profile_revision = ModelProfileRevision(
                    model_profile_id=profile.id,
                    version=1,
                    **profile_payload,
                    config_hash=canonical_hash(profile_payload),
                    published_at=datetime.now(timezone.utc),
                )
                self.session.add_all([profile, profile_revision])
                self.session.flush()
                profile.active_revision_id = profile_revision.id
                created_profiles.append((profile, profile_revision))

            self.session.commit()
            self.session.refresh(connection)
            self.session.refresh(connection_revision)
            for profile, revision in created_profiles:
                self.session.refresh(profile)
                self.session.refresh(revision)
            return connection, connection_revision, created_profiles
        except Exception:
            self.session.rollback()
            raise

    def create_connection(
        self,
        *,
        name: str,
        slug: str,
        provider: str,
        base_url: str,
        api_key: str | None = None,
        secret_ref: str | None = None,
    ) -> tuple[ModelConnection, ModelConnectionRevision]:
        self._validate_slug(slug)
        if self.session.scalar(
            select(ModelConnection).where(
                ModelConnection.tenant_id == SYSTEM_TENANT_ID,
                ModelConnection.slug == slug,
            )
        ):
            raise ValueError(f"Model connection slug already exists: {slug}")
        adapter = self.adapters.get(provider)
        normalized_url = validate_model_base_url(
            base_url, allow_private=self.settings.agent_allow_private_model_urls
        )
        connection_id = str(uuid4())
        reference = self._secret_reference(
            connection_id=connection_id,
            name=slug,
            api_key=api_key,
            secret_ref=secret_ref,
        )
        connection = ModelConnection(
            id=connection_id,
            tenant_id=SYSTEM_TENANT_ID,
            name=name.strip(),
            slug=slug,
        )
        payload = {
            "provider": adapter.provider_key,
            "base_url": normalized_url,
            "authentication_type": "api_key",
            "secret_ref": reference,
        }
        revision = ModelConnectionRevision(
            model_connection_id=connection.id,
            version=1,
            **payload,
            config_hash=canonical_hash(payload),
            published_at=datetime.now(timezone.utc),
        )
        self.session.add_all([connection, revision])
        self.session.flush()
        connection.active_revision_id = revision.id
        self.session.commit()
        self.session.refresh(connection)
        self.session.refresh(revision)
        return connection, revision

    def revise_connection(
        self,
        connection_id: str,
        *,
        name: str | None = None,
        provider: str,
        base_url: str,
        secret_ref: str,
    ) -> ModelConnectionRevision:
        connection = self._connection(connection_id)
        if name is not None:
            connection.name = name.strip()
        adapter = self.adapters.get(provider)
        normalized_url = validate_model_base_url(
            base_url, allow_private=self.settings.agent_allow_private_model_urls
        )
        self._validate_secret_ref(secret_ref)
        version = self._next_version(ModelConnectionRevision, "model_connection_id", connection.id)
        payload = {
            "provider": adapter.provider_key,
            "base_url": normalized_url,
            "authentication_type": "api_key",
            "secret_ref": secret_ref,
        }
        revision = ModelConnectionRevision(
            model_connection_id=connection.id,
            version=version,
            **payload,
            config_hash=canonical_hash(payload),
            published_at=datetime.now(timezone.utc),
        )
        self.session.add(revision)
        self.session.flush()
        connection.active_revision_id = revision.id
        self.session.commit()
        self.session.refresh(revision)
        return revision

    def rotate_connection_secret(self, connection_id: str, api_key: str) -> str:
        connection = self._connection(connection_id)
        revision = self.session.get(ModelConnectionRevision, connection.active_revision_id)
        if revision is None or not revision.secret_ref.startswith("db:"):
            raise ValueError("Only database-backed secrets can be rotated")
        secret = self.secrets.rotate(revision.secret_ref, api_key)
        self.session.commit()
        return secret.fingerprint

    def create_profile(
        self,
        *,
        name: str,
        slug: str,
        connection_revision_id: str,
        model_id: str,
        api_mode: str,
        parameters: dict,
    ) -> tuple[ModelProfile, ModelProfileRevision]:
        self._validate_slug(slug)
        if self.session.scalar(
            select(ModelProfile).where(
                ModelProfile.tenant_id == SYSTEM_TENANT_ID,
                ModelProfile.slug == slug,
            )
        ):
            raise ValueError(f"Model profile slug already exists: {slug}")
        self._connection_revision(connection_revision_id)
        params = self._validate_parameters(parameters)
        if api_mode not in {"chat_completions", "responses"}:
            raise ValueError(f"Unsupported API mode: {api_mode}")
        profile = ModelProfile(
            id=str(uuid4()),
            tenant_id=SYSTEM_TENANT_ID,
            name=name.strip(),
            slug=slug,
        )
        payload = {
            "connection_revision_id": connection_revision_id,
            "model_id": model_id.strip(),
            "api_mode": api_mode,
            "parameters": params,
        }
        revision = ModelProfileRevision(
            model_profile_id=profile.id,
            version=1,
            **payload,
            config_hash=canonical_hash(payload),
            published_at=datetime.now(timezone.utc),
        )
        self.session.add_all([profile, revision])
        self.session.flush()
        profile.active_revision_id = revision.id
        self.session.commit()
        self.session.refresh(profile)
        self.session.refresh(revision)
        return profile, revision

    def revise_profile(
        self,
        profile_id: str,
        *,
        name: str | None = None,
        connection_revision_id: str,
        model_id: str,
        api_mode: str,
        parameters: dict,
    ) -> ModelProfileRevision:
        profile = self._profile(profile_id)
        if name is not None:
            profile.name = name.strip()
        self._connection_revision(connection_revision_id)
        params = self._validate_parameters(parameters)
        if api_mode not in {"chat_completions", "responses"}:
            raise ValueError(f"Unsupported API mode: {api_mode}")
        payload = {
            "connection_revision_id": connection_revision_id,
            "model_id": model_id.strip(),
            "api_mode": api_mode,
            "parameters": params,
        }
        revision = ModelProfileRevision(
            model_profile_id=profile.id,
            version=self._next_version(ModelProfileRevision, "model_profile_id", profile.id),
            **payload,
            config_hash=canonical_hash(payload),
            published_at=datetime.now(timezone.utc),
        )
        self.session.add(revision)
        self.session.flush()
        profile.active_revision_id = revision.id
        self.session.commit()
        self.session.refresh(revision)
        return revision

    def resolve_profile_revision(self, revision_id: str) -> dict:
        profile_revision = self.session.get(ModelProfileRevision, revision_id)
        if profile_revision is None or profile_revision.status != "PUBLISHED":
            raise KeyError(f"Model profile revision not found: {revision_id}")
        connection = self._connection_revision(profile_revision.connection_revision_id)
        profile_payload = {
            "connection_revision_id": profile_revision.connection_revision_id,
            "model_id": profile_revision.model_id,
            "api_mode": profile_revision.api_mode,
            "parameters": profile_revision.parameters,
        }
        if canonical_hash(profile_payload) != profile_revision.config_hash:
            raise ValueError(f"Model profile revision failed integrity validation: {revision_id}")
        connection_payload = {
            "provider": connection.provider,
            "base_url": connection.base_url,
            "authentication_type": connection.authentication_type,
            "secret_ref": connection.secret_ref,
        }
        if canonical_hash(connection_payload) != connection.config_hash:
            raise ValueError(
                f"Model connection revision failed integrity validation: {connection.id}"
            )
        payload = {
            "model_profile_revision_id": profile_revision.id,
            "connection_revision_id": connection.id,
            "provider": connection.provider,
            "base_url": connection.base_url,
            "authentication_type": connection.authentication_type,
            "secret_ref": connection.secret_ref,
            "safety_identifier_salt_ref": "env:LLM_SAFETY_SALT",
            "model_id": profile_revision.model_id,
            "api_mode": profile_revision.api_mode,
            **profile_revision.parameters,
        }
        payload.setdefault("enabled", True)
        payload.setdefault("store", False)
        payload.setdefault("response_storage_disabled", True)
        payload.setdefault("timeout_seconds", 120)
        payload.setdefault("max_retries", 1)
        payload.setdefault("max_output_tokens", 8000)
        return payload

    def test_connection(self, connection_id: str) -> dict:
        connection = self._connection(connection_id)
        revision = self._connection_revision(connection.active_revision_id)
        api_key = self.secrets.resolve(revision.secret_ref)
        adapter = self.adapters.get(revision.provider)
        return adapter.test_connection(
            {
                "base_url": revision.base_url,
                "timeout_seconds": 30,
                "_allow_private": self.settings.agent_allow_private_model_urls,
                "_trusted_hosts": self.settings.agent_trusted_model_hosts,
            },
            api_key,
        )

    def _secret_reference(
        self,
        *,
        connection_id: str,
        name: str,
        api_key: str | None,
        secret_ref: str | None,
    ) -> str:
        if bool(api_key) == bool(secret_ref):
            raise ValueError("Provide exactly one of api_key or secret_ref")
        if secret_ref:
            self._validate_secret_ref(secret_ref)
            return secret_ref
        secret = self.secrets.create(
            SYSTEM_TENANT_ID,
            f"model-{name}-{connection_id[:8]}",
            api_key or "",
        )
        return f"db:{secret.id}"

    @staticmethod
    def _validate_slug(slug: str) -> None:
        if not SLUG_PATTERN.fullmatch(slug):
            raise ValueError("Slug must contain 3-64 lowercase letters, numbers, or hyphens")

    @staticmethod
    def _validate_parameters(parameters: dict) -> dict:
        unknown = set(parameters) - ALLOWED_MODEL_PARAMETERS
        if unknown:
            raise ValueError(f"Unsupported model parameters: {', '.join(sorted(unknown))}")
        output = dict(parameters)
        if output.get("store") not in (None, False):
            raise ValueError("Model response storage must remain disabled")
        if "temperature" in output and not 0 <= float(output["temperature"]) <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if "top_p" in output and not 0 < float(output["top_p"]) <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if "max_retries" in output and not 0 <= int(output["max_retries"]) <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        return output

    def _validate_secret_ref(self, secret_ref: str) -> None:
        if secret_ref.startswith("db:"):
            secret = self.session.get(ConfigSecret, secret_ref.removeprefix("db:"))
            if (
                secret is None
                or secret.tenant_id != SYSTEM_TENANT_ID
                or secret.status != "ACTIVE"
            ):
                raise ValueError(f"Secret reference is not active: {secret_ref}")
            return
        if secret_ref not in ALLOWED_ENV_SECRET_REFS:
            raise ValueError(f"Unsupported secret reference: {secret_ref}")

    def _connection(self, connection_id: str) -> ModelConnection:
        connection = self.session.get(ModelConnection, connection_id)
        if connection is None or connection.status != "ACTIVE":
            raise KeyError(f"Model connection not found: {connection_id}")
        return connection

    def _connection_revision(self, revision_id: str | None) -> ModelConnectionRevision:
        revision = self.session.get(ModelConnectionRevision, revision_id)
        if revision is None or revision.status != "PUBLISHED":
            raise KeyError(f"Model connection revision not found: {revision_id}")
        return revision

    def _profile(self, profile_id: str) -> ModelProfile:
        profile = self.session.get(ModelProfile, profile_id)
        if profile is None or profile.status != "ACTIVE":
            raise KeyError(f"Model profile not found: {profile_id}")
        return profile

    def _next_version(self, model, field: str, parent_id: str) -> int:
        column = getattr(model, field)
        return (
            self.session.scalar(select(func.max(model.version)).where(column == parent_id)) or 0
        ) + 1
