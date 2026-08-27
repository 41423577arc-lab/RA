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
from app.services.agent_config.secrets import ALLOWED_ENV_SECRET_REFS, SecretStore
from app.services.agent_config.service import SYSTEM_TENANT_ID
from app.services.agent_config.snapshot import canonical_hash


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
        provider: str,
        base_url: str,
        secret_ref: str,
    ) -> ModelConnectionRevision:
        connection = self._connection(connection_id)
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
        connection_revision_id: str,
        model_id: str,
        api_mode: str,
        parameters: dict,
    ) -> ModelProfileRevision:
        profile = self._profile(profile_id)
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
