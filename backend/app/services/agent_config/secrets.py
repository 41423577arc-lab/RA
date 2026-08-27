import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models.database import ConfigSecret


ALLOWED_ENV_SECRET_REFS = {
    "env:OPENAI_API_KEY": "openai_api_key",
    "env:TAVILY_API_KEY": "tavily_api_key",
    "env:LLM_SAFETY_SALT": "llm_safety_salt",
}


class SecretConfigurationError(RuntimeError):
    pass


class SecretStore:
    def __init__(self, session: Session, settings: Settings):
        self.session = session
        self.settings = settings

    def create(self, tenant_id: str, name: str, plaintext: str) -> ConfigSecret:
        value = plaintext.strip()
        if not value:
            raise ValueError("Secret value cannot be empty")
        if self.session.scalar(
            select(ConfigSecret).where(
                ConfigSecret.tenant_id == tenant_id,
                ConfigSecret.name == name,
            )
        ):
            raise ValueError(f"Secret name already exists: {name}")
        secret = ConfigSecret(
            id=str(uuid4()),
            tenant_id=tenant_id,
            name=name,
            ciphertext=self._fernet().encrypt(value.encode("utf-8")).decode("ascii"),
            fingerprint=self._fingerprint(value),
        )
        self.session.add(secret)
        self.session.flush()
        return secret

    def rotate(self, secret_ref: str, plaintext: str) -> ConfigSecret:
        secret = self._get_db_secret(secret_ref)
        value = plaintext.strip()
        if not value:
            raise ValueError("Secret value cannot be empty")
        secret.ciphertext = self._fernet().encrypt(value.encode("utf-8")).decode("ascii")
        secret.fingerprint = self._fingerprint(value)
        secret.version += 1
        secret.rotated_at = datetime.now(timezone.utc)
        self.session.flush()
        return secret

    def resolve(self, secret_ref: str) -> str:
        setting_name = ALLOWED_ENV_SECRET_REFS.get(secret_ref)
        if setting_name:
            return str(getattr(self.settings, setting_name, ""))
        secret = self._get_db_secret(secret_ref)
        if secret.status != "ACTIVE":
            raise SecretConfigurationError(f"Secret is not active: {secret_ref}")
        try:
            return self._fernet().decrypt(secret.ciphertext.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise SecretConfigurationError("Secret cannot be decrypted with the active key") from exc

    def _get_db_secret(self, secret_ref: str) -> ConfigSecret:
        if not secret_ref.startswith("db:"):
            raise SecretConfigurationError(f"Unsupported secret reference: {secret_ref}")
        secret = self.session.get(ConfigSecret, secret_ref.removeprefix("db:"))
        if secret is None:
            raise SecretConfigurationError(f"Secret not found: {secret_ref}")
        return secret

    def _fernet(self) -> Fernet:
        if not self.settings.agent_secret_key:
            raise SecretConfigurationError(
                "AGENT_SECRET_KEY is required for encrypted configuration secrets"
            )
        try:
            return Fernet(self.settings.agent_secret_key.encode("ascii"))
        except (ValueError, UnicodeEncodeError) as exc:
            raise SecretConfigurationError("AGENT_SECRET_KEY is not a valid Fernet key") from exc

    @staticmethod
    def _fingerprint(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
