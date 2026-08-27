import hashlib
import hmac
import secrets
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, settings
from app.database import get_session
from app.models.database import AuthSession, Tenant, User


SYSTEM_TENANT_ID = "00000000-0000-0000-0000-000000000001"
SYSTEM_USER_ID = "00000000-0000-0000-0000-000000000003"
AUTH_COOKIE_NAME = "resource_agent_session"
PBKDF2_ITERATIONS = 600_000


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str
    email: str
    display_name: str
    role: str
    auth_enabled: bool


class AuthService:
    def __init__(self, session: Session, config: Settings):
        self.session = session
        self.config = config

    def ensure_system_user(self) -> User:
        tenant = self.session.get(Tenant, SYSTEM_TENANT_ID)
        if tenant is None:
            tenant = Tenant(
                id=SYSTEM_TENANT_ID,
                name="System Tenant",
                slug="system-tenant",
            )
            self.session.add(tenant)
            self.session.flush()
        user = self.session.get(User, SYSTEM_USER_ID)
        if user is None:
            user = User(
                id=SYSTEM_USER_ID,
                tenant_id=SYSTEM_TENANT_ID,
                email="system@local.invalid",
                display_name="Local User",
                password_hash="disabled",
                role="SYSTEM",
            )
            self.session.add(user)
            self.session.commit()
        return user

    def register(self, *, email: str, display_name: str, password: str) -> User:
        if not self.config.auth_allow_registration:
            raise ValueError("User registration is disabled")
        normalized_email = email.strip().casefold()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", normalized_email):
            raise ValueError("Email address is invalid")
        if self.session.scalar(
            select(User).where(
                User.tenant_id == SYSTEM_TENANT_ID,
                func.lower(User.email) == normalized_email,
            )
        ):
            raise ValueError("Email is already registered")
        human_count = self.session.scalar(
            select(func.count()).select_from(User).where(User.role != "SYSTEM")
        ) or 0
        user = User(
            tenant_id=SYSTEM_TENANT_ID,
            email=normalized_email,
            display_name=display_name.strip(),
            password_hash=hash_password(password),
            role="ADMIN" if human_count == 0 else "MEMBER",
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    def authenticate(self, email: str, password: str) -> User:
        normalized_email = email.strip().casefold()
        user = self.session.scalar(
            select(User).where(
                User.tenant_id == SYSTEM_TENANT_ID,
                func.lower(User.email) == normalized_email,
            )
        )
        if (
            user is None
            or user.status != "ACTIVE"
            or not verify_password(password, user.password_hash)
        ):
            raise ValueError("Email or password is incorrect")
        user.last_login_at = datetime.now(timezone.utc)
        self.session.commit()
        return user

    def create_session(self, user: User) -> tuple[str, AuthSession]:
        token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        auth_session = AuthSession(
            user_id=user.id,
            token_hash=token_hash(token),
            expires_at=now + timedelta(days=self.config.auth_session_days),
            last_seen_at=now,
        )
        self.session.add(auth_session)
        self.session.commit()
        return token, auth_session

    def resolve_session(self, token: str) -> User | None:
        auth_session = self.session.scalar(
            select(AuthSession).where(AuthSession.token_hash == token_hash(token))
        )
        now = datetime.now(timezone.utc)
        if (
            auth_session is None
            or auth_session.revoked_at is not None
            or _aware(auth_session.expires_at) <= now
        ):
            return None
        user = self.session.get(User, auth_session.user_id)
        if user is None or user.status != "ACTIVE":
            return None
        if now - _aware(auth_session.last_seen_at) > timedelta(minutes=5):
            auth_session.last_seen_at = now
            self.session.commit()
        return user

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        auth_session = self.session.scalar(
            select(AuthSession).where(AuthSession.token_hash == token_hash(token))
        )
        if auth_session is not None and auth_session.revoked_at is None:
            auth_session.revoked_at = datetime.now(timezone.utc)
            self.session.commit()


def get_current_principal(
    request: Request,
    session: Session = Depends(get_session),
) -> Principal:
    service = AuthService(session, settings)
    if not settings.auth_enabled:
        user = service.ensure_system_user()
        return _principal(user, auth_enabled=False)
    token = request.cookies.get(AUTH_COOKIE_NAME)
    user = service.resolve_session(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return _principal(user, auth_enabled=True)


def optional_principal(
    request: Request,
    session: Session = Depends(get_session),
) -> Principal | None:
    if not settings.auth_enabled:
        return _principal(AuthService(session, settings).ensure_system_user(), False)
    token = request.cookies.get(AUTH_COOKIE_NAME)
    user = AuthService(session, settings).resolve_session(token) if token else None
    return _principal(user, True) if user else None


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(candidate.hex(), digest_hex)
    except (TypeError, ValueError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _principal(user: User, auth_enabled: bool) -> Principal:
    return Principal(
        user_id=user.id,
        tenant_id=user.tenant_id,
        email=user.email,
        display_name=user.display_name,
        role=user.role,
        auth_enabled=auth_enabled,
    )


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
