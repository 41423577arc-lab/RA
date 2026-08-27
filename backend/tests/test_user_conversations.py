from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.admin_models import require_agent_admin
from app.api.auth import router as auth_router
from app.api.conversations import router as conversation_router
from app.api.intake import router as intake_router
from app.config import Settings, settings as app_settings
from app.database import get_session
from app.models.database import AuthSession, Base, ConversationMessage, IntakeSession
from app.services.auth import AuthService, Principal, token_hash, verify_password
from app.services.conversations import ConversationService


ROOT = Path(__file__).resolve().parents[2]


def _settings(**updates) -> Settings:
    return Settings(
        _env_file=None,
        database_url="sqlite://",
        prompt_dir=ROOT / "backend/prompts",
        report_template=ROOT / "backend/templates/report.md.j2",
        detailed_report_template=ROOT / "backend/templates/detailed_report.md.j2",
        action_brief_template=ROOT / "backend/templates/action_brief.md.j2",
        auth_enabled=True,
        auth_allow_registration=True,
        **updates,
    )


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as database_session:
        AuthService(database_session, _settings()).ensure_system_user()
        yield database_session
    Base.metadata.drop_all(engine)


def test_password_and_session_tokens_are_never_stored_in_plaintext(session) -> None:
    service = AuthService(session, _settings())
    user = service.register(
        email="owner@example.com",
        display_name="Owner",
        password="correct-horse-battery-staple",
    )
    token, auth_session = service.create_session(user)

    assert "correct-horse" not in user.password_hash
    assert verify_password("correct-horse-battery-staple", user.password_hash)
    assert not verify_password("wrong-password", user.password_hash)
    assert token not in auth_session.token_hash
    assert auth_session.token_hash == token_hash(token)
    assert service.resolve_session(token).id == user.id

    service.revoke(token)
    assert service.resolve_session(token) is None


def test_conversation_messages_are_append_only_and_owned(session) -> None:
    service = AuthService(session, _settings())
    first_user = service.register(
        email="first@example.com", display_name="First", password="password-one-123"
    )
    second_user = service.register(
        email="second@example.com", display_name="Second", password="password-two-123"
    )
    first = Principal(
        first_user.id, first_user.tenant_id, first_user.email, first_user.display_name,
        first_user.role, True,
    )
    second = Principal(
        second_user.id, second_user.tenant_id, second_user.email, second_user.display_name,
        second_user.role, True,
    )
    conversations = ConversationService(session)
    conversation = conversations.ensure_for_intake(first, "intake-owned")
    conversations.sync_messages(
        conversation,
        [
            {"role": "user", "content": "今晚和中建二局刘希川吃饭"},
            {"role": "assistant", "content": "请确认人物身份。"},
        ],
        channel="intake",
        author_id=first.user_id,
    )
    conversations.sync_messages(
        conversation,
        [
            {"role": "user", "content": "今晚和中建二局刘希川吃饭"},
            {"role": "assistant", "content": "请确认人物身份。"},
            {"role": "user", "content": "对，是这样的"},
        ],
        channel="intake",
        author_id=first.user_id,
    )

    messages = list(
        session.scalars(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation.id)
            .order_by(ConversationMessage.sequence)
        )
    )
    assert [item.sequence for item in messages] == [1, 2, 3]
    assert conversation.title == "今晚和中建二局刘希川吃饭"
    assert conversations.get_owned(first, conversation.id) is not None
    assert conversations.get_owned(second, conversation.id) is None


def test_auth_and_history_api_isolate_users(session, monkeypatch) -> None:
    monkeypatch.setattr(app_settings, "auth_enabled", True)
    monkeypatch.setattr(app_settings, "auth_allow_registration", True)
    test_app = FastAPI()
    test_app.include_router(auth_router)
    test_app.include_router(conversation_router)
    test_app.include_router(intake_router)
    test_app.dependency_overrides[get_session] = lambda: session

    with TestClient(test_app) as client:
        first = client.post(
            "/api/v1/auth/register",
            json={
                "email": "api-first@example.com",
                "display_name": "API First",
                "password": "api-password-one",
            },
        )
        assert first.status_code == 200
        assert "resource_agent_session=" in first.headers["set-cookie"]
        assert "httponly" in first.headers["set-cookie"].lower()
        principal = Principal(
            first.json()["user_id"],
            "00000000-0000-0000-0000-000000000001",
            first.json()["email"],
            first.json()["display_name"],
            first.json()["role"],
            True,
        )
        intake_id = "11111111-1111-1111-1111-111111111111"
        conversation = ConversationService(session).ensure_for_intake(
            principal, intake_id
        )
        session.add(
            IntakeSession(
                id=intake_id,
                tenant_id=principal.tenant_id,
                owner_id=principal.user_id,
                conversation_id=conversation.id,
                status="COLLECTING",
                messages=[],
                structured_context={},
                missing_information=[],
                analysis_input="",
            )
        )
        session.commit()
        assert client.get("/api/v1/conversations").json()[0]["id"] == conversation.id

        assert client.post("/api/v1/auth/logout").status_code == 204
        second = client.post(
            "/api/v1/auth/register",
            json={
                "email": "api-second@example.com",
                "display_name": "API Second",
                "password": "api-password-two",
            },
        )
        assert second.status_code == 200
        assert client.get("/api/v1/conversations").json() == []
        assert client.get(f"/api/v1/conversations/{conversation.id}").status_code == 404
        assert client.get(f"/api/v1/intake/{intake_id}").status_code == 404

    assert session.scalar(select(AuthSession)) is not None


def test_agent_admin_requires_admin_role_when_auth_is_enabled(monkeypatch) -> None:
    monkeypatch.setattr(app_settings, "agent_admin_enabled", True)
    monkeypatch.setattr(app_settings, "auth_enabled", True)
    member = Principal(
        "member-id", "tenant-id", "member@example.com", "Member", "MEMBER", True
    )
    admin = Principal(
        "admin-id", "tenant-id", "admin@example.com", "Admin", "ADMIN", True
    )

    with pytest.raises(HTTPException) as exc_info:
        require_agent_admin(member)
    assert exc_info.value.status_code == 403
    assert require_agent_admin(admin) is None
