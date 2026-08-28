import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings, settings
from app.main import app
from app.models.database import (
    AgentNodeBinding,
    Base,
    ConfigSecret,
    ModelConnection,
    ModelProfile,
    ModelProfileRevision,
)
from app.schemas.task import WebSearchPlan, WebSearchQuery
from app.services.agent_config.models import ModelConfigService
from app.services.agent_config.model_imports import parse_model_configuration
from app.services.agent_config.provider_adapters import (
    ModelProviderConfigurationError,
    OpenAIProviderAdapter,
    validate_model_base_url,
)
from app.services.agent_config.service import (
    DEFAULT_AGENT_DEFINITION_ID,
    AgentConfigService,
)
from app.services.agent_config.snapshot import canonical_hash
from app.services.integrations.llm_client import StructuredLLM


ROOT = Path(__file__).resolve().parents[2]


def _settings(**updates) -> Settings:
    values = {
        "database_url": "sqlite://",
        "prompt_dir": ROOT / "backend/prompts",
        "report_template": ROOT / "backend/templates/report.md.j2",
        "detailed_report_template": ROOT / "backend/templates/detailed_report.md.j2",
        "action_brief_template": ROOT / "backend/templates/action_brief.md.j2",
        "openai_api_key": "legacy-key",
        "agent_secret_key": Fernet.generate_key().decode("ascii"),
        "agent_allow_private_model_urls": True,
        **updates,
    }
    return Settings(
        _env_file=None,
        **values,
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
        yield database_session
    Base.metadata.drop_all(engine)


def test_api_key_is_encrypted_and_rotation_does_not_create_config_revision(session) -> None:
    config = _settings()
    AgentConfigService(session, config).ensure_default_agent()
    service = ModelConfigService(session, config)
    connection, revision = service.create_connection(
        name="Customer Gateway",
        slug="customer-gateway",
        provider="openai_compatible",
        base_url="http://model.internal/v1",
        api_key="first-sensitive-api-key",
    )
    secret = session.get(ConfigSecret, revision.secret_ref.removeprefix("db:"))

    assert secret is not None
    assert "first-sensitive-api-key" not in secret.ciphertext
    assert service.secrets.resolve(revision.secret_ref) == "first-sensitive-api-key"
    revision_id = revision.id
    revision_hash = revision.config_hash

    service.rotate_connection_secret(connection.id, "rotated-sensitive-api-key")

    assert connection.active_revision_id == revision_id
    assert revision.config_hash == revision_hash
    assert service.secrets.resolve(revision.secret_ref) == "rotated-sensitive-api-key"
    assert secret.version == 2


def test_model_import_parses_generic_toml_and_ignores_unrelated_fields() -> None:
    preview = parse_model_configuration(
        '''
model_provider = "OpenAI"
model = "gpt-5.5"
review_model = "gpt-5.5"
model_reasoning_effort = "xhigh"
disable_response_storage = true
network_access = "enabled"
windows_wsl_setup_acknowledged = true

[model_providers.OpenAI]
name = "OpenAI"
base_url = "[https://vftsub.vf-tech.cn](https://vftsub.vf-tech.cn)"
wire_api = "responses"
requires_openai_auth = true

[features]
goals = true
'''
    )

    assert preview.source_format == "toml"
    assert preview.provider == "openai"
    assert preview.base_url == "https://vftsub.vf-tech.cn"
    assert preview.api_mode == "responses"
    assert preview.parameters == {
        "reasoning_effort": "xhigh",
        "store": False,
        "response_storage_disabled": True,
    }
    assert [(item.role, item.model_id) for item in preview.profiles] == [
        ("primary", "gpt-5.5")
    ]
    assert preview.ignored_fields == (
        "features.goals",
        "network_access",
        "windows_wsl_setup_acknowledged",
    )


@pytest.mark.parametrize(
    ("content", "source_format"),
    [
        (
            json.dumps(
                {
                    "model_provider": "openai_compatible",
                    "model": "customer-model",
                    "base_url": "https://models.example/v1",
                    "api_mode": "chat_completions",
                    "parameters": {
                        "reasoning_effort": "high",
                        "max_output_tokens": 4096,
                        "store": False,
                    },
                }
            ),
            "json",
        ),
        (
            "\n".join(
                [
                    "MODEL_PROVIDER=openai_compatible",
                    "LLM_MODEL=customer-model",
                    "OPENAI_BASE_URL=https://models.example/v1",
                    "LLM_API_MODE=chat_completions",
                    "LLM_REASONING_EFFORT=high",
                    "OPENAI_API_KEY=must-not-import-from-content",
                ]
            ),
            "env",
        ),
    ],
)
def test_model_import_parses_json_and_env(content: str, source_format: str) -> None:
    preview = parse_model_configuration(content)

    assert preview.source_format == source_format
    assert preview.provider == "openai_compatible"
    assert preview.base_url == "https://models.example/v1"
    assert preview.api_mode == "chat_completions"
    assert preview.parameters["reasoning_effort"] == "high"
    if source_format == "json":
        assert preview.parameters["max_output_tokens"] == 4096
    else:
        assert "openai_api_key" in preview.ignored_fields
        assert "must-not-import-from-content" not in repr(preview)
    assert preview.profiles[0].model_id == "customer-model"


def test_model_import_atomically_creates_connection_profiles_and_secret(session) -> None:
    config = _settings()
    AgentConfigService(session, config).ensure_default_agent()
    service = ModelConfigService(session, config)
    content = '''
model_provider = "OpenAI"
model = "primary-model"
review_model = "review-model"
model_reasoning_effort = "xhigh"
disable_response_storage = true

[model_providers.OpenAI]
name = "Customer OpenAI"
base_url = "https://models.example/v1"
wire_api = "responses"
requires_openai_auth = true
'''

    connection, connection_revision, profiles = service.import_configuration(
        content,
        api_key="imported-plaintext-secret",
        connection_name="Customer OpenAI",
        connection_slug="customer-openai-connection",
        profiles=[
            {"role": "primary", "name": "Primary Model", "slug": "primary-model"},
            {"role": "review", "name": "Review Model", "slug": "review-model"},
        ],
    )

    assert connection.active_revision_id == connection_revision.id
    assert connection_revision.provider == "openai"
    assert connection_revision.secret_ref.startswith("db:")
    secret = session.get(
        ConfigSecret, connection_revision.secret_ref.removeprefix("db:")
    )
    assert secret is not None
    assert "imported-plaintext-secret" not in secret.ciphertext
    assert service.secrets.resolve(connection_revision.secret_ref) == (
        "imported-plaintext-secret"
    )
    assert len(profiles) == 2
    assert {profile.slug for profile, _ in profiles} == {
        "primary-model",
        "review-model",
    }
    for _, revision in profiles:
        assert revision.connection_revision_id == connection_revision.id
        assert revision.api_mode == "responses"
        assert revision.parameters["reasoning_effort"] == "xhigh"
        assert revision.parameters["store"] is False
    assert session.scalar(
        select(func.count()).select_from(ModelConnection)
    ) >= 2
    assert session.scalar(select(func.count()).select_from(ModelProfile)) >= 4


def test_model_import_rolls_back_secret_and_connection_when_profile_slug_exists(
    session,
) -> None:
    config = _settings()
    AgentConfigService(session, config).ensure_default_agent()
    service = ModelConfigService(session, config)
    before_connections = session.scalar(
        select(func.count()).select_from(ModelConnection)
    )
    before_secrets = session.scalar(select(func.count()).select_from(ConfigSecret))

    with pytest.raises(ValueError, match="profile slug already exists"):
        service.import_configuration(
            'model_provider="OpenAI"\nmodel="gpt-5.5"\nbase_url="https://models.example/v1"',
            api_key="must-not-persist",
            connection_name="Duplicate Test",
            connection_slug="duplicate-test-connection",
            profiles=[
                {
                    "role": "primary",
                    "name": "Duplicate",
                    "slug": "default-default-model",
                }
            ],
        )

    assert session.scalar(select(func.count()).select_from(ModelConnection)) == (
        before_connections
    )
    assert session.scalar(select(func.count()).select_from(ConfigSecret)) == before_secrets


def test_default_agent_binds_every_llm_node_to_database_profile(session) -> None:
    config = _settings()
    service = AgentConfigService(session, config)
    published = service.ensure_default_agent()
    bindings = service._bindings(published.id)

    assert bindings
    assert all(binding.model_profile_revision_id for binding in bindings)
    model_service = ModelConfigService(session, config)
    for binding in bindings:
        assert model_service.resolve_profile_revision(
            binding.model_profile_revision_id
        ) == binding.model_config


def test_model_profile_draft_binding_changes_only_new_runs_without_publish(session) -> None:
    config = _settings()
    agent_service = AgentConfigService(session, config)
    agent_service.ensure_default_agent()
    old_run = agent_service.ensure_intake_run("old-intake")
    old_snapshot = json.loads(json.dumps(old_run.resolved_config_snapshot))
    old_config_hash = old_run.config_hash
    model_service = ModelConfigService(session, config)
    _, connection_revision = model_service.create_connection(
        name="Second Gateway",
        slug="second-gateway",
        provider="openai_compatible",
        base_url="http://model.internal/v1",
        secret_ref="env:OPENAI_API_KEY",
    )
    _, profile_revision = model_service.create_profile(
        name="Fast Intake Model",
        slug="fast-intake-model",
        connection_revision_id=connection_revision.id,
        model_id="customer-fast-model",
        api_mode="chat_completions",
        parameters={
            "temperature": 0.2,
            "max_output_tokens": 4096,
            "max_retries": 2,
        },
    )
    revision_count = session.scalar(
        select(func.count()).select_from(ModelProfileRevision)
    )
    draft = agent_service.create_draft(DEFAULT_AGENT_DEFINITION_ID)
    agent_service.set_draft_node_model(
        draft.id, "intake_chat", profile_revision.id
    )
    agent_service.set_draft_node_model(
        draft.id, "intake_chat", profile_revision.id
    )
    new_run = agent_service.ensure_intake_run(
        "new-intake", initiator_role="ADMIN"
    )

    assert old_run.agent_version_id != draft.id
    assert old_run.resolved_config_snapshot == old_snapshot
    assert old_run.config_hash == old_config_hash
    assert old_run.resolved_config_snapshot["nodes"]["intake_chat"]["model"][
        "model_id"
    ] != "customer-fast-model"
    assert new_run.agent_version_id == draft.id
    configured = new_run.resolved_config_snapshot["nodes"]["intake_chat"]["model"]
    assert configured["model_id"] == "customer-fast-model"
    assert configured["temperature"] == 0.2
    assert configured["secret_ref"] == "env:OPENAI_API_KEY"
    assert new_run.resolved_config_snapshot["nodes"]["final_synthesis"]["model"][
        "model_id"
    ] == config.llm_model
    binding = session.scalar(
        select(AgentNodeBinding).where(
            AgentNodeBinding.agent_version_id == draft.id,
            AgentNodeBinding.node_key == "intake_chat",
        )
    )
    assert binding.model_profile_revision_id == profile_revision.id
    assert session.scalar(
        select(func.count()).select_from(ModelProfileRevision)
    ) == revision_count


class _FakeChatCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        result = WebSearchPlan(
            queries=[WebSearchQuery(query="配置模型测试", purpose="验证")]
        )
        return SimpleNamespace(
            id="configured-response",
            choices=[SimpleNamespace(message=SimpleNamespace(content=result.model_dump_json()))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
        )


class _FakeAdapter:
    def __init__(self):
        self.config = None
        self.api_key = None
        self.completions = _FakeChatCompletions()

    def create_client(self, config, api_key):
        self.config = config
        self.api_key = api_key
        return SimpleNamespace(chat=SimpleNamespace(completions=self.completions))


class _FakeAdapters:
    def __init__(self, adapter):
        self.adapter = adapter

    def get(self, _provider):
        return self.adapter


def test_structured_llm_uses_node_model_from_run_snapshot() -> None:
    config = _settings(
        openai_api_key="runtime-secret",
        agent_trusted_model_hosts="configured.example",
    )
    node_model = {
        "provider": "openai_compatible",
        "base_url": "https://configured.example/v1",
        "secret_ref": "env:OPENAI_API_KEY",
        "safety_identifier_salt_ref": "env:LLM_SAFETY_SALT",
        "model_id": "configured-node-model",
        "api_mode": "chat_completions",
        "temperature": 0.35,
        "top_p": 0.8,
        "timeout_seconds": 17,
        "max_retries": 0,
        "max_output_tokens": 1234,
        "enabled": True,
        "response_storage_disabled": True,
        "store": False,
    }
    prompt_content = (ROOT / "backend/prompts/evidence_verify_v1.txt").read_text(
        encoding="utf-8"
    )
    snapshot = {
        "nodes": {
            "evidence_verify": {
                "model": node_model,
                "prompt": {
                    "revision_id": "test-evidence-prompt",
                    "version": 1,
                    "node_key": "evidence_verify",
                    "content": prompt_content,
                    "content_hash": canonical_hash(prompt_content),
                    "skills": [],
                },
            }
        }
    }
    adapter = _FakeAdapter()
    llm = StructuredLLM(
        config,
        resolved_config=snapshot,
        provider_adapters=_FakeAdapters(adapter),
    )

    result = llm.parse(
        "task-configured",
        "evidence_verify",
        {"candidate": "demo"},
        WebSearchPlan,
    )

    assert result.queries[0].query == "配置模型测试"
    assert adapter.api_key == "runtime-secret"
    assert adapter.config["model_id"] == "configured-node-model"
    assert adapter.config["_trusted_hosts"] == "configured.example"
    assert adapter.completions.kwargs["model"] == "configured-node-model"
    assert adapter.completions.kwargs["temperature"] == 0.35
    assert adapter.completions.kwargs["top_p"] == 0.8
    assert adapter.completions.kwargs["max_tokens"] == 1234
    assert adapter.completions.kwargs["timeout"] == 17


def test_model_parameters_and_base_urls_are_restricted(session) -> None:
    service = ModelConfigService(session, _settings())

    with pytest.raises(ValueError, match="Unsupported model parameters"):
        service._validate_parameters({"arbitrary_provider_payload": "unsafe"})
    with pytest.raises(ValueError, match="storage"):
        service._validate_parameters({"store": True})
    with pytest.raises(ModelProviderConfigurationError, match="HTTPS"):
        validate_model_base_url("http://models.example/v1", allow_private=False)
    with pytest.raises(ModelProviderConfigurationError, match="Private"):
        validate_model_base_url("https://127.0.0.1/v1", allow_private=False)
    assert (
        validate_model_base_url("https://models.example/v1/", allow_private=False)
        == "https://models.example/v1"
    )


def test_provider_rejects_hostname_that_resolves_to_private_address(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.agent_config.provider_adapters.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("10.0.0.8", 443))],
    )

    with pytest.raises(ModelProviderConfigurationError, match="non-global"):
        OpenAIProviderAdapter("openai_compatible").create_client(
            {
                "base_url": "https://models.example/v1",
                "timeout_seconds": 30,
                "_allow_private": False,
            },
            "secret",
        )


def test_provider_allows_only_exact_trusted_hostname_with_non_global_dns(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.agent_config.provider_adapters.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("198.18.1.78", 443))],
    )

    client = OpenAIProviderAdapter("openai_compatible").create_client(
        {
            "base_url": "https://vftllmapi.vf-tech.cn/v1",
            "timeout_seconds": 30,
            "_allow_private": False,
            "_trusted_hosts": "VFTLLMAPI.VF-TECH.CN.",
        },
        "secret",
    )

    assert str(client.base_url) == "https://vftllmapi.vf-tech.cn/v1/"


def test_provider_does_not_apply_trust_to_lookalike_hostname_or_literal_ip(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.agent_config.provider_adapters.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("198.18.1.78", 443))],
    )
    adapter = OpenAIProviderAdapter("openai_compatible")

    with pytest.raises(ModelProviderConfigurationError, match="non-global"):
        adapter.create_client(
            {
                "base_url": "https://vftllmapi.vf-tech.cn.attacker.example/v1",
                "_allow_private": False,
                "_trusted_hosts": "vftllmapi.vf-tech.cn",
            },
            "secret",
        )
    with pytest.raises(ModelProviderConfigurationError, match="Private"):
        adapter.create_client(
            {
                "base_url": "https://198.18.1.78/v1",
                "_allow_private": False,
                "_trusted_hosts": "198.18.1.78",
            },
            "secret",
        )


def test_model_admin_api_is_disabled_by_default() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/admin/model-connections")

    assert response.status_code == 404
    assert response.json()["detail"] == "Agent administration is disabled"


def test_model_admin_api_never_returns_plaintext_secret(monkeypatch) -> None:
    monkeypatch.setattr(settings, "agent_admin_enabled", True)
    monkeypatch.setattr(settings, "agent_secret_key", Fernet.generate_key().decode("ascii"))

    slug = f"api-managed-{uuid4().hex[:12]}"
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/admin/model-connections",
            json={
                "name": "API Managed Gateway",
                "slug": slug,
                "provider": "openai_compatible",
                "base_url": "https://models.example/v1",
                "api_key": "api-managed-plaintext-secret",
            },
        )

    assert response.status_code == 200, response.text
    serialized = json.dumps(response.json())
    assert "api-managed-plaintext-secret" not in serialized
    assert "ciphertext" not in serialized
    assert response.json()["secret_ref"].startswith("db:")
