import json
import re
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.database import SessionLocal
from app.main import app
from app.models.database import (
    AgentDefinition,
    AgentNodeBinding,
    AgentRun,
    AgentVersion,
    Base,
    Tenant,
)
from app.services.agent_config.registry import NODE_REGISTRY
from app.services.agent_config.service import (
    DEFAULT_AGENT_DEFINITION_ID,
    SYSTEM_TENANT_ID,
    AgentConfigService,
)
from app.services.agent_config.snapshot import build_legacy_behavior_config, canonical_hash
import app.api.tasks as task_api
from app.config import settings as app_settings


ROOT = Path(__file__).resolve().parents[2]


def _settings(**updates) -> Settings:
    values = {
        "database_url": "sqlite://",
        "prompt_dir": ROOT / "backend/prompts",
        "report_template": ROOT / "backend/templates/report.md.j2",
        "detailed_report_template": ROOT / "backend/templates/detailed_report.md.j2",
        "action_brief_template": ROOT / "backend/templates/action_brief.md.j2",
        "openai_api_key": "test-secret-that-must-not-be-snapshotted",
        **updates,
    }
    return Settings(_env_file=None, **values)


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


def test_node_registry_matches_production_llm_calls_and_prompt_files() -> None:
    parse_pattern = re.compile(r'\.parse\(\s*[^,]+,\s*["\']([^"\']+)["\']', re.MULTILINE)
    production_nodes = {
        match.group(1)
        for path in (ROOT / "backend/app").rglob("*.py")
        for match in parse_pattern.finditer(path.read_text(encoding="utf-8"))
    }
    prompt_nodes = {
        path.stem.removesuffix("_v1")
        for path in (ROOT / "backend/prompts").glob("*_v1.txt")
    }

    assert production_nodes == set(NODE_REGISTRY)
    assert prompt_nodes == set(NODE_REGISTRY)
    assert NODE_REGISTRY["evidence_verify"].conditional is True


def test_alembic_clean_database_upgrade_downgrade_cycle(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "agent-config-migrations.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    monkeypatch.setattr(app_settings, "database_url", database_url)
    config = Config(str(ROOT / "backend/alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "backend/migrations"))

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    assert inspect(engine).has_table("model_profiles")
    assert "model_profile_revision_id" in {
        item["name"] for item in inspect(engine).get_columns("agent_node_bindings")
    }

    command.downgrade(config, "base")
    assert not inspect(engine).has_table("agent_definitions")
    assert not inspect(engine).has_table("model_profiles")

    command.upgrade(config, "head")
    assert inspect(engine).has_table("agent_definitions")
    assert inspect(engine).has_table("model_profiles")
    engine.dispose()


def test_default_agent_seed_is_idempotent(session) -> None:
    service = AgentConfigService(session, _settings())

    first = service.ensure_default_agent()
    second = service.ensure_default_agent()

    assert first.id == second.id
    assert session.scalar(select(func.count()).select_from(Tenant)) == 1
    assert session.scalar(select(func.count()).select_from(AgentDefinition)) == 1
    assert session.scalar(select(func.count()).select_from(AgentVersion)) == 1
    assert session.scalar(select(func.count()).select_from(AgentNodeBinding)) == len(
        NODE_REGISTRY
    )
    definition = session.get(AgentDefinition, DEFAULT_AGENT_DEFINITION_ID)
    assert definition is not None
    assert definition.tenant_id == SYSTEM_TENANT_ID
    assert definition.published_version_id == first.id


def test_secret_rotation_does_not_change_agent_version(session) -> None:
    first = AgentConfigService(
        session, _settings(openai_api_key="first-plaintext-secret")
    ).ensure_default_agent()
    second = AgentConfigService(
        session, _settings(openai_api_key="rotated-plaintext-secret")
    ).ensure_default_agent()

    assert second.id == first.id
    snapshot, _ = AgentConfigService(
        session, _settings(openai_api_key="rotated-plaintext-secret")
    ).resolve_published(DEFAULT_AGENT_DEFINITION_ID)
    serialized = json.dumps(snapshot, ensure_ascii=False)
    assert "first-plaintext-secret" not in serialized
    assert "rotated-plaintext-secret" not in serialized
    assert "env:OPENAI_API_KEY" in serialized


def test_behavior_change_publishes_new_version_without_changing_old_run(session) -> None:
    first_service = AgentConfigService(session, _settings(llm_model="model-v1"))
    first_service.ensure_default_agent()
    old_run = first_service.ensure_intake_run("intake-1")
    first_service.link_research_task("intake-1", "task-1")
    old_snapshot = json.loads(json.dumps(old_run.resolved_config_snapshot))

    second_service = AgentConfigService(session, _settings(llm_model="model-v2"))
    second_version = second_service.ensure_default_agent()
    new_run = second_service.ensure_intake_run("intake-2")

    assert second_version.version == 2
    assert old_run.agent_version_id != new_run.agent_version_id
    assert second_service.get_for_task("task-1").id == old_run.id
    assert old_run.resolved_config_snapshot == old_snapshot
    assert old_run.resolved_config_snapshot["nodes"]["intake_chat"]["model"][
        "model_id"
    ] == "model-v1"
    assert new_run.resolved_config_snapshot["nodes"]["intake_chat"]["model"][
        "model_id"
    ] == "model-v2"
    assert session.get(Tenant, SYSTEM_TENANT_ID) is not None
    assert session.scalar(select(func.count()).select_from(AgentRun)) == 2


def test_resolver_rejects_mutated_published_binding(session) -> None:
    service = AgentConfigService(session, _settings())
    version = service.ensure_default_agent()
    binding = session.scalar(
        select(AgentNodeBinding).where(
            AgentNodeBinding.agent_version_id == version.id,
            AgentNodeBinding.node_key == "final_synthesis",
        )
    )
    assert binding is not None
    binding.model_config = {**binding.model_config, "model_id": "tampered-model"}
    session.commit()

    with pytest.raises(ValueError, match="integrity validation"):
        service.resolve_published(DEFAULT_AGENT_DEFINITION_ID)


def test_snapshot_hash_is_canonical_and_excludes_secret_values() -> None:
    first = build_legacy_behavior_config(_settings(openai_api_key="secret-one"))
    second = build_legacy_behavior_config(_settings(openai_api_key="secret-two"))

    assert canonical_hash(first) == canonical_hash(second)
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    assert "secret-one" not in json.dumps(first, ensure_ascii=False)


def test_compatibility_task_endpoint_creates_agent_run(monkeypatch) -> None:
    dispatched: list[str] = []
    monkeypatch.setattr(task_api.run_research_pipeline, "delay", dispatched.append)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tasks/text",
            json={"text": "准备拜访客户并查询内部项目。"},
        )

    assert response.status_code == 202
    task_id = response.json()["task_id"]
    assert dispatched == [task_id]
    with SessionLocal() as database_session:
        run = database_session.scalar(
            select(AgentRun).where(AgentRun.research_task_id == task_id)
        )
        assert run is not None
        assert run.tenant_id == SYSTEM_TENANT_ID
        assert run.resolved_config_snapshot["agent_version_id"] == run.agent_version_id
