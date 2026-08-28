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
from app.api.admin_models import get_default_agent
from app.database import SessionLocal
from app.main import app
from app.models.database import (
    AgentDefinition,
    AgentNodeBinding,
    AgentRun,
    AgentToolBinding,
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
    assert inspect(engine).has_table("prompt_revisions")
    assert inspect(engine).has_table("mcp_server_revisions")
    assert inspect(engine).has_table("tool_mapping_revisions")
    assert inspect(engine).has_table("agent_tool_bindings")
    assert inspect(engine).has_table("users")
    assert inspect(engine).has_table("conversations")
    assert inspect(engine).has_table("conversation_messages")
    assert "owner_id" in {
        item["name"] for item in inspect(engine).get_columns("intake_sessions")
    }
    assert "model_profile_revision_id" in {
        item["name"] for item in inspect(engine).get_columns("agent_node_bindings")
    }
    assert "prompt_revision_id" in {
        item["name"] for item in inspect(engine).get_columns("agent_node_bindings")
    }
    assert "release_note" in {
        item["name"] for item in inspect(engine).get_columns("agent_versions")
    }

    command.downgrade(config, "base")
    assert not inspect(engine).has_table("agent_definitions")
    assert not inspect(engine).has_table("model_profiles")
    assert not inspect(engine).has_table("prompt_revisions")
    assert not inspect(engine).has_table("mcp_server_revisions")
    assert not inspect(engine).has_table("users")
    assert not inspect(engine).has_table("conversations")

    command.upgrade(config, "head")
    assert inspect(engine).has_table("agent_definitions")
    assert inspect(engine).has_table("model_profiles")
    assert inspect(engine).has_table("prompt_revisions")
    assert inspect(engine).has_table("agent_tool_bindings")
    assert inspect(engine).has_table("auth_sessions")
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


def test_new_runs_use_published_version_when_no_draft_exists(session) -> None:
    service = AgentConfigService(session, _settings())
    published = service.ensure_default_agent()

    intake_run = service.ensure_intake_run("published-intake")
    task_run = service.ensure_task_run("published-task")

    assert intake_run.agent_version_id == published.id
    assert task_run.agent_version_id == published.id


def test_new_runs_use_single_draft_and_existing_intake_run_stays_frozen(session) -> None:
    service = AgentConfigService(session, _settings())
    published = service.ensure_default_agent()
    existing = service.ensure_intake_run("existing-intake")
    existing_snapshot = json.loads(json.dumps(existing.resolved_config_snapshot))
    existing_hash = existing.config_hash
    draft = service.create_draft(DEFAULT_AGENT_DEFINITION_ID)

    new_intake = service.ensure_intake_run("draft-intake", initiator_role="ADMIN")
    new_task = service.ensure_task_run("draft-task", initiator_role="SYSTEM")
    reused = service.ensure_intake_run("existing-intake", initiator_role="ADMIN")
    member_run = service.ensure_task_run("member-task", initiator_role="MEMBER")
    unknown_role_run = service.ensure_task_run(
        "unknown-role-task", initiator_role="FUTURE_ROLE"
    )
    worker_fallback_run = service.ensure_task_run("worker-fallback-task")

    assert existing.agent_version_id == published.id
    assert new_intake.agent_version_id == draft.id
    assert new_task.agent_version_id == draft.id
    assert reused.id == existing.id
    assert reused.agent_version_id == published.id
    assert reused.resolved_config_snapshot == existing_snapshot
    assert reused.config_hash == existing_hash
    assert member_run.agent_version_id == published.id
    assert unknown_role_run.agent_version_id == published.id
    assert worker_fallback_run.agent_version_id == published.id


def test_create_draft_rejects_a_second_draft(session) -> None:
    service = AgentConfigService(session, _settings())
    service.ensure_default_agent()
    service.create_draft(DEFAULT_AGENT_DEFINITION_ID)

    with pytest.raises(ValueError, match="already has a draft version"):
        service.create_draft(DEFAULT_AGENT_DEFINITION_ID)


def test_new_admin_run_rejects_corrupted_multiple_drafts(session) -> None:
    service = AgentConfigService(session, _settings())
    published = service.ensure_default_agent()
    first = service.create_draft(DEFAULT_AGENT_DEFINITION_ID)
    duplicate = AgentVersion(
        agent_definition_id=DEFAULT_AGENT_DEFINITION_ID,
        version=first.version + 1,
        status="DRAFT",
        config_schema_version=published.config_schema_version,
        config=published.config,
        config_hash=published.config_hash,
    )
    session.add(duplicate)
    session.commit()

    with pytest.raises(ValueError, match="multiple draft versions"):
        service.ensure_intake_run(
            "ambiguous-draft-intake", initiator_role="ADMIN"
        )


def test_new_run_rejects_draft_with_unsupported_schema(session) -> None:
    service = AgentConfigService(session, _settings())
    service.ensure_default_agent()
    draft = service.create_draft(DEFAULT_AGENT_DEFINITION_ID)
    draft.config_schema_version += 1
    session.commit()

    with pytest.raises(ValueError, match="Unsupported agent config schema version"):
        service.ensure_intake_run("invalid-schema-intake", initiator_role="ADMIN")


def test_new_run_rejects_incomplete_draft_bindings(session) -> None:
    service = AgentConfigService(session, _settings())
    service.ensure_default_agent()
    draft = service.create_draft(DEFAULT_AGENT_DEFINITION_ID)
    binding = session.scalar(
        select(AgentToolBinding).where(
            AgentToolBinding.agent_version_id == draft.id,
            AgentToolBinding.logical_tool_key == "projects.search",
        )
    )
    assert binding is not None
    session.delete(binding)
    session.commit()

    with pytest.raises(ValueError, match="missing required logical tools"):
        service.ensure_task_run("incomplete-draft-task", initiator_role="ADMIN")


def test_agent_admin_detail_exposes_unified_version_configuration(session) -> None:
    AgentConfigService(session, _settings()).ensure_default_agent()

    detail = get_default_agent(session=session)

    assert {node.node_key for node in detail.published_version.nodes} == set(
        NODE_REGISTRY
    )
    assert {tool.logical_tool_key for tool in detail.published_version.tools} == {
        "identity.find_candidates",
        "projects.search",
    }


def test_agent_admin_routes_expose_fixed_agent_without_runtime_editor() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/api/v1/admin/agent" in route_paths
    assert "/api/v1/admin/agents" not in route_paths
    assert "/api/v1/admin/agent-versions/{agent_version_id}/runtime-config" not in route_paths
    assert (
        "/api/v1/admin/agent-versions/{agent_version_id}/nodes/{node_key}/prompt-working-copy"
        in route_paths
    )


def test_prompt_working_copy_put_passes_browser_cors_preflight() -> None:
    with TestClient(app) as client:
        response = client.options(
            "/api/v1/admin/agent-versions/draft-id/nodes/intake_chat/prompt-working-copy",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "PUT",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert response.status_code == 200
    assert "PUT" in response.headers["access-control-allow-methods"]


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
    assert old_snapshot["loop"]["max_loops"] >= 1
    assert old_snapshot["output"]["formats"]

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
