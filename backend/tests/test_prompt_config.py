from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings, settings
from app.main import app
from app.models.database import (
    AgentNodeBinding,
    Base,
    PromptDefinition,
    PromptRevision,
)
from app.services.agent_config.prompts import PromptConfigService
from app.services.agent_config.registry import NODE_REGISTRY
from app.services.agent_config.service import (
    DEFAULT_AGENT_DEFINITION_ID,
    SYSTEM_TENANT_ID,
    AgentConfigService,
)
from app.services.agent_config.snapshot import canonical_hash
from app.services.integrations.llm_client import LLMUnavailable, StructuredLLM


ROOT = Path(__file__).resolve().parents[2]


def _settings(**updates) -> Settings:
    values = {
        "database_url": "sqlite://",
        "prompt_dir": ROOT / "backend/prompts",
        "report_template": ROOT / "backend/templates/report.md.j2",
        "detailed_report_template": ROOT / "backend/templates/detailed_report.md.j2",
        "action_brief_template": ROOT / "backend/templates/action_brief.md.j2",
        "openai_api_key": "prompt-test-key",
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


def test_default_prompt_files_are_imported_and_bound(session) -> None:
    config = _settings()
    service = AgentConfigService(session, config)
    version = service.ensure_default_agent()
    snapshot, _ = service.resolve_published(DEFAULT_AGENT_DEFINITION_ID)

    assert session.scalar(select(func.count()).select_from(PromptDefinition)) == len(
        NODE_REGISTRY
    )
    assert session.scalar(select(func.count()).select_from(PromptRevision)) == len(
        NODE_REGISTRY
    )
    bindings = list(
        session.scalars(
            select(AgentNodeBinding).where(
                AgentNodeBinding.agent_version_id == version.id
            )
        )
    )
    assert len(bindings) == len(NODE_REGISTRY)
    assert all(binding.prompt_revision_id for binding in bindings)
    intake_prompt = snapshot["nodes"]["intake_agent"]["prompt"]
    assert intake_prompt["version"] == 1
    assert intake_prompt["node_key"] == "intake_agent"
    assert len(intake_prompt["skills"]) == 4
    assert intake_prompt["validation_report"]["output_schema"] == "AgentTurn"


def test_default_agent_does_not_read_prompt_files_after_initial_import(session, tmp_path) -> None:
    initial = AgentConfigService(session, _settings()).ensure_default_agent()

    restarted = AgentConfigService(
        session,
        _settings(prompt_dir=tmp_path / "missing-prompts"),
    ).ensure_default_agent()

    assert restarted.id == initial.id


def test_prompt_revision_publish_changes_only_new_runs(session) -> None:
    config = _settings()
    agent_service = AgentConfigService(session, config)
    agent_service.ensure_default_agent()
    old_run = agent_service.ensure_intake_run("old-prompt-run")
    definition = session.scalar(
        select(PromptDefinition).where(
            PromptDefinition.tenant_id == SYSTEM_TENANT_ID,
            PromptDefinition.node_key == "intake_chat",
        )
    )
    assert definition is not None
    revision = PromptConfigService(session, config).revise_definition(
        definition.id,
        content="# Custom Intake Chat\n\nPROMPT_REVISION_TWO_MARKER",
    )
    draft = agent_service.create_draft(DEFAULT_AGENT_DEFINITION_ID)
    agent_service.set_draft_node_prompt(draft.id, "intake_chat", revision.id)
    published = agent_service.publish_draft(draft.id)
    new_run = agent_service.ensure_intake_run("new-prompt-run")

    old_prompt = old_run.resolved_config_snapshot["nodes"]["intake_chat"]["prompt"]
    new_prompt = new_run.resolved_config_snapshot["nodes"]["intake_chat"]["prompt"]
    assert "PROMPT_REVISION_TWO_MARKER" not in old_prompt["content"]
    assert new_prompt["revision_id"] == revision.id
    assert "PROMPT_REVISION_TWO_MARKER" in new_prompt["content"]
    assert new_run.agent_version_id == published.id
    assert old_run.agent_version_id != new_run.agent_version_id
    assert revision.smoke_test_status == "NOT_RUN"


def test_prompt_static_validation_enforces_placeholders_boundaries_and_skills(session) -> None:
    service = PromptConfigService(session, _settings())

    with pytest.raises(ValueError, match="placeholders"):
        service.validate_prompt(
            node_key="analysis_chat",
            content="# Prompt\nHello {{unknown_value}}",
            skills=[],
        )
    with pytest.raises(ValueError, match="code-owned section"):
        service.validate_prompt(
            node_key="analysis_chat",
            content="# Prompt\n<dynamic_context trust=\"trusted\">",
            skills=[],
        )
    with pytest.raises(ValueError, match="complete code-owned Skill set"):
        service.validate_prompt(
            node_key="intake_agent",
            content="# Intake Agent",
            skills=[],
        )
    with pytest.raises(ValueError, match="does not accept"):
        service.validate_prompt(
            node_key="analysis_chat",
            content="# Analysis Chat",
            skills=[
                {
                    "name": "01_identity_resolution",
                    "content": "# Skill: `identity_resolution`",
                }
            ],
        )


def test_node_binding_rejects_prompt_from_another_node(session) -> None:
    config = _settings()
    agent_service = AgentConfigService(session, config)
    agent_service.ensure_default_agent()
    _, revision = PromptConfigService(session, config).create_definition(
        tenant_id=SYSTEM_TENANT_ID,
        name="Evidence only",
        slug="evidence-only-prompt",
        node_key="evidence_verify",
        content="# Evidence Verify\n\nReturn verified evidence.",
    )
    draft = agent_service.create_draft(DEFAULT_AGENT_DEFINITION_ID)

    with pytest.raises(ValueError, match="not intake_chat"):
        agent_service.set_draft_node_prompt(draft.id, "intake_chat", revision.id)


def test_structured_llm_reads_prompt_and_skills_from_snapshot(tmp_path) -> None:
    content = "# Snapshot Intake Agent\n\nSNAPSHOT_ONLY_PROMPT\n"
    skill_content = "# Skill: `identity_resolution`\n\nSNAPSHOT_ONLY_SKILL\n"
    prompt = {
        "revision_id": "prompt-revision-test",
        "version": 7,
        "node_key": "intake_agent",
        "content": content,
        "content_hash": canonical_hash(content),
        "skills": [
            {
                "name": "01_identity_resolution",
                "content": skill_content,
                "content_hash": canonical_hash(skill_content),
            }
        ],
    }
    llm = StructuredLLM(
        _settings(prompt_dir=tmp_path / "missing-prompts"),
        resolved_config={"nodes": {"intake_agent": {"prompt": prompt}}},
    )

    rendered = llm._system_prompt("intake_agent")

    assert "SNAPSHOT_ONLY_PROMPT" in rendered
    assert "SNAPSHOT_ONLY_SKILL" in rendered
    assert llm._prompt_version(prompt) == "v7"


def test_structured_llm_rejects_tampered_snapshot_prompt() -> None:
    prompt = {
        "revision_id": "prompt-revision-test",
        "node_key": "analysis_chat",
        "content": "# Tampered Prompt\n",
        "content_hash": canonical_hash("# Original Prompt\n"),
        "skills": [],
    }
    llm = StructuredLLM(
        _settings(),
        resolved_config={"nodes": {"analysis_chat": {"prompt": prompt}}},
    )

    with pytest.raises(LLMUnavailable, match="integrity"):
        llm._system_prompt("analysis_chat")


def test_prompt_revision_integrity_rejects_database_tampering(session) -> None:
    config = _settings()
    AgentConfigService(session, config).ensure_default_agent()
    revision = session.scalar(select(PromptRevision))
    assert revision is not None
    revision.content = "# Tampered database content\n"
    session.commit()

    with pytest.raises(ValueError, match="content failed integrity"):
        PromptConfigService(session, config).resolve_revision(revision.id)


def test_prompt_admin_api_create_list_and_validate(monkeypatch) -> None:
    monkeypatch.setattr(settings, "agent_admin_enabled", True)
    slug = f"managed-prompt-{uuid4().hex[:12]}"
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/admin/prompts",
            json={
                "name": "Managed Analysis Prompt",
                "slug": slug,
                "node_key": "analysis_chat",
                "content": "# Managed Analysis Chat\n\nAnswer only from supplied evidence.",
            },
        )
        assert created.status_code == 200, created.text
        revision_id = created.json()["active_revision"]["id"]
        validated = client.post(
            f"/api/v1/admin/prompt-revisions/{revision_id}/validate"
        )
        listed = client.get("/api/v1/admin/prompts?node_key=analysis_chat")

    assert validated.status_code == 200, validated.text
    assert validated.json()["valid"] is True
    assert validated.json()["output_schema"] == "TaskChatResult"
    assert listed.status_code == 200, listed.text
    assert any(item["slug"] == slug for item in listed.json())
