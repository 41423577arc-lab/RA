import json
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.models.database import (
    AgentToolBinding,
    Base,
    ConfigSecret,
    McpServerDefinition,
    McpServerRevision,
    ToolMappingDefinition,
    ToolMappingRevision,
)
from app.services.agent_config.mcp import DEFAULT_TOOL_MAPPINGS, McpConfigService
from app.services.agent_config.service import (
    DEFAULT_AGENT_DEFINITION_ID,
    SYSTEM_TENANT_ID,
    AgentConfigService,
)


ROOT = Path(__file__).resolve().parents[2]


def _settings(**updates) -> Settings:
    return Settings(
        _env_file=None,
        database_url="sqlite://",
        prompt_dir=ROOT / "backend/prompts",
        report_template=ROOT / "backend/templates/report.md.j2",
        detailed_report_template=ROOT / "backend/templates/detailed_report.md.j2",
        action_brief_template=ROOT / "backend/templates/action_brief.md.j2",
        mcp_server_url="http://default-mcp:8001/mcp",
        agent_allow_private_mcp_urls=True,
        agent_secret_key=Fernet.generate_key().decode("ascii"),
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
        yield database_session
    Base.metadata.drop_all(engine)


def test_default_mcp_registry_and_agent_bindings_are_idempotent(session) -> None:
    service = AgentConfigService(session, _settings())
    version = service.ensure_default_agent()
    again = service.ensure_default_agent()
    snapshot, _ = service.resolve_published(DEFAULT_AGENT_DEFINITION_ID)

    assert again.id == version.id
    assert session.scalar(select(func.count()).select_from(McpServerDefinition)) == 1
    assert session.scalar(select(func.count()).select_from(McpServerRevision)) == 1
    assert session.scalar(select(func.count()).select_from(ToolMappingDefinition)) == 2
    assert session.scalar(select(func.count()).select_from(ToolMappingRevision)) == 2
    assert session.scalar(select(func.count()).select_from(AgentToolBinding)) == 2
    mappings = {
        item["logical_tool_key"]: item for item in snapshot["tool_mappings"]
    }
    assert set(DEFAULT_TOOL_MAPPINGS) <= set(mappings)
    assert mappings["identity.find_candidates"]["allowed_nodes"] == [
        "intake_agent",
        "intake_identity_update",
    ]
    assert mappings["projects.search"]["allowed_nodes"] == ["research_pipeline"]


def test_new_mcp_revision_only_affects_runs_from_new_agent_version(session) -> None:
    config = _settings()
    agent_service = AgentConfigService(session, config)
    agent_service.ensure_default_agent()
    old_run = agent_service.ensure_intake_run("old-run")
    old_snapshot = json.loads(json.dumps(old_run.resolved_config_snapshot))

    mcp_service = McpConfigService(session, config)
    server_definition, server_revision = mcp_service.create_server(
        tenant_id=SYSTEM_TENANT_ID,
        name="Customer CRM",
        slug="customer-crm",
        url="http://customer-crm:9000/mcp",
        authentication_type="bearer",
        api_token="sensitive-customer-token",
        secret_ref=None,
        timeout_seconds=20,
    )
    mapping_definition, mapping_revision = mcp_service.create_mapping(
        tenant_id=SYSTEM_TENANT_ID,
        name="Customer project search",
        logical_tool_key="customer.projects_search",
        mcp_server_revision_id=server_revision.id,
        remote_tool_name="query_projects",
        adapter_key="declarative",
        input_mapping={"rename": {"organization_names": "customer_names"}},
        output_mapping={"rename": {"xm_mc": "project_name"}},
        timeout_seconds=20,
    )
    assert server_definition.active_revision_id == server_revision.id
    assert mapping_definition.active_revision_id == mapping_revision.id
    secret = session.scalar(select(ConfigSecret))
    assert secret is not None
    assert "sensitive-customer-token" not in secret.ciphertext

    draft = agent_service.create_draft(DEFAULT_AGENT_DEFINITION_ID)
    agent_service.set_draft_tool_mapping(
        draft.id,
        "customer.projects_search",
        mapping_revision.id,
        ["research_pipeline"],
    )
    published = agent_service.publish_draft(draft.id)
    new_run = agent_service.ensure_intake_run("new-run")

    serialized = json.dumps(new_run.resolved_config_snapshot, ensure_ascii=False)
    assert published.id == new_run.agent_version_id
    assert old_run.resolved_config_snapshot == old_snapshot
    assert "sensitive-customer-token" not in serialized
    assert server_revision.secret_ref in serialized
    assert "customer.projects_search" in serialized


def test_mapping_validation_rejects_executable_or_nested_mapping(session) -> None:
    config = _settings()
    AgentConfigService(session, config).ensure_default_agent()
    server_revision = session.scalar(select(McpServerRevision))

    with pytest.raises(ValueError, match="Unsupported declarative mapping"):
        McpConfigService(session, config).create_mapping(
            tenant_id=SYSTEM_TENANT_ID,
            name="Unsafe mapping",
            logical_tool_key="unsafe.mapping",
            mcp_server_revision_id=server_revision.id,
            remote_tool_name="unsafe_tool",
            adapter_key="declarative",
            input_mapping={"expression": "__import__('os')"},
            output_mapping={},
            timeout_seconds=10,
        )
