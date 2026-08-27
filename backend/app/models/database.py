import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


INTAKE_JSON = JSON().with_variant(JSONB(), "postgresql")


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ConfigSecret(Base):
    __tablename__ = "config_secrets"
    __table_args__ = (UniqueConstraint("tenant_id", "name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    ciphertext: Mapped[str] = mapped_column(Text, nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column("key_version", nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    rotated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AgentDefinition(Base):
    __tablename__ = "agent_definitions"
    __table_args__ = (UniqueConstraint("tenant_id", "slug"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    published_version_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AgentVersion(Base):
    __tablename__ = "agent_versions"
    __table_args__ = (UniqueConstraint("agent_definition_id", "version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_definition_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_definitions.id"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="DRAFT")
    config_schema_version: Mapped[int] = mapped_column(nullable=False, default=1)
    config: Mapped[dict] = mapped_column(INTAKE_JSON, nullable=False, default=dict)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ModelConnection(Base):
    __tablename__ = "model_connections"
    __table_args__ = (UniqueConstraint("tenant_id", "slug"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    active_revision_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ModelConnectionRevision(Base):
    __tablename__ = "model_connection_revisions"
    __table_args__ = (UniqueConstraint("model_connection_id", "version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_connection_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("model_connections.id"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    base_url: Mapped[str] = mapped_column(Text, nullable=False)
    authentication_type: Mapped[str] = mapped_column(String(32), nullable=False)
    secret_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PUBLISHED")
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelProfile(Base):
    __tablename__ = "model_profiles"
    __table_args__ = (UniqueConstraint("tenant_id", "slug"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    active_revision_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ModelProfileRevision(Base):
    __tablename__ = "model_profile_revisions"
    __table_args__ = (UniqueConstraint("model_profile_id", "version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_profile_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("model_profiles.id"), nullable=False, index=True
    )
    connection_revision_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("model_connection_revisions.id"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(nullable=False)
    model_id: Mapped[str] = mapped_column(String(200), nullable=False)
    api_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    parameters: Mapped[dict] = mapped_column(INTAKE_JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PUBLISHED")
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PromptDefinition(Base):
    __tablename__ = "prompt_definitions"
    __table_args__ = (UniqueConstraint("tenant_id", "slug"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    node_key: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    active_revision_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PromptRevision(Base):
    __tablename__ = "prompt_revisions"
    __table_args__ = (UniqueConstraint("prompt_definition_id", "version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt_definition_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("prompt_definitions.id"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    required_variables: Mapped[list] = mapped_column(INTAKE_JSON, nullable=False, default=list)
    skill_bundle: Mapped[list] = mapped_column(INTAKE_JSON, nullable=False, default=list)
    validation_report: Mapped[dict] = mapped_column(INTAKE_JSON, nullable=False, default=dict)
    smoke_test_status: Mapped[str] = mapped_column(String(32), nullable=False, default="NOT_RUN")
    source: Mapped[str] = mapped_column(String(255), nullable=False, default="admin")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PUBLISHED")
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class McpServerDefinition(Base):
    __tablename__ = "mcp_server_definitions"
    __table_args__ = (UniqueConstraint("tenant_id", "slug"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    active_revision_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class McpServerRevision(Base):
    __tablename__ = "mcp_server_revisions"
    __table_args__ = (UniqueConstraint("mcp_server_definition_id", "version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    mcp_server_definition_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("mcp_server_definitions.id"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(nullable=False)
    transport: Mapped[str] = mapped_column(String(32), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    authentication_type: Mapped[str] = mapped_column(String(32), nullable=False)
    secret_ref: Mapped[str | None] = mapped_column(String(255))
    timeout_seconds: Mapped[int] = mapped_column(nullable=False, default=10)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PUBLISHED")
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ToolMappingDefinition(Base):
    __tablename__ = "tool_mapping_definitions"
    __table_args__ = (UniqueConstraint("tenant_id", "logical_tool_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    logical_tool_key: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    active_revision_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ToolMappingRevision(Base):
    __tablename__ = "tool_mapping_revisions"
    __table_args__ = (UniqueConstraint("tool_mapping_definition_id", "version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tool_mapping_definition_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tool_mapping_definitions.id"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(nullable=False)
    mcp_server_revision_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("mcp_server_revisions.id"), nullable=False, index=True
    )
    remote_tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    adapter_key: Mapped[str] = mapped_column(String(100), nullable=False, default="declarative")
    input_mapping: Mapped[dict] = mapped_column(INTAKE_JSON, nullable=False, default=dict)
    output_mapping: Mapped[dict] = mapped_column(INTAKE_JSON, nullable=False, default=dict)
    timeout_seconds: Mapped[int] = mapped_column(nullable=False, default=10)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PUBLISHED")
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentNodeBinding(Base):
    __tablename__ = "agent_node_bindings"
    __table_args__ = (UniqueConstraint("agent_version_id", "node_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_versions.id"), nullable=False, index=True
    )
    node_key: Mapped[str] = mapped_column(String(100), nullable=False)
    model_profile_revision_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("model_profile_revisions.id"), index=True
    )
    prompt_revision_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("prompt_revisions.id"), index=True
    )
    model_config: Mapped[dict] = mapped_column(INTAKE_JSON, nullable=False, default=dict)
    prompt_config: Mapped[dict] = mapped_column(INTAKE_JSON, nullable=False, default=dict)
    allowed_tools: Mapped[list] = mapped_column(INTAKE_JSON, nullable=False, default=list)


class AgentToolBinding(Base):
    __tablename__ = "agent_tool_bindings"
    __table_args__ = (UniqueConstraint("agent_version_id", "logical_tool_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_versions.id"), nullable=False, index=True
    )
    logical_tool_key: Mapped[str] = mapped_column(String(100), nullable=False)
    tool_mapping_revision_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tool_mapping_revisions.id"), nullable=False, index=True
    )
    allowed_nodes: Mapped[list] = mapped_column(INTAKE_JSON, nullable=False, default=list)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id"), nullable=False, index=True
    )
    agent_definition_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_definitions.id"), nullable=False, index=True
    )
    agent_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_versions.id"), nullable=False, index=True
    )
    config_schema_version: Mapped[int] = mapped_column(nullable=False)
    resolved_config_snapshot: Mapped[dict] = mapped_column(
        INTAKE_JSON, nullable=False, default=dict
    )
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="COLLECTING")
    intake_session_id: Mapped[str | None] = mapped_column(
        String(36), unique=True, index=True
    )
    research_task_id: Mapped[str | None] = mapped_column(
        String(36), unique=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ResearchTask(Base):
    __tablename__ = "research_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    input_type: Mapped[str] = mapped_column(String(16), nullable=False)
    audio_path: Mapped[str | None] = mapped_column(Text)
    input_text: Mapped[str | None] = mapped_column(Text)
    intake_session_id: Mapped[str | None] = mapped_column(String(36), unique=True)
    input_snapshot: Mapped[dict | None] = mapped_column(INTAKE_JSON)
    extracted_info: Mapped[dict | None] = mapped_column(JSON)
    llm_understanding: Mapped[dict | None] = mapped_column(JSON)
    confirmation_request: Mapped[dict | None] = mapped_column(JSON)
    confirmed_context: Mapped[dict | None] = mapped_column(JSON)
    confirmation_version: Mapped[int] = mapped_column(default=0)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    web_search_plan: Mapped[dict | None] = mapped_column(JSON)
    web_results: Mapped[list | None] = mapped_column(JSON)
    web_search_status: Mapped[str | None] = mapped_column(String(16))
    web_pages: Mapped[list | None] = mapped_column(JSON)
    web_fetch_status: Mapped[str | None] = mapped_column(String(16))
    public_claims: Mapped[list | None] = mapped_column(JSON)
    verified_web_results: Mapped[list | None] = mapped_column(JSON)
    project_query_plan: Mapped[dict | None] = mapped_column(JSON)
    internal_results: Mapped[list | None] = mapped_column(JSON)
    ranked_internal_results: Mapped[list | None] = mapped_column(JSON)
    internal_search_status: Mapped[str | None] = mapped_column(String(16))
    association_analysis: Mapped[dict | None] = mapped_column(JSON)
    generated_report_content: Mapped[dict | None] = mapped_column(JSON)
    detailed_report_markdown: Mapped[str | None] = mapped_column(Text)
    action_brief_markdown: Mapped[str | None] = mapped_column(Text)
    degraded_nodes: Mapped[list | None] = mapped_column(JSON, default=list)
    prompt_versions: Mapped[dict | None] = mapped_column(JSON, default=dict)
    report_markdown: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class IntakeSession(Base):
    __tablename__ = "intake_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="COLLECTING")
    messages: Mapped[list] = mapped_column(INTAKE_JSON, nullable=False, default=list)
    structured_context: Mapped[dict] = mapped_column(INTAKE_JSON, nullable=False, default=dict)
    missing_information: Mapped[list] = mapped_column(INTAKE_JSON, nullable=False, default=list)
    confirmation_request: Mapped[dict | None] = mapped_column(INTAKE_JSON)
    analysis_input: Mapped[str] = mapped_column(Text, nullable=False, default="")
    ready_to_analyze: Mapped[bool] = mapped_column(nullable=False, default=False)
    version: Mapped[int] = mapped_column(nullable=False, default=0)
    research_task_id: Mapped[str | None] = mapped_column(String(36), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class IntakeAudioJob(Base):
    __tablename__ = "intake_audio_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="QUEUED")
    audio_path: Mapped[str | None] = mapped_column(Text)
    transcript: Mapped[str | None] = mapped_column(Text)
    corrected_transcript: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class LlmCallLog(Base):
    __tablename__ = "llm_call_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    node_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    response_id: Mapped[str | None] = mapped_column(String(255))
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False, default="v1")
    latency_ms: Mapped[int] = mapped_column(nullable=False, default=0)
    input_tokens: Mapped[int | None] = mapped_column()
    output_tokens: Mapped[int | None] = mapped_column()
    error_type: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ExecutionEvent(Base):
    __tablename__ = "execution_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    scope_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    node_name: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str | None] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False, default="")
    payload: Mapped[dict | list | str | None] = mapped_column(INTAKE_JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
