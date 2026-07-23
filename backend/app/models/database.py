import uuid
from datetime import datetime

from sqlalchemy import DateTime, JSON, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


INTAKE_JSON = JSON().with_variant(JSONB(), "postgresql")


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
