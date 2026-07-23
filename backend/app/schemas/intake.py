from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.schemas.task import ConfirmationRequest


class IntakeMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2_000)


class IntakeChatRequest(BaseModel):
    session_id: UUID = Field(default_factory=uuid4)
    messages: list[IntakeMessage] = Field(min_length=1, max_length=30)
    audio_job_id: UUID | None = None


class IntakeStructuredContext(BaseModel):
    people: list[str] = Field(default_factory=list, max_length=20)
    organizations: list[str] = Field(default_factory=list, max_length=20)
    projects: list[str] = Field(default_factory=list, max_length=20)
    focus_questions: list[str] = Field(default_factory=list, max_length=20)


class IntakeChatResult(BaseModel):
    assistant_reply: str = Field(min_length=1, max_length=1_000)
    analysis_input: str = Field(min_length=1, max_length=10_000)
    ready_to_analyze: bool
    missing_information: list[str] = Field(default_factory=list, max_length=8)
    structured_context: IntakeStructuredContext = Field(default_factory=IntakeStructuredContext)
    next_action: Literal["ASK_USER", "LOOKUP_ENTITY", "PROPOSE_READY"] = "ASK_USER"


class IntakeFollowupResult(BaseModel):
    assistant_reply: str = Field(min_length=1, max_length=1_000)


class IntakeChatResponse(IntakeChatResult):
    session_id: UUID
    status: Literal[
        "COLLECTING",
        "PROCESSING_AUDIO",
        "NEEDS_CONFIRMATION",
        "READY",
        "STARTING_ANALYSIS",
        "ANALYZING",
    ]
    version: int = Field(ge=0)
    confirmation_request: ConfirmationRequest | None = None


class IntakeSessionResponse(IntakeChatResponse):
    messages: list[IntakeMessage]
    research_task_id: UUID | None = None
    active_audio_job: dict | None = None


class StartAnalysisRequest(BaseModel):
    expected_version: int | None = Field(default=None, ge=0)


class IntakeAudioJobResponse(BaseModel):
    job_id: UUID
    session_id: UUID
    status: Literal[
        "QUEUED", "TRANSCRIBING", "NEEDS_REVIEW", "TRANSCRIBED", "FAILED"
    ]
    transcript: str | None = None
    corrected_transcript: str | None = None
    error_message: str | None = None
    retry_count: int = 0
