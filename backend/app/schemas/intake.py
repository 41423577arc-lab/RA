from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, BeforeValidator, Field, model_validator

from app.schemas.task import ConfirmationRequest


IntakeContextFieldStatus = Literal[
    "MISSING", "CANDIDATE", "AMBIGUOUS", "CONFIRMED", "NOT_REQUIRED"
]
LegacyIntakeFieldStatus = Literal[
    "NEEDS_COMPLETION",
    "STANDARD_COMPLETE",
    "USER_CONFIRMED",
    "NOT_PROVIDED",
]
IntakeAction = Literal[
    "SEARCH_INTERNAL", "SEARCH_PUBLIC", "ASK_USER", "READY"
]
IntakeToolAction = Literal["SEARCH_INTERNAL", "SEARCH_PUBLIC"]
IntakeEntityType = Literal["ORGANIZATION", "PERSON"]
IntakeQuestionKind = Literal["GENERAL", "CONFIRM_CANDIDATE", "CONFIRM_NEW_ENTITY"]


def _normalize_legacy_intake_action(value: object) -> object:
    if not isinstance(value, str):
        return value
    return {
        "LOOKUP_INTERNAL": "SEARCH_INTERNAL",
        "SEARCH_EXTERNAL": "SEARCH_PUBLIC",
        "REQUEST_CONFIRMATION": "ASK_USER",
        "PROPOSE_READY": "READY",
    }.get(value, value)


CompatibleIntakeAction = Annotated[
    IntakeAction, BeforeValidator(_normalize_legacy_intake_action)
]


class IntakeMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2_000)


class IntakeChatRequest(BaseModel):
    session_id: UUID = Field(default_factory=uuid4)
    messages: list[IntakeMessage] = Field(min_length=1, max_length=30)
    audio_job_id: UUID | None = None


class IntakeEntityResolution(BaseModel):
    candidate_id: str | None = None
    entity_type: IntakeEntityType
    canonical_name: str
    mention: str
    organization: str | None = None
    title: str | None = None
    region: str | None = None
    confidence: float = Field(default=1, ge=0, le=1)
    confirmed_by: Literal[
        "USER_INPUT", "INTERNAL", "EXTERNAL_AUTO", "AUTO", "USER"
    ]
    source_url: str | None = None
    evidence_quote: str | None = None


class IntakeToolAttempt(BaseModel):
    action: IntakeToolAction
    target_fields: list[str] = Field(default_factory=list, max_length=20)
    query: str = Field(min_length=1, max_length=500)
    technical_status: Literal["SUCCESS", "FAILED"]
    information_status: Literal["RESOLVED", "PARTIAL", "NO_RESULT"]
    observation: str = Field(default="", max_length=4_000)

class IntakeResolutionResult(BaseModel):
    status: Literal["RESOLVED", "PARTIAL", "UNRESOLVED"]
    updated_fields: list[str] = Field(default_factory=list, max_length=20)
    summary: str = Field(default="", max_length=2_000)


class IntakeStructuredContext(BaseModel):
    people: list[str] = Field(default_factory=list, max_length=20)
    people_details: list["IntakePersonCandidate"] = Field(default_factory=list, max_length=20)
    organizations: list[str] = Field(default_factory=list, max_length=20)
    projects: list[str] = Field(default_factory=list, max_length=20)
    business_directions: list[str] = Field(default_factory=list, max_length=20)
    focus_questions: list[str] = Field(default_factory=list, max_length=20)
    event_type: Literal["宴请", "拜访", "会议", "其他"] | None = None
    event_time: str | None = None
    event_location: str | None = None
    entity_assessments: list["IntakeEntityAssessment"] = Field(default_factory=list, max_length=40)
    entity_resolutions: list[IntakeEntityResolution] = Field(default_factory=list, max_length=40)
    field_states: dict[str, "IntakeFieldState"] = Field(default_factory=dict)
    target_fields: list[str] = Field(default_factory=list, max_length=20)
    next_action: IntakeAction | None = None
    success_criteria: list[str] = Field(default_factory=list, max_length=20)
    resolution_result: IntakeResolutionResult | None = None
    user_question: str | None = Field(default=None, min_length=1, max_length=1_000)
    tool_attempts: list[IntakeToolAttempt] = Field(default_factory=list, max_length=20)
    final_confirmation: "IntakeFinalConfirmation | None" = None


class IntakePersonCandidate(BaseModel):
    name: str | None = None
    title: str | None = None
    organization: str | None = None


class IntakeEntityAssessment(BaseModel):
    entity_type: IntakeEntityType
    mention: str
    is_standard: bool
    reason: str = ""


class IntakeFieldState(BaseModel):
    status: IntakeContextFieldStatus | LegacyIntakeFieldStatus
    required: bool
    value: Any | None = None
    source: str | None = None
    reason: str = ""


class IntakeFinalConfirmation(BaseModel):
    version: int = Field(ge=1)
    question: str = Field(min_length=1, max_length=1_000)
    status: Literal["PENDING", "CONFIRMED"] = "PENDING"


class IntakeFinalConfirmationResult(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)


class IntakeChatResult(BaseModel):
    assistant_reply: str = Field(min_length=1, max_length=1_000)
    analysis_input: str = Field(min_length=1, max_length=10_000)
    ready_to_analyze: bool
    missing_information: list[str] = Field(default_factory=list, max_length=8)
    structured_context: IntakeStructuredContext = Field(default_factory=IntakeStructuredContext)
    next_action: CompatibleIntakeAction = "ASK_USER"


class IntakeFollowupResult(BaseModel):
    assistant_reply: str = Field(min_length=1, max_length=1_000)
    next_action: CompatibleIntakeAction = "ASK_USER"


class IntakeReadinessResult(BaseModel):
    assistant_reply: str = Field(min_length=1, max_length=1_000)
    ready_to_analyze: bool
    missing_information: list[str] = Field(default_factory=list, max_length=8)
    next_action: CompatibleIntakeAction

    @model_validator(mode="after")
    def validate_decision(self):
        if self.ready_to_analyze:
            if self.next_action != "READY" or self.missing_information:
                raise ValueError("就绪结论必须使用 READY 且不能包含缺失信息")
        elif self.next_action != "ASK_USER":
            raise ValueError("未就绪结论必须使用 ASK_USER")
        return self


class ConfirmIntakeSummaryRequest(BaseModel):
    expected_version: int = Field(ge=1)


class ExternalIdentityCandidate(BaseModel):
    mention: str
    entity_type: IntakeEntityType
    canonical_name: str
    organization: str | None = None
    title: str | None = None
    source_url: str
    evidence_quote: str
    confidence: float = Field(ge=0, le=1)


class ExternalIdentityNormalizationResult(BaseModel):
    candidates: list[ExternalIdentityCandidate] = Field(default_factory=list, max_length=10)


class IntakeChatResponse(IntakeChatResult):
    session_id: UUID
    status: Literal[
        "COLLECTING",
        "PROCESSING_AUDIO",
        "NEEDS_CONFIRMATION",
        "AWAITING_FINAL_CONFIRMATION",
        "READY",
        "STARTING_ANALYSIS",
        "ANALYZING",
    ]
    version: int = Field(ge=0)
    confirmation_request: ConfirmationRequest | None = None
    final_confirmation: IntakeFinalConfirmation | None = None


class IntakeSessionResponse(IntakeChatResponse):
    messages: list[IntakeMessage]
    research_task_id: UUID | None = None
    active_audio_job: dict | None = None


class IntakeActivityResponse(BaseModel):
    session_id: UUID
    phase: Literal[
        "IDLE",
        "THINKING",
        "CHECKING_CONTEXT",
        "CALLING_TOOL",
        "PROCESSING_TOOL_RESULT",
        "COMPLETED",
        "FAILED",
    ]
    detail: str
    active: bool
    tool_name: str | None = None
    sequence: int = Field(ge=0)
    updated_at: datetime | None = None


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
