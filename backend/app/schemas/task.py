from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


TaskStatus = Literal[
    "PENDING",
    "TRANSCRIBING",
    "CONTEXT_EXTRACTING",
    "EXTRACTING",
    "RULE_EXTRACTING",
    "LLM_UNDERSTANDING",
    "RESOLVING_ENTITIES",
    "NEEDS_CONFIRMATION",
    "PLANNING_WEB_SEARCH",
    "WEB_SEARCHING",
    "WEB_FETCHING",
    "VERIFYING_WEB_RESULTS",
    "PLANNING_PROJECT_SEARCH",
    "PROJECT_SEARCHING",
    "RERANKING_PROJECTS",
    "ANALYZING_ASSOCIATIONS",
    "GENERATING_REPORT_CONTENT",
    "GENERATING",
    "RENDERING_REPORT",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
]
IntentType = Literal[
    "MEETING_PREPARATION",
    "PERSON_BACKGROUND_RESEARCH",
    "INTERNAL_PROJECT_QUERY",
    "RESOURCE_RELATION_QUERY",
    "PROJECT_ADVANCEMENT_ADVICE",
    "REPORT_GENERATION",
]
EntityType = Literal["PERSON", "ORGANIZATION", "PROJECT"]
EntityResolution = Literal["CONFIRMED", "NEEDS_CONFIRMATION", "MISSING"]
StatementType = Literal["FACT", "INFERENCE", "RECOMMENDATION"]


class Person(BaseModel):
    name: str | None = None
    organization: str | None = None
    title: str | None = None


class ExtractedInfo(BaseModel):
    event_type: Literal["宴请", "拜访", "会议", "其他"]
    event_time: str | None = None
    event_location: str | None = None
    people: list[Person] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class EntityMention(BaseModel):
    mention: str
    canonical_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    organization: str | None = None
    title: str | None = None
    region: str | None = None
    evidence_text: str = ""
    confidence: float = Field(default=0, ge=0, le=1)
    resolution: EntityResolution

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_resolution(cls, value):
        if isinstance(value, dict) and "resolution" not in value:
            value = dict(value)
            if value.get("needs_confirmation"):
                value["resolution"] = "NEEDS_CONFIRMATION"
            elif value.get("canonical_name"):
                value["resolution"] = "CONFIRMED"
            else:
                value["resolution"] = "MISSING"
        return value


class ProjectMention(BaseModel):
    mention: str
    canonical_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    business_directions: list[str] = Field(default_factory=list)
    evidence_text: str = ""
    confidence: float = Field(default=0, ge=0, le=1)
    needs_confirmation: bool = False


class IntentUnderstanding(BaseModel):
    intents: list[IntentType]
    people: list[EntityMention] = Field(default_factory=list)
    organizations: list[EntityMention] = Field(default_factory=list)
    projects: list[ProjectMention] = Field(default_factory=list)
    event_type: Literal["宴请", "拜访", "会议", "其他"]
    event_time: str | None = None
    event_location: str | None = None
    business_directions: list[str] = Field(default_factory=list)
    focus_questions: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0, le=1)


class CandidateOption(BaseModel):
    candidate_id: str
    entity_type: EntityType
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    organization: str | None = None
    title: str | None = None
    region: str | None = None
    reason: str
    confidence: float = Field(ge=0, le=1)
    source_url: str | None = None
    evidence_quote: str | None = None


class ConfirmationItem(BaseModel):
    mention: str
    entity_type: EntityType
    candidates: list[CandidateOption]
    required: bool = True


class ConfirmationRequest(BaseModel):
    version: int
    items: list[ConfirmationItem]


class ConfirmationSelection(BaseModel):
    mention: str
    candidate_id: str | None = None
    manual_value: str | None = None


class ConfirmationPayload(BaseModel):
    confirmation_version: int
    selections: list[ConfirmationSelection]


class ConfirmedEntity(BaseModel):
    candidate_id: str | None = None
    entity_type: EntityType
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    organization: str | None = None
    title: str | None = None
    region: str | None = None
    confirmed_by: Literal["AUTO", "USER"]


class ConfirmedContext(BaseModel):
    intents: list[IntentType]
    entities: list[ConfirmedEntity]
    event_type: Literal["宴请", "拜访", "会议", "其他"]
    event_time: str | None = None
    event_location: str | None = None
    business_directions: list[str] = Field(default_factory=list)
    focus_questions: list[str] = Field(default_factory=list)


class WebSearchQuery(BaseModel):
    query: str = Field(min_length=2, max_length=120)
    purpose: str
    target_person: str | None = None
    target_organization: str | None = None
    required_terms: list[str] = Field(default_factory=list, max_length=8)


class WebSearchPlan(BaseModel):
    queries: list[WebSearchQuery] = Field(min_length=1, max_length=6)


class SearchResult(BaseModel):
    web_result_id: str = ""
    title: str
    url: str
    content: str = ""
    query: str
    rank: int
    published_at: datetime | None = None


class WebPage(BaseModel):
    web_result_id: str = ""
    title: str
    url: str
    raw_content: str
    rank: int
    published_at: datetime | None = None


class WebEvidence(BaseModel):
    evidence_id: str
    quote: str
    claim: str
    matched_terms: list[str] = Field(default_factory=list)


class PublicClaim(BaseModel):
    web_result_id: str = ""
    evidence_id: str = ""
    subject: str
    claim: str
    evidence_quote: str = ""
    source_title: str
    source_url: str
    published_at: datetime | None = None
    matched_keywords: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1, ge=0, le=1)


class WebVerification(BaseModel):
    web_result_id: str
    keep: bool
    matched_person: str | None = None
    matched_organization: str | None = None
    identity_reason: str
    confidence: float = Field(ge=0, le=1)
    same_name_risk: bool
    conflicts: list[str] = Field(default_factory=list)
    evidence: list[WebEvidence] = Field(default_factory=list)


class WebVerificationBatch(BaseModel):
    results: list[WebVerification]


class ProjectQueryPlan(BaseModel):
    person_names: list[str] = Field(default_factory=list)
    organization_names: list[str] = Field(default_factory=list)
    project_names: list[str] = Field(default_factory=list)
    business_terms: list[str] = Field(default_factory=list)
    statuses: list[Literal["ACTIVE", "COMPLETED"]] = Field(
        default_factory=lambda: ["ACTIVE", "COMPLETED"]
    )
    purpose: str = "资源调查"


class ProjectResult(BaseModel):
    project_id: str
    project_name: str
    project_aliases: list[str] = Field(default_factory=list)
    customer_name: str
    contact_name: str | None = None
    status: Literal["ACTIVE", "COMPLETED"]
    owner_name: str
    start_date: date
    end_date: date | None = None
    description: str
    match_type: Literal[
        "PERSON_EXACT", "ORG_EXACT", "PROJECT_EXACT", "TEXT_MATCH", "VECTOR_MATCH"
    ]
    similarity: float | None = None


class ProjectRanking(BaseModel):
    project_id: str
    relevance_score: int = Field(ge=0, le=100)
    relevance_reason: str
    recommended_use: str
    related_internal_resource: str | None = None
    confidence: float = Field(ge=0, le=1)
    evidence_refs: list[str] = Field(default_factory=list)


class ProjectRankingBatch(BaseModel):
    rankings: list[ProjectRanking]


class EvidenceBackedItem(BaseModel):
    text: str
    statement_type: StatementType
    evidence_refs: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class AssociationAnalysis(BaseModel):
    key_findings: list[EvidenceBackedItem] = Field(default_factory=list)
    related_projects: list[EvidenceBackedItem] = Field(default_factory=list)
    available_resources: list[EvidenceBackedItem] = Field(default_factory=list)
    recommended_topics: list[EvidenceBackedItem] = Field(default_factory=list)
    risks: list[EvidenceBackedItem] = Field(default_factory=list)
    information_gaps: list[EvidenceBackedItem] = Field(default_factory=list)
    next_actions: list[EvidenceBackedItem] = Field(default_factory=list)


class ActionBrief(BaseModel):
    destination: str | None = None
    meeting_people: list[str] = Field(default_factory=list)
    objective: str
    discussion_topics: list[str] = Field(default_factory=list)
    internal_contacts: list[str] = Field(default_factory=list)
    preparation_items: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class GeneratedReportContent(BaseModel):
    task_overview: list[EvidenceBackedItem] = Field(default_factory=list)
    person_and_company_summary: list[EvidenceBackedItem] = Field(default_factory=list)
    public_information_summary: list[EvidenceBackedItem] = Field(default_factory=list)
    priority_projects: list[EvidenceBackedItem] = Field(default_factory=list)
    resource_analysis: list[EvidenceBackedItem] = Field(default_factory=list)
    recommended_topics: list[EvidenceBackedItem] = Field(default_factory=list)
    advancement_advice: list[EvidenceBackedItem] = Field(default_factory=list)
    preparation_items: list[EvidenceBackedItem] = Field(default_factory=list)
    gaps_and_risks: list[EvidenceBackedItem] = Field(default_factory=list)
    action_brief: ActionBrief


class TextTaskRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class TaskCreated(BaseModel):
    task_id: UUID
    status: Literal["PENDING"] = "PENDING"
    input_type: Literal["text", "audio"]


class TaskResponse(BaseModel):
    task_id: UUID
    status: TaskStatus
    input_type: Literal["text", "audio"]
    input_text: str | None = None
    extracted_info: ExtractedInfo | None = None
    llm_understanding: IntentUnderstanding | None = None
    confirmation_request: ConfirmationRequest | None = None
    confirmed_context: ConfirmedContext | None = None
    web_search_plan: WebSearchPlan | None = None
    web_search_status: str | None = None
    web_fetch_status: str | None = None
    verified_web_results: list[WebVerification] = Field(default_factory=list)
    public_claims: list[PublicClaim] = Field(default_factory=list)
    project_query_plan: ProjectQueryPlan | None = None
    internal_search_status: str | None = None
    internal_results: list[ProjectResult] = Field(default_factory=list)
    ranked_internal_results: list[ProjectRanking] = Field(default_factory=list)
    association_analysis: AssociationAnalysis | None = None
    detailed_report_markdown: str | None = None
    action_brief_markdown: str | None = None
    report_markdown: str | None = None
    degraded_nodes: list[str] = Field(default_factory=list)
    error_message: str | None = None
