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
WebEvidenceKind = Literal["IDENTITY", "ORGANIZATION_TOPIC"]


class Person(BaseModel):  # 人物信息模型
    name: str | None = None
    organization: str | None = None
    title: str | None = None


class ExtractedInfo(BaseModel):  # 基础信息提取结果模型
    event_type: Literal["宴请", "拜访", "会议", "其他"]
    event_time: str | None = None
    event_location: str | None = None
    people: list[Person] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class EntityMention(BaseModel):  # 实体提及与解析信息模型
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


class ProjectMention(BaseModel):  # 项目提及信息模型
    mention: str
    canonical_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    business_directions: list[str] = Field(default_factory=list)
    evidence_text: str = ""
    confidence: float = Field(default=0, ge=0, le=1)
    needs_confirmation: bool = False


class IntentUnderstanding(BaseModel):  # 意图理解结果模型
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


class CandidateOption(BaseModel):  # 实体候选选项模型
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


class ConfirmationItem(BaseModel):  # 待确认实体项模型
    mention: str
    entity_type: EntityType
    candidates: list[CandidateOption]
    required: bool = True


class ConfirmationRequest(BaseModel):  # 身份确认请求模型
    version: int
    items: list[ConfirmationItem]


class ConfirmationSelection(BaseModel):  # 用户确认选择模型
    mention: str
    candidate_id: str | None = None
    manual_value: str | None = None


class ConfirmationPayload(BaseModel):  # 身份确认提交数据模型
    confirmation_version: int
    selections: list[ConfirmationSelection]


class ConfirmedEntity(BaseModel):  # 已确认实体模型
    candidate_id: str | None = None
    entity_type: EntityType
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    organization: str | None = None
    title: str | None = None
    region: str | None = None
    confirmed_by: Literal["AUTO", "USER"]


class ConfirmedContext(BaseModel):  # 已确认分析上下文模型
    intents: list[IntentType]
    entities: list[ConfirmedEntity]
    event_type: Literal["宴请", "拜访", "会议", "其他"]
    event_time: str | None = None
    event_location: str | None = None
    business_directions: list[str] = Field(default_factory=list)
    focus_questions: list[str] = Field(default_factory=list)


class WebSearchQuery(BaseModel):  # 联网检索查询项模型
    query: str = Field(min_length=2, max_length=120)
    purpose: str
    target_person: str | None = None
    target_organization: str | None = None
    required_terms: list[str] = Field(default_factory=list, max_length=8)


class WebSearchPlan(BaseModel):  # 联网检索计划模型
    queries: list[WebSearchQuery] = Field(min_length=1, max_length=6)


class SearchResult(BaseModel):  # 联网搜索结果模型
    web_result_id: str = ""
    title: str
    url: str
    content: str = ""
    query: str
    rank: int
    target_person: str | None = None
    target_organization: str | None = None
    published_at: datetime | None = None


class WebPage(BaseModel):  # 网页正文内容模型
    web_result_id: str = ""
    title: str
    url: str
    raw_content: str
    rank: int
    query: str = ""
    target_person: str | None = None
    target_organization: str | None = None
    search_snippet: str = ""
    content_source: Literal["PAGE_TEXT", "SEARCH_SNIPPET"] = "PAGE_TEXT"
    published_at: datetime | None = None


class WebEvidence(BaseModel):  # 网页证据模型
    evidence_id: str
    quote: str
    claim: str
    matched_terms: list[str] = Field(default_factory=list)


class PublicClaim(BaseModel):  # 经核验的公开事实模型
    web_result_id: str = ""
    evidence_id: str = ""
    subject: str
    claim: str
    evidence_quote: str = ""
    source_title: str
    source_url: str
    evidence_source: Literal["PAGE_TEXT", "SEARCH_SNIPPET"] = "PAGE_TEXT"
    published_at: datetime | None = None
    matched_keywords: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1, ge=0, le=1)


class WebVerification(BaseModel):  # 网页结果核验模型
    web_result_id: str
    keep: bool
    matched_person: str | None = None
    matched_organization: str | None = None
    identity_reason: str
    confidence: float = Field(ge=0, le=1)
    same_name_risk: bool
    conflicts: list[str] = Field(default_factory=list)
    evidence: list[WebEvidence] = Field(default_factory=list)


class WebVerificationBatch(BaseModel):  # 网页结果批量核验模型
    results: list[WebVerification]


class WebEvidenceCandidate(BaseModel):  # 送入模型的精简网页证据候选
    candidate_id: str = Field(min_length=1, max_length=64)
    web_result_id: str = Field(min_length=1, max_length=64)
    kind: WebEvidenceKind
    text: str = Field(min_length=10, max_length=1000)
    target_person: str | None = None
    target_organization: str | None = None
    matched_terms: list[str] = Field(default_factory=list, max_length=8)


class SupportedWebEvidence(BaseModel):  # 模型只返回支持结论的候选
    candidate_id: str = Field(min_length=1, max_length=64)
    position: str | None = Field(default=None, max_length=100)


class WebEvidenceDecision(BaseModel):  # 未列出的候选视为不支持
    supported: list[SupportedWebEvidence] = Field(default_factory=list, max_length=30)
    ambiguous_candidate_ids: list[str] = Field(default_factory=list, max_length=30)


class ProjectQueryPlan(BaseModel):  # 内部项目查询计划模型
    person_names: list[str] = Field(default_factory=list)
    organization_names: list[str] = Field(default_factory=list)
    project_names: list[str] = Field(default_factory=list)
    business_terms: list[str] = Field(default_factory=list)
    statuses: list[Literal["ACTIVE", "COMPLETED"]] = Field(
        default_factory=lambda: ["ACTIVE", "COMPLETED"]
    )
    purpose: str = "资源调查"


class ProjectResult(BaseModel):  # 内部项目查询结果模型
    project_id: str
    project_name: str
    project_aliases: list[str] = Field(default_factory=list)
    customer_name: str
    contact_name: str | None = None
    customer_contact_title: str | None = None
    customer_contact_phone: str | None = None
    status: Literal["ACTIVE", "COMPLETED"]
    owner_name: str
    owner_phone: str | None = None
    owner_email: str | None = None
    owner_manager_name: str | None = None
    owner_region: str | None = None
    start_date: date
    end_date: date | None = None
    description: str
    sales_rep_id: str | None = None
    project_stage: str | None = None
    health_status: Literal["GREEN", "AMBER", "RED"] | None = None
    priority: Literal["P0", "P1", "P2", "P3"] | None = None
    contract_value: float | None = None
    win_probability: int | None = Field(default=None, ge=0, le=100)
    last_activity_date: date | None = None
    next_followup_date: date | None = None
    match_type: Literal[
        "PERSON_EXACT", "ORG_EXACT", "PROJECT_EXACT", "TEXT_MATCH", "VECTOR_MATCH"
    ]
    similarity: float | None = None


class ProjectRanking(BaseModel):  # 项目相关性排序结果模型
    project_id: str
    relevance_score: int = Field(ge=0, le=100)
    relevance_reason: str
    recommended_use: str
    related_internal_resource: str | None = None
    confidence: float = Field(ge=0, le=1)
    evidence_refs: list[str] = Field(default_factory=list)
    score: int | None = Field(default=None, ge=0, le=100)
    reason_codes: list[str] = Field(default_factory=list)
    rank: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def synchronize_score(self):
        if self.score is None:
            self.score = self.relevance_score
        elif self.score != self.relevance_score:
            raise ValueError("score must equal relevance_score")
        return self


class ProjectRankingBatch(BaseModel):  # 项目相关性批量排序模型
    rankings: list[ProjectRanking]


class EvidenceBackedItem(BaseModel):  # 有证据支撑的内容条目模型
    text: str
    statement_type: StatementType
    evidence_refs: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class AssociationAnalysis(BaseModel):  # 资源关联分析结果模型
    key_findings: list[EvidenceBackedItem] = Field(default_factory=list)
    related_projects: list[EvidenceBackedItem] = Field(default_factory=list)
    available_resources: list[EvidenceBackedItem] = Field(default_factory=list)
    recommended_topics: list[EvidenceBackedItem] = Field(default_factory=list)
    risks: list[EvidenceBackedItem] = Field(default_factory=list)
    information_gaps: list[EvidenceBackedItem] = Field(default_factory=list)
    next_actions: list[EvidenceBackedItem] = Field(default_factory=list)


class ActionBrief(BaseModel):  # 行动说明模型
    destination: str | None = None
    meeting_people: list[str] = Field(default_factory=list)
    objective: str
    discussion_topics: list[str] = Field(default_factory=list)
    internal_contacts: list[str] = Field(default_factory=list)
    preparation_items: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class GeneratedReportContent(BaseModel):  # 结构化报告内容模型
    task_overview: list[EvidenceBackedItem] = Field(default_factory=list)
    person_and_company_summary: list[EvidenceBackedItem] = Field(default_factory=list)
    public_information_summary: list[EvidenceBackedItem] = Field(default_factory=list)
    priority_projects: list[EvidenceBackedItem] = Field(default_factory=list, max_length=3)
    resource_analysis: list[EvidenceBackedItem] = Field(default_factory=list)
    recommended_topics: list[EvidenceBackedItem] = Field(default_factory=list)
    advancement_advice: list[EvidenceBackedItem] = Field(default_factory=list)
    preparation_items: list[EvidenceBackedItem] = Field(default_factory=list)
    gaps_and_risks: list[EvidenceBackedItem] = Field(default_factory=list)
    action_brief: ActionBrief


class TextTaskRequest(BaseModel):  # 文本任务创建请求模型
    text: str = Field(min_length=1, max_length=10_000)


class TaskCreated(BaseModel):  # 任务创建结果模型
    task_id: UUID
    status: Literal["PENDING"] = "PENDING"
    input_type: Literal["text", "audio"]


class ExecutionEventResponse(BaseModel):  # 单条执行事件响应模型
    sequence: int = Field(ge=1)
    event_type: str
    node_name: str | None = None
    status: str | None = None
    title: str
    detail: str
    payload: dict | list | str | None = None
    created_at: datetime


class ExecutionLogResponse(BaseModel):  # 执行日志响应模型
    task_id: UUID
    latest_sequence: int = Field(ge=0)
    events: list[ExecutionEventResponse] = Field(default_factory=list)


class StreamExecutionEvent(BaseModel):
    sequence: int = Field(ge=1)
    event_type: Literal[
        "PHASE_CHANGED",
        "AGENT_ACTION",
        "TOOL_STARTED",
        "TOOL_RESULT",
        "CONTEXT_UPDATED",
        "LLM_STARTED",
        "LLM_TOKEN",
        "DEGRADED",
        "DONE",
    ]
    node_name: str | None = None
    status: str | None = None
    title: str
    detail: str
    payload: dict | list | str | None = None
    created_at: datetime


class TaskChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2_000)


class TaskChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2_000)


class TaskChatResult(BaseModel):
    assistant_reply: str = Field(min_length=1, max_length=2_000)


class TaskChatResponse(BaseModel):
    task_id: UUID
    task_status: TaskStatus
    messages: list[TaskChatMessage] = Field(default_factory=list, max_length=40)


class TaskClearResponse(BaseModel):
    task_id: UUID
    cleared: Literal[True] = True
    intake_session_version: int | None = Field(default=None, ge=0)
    ready_to_analyze: bool = False


class TaskResponse(BaseModel):  # 完整任务状态响应模型
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
    analysis_chat_messages: list[TaskChatMessage] = Field(default_factory=list)
