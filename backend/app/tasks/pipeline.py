import re
from pathlib import Path
from typing import Protocol

from app.config import settings
from app.database import SessionLocal, TaskRepository
from app.schemas.task import (
    ConfirmedContext,
    ConfirmedEntity,
    ExtractedInfo,
    IntentUnderstanding,
    ProjectResult,
    PublicClaim,
    SearchResult,
    WebPage,
    WebVerification,
)
from app.services.research.agent_nodes import (
    AgentNodes,
    complete_report_content,
    fallback_report_content,
    fallback_understanding,
)
from app.services.research.agent_tools import ResearchToolExecutor
from app.services.research.evidence_verify import AgentEvidenceProcessor
from app.services.research.final_synthesis import (
    ordered_ranked_projects,
    validate_final_synthesis,
)
from app.services.intake.entity_resolver import EntityResolver
from app.services.research.extractor import RuleExtractor
from app.services.integrations.llm_client import StructuredLLM
from app.services.integrations.mcp_client import ProjectMcpClient
from app.services.research.project_ranker import ProjectRanker
from app.services.reporting.renderer import ReportRenderer
from app.services.research.resource_association import ResourceAssociationBuilder
from app.services.integrations.tavily_client import TavilyClient
from app.services.integrations.transcriber import LocalWhisperTranscriber
from app.tasks.celery_app import celery_app


class Repository(Protocol):
    def get(self, task_id: str): ...

    def update(self, task_id: str, **values: object): ...


class Transcriber(Protocol):
    def transcribe(self, webm_path: Path) -> str: ...


class SearchService(Protocol):
    async def search(self, queries: list[str]) -> list[SearchResult]: ...

    async def extract(self, results: list[SearchResult]) -> list[WebPage]: ...


class ProjectService(Protocol):
    async def search_projects(
        self, person_names: list[str], organization_names: list[str], keywords: list[str]
    ) -> list[ProjectResult]: ...


class PipelineCancelled(Exception):
    pass


class ResearchPipeline:
    def __init__(
        self,
        repository: Repository,
        transcriber: Transcriber,
        extractor: RuleExtractor,
        web: SearchService,
        projects: ProjectService,
        renderer: ReportRenderer,
        agents: AgentNodes | None = None,
        entity_resolver: EntityResolver | None = None,
        project_ranker: ProjectRanker | None = None,
        association_builder: ResourceAssociationBuilder | None = None,
    ):
        self.repository = repository
        self.transcriber = transcriber
        self.extractor = extractor
        self.web = web
        self.projects = projects
        self.renderer = renderer
        self.agents = agents or AgentNodes(StructuredLLM(settings, repository))
        self.entity_resolver = entity_resolver or EntityResolver()
        self.project_ranker = project_ranker or ProjectRanker()
        self.association_builder = association_builder or ResourceAssociationBuilder()

    def run(self, task_id: str) -> None:
        self._run_pipeline(task_id)

    def _run_pipeline(self, task_id: str) -> None:
        task = self.repository.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")
        if getattr(task, "status", None) == "CANCELLED":
            return
        audio_path = Path(task.audio_path) if task.audio_path else None
        degraded = list(task.degraded_nodes or [])
        try:
            self._checkpoint(task_id)
            input_text = sanitize_research_input(
                task.input_text or "", getattr(task, "input_snapshot", None)
            )
            if task.input_type == "audio" and not input_text:
                self.repository.update(task_id, status="TRANSCRIBING")
                input_text = self.transcriber.transcribe(audio_path)
                if not input_text:
                    raise ValueError("No valid speech was recognized")
                self.repository.update(task_id, input_text=input_text)

            self._checkpoint(task_id)
            if task.confirmed_context:
                context = ConfirmedContext.model_validate(task.confirmed_context)
                extracted = (
                    ExtractedInfo.model_validate(task.extracted_info)
                    if task.extracted_info
                    else extracted_from_context(context)
                )
                understanding = (
                    IntentUnderstanding.model_validate(task.llm_understanding)
                    if task.llm_understanding
                    else understanding_from_context(context, extracted)
                )
                if not task.extracted_info or not task.llm_understanding:
                    self.repository.update(
                        task_id,
                        extracted_info=extracted.model_dump(mode="json"),
                        llm_understanding=understanding.model_dump(mode="json"),
                    )
            else:
                self.repository.update(task_id, status="CONTEXT_EXTRACTING")
                extracted = self.extractor.extract(input_text)
                self.repository.update(
                    task_id, extracted_info=extracted.model_dump(mode="json")
                )
                understanding = fallback_understanding(extracted)
                self.repository.update(
                    task_id, llm_understanding=understanding.model_dump(mode="json")
                )
                context = context_from_intake_snapshot(
                    getattr(task, "input_snapshot", None), understanding
                )
                confirmation = None
                if context is None:
                    version = int(task.confirmation_version or 0) + 1
                    context, confirmation = self.entity_resolver.resolve(
                        input_text, understanding, version
                    )
                if confirmation:
                    self.repository.update(
                        task_id,
                        status="NEEDS_CONFIRMATION",
                        confirmation_version=confirmation.version,
                        confirmation_request=confirmation.model_dump(mode="json"),
                        degraded_nodes=degraded,
                    )
                    return
                self.repository.update(
                    task_id,
                    confirmed_context=context.model_dump(mode="json"),
                    confirmation_request=None,
                )

            self._checkpoint(task_id)
            intake_claims = identity_claims_from_intake_snapshot(
                getattr(task, "input_snapshot", None)
            )
            evidence_processor = AgentEvidenceProcessor(self.agents, self.repository)
            tool_executor = ResearchToolExecutor(
                self.web,
                self.projects,
                state_recorder=self.repository,
                evidence_processor=evidence_processor,
            )
            public_result = tool_executor.search_public(task_id, context)
            for node_name in public_result.degraded_nodes:
                if node_name not in degraded:
                    degraded.append(node_name)

            self._checkpoint(task_id)
            internal_result = tool_executor.search_internal(task_id, context)
            for node_name in internal_result.degraded_nodes:
                if node_name not in degraded:
                    degraded.append(node_name)

            claims = merge_public_claims(intake_claims, list(public_result.public_claims))
            project_results = list(internal_result.project_results)
            self.repository.update(
                task_id,
                web_results=[
                    item.model_dump(mode="json") for item in public_result.search_results
                ],
                web_pages=[
                    item.model_dump(mode="json") for item in public_result.web_pages
                ],
                verified_web_results=[
                    item.model_dump(mode="json")
                    for item in public_result.web_verifications
                ],
                public_claims=[item.model_dump(mode="json") for item in claims],
                internal_results=[
                    item.model_dump(mode="json") for item in project_results
                ],
                web_search_status=public_result.web_search_status,
                web_fetch_status=public_result.web_fetch_status,
                internal_search_status=internal_result.internal_search_status,
            )

            self._checkpoint(task_id)
            self.repository.update(task_id, status="RERANKING_PROJECTS")
            rankings = self.project_ranker.rank(
                project_results,
                context,
                reference_date=getattr(task, "created_at", None),
            )
            self._record_event(
                task_id,
                event_type="RULE_GENERATION",
                node_name="deterministic_project_ranker",
                status="SUCCESS",
                title="Deterministic project ranking completed",
                detail=f"Ranked {len(rankings)} projects.",
                payload={
                    "generator": self.project_ranker.version,
                    "results": [
                        {
                            "project_id": item.project_id,
                            "score": item.score,
                            "reason_codes": item.reason_codes,
                            "rank": item.rank,
                        }
                        for item in rankings
                    ],
                },
            )
            self.repository.update(
                task_id,
                ranked_internal_results=[
                    item.model_dump(mode="json") for item in rankings
                ],
            )

            self._checkpoint(task_id)
            self.repository.update(task_id, status="ANALYZING_ASSOCIATIONS")
            analysis = self.association_builder.build(
                context,
                claims,
                project_results,
                rankings,
                reference_date=getattr(task, "created_at", None),
            )
            self._record_event(
                task_id,
                event_type="RULE_GENERATION",
                node_name="resource_association_builder",
                status="SUCCESS",
                title="Deterministic association building completed",
                detail=f"Organized {len(analysis.related_projects)} related projects.",
                payload={
                    "generator": self.association_builder.version,
                    "related_project_refs": [
                        item.evidence_refs for item in analysis.related_projects
                    ],
                    "resource_refs": [
                        item.evidence_refs for item in analysis.available_resources
                    ],
                    "information_gaps": [
                        item.text for item in analysis.information_gaps
                    ],
                    "risk_flags": [item.text for item in analysis.risks],
                },
            )
            self.repository.update(
                task_id, association_analysis=analysis.model_dump(mode="json")
            )

            self._checkpoint(task_id)
            self.repository.update(task_id, status="GENERATING_REPORT_CONTENT")
            ranked_project_pairs = ordered_ranked_projects(project_results, rankings)
            ranked_project_results = [item[0] for item in ranked_project_pairs]
            fallback_content = validate_final_synthesis(
                fallback_report_content(
                    "", context, analysis, claims, ranked_project_results
                ),
                context,
                claims,
                ranked_project_pairs,
                analysis,
            )
            report_content = self._with_fallback(
                task_id,
                "final_synthesis",
                degraded,
                lambda: validate_final_synthesis(
                    self.agents.final_synthesis(
                        task_id,
                        context,
                        claims,
                        ranked_project_pairs,
                        analysis,
                    ),
                    context,
                    claims,
                    ranked_project_pairs,
                    analysis,
                ),
                lambda: fallback_content,
            )
            report_content = validate_final_synthesis(
                complete_report_content(report_content, fallback_content),
                context,
                claims,
                ranked_project_pairs,
                analysis,
            )
            self.repository.update(
                task_id,
                generated_report_content=report_content.model_dump(mode="json"),
            )

            self._checkpoint(task_id)
            self.repository.update(task_id, status="RENDERING_REPORT")
            detailed, action = self.renderer.render_generated(
                report_content,
                claims,
                project_results,
                public_result.web_search_status,
                public_result.web_fetch_status,
                internal_result.internal_search_status,
            )
            self._checkpoint(task_id)
            self.repository.update(
                task_id,
                status="COMPLETED",
                detailed_report_markdown=detailed,
                action_brief_markdown=action,
                report_markdown=detailed,
                degraded_nodes=degraded,
                error_message=None,
            )
        except PipelineCancelled:
            self._record_event(
                task_id,
                event_type="PIPELINE_CANCELLED",
                node_name="research_pipeline",
                status="CANCELLED",
                title="Research pipeline cancelled",
                detail="The task was cancelled before completion.",
            )
        except Exception as exc:
            if self._is_cancelled(task_id):
                self._record_event(
                    task_id,
                    event_type="PIPELINE_CANCELLED",
                    node_name="research_pipeline",
                    status="CANCELLED",
                    title="Research pipeline cancelled",
                    detail="The task was cancelled during execution.",
                )
            else:
                self._record_event(
                    task_id,
                    event_type="PIPELINE_ERROR",
                    node_name="research_pipeline",
                    status="FAILED",
                    title="Research pipeline failed",
                    detail=str(exc)[:1000],
                    payload={"error_type": type(exc).__name__},
                )
                self.repository.update(
                    task_id,
                    status="FAILED",
                    error_message=str(exc),
                    degraded_nodes=degraded,
                )
        finally:
            if audio_path:
                audio_path.unlink(missing_ok=True)
                audio_path.with_suffix(".wav").unlink(missing_ok=True)

    def _with_fallback(self, task_id: str, node_name: str, degraded: list[str], call, fallback):
        try:
            return call()
        except Exception as exc:
            if node_name not in degraded:
                degraded.append(node_name)
            self._record_event(
                task_id,
                event_type="FALLBACK",
                node_name=node_name,
                status="DEGRADED",
                title=f"节点已降级：{node_name}",
                detail=str(exc)[:1000],
            )
            return fallback()

    def _record_event(self, task_id: str, **values) -> None:
        logger = getattr(self.repository, "log_execution_event", None)
        if logger is not None:
            logger(task_id, **values)

    def _checkpoint(self, task_id: str) -> None:
        if self._is_cancelled(task_id):
            raise PipelineCancelled

    def _is_cancelled(self, task_id: str) -> bool:
        getter = getattr(self.repository, "get_fresh", self.repository.get)
        task = getter(task_id)
        return bool(task is not None and getattr(task, "status", None) == "CANCELLED")

def merge_public_claims(*groups: list[PublicClaim]) -> list[PublicClaim]:
    merged: list[PublicClaim] = []
    seen: set[tuple[str, str, str]] = set()
    for claim in (item for group in groups for item in group):
        key = (claim.subject.casefold(), claim.claim.casefold(), claim.source_url)
        if key in seen:
            continue
        seen.add(key)
        merged.append(claim)
    return merged[:30]


def context_from_intake_snapshot(
    snapshot: dict | None, understanding: IntentUnderstanding | None = None
) -> ConfirmedContext | None:
    structured = (snapshot or {}).get("structured_context", {})
    resolutions = structured.get("entity_resolutions", [])
    requester = structured.get("requester_context", {})
    requester_identity_terms = {
        str(requester.get(key) or "").strip()
        for key in ("name", "organization")
        if str(requester.get(key) or "").strip()
    }
    entities: list[ConfirmedEntity] = []
    for item in resolutions:
        entity_type = item.get("entity_type")
        canonical_name = (item.get("canonical_name") or "").strip()
        if entity_type not in {"PERSON", "ORGANIZATION", "PROJECT"} or not canonical_name:
            continue
        if canonical_name in requester_identity_terms or (
            item.get("mention") or ""
        ).strip() in requester_identity_terms:
            continue
        entities.append(
            ConfirmedEntity(
                candidate_id=item.get("candidate_id"),
                entity_type=entity_type,
                canonical_name=canonical_name,
                aliases=item.get("aliases") or [],
                organization=item.get("organization"),
                title=item.get("title"),
                region=item.get("region"),
                confirmed_by="AUTO"
                if item.get("confirmed_by") in {"INTERNAL", "EXTERNAL_AUTO", "AUTO"}
                else "USER",
            )
        )
    if not entities:
        return None
    requester_terms = {
        str(value).strip()
        for value in requester.values()
        if isinstance(value, str) and value.strip()
    }
    snapshot_text = str((snapshot or {}).get("analysis_input") or "")
    focus_questions = (
        list(structured.get("focus_questions") or [])
        or (understanding.focus_questions if understanding else [])
    )
    focus_questions = [
        item
        for item in focus_questions
        if not any(term in item for term in requester_terms)
    ]
    event_type = structured.get("event_type") or (
        understanding.event_type
        if understanding
        else infer_event_type(snapshot_text)
    )
    return ConfirmedContext(
        intents=(
            understanding.intents
            if understanding
            else [
                "MEETING_PREPARATION",
                "PERSON_BACKGROUND_RESEARCH",
                "INTERNAL_PROJECT_QUERY",
                "RESOURCE_RELATION_QUERY",
                "REPORT_GENERATION",
            ]
        ),
        entities=entities,
        event_type=event_type,
        event_time=structured.get("event_time")
        or (understanding.event_time if understanding else None),
        event_location=structured.get("event_location")
        or (understanding.event_location if understanding else None),
        business_directions=list(structured.get("business_directions") or [])
        or (understanding.business_directions if understanding else []),
        focus_questions=focus_questions,
    )


def infer_event_type(text: str) -> str:
    if any(term in text for term in ("吃饭", "宴请", "饭局", "晚宴")):
        return "宴请"
    if any(term in text for term in ("拜访", "走访")):
        return "拜访"
    if any(term in text for term in ("会议", "开会", "会面")):
        return "会议"
    return "其他"


def extracted_from_context(context: ConfirmedContext) -> ExtractedInfo:
    people = [
        {
            "name": entity.canonical_name,
            "organization": entity.organization,
            "title": entity.title,
        }
        for entity in context.entities
        if entity.entity_type == "PERSON"
    ]
    return ExtractedInfo(
        event_type=context.event_type,
        event_time=context.event_time,
        event_location=context.event_location,
        people=people,
        keywords=context.business_directions,
    )


def understanding_from_context(
    context: ConfirmedContext, extracted: ExtractedInfo
) -> IntentUnderstanding:
    understanding = fallback_understanding(extracted)
    return understanding.model_copy(
        update={
            "intents": context.intents,
            "focus_questions": context.focus_questions,
            "business_directions": context.business_directions,
            "overall_confidence": 1.0,
        }
    )


def sanitize_research_input(text: str, snapshot: dict | None) -> str:
    requester = ((snapshot or {}).get("structured_context") or {}).get(
        "requester_context", {}
    )
    sanitized = text
    for key in ("organization", "title", "name"):
        value = requester.get(key)
        if isinstance(value, str) and value:
            sanitized = sanitized.replace(value, "")
    return " ".join(sanitized.split())


def identity_claims_from_intake_snapshot(snapshot: dict | None) -> list[PublicClaim]:
    structured = (snapshot or {}).get("structured_context", {})
    resolutions = structured.get("entity_resolutions", [])
    claims: list[PublicClaim] = []
    for item in resolutions:
        source_url = (item.get("source_url") or "").strip()
        evidence_quote = (item.get("evidence_quote") or "").strip()
        canonical_name = (item.get("canonical_name") or "").strip()
        if not source_url or not evidence_quote or not canonical_name:
            continue
        organization = (item.get("organization") or "").strip()
        title = (item.get("title") or "").strip()
        details = "、".join(value for value in (organization, title) if value)
        claim = f"{canonical_name}（{details}）" if details else canonical_name
        index = len(claims) + 1
        claims.append(
            PublicClaim(
                web_result_id=f"INTAKE{index:03d}",
                evidence_id="IDENTITY",
                subject=canonical_name,
                claim=claim,
                evidence_quote=evidence_quote,
                source_title="关键人身份核验来源",
                source_url=source_url,
                matched_keywords=[],
                confidence=float(item.get("confidence") or 1),
            )
        )
    return claims


@celery_app.task(name="run_research_pipeline")
def run_research_pipeline(task_id: str) -> None:
    with SessionLocal() as session:
        repository = TaskRepository(session)
        llm = StructuredLLM(settings, repository)
        pipeline = ResearchPipeline(
            repository=repository,
            transcriber=LocalWhisperTranscriber(settings.whisper_model_path),
            extractor=RuleExtractor(settings.seed_dir),
            web=TavilyClient(settings.tavily_api_key),
            projects=ProjectMcpClient(settings.mcp_server_url),
            renderer=ReportRenderer(
                settings.report_template,
                settings.detailed_report_template,
                settings.action_brief_template,
            ),
            agents=AgentNodes(llm),
            entity_resolver=EntityResolver(),
        )
        pipeline.run(task_id)
