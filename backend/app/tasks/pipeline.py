import asyncio
from pathlib import Path
from typing import Protocol

from app.config import settings
from app.database import SessionLocal, TaskRepository
from app.schemas.task import (
    ConfirmedContext,
    ConfirmedEntity,
    ExtractedInfo,
    IntentUnderstanding,
    ProjectQueryPlan,
    ProjectResult,
    PublicClaim,
    SearchResult,
    WebPage,
    WebSearchPlan,
)
from app.services.agent_nodes import (
    AgentNodes,
    claims_from_verifications,
    complete_analysis,
    complete_report_content,
    deterministic_rankings,
    fallback_association,
    fallback_project_query,
    fallback_report_content,
    fallback_understanding,
    fallback_web_plan,
    strict_rule_verifications,
    validate_analysis,
    validate_rankings,
    validate_report_content,
    validate_web_results,
)
from app.services.entity_resolver import EntityResolver
from app.services.extractor import RuleExtractor
from app.services.llm_client import StructuredLLM
from app.services.mcp_client import ProjectMcpClient
from app.services.report_renderer import ReportRenderer
from app.services.tavily_client import TavilyClient
from app.services.transcriber import LocalWhisperTranscriber
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
    ):
        self.repository = repository
        self.transcriber = transcriber
        self.extractor = extractor
        self.web = web
        self.projects = projects
        self.renderer = renderer
        self.agents = agents
        self.entity_resolver = entity_resolver

    def run(self, task_id: str) -> None:
        if self.agents is None or self.entity_resolver is None:
            self._run_legacy(task_id)
            return
        task = self.repository.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")
        audio_path = Path(task.audio_path) if task.audio_path else None
        degraded = list(task.degraded_nodes or [])
        try:
            input_text = sanitize_research_input(
                task.input_text or "", getattr(task, "input_snapshot", None)
            )
            if task.input_type == "audio" and not input_text:
                self.repository.update(task_id, status="TRANSCRIBING")
                input_text = self.transcriber.transcribe(audio_path)
                if not input_text:
                    raise ValueError("未识别到有效语音，请重新录制")
                self.repository.update(task_id, input_text=input_text)

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
                self.repository.update(task_id, extracted_info=extracted.model_dump(mode="json"))

                understanding = self._with_fallback(
                    "understanding",
                    degraded,
                    lambda: self.agents.understanding(task_id, input_text, extracted),
                    lambda: fallback_understanding(extracted),
                )
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

            intake_claims = identity_claims_from_intake_snapshot(
                getattr(task, "input_snapshot", None)
            )

            self.repository.update(task_id, status="PLANNING_WEB_SEARCH")
            web_plan = self._with_fallback(
                "web_plan",
                degraded,
                lambda: self.agents.web_plan(task_id, context),
                lambda: fallback_web_plan(context),
            )
            web_plan = sanitize_web_plan(web_plan, context)
            self.repository.update(
                task_id, web_search_plan=web_plan.model_dump(mode="json")
            )
            _, pages, web_search_status, web_fetch_status = self._run_web(
                task_id, [item.query for item in web_plan.queries]
            )
            if web_search_status == "FAILED" and "web_search" not in degraded:
                degraded.append("web_search")
            if web_fetch_status == "FAILED" and "web_fetch" not in degraded:
                degraded.append("web_fetch")

            self.repository.update(task_id, status="VERIFYING_WEB_RESULTS")
            if pages:
                verifications = self._with_fallback(
                    "web_verify",
                    degraded,
                    lambda: validate_web_results(
                        self.agents.web_verify(task_id, context, pages),
                        pages,
                        context,
                        settings.llm_web_identity_threshold,
                    ),
                    lambda: strict_rule_verifications(
                        pages, context, extracted.keywords
                    ),
                )
            else:
                verifications = []
            claims = merge_public_claims(
                intake_claims, claims_from_verifications(verifications, pages)
            )
            self.repository.update(
                task_id,
                verified_web_results=[
                    item.model_dump(mode="json") for item in verifications
                ],
                public_claims=[item.model_dump(mode="json") for item in claims],
            )

            self.repository.update(task_id, status="PLANNING_PROJECT_SEARCH")
            project_plan = self._with_fallback(
                "project_query",
                degraded,
                lambda: self.agents.project_query(task_id, context),
                lambda: fallback_project_query(context),
            )
            project_plan = sanitize_project_plan(project_plan, context)
            self.repository.update(
                task_id, project_query_plan=project_plan.model_dump(mode="json")
            )

            self.repository.update(task_id, status="PROJECT_SEARCHING")
            project_results: list[ProjectResult] = []
            internal_search_status = "SUCCESS"
            try:
                project_results = asyncio.run(
                    self.projects.search_projects(
                        project_plan.person_names,
                        project_plan.organization_names,
                        unique_non_empty([*project_plan.project_names, *project_plan.business_terms]),
                    )
                )
                project_results = [
                    item for item in project_results if item.status in project_plan.statuses
                ]
            except Exception:
                internal_search_status = "FAILED"
            self.repository.update(
                task_id,
                internal_results=[item.model_dump(mode="json") for item in project_results],
                internal_search_status=internal_search_status,
            )

            self.repository.update(task_id, status="RERANKING_PROJECTS")
            rankings = self._with_fallback(
                "project_rerank",
                degraded,
                lambda: validate_rankings(
                    self.agents.project_rerank(task_id, context, project_results).rankings,
                    project_results,
                    settings.llm_project_confidence_threshold,
                ),
                lambda: deterministic_rankings(project_results, context),
            )
            self.repository.update(
                task_id,
                ranked_internal_results=[item.model_dump(mode="json") for item in rankings],
            )

            self.repository.update(task_id, status="ANALYZING_ASSOCIATIONS")
            fallback_analysis = fallback_association(
                claims, project_results, rankings
            )
            analysis = self._with_fallback(
                "association",
                degraded,
                lambda: validate_analysis(
                    self.agents.association(task_id, context, claims, project_results, rankings),
                    claims,
                    project_results,
                    settings.llm_analysis_confidence_threshold,
                ),
                lambda: fallback_analysis,
            )
            analysis = complete_analysis(analysis, fallback_analysis)
            self.repository.update(
                task_id, association_analysis=analysis.model_dump(mode="json")
            )

            self.repository.update(task_id, status="GENERATING_REPORT_CONTENT")
            fallback_content = validate_report_content(
                fallback_report_content(
                    input_text, context, analysis, claims, project_results
                ),
                claims,
                project_results,
                context,
            )
            report_content = self._with_fallback(
                "report_content",
                degraded,
                lambda: validate_report_content(
                    self.agents.report_content(
                        task_id, input_text, context, claims, project_results, analysis
                    ),
                    claims,
                    project_results,
                    context,
                ),
                lambda: fallback_content,
            )
            report_content = validate_report_content(
                complete_report_content(report_content, fallback_content),
                claims,
                project_results,
                context,
            )
            self.repository.update(
                task_id, generated_report_content=report_content.model_dump(mode="json")
            )

            self.repository.update(task_id, status="RENDERING_REPORT")
            detailed, action = self.renderer.render_generated(
                report_content,
                claims,
                project_results,
                web_search_status,
                web_fetch_status,
                internal_search_status,
            )
            self.repository.update(
                task_id,
                status="COMPLETED",
                detailed_report_markdown=detailed,
                action_brief_markdown=action,
                report_markdown=detailed,
                degraded_nodes=degraded,
                error_message=None,
            )
        except Exception as exc:
            self.repository.update(task_id, status="FAILED", error_message=str(exc), degraded_nodes=degraded)
        finally:
            if audio_path:
                audio_path.unlink(missing_ok=True)
                audio_path.with_suffix(".wav").unlink(missing_ok=True)

    def _with_fallback(self, node_name: str, degraded: list[str], call, fallback):
        try:
            return call()
        except Exception:
            if node_name not in degraded:
                degraded.append(node_name)
            return fallback()

    def _run_web(
        self, task_id: str, queries: list[str]
    ) -> tuple[list[SearchResult], list[WebPage], str, str]:
        search_results: list[SearchResult] = []
        pages: list[WebPage] = []
        web_search_status = "SKIPPED"
        web_fetch_status = "SKIPPED"
        if queries:
            self.repository.update(task_id, status="WEB_SEARCHING")
            try:
                search_results = asyncio.run(self.web.search(queries))
                for index, item in enumerate(search_results, 1):
                    if not item.web_result_id:
                        item.web_result_id = f"W{index:03d}"
                web_search_status = "SUCCESS"
            except Exception:
                web_search_status = "FAILED"
            self.repository.update(
                task_id,
                web_results=[item.model_dump(mode="json") for item in search_results],
                web_search_status=web_search_status,
            )
            if search_results:
                self.repository.update(task_id, status="WEB_FETCHING")
                try:
                    pages = asyncio.run(self.web.extract(search_results))
                    by_url = {item.url: item for item in search_results}
                    for page in pages:
                        if not page.web_result_id and page.url in by_url:
                            page.web_result_id = by_url[page.url].web_result_id
                    web_fetch_status = "SUCCESS" if pages else "FAILED"
                except Exception:
                    web_fetch_status = "FAILED"
            self.repository.update(
                task_id,
                web_pages=[item.model_dump(mode="json") for item in pages],
                web_fetch_status=web_fetch_status,
            )
        return search_results, pages, web_search_status, web_fetch_status

    def _run_legacy(self, task_id: str) -> None:
        task = self.repository.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")
        audio_path = Path(task.audio_path) if task.audio_path else None
        try:
            input_text = task.input_text or ""
            if task.input_type == "audio":
                self.repository.update(task_id, status="TRANSCRIBING")
                input_text = self.transcriber.transcribe(audio_path)
                if not input_text:
                    raise ValueError("未识别到有效语音，请重新录制")
                self.repository.update(task_id, input_text=input_text)
            self.repository.update(task_id, status="EXTRACTING")
            extracted = self.extractor.extract(input_text)
            self.repository.update(task_id, extracted_info=extracted.model_dump(mode="json"))
            claims = identity_claims_from_intake_snapshot(
                getattr(task, "input_snapshot", None)
            )
            web_search_status = "REUSED_INTAKE" if claims else "SKIPPED"
            web_fetch_status = web_search_status
            self.repository.update(
                task_id,
                web_search_plan=None,
                web_results=[],
                web_pages=[],
                web_search_status=web_search_status,
                web_fetch_status=web_fetch_status,
                verified_web_results=[],
                public_claims=[item.model_dump(mode="json") for item in claims],
            )
            self.repository.update(task_id, status="PROJECT_SEARCHING")
            person_names = unique_non_empty(person.name for person in extracted.people)
            organization_names = unique_non_empty(person.organization for person in extracted.people)
            project_results: list[ProjectResult] = []
            internal_search_status = "SUCCESS"
            try:
                project_results = asyncio.run(
                    self.projects.search_projects(person_names, organization_names, extracted.keywords)
                )
            except Exception:
                internal_search_status = "FAILED"
            self.repository.update(
                task_id,
                internal_results=[item.model_dump(mode="json") for item in project_results],
                internal_search_status=internal_search_status,
            )
            self.repository.update(task_id, status="GENERATING")
            report = self.renderer.render(
                input_text,
                extracted,
                claims,
                project_results,
                web_search_status,
                web_fetch_status,
                internal_search_status,
            )
            self.repository.update(task_id, status="COMPLETED", report_markdown=report)
        except Exception as exc:
            self.repository.update(task_id, status="FAILED", error_message=str(exc))
        finally:
            if audio_path:
                audio_path.unlink(missing_ok=True)
                audio_path.with_suffix(".wav").unlink(missing_ok=True)


def sanitize_project_plan(
    plan: ProjectQueryPlan, context: ConfirmedContext
) -> ProjectQueryPlan:
    base = fallback_project_query(context)
    return plan.model_copy(
        update={
            "person_names": unique_non_empty([*base.person_names, *plan.person_names]),
            "organization_names": unique_non_empty(
                [*base.organization_names, *plan.organization_names]
            ),
            "project_names": unique_non_empty([*base.project_names, *plan.project_names]),
            "business_terms": unique_non_empty(
                [*base.business_terms, *plan.business_terms]
            ),
            "statuses": plan.statuses or ["ACTIVE", "COMPLETED"],
        }
    )


def sanitize_web_plan(plan: WebSearchPlan, context: ConfirmedContext) -> WebSearchPlan:
    people = {
        item.canonical_name for item in context.entities if item.entity_type == "PERSON"
    }
    organizations = {
        item.organization or item.canonical_name
        for item in context.entities
        if item.organization or item.entity_type == "ORGANIZATION"
    }
    valid = [
        item
        for item in plan.queries
        if (not item.target_person or item.target_person in people)
        and (
            not item.target_organization
            or item.target_organization in organizations
        )
    ]
    return WebSearchPlan(queries=valid) if valid else fallback_web_plan(context)


def unique_non_empty(values) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


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
    event_type = (
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
