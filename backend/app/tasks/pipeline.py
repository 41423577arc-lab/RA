import asyncio
from pathlib import Path
from typing import Protocol

from app.config import settings
from app.database import SessionLocal, TaskRepository
from app.schemas.task import (
    ConfirmedContext,
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
            input_text = task.input_text or ""
            if task.input_type == "audio" and not input_text:
                self.repository.update(task_id, status="TRANSCRIBING")
                input_text = self.transcriber.transcribe(audio_path)
                if not input_text:
                    raise ValueError("未识别到有效语音，请重新录制")
                self.repository.update(task_id, input_text=input_text)

            if task.confirmed_context:
                extracted = ExtractedInfo.model_validate(task.extracted_info)
                understanding = IntentUnderstanding.model_validate(task.llm_understanding)
                context = ConfirmedContext.model_validate(task.confirmed_context)
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

                version = int(task.confirmation_version or 0) + 1
                context, confirmation = self.entity_resolver.resolve(
                    input_text, understanding, version
                )
                lookup = self.entity_resolver.candidate_lookup(
                    input_text, understanding
                )
                if confirmation and lookup:
                    mention, organization = lookup
                    candidates = self._discover_identity_candidates(
                        task_id, mention, organization, degraded
                    )
                    context, confirmation = self.entity_resolver.resolve(
                        input_text,
                        understanding,
                        version,
                        external_candidates=candidates,
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

            self.repository.update(task_id, status="PLANNING_WEB_SEARCH")
            web_plan = self._with_fallback(
                "web_plan",
                degraded,
                lambda: self.agents.web_plan(task_id, context),
                lambda: fallback_web_plan(context),
            )
            web_plan = sanitize_web_plan(web_plan, context)
            self.repository.update(task_id, web_search_plan=web_plan.model_dump(mode="json"))

            search_results, pages, web_search_status, web_fetch_status = self._run_web(
                task_id, [item.query for item in web_plan.queries]
            )

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
                    lambda: strict_rule_verifications(pages, context, extracted.keywords),
                )
            else:
                verifications = []
            claims = claims_from_verifications(verifications, pages)
            self.repository.update(
                task_id,
                verified_web_results=[item.model_dump(mode="json") for item in verifications],
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
            analysis = self._with_fallback(
                "association",
                degraded,
                lambda: validate_analysis(
                    self.agents.association(task_id, context, claims, project_results, rankings),
                    claims,
                    project_results,
                    settings.llm_analysis_confidence_threshold,
                ),
                lambda: fallback_association(claims, project_results, rankings),
            )
            self.repository.update(
                task_id, association_analysis=analysis.model_dump(mode="json")
            )

            self.repository.update(task_id, status="GENERATING_REPORT_CONTENT")
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
                lambda: validate_report_content(
                    fallback_report_content(input_text, context, analysis, project_results),
                    claims,
                    project_results,
                    context,
                ),
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

    def _discover_identity_candidates(
        self,
        task_id: str,
        mention: str,
        organization: str,
        degraded: list[str],
    ):
        try:
            query = f'"{organization}" "{mention}" 董事长 总经理 高管'
            results = asyncio.run(self.web.search([query]))
            pages = asyncio.run(self.web.extract(results)) if results else []
            if not pages:
                return []
            return self.entity_resolver.candidates_from_web(
                mention, organization, pages
            )
        except Exception:
            if "identity_candidates" not in degraded:
                degraded.append("identity_candidates")
            return []

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
            queries = build_search_queries(extracted)
            _, pages, web_search_status, web_fetch_status = self._run_web(task_id, queries)
            claims = self.extractor.extract_public_claims(pages, extracted) if pages else []
            self.repository.update(
                task_id, public_claims=[item.model_dump(mode="json") for item in claims]
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


def sanitize_web_plan(plan: WebSearchPlan, context: ConfirmedContext) -> WebSearchPlan:
    people = {item.canonical_name for item in context.entities if item.entity_type == "PERSON"}
    organizations = {
        item.organization or item.canonical_name
        for item in context.entities
        if item.organization or item.entity_type == "ORGANIZATION"
    }
    valid = [
        item
        for item in plan.queries
        if (not item.target_person or item.target_person in people)
        and (not item.target_organization or item.target_organization in organizations)
    ]
    return WebSearchPlan(queries=valid) if valid else fallback_web_plan(context)


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


def unique_non_empty(values) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def build_search_queries(extracted: ExtractedInfo) -> list[str]:
    queries: list[str] = []
    organizations: list[str] = []
    for person in extracted.people:
        if person.organization:
            organizations.append(person.organization)
        if person.name and person.organization:
            parts = [person.name, person.organization]
            if person.title:
                parts.append(person.title)
            parts.append("负责业务")
            queries.append(" ".join(parts))
        elif person.name:
            queries.append(f"{person.name} 负责业务")
        elif person.organization:
            queries.append(f"{person.organization} 主营业务 项目")
    for organization in unique_non_empty(organizations):
        queries.append(f"{organization} 主营业务 项目")
    if not queries and extracted.keywords:
        queries.append(" ".join(extracted.keywords[:3]))
    return list(dict.fromkeys(queries))


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
