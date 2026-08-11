import asyncio
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
    ProjectQueryPlan,
    ProjectResult,
    PublicClaim,
    SearchResult,
    WebPage,
    WebSearchPlan,
    WebSearchQuery,
    WebVerification,
)
from app.services.agent_nodes import (
    AgentNodes,
    build_web_verification_candidates,
    claims_from_verifications,
    complete_analysis,
    complete_report_content,
    deterministic_rankings,
    fallback_association,
    fallback_project_query,
    fallback_report_content,
    fallback_understanding,
    fallback_web_plan,
    materialize_web_verifications,
    organization_aliases,
    validate_analysis,
    validate_rankings,
    validate_report_content,
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
                    raise ValueError("未识别到有效语音，请重新录制")
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
                self.repository.update(task_id, extracted_info=extracted.model_dump(mode="json"))

                understanding = self._with_fallback(
                    task_id,
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

            self._checkpoint(task_id)
            intake_claims = identity_claims_from_intake_snapshot(
                getattr(task, "input_snapshot", None)
            )

            self.repository.update(task_id, status="PLANNING_WEB_SEARCH")
            web_plan = self._with_fallback(
                task_id,
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
                task_id, web_plan.queries
            )
            self._checkpoint(task_id)
            if web_search_status == "FAILED" and "web_search" not in degraded:
                degraded.append("web_search")
            if web_fetch_status == "FAILED" and "web_fetch" not in degraded:
                degraded.append("web_fetch")

            self.repository.update(task_id, status="VERIFYING_WEB_RESULTS")
            if pages:
                candidates = build_web_verification_candidates(
                    pages, context, web_plan.queries
                )
                verifications = (
                    materialize_web_verifications(
                        self.agents.web_verify(task_id, candidates), candidates
                    )
                    if candidates
                    else []
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
                task_id,
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
            self._checkpoint(task_id)
            project_results: list[ProjectResult] = []
            internal_search_status = "SUCCESS"
            project_arguments = {
                "person_names": project_plan.person_names,
                "organization_names": project_plan.organization_names,
                "keywords": unique_non_empty(
                    [*project_plan.project_names, *project_plan.business_terms]
                ),
            }
            self._record_event(
                task_id,
                event_type="TOOL_REQUEST",
                node_name="mcp.search_projects",
                status="RUNNING",
                title="查询内部项目",
                detail="调用 MCP 工具 search_projects。",
                payload={
                    "tool": "search_projects",
                    "arguments": project_arguments,
                },
            )
            try:
                project_results = asyncio.run(
                    self.projects.search_projects(
                        project_arguments["person_names"],
                        project_arguments["organization_names"],
                        project_arguments["keywords"],
                    )
                )
                project_results = [
                    item for item in project_results if item.status in project_plan.statuses
                ]
                self._checkpoint(task_id)
                self._record_event(
                    task_id,
                    event_type="TOOL_RESPONSE",
                    node_name="mcp.search_projects",
                    status="SUCCESS",
                    title="内部项目查询完成",
                    detail=f"返回 {len(project_results)} 个符合状态条件的项目。",
                    payload={
                        "results": [
                            item.model_dump(mode="json") for item in project_results
                        ]
                    },
                )
            except PipelineCancelled:
                raise
            except Exception as exc:
                internal_search_status = "FAILED"
                self._record_event(
                    task_id,
                    event_type="TOOL_ERROR",
                    node_name="mcp.search_projects",
                    status="FAILED",
                    title="内部项目查询失败",
                    detail=str(exc)[:1000],
                    payload={
                        "tool": "search_projects",
                        "arguments": project_arguments,
                    },
                )
            self.repository.update(
                task_id,
                internal_results=[item.model_dump(mode="json") for item in project_results],
                internal_search_status=internal_search_status,
            )

            self._checkpoint(task_id)
            self.repository.update(task_id, status="RERANKING_PROJECTS")
            rankings = self._with_fallback(
                task_id,
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

            self._checkpoint(task_id)
            self.repository.update(task_id, status="ANALYZING_ASSOCIATIONS")
            fallback_analysis = fallback_association(
                claims, project_results, rankings
            )
            analysis = self._with_fallback(
                task_id,
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

            self._checkpoint(task_id)
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
                task_id,
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

            self._checkpoint(task_id)
            self.repository.update(task_id, status="RENDERING_REPORT")
            detailed, action = self.renderer.render_generated(
                report_content,
                claims,
                project_results,
                web_search_status,
                web_fetch_status,
                internal_search_status,
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
                title="研究流水线已停止",
                detail="收到用户停止请求，后续分析阶段不再执行。",
            )
        except Exception as exc:
            if self._is_cancelled(task_id):
                self._record_event(
                    task_id,
                    event_type="PIPELINE_CANCELLED",
                    node_name="research_pipeline",
                    status="CANCELLED",
                    title="研究流水线已停止",
                    detail="任务执行期间收到用户停止请求。",
                )
            else:
                self._record_event(
                    task_id,
                    event_type="PIPELINE_ERROR",
                    node_name="research_pipeline",
                    status="FAILED",
                    title="研究流水线执行失败",
                    detail=str(exc)[:1000],
                    payload={"error_type": type(exc).__name__},
                )
                self.repository.update(task_id, status="FAILED", error_message=str(exc), degraded_nodes=degraded)
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

    def _run_web(
        self, task_id: str, queries: list[WebSearchQuery]
    ) -> tuple[list[SearchResult], list[WebPage], str, str]:
        search_results: list[SearchResult] = []
        pages: list[WebPage] = []
        web_search_status = "SKIPPED"
        web_fetch_status = "SKIPPED"
        if queries:
            self.repository.update(task_id, status="WEB_SEARCHING")
            self._record_event(
                task_id,
                event_type="SEARCH_REQUEST",
                node_name="tavily_search",
                status="RUNNING",
                title="执行公开网络搜索",
                detail=f"向 Tavily 提交 {len(queries)} 条搜索指令。",
                payload={
                    "provider": "Tavily",
                    "requests": [
                        {
                            "query": item.query,
                            "purpose": item.purpose,
                            "target_person": item.target_person,
                            "target_organization": item.target_organization,
                            "request": {
                                "query": item.query,
                                "search_depth": "basic",
                                "max_results": 5,
                            },
                        }
                        for item in queries
                    ],
                },
            )
            try:
                search_results = asyncio.run(
                    self.web.search([item.query for item in queries])
                )
                self._checkpoint(task_id)
                query_targets = {item.query: item for item in queries}
                for index, item in enumerate(search_results, 1):
                    if not item.web_result_id:
                        item.web_result_id = f"W{index:03d}"
                    target = query_targets.get(item.query)
                    if target:
                        item.target_person = target.target_person
                        item.target_organization = target.target_organization
                web_search_status = "SUCCESS"
                self._record_event(
                    task_id,
                    event_type="SEARCH_RESPONSE",
                    node_name="tavily_search",
                    status="SUCCESS",
                    title="公开网络搜索完成",
                    detail=f"获得 {len(search_results)} 条去重结果。",
                    payload={
                        "results": [
                            item.model_dump(mode="json") for item in search_results
                        ]
                    },
                )
            except PipelineCancelled:
                raise
            except Exception as exc:
                web_search_status = "FAILED"
                self._record_event(
                    task_id,
                    event_type="TOOL_ERROR",
                    node_name="tavily_search",
                    status="FAILED",
                    title="公开网络搜索失败",
                    detail=str(exc)[:1000],
                )
            self.repository.update(
                task_id,
                web_results=[item.model_dump(mode="json") for item in search_results],
                web_search_status=web_search_status,
            )
            if search_results:
                self.repository.update(task_id, status="WEB_FETCHING")
                self._record_event(
                    task_id,
                    event_type="SEARCH_REQUEST",
                    node_name="tavily_extract",
                    status="RUNNING",
                    title="抓取候选网页正文",
                    detail=f"向 Tavily Extract 提交 {len(search_results)} 个 URL。",
                    payload={
                        "provider": "Tavily",
                        "request": {
                            "urls": [item.url for item in search_results],
                            "extract_depth": "basic",
                        },
                    },
                )
                extracted_pages: list[WebPage] = []
                try:
                    extracted_pages = asyncio.run(self.web.extract(search_results))
                    self._checkpoint(task_id)
                    by_url = {item.url: item for item in search_results}
                    for page in extracted_pages:
                        source = by_url.get(page.url)
                        if source is None:
                            continue
                        page.web_result_id = page.web_result_id or source.web_result_id
                        page.query = page.query or source.query
                        page.target_person = page.target_person or source.target_person
                        page.target_organization = (
                            page.target_organization or source.target_organization
                        )
                        page.search_snippet = page.search_snippet or source.content
                except PipelineCancelled:
                    raise
                except Exception as exc:
                    web_fetch_status = "FAILED"
                    self._record_event(
                        task_id,
                        event_type="TOOL_ERROR",
                        node_name="tavily_extract",
                        status="FAILED",
                        title="网页正文抓取失败",
                        detail=str(exc)[:1000],
                    )
                pages = list(extracted_pages)
                extracted_urls = {page.url for page in extracted_pages}
                pages.extend(
                    WebPage(
                        web_result_id=item.web_result_id,
                        title=item.title,
                        url=item.url,
                        raw_content=item.content,
                        rank=item.rank,
                        query=item.query,
                        target_person=item.target_person,
                        target_organization=item.target_organization,
                        search_snippet=item.content,
                        content_source="SEARCH_SNIPPET",
                        published_at=item.published_at,
                    )
                    for item in search_results
                    if item.url not in extracted_urls and item.content.strip()
                )
                if extracted_pages and len(extracted_urls) == len(search_results):
                    web_fetch_status = "SUCCESS"
                elif extracted_pages:
                    web_fetch_status = "PARTIAL"
                elif pages:
                    web_fetch_status = "SNIPPET_FALLBACK"
                else:
                    web_fetch_status = "FAILED"
                self._record_event(
                    task_id,
                    event_type="SEARCH_RESPONSE",
                    node_name="tavily_extract",
                    status=web_fetch_status,
                    title="网页正文处理完成",
                    detail=f"形成 {len(pages)} 个可供核验的页面。",
                    payload={
                        "pages": [
                            {
                                "web_result_id": page.web_result_id,
                                "title": page.title,
                                "url": page.url,
                                "query": page.query,
                                "content_source": page.content_source,
                                "content_length": len(page.raw_content),
                            }
                            for page in pages
                        ]
                    },
                )
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
    organization_names = unique_non_empty(
        [*base.organization_names, *plan.organization_names]
    )
    expanded_organizations = unique_non_empty(
        alias
        for name in organization_names
        for alias in project_organization_terms(name)
    )
    return plan.model_copy(
        update={
            "person_names": unique_non_empty([*base.person_names, *plan.person_names]),
            "organization_names": expanded_organizations,
            "project_names": unique_non_empty([*base.project_names, *plan.project_names]),
            "business_terms": unique_non_empty(
                [*base.business_terms, *plan.business_terms]
            ),
            "statuses": unique_non_empty([*base.statuses, *plan.statuses]),
        }
    )


def project_organization_terms(name: str) -> list[str]:
    terms = organization_aliases(name)
    match = re.match(r"^(中建[一二三四五六七八九十]局)", name)
    if match:
        terms.append(match.group(1))
    return unique_non_empty(terms)


def sanitize_web_plan(plan: WebSearchPlan, context: ConfirmedContext) -> WebSearchPlan:
    person_entities = [
        item for item in context.entities if item.entity_type == "PERSON"
    ]
    people = {item.canonical_name for item in person_entities}
    organizations = {
        item.organization or item.canonical_name
        for item in context.entities
        if item.organization or item.entity_type == "ORGANIZATION"
    }
    default_person = person_entities[0] if len(person_entities) == 1 else None
    default_organization = (
        default_person.organization
        if default_person and default_person.organization
        else next(iter(organizations), None)
    )
    valid = []
    for item in plan.queries:
        if item.target_person and item.target_person not in people:
            continue
        if item.target_organization and item.target_organization not in organizations:
            continue
        target_person = item.target_person or (
            default_person.canonical_name if default_person else None
        )
        target_organization = item.target_organization or default_organization
        required_terms = unique_non_empty(
            [*item.required_terms, target_person, target_organization]
        )[:8]
        valid.append(
            item.model_copy(
                update={
                    "target_person": target_person,
                    "target_organization": target_organization,
                    "required_terms": required_terms,
                }
            )
        )
    if valid:
        return WebSearchPlan(queries=valid)
    return sanitize_web_plan(fallback_web_plan(context), context)


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


def merge_snippet_verifications(
    primary: list[WebVerification], snippet_rules: list[WebVerification]
) -> list[WebVerification]:
    output = list(primary)
    positions = {item.web_result_id: index for index, item in enumerate(output)}
    for item in snippet_rules:
        if not item.keep:
            continue
        position = positions.get(item.web_result_id)
        if position is None:
            positions[item.web_result_id] = len(output)
            output.append(item)
        elif not output[position].keep:
            output[position] = item
    return output


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
