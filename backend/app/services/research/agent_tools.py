import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.schemas.task import (
    ConfirmedContext,
    ProjectResult,
    PublicClaim,
    SearchResult,
    WebPage,
    WebSearchQuery,
    WebVerification,
)
from app.services.research.agent_nodes import (
    fallback_project_query,
    fallback_web_plan,
    organization_aliases,
)
from app.services.research.evidence_verify import EvidenceProcessingResult


PUBLIC_RESULT_LIMIT = 10
PROJECT_RESULT_LIMIT = 20
PUBLIC_SNIPPET_LIMIT = 1_000
PROJECT_DESCRIPTION_LIMIT = 1_000
WEB_PAGE_CONTENT_LIMIT = 20_000


class PublicSearchClient(Protocol):
    async def search(self, queries: list[str]) -> list[SearchResult]: ...

    async def extract(self, results: list[SearchResult]) -> list[WebPage]: ...


class InternalProjectClient(Protocol):
    async def search_projects(
        self,
        person_names: list[str],
        organization_names: list[str],
        keywords: list[str],
    ) -> list[ProjectResult]: ...


class ToolStateRecorder(Protocol):
    def update(self, task_id: str, **values: object) -> object: ...

    def log_execution_event(self, scope_id: str, **values: object) -> object: ...


class EvidenceProcessor(Protocol):
    def process(
        self,
        task_id: str,
        pages: list[WebPage],
        context,
        queries: list[WebSearchQuery],
    ) -> EvidenceProcessingResult: ...


@dataclass(frozen=True)
class ResearchToolResult:
    search_results: tuple[SearchResult, ...] = ()
    web_pages: tuple[WebPage, ...] = ()
    web_verifications: tuple[WebVerification, ...] = ()
    public_claims: tuple[PublicClaim, ...] = ()
    project_results: tuple[ProjectResult, ...] = ()
    degraded_nodes: tuple[str, ...] = ()
    web_search_status: str = "SKIPPED"
    web_fetch_status: str = "SKIPPED"
    internal_search_status: str = "SKIPPED"


class ResearchToolExecutor:
    def __init__(
        self,
        public_search: PublicSearchClient,
        internal_projects: InternalProjectClient,
        *,
        state_recorder: ToolStateRecorder | None = None,
        evidence_processor: EvidenceProcessor | None = None,
    ):
        self.public_search = public_search
        self.internal_projects = internal_projects
        self.state_recorder = state_recorder
        self.evidence_processor = evidence_processor

    def search_public(
        self,
        task_id: str,
        context: ConfirmedContext,
    ) -> ResearchToolResult:
        plan = fallback_web_plan(context)
        self._record(
            task_id,
            event_type="RULE_GENERATION",
            node_name="SEARCH_PUBLIC",
            status="SUCCESS",
            title="规则生成公开搜索参数",
            detail=f"根据已确认上下文生成 {len(plan.queries)} 条查询。",
            payload={
                "generator": "fallback_web_plan",
                "action": "SEARCH_PUBLIC",
                "result": plan.model_dump(mode="json"),
            },
        )
        self._update(
            task_id,
            status="WEB_SEARCHING",
            web_search_plan=plan.model_dump(mode="json"),
        )
        queries = [item.query for item in plan.queries]
        self._record(
            task_id,
            event_type="SEARCH_REQUEST",
            node_name="tavily_search",
            status="RUNNING",
            title="执行公开网络搜索",
            detail=f"向 Tavily 提交 {len(queries)} 条查询。",
            payload={"provider": "Tavily", "queries": queries},
        )
        try:
            query_targets = {item.query: item for item in plan.queries}
            raw_results = asyncio.run(
                self.public_search.search(queries)
            )
            results = _normalize_public_results(raw_results, query_targets)
        except Exception as exc:
            self._record(
                task_id,
                event_type="TOOL_ERROR",
                node_name="tavily_search",
                status="FAILED",
                title="公开网络搜索失败",
                detail=f"错误类型：{type(exc).__name__}。",
            )
            self._update(
                task_id,
                web_results=[],
                web_pages=[],
                verified_web_results=[],
                public_claims=[],
                web_search_status="FAILED",
                web_fetch_status="SKIPPED",
            )
            return ResearchToolResult(
                degraded_nodes=("web_search",),
                web_search_status="FAILED",
                web_fetch_status="SKIPPED",
            )

        self._record(
            task_id,
            event_type="SEARCH_RESPONSE",
            node_name="tavily_search",
            status="SUCCESS",
            title="公开网络搜索完成",
            detail=f"获得 {len(results)} 条标准化结果。",
            payload={
                "results": [
                    {
                        "web_result_id": item.web_result_id,
                        "title": item.title,
                        "url": item.url,
                        "rank": item.rank,
                    }
                    for item in results
                ]
            },
        )
        self._update(
            task_id,
            status="WEB_FETCHING",
            web_results=[item.model_dump(mode="json") for item in results],
            web_search_status="SUCCESS",
        )
        raw_pages = []
        fetch_failed = False
        if results:
            self._record(
                task_id,
                event_type="SEARCH_REQUEST",
                node_name="tavily_extract",
                status="RUNNING",
                title="抓取候选网页正文",
                detail=f"向 Tavily Extract 提交 {len(results)} 个 URL。",
                payload={"urls": [item.url for item in results]},
            )
            try:
                raw_pages = asyncio.run(self.public_search.extract(results))
            except Exception as exc:
                fetch_failed = True
                self._record(
                    task_id,
                    event_type="TOOL_ERROR",
                    node_name="tavily_extract",
                    status="FAILED",
                    title="网页正文抓取失败",
                    detail=f"错误类型：{type(exc).__name__}。",
                )
        pages = _normalize_pages(raw_pages, results)
        extracted_count = sum(
            item.content_source == "PAGE_TEXT" for item in pages
        )
        if extracted_count == len(results) and results:
            fetch_status = "SUCCESS"
        elif extracted_count:
            fetch_status = "PARTIAL"
        elif pages:
            fetch_status = "SNIPPET_FALLBACK"
        else:
            fetch_status = "FAILED" if results else "SKIPPED"
        self._record(
            task_id,
            event_type="SEARCH_RESPONSE",
            node_name="tavily_extract",
            status=fetch_status,
            title="网页正文处理完成",
            detail=f"形成 {len(pages)} 个标准化页面。",
            payload={
                "pages": [
                    {
                        "web_result_id": item.web_result_id,
                        "url": item.url,
                        "content_source": item.content_source,
                        "content_length": len(item.raw_content),
                    }
                    for item in pages
                ]
            },
        )

        evidence_result = EvidenceProcessingResult((), ())
        if pages and self.evidence_processor is not None:
            self._update(task_id, status="VERIFYING_WEB_RESULTS")
            evidence_result = self.evidence_processor.process(
                task_id,
                pages,
                context,
                plan.queries,
            )
        degraded = list(evidence_result.degraded_nodes)
        if fetch_failed or fetch_status in {"FAILED", "SNIPPET_FALLBACK"}:
            degraded.append("web_fetch")
        self._update(
            task_id,
            web_pages=[item.model_dump(mode="json") for item in pages],
            web_fetch_status=fetch_status,
            verified_web_results=[
                item.model_dump(mode="json")
                for item in evidence_result.verifications
            ],
            public_claims=[
                item.model_dump(mode="json") for item in evidence_result.claims
            ],
        )

        return ResearchToolResult(
            search_results=tuple(results),
            web_pages=tuple(pages),
            web_verifications=evidence_result.verifications,
            public_claims=evidence_result.claims,
            degraded_nodes=tuple(dict.fromkeys(degraded)),
            web_search_status="SUCCESS",
            web_fetch_status=fetch_status,
        )

    def search_internal(
        self,
        task_id: str,
        context: ConfirmedContext,
    ) -> ResearchToolResult:
        plan = _expand_project_plan(fallback_project_query(context))
        keywords = _unique([*plan.project_names, *plan.business_terms])
        arguments = {
            "person_names": plan.person_names,
            "organization_names": plan.organization_names,
            "keywords": keywords,
        }
        self._record(
            task_id,
            event_type="RULE_GENERATION",
            node_name="SEARCH_INTERNAL",
            status="SUCCESS",
            title="规则生成内部项目查询参数",
            detail="根据已确认上下文生成 MCP 查询参数。",
            payload={
                "generator": "fallback_project_query",
                "action": "SEARCH_INTERNAL",
                "arguments": arguments,
            },
        )
        self._update(
            task_id,
            status="PROJECT_SEARCHING",
            project_query_plan=plan.model_dump(mode="json"),
        )
        self._record(
            task_id,
            event_type="TOOL_REQUEST",
            node_name="mcp.search_projects",
            status="RUNNING",
            title="查询内部项目",
            detail="调用 MCP 工具 search_projects。",
            payload={"tool": "search_projects", "arguments": arguments},
        )
        try:
            raw_projects = asyncio.run(
                self.internal_projects.search_projects(
                    plan.person_names,
                    plan.organization_names,
                    keywords,
                )
            )
            projects = _normalize_projects(raw_projects, plan.statuses)
        except Exception as exc:
            self._record(
                task_id,
                event_type="TOOL_ERROR",
                node_name="mcp.search_projects",
                status="FAILED",
                title="内部项目查询失败",
                detail=f"错误类型：{type(exc).__name__}。",
                payload={"tool": "search_projects", "arguments": arguments},
            )
            self._update(
                task_id,
                internal_results=[],
                internal_search_status="FAILED",
            )
            return ResearchToolResult(
                degraded_nodes=("internal_search",),
                internal_search_status="FAILED",
            )

        status = "SUCCESS" if projects else "EMPTY"
        summary = f"内部项目搜索得到 {len(projects)} 个去重项目。"
        self._record(
            task_id,
            event_type="TOOL_RESPONSE",
            node_name="mcp.search_projects",
            status=status,
            title="内部项目查询完成",
            detail=summary,
            payload={
                "project_ids": [item.project_id for item in projects],
                "result_count": len(projects),
            },
        )
        self._update(
            task_id,
            internal_results=[item.model_dump(mode="json") for item in projects],
            internal_search_status="SUCCESS",
        )
        return ResearchToolResult(
            project_results=tuple(projects),
            internal_search_status="SUCCESS",
        )

    def _record(self, task_id: str, **values) -> None:
        if self.state_recorder is not None:
            logger = getattr(self.state_recorder, "log_execution_event", None)
            if logger is not None:
                logger(task_id, **values)

    def _update(self, task_id: str, **values) -> None:
        if self.state_recorder is not None:
            updater = getattr(self.state_recorder, "update", None)
            if updater is not None:
                updater(task_id, **values)


def _normalize_public_results(raw_results, query_targets) -> list[SearchResult]:
    output = []
    seen = set()
    for raw in raw_results:
        try:
            result = SearchResult.model_validate(raw)
        except (TypeError, ValueError):
            continue
        url = _canonical_url(result.url)
        if url is None or url in seen:
            continue
        seen.add(url)
        target = query_targets.get(result.query)
        output.append(
            result.model_copy(
                update={
                    "web_result_id": f"W{len(output) + 1:03d}",
                    "title": (result.title or url)[:160],
                    "url": url,
                    "content": result.content[:PUBLIC_SNIPPET_LIMIT],
                    "rank": len(output),
                    "target_person": target.target_person if target else None,
                    "target_organization": target.target_organization if target else None,
                }
            )
        )
        if len(output) == PUBLIC_RESULT_LIMIT:
            break
    return output


def _normalize_pages(
    raw_pages: Sequence[WebPage | dict],
    results: Sequence[SearchResult],
) -> list[WebPage]:
    result_by_url = {item.url: item for item in results}
    output = []
    seen = set()
    for raw in raw_pages:
        try:
            page = WebPage.model_validate(raw)
        except (TypeError, ValueError):
            continue
        url = _canonical_url(page.url)
        source = result_by_url.get(url or "")
        if source is None or url in seen:
            continue
        seen.add(url)
        output.append(
            page.model_copy(
                update={
                    "web_result_id": source.web_result_id,
                    "title": page.title[:160],
                    "url": url,
                    "raw_content": page.raw_content[:WEB_PAGE_CONTENT_LIMIT],
                    "rank": source.rank,
                    "query": source.query,
                    "target_person": source.target_person,
                    "target_organization": source.target_organization,
                    "search_snippet": source.content[:PUBLIC_SNIPPET_LIMIT],
                    "content_source": "PAGE_TEXT",
                    "published_at": page.published_at or source.published_at,
                }
            )
        )
    for source in results:
        if source.url in seen or not source.content.strip():
            continue
        output.append(
            WebPage(
                web_result_id=source.web_result_id,
                title=source.title,
                url=source.url,
                raw_content=source.content[:PUBLIC_SNIPPET_LIMIT],
                rank=source.rank,
                query=source.query,
                target_person=source.target_person,
                target_organization=source.target_organization,
                search_snippet=source.content[:PUBLIC_SNIPPET_LIMIT],
                content_source="SEARCH_SNIPPET",
                published_at=source.published_at,
            )
        )
    return output[:PUBLIC_RESULT_LIMIT]


def _normalize_projects(raw_projects, statuses) -> list[ProjectResult]:
    output = []
    seen = set()
    for raw in raw_projects:
        try:
            project = ProjectResult.model_validate(raw)
        except (TypeError, ValueError):
            continue
        if project.project_id in seen or project.status not in statuses:
            continue
        seen.add(project.project_id)
        output.append(
            project.model_copy(
                update={
                    "project_aliases": project.project_aliases[:10],
                    "description": project.description[:PROJECT_DESCRIPTION_LIMIT],
                }
            )
        )
        if len(output) == PROJECT_RESULT_LIMIT:
            break
    return output


def _expand_project_plan(plan):
    organizations = _unique(
        alias
        for name in plan.organization_names
        for alias in _project_organization_terms(name)
    )
    return plan.model_copy(update={"organization_names": organizations})


def _project_organization_terms(name: str) -> list[str]:
    terms = organization_aliases(name)
    match = re.match(r"^(中建[一二三四五六七八九十]局)", name)
    if match:
        terms.append(match.group(1))
    return _unique(terms)


def _canonical_url(value: str) -> str | None:
    parts = urlsplit(value.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return None
    query = urlencode(
        [
            (key, item)
            for key, item in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_")
        ]
    )
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def _unique(values) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
