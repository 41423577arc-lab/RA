from datetime import date

from app.schemas.task import (
    ConfirmedContext,
    ConfirmedEntity,
    ProjectResult,
    SearchResult,
    WebPage,
)
from app.services.research.agent_nodes import fallback_project_query, fallback_web_plan
from app.services.research.agent_tools import ResearchToolExecutor, _expand_project_plan


RAW_WEB_MARKER = "RAW_WEB_BODY_MUST_NOT_ENTER_CONTEXT"
RAW_MCP_MARKER = "RAW_MCP_DESCRIPTION_MUST_BE_TRIMMED"


class PublicSearch:
    def __init__(self):
        self.queries = []
        self.extract_input = []

    async def search(self, queries):
        self.queries = queries
        query = queries[0]
        return [
            SearchResult(
                title="领导信息",
                url="HTTPS://Example.com/leader/?utm_source=test#bio",
                content=RAW_WEB_MARKER * 100,
                query=query,
                rank=5,
            ),
            SearchResult(
                title="重复页面",
                url="https://example.com/leader",
                content="duplicate",
                query=query,
                rank=6,
            ),
            SearchResult(
                title="无效页面",
                url="file:///tmp/result",
                content="invalid",
                query=query,
                rank=7,
            ),
            {"malformed": True},
        ]

    async def extract(self, results):
        self.extract_input = results
        result = results[0]
        return [
            WebPage(
                web_result_id=result.web_result_id,
                title=result.title,
                url=result.url,
                raw_content=RAW_WEB_MARKER * 1_000,
                rank=result.rank,
                query=result.query,
            )
        ]


class InternalProjects:
    def __init__(self):
        self.calls = []

    async def search_projects(
        self,
        person_names,
        organization_names,
        keywords,
    ):
        self.calls.append((person_names, organization_names, keywords))
        project = _project("P001", "D" * 2_000 + RAW_MCP_MARKER)
        return [project, project.model_copy(update={"project_name": "重复项目"}), {"bad": 1}]


def _confirmed_context():
    return ConfirmedContext(
        intents=["MEETING_PREPARATION", "INTERNAL_PROJECT_QUERY"],
        entities=[
            ConfirmedEntity(
                entity_type="PERSON",
                canonical_name="范玉峰",
                aliases=["范总"],
                organization="中建二局安装工程有限公司",
                title="党委书记、董事长",
                confirmed_by="USER",
            ),
            ConfirmedEntity(
                entity_type="PROJECT",
                canonical_name="城市更新项目",
                confirmed_by="USER",
            ),
        ],
        event_type="会议",
        business_directions=["钢结构", "城市更新"],
    )


def _project(project_id, description):
    return ProjectResult(
        project_id=project_id,
        project_name="示例项目",
        project_aliases=[f"别名{i}" for i in range(20)],
        customer_name="中建二局安装工程有限公司",
        status="ACTIVE",
        owner_name="项目负责人",
        start_date=date(2026, 1, 1),
        description=description,
        match_type="ORG_EXACT",
    )


def test_public_tool_uses_rule_plan_and_returns_bounded_results():
    public = PublicSearch()
    executor = ResearchToolExecutor(public, InternalProjects())

    result = executor.search_public("task-tools", _confirmed_context())

    plan = fallback_web_plan(_confirmed_context())
    assert public.queries == [item.query for item in plan.queries]
    assert len(public.extract_input) == 1
    assert public.extract_input[0].url == "https://example.com/leader"
    assert len(public.extract_input[0].content) == 1_000
    assert result.web_search_status == "SUCCESS"
    assert [item.web_result_id for item in result.search_results] == ["W001"]
    assert RAW_WEB_MARKER not in "".join(item.content for item in result.search_results)


def test_internal_tool_preserves_mcp_contract_and_normalizes_projects():
    internal = InternalProjects()
    executor = ResearchToolExecutor(PublicSearch(), internal)

    result = executor.search_internal("task-tools", _confirmed_context())

    plan = _expand_project_plan(fallback_project_query(_confirmed_context()))
    assert internal.calls == [
        (
            plan.person_names,
            plan.organization_names,
            [*plan.project_names, *plan.business_terms],
        )
    ]
    assert len(result.project_results) == 1
    assert len(result.project_results[0].project_aliases) == 10
    assert len(result.project_results[0].description) == 1_000
