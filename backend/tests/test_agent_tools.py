from datetime import date
from types import SimpleNamespace

from app.schemas.task import (
    ConfirmedContext,
    ConfirmedEntity,
    ProjectResult,
    SearchResult,
    WebPage,
)
from app.services.agent_context import AgentContextBuilder
from app.services.agent_loop import AgentLoopRunner
from app.services.agent_nodes import fallback_project_query, fallback_web_plan
from app.services.agent_tools import AgentToolExecutor, _expand_project_plan


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


class EventRecorder:
    def __init__(self):
        self.events = []

    def log_execution_event(self, scope_id, **values):
        self.events.append({"scope_id": scope_id, **values})


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


def _task():
    return SimpleNamespace(
        id="task-tools",
        input_text="准备会面",
        confirmation_request=None,
        ranked_internal_results=[],
        association_analysis=None,
    )


def test_public_tool_uses_rule_plan_and_builds_bounded_observation():
    public = PublicSearch()
    executor = AgentToolExecutor(public, InternalProjects())
    context = AgentContextBuilder().build(
        "PUBLIC_RESEARCH", _task(), _confirmed_context(), [], [], []
    )

    result = executor.execute("task-tools", "SEARCH_PUBLIC", context)

    plan = fallback_web_plan(_confirmed_context())
    assert public.queries == [item.query for item in plan.queries]
    assert len(public.extract_input) == 1
    assert public.extract_input[0].url == "https://example.com/leader"
    assert len(public.extract_input[0].content) == 1_000
    assert result.observation.status == "SUCCESS"
    assert result.observation.result_refs == ["WEB_RESULT:W001"]
    assert RAW_WEB_MARKER not in result.observation.model_dump_json()


def test_internal_tool_preserves_mcp_contract_and_normalizes_projects():
    internal = InternalProjects()
    executor = AgentToolExecutor(PublicSearch(), internal)
    context = AgentContextBuilder().build(
        "PROJECT_RESEARCH", _task(), _confirmed_context(), [], [], []
    )

    result = executor.execute("task-tools", "SEARCH_INTERNAL", context)

    plan = _expand_project_plan(fallback_project_query(_confirmed_context()))
    assert internal.calls == [
        (
            plan.person_names,
            plan.organization_names,
            [*plan.project_names, *plan.business_terms],
        )
    ]
    assert result.observation.project_ids == ["P001"]
    assert len(result.project_results) == 1
    assert len(result.project_results[0].project_aliases) == 10
    assert len(result.project_results[0].description) == 1_000


def test_runner_rebuilds_context_from_observations_after_each_tool():
    public = PublicSearch()
    internal = InternalProjects()
    recorder = EventRecorder()
    contexts = []
    actions = iter(["SEARCH_PUBLIC", "SEARCH_INTERNAL", "SYNTHESIZE"])

    def agent_turn(_task_id, context):
        contexts.append(context)
        return next(actions)

    runner = AgentLoopRunner(
        AgentContextBuilder(),
        recorder,
        agent_turn,
        AgentToolExecutor(public, internal),
    )
    result = runner.run(
        "PUBLIC_RESEARCH",
        _task(),
        _confirmed_context(),
        evidence=[],
        project_results=[],
        recent_messages=[],
    )

    assert result.phase == "DONE"
    assert [context.phase for context in contexts] == [
        "PUBLIC_RESEARCH",
        "PROJECT_RESEARCH",
        "SYNTHESIS",
    ]
    assert contexts[0].observations == []
    assert [item.action for item in contexts[1].observations] == ["SEARCH_PUBLIC"]
    assert [item.action for item in contexts[2].observations] == [
        "SEARCH_PUBLIC",
        "SEARCH_INTERNAL",
    ]
    assert [item.project_id for item in contexts[2].project_results] == ["P001"]
    serialized_contexts = "".join(item.model_dump_json() for item in contexts)
    assert RAW_WEB_MARKER not in serialized_contexts
    assert RAW_MCP_MARKER not in serialized_contexts
    observation_events = [
        item for item in recorder.events if item["event_type"] == "AGENT_OBSERVATION"
    ]
    assert len(observation_events) == 2
    assert all(
        item["payload"]["plan_source"] == "DETERMINISTIC_RULE"
        for item in observation_events
    )
