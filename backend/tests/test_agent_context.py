from datetime import date
from types import SimpleNamespace

from app.schemas.task import (
    AgentContext,
    ConfirmedContext,
    ConfirmedEntity,
    EvidenceBackedItem,
    ProjectResult,
    PublicClaim,
    TaskChatMessage,
)
from app.services.agent_context import AgentContextBuilder


def _confirmed_context() -> ConfirmedContext:
    return ConfirmedContext(
        intents=["MEETING_PREPARATION", "INTERNAL_PROJECT_QUERY"],
        entities=[
            ConfirmedEntity(
                entity_type="PERSON",
                canonical_name="范玉峰",
                organization="中建二局安装工程有限公司",
                title="党委书记、董事长",
                confirmed_by="USER",
            ),
            ConfirmedEntity(
                entity_type="PROJECT",
                canonical_name="示例项目",
                confirmed_by="USER",
            ),
        ],
        event_type="宴请",
        event_time="下周",
        event_location="北京",
        business_directions=["钢结构", "城市更新"],
        focus_questions=["近期重点项目是什么"],
    )


def _evidence() -> list[PublicClaim]:
    return [
        PublicClaim(
            web_result_id="W001",
            evidence_id="E001",
            subject="范玉峰",
            claim="范玉峰任中建二局安装公司党委书记、董事长",
            evidence_quote="范玉峰任中建二局安装公司党委书记、董事长。" + "Q" * 2_000,
            source_title="公司领导",
            source_url="https://example.com/leader",
        )
    ]


def _project(project_id: str, name: str) -> ProjectResult:
    return ProjectResult(
        project_id=project_id,
        project_name=name,
        customer_name="中建二局安装工程有限公司",
        status="ACTIVE",
        owner_name="项目负责人",
        start_date=date(2026, 1, 1),
        description="D" * 2_000,
        match_type="ORG_EXACT",
    )


def _task():
    gap = EvidenceBackedItem(
        text="尚不清楚对方当前最关注的项目",
        statement_type="INFERENCE",
        evidence_refs=["PROJECT:P001"],
        confidence=0.7,
    )
    return SimpleNamespace(
        input_text="和范总吃饭，了解近期项目",
        confirmation_request={
            "version": 1,
            "items": [
                {
                    "mention": "范总",
                    "entity_type": "PERSON",
                    "required": True,
                    "candidates": [
                        {
                            "candidate_id": "person-1",
                            "entity_type": "PERSON",
                            "canonical_name": "范玉峰",
                            "organization": "中建二局安装工程有限公司",
                            "title": "党委书记、董事长",
                            "reason": "官方页面精确候选",
                            "confidence": 0.98,
                            "source_url": "https://example.com/leader",
                            "evidence_quote": "Q" * 2_000,
                        }
                    ],
                }
            ],
        },
        ranked_internal_results=[
            {"project_id": "P002"},
            {"project_id": "P001"},
        ],
        association_analysis={
            "information_gaps": [gap.model_dump(mode="json")],
        },
        web_pages=[{"raw_content": "TAVILY_RAW_BODY_SHOULD_NOT_APPEAR" * 1_000}],
        web_results=[{"content": "RAW_TOOL_RESULT_SHOULD_NOT_APPEAR" * 1_000}],
    )


def test_builder_produces_distinct_phase_contexts() -> None:
    builder = AgentContextBuilder()
    task = _task()
    confirmed = _confirmed_context()
    evidence = _evidence()
    projects = [_project("P001", "项目一"), _project("P002", "项目二")]
    messages = [
        TaskChatMessage(role="user", content=f"消息 {index}")
        for index in range(10)
    ]

    identity = builder.build("IDENTITY", task, confirmed, evidence, projects, messages)
    public = builder.build("PUBLIC_RESEARCH", task, confirmed, evidence, projects, messages)
    project = builder.build("PROJECT_RESEARCH", task, confirmed, evidence, projects, messages)
    synthesis = builder.build("SYNTHESIS", task, confirmed, evidence, projects, messages)

    assert isinstance(identity, AgentContext)
    assert identity.identity_candidates is not None
    assert [message.content for message in identity.recent_messages] == [
        f"消息 {index}" for index in range(2, 10)
    ]
    assert identity.public_evidence == []
    assert identity.project_results == []

    assert public.identity_candidates is None
    assert public.recent_messages == []
    assert public.confirmed_context.focus_questions == ["近期重点项目是什么"]
    assert public.confirmed_context.business_directions == []
    assert len(public.public_evidence[0].evidence_quote) == 600
    assert public.project_results == []

    assert project.confirmed_context.business_directions == ["钢结构", "城市更新"]
    assert project.confirmed_context.focus_questions == []
    assert project.public_evidence[0].evidence_quote == ""
    assert [item.project_id for item in project.project_results] == ["P001", "P002"]
    assert len(project.project_results[0].description) == 600

    assert synthesis.confirmed_context == confirmed
    assert [item.project_id for item in synthesis.project_results] == ["P002", "P001"]
    assert synthesis.information_gaps[0].text == "尚不清楚对方当前最关注的项目"
    assert len(synthesis.public_evidence[0].evidence_quote) == 800

    serialized = {
        context.model_dump_json()
        for context in (identity, public, project, synthesis)
    }
    assert len(serialized) == 4


def test_project_research_never_carries_raw_web_or_tool_results() -> None:
    context = AgentContextBuilder().build(
        "PROJECT_RESEARCH",
        _task(),
        _confirmed_context(),
        _evidence(),
        [_project("P001", "项目一")],
        [TaskChatMessage(role="user", content="近期对话")],
    )

    payload = context.model_dump_json()
    assert "TAVILY_RAW_BODY_SHOULD_NOT_APPEAR" not in payload
    assert "RAW_TOOL_RESULT_SHOULD_NOT_APPEAR" not in payload
    assert "raw_content" not in payload
    assert "web_pages" not in payload
    assert "web_results" not in payload
