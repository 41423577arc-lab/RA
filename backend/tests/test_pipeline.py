from datetime import date
from pathlib import Path
from types import SimpleNamespace

from app.schemas.task import (
    ProjectResult,
    SearchResult,
    SupportedWebEvidence,
    WebEvidenceDecision,
    WebPage,
)
from app.services.research.extractor import RuleExtractor
from app.services.intake.entity_resolver import EntityResolver
from app.services.reporting.renderer import ReportRenderer
from app.tasks.pipeline import (
    ResearchPipeline,
    context_from_intake_snapshot,
    identity_claims_from_intake_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]


class FakeRepository:
    def __init__(self, task):
        self.task = task
        for name, value in {
            "degraded_nodes": [],
            "confirmation_version": 0,
            "confirmed_context": None,
            "extracted_info": None,
            "llm_understanding": None,
            "input_snapshot": {},
        }.items():
            if not hasattr(task, name):
                setattr(task, name, value)
        self.statuses: list[str] = []
        self.events: list[dict] = []

    def get(self, task_id: str):
        return self.task if task_id == self.task.id else None

    def update(self, task_id: str, **values):
        assert task_id == self.task.id
        for key, value in values.items():
            setattr(self.task, key, value)
        if "status" in values:
            self.statuses.append(values["status"])
        return self.task

    def log_execution_event(self, task_id: str, **values):
        assert task_id == self.task.id
        self.events.append(values)


class NoopTranscriber:
    def transcribe(self, _):
        raise AssertionError("Text tasks must not call the transcriber")


class FakeTranscriber:
    def transcribe(self, path):
        assert path.read_bytes() == b"webm"
        return "老板周五要和比亚迪股份有限公司的王传福董事长兼总裁吃饭，主要聊新能源和储能项目。"


class FakeWeb:
    async def search(self, queries):
        assert queries[0].startswith("王传福 比亚迪股份有限公司")
        return [
            SearchResult(
                title="比亚迪公司简介",
                url="https://example.com/byd",
                content="摘要",
                query=queries[0],
                rank=0,
            )
        ]

    async def extract(self, _):
        return [
            WebPage(
                title="比亚迪公司简介",
                url="https://example.com/byd",
                raw_content="王传福表示比亚迪股份有限公司持续发展新能源汽车和储能业务。",
                rank=0,
            )
        ]


class FailedWeb:
    async def search(self, _):
        raise RuntimeError("tavily unavailable")

    async def extract(self, _):
        raise AssertionError("Extract must be skipped after search failure")


class FakeProjects:
    async def search_projects(self, person_names, organization_names, keywords):
        assert person_names == ["王传福"]
        assert "比亚迪股份有限公司" in organization_names
        assert keywords == ["新能源", "储能"]
        return [
            ProjectResult(
                project_id="P001",
                project_name="比亚迪园区储能管理平台",
                customer_name="比亚迪股份有限公司",
                contact_name="王传福",
                status="ACTIVE",
                owner_name="张伟",
                start_date=date(2026, 1, 10),
                description="建设园区储能监控与能源管理平台",
                match_type="PERSON_EXACT",
            ),
            ProjectResult(
                project_id="P002",
                project_name="比亚迪新能源汽车供应链分析",
                customer_name="比亚迪股份有限公司",
                contact_name="王传福",
                status="COMPLETED",
                owner_name="刘芳",
                start_date=date(2024, 3, 1),
                end_date=date(2024, 11, 30),
                description="完成新能源汽车供应链分析",
                match_type="PERSON_EXACT",
            ),
        ]


class FailedProjects:
    async def search_projects(self, person_names, organization_names, keywords):
        raise RuntimeError("mcp unavailable")


class MacroWeb:
    async def search(self, queries):
        assert all("林致远" not in query for query in queries)
        return [
            SearchResult(
                title="宏远制造公开资料",
                url="https://example.com/hongyuan",
                content="张伟负责宏远制造有限公司持续推进制造园区能源管理。",
                query=queries[0],
                rank=0,
            )
        ]

    async def extract(self, _):
        return [
            WebPage(
                title="宏远制造公开资料",
                url="https://example.com/hongyuan",
                raw_content="张伟负责宏远制造有限公司持续推进制造园区能源管理和节能改造。",
                rank=0,
            )
        ]


class MacroProjects:
    def __init__(self):
        self.arguments = None

    async def search_projects(self, person_names, organization_names, keywords):
        self.arguments = (person_names, organization_names, keywords)
        return [
            ProjectResult(
                project_id="P007",
                project_name="制造园区能源管理",
                customer_name="宏远制造有限公司",
                contact_name="郑伟",
                customer_contact_title="能源管理部经理",
                customer_contact_phone="17010000006",
                status="ACTIVE",
                owner_name="张伟",
                owner_phone="17000001001",
                owner_manager_name="周岚",
                owner_region="华东大区",
                start_date=date(2025, 8, 1),
                description="建设制造园区综合能源管理系统",
                project_stage="DELIVERY",
                match_type="ORG_EXACT",
            )
        ]


class FallbackAgents:
    def __init__(self):
        self.calls: list[str] = []

    def evidence_verify(self, _task_id, candidates):
        assert candidates
        return WebEvidenceDecision(
            supported=[
                SupportedWebEvidence(
                    candidate_id=candidates[0].candidate_id,
                    position="负责",
                )
            ]
        )

    def agent_turn(self, *_args, **_kwargs):
        self.calls.append("agent_turn")
        raise AssertionError("ResearchPipeline must not call agent_turn")

    def __getattr__(self, _):
        def fail(*args, **kwargs):
            raise RuntimeError("use deterministic fallback")

        return fail


def make_pipeline(repository, web):
    return ResearchPipeline(
        repository=repository,
        transcriber=NoopTranscriber(),
        extractor=RuleExtractor(ROOT / "seed"),
        web=web,
        projects=FakeProjects(),
        renderer=ReportRenderer(
            ROOT / "backend/templates/report.md.j2",
            ROOT / "backend/templates/detailed_report.md.j2",
            ROOT / "backend/templates/action_brief.md.j2",
        ),
        agents=FallbackAgents(),
        entity_resolver=EntityResolver(),
    )


def test_pipeline_constructor_defaults_to_research_dependencies() -> None:
    task = SimpleNamespace(id="constructor-test")
    repository = FakeRepository(task)

    pipeline = ResearchPipeline(
        repository=repository,
        transcriber=NoopTranscriber(),
        extractor=RuleExtractor(ROOT / "seed"),
        web=FailedWeb(),
        projects=FailedProjects(),
        renderer=ReportRenderer(ROOT / "backend/templates/report.md.j2"),
    )

    assert pipeline.agents.__class__.__name__ == "AgentNodes"
    assert pipeline.agents.llm.repository is repository
    assert isinstance(pipeline.entity_resolver, EntityResolver)


def test_pipeline_reuses_only_intake_identity_evidence() -> None:
    claims = identity_claims_from_intake_snapshot(
        {
            "structured_context": {
                "entity_resolutions": [
                    {
                        "entity_type": "PERSON",
                        "mention": "王总",
                        "canonical_name": "王传福",
                        "organization": "比亚迪股份有限公司",
                        "title": "董事长兼总裁",
                        "confidence": 0.9,
                        "confirmed_by": "EXTERNAL_AUTO",
                        "source_url": "https://example.com/identity",
                        "evidence_quote": "王传福任比亚迪股份有限公司董事长兼总裁。",
                    }
                ]
            }
        }
    )

    assert len(claims) == 1
    assert claims[0].claim == "王传福（比亚迪股份有限公司、董事长兼总裁）"
    assert claims[0].matched_keywords == []
    assert "业务" not in claims[0].claim


def test_context_from_snapshot_excludes_requester_even_if_resolution_is_polluted() -> None:
    context = context_from_intake_snapshot(
        {
            "analysis_input": "林致远与宏远制造有限公司的张伟吃饭。",
            "structured_context": {
                "requester_context": {
                    "name": "林致远",
                    "organization": "澄岳产业发展有限公司",
                },
                "entity_resolutions": [
                    {
                        "entity_type": "PERSON",
                        "mention": "林致远",
                        "canonical_name": "林致远",
                        "confirmed_by": "AUTO",
                    },
                    {
                        "entity_type": "ORGANIZATION",
                        "mention": "澄岳产业",
                        "canonical_name": "澄岳产业发展有限公司",
                        "confirmed_by": "AUTO",
                    },
                    {
                        "entity_type": "PERSON",
                        "mention": "张总",
                        "canonical_name": "张伟",
                        "organization": "宏远制造有限公司",
                        "confirmed_by": "USER",
                    },
                    {
                        "entity_type": "ORGANIZATION",
                        "mention": "宏远制造",
                        "canonical_name": "宏远制造有限公司",
                        "confirmed_by": "INTERNAL",
                    },
                ],
            },
        }
    )

    assert context is not None
    assert {item.canonical_name for item in context.entities} == {
        "张伟",
        "宏远制造有限公司",
    }


def test_full_text_pipeline_generates_report_and_all_states() -> None:
    task = SimpleNamespace(
        id="task-1",
        input_type="text",
        input_text="老板周五要和比亚迪股份有限公司的王传福董事长兼总裁吃饭，主要聊新能源和储能项目。",
        audio_path=None,
    )
    repository = FakeRepository(task)

    make_pipeline(repository, FailedWeb()).run(task.id)

    assert repository.statuses == [
        "CONTEXT_EXTRACTING",
        "WEB_SEARCHING",
        "PROJECT_SEARCHING",
        "RERANKING_PROJECTS",
        "ANALYZING_ASSOCIATIONS",
        "GENERATING_REPORT_CONTENT",
        "RENDERING_REPORT",
        "COMPLETED",
    ]
    assert "比亚迪园区储能管理平台" in task.report_markdown
    assert "比亚迪新能源汽车供应链分析" in task.report_markdown
    assert task.web_search_status == "FAILED"
    assert task.internal_search_status == "SUCCESS"


def test_search_failure_is_partial_and_internal_projects_continue() -> None:
    task = SimpleNamespace(
        id="task-2",
        input_type="text",
        input_text="老板周五要和比亚迪股份有限公司的王传福董事长兼总裁吃饭，主要聊新能源和储能项目。",
        audio_path=None,
    )
    repository = FakeRepository(task)

    make_pipeline(repository, FailedWeb()).run(task.id)

    assert task.status == "COMPLETED"
    assert task.web_search_status == "FAILED"
    assert task.web_fetch_status == "SKIPPED"
    assert "MISSING_VERIFIED_PERSON_EVIDENCE" in task.report_markdown
    assert "比亚迪园区储能管理平台" in task.report_markdown


def test_audio_pipeline_transcribes_and_deletes_shared_file(tmp_path) -> None:
    audio_path = tmp_path / "task-audio.webm"
    audio_path.write_bytes(b"webm")
    task = SimpleNamespace(
        id="task-audio",
        input_type="audio",
        input_text=None,
        audio_path=str(audio_path),
    )
    repository = FakeRepository(task)
    pipeline = ResearchPipeline(
        repository=repository,
        transcriber=FakeTranscriber(),
        extractor=RuleExtractor(ROOT / "seed"),
        web=FailedWeb(),
        projects=FakeProjects(),
        renderer=ReportRenderer(
            ROOT / "backend/templates/report.md.j2",
            ROOT / "backend/templates/detailed_report.md.j2",
            ROOT / "backend/templates/action_brief.md.j2",
        ),
        agents=FallbackAgents(),
        entity_resolver=EntityResolver(),
    )

    pipeline.run(task.id)

    assert repository.statuses[0] == "TRANSCRIBING"
    assert task.status == "COMPLETED"
    assert task.input_text.startswith("老板周五")
    assert not audio_path.exists()


def test_mcp_failure_is_partial_and_public_information_continues() -> None:
    task = SimpleNamespace(
        id="task-mcp-failed",
        input_type="text",
        input_text="老板周五要和比亚迪股份有限公司的王传福董事长兼总裁吃饭，主要聊新能源和储能项目。",
        audio_path=None,
    )
    repository = FakeRepository(task)
    pipeline = ResearchPipeline(
        repository=repository,
        transcriber=NoopTranscriber(),
        extractor=RuleExtractor(ROOT / "seed"),
        web=FailedWeb(),
        projects=FailedProjects(),
        renderer=ReportRenderer(
            ROOT / "backend/templates/report.md.j2",
            ROOT / "backend/templates/detailed_report.md.j2",
            ROOT / "backend/templates/action_brief.md.j2",
        ),
        agents=FallbackAgents(),
        entity_resolver=EntityResolver(),
    )

    pipeline.run(task.id)

    assert task.status == "COMPLETED"
    assert task.internal_search_status == "FAILED"
    assert "NO_RANKED_PROJECTS" in task.report_markdown
    assert task.web_search_status == "FAILED"


def test_confirmed_intake_merges_public_and_internal_evidence_without_requester() -> None:
    snapshot = {
        "analysis_input": "林致远要和宏远制造有限公司的张伟吃饭。",
        "structured_context": {
            "people": ["张伟"],
            "organizations": ["宏远制造有限公司"],
            "requester_context": {
                "name": "林致远",
                "organization": "澄岳产业发展有限公司",
                "title": "副总经理",
            },
            "entity_resolutions": [
                {
                    "entity_type": "ORGANIZATION",
                    "mention": "宏远制造",
                    "canonical_name": "宏远制造有限公司",
                    "confirmed_by": "INTERNAL",
                },
                {
                    "entity_type": "PERSON",
                    "mention": "张总",
                    "canonical_name": "张伟",
                    "organization": "宏远制造有限公司",
                    "confirmed_by": "USER",
                },
            ],
        },
    }
    confirmed_context = {
        "intents": [
            "MEETING_PREPARATION",
            "PERSON_BACKGROUND_RESEARCH",
            "INTERNAL_PROJECT_QUERY",
            "RESOURCE_RELATION_QUERY",
            "REPORT_GENERATION",
        ],
        "entities": [
            {
                "entity_type": "ORGANIZATION",
                "canonical_name": "宏远制造有限公司",
                "confirmed_by": "AUTO",
            },
            {
                "entity_type": "PERSON",
                "canonical_name": "张伟",
                "organization": "宏远制造有限公司",
                "confirmed_by": "USER",
            },
        ],
        "event_type": "宴请",
        "business_directions": [],
        "focus_questions": [],
    }
    task = SimpleNamespace(
        id="task-confirmed-intake",
        input_type="text",
        input_text=snapshot["analysis_input"],
        input_snapshot=snapshot,
        audio_path=None,
        degraded_nodes=[],
        confirmation_version=0,
        confirmed_context=confirmed_context,
        extracted_info=None,
        llm_understanding=None,
    )
    repository = FakeRepository(task)
    projects = MacroProjects()
    agents = FallbackAgents()
    pipeline = ResearchPipeline(
        repository=repository,
        transcriber=NoopTranscriber(),
        extractor=RuleExtractor(ROOT / "seed"),
        web=MacroWeb(),
        projects=projects,
        renderer=ReportRenderer(
            ROOT / "backend/templates/report.md.j2",
            ROOT / "backend/templates/detailed_report.md.j2",
            ROOT / "backend/templates/action_brief.md.j2",
        ),
        agents=agents,
        entity_resolver=object(),
    )

    pipeline.run(task.id)

    assert task.status == "COMPLETED", getattr(task, "error_message", None)
    assert "CONTEXT_EXTRACTING" not in repository.statuses
    assert agents.calls == []
    assert not {
        "web_plan",
        "web_verify",
        "project_query",
        "project_rerank",
        "association",
        "report_content",
    }.intersection(event.get("node_name") for event in repository.events)
    public_plan_index = next(
        index
        for index, event in enumerate(repository.events)
        if event.get("event_type") == "RULE_GENERATION"
        and event.get("node_name") == "SEARCH_PUBLIC"
    )
    tavily_started_index = next(
        index
        for index, event in enumerate(repository.events)
        if event.get("event_type") == "SEARCH_REQUEST"
        and event.get("node_name") == "tavily_search"
    )
    tavily_completed_index = next(
        index
        for index, event in enumerate(repository.events)
        if event.get("event_type") == "SEARCH_RESPONSE"
        and event.get("node_name") == "tavily_search"
    )
    internal_plan_index = next(
        index
        for index, event in enumerate(repository.events)
        if event.get("event_type") == "RULE_GENERATION"
        and event.get("node_name") == "SEARCH_INTERNAL"
    )
    assert (
        public_plan_index
        < tavily_started_index
        < tavily_completed_index
        < internal_plan_index
    )
    rule_events = [
        event
        for event in repository.events
        if event.get("event_type") == "RULE_GENERATION"
    ]
    assert [event["node_name"] for event in rule_events] == [
        "SEARCH_PUBLIC",
        "SEARCH_INTERNAL",
        "deterministic_project_ranker",
        "resource_association_builder",
    ]
    assert all(event["status"] == "SUCCESS" for event in rule_events)
    assert all(event["payload"]["generator"] for event in rule_events)
    assert rule_events[0]["payload"]["result"]
    assert rule_events[1]["payload"]["arguments"]
    assert set(rule_events[2]["payload"]["results"][0]) == {
        "project_id",
        "score",
        "reason_codes",
        "rank",
    }
    assert rule_events[3]["payload"]["generator"] == "resource_association_v1"
    assert "risk_flags" in rule_events[3]["payload"]
    assert projects.arguments == (["张伟"], ["宏远制造有限公司", "宏远制造"], [])
    assert "林致远" not in str(task.project_query_plan)
    assert task.web_search_status == "SUCCESS"
    assert task.project_query_plan["statuses"] == ["ACTIVE", "COMPLETED"]
    assert len(task.public_claims) == 1
    assert task.internal_results[0]["project_id"] == "P007"
    assert task.internal_results[0]["contact_name"] == "郑伟"
    assert task.internal_results[0]["owner_name"] == "张伟"
    assert "宏远制造有限公司持续推进制造园区能源管理" in task.detailed_report_markdown
    assert "制造园区能源管理" in task.detailed_report_markdown
    assert "客户联系人为郑伟（能源管理部经理），联系电话 17010000006" in task.detailed_report_markdown
    assert "我方项目销售员为张伟；联系电话 17000001001" in task.detailed_report_markdown
