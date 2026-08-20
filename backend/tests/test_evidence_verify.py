from pathlib import Path
from types import SimpleNamespace

from app.schemas.task import (
    ConfirmedContext,
    ConfirmedEntity,
    WebEvidenceCandidate,
)
from app.services.intake.entity_resolver import EntityResolver
from app.services.research.evidence_verify import (
    materialize_routed_web_verifications,
    route_web_evidence_candidates,
)
from app.services.research.extractor import RuleExtractor
from app.services.reporting.report_renderer import ReportRenderer
from app.tasks.pipeline import ResearchPipeline


ROOT = Path(__file__).resolve().parents[2]


def _context() -> ConfirmedContext:
    return ConfirmedContext(
        intents=["MEETING_PREPARATION", "PERSON_BACKGROUND_RESEARCH"],
        entities=[
            ConfirmedEntity(
                entity_type="PERSON",
                canonical_name="王传福",
                organization="比亚迪股份有限公司",
                title="董事长兼总裁",
                confirmed_by="USER",
            ),
            ConfirmedEntity(
                entity_type="ORGANIZATION",
                canonical_name="比亚迪股份有限公司",
                confirmed_by="USER",
            ),
        ],
        event_type="会议",
        business_directions=["储能"],
    )


def _candidate(
    candidate_id: str,
    text: str,
    *,
    kind: str = "IDENTITY",
    organization: str | None = "比亚迪股份有限公司",
    matched_terms: list[str] | None = None,
) -> WebEvidenceCandidate:
    return WebEvidenceCandidate(
        candidate_id=candidate_id,
        web_result_id=candidate_id.split("-")[0],
        kind=kind,
        text=text,
        target_person="王传福" if kind == "IDENTITY" else None,
        target_organization=organization,
        matched_terms=matched_terms or [],
    )


def test_rules_route_candidates_without_accepting_identity_cooccurrence() -> None:
    candidates = [
        _candidate(
            "W001-C01",
            "王传福现任比亚迪股份有限公司董事长兼总裁。",
        ),
        _candidate(
            "W002-C01",
            "比亚迪股份有限公司持续推进储能业务布局。",
            kind="ORGANIZATION_TOPIC",
            matched_terms=["储能"],
        ),
        _candidate("W003-C01", "王传福参加行业活动并发表演讲。"),
        _candidate(
            "W004-C01",
            "比亚迪股份有限公司发布年度报告。",
            kind="ORGANIZATION_TOPIC",
            matched_terms=["储能"],
        ),
        _candidate(
            "W005-C01",
            "王传福与比亚迪股份有限公司参加会议，张三担任董事长。",
        ),
        _candidate(
            "W006-C01",
            "王传福曾任比亚迪股份有限公司董事长。",
        ),
    ]

    routing = route_web_evidence_candidates(candidates, _context())

    assert [item.candidate_id for item in routing.accepted] == [
        "W001-C01",
        "W002-C01",
    ]
    assert [item.candidate_id for item in routing.rejected] == [
        "W003-C01",
        "W004-C01",
    ]
    assert [item.candidate_id for item in routing.ambiguous] == [
        "W005-C01",
        "W006-C01",
    ]
    assert routing.accepted_support[0].position == "董事长兼总裁"


def test_failed_llm_marks_only_ambiguous_candidates_unverified() -> None:
    candidates = [
        _candidate(
            "W001-C01",
            "王传福现任比亚迪股份有限公司董事长兼总裁。",
        ),
        _candidate(
            "W002-C01",
            "王传福参加会议，比亚迪股份有限公司介绍近期业务。",
        ),
    ]
    routing = route_web_evidence_candidates(candidates, _context())

    verifications = materialize_routed_web_verifications(
        routing,
        llm_failed=True,
    )

    by_id = {item.web_result_id: item for item in verifications}
    assert by_id["W001"].keep is True
    assert by_id["W001"].evidence[0].claim == (
        "王传福在比亚迪股份有限公司担任董事长兼总裁"
    )
    assert by_id["W002"].keep is False
    assert "未核验" in by_id["W002"].identity_reason
    assert by_id["W002"].conflicts == ["候选原文存在歧义", "歧义候选未核验"]


class Repository:
    def __init__(self, task):
        self.task = task
        self.events = []

    def get(self, task_id):
        return self.task if task_id == self.task.id else None

    def update(self, task_id, **values):
        assert task_id == self.task.id
        for key, value in values.items():
            setattr(self.task, key, value)
        return self.task

    def log_execution_event(self, task_id, **values):
        assert task_id == self.task.id
        self.events.append(values)


class AmbiguousWeb:
    async def search(self, queries):
        from app.schemas.task import SearchResult

        return [
            SearchResult(
                title="活动报道",
                url="https://example.com/activity",
                content="王传福参加会议，比亚迪股份有限公司介绍近期业务。",
                query=queries[0],
                rank=0,
            )
        ]

    async def extract(self, results):
        from app.schemas.task import WebPage

        return [
            WebPage(
                web_result_id=results[0].web_result_id,
                title=results[0].title,
                url=results[0].url,
                raw_content=results[0].content,
                rank=0,
                query=results[0].query,
                target_person=results[0].target_person,
                target_organization=results[0].target_organization,
            )
        ]


class EmptyProjects:
    async def search_projects(self, *_args):
        return []


class ChoicesNoneAgents:
    def __init__(self):
        self.evidence_verify_candidates = []

    def evidence_verify(self, _task_id, candidates):
        self.evidence_verify_candidates = candidates
        raise TypeError("'NoneType' object is not subscriptable")

    def __getattr__(self, _name):
        def fail(*_args, **_kwargs):
            raise RuntimeError("model unavailable")

        return fail


class NoopTranscriber:
    def transcribe(self, _path):
        raise AssertionError("text task must not transcribe")


def test_choices_none_degrades_evidence_verify_and_pipeline_continues() -> None:
    context = _context()
    task = SimpleNamespace(
        id="task-evidence-degraded",
        input_type="text",
        input_text="准备与王传福会面，了解储能业务。",
        input_snapshot=None,
        audio_path=None,
        degraded_nodes=[],
        confirmation_version=0,
        confirmed_context=context.model_dump(mode="json"),
        extracted_info=None,
        llm_understanding=None,
    )
    repository = Repository(task)
    agents = ChoicesNoneAgents()
    pipeline = ResearchPipeline(
        repository=repository,
        transcriber=NoopTranscriber(),
        extractor=RuleExtractor(ROOT / "seed"),
        web=AmbiguousWeb(),
        projects=EmptyProjects(),
        renderer=ReportRenderer(
            ROOT / "backend/templates/report.md.j2",
            ROOT / "backend/templates/detailed_report.md.j2",
            ROOT / "backend/templates/action_brief.md.j2",
        ),
        agents=agents,
        entity_resolver=EntityResolver(),
    )

    pipeline.run(task.id)

    assert task.status == "COMPLETED", getattr(task, "error_message", None)
    assert len(agents.evidence_verify_candidates) == 1
    assert agents.evidence_verify_candidates[0].candidate_id.endswith("-C01")
    assert "evidence_verify" in task.degraded_nodes
    assert task.public_claims == []
    assert task.verified_web_results[0]["keep"] is False
    assert "未核验" in task.verified_web_results[0]["identity_reason"]
    degraded_events = [
        item
        for item in repository.events
        if item.get("node_name") == "evidence_verify"
        and item.get("status") == "DEGRADED"
    ]
    assert len(degraded_events) == 1
