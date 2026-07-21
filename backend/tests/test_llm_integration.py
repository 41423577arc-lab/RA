from datetime import date
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.schemas.task import (
    ConfirmedContext,
    ConfirmedEntity,
    ProjectResult,
    WebEvidence,
    WebPage,
    WebSearchPlan,
    WebSearchQuery,
    WebVerification,
    WebVerificationBatch,
)
from app.services.agent_nodes import validate_web_results
from app.services.entity_resolver import EntityResolver
from app.services.extractor import RuleExtractor
from app.services.llm_client import StructuredLLM
from app.tasks.pipeline import ResearchPipeline
from app.services.report_renderer import ReportRenderer


ROOT = Path(__file__).resolve().parents[2]


class FakeResponses:
    def __init__(self):
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="resp-test",
            output_parsed=WebSearchPlan(
                queries=[WebSearchQuery(query="王传福 比亚迪", purpose="身份核验")]
            ),
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )


def test_llm_client_uses_fixed_model_reasoning_and_no_storage() -> None:
    config = Settings(
        openai_api_key="test-key",
        openai_base_url="https://vftsub.vf-tech.cn",
        llm_model="gpt-5.5",
        llm_review_model="gpt-5.5",
        llm_reasoning_effort="xhigh",
        llm_disable_response_storage=True,
        prompt_dir=ROOT / "backend/prompts",
    )
    service = StructuredLLM(config)
    fake = FakeResponses()
    service.client = SimpleNamespace(responses=fake)

    result = service.parse("task-1", "web_plan", {"name": "王传福"}, WebSearchPlan)

    assert result.queries[0].query == "王传福 比亚迪"
    assert fake.kwargs["model"] == "gpt-5.5"
    assert fake.kwargs["reasoning"] == {"effort": "xhigh"}
    assert fake.kwargs["store"] is False
    assert "tools" not in fake.kwargs


def test_web_verification_rejects_same_name_page_without_target_company() -> None:
    context = ConfirmedContext(
        intents=["PERSON_BACKGROUND_RESEARCH"],
        entities=[
            ConfirmedEntity(
                entity_type="PERSON",
                canonical_name="王传福",
                organization="比亚迪股份有限公司",
                confirmed_by="AUTO",
            )
        ],
        event_type="会议",
    )
    pages = [
        WebPage(
            web_result_id="W001",
            title="同名人物",
            url="https://example.com/other",
            raw_content="王传福参加某大学活动并发表演讲。",
            rank=0,
        )
    ]
    batch = WebVerificationBatch(
        results=[
            WebVerification(
                web_result_id="W001",
                keep=True,
                matched_person="王传福",
                identity_reason="姓名相同",
                confidence=0.99,
                same_name_risk=False,
                evidence=[
                    WebEvidence(
                        evidence_id="E1",
                        quote="王传福参加某大学活动并发表演讲",
                        claim="参加大学活动",
                    )
                ],
            )
        ]
    )

    results = validate_web_results(batch, pages, context, 0.8)

    assert results[0].keep is False
    assert results[0].evidence == []


def test_entity_resolver_requires_confirmation_for_huaxing_li_alias() -> None:
    extractor = RuleExtractor(ROOT / "seed")
    extracted = extractor.extract("华星的李总明天参加会议")
    from app.services.agent_nodes import fallback_understanding

    understanding = fallback_understanding(extracted)
    context, confirmation = EntityResolver(ROOT / "seed").resolve(
        "华星的李总明天参加会议", understanding, extracted, 1
    )

    assert context is None
    assert confirmation is not None
    assert {item.canonical_name for item in confirmation.items[0].candidates} == {"李明", "李伟"}


class Repo:
    def __init__(self, task):
        self.task = task

    def get(self, _):
        return self.task

    def update(self, _, **values):
        for key, value in values.items():
            setattr(self.task, key, value)
        return self.task


class NoopTranscriber:
    def transcribe(self, _):
        raise AssertionError


class Web:
    async def search(self, queries):
        from app.schemas.task import SearchResult

        return [SearchResult(title="比亚迪动态", url="https://example.com/byd", query=queries[0], rank=0)]

    async def extract(self, _):
        return [WebPage(title="比亚迪动态", url="https://example.com/byd", raw_content="王传福表示比亚迪股份有限公司继续推进新能源和储能业务。", rank=0)]


class Projects:
    async def search_projects(self, *_):
        return [
            ProjectResult(
                project_id="P001",
                project_name="比亚迪园区储能管理平台",
                customer_name="比亚迪股份有限公司",
                contact_name="王传福",
                status="ACTIVE",
                owner_name="张伟",
                start_date=date(2026, 1, 10),
                description="储能管理",
                match_type="PERSON_EXACT",
            )
        ]


class DisabledAgents:
    def __getattr__(self, _):
        def fail(*args, **kwargs):
            raise RuntimeError("llm disabled")

        return fail


def test_v05_pipeline_completes_with_all_llm_nodes_degraded() -> None:
    task = SimpleNamespace(
        id="task-v05",
        input_type="text",
        input_text="老板周五要和比亚迪股份有限公司的王传福董事长兼总裁吃饭，主要聊新能源和储能项目。",
        audio_path=None,
        degraded_nodes=[],
        confirmation_version=0,
        confirmed_context=None,
    )
    pipeline = ResearchPipeline(
        repository=Repo(task),
        transcriber=NoopTranscriber(),
        extractor=RuleExtractor(ROOT / "seed"),
        web=Web(),
        projects=Projects(),
        renderer=ReportRenderer(
            ROOT / "backend/templates/report.md.j2",
            ROOT / "backend/templates/detailed_report.md.j2",
            ROOT / "backend/templates/action_brief.md.j2",
        ),
        agents=DisabledAgents(),
        entity_resolver=EntityResolver(ROOT / "seed"),
    )

    pipeline.run(task.id)

    assert task.status == "COMPLETED"
    assert set(task.degraded_nodes) == {
        "understanding",
        "web_plan",
        "web_verify",
        "project_query",
        "project_rerank",
        "association",
        "report_content",
    }
    assert "P001" in task.detailed_report_markdown
    assert "和谁见面：王传福" in task.action_brief_markdown
    assert "ACTIVE" not in task.detailed_report_markdown
    assert "end_date" not in task.detailed_report_markdown
