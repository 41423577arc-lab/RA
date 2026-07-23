from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.schemas.task import (
    ConfirmationSelection,
    ConfirmedContext,
    ConfirmedEntity,
    EntityMention,
    IntentUnderstanding,
    ProjectResult,
    WebEvidence,
    WebPage,
    WebSearchPlan,
    WebSearchQuery,
    WebVerification,
    WebVerificationBatch,
)
from app.services.agent_nodes import validate_web_results
from app.services.entity_resolver import EntityResolver, InsufficientContextError
from app.services.extractor import RuleExtractor
from app.services.llm_client import LLMCallFailed, StructuredLLM
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


class FakeChatCompletions:
    def __init__(self, content: str):
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="chat-test",
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self.content))
            ],
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )


def test_llm_client_uses_chat_completions_with_pydantic_schema() -> None:
    config = Settings(
        openai_api_key="test-key",
        openai_base_url="https://vftllmapi.vf-tech.cn/v1",
        llm_model="MiniMax-M3",
        llm_review_model="MiniMax-M3",
        llm_api_mode="chat_completions",
        llm_reasoning_effort="xhigh",
        llm_disable_response_storage=True,
        prompt_dir=ROOT / "backend/prompts",
    )
    service = StructuredLLM(config)
    expected = WebSearchPlan(
        queries=[WebSearchQuery(query="王传福 比亚迪", purpose="身份核验")]
    )
    fake = FakeChatCompletions(expected.model_dump_json())
    service.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

    result = service.parse("task-1", "web_plan", {"name": "王传福"}, WebSearchPlan)

    assert result.queries[0].query == "王传福 比亚迪"
    assert fake.kwargs["model"] == "MiniMax-M3"
    assert fake.kwargs["store"] is False
    assert "JSON Schema" in fake.kwargs["messages"][0]["content"]
    assert "UNTRUSTED_DATA" in fake.kwargs["messages"][1]["content"]
    assert "tools" not in fake.kwargs


def test_llm_client_rejects_invalid_chat_completion_json() -> None:
    config = Settings(
        openai_api_key="test-key",
        llm_api_mode="chat_completions",
        llm_max_retries=0,
        llm_disable_response_storage=True,
        prompt_dir=ROOT / "backend/prompts",
    )
    service = StructuredLLM(config)
    fake = FakeChatCompletions("这不是 JSON")
    service.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

    with pytest.raises(LLMCallFailed, match="调用失败"):
        service.parse("task-1", "web_plan", {"name": "王传福"}, WebSearchPlan)


def test_llm_client_keeps_responses_mode_available() -> None:
    config = Settings(
        openai_api_key="test-key",
        llm_api_mode="responses",
        llm_disable_response_storage=True,
        prompt_dir=ROOT / "backend/prompts",
    )
    service = StructuredLLM(config)
    fake = FakeResponses()
    service.client = SimpleNamespace(responses=fake)

    result = service.parse("task-1", "web_plan", {"name": "王传福"}, WebSearchPlan)

    assert result.queries[0].query == "王传福 比亚迪"
    assert fake.kwargs["reasoning"] == {"effort": "xhigh"}


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
    input_text = "华星能源集团的李总明天参加会议"
    extracted = extractor.extract(input_text)
    understanding = IntentUnderstanding(
        intents=["MEETING_PREPARATION"],
        people=[
            EntityMention(
                mention="李总",
                organization="华星能源集团",
                evidence_text="华星能源集团的李总",
                confidence=0.9,
                resolution="NEEDS_CONFIRMATION",
            )
        ],
        organizations=[
            EntityMention(
                mention="华星能源集团",
                canonical_name="华星能源集团",
                evidence_text="华星能源集团",
                confidence=0.98,
                resolution="CONFIRMED",
            )
        ],
        event_type="会议",
        overall_confidence=0.9,
    )
    context, confirmation = EntityResolver().resolve(
        input_text, understanding, 1
    )

    assert context is None
    assert confirmation is not None
    assert [item.entity_type for item in confirmation.items] == ["PERSON"]
    assert confirmation.items[0].candidates == []
    assert EntityResolver().candidate_lookup(
        input_text, understanding
    ) == ("李总", "华星能源集团")


def test_entity_resolver_accepts_explicit_person_not_in_internal_data() -> None:
    input_text = "今天中午去丰收农业有限公司和杜鹏吃饭，沟通农业物联网平台。"
    extracted = RuleExtractor(ROOT / "seed").extract(input_text)
    understanding = IntentUnderstanding(
        intents=["MEETING_PREPARATION"],
        people=[
            EntityMention(
                mention="杜鹏",
                canonical_name="杜鹏",
                organization="丰收农业有限公司",
                evidence_text="丰收农业有限公司和杜鹏",
                confidence=0.98,
                resolution="CONFIRMED",
            )
        ],
        organizations=[
            EntityMention(
                mention="丰收农业有限公司",
                canonical_name="丰收农业有限公司",
                evidence_text="丰收农业有限公司",
                confidence=0.98,
                resolution="CONFIRMED",
            )
        ],
        event_type="宴请",
        overall_confidence=0.98,
    )

    context, confirmation = EntityResolver().resolve(
        input_text, understanding, 1
    )

    assert confirmation is None
    assert context is not None
    assert {(item.entity_type, item.canonical_name) for item in context.entities} == {
        ("PERSON", "杜鹏"),
        ("ORGANIZATION", "丰收农业有限公司"),
    }


def test_model_confirmed_fang_zheng_is_accepted_without_confidence_threshold() -> None:
    input_text = "我晚上去和新城水务的方正赴宴，讨论水务管网监测项目的现场部署问题。"
    understanding = IntentUnderstanding(
        intents=["MEETING_PREPARATION", "PROJECT_ADVANCEMENT_ADVICE"],
        people=[
            EntityMention(
                mention="方正",
                canonical_name="方正",
                organization="新城水务",
                evidence_text="我晚上去和新城水务的方正赴宴",
                confidence=0.6,
                resolution="CONFIRMED",
            )
        ],
        organizations=[
            EntityMention(
                mention="新城水务",
                canonical_name="新城水务",
                evidence_text="我晚上去和新城水务的方正赴宴",
                confidence=0.7,
                resolution="CONFIRMED",
            )
        ],
        event_type="宴请",
        overall_confidence=0.7,
    )

    context, confirmation = EntityResolver().resolve(input_text, understanding, 1)

    assert confirmation is None
    assert context is not None
    person = next(item for item in context.entities if item.entity_type == "PERSON")
    assert person.canonical_name == "方正"
    assert person.organization == "新城水务"


def test_model_entity_with_fabricated_evidence_is_not_accepted() -> None:
    input_text = "我晚上去和新城水务的方正赴宴。"
    understanding = IntentUnderstanding(
        intents=["MEETING_PREPARATION"],
        people=[
            EntityMention(
                mention="方正",
                canonical_name="方正",
                organization="新城水务",
                evidence_text="方正是新城水务董事长",
                confidence=0.99,
                resolution="CONFIRMED",
            )
        ],
        organizations=[
            EntityMention(
                mention="新城水务",
                canonical_name="新城水务",
                evidence_text="我晚上去和新城水务的方正赴宴",
                confidence=0.99,
                resolution="CONFIRMED",
            )
        ],
        event_type="宴请",
        overall_confidence=0.99,
    )

    context, confirmation = EntityResolver().resolve(input_text, understanding, 1)

    assert context is None
    assert confirmation is not None
    assert [item.entity_type for item in confirmation.items] == ["PERSON"]


def test_web_identity_candidates_require_exact_page_evidence() -> None:
    page = WebPage(
        web_result_id="W001",
        title="管理团队",
        url="https://example.com/team",
        raw_content="华星能源集团总经理李海负责新能源业务。",
        rank=0,
    )
    resolver = EntityResolver()
    candidates = resolver.candidates_from_web(
        "李总",
        "华星能源集团",
        [page],
    )

    assert [item.canonical_name for item in candidates] == ["李海"]
    assert candidates[0].source_url == page.url


def test_missing_organization_can_be_filled_without_internal_match() -> None:
    input_text = "今天中午和杜鹏吃饭，沟通农业物联网平台。"
    extracted = RuleExtractor(ROOT / "seed").extract(input_text)
    understanding = IntentUnderstanding(
        intents=["MEETING_PREPARATION"],
        people=[
            EntityMention(
                mention="杜鹏",
                canonical_name="杜鹏",
                evidence_text="和杜鹏吃饭",
                confidence=0.98,
                resolution="CONFIRMED",
            )
        ],
        organizations=[],
        event_type="宴请",
        overall_confidence=0.8,
    )
    resolver = EntityResolver()
    context, confirmation = resolver.resolve(input_text, understanding, 1)

    assert context is None
    assert confirmation is not None
    assert [item.entity_type for item in confirmation.items] == ["ORGANIZATION"]

    context = resolver.apply_confirmation(
        confirmation,
        [
            ConfirmationSelection(
                mention="企业名称", manual_value="丰收农业有限公司"
            )
        ],
        understanding,
        input_text,
    )

    person = next(item for item in context.entities if item.entity_type == "PERSON")
    assert person.organization == "丰收农业有限公司"


def test_task_without_person_and_organization_stops_clearly() -> None:
    input_text = "帮我生成一份报告"
    extracted = RuleExtractor(ROOT / "seed").extract(input_text)
    understanding = IntentUnderstanding(
        intents=["REPORT_GENERATION"],
        people=[],
        organizations=[],
        event_type="其他",
        overall_confidence=0.4,
    )

    with pytest.raises(InsufficientContextError, match="人物姓名和企业名称"):
        EntityResolver().resolve(input_text, understanding, 1)


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
        raise AssertionError("Pipeline must not search the public web")

    async def extract(self, _):
        raise AssertionError("Pipeline must not extract public pages")


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
        entity_resolver=EntityResolver(),
    )

    pipeline.run(task.id)

    assert task.status == "COMPLETED", getattr(task, "error_message", None)
    assert set(task.degraded_nodes) == {
        "understanding",
        "project_query",
        "project_rerank",
        "association",
        "report_content",
    }
    assert "P001" in task.detailed_report_markdown
    assert "和谁见面：王传福" in task.action_brief_markdown
    assert "ACTIVE" not in task.detailed_report_markdown
    assert "end_date" not in task.detailed_report_markdown
