from app.schemas.task import (
    ConfirmedContext,
    ConfirmedEntity,
    SupportedWebEvidence,
    WebEvidenceCandidate,
    WebEvidenceDecision,
    WebPage,
    WebSearchQuery,
)
from app.services.agent_nodes import (
    AgentNodes,
    build_web_verification_candidates,
    materialize_web_verifications,
)


def context() -> ConfirmedContext:
    return ConfirmedContext(
        intents=["PERSON_BACKGROUND_RESEARCH"],
        entities=[
            ConfirmedEntity(
                entity_type="PERSON",
                canonical_name="王传福",
                organization="比亚迪股份有限公司",
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


def test_candidates_classify_identity_and_organization_topic() -> None:
    queries = [
        WebSearchQuery(
            query="王传福 比亚迪 董事长 履历",
            purpose="核验人物身份和公开履历",
            target_person="王传福",
            target_organization="比亚迪股份有限公司",
            required_terms=["王传福", "比亚迪股份有限公司", "董事长"],
        ),
        WebSearchQuery(
            query="比亚迪 储能 业务布局",
            purpose="了解企业储能业务布局",
            target_person="王传福",
            target_organization="比亚迪股份有限公司",
            required_terms=["王传福", "比亚迪股份有限公司", "储能"],
        ),
    ]
    pages = [
        WebPage(
            web_result_id="W001",
            title="人物资料",
            url="https://example.com/person",
            raw_content="王传福现任比亚迪股份有限公司董事长兼总裁。",
            rank=0,
            query=queries[0].query,
            target_person="王传福",
            target_organization="比亚迪股份有限公司",
        ),
        WebPage(
            web_result_id="W002",
            title="企业资料",
            url="https://example.com/company",
            raw_content="比亚迪股份有限公司持续扩大储能业务布局。",
            rank=1,
            query=queries[1].query,
            target_person="王传福",
            target_organization="比亚迪股份有限公司",
        ),
    ]

    candidates = build_web_verification_candidates(pages, context(), queries)

    assert [(item.web_result_id, item.kind) for item in candidates] == [
        ("W001", "IDENTITY"),
        ("W002", "ORGANIZATION_TOPIC"),
    ]
    assert candidates[1].target_person is None


def test_candidates_enforce_page_segment_and_batch_limits() -> None:
    query = WebSearchQuery(
        query="王传福 比亚迪 董事长 履历",
        purpose="核验人物身份和履历",
        target_person="王传福",
        target_organization="比亚迪股份有限公司",
        required_terms=["王传福", "比亚迪股份有限公司", "董事长"],
    )
    repeated = ("王传福现任比亚迪股份有限公司董事长。" + "背景资料" * 300) * 5
    page = WebPage(
        web_result_id="W001",
        title="人物资料",
        url="https://example.com/person",
        raw_content=repeated,
        rank=0,
        query=query.query,
        target_person="王传福",
        target_organization="比亚迪股份有限公司",
    )

    candidates = build_web_verification_candidates(
        [page], context(), [query], max_batch_chars=2500
    )

    assert 1 <= len(candidates) <= 3
    assert all(len(item.text) <= 1000 for item in candidates)
    assert sum(len(item.text) for item in candidates) <= 2500


class CapturingLlm:
    def __init__(self) -> None:
        self.call = None

    def parse(self, task_id, node_name, input_payload, output_model):
        self.call = (task_id, node_name, input_payload, output_model)
        return WebEvidenceDecision()


def test_agent_sends_only_bounded_candidates_to_web_verify() -> None:
    candidate = WebEvidenceCandidate(
        candidate_id="W001-C01",
        web_result_id="W001",
        kind="IDENTITY",
        text="王传福现任比亚迪股份有限公司董事长。",
        target_person="王传福",
        target_organization="比亚迪股份有限公司",
        matched_terms=["董事长"],
    )
    llm = CapturingLlm()

    AgentNodes(llm).web_verify("task-1", [candidate])

    _, node_name, payload, output_model = llm.call
    assert node_name == "web_verify"
    assert set(payload) == {"candidates"}
    assert payload["candidates"] == [
        candidate.model_dump(mode="json", exclude={"web_result_id"})
    ]
    assert output_model is WebEvidenceDecision


def test_materialization_binds_supported_decision_to_source_text() -> None:
    candidate = WebEvidenceCandidate(
        candidate_id="W001-C01",
        web_result_id="W001",
        kind="IDENTITY",
        text="王传福现任比亚迪股份有限公司董事长。",
        target_person="王传福",
        target_organization="比亚迪股份有限公司",
        matched_terms=["董事长"],
    )
    decision = WebEvidenceDecision(
        supported=[
            SupportedWebEvidence(candidate_id="W001-C01", position="董事长")
        ]
    )

    result = materialize_web_verifications(decision, [candidate])[0]

    assert result.keep is True
    assert result.evidence[0].quote == candidate.text
    assert result.evidence[0].claim == "王传福在比亚迪股份有限公司担任董事长"


def test_materialization_rejects_position_not_present_in_source_text() -> None:
    candidate = WebEvidenceCandidate(
        candidate_id="W001-C01",
        web_result_id="W001",
        kind="IDENTITY",
        text="王传福现任比亚迪股份有限公司董事长。",
        target_person="王传福",
        target_organization="比亚迪股份有限公司",
    )
    decision = WebEvidenceDecision(
        supported=[
            SupportedWebEvidence(candidate_id="W001-C01", position="首席执行官")
        ]
    )

    result = materialize_web_verifications(decision, [candidate])[0]

    assert result.keep is False
    assert result.evidence == []
    assert result.conflicts == ["候选原文存在歧义"]
