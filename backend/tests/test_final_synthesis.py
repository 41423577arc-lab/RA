from datetime import date
from pathlib import Path

from app.schemas.task import (
    ActionBrief,
    AssociationAnalysis,
    ConfirmedContext,
    ConfirmedEntity,
    EvidenceBackedItem,
    GeneratedReportContent,
    ProjectRanking,
    ProjectResult,
    PublicClaim,
)
from app.services.research.agent_nodes import AgentNodes
from app.services.research.final_synthesis import (
    ordered_ranked_projects,
    validate_final_synthesis,
)


ROOT = Path(__file__).resolve().parents[2]


class CapturingLlm:
    def __init__(self):
        self.call = None

    def parse(self, task_id, node_name, input_payload, output_model):
        self.call = (task_id, node_name, input_payload, output_model)
        return GeneratedReportContent(
            action_brief=ActionBrief(objective="整理现有材料"),
        )


def _context() -> ConfirmedContext:
    return ConfirmedContext(
        intents=["MEETING_PREPARATION"],
        entities=[
            ConfirmedEntity(
                entity_type="PERSON",
                canonical_name="王志远",
                organization="华北筑联供应链有限公司",
                title="总经理",
                confirmed_by="USER",
            )
        ],
        event_type="会议",
        event_location="北京",
        business_directions=["供应链"],
    )


def _claim() -> PublicClaim:
    return PublicClaim(
        web_result_id="W001",
        evidence_id="E001",
        subject="王志远",
        claim="王志远现任华北筑联供应链有限公司总经理",
        evidence_quote="王志远现任华北筑联供应链有限公司总经理。",
        source_title="企业页面",
        source_url="https://example.com/person",
    )


def _project(project_id: str, owner: str) -> ProjectResult:
    return ProjectResult(
        project_id=project_id,
        project_name=f"项目{project_id}",
        customer_name="华北筑联供应链有限公司",
        status="ACTIVE",
        owner_name=owner,
        start_date=date(2026, 1, 1),
        description="供应链协同",
        match_type="ORG_EXACT",
    )


def _ranking(project_id: str, score: int, rank: int) -> ProjectRanking:
    return ProjectRanking(
        project_id=project_id,
        relevance_score=score,
        score=score,
        reason_codes=[f"FIXTURE:+{score}"],
        rank=rank,
        relevance_reason=f"FIXTURE:+{score}",
        recommended_use="",
        confidence=1,
        evidence_refs=[f"PROJECT:{project_id}"],
    )


def _association() -> AssociationAnalysis:
    return AssociationAnalysis(
        related_projects=[
            _fact("项目P002；排序：1；评分：90", "PROJECT:P002"),
            _fact("项目P001；排序：2；评分：60", "PROJECT:P001"),
        ],
        available_resources=[
            _fact("我方项目销售员为张伟", "PROJECT:P002"),
            _fact("我方项目销售员为李华", "PROJECT:P001"),
        ],
        risks=[_fact("PROJECT_HEALTH_RED:P002", "PROJECT:P002")],
        information_gaps=[
            _fact("MISSING_CUSTOMER_CONTACT:P001", "PROJECT:P001")
        ],
    )


def _fact(text: str, ref: str) -> EvidenceBackedItem:
    return EvidenceBackedItem(
        text=text,
        statement_type="FACT",
        evidence_refs=[ref],
        confidence=1,
    )


def test_final_synthesis_receives_only_whitelisted_ranked_inputs() -> None:
    p1 = _project("P001", "李华")
    p2 = _project("P002", "张伟")
    ranked = ordered_ranked_projects(
        [p1, p2],
        [_ranking("P001", 60, 2), _ranking("P002", 90, 1)],
    )
    llm = CapturingLlm()

    AgentNodes(llm).final_synthesis(
        "task-final",
        _context(),
        [_claim()],
        ranked,
        _association(),
    )

    task_id, node_name, payload, output_model = llm.call
    assert task_id == "task-final"
    assert node_name == "final_synthesis"
    assert output_model is GeneratedReportContent
    assert set(payload) == {
        "confirmed_context",
        "verified_evidence",
        "ranked_projects",
        "deterministic_association",
        "information_gaps",
    }
    assert [item["project"]["project_id"] for item in payload["ranked_projects"]] == [
        "P002",
        "P001",
    ]
    assert "input_text" not in payload
    assert set(payload["deterministic_association"]) == {
        "related_projects",
        "customer_and_internal_resources",
        "risk_flags",
    }


def test_validator_rebinds_facts_filters_refs_and_preserves_project_rank() -> None:
    p1 = _project("P001", "李华")
    p2 = _project("P002", "张伟")
    ranked = ordered_ranked_projects(
        [p1, p2],
        [_ranking("P001", 60, 2), _ranking("P002", 90, 1)],
    )
    fabricated = GeneratedReportContent(
        public_information_summary=[
            _fact("王志远还负责一个输入中不存在的事业部", "WEB:W001:E001")
        ],
        priority_projects=[
            _fact("P001 是最相关项目", "PROJECT:P001"),
            _fact("P002 排名靠后", "PROJECT:P002"),
        ],
        resource_analysis=[_fact("张伟是客户董事长", "PROJECT:P002")],
        recommended_topics=[
            EvidenceBackedItem(
                text="讨论供应链协同",
                statement_type="FACT",
                evidence_refs=["PROJECT:P002"],
                confidence=1,
            ),
            EvidenceBackedItem(
                text="讨论不存在的项目",
                statement_type="RECOMMENDATION",
                evidence_refs=["PROJECT:UNKNOWN"],
                confidence=1,
            ),
        ],
        gaps_and_risks=[_fact("模型自行补充的风险", "PROJECT:P002")],
        action_brief=ActionBrief(
            destination="上海",
            meeting_people=["王志远", "虚构人物"],
            objective="基于已整理材料沟通",
            internal_contacts=["张伟", "虚构负责人"],
            risks=["模型自行补充的风险"],
            evidence_refs=["PROJECT:P002", "PROJECT:UNKNOWN"],
        ),
    )

    validated = validate_final_synthesis(
        fabricated,
        _context(),
        [_claim()],
        ranked,
        _association(),
    )

    assert [item.text for item in validated.public_information_summary] == [
        _claim().claim
    ]
    assert [item.text for item in validated.priority_projects] == [
        "项目P002；排序：1；评分：90",
        "项目P001；排序：2；评分：60",
    ]
    assert [item.text for item in validated.resource_analysis] == [
        "我方项目销售员为张伟",
        "我方项目销售员为李华",
    ]
    assert [item.text for item in validated.gaps_and_risks] == [
        "PROJECT_HEALTH_RED:P002",
        "MISSING_CUSTOMER_CONTACT:P001",
    ]
    assert [item.text for item in validated.recommended_topics] == [
        "讨论供应链协同"
    ]
    assert validated.recommended_topics[0].statement_type == "RECOMMENDATION"
    assert validated.action_brief.destination == "北京"
    assert validated.action_brief.meeting_people == ["王志远"]
    assert validated.action_brief.internal_contacts == ["张伟"]
    assert validated.action_brief.risks == ["PROJECT_HEALTH_RED:P002"]
    assert validated.action_brief.evidence_refs == ["PROJECT:P002"]


def test_final_synthesis_prompt_forbids_new_facts_and_reordering() -> None:
    prompt = (ROOT / "backend/prompts/final_synthesis_v1.txt").read_text(
        encoding="utf-8"
    )

    assert "不得新增" in prompt
    assert "不得重新排序" in prompt
    assert "不得自行补全" in prompt
    assert "不得提出继续搜索" in prompt
