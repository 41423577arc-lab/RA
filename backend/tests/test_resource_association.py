from datetime import date

from app.schemas.task import (
    ConfirmedContext,
    ConfirmedEntity,
    ProjectRanking,
    ProjectResult,
    PublicClaim,
)
from app.services.resource_association import ResourceAssociationBuilder


def _context() -> ConfirmedContext:
    return ConfirmedContext(
        intents=["MEETING_PREPARATION", "INTERNAL_PROJECT_QUERY"],
        entities=[
            ConfirmedEntity(
                entity_type="PERSON",
                canonical_name="王志远",
                organization="华北筑联供应链有限公司",
                confirmed_by="USER",
            )
        ],
        event_type="会议",
    )


def _claim() -> PublicClaim:
    return PublicClaim(
        web_result_id="W001",
        evidence_id="E1",
        subject="王志远",
        claim="王志远现任华北筑联供应链有限公司总经理",
        evidence_quote="王志远现任华北筑联供应链有限公司总经理。",
        source_title="企业信息",
        source_url="https://example.com/person",
    )


def _project(
    project_id: str,
    *,
    contact_name: str | None,
    owner_phone: str | None,
    match_type: str,
    health: str,
    activity: date,
) -> ProjectResult:
    return ProjectResult(
        project_id=project_id,
        project_name=f"项目{project_id}",
        customer_name="华北筑联供应链有限公司",
        contact_name=contact_name,
        customer_contact_title="总经理" if contact_name else None,
        customer_contact_phone="17010000001" if contact_name else None,
        status="ACTIVE",
        owner_name="张伟",
        owner_phone=owner_phone,
        owner_email="zhangwei@example.com" if owner_phone else None,
        start_date=date(2025, 1, 1),
        description="供应链协同项目",
        project_stage="PROPOSAL",
        health_status=health,
        priority="P1",
        last_activity_date=activity,
        match_type=match_type,
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


def test_builder_only_organizes_resources_gaps_and_deterministic_risks() -> None:
    p1 = _project(
        "P001",
        contact_name=None,
        owner_phone=None,
        match_type="VECTOR_MATCH",
        health="GREEN",
        activity=date(2025, 10, 1),
    )
    p2 = _project(
        "P002",
        contact_name="王志远",
        owner_phone="17000000001",
        match_type="PERSON_EXACT",
        health="RED",
        activity=date(2026, 8, 1),
    )

    analysis = ResourceAssociationBuilder().build(
        _context(),
        [_claim()],
        [p1, p2],
        [_ranking("P001", 30, 2), _ranking("P002", 85, 1)],
        reference_date=date(2026, 8, 14),
    )

    assert analysis.key_findings == []
    assert analysis.recommended_topics == []
    assert analysis.next_actions == []
    assert [item.evidence_refs for item in analysis.related_projects] == [
        ["PROJECT:P002"],
        ["PROJECT:P001"],
    ]
    assert analysis.available_resources[0].text.startswith("客户联系人为王志远")
    assert analysis.available_resources[1].text.startswith("我方项目销售员为张伟")
    assert analysis.available_resources[2].text.startswith("我方项目销售员为张伟")
    assert analysis.available_resources[1].evidence_refs == ["PROJECT:P002"]
    assert analysis.available_resources[2].evidence_refs == ["PROJECT:P001"]
    assert [item.text for item in analysis.information_gaps] == [
        "MISSING_CUSTOMER_CONTACT:P001",
        "MISSING_INTERNAL_OWNER_CONTACT:P001",
    ]
    assert [item.text for item in analysis.risks] == [
        "PROJECT_HEALTH_RED:P002",
        "FUZZY_PROJECT_MATCH:P001",
        "LOW_RELEVANCE_SCORE:P001",
        "STALE_PROJECT_ACTIVITY:P001",
    ]
    all_items = [
        *analysis.related_projects,
        *analysis.available_resources,
        *analysis.information_gaps,
        *analysis.risks,
    ]
    assert all(item.statement_type == "FACT" for item in all_items)
    assert not any(
        word in item.text
        for item in all_items
        for word in ("建议", "可围绕", "下一步", "机会")
    )


def test_builder_reports_missing_verified_identity_and_ranked_projects() -> None:
    analysis = ResourceAssociationBuilder().build(_context(), [], [], [])

    assert [item.text for item in analysis.information_gaps] == [
        "MISSING_VERIFIED_PERSON_EVIDENCE:王志远",
        "NO_RANKED_PROJECTS",
    ]
    assert analysis.related_projects == []
    assert analysis.available_resources == []
    assert analysis.risks == []
