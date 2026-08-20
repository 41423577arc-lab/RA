from datetime import date

from app.schemas.task import ConfirmedContext, ConfirmedEntity, ProjectResult
from app.services.research.project_ranker import ProjectRanker


REFERENCE_DATE = date(2026, 8, 14)


def _context() -> ConfirmedContext:
    return ConfirmedContext(
        intents=["INTERNAL_PROJECT_QUERY"],
        entities=[
            ConfirmedEntity(
                entity_type="PERSON",
                canonical_name="王志远",
                organization="华北筑联供应链有限公司",
                confirmed_by="USER",
            ),
            ConfirmedEntity(
                entity_type="ORGANIZATION",
                canonical_name="华北筑联供应链有限公司",
                confirmed_by="USER",
            ),
        ],
        event_type="会议",
        business_directions=["供应链"],
    )


def _p021() -> ProjectResult:
    return ProjectResult(
        project_id="P021",
        project_name="工程物资供应链协同平台",
        customer_name="华北筑联供应链有限公司",
        contact_name="王志远",
        status="ACTIVE",
        owner_name="张伟",
        start_date=date(2026, 4, 12),
        description="工程物资供应链协同",
        project_stage="PROPOSAL",
        health_status="AMBER",
        priority="P1",
        last_activity_date=date(2026, 7, 20),
        match_type="PERSON_EXACT",
    )


def _p015() -> ProjectResult:
    return ProjectResult(
        project_id="P015",
        project_name="物流车辆调度系统",
        customer_name="快达物流集团",
        contact_name="罗杰",
        status="ACTIVE",
        owner_name="陈杰",
        start_date=date(2025, 10, 10),
        description="物流车辆智能调度",
        project_stage="QUALIFICATION",
        health_status="GREEN",
        priority="P2",
        last_activity_date=date(2026, 3, 1),
        match_type="TEXT_MATCH",
        similarity=0.3,
    )


def test_project_ranker_produces_explainable_fixed_scores() -> None:
    rankings = ProjectRanker().rank(
        [_p015(), _p021()],
        _context(),
        reference_date=REFERENCE_DATE,
    )

    assert [(item.project_id, item.score, item.rank) for item in rankings] == [
        ("P021", 87, 1),
        ("P015", 34, 2),
    ]
    assert rankings[0].reason_codes == [
        "MATCH_PERSON_EXACT:+35",
        "CONTEXT_PERSON_EXACT:+12",
        "CONTEXT_ORGANIZATION_EXACT:+10",
        "CONTEXT_BUSINESS_TERM:供应链:+4",
        "STATUS_ACTIVE:+8",
        "STAGE_PROPOSAL:+6",
        "PRIORITY_P1:+4",
        "HEALTH_AMBER:+2",
        "ACTIVITY_WITHIN_30D:+6",
    ]
    assert rankings[1].reason_codes == [
        "MATCH_TEXT_MATCH:+12",
        "SIMILARITY_0.30:+3",
        "STATUS_ACTIVE:+8",
        "STAGE_QUALIFICATION:+4",
        "PRIORITY_P2:+2",
        "HEALTH_GREEN:+3",
        "ACTIVITY_WITHIN_180D:+2",
    ]
    assert all(item.relevance_score == item.score for item in rankings)
    assert all(item.recommended_use == "" for item in rankings)


def test_project_ranker_is_stable_and_filters_duplicate_project_ids() -> None:
    ranker = ProjectRanker()
    first = ranker.rank(
        [_p015(), _p021(), _p021()],
        _context(),
        reference_date=REFERENCE_DATE,
    )
    second = ranker.rank(
        [_p021(), _p015()],
        _context(),
        reference_date=REFERENCE_DATE,
    )

    assert [item.model_dump() for item in first] == [
        item.model_dump() for item in second
    ]
    assert len(first) == 2


def test_legacy_project_ranking_payload_derives_score() -> None:
    from app.schemas.task import ProjectRanking

    ranking = ProjectRanking(
        project_id="P001",
        relevance_score=60,
        relevance_reason="旧数据",
        recommended_use="",
        confidence=0.8,
    )

    assert ranking.score == 60
    assert ranking.reason_codes == []
    assert ranking.rank is None
