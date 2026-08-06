from app.schemas.intake import (
    IntakeChatResult,
    IntakeEntityAssessment,
    IntakeEntityResolution,
    IntakePersonCandidate,
    IntakeStructuredContext,
)
from app.schemas.task import EntityMention, IntentUnderstanding, WebPage
from app.services.entity_resolver import EntityResolver
from app.services.intake_completeness import is_intake_ready
from app.services.intake_field_state import (
    derive_field_states,
    fallback_confirmation_question,
    fields_ready_for_confirmation,
)


def _understanding(*, people=None, organizations=None) -> IntentUnderstanding:
    return IntentUnderstanding(
        intents=["MEETING_PREPARATION"],
        people=people or [],
        organizations=organizations or [],
        event_type="会议",
        overall_confidence=0.9,
    )


def test_completeness_uses_unicode_normalization() -> None:
    result = IntakeChatResult(
        assistant_reply="已记录。",
        analysis_input="了解 ABC 有限公司。",
        ready_to_analyze=True,
        structured_context=IntakeStructuredContext(
            organizations=["ＡＢＣ有限公司"]
        ),
    )

    assert is_intake_ready(result, "了解 ABC 有限公司。") is True


def test_organization_candidate_matches_normalized_short_name() -> None:
    resolver = EntityResolver()

    assert resolver.organization_candidate_matches(
        "比亚迪", "比亚迪股份有限公司"
    )
    assert not resolver.organization_candidate_matches(
        "比亚迪", "深圳迪比亚科技有限公司"
    )


def test_person_candidate_uses_name_fragment_and_compatible_title() -> None:
    resolver = EntityResolver()

    assert resolver.person_candidate_matches("王总", "王传福", "董事长兼总裁")
    assert not resolver.person_candidate_matches("王总", "王海", "工程师")
    assert resolver.person_candidate_matches("亚辉先生", "赵亚辉", "总经理")


def test_relationship_candidate_requires_person_and_organization_match() -> None:
    resolver = EntityResolver()

    assert resolver.relationship_candidate_matches(
        "王总",
        "比亚迪",
        candidate_name="王传福",
        candidate_organization="比亚迪股份有限公司",
        candidate_title="董事长兼总裁",
    )
    assert not resolver.relationship_candidate_matches(
        "王总",
        "比亚迪",
        candidate_name="王传福",
        candidate_organization="华星能源集团有限公司",
        candidate_title="董事长兼总裁",
    )


def test_candidate_lookup_accepts_full_title_and_courtesy_reference() -> None:
    organization = EntityMention(
        mention="华星能源集团有限公司",
        canonical_name="华星能源集团有限公司",
        evidence_text="华星能源集团有限公司",
        confidence=0.99,
        resolution="CONFIRMED",
    )
    resolver = EntityResolver()

    for mention in ("李董事长", "亚辉先生"):
        understanding = _understanding(
            people=[
                EntityMention(
                    mention=mention,
                    evidence_text=mention,
                    confidence=0.8,
                    resolution="NEEDS_CONFIRMATION",
                )
            ],
            organizations=[organization],
        )

        assert resolver.candidate_lookup(
            f"与华星能源集团有限公司的{mention}会面", understanding
        ) == (mention, "华星能源集团有限公司")


def test_web_rule_candidate_requires_compatible_reference_title() -> None:
    page = WebPage(
        title="管理团队",
        url="https://example.com/team",
        raw_content=(
            "华星能源集团有限公司总经理李海负责日常经营。"
            "华星能源集团有限公司董事长李明负责主持董事会。"
        ),
        rank=0,
    )

    candidates = EntityResolver().candidates_from_web(
        "李董事长", "华星能源集团有限公司", [page]
    )

    assert [candidate.canonical_name for candidate in candidates] == ["李明"]


def test_model_cannot_confirm_organization_canonical_name_absent_from_source() -> None:
    understanding = _understanding(
        people=[
            EntityMention(
                mention="张伟",
                canonical_name="张伟",
                evidence_text="张伟",
                confidence=0.99,
                resolution="CONFIRMED",
            )
        ],
        organizations=[
            EntityMention(
                mention="华星",
                canonical_name="华星能源集团有限公司",
                evidence_text="华星",
                confidence=0.99,
                resolution="CONFIRMED",
            )
        ],
    )

    context, confirmation = EntityResolver().resolve(
        "张伟与华星开会", understanding, version=1
    )

    assert context is None
    assert confirmation is not None
    assert [item.entity_type for item in confirmation.items] == ["ORGANIZATION"]
    assert confirmation.items[0].candidates == []


def test_unsupported_organization_canonical_keeps_empty_confirmation_item() -> None:
    understanding = _understanding(
        organizations=[
            EntityMention(
                mention="华星",
                canonical_name="华星能源集团有限公司",
                evidence_text="华星",
                confidence=0.99,
                resolution="NEEDS_CONFIRMATION",
            )
        ]
    )

    context, confirmation = EntityResolver().resolve(
        "与华星开会", understanding, version=1
    )

    assert context is None
    assert confirmation is not None
    organization_item = next(
        item for item in confirmation.items if item.entity_type == "ORGANIZATION"
    )
    assert organization_item.candidates == []
    assert all(
        candidate.canonical_name
        for item in confirmation.items
        for candidate in item.candidates
    )


def test_entity_deduplication_uses_normalized_identity_key() -> None:
    understanding = _understanding(
        organizations=[
            EntityMention(
                mention="ＡＢＣ有限公司",
                canonical_name="ＡＢＣ有限公司",
                evidence_text="ＡＢＣ有限公司",
                confidence=0.99,
                resolution="CONFIRMED",
            ),
            EntityMention(
                mention="ABC有限公司",
                canonical_name="ABC有限公司",
                evidence_text="ABC有限公司",
                confidence=0.99,
                resolution="CONFIRMED",
            ),
        ]
    )

    confirmed, _, _ = EntityResolver()._supported_entities(
        "ＡＢＣ有限公司和ABC有限公司", understanding
    )

    assert len(confirmed) == 1


def test_field_states_advance_from_completion_to_user_confirmation() -> None:
    incomplete = IntakeStructuredContext(
        people=["王总"],
        event_type="宴请",
        entity_assessments=[
            IntakeEntityAssessment(
                entity_type="PERSON",
                mention="王总",
                is_standard=False,
                reason="不是完整姓名",
            )
        ],
    )

    incomplete_states = derive_field_states(incomplete)

    assert incomplete_states["people"].status == "NEEDS_COMPLETION"
    assert fields_ready_for_confirmation(incomplete_states) is False

    completed = incomplete.model_copy(
        update={
            "people": ["王伟"],
            "people_details": [
                IntakePersonCandidate(
                    name="王伟",
                    title="总经理",
                    organization="示例科技有限公司",
                )
            ],
            "organizations": ["示例科技有限公司"],
            "entity_resolutions": [
                IntakeEntityResolution(
                    entity_type="PERSON",
                    mention="王总",
                    canonical_name="王伟",
                    organization="示例科技有限公司",
                    title="总经理",
                    confirmed_by="USER",
                ),
                IntakeEntityResolution(
                    entity_type="ORGANIZATION",
                    mention="示例科技",
                    canonical_name="示例科技有限公司",
                    confirmed_by="USER",
                ),
            ],
        }
    )
    completed_states = derive_field_states(completed)

    assert completed_states["people"].status == "STANDARD_COMPLETE"
    assert completed_states["organizations"].status == "STANDARD_COMPLETE"
    assert completed_states["event_type"].status == "STANDARD_COMPLETE"
    assert completed_states["event_time"].status == "NOT_PROVIDED"
    assert fields_ready_for_confirmation(completed_states) is True

    confirmed_states = derive_field_states(completed, final_confirmed=True)
    assert confirmed_states["people"].status == "USER_CONFIRMED"
    assert confirmed_states["event_type"].status == "USER_CONFIRMED"
    assert confirmed_states["event_time"].status == "NOT_PROVIDED"


def test_fallback_confirmation_uses_only_explicit_relationship() -> None:
    context = IntakeStructuredContext(
        people=["王伟"],
        organizations=["示例科技有限公司"],
        people_details=[
            IntakePersonCandidate(
                name="王伟",
                title="总经理",
                organization="示例科技有限公司",
            )
        ],
        event_type="宴请",
        event_time="今天晚上",
        event_location="滨江餐厅",
        focus_questions=["合作项目进展"],
    )

    question = fallback_confirmation_question(context)

    assert "今天晚上" in question
    assert "在滨江餐厅" in question
    assert "示例科技有限公司的总经理王伟" in question
    assert "见面吃饭" in question
    assert "合作项目进展" in question
