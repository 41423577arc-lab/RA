from app.schemas.intake import (
    IntakeChatRequest,
    IntakeChatResult,
    IntakeEntityAssessment,
    IntakeEntityResolution,
    IntakeFieldState,
    IntakePersonCandidate,
    IntakeResolutionResult,
    IntakeStructuredContext,
    IntakeToolAttempt,
)
from app.schemas.task import (
    CandidateOption,
    ConfirmationItem,
    ConfirmationRequest,
    EntityMention,
    IntentUnderstanding,
    WebPage,
)
from app.services.intake.entity_resolver import EntityResolver
from app.services.intake.completeness import is_intake_ready
from app.services.intake.field_state import (
    derive_field_states,
    fallback_confirmation_question,
    fields_ready_for_confirmation,
)
from app.services.intake.identity_loop import IntakeIdentityLoop


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


def test_identity_context_schema_expresses_initialization_to_ready_flow() -> None:
    initialized = IntakeStructuredContext(
        people=["王总"],
        field_states={
            "people": IntakeFieldState(
                value=["王总"],
                status="AMBIGUOUS",
                source="USER_INPUT",
                required=True,
                reason="称谓不能唯一确定身份",
            ),
            "organizations": IntakeFieldState(
                value=None,
                status="MISSING",
                source="USER_INPUT",
                required=True,
                reason="未提供所属组织",
            ),
        },
        target_fields=["people", "organizations"],
        next_action="SEARCH_INTERNAL",
        success_criteria=["找到与王总称谓匹配的人物及所属组织候选"],
    )

    after_internal_search = initialized.model_copy(
        update={
            "field_states": {
                "people": IntakeFieldState(
                    value=["王伟", "王强"],
                    status="CANDIDATE",
                    source="INTERNAL:mcp.search_projects",
                    required=True,
                    reason="内部查询返回两个同称谓候选人",
                ),
                "organizations": IntakeFieldState(
                    value=["示例科技有限公司", "示例能源有限公司"],
                    status="AMBIGUOUS",
                    source="INTERNAL:mcp.search_projects",
                    required=True,
                    reason="候选人分别属于不同组织",
                ),
            },
            "target_fields": ["people", "organizations"],
            "next_action": "ASK_USER",
            "success_criteria": ["用户明确选择人物及其所属组织"],
            "resolution_result": IntakeResolutionResult(
                status="PARTIAL",
                updated_fields=["people", "organizations"],
                summary="内部查询获得候选，但仍需用户消歧",
            ),
            "user_question": "您指的是示例科技的王伟，还是示例能源的王强？",
            "tool_attempts": [
                IntakeToolAttempt(
                    action="SEARCH_INTERNAL",
                    target_fields=["people", "organizations"],
                    query="王总",
                    technical_status="SUCCESS",
                    information_status="PARTIAL",
                    observation="返回王伟和王强两个候选人",
                )
            ],
        }
    )

    ready = after_internal_search.model_copy(
        update={
            "people": ["王伟"],
            "organizations": ["示例科技有限公司"],
            "field_states": {
                "people": IntakeFieldState(
                    value=["王伟"],
                    status="CONFIRMED",
                    source="USER_REPLY",
                    required=True,
                ),
                "organizations": IntakeFieldState(
                    value=["示例科技有限公司"],
                    status="CONFIRMED",
                    source="USER_REPLY",
                    required=True,
                ),
            },
            "target_fields": [],
            "next_action": "READY",
            "success_criteria": [],
            "resolution_result": IntakeResolutionResult(
                status="RESOLVED",
                updated_fields=["people", "organizations"],
                summary="用户已确认人物和所属组织",
            ),
            "user_question": None,
        }
    )

    assert initialized.field_states["organizations"].status == "MISSING"
    assert after_internal_search.tool_attempts[0].technical_status == "SUCCESS"
    assert after_internal_search.tool_attempts[0].information_status == "PARTIAL"
    assert after_internal_search.next_action == "ASK_USER"
    assert ready.next_action == "READY"
    assert all(state.status == "CONFIRMED" for state in ready.field_states.values())


class _IdentityLoopAgent:
    def initialize_context(self, _request, extracted_context):
        return extracted_context.model_copy(
            update={
                "field_states": {
                    "people": IntakeFieldState(
                        value=["王总"],
                        status="AMBIGUOUS",
                        source="USER_INPUT",
                        required=True,
                    )
                },
                "target_fields": ["people"],
                "next_action": "SEARCH_INTERNAL",
                "success_criteria": ["找到王总的唯一身份"],
            }
        )

    def update_context(
        self,
        _request,
        old_context,
        *,
        extracted_context=None,
        tool_observation=None,
    ):
        if tool_observation is None:
            return old_context.model_copy(
                update={
                    "people": ["王伟"],
                    "organizations": ["示例科技有限公司"],
                    "target_fields": ["people", "organizations"],
                    "next_action": "READY",
                    "success_criteria": ["核验用户确认的标准身份"],
                    "user_question": None,
                }
            )
        if tool_observation["action"] == "SEARCH_INTERNAL" and not tool_observation[
            "resolutions"
        ]:
            return old_context.model_copy(
                update={
                    "next_action": "SEARCH_PUBLIC",
                    "success_criteria": ["从公开证据中找到唯一身份"],
                }
            )
        if tool_observation["action"] == "SEARCH_PUBLIC":
            return old_context.model_copy(
                update={
                    "next_action": "ASK_USER",
                    "user_question": "您指的是示例科技有限公司的王伟吗？",
                    "resolution_result": IntakeResolutionResult(
                        status="PARTIAL",
                        updated_fields=["people"],
                        summary="已找到候选，但仍需用户确认",
                    ),
                }
            )
        return old_context.model_copy(update={"next_action": "READY"})


class _EventRecorder:
    def __init__(self):
        self.events = []

    def log_execution_event(self, scope_id, **values):
        self.events.append((scope_id, values))


def _identity_hard_gate(context: IntakeStructuredContext) -> bool:
    resolved = {
        value
        for item in context.entity_resolutions
        for value in (item.mention, item.canonical_name)
    }
    return {*context.people, *context.organizations}.issubset(resolved)


def test_intake_identity_loop_resumes_after_internal_public_and_user_reply() -> None:
    request = IntakeChatRequest(
        messages=[{"role": "user", "content": "我要见王总"}]
    )
    candidate = CandidateOption(
        candidate_id="web:wang-wei",
        entity_type="PERSON",
        canonical_name="王伟",
        organization="示例科技有限公司",
        title="总经理",
        reason="公开页面同时出现姓名、企业与职位",
        confidence=0.9,
        source_url="https://example.com/wang-wei",
        evidence_quote="王伟现任示例科技有限公司总经理",
    )
    internal_confirmation = ConfirmationRequest(
        version=1,
        items=[
            ConfirmationItem(
                mention="王总", entity_type="PERSON", candidates=[]
            )
        ],
    )
    public_confirmation = ConfirmationRequest(
        version=1,
        items=[
            ConfirmationItem(
                mention="王总", entity_type="PERSON", candidates=[candidate]
            )
        ],
    )
    recorder = _EventRecorder()
    checkpoints = []
    loop = IntakeIdentityLoop(
        _IdentityLoopAgent(), recorder, str(request.session_id)
    )

    waiting = loop.run(
        request,
        IntakeStructuredContext(people=["王总"]),
        None,
        lookup_internal=lambda _context: ([], internal_confirmation),
        lookup_public=lambda _context, _confirmation: ([], public_confirmation),
        hard_gate=_identity_hard_gate,
        checkpoint=lambda context, confirmation: checkpoints.append(
            (context.model_dump(mode="json"), confirmation)
        ),
    )

    assert waiting.stop_reason == "WAITING_USER"
    assert [attempt.action for attempt in waiting.context.tool_attempts] == [
        "SEARCH_INTERNAL",
        "SEARCH_PUBLIC",
    ]
    assert waiting.context.tool_attempts[0].technical_status == "SUCCESS"
    assert waiting.context.tool_attempts[0].information_status == "NO_RESULT"
    assert waiting.context.tool_attempts[1].information_status == "PARTIAL"
    assert waiting.context.user_question == "您指的是示例科技有限公司的王伟吗？"
    assert checkpoints[-1][0]["next_action"] == "ASK_USER"
    assert len(checkpoints[-1][0]["tool_attempts"]) == 2

    confirmed_resolution = IntakeEntityResolution(
        entity_type="PERSON",
        mention="王伟",
        canonical_name="王伟",
        organization="示例科技有限公司",
        title="总经理",
        confirmed_by="USER_INPUT",
    ).model_dump(mode="json")
    organization_resolution = IntakeEntityResolution(
        entity_type="ORGANIZATION",
        mention="示例科技有限公司",
        canonical_name="示例科技有限公司",
        confirmed_by="USER_INPUT",
    ).model_dump(mode="json")
    followup_request = IntakeChatRequest(
        session_id=request.session_id,
        messages=[
            {"role": "user", "content": "我要见王总"},
            {"role": "assistant", "content": waiting.context.user_question},
            {"role": "user", "content": "是示例科技的王伟"},
        ],
    )
    resumed = loop.run(
        followup_request,
        IntakeStructuredContext(
            people=["王伟"], organizations=["示例科技有限公司"]
        ),
        waiting.context,
        waiting.confirmation,
        lookup_internal=lambda _context: (
            [confirmed_resolution, organization_resolution],
            None,
        ),
        lookup_public=lambda _context, confirmation: ([], confirmation),
        hard_gate=_identity_hard_gate,
    )

    assert resumed.stop_reason == "READY"
    assert resumed.tool_calls == 1
    assert resumed.context.next_action == "READY"
    assert len(resumed.context.entity_resolutions) == 2


def test_intake_identity_loop_hard_gate_verifies_clear_identity() -> None:
    class ClearIdentityAgent(_IdentityLoopAgent):
        def initialize_context(self, _request, extracted_context):
            return extracted_context.model_copy(
                update={
                    "next_action": "READY",
                    "target_fields": [],
                    "success_criteria": [],
                }
            )

    request = IntakeChatRequest(
        messages=[
            {
                "role": "user",
                "content": "我要见示例科技有限公司总经理王伟",
            }
        ]
    )
    context = IntakeStructuredContext(
        people=["王伟"], organizations=["示例科技有限公司"]
    )
    resolutions = [
        IntakeEntityResolution(
            entity_type="PERSON",
            mention="王伟",
            canonical_name="王伟",
            organization="示例科技有限公司",
            title="总经理",
            confirmed_by="USER_INPUT",
        ).model_dump(mode="json"),
        IntakeEntityResolution(
            entity_type="ORGANIZATION",
            mention="示例科技有限公司",
            canonical_name="示例科技有限公司",
            confirmed_by="USER_INPUT",
        ).model_dump(mode="json"),
    ]
    loop = IntakeIdentityLoop(
        ClearIdentityAgent(), _EventRecorder(), str(request.session_id)
    )

    result = loop.run(
        request,
        context,
        None,
        lookup_internal=lambda _context: (resolutions, None),
        lookup_public=lambda _context, confirmation: ([], confirmation),
        hard_gate=_identity_hard_gate,
    )

    assert result.stop_reason == "READY"
    assert result.tool_calls == 1
    assert result.context.tool_attempts[0].information_status == "RESOLVED"


def test_intake_identity_loop_separates_tool_failure_from_information_result() -> None:
    request = IntakeChatRequest(
        messages=[{"role": "user", "content": "我要见王总"}]
    )
    loop = IntakeIdentityLoop(
        _IdentityLoopAgent(), _EventRecorder(), str(request.session_id)
    )

    def failed_internal(_context):
        raise TimeoutError("MCP timeout")

    result = loop.run(
        request,
        IntakeStructuredContext(people=["王总"]),
        None,
        lookup_internal=failed_internal,
        lookup_public=lambda _context, confirmation: ([], confirmation),
        hard_gate=_identity_hard_gate,
    )

    assert result.stop_reason == "WAITING_USER"
    assert result.context.tool_attempts[0].technical_status == "FAILED"
    assert result.context.tool_attempts[0].information_status == "NO_RESULT"
