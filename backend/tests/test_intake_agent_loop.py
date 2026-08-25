import pytest
from pydantic import ValidationError

from app.schemas.intake import (
    IntakeEntityResolution,
    IntakeStructuredContext,
    IntakeToolAttempt,
)
from app.schemas.task import CandidateOption, ConfirmationItem, ConfirmationRequest
from app.services.intake.agent_loop import (
    AgentState,
    AgentTurn,
    IntakeContextPatch,
    MechanicalIntakeAgentLoop,
    QueryPlan,
    ToolObservation,
)
from app.services.intake.query_executor import IntakeQueryExecutor
from app.services.intake.state_reducer import IntakeStateReducer


def _context() -> IntakeStructuredContext:
    return IntakeStructuredContext(
        people=["刘希川"],
        organizations=["中建二局"],
        entity_resolutions=[
            IntakeEntityResolution(
                entity_type="PERSON",
                canonical_name="刘希川",
                mention="刘希川",
                organization="中建二局",
                confirmed_by="USER_INPUT",
            )
        ],
        tool_attempts=[
            IntakeToolAttempt(
                action="SEARCH_INTERNAL",
                query="人物:刘希川；企业:中建二局",
                technical_status="SUCCESS",
                information_status="PARTIAL",
                observation="找到企业候选",
            )
        ],
    )


def test_agent_state_keeps_business_facts_only_in_structured_context() -> None:
    state = AgentState(context=_context())

    assert set(AgentState.model_fields) == {
        "context",
        "latest_observation",
        "loop_count",
        "llm_turn_count",
    }
    assert state.context.people == ["刘希川"]
    assert not hasattr(state, "people")
    assert not hasattr(state, "entity_resolutions")


def test_context_patch_rejects_python_controlled_fields() -> None:
    with pytest.raises(ValidationError, match="不可修改字段"):
        IntakeContextPatch(updates={"entity_resolutions": []})

    with pytest.raises(ValidationError, match="不可修改字段"):
        IntakeContextPatch(updates={"tool_attempts": []})


def test_context_patch_uses_canonical_context_field_validation() -> None:
    patch = IntakeContextPatch(
        updates={
            "organizations": ["中国建筑第二工程局有限公司"],
            "target_fields": ["organization_full_name"],
        }
    )

    assert patch.updates["organizations"] == ["中国建筑第二工程局有限公司"]
    with pytest.raises(ValidationError):
        IntakeContextPatch(updates={"organizations": "中建二局"})


def test_agent_turn_requires_action_specific_payload() -> None:
    with pytest.raises(ValidationError, match="必须提供 QueryPlan"):
        AgentTurn(
            skill="internal_lookup",
            next_action="SEARCH_INTERNAL",
            reason="需要查询内部候选",
        )

    with pytest.raises(ValidationError, match="必须提供用户问题"):
        AgentTurn(
            skill="identity_resolution",
            next_action="ASK_USER",
            reason="身份仍有歧义",
        )


def test_reducer_applies_semantic_patch_without_losing_controlled_state() -> None:
    state = AgentState(context=_context())
    turn = AgentTurn(
        context_patch=IntakeContextPatch(
            updates={
                "organizations": ["中国建筑第二工程局有限公司"],
                "target_fields": ["organization_full_name"],
            }
        ),
        skill="identity_resolution",
        next_action="ASK_USER",
        user_message="请确认企业全称。",
        reason="内部候选需要用户确认",
    )

    reduced = IntakeStateReducer.apply_turn(state, turn)

    assert reduced.context.organizations == ["中国建筑第二工程局有限公司"]
    assert reduced.context.people == ["刘希川"]
    assert reduced.context.entity_resolutions == state.context.entity_resolutions
    assert reduced.context.tool_attempts == state.context.tool_attempts
    assert reduced.context.user_question == "请确认企业全称。"
    assert reduced.loop_count == 1
    assert reduced.llm_turn_count == 1


def test_reducer_records_typed_observation_without_promoting_candidate() -> None:
    state = AgentState(context=_context(), loop_count=1, llm_turn_count=1)
    observation = ToolObservation(
        action="SEARCH_INTERNAL",
        target_fields=["organization_full_name"],
        executed_query="人物:刘希川；企业:中建二局",
        technical_status="SUCCESS",
        information_status="PARTIAL",
        resolutions=[],
        summary="找到一个企业候选，仍需用户确认",
    )

    reduced = IntakeStateReducer.apply_observation(state, observation)

    assert reduced.latest_observation == observation
    assert reduced.context.entity_resolutions == state.context.entity_resolutions
    assert reduced.context.tool_attempts[-1].information_status == "PARTIAL"
    assert reduced.loop_count == 1
    assert reduced.llm_turn_count == 1


def test_tool_observation_distinguishes_technical_failure() -> None:
    observation = ToolObservation(
        action="SEARCH_PUBLIC",
        executed_query="中建二局 刘希川",
        technical_status="FAILED",
        information_status="NO_RESULT",
        error="Tavily 请求超时",
    )

    assert observation.technical_status == "FAILED"
    assert observation.information_status == "NO_RESULT"
    with pytest.raises(ValidationError, match="必须为 NO_RESULT"):
        ToolObservation(
            action="SEARCH_PUBLIC",
            executed_query="中建二局 刘希川",
            technical_status="FAILED",
            information_status="PARTIAL",
            error="Tavily 请求超时",
        )


def test_llm_failure_preserves_the_complete_context() -> None:
    state = AgentState(context=_context(), loop_count=2, llm_turn_count=2)

    reduced = IntakeStateReducer.preserve_after_llm_failure(state)

    assert reduced.context == state.context
    assert reduced.context is not state.context
    assert reduced.loop_count == 2
    assert reduced.llm_turn_count == 3


def test_query_plan_contains_semantic_targets_but_no_executable_query() -> None:
    plan = QueryPlan(
        action="SEARCH_INTERNAL",
        target_fields=["person_organization"],
        person_mentions=["刘希川"],
        organization_mentions=["中建二局"],
    )

    assert plan.action == "SEARCH_INTERNAL"
    assert "query" not in QueryPlan.model_fields
    assert "sql" not in QueryPlan.model_fields


class _CandidateBackend:
    def __init__(
        self,
        *,
        internal_result=([], None),
        public_result=None,
        error: Exception | None = None,
    ):
        self.internal_result = internal_result
        self.public_result = public_result
        self.error = error
        self.contexts = []

    def lookup_internal(
        self,
        context,
        _version,
        _source_text,
        *,
        raise_on_error,
    ):
        assert raise_on_error is True
        self.contexts.append(context)
        if self.error:
            raise self.error
        return self.internal_result

    def search_key_person_identity_web(
        self,
        context,
        _confirmation,
        _normalizer,
        *,
        raise_on_error,
    ):
        assert raise_on_error is True
        self.contexts.append(context)
        if self.error:
            raise self.error
        return self.public_result

    @staticmethod
    def apply_automatic_candidates(resolutions, confirmation, _threshold):
        return resolutions, confirmation


def _query_plan(action="SEARCH_INTERNAL") -> QueryPlan:
    return QueryPlan(
        action=action,
        target_fields=["person_organization"],
        person_mentions=["刘希川"],
        organization_mentions=["中建二局"],
    )


def test_query_executor_uses_only_names_already_in_context() -> None:
    backend = _CandidateBackend()
    plan = QueryPlan(
        action="SEARCH_INTERNAL",
        target_fields=["person_organization"],
        person_mentions=["模型虚构的人名"],
        organization_mentions=["模型虚构的企业"],
    )

    observation = IntakeQueryExecutor(backend).execute(
        plan,
        _context(),
        version=1,
        source_text="今晚和中建二局刘希川吃饭",
    )

    assert backend.contexts[0].people == ["刘希川"]
    assert backend.contexts[0].organizations == ["中建二局"]
    assert observation.executed_query == "人物:刘希川；企业:中建二局"


def test_query_executor_returns_typed_partial_observation() -> None:
    candidate = CandidateOption(
        candidate_id="internal:customer:C024",
        entity_type="ORGANIZATION",
        canonical_name="中建二局安装工程有限公司",
        reason="内部客户候选",
        confidence=0.8,
    )
    confirmation = ConfirmationRequest(
        version=1,
        items=[
            ConfirmationItem(
                mention="中建二局",
                entity_type="ORGANIZATION",
                candidates=[candidate],
            )
        ],
    )
    backend = _CandidateBackend(internal_result=([], confirmation))

    observation = IntakeQueryExecutor(backend).execute(
        _query_plan(),
        _context(),
        version=1,
    )

    assert observation.technical_status == "SUCCESS"
    assert observation.information_status == "PARTIAL"
    assert observation.confirmation.items[0].candidates[0] == candidate


def test_query_executor_distinguishes_empty_result_from_tool_failure() -> None:
    empty = IntakeQueryExecutor(_CandidateBackend()).execute(
        _query_plan(),
        _context(),
        version=1,
    )
    failed = IntakeQueryExecutor(
        _CandidateBackend(error=TimeoutError("MCP 请求超时"))
    ).execute(
        _query_plan(),
        _context(),
        version=1,
    )

    assert empty.technical_status == "SUCCESS"
    assert empty.information_status == "NO_RESULT"
    assert empty.error is None
    assert failed.technical_status == "FAILED"
    assert failed.information_status == "NO_RESULT"
    assert "MCP 请求超时" in failed.error


def test_query_executor_public_search_requires_prior_confirmation() -> None:
    observation = IntakeQueryExecutor(_CandidateBackend()).execute(
        _query_plan("SEARCH_PUBLIC"),
        _context(),
        version=1,
    )

    assert observation.technical_status == "FAILED"
    assert "缺少内部候选" in observation.error


def test_query_executor_calls_existing_public_candidate_service() -> None:
    confirmation = ConfirmationRequest(
        version=2,
        items=[
            ConfirmationItem(
                mention="刘希川",
                entity_type="PERSON",
                candidates=[],
            )
        ],
    )
    backend = _CandidateBackend(public_result=confirmation)

    observation = IntakeQueryExecutor(backend).execute(
        _query_plan("SEARCH_PUBLIC"),
        _context(),
        version=2,
        confirmation=confirmation,
        external_normalizer=lambda *_: None,
    )

    assert len(backend.contexts) == 1
    assert observation.technical_status == "SUCCESS"
    assert observation.information_status == "NO_RESULT"
    assert observation.confirmation == confirmation


class _DecisionProvider:
    def __init__(self, turns=None, error: Exception | None = None):
        self.turns = list(turns or [])
        self.error = error
        self.states = []

    def decide(self, state):
        self.states.append(state)
        if self.error:
            raise self.error
        return self.turns.pop(0)


def _agent_turn(action, *, question=None) -> AgentTurn:
    if action == "ASK_USER":
        return AgentTurn(
            skill="identity_resolution",
            next_action=action,
            user_message=question or "请确认目标人物和企业。",
            reason="仍需用户确认",
        )
    if action == "READY":
        return AgentTurn(
            skill="intake_readiness",
            next_action=action,
            reason="身份信息已经完整",
        )
    return AgentTurn(
        skill="internal_lookup"
        if action == "SEARCH_INTERNAL"
        else "public_lookup",
        next_action=action,
        query_plan=_query_plan(action),
        reason="需要继续查询身份",
    )


def _mechanical_loop(provider, backend, **limits) -> MechanicalIntakeAgentLoop:
    return MechanicalIntakeAgentLoop(
        provider,
        IntakeQueryExecutor(backend),
        IntakeStateReducer,
        **limits,
    )


def test_mechanical_loop_runs_internal_public_then_waits_for_user() -> None:
    confirmation = ConfirmationRequest(
        version=1,
        items=[
            ConfirmationItem(
                mention="刘希川",
                entity_type="PERSON",
                candidates=[],
            )
        ],
    )
    provider = _DecisionProvider(
        [
            _agent_turn("SEARCH_INTERNAL"),
            _agent_turn("SEARCH_PUBLIC"),
            _agent_turn("ASK_USER"),
        ]
    )
    backend = _CandidateBackend(
        internal_result=([], confirmation),
        public_result=confirmation,
    )
    checkpoints = []

    result = _mechanical_loop(provider, backend).run(
        _context(),
        version=1,
        source_text="今晚和中建二局刘希川吃饭",
        hard_gate=lambda _context: False,
        external_normalizer=lambda *_: None,
        checkpoint=lambda context, pending: checkpoints.append((context, pending)),
    )

    assert result.stop_reason == "WAITING_USER"
    assert result.tool_calls == 2
    assert [item.action for item in result.state.context.tool_attempts[-2:]] == [
        "SEARCH_INTERNAL",
        "SEARCH_PUBLIC",
    ]
    assert result.state.context.user_question == "请确认目标人物和企业。"
    assert result.confirmation == confirmation
    assert len(checkpoints) == 4


def test_mechanical_loop_python_hard_gate_can_promote_ask_user_to_ready() -> None:
    context = _context().model_copy(
        update={
            "entity_resolutions": [
                *_context().entity_resolutions,
                IntakeEntityResolution(
                    entity_type="ORGANIZATION",
                    canonical_name="中建二局",
                    mention="中建二局",
                    confirmed_by="USER_INPUT",
                ),
            ]
        }
    )
    provider = _DecisionProvider([_agent_turn("ASK_USER")])

    result = _mechanical_loop(provider, _CandidateBackend()).run(
        context,
        version=1,
        source_text=None,
        hard_gate=lambda value: len(value.entity_resolutions) == 2,
    )

    assert result.stop_reason == "READY"
    assert result.tool_calls == 0
    assert result.state.context.next_action == "READY"


def test_mechanical_loop_rejects_model_ready_before_hard_gate() -> None:
    provider = _DecisionProvider(
        [_agent_turn("READY"), _agent_turn("ASK_USER")]
    )

    result = _mechanical_loop(provider, _CandidateBackend()).run(
        _context().model_copy(update={"tool_attempts": []}),
        version=1,
        source_text=None,
        hard_gate=lambda _context: False,
    )

    assert result.stop_reason == "WAITING_USER"
    assert result.tool_calls == 1
    assert result.state.context.tool_attempts[-1].action == "SEARCH_INTERNAL"


def test_mechanical_loop_stops_duplicate_controlled_query() -> None:
    provider = _DecisionProvider(
        [_agent_turn("SEARCH_INTERNAL"), _agent_turn("SEARCH_INTERNAL")]
    )

    result = _mechanical_loop(provider, _CandidateBackend()).run(
        _context(),
        version=1,
        source_text=None,
        hard_gate=lambda _context: False,
    )

    assert result.stop_reason == "REPEATED_ACTION"
    assert result.tool_calls == 1
    assert "相同身份查询" in result.state.context.user_question


def test_mechanical_loop_preserves_context_when_model_fails() -> None:
    context = _context()
    provider = _DecisionProvider(error=RuntimeError("模型输出校验失败"))

    result = _mechanical_loop(provider, _CandidateBackend()).run(
        context,
        version=1,
        source_text=None,
        hard_gate=lambda _context: False,
    )

    assert result.stop_reason == "WAITING_USER"
    assert result.state.context.people == context.people
    assert result.state.context.organizations == context.organizations
    assert result.state.context.entity_resolutions == context.entity_resolutions
    assert result.state.llm_turn_count == 1


def test_mechanical_loop_enforces_tool_limit() -> None:
    confirmation = ConfirmationRequest(
        version=1,
        items=[
            ConfirmationItem(
                mention="刘希川",
                entity_type="PERSON",
                candidates=[],
            )
        ],
    )
    provider = _DecisionProvider(
        [_agent_turn("SEARCH_INTERNAL"), _agent_turn("SEARCH_PUBLIC")]
    )
    backend = _CandidateBackend(
        internal_result=([], confirmation),
        public_result=confirmation,
    )

    result = _mechanical_loop(provider, backend, max_tool_calls=1).run(
        _context(),
        version=1,
        source_text=None,
        hard_gate=lambda _context: False,
        external_normalizer=lambda *_: None,
    )

    assert result.stop_reason == "MAX_TOOL_CALLS"
    assert result.tool_calls == 1
