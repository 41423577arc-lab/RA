import pytest
from pydantic import ValidationError

from app.schemas.intake import (
    IntakeEntityResolution,
    IntakeStructuredContext,
    IntakeToolAttempt,
)
from app.services.intake.agent_loop import (
    AgentState,
    AgentTurn,
    IntakeContextPatch,
    QueryPlan,
    ToolObservation,
)
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
            "next_action": "ASK_USER",
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
