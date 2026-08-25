from app.schemas.intake import (
    IntakeEntityResolution,
    IntakeStructuredContext,
    IntakeToolAttempt,
)
from app.services.intake.agent_loop import AgentState, AgentTurn, ToolObservation


class IntakeStateReducer:
    """用经过 Schema 校验的输入生成新的不可变运行态。"""

    @staticmethod
    def apply_turn(state: AgentState, turn: AgentTurn) -> AgentState:
        context_data = state.context.model_dump(mode="python")
        context_data.update(turn.context_patch.updates)
        context_data["next_action"] = turn.next_action
        context_data["user_question"] = (
            turn.user_message if turn.next_action == "ASK_USER" else None
        )
        context = IntakeStructuredContext.model_validate(context_data)
        return AgentState(
            context=context,
            latest_observation=state.latest_observation,
            loop_count=state.loop_count + 1,
            llm_turn_count=state.llm_turn_count + 1,
        )

    @staticmethod
    def apply_observation(
        state: AgentState,
        observation: ToolObservation,
    ) -> AgentState:
        resolutions = IntakeStateReducer._merge_resolutions(
            state.context.entity_resolutions,
            observation.resolutions,
        )
        attempt = IntakeToolAttempt(
            action=observation.action,
            target_fields=observation.target_fields,
            query=observation.executed_query,
            technical_status=observation.technical_status,
            information_status=observation.information_status,
            observation=observation.error or observation.summary,
        )
        context_data = state.context.model_dump(mode="python")
        context_data.update(
            {
                "entity_resolutions": resolutions[-40:],
                "tool_attempts": [*state.context.tool_attempts, attempt][-20:],
            }
        )
        context = IntakeStructuredContext.model_validate(context_data)
        return AgentState(
            context=context,
            latest_observation=observation,
            loop_count=state.loop_count,
            llm_turn_count=state.llm_turn_count,
        )

    @staticmethod
    def preserve_after_llm_failure(state: AgentState) -> AgentState:
        return AgentState(
            context=state.context.model_copy(deep=True),
            latest_observation=state.latest_observation,
            loop_count=state.loop_count,
            llm_turn_count=state.llm_turn_count + 1,
        )

    @staticmethod
    def _merge_resolutions(
        existing: list[IntakeEntityResolution],
        additions: list[IntakeEntityResolution],
    ) -> list[IntakeEntityResolution]:
        merged = {
            (item.entity_type, item.canonical_name, item.organization): item
            for item in existing
        }
        for item in additions:
            merged[(item.entity_type, item.canonical_name, item.organization)] = item
        return list(merged.values())
