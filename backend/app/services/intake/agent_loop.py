from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.intake import (
    IntakeAction,
    IntakeEntityResolution,
    IntakeEntityType,
    IntakeStructuredContext,
    IntakeToolAction,
)
from app.schemas.task import ConfirmationRequest
from app.services.infrastructure.loop_controls import RepeatedActionGuard


IntakeSkillName = Literal[
    "identity_resolution",
    "internal_lookup",
    "public_lookup",
    "intake_readiness",
]
TechnicalStatus = Literal["SUCCESS", "FAILED"]
InformationStatus = Literal["RESOLVED", "PARTIAL", "NO_RESULT"]

_CONTROLLED_CONTEXT_FIELDS = {
    "entity_resolutions",
    "tool_attempts",
    "final_confirmation",
    "next_action",
    "user_question",
}
_PATCHABLE_CONTEXT_FIELDS = (
    set(IntakeStructuredContext.model_fields) - _CONTROLLED_CONTEXT_FIELDS
)


class IntakeContextPatch(BaseModel):
    """表示单次模型推理建议更新的业务字段。"""

    model_config = ConfigDict(frozen=True)

    updates: dict[str, Any] = Field(default_factory=dict, max_length=30)

    @field_validator("updates")
    @classmethod
    def validate_updates(cls, updates: dict[str, Any]) -> dict[str, Any]:
        unknown_fields = set(updates) - _PATCHABLE_CONTEXT_FIELDS
        if unknown_fields:
            fields = "、".join(sorted(unknown_fields))
            raise ValueError(f"Context Patch 包含不可修改字段：{fields}")

        normalized: dict[str, Any] = {}
        for field_name, value in updates.items():
            context = IntakeStructuredContext.model_validate({field_name: value})
            normalized[field_name] = getattr(context, field_name)
        return normalized


class QueryPlan(BaseModel):
    """模型只描述查询意图，实际工具参数由 Python 执行器生成。"""

    model_config = ConfigDict(frozen=True)

    action: IntakeToolAction
    target_fields: list[str] = Field(default_factory=list, max_length=20)
    entity_types: list[IntakeEntityType] = Field(default_factory=list, max_length=2)
    person_mentions: list[str] = Field(default_factory=list, max_length=20)
    organization_mentions: list[str] = Field(default_factory=list, max_length=20)
    relationship_hints: list[str] = Field(default_factory=list, max_length=20)
    result_limit: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def require_query_target(self):
        if not (
            self.target_fields
            or self.person_mentions
            or self.organization_mentions
            or self.relationship_hints
        ):
            raise ValueError("QueryPlan 至少需要一个查询目标")
        return self


class ToolObservation(BaseModel):
    """统一表示工具技术状态和信息补充状态。"""

    model_config = ConfigDict(frozen=True)

    action: IntakeToolAction
    target_fields: list[str] = Field(default_factory=list, max_length=20)
    executed_query: str = Field(min_length=1, max_length=500)
    technical_status: TechnicalStatus
    information_status: InformationStatus
    resolutions: list[IntakeEntityResolution] = Field(default_factory=list, max_length=40)
    confirmation: ConfirmationRequest | None = None
    summary: str = Field(default="", max_length=4_000)
    error: str | None = Field(default=None, max_length=1_000)

    @model_validator(mode="after")
    def validate_statuses(self):
        if self.technical_status == "FAILED":
            if self.information_status != "NO_RESULT":
                raise ValueError("工具技术失败时信息状态必须为 NO_RESULT")
            if not self.error:
                raise ValueError("工具技术失败时必须记录错误信息")
        elif self.error:
            raise ValueError("工具技术成功时不能携带错误信息")
        return self


class AgentTurn(BaseModel):
    """表示模型在一次决策中输出的受控动作。"""

    model_config = ConfigDict(frozen=True)

    context_patch: IntakeContextPatch = Field(default_factory=IntakeContextPatch)
    skill: IntakeSkillName
    next_action: IntakeAction
    query_plan: QueryPlan | None = None
    user_message: str | None = Field(default=None, min_length=1, max_length=1_000)
    reason: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_action_payload(self):
        if self.next_action in ("SEARCH_INTERNAL", "SEARCH_PUBLIC"):
            if self.query_plan is None:
                raise ValueError("查询动作必须提供 QueryPlan")
            if self.query_plan.action != self.next_action:
                raise ValueError("QueryPlan 动作必须与 next_action 一致")
            if self.user_message is not None:
                raise ValueError("查询动作不能同时向用户提问")
        elif self.next_action == "ASK_USER":
            if self.user_message is None:
                raise ValueError("ASK_USER 必须提供用户问题")
            if self.query_plan is not None:
                raise ValueError("ASK_USER 不能携带 QueryPlan")
        elif self.query_plan is not None or self.user_message is not None:
            raise ValueError("READY 不能携带 QueryPlan 或用户问题")
        return self


class AgentState(BaseModel):
    """Intake Agent 的请求内运行态，业务事实只保存在 context 中。"""

    model_config = ConfigDict(frozen=True)

    context: IntakeStructuredContext
    latest_observation: ToolObservation | None = None
    loop_count: int = Field(default=0, ge=0)
    llm_turn_count: int = Field(default=0, ge=0)


AgentLoopStopReason = Literal[
    "READY",
    "WAITING_USER",
    "MAX_LOOPS",
    "MAX_TOOL_CALLS",
    "REPEATED_ACTION",
]
AgentCheckpoint = Callable[
    [IntakeStructuredContext, ConfirmationRequest | None], None
]


class AgentDecisionProvider(Protocol):
    def decide(self, state: AgentState) -> AgentTurn: ...


class QueryExecutor(Protocol):
    def controlled_query(
        self,
        plan: QueryPlan,
        context: IntakeStructuredContext,
    ) -> str: ...

    def execute(
        self,
        plan: QueryPlan,
        context: IntakeStructuredContext,
        *,
        version: int,
        source_text: str | None = None,
        confirmation: ConfirmationRequest | None = None,
        external_normalizer=None,
    ) -> ToolObservation: ...


@dataclass(frozen=True)
class MechanicalAgentLoopResult:
    state: AgentState
    confirmation: ConfirmationRequest | None
    stop_reason: AgentLoopStopReason
    tool_calls: int


class MechanicalIntakeAgentLoop:
    """执行模型决策、受控工具调用和状态归并，不承载业务判断。"""

    def __init__(
        self,
        decision_provider: AgentDecisionProvider,
        query_executor: QueryExecutor,
        reducer,
        *,
        max_loops: int = 8,
        max_tool_calls: int = 4,
        max_repeated_actions: int = 2,
    ):
        if max_loops < 1:
            raise ValueError("max_loops 必须大于等于 1")
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls 必须大于等于 1")
        self.decision_provider = decision_provider
        self.query_executor = query_executor
        self.reducer = reducer
        self.max_loops = max_loops
        self.max_tool_calls = max_tool_calls
        self.max_repeated_actions = max_repeated_actions

    def run(
        self,
        initial_context: IntakeStructuredContext,
        *,
        version: int,
        source_text: str | None,
        hard_gate: Callable[[IntakeStructuredContext], bool],
        confirmation: ConfirmationRequest | None = None,
        external_normalizer=None,
        checkpoint: AgentCheckpoint | None = None,
    ) -> MechanicalAgentLoopResult:
        state = AgentState(context=initial_context.model_copy(deep=True))
        tool_calls = 0
        guard = RepeatedActionGuard(self.max_repeated_actions)
        turn_queries: set[tuple[str, str]] = set()
        self._checkpoint(checkpoint, state.context, confirmation)

        for _ in range(self.max_loops):
            try:
                proposed_turn = self.decision_provider.decide(state)
            except Exception:
                state = self.reducer.preserve_after_llm_failure(state)
                state = self._waiting_state(
                    state,
                    "本轮身份判断暂时失败，请补充或确认目标人物和企业。",
                )
                return self._finish(
                    state,
                    confirmation,
                    "WAITING_USER",
                    tool_calls,
                    checkpoint,
                )

            turn = self._apply_hard_constraints(
                proposed_turn,
                state,
                confirmation,
                hard_gate,
            )
            state = self.reducer.apply_turn(state, turn)
            action_fingerprint = self._action_fingerprint(turn, state.context)
            if guard.observe(action_fingerprint):
                state = self._waiting_state(
                    state,
                    "连续重复同一动作，已停止自动处理，请确认身份信息。",
                )
                return self._finish(
                    state,
                    confirmation,
                    "REPEATED_ACTION",
                    tool_calls,
                    checkpoint,
                )

            if turn.next_action == "READY":
                if hard_gate(state.context):
                    return self._finish(
                        state,
                        confirmation,
                        "READY",
                        tool_calls,
                        checkpoint,
                    )
                state = self._waiting_state(state)
                return self._finish(
                    state,
                    confirmation,
                    "WAITING_USER",
                    tool_calls,
                    checkpoint,
                )

            if turn.next_action == "ASK_USER":
                return self._finish(
                    state,
                    confirmation,
                    "WAITING_USER",
                    tool_calls,
                    checkpoint,
                )

            if tool_calls >= self.max_tool_calls:
                state = self._waiting_state(
                    state,
                    "已达到本轮工具调用上限，请补充或确认身份信息。",
                )
                return self._finish(
                    state,
                    confirmation,
                    "MAX_TOOL_CALLS",
                    tool_calls,
                    checkpoint,
                )

            query_key = (
                turn.next_action,
                self.query_executor.controlled_query(turn.query_plan, state.context),
            )
            if query_key in turn_queries:
                state = self._waiting_state(
                    state,
                    "相同身份查询已经执行，请补充或确认身份信息。",
                )
                return self._finish(
                    state,
                    confirmation,
                    "REPEATED_ACTION",
                    tool_calls,
                    checkpoint,
                )
            turn_queries.add(query_key)

            observation = self.query_executor.execute(
                turn.query_plan,
                state.context,
                version=version,
                source_text=source_text,
                confirmation=confirmation,
                external_normalizer=external_normalizer,
            )
            tool_calls += 1
            state = self.reducer.apply_observation(state, observation)
            if observation.confirmation is not None:
                confirmation = observation.confirmation
            self._checkpoint(checkpoint, state.context, confirmation)

        state = self._waiting_state(
            state,
            "已达到本轮身份确认循环上限，请补充或确认身份信息。",
        )
        return self._finish(
            state,
            confirmation,
            "MAX_LOOPS",
            tool_calls,
            checkpoint,
        )

    def _apply_hard_constraints(
        self,
        turn: AgentTurn,
        state: AgentState,
        confirmation: ConfirmationRequest | None,
        hard_gate: Callable[[IntakeStructuredContext], bool],
    ) -> AgentTurn:
        attempts = {item.action for item in state.context.tool_attempts}
        if turn.next_action == "ASK_USER" and hard_gate(state.context):
            return self._ready_turn("身份硬校验已满足")
        if turn.next_action == "READY" and not hard_gate(state.context):
            if "SEARCH_INTERNAL" not in attempts:
                return self._search_turn("SEARCH_INTERNAL", state.context)
            if (
                confirmation is not None
                and any(len(item.candidates) != 1 for item in confirmation.items)
                and "SEARCH_PUBLIC" not in attempts
            ):
                return self._search_turn("SEARCH_PUBLIC", state.context)
            return self._ask_turn(state.context)
        if turn.next_action == "SEARCH_PUBLIC":
            if "SEARCH_INTERNAL" not in attempts:
                return self._search_turn("SEARCH_INTERNAL", state.context)
            if confirmation is None:
                return self._ask_turn(state.context)
        return turn

    def _action_fingerprint(
        self,
        turn: AgentTurn,
        context: IntakeStructuredContext,
    ) -> tuple[str, str]:
        if turn.query_plan is None:
            return turn.next_action, turn.user_message or ""
        return turn.next_action, self.query_executor.controlled_query(
            turn.query_plan,
            context,
        )

    @staticmethod
    def _search_turn(
        action: IntakeToolAction,
        context: IntakeStructuredContext,
    ) -> AgentTurn:
        return AgentTurn(
            skill="internal_lookup" if action == "SEARCH_INTERNAL" else "public_lookup",
            next_action=action,
            query_plan=QueryPlan(
                action=action,
                target_fields=context.target_fields or ["identity"],
                entity_types=[
                    *(("PERSON",) if context.people else ()),
                    *(("ORGANIZATION",) if context.organizations else ()),
                ],
                person_mentions=context.people,
                organization_mentions=context.organizations,
            ),
            reason="Python 硬约束要求先完成身份查询",
        )

    @staticmethod
    def _ask_turn(context: IntakeStructuredContext) -> AgentTurn:
        return AgentTurn(
            skill="identity_resolution",
            next_action="ASK_USER",
            user_message=context.user_question
            or "请确认目标人物或企业的完整名称、职位或所属关系。",
            reason="身份硬校验尚未满足，需要用户确认",
        )

    @staticmethod
    def _ready_turn(reason: str) -> AgentTurn:
        return AgentTurn(
            skill="intake_readiness",
            next_action="READY",
            reason=reason,
        )

    @staticmethod
    def _waiting_state(
        state: AgentState,
        question: str | None = None,
    ) -> AgentState:
        context_data = state.context.model_dump(mode="python")
        context_data.update(
            {
                "next_action": "ASK_USER",
                "user_question": question
                or state.context.user_question
                or "请确认目标人物或企业的完整名称、职位或所属关系。",
            }
        )
        return AgentState(
            context=IntakeStructuredContext.model_validate(context_data),
            latest_observation=state.latest_observation,
            loop_count=state.loop_count,
            llm_turn_count=state.llm_turn_count,
        )

    @staticmethod
    def _checkpoint(
        checkpoint: AgentCheckpoint | None,
        context: IntakeStructuredContext,
        confirmation: ConfirmationRequest | None,
    ) -> None:
        if checkpoint is not None:
            checkpoint(context.model_copy(deep=True), confirmation)

    @staticmethod
    def _finish(
        state: AgentState,
        confirmation: ConfirmationRequest | None,
        stop_reason: AgentLoopStopReason,
        tool_calls: int,
        checkpoint: AgentCheckpoint | None,
    ) -> MechanicalAgentLoopResult:
        MechanicalIntakeAgentLoop._checkpoint(
            checkpoint,
            state.context,
            confirmation,
        )
        return MechanicalAgentLoopResult(
            state=state,
            confirmation=confirmation,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
        )
