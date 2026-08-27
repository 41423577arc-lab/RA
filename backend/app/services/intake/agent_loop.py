from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.intake import (
    IntakeAction,
    IntakeEntityResolution,
    IntakeEntityAssessment,
    IntakeEntityType,
    IntakeFieldState,
    IntakePersonCandidate,
    IntakeResolutionResult,
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

class IntakeContextPatch(BaseModel):
    """表示单次模型推理建议更新的业务字段。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    people: list[str] = Field(default_factory=list, max_length=20)
    people_details: list[IntakePersonCandidate] = Field(
        default_factory=list,
        max_length=20,
    )
    organizations: list[str] = Field(default_factory=list, max_length=20)
    projects: list[str] = Field(default_factory=list, max_length=20)
    business_directions: list[str] = Field(default_factory=list, max_length=20)
    focus_questions: list[str] = Field(default_factory=list, max_length=20)
    event_type: Literal["宴请", "拜访", "会议", "其他"] | None = None
    event_time: str | None = None
    event_location: str | None = None
    entity_assessments: list[IntakeEntityAssessment] = Field(
        default_factory=list,
        max_length=40,
    )
    field_states: dict[str, IntakeFieldState] = Field(default_factory=dict)
    target_fields: list[str] = Field(default_factory=list, max_length=20)
    success_criteria: list[str] = Field(default_factory=list, max_length=20)
    resolution_result: IntakeResolutionResult | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_updates_wrapper(cls, value):
        if isinstance(value, dict) and set(value) == {"updates"}:
            updates = value["updates"]
            if isinstance(updates, dict):
                return updates
        return value

    @property
    def updates(self) -> dict:
        return self.model_dump(mode="python", exclude_unset=True)


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
        on_observation: Callable[[ToolObservation], None] | None = None,
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
        self.on_observation = on_observation

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
                if confirmation is None and hard_gate(state.context):
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

            query_context = state.context
            observation = self.query_executor.execute(
                turn.query_plan,
                query_context,
                version=version,
                source_text=source_text,
                confirmation=confirmation,
                external_normalizer=external_normalizer,
            )
            if self.on_observation is not None:
                self.on_observation(observation)
            tool_calls += 1
            state = self.reducer.apply_observation(state, observation)
            if observation.confirmation is not None:
                confirmation = observation.confirmation
            if confirmation is not None and observation.resolutions:
                confirmation = self._reconcile_confirmation(
                    confirmation,
                    observation,
                    query_context,
                    state.context,
                )
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

    @staticmethod
    def _reconcile_confirmation(
        confirmation: ConfirmationRequest,
        observation: ToolObservation,
        query_context: IntakeStructuredContext,
        updated_context: IntakeStructuredContext,
    ) -> ConfirmationRequest | None:
        active_mentions = {
            *updated_context.people,
            *updated_context.organizations,
        }
        resolved_mentions = {
            (resolution.entity_type, resolution.mention)
            for resolution in observation.resolutions
        } | {
            (resolution.entity_type, resolution.canonical_name)
            for resolution in observation.resolutions
        }
        linked_organization_mentions: set[str] = set()
        if len(query_context.people) == 1 and len(query_context.organizations) == 1:
            queried_person = query_context.people[0]
            if any(
                resolution.entity_type == "PERSON"
                and queried_person in {resolution.mention, resolution.canonical_name}
                and resolution.organization
                for resolution in observation.resolutions
            ):
                linked_organization_mentions.add(query_context.organizations[0])

        remaining_items = [
            item
            for item in confirmation.items
            if item.mention in active_mentions
            and (item.entity_type, item.mention) not in resolved_mentions
            and not (
                item.entity_type == "ORGANIZATION"
                and item.mention in linked_organization_mentions
            )
        ]
        if not remaining_items:
            return None
        return ConfirmationRequest(
            version=confirmation.version,
            items=remaining_items,
        )

    def _apply_hard_constraints(
        self,
        turn: AgentTurn,
        state: AgentState,
        confirmation: ConfirmationRequest | None,
        hard_gate: Callable[[IntakeStructuredContext], bool],
    ) -> AgentTurn:
        attempts = {item.action for item in state.context.tool_attempts}
        gate_ready = confirmation is None and hard_gate(state.context)
        if gate_ready:
            return turn if turn.next_action == "READY" else self._ready_turn(
                "身份硬校验已满足"
            )
        if turn.next_action == "SEARCH_INTERNAL":
            turn = self._filter_nonstandard_person_mentions(turn, state.context)
            if turn.next_action == "ASK_USER":
                return turn
        if (
            turn.next_action == "ASK_USER"
            and confirmation is None
            and "SEARCH_INTERNAL" not in attempts
            and self._has_identity_mentions(state.context, turn.context_patch)
        ):
            forced_search = self._search_turn(
                "SEARCH_INTERNAL",
                state.context,
                context_patch=turn.context_patch,
            )
            filtered_turn = self._filter_nonstandard_person_mentions(
                forced_search,
                state.context,
            )
            if filtered_turn.next_action == "ASK_USER":
                return filtered_turn.model_copy(
                    update={
                        "user_message": turn.user_message,
                        "reason": turn.reason,
                    }
                )
            return filtered_turn
        if turn.next_action == "READY" and not gate_ready:
            if "SEARCH_INTERNAL" not in attempts:
                return self._search_turn(
                    "SEARCH_INTERNAL",
                    state.context,
                    context_patch=turn.context_patch,
                )
            if (
                confirmation is not None
                and any(len(item.candidates) != 1 for item in confirmation.items)
                and "SEARCH_PUBLIC" not in attempts
            ):
                return self._search_turn("SEARCH_PUBLIC", state.context)
            return self._ask_turn(state.context)
        if turn.next_action == "SEARCH_PUBLIC":
            if "SEARCH_INTERNAL" not in attempts:
                return self._search_turn(
                    "SEARCH_INTERNAL",
                    state.context,
                    context_patch=turn.context_patch,
                )
            if confirmation is None:
                return self._ask_turn(state.context)
        return turn

    @staticmethod
    def _has_identity_mentions(
        context: IntakeStructuredContext,
        context_patch: IntakeContextPatch,
    ) -> bool:
        return bool(
            context_patch.people
            or context_patch.organizations
            or context.people
            or context.organizations
        )

    @staticmethod
    def _filter_nonstandard_person_mentions(
        turn: AgentTurn,
        context: IntakeStructuredContext,
    ) -> AgentTurn:
        if turn.query_plan is None or not turn.query_plan.person_mentions:
            return turn
        title_suffixes = (
            "总",
            "董",
            "经理",
            "主任",
            "董事长",
            "负责人",
            "领导",
            "先生",
            "女士",
            "老师",
        )
        generic_mentions = {"客户", "老板", "负责人", "联系人", "领导"}
        searchable_mentions = [
            mention
            for mention in turn.query_plan.person_mentions
            if "".join(mention.split()) not in generic_mentions
            and not "".join(mention.split()).endswith(title_suffixes)
        ]
        if searchable_mentions == turn.query_plan.person_mentions:
            return turn
        if searchable_mentions:
            return turn.model_copy(
                update={
                    "query_plan": turn.query_plan.model_copy(
                        update={"person_mentions": searchable_mentions}
                    )
                }
            )
        return AgentTurn(
            context_patch=turn.context_patch,
            skill="identity_resolution",
            next_action="ASK_USER",
            user_message=(
                context.user_question
                or "请补充目标人物的完整姓名、所在企业或具体职位。"
            ),
            reason="非标准人物称谓不能用于内部身份查询",
        )

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
        *,
        context_patch: IntakeContextPatch | None = None,
    ) -> AgentTurn:
        patch = context_patch or IntakeContextPatch()
        people = patch.people or context.people
        organizations = patch.organizations or context.organizations
        return AgentTurn(
            context_patch=patch,
            skill="internal_lookup" if action == "SEARCH_INTERNAL" else "public_lookup",
            next_action=action,
            query_plan=QueryPlan(
                action=action,
                target_fields=patch.target_fields or context.target_fields or ["identity"],
                entity_types=[
                    *(("PERSON",) if people else ()),
                    *(("ORGANIZATION",) if organizations else ()),
                ],
                person_mentions=people,
                organization_mentions=organizations,
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
