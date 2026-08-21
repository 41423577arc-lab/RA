from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from app.schemas.intake import (
    IntakeChatRequest,
    IntakeContextNextAction,
    IntakeResolutionResult,
    IntakeStructuredContext,
    IntakeToolAttempt,
)
from app.schemas.task import ConfirmationRequest
from app.services.infrastructure.loop_controls import RepeatedActionGuard


IntakeLoopStopReason = Literal[
    "READY",
    "WAITING_USER",
    "MAX_LOOPS",
    "MAX_TOOL_CALLS",
    "REPEATED_ACTION",
]
InternalLookup = Callable[
    [IntakeStructuredContext], tuple[list[dict], ConfirmationRequest | None]
]
PublicLookup = Callable[
    [IntakeStructuredContext, ConfirmationRequest],
    tuple[list[dict], ConfirmationRequest | None],
]
ContextCheckpoint = Callable[
    [IntakeStructuredContext, ConfirmationRequest | None], None
]


class IntakeContextAgent(Protocol):
    def initialize_context(
        self,
        request: IntakeChatRequest,
        extracted_context: IntakeStructuredContext,
    ) -> IntakeStructuredContext: ...

    def update_context(
        self,
        request: IntakeChatRequest,
        old_context: IntakeStructuredContext,
        *,
        extracted_context: IntakeStructuredContext | None = None,
        tool_observation: dict | None = None,
    ) -> IntakeStructuredContext: ...


@dataclass(frozen=True)
class IntakeIdentityLoopResult:
    context: IntakeStructuredContext
    resolutions: tuple[dict, ...]
    confirmation: ConfirmationRequest | None
    stop_reason: IntakeLoopStopReason
    tool_calls: int


class IntakeActionValidator:
    allowed_actions = {"SEARCH_INTERNAL", "SEARCH_PUBLIC", "ASK_USER", "READY"}

    def validate(self, requested_action: object) -> IntakeContextNextAction:
        if isinstance(requested_action, str) and requested_action in self.allowed_actions:
            return cast(IntakeContextNextAction, requested_action)
        return "ASK_USER"


class IntakeIdentityLoop:
    def __init__(
        self,
        agent: IntakeContextAgent,
        event_recorder,
        scope_id: str,
        *,
        max_loops: int = 8,
        max_tool_calls: int = 4,
        max_repeated_actions: int = 2,
    ):
        if max_loops < 1:
            raise ValueError("max_loops must be at least 1")
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        self.agent = agent
        self.event_recorder = event_recorder
        self.scope_id = scope_id
        self.max_loops = max_loops
        self.max_tool_calls = max_tool_calls
        self.max_repeated_actions = max_repeated_actions
        self.validator = IntakeActionValidator()

    def run(
        self,
        request: IntakeChatRequest,
        extracted_context: IntakeStructuredContext,
        previous_context: IntakeStructuredContext | None,
        previous_confirmation: ConfirmationRequest | None = None,
        *,
        lookup_internal: InternalLookup,
        lookup_public: PublicLookup,
        hard_gate: Callable[[IntakeStructuredContext], bool],
        checkpoint: ContextCheckpoint | None = None,
    ) -> IntakeIdentityLoopResult:
        context = self._initial_context(request, extracted_context, previous_context)
        resolutions = list(context.entity_resolutions)
        confirmation = previous_confirmation
        tool_calls = 0
        guard = RepeatedActionGuard(self.max_repeated_actions)

        self._checkpoint(checkpoint, context, confirmation)
        for iteration in range(1, self.max_loops + 1):
            requested_action = context.next_action
            action = self.validator.validate(requested_action)
            action = self._apply_hard_constraints(
                action, context, confirmation, hard_gate
            )
            self._record_action(iteration, requested_action, action)

            if guard.observe(action):
                context = self._waiting_context(
                    context, "连续重复同一动作，已停止自动查询。"
                )
                return self._finish(
                    context,
                    resolutions,
                    confirmation,
                    "REPEATED_ACTION",
                    tool_calls,
                    checkpoint,
                )

            if action == "READY":
                return self._finish(
                    context,
                    resolutions,
                    confirmation,
                    "READY",
                    tool_calls,
                    checkpoint,
                )
            if action == "ASK_USER":
                context = self._waiting_context(context)
                return self._finish(
                    context,
                    resolutions,
                    confirmation,
                    "WAITING_USER",
                    tool_calls,
                    checkpoint,
                )
            if tool_calls >= self.max_tool_calls:
                context = self._waiting_context(
                    context, "已达到本轮工具调用上限，请补充或确认身份信息。"
                )
                return self._finish(
                    context,
                    resolutions,
                    confirmation,
                    "MAX_TOOL_CALLS",
                    tool_calls,
                    checkpoint,
                )

            tool_calls += 1
            previous_criteria = list(context.success_criteria)
            query = self._controlled_query(context)
            technical_status: Literal["SUCCESS", "FAILED"] = "SUCCESS"
            error: str | None = None
            additions: list[dict] = []
            try:
                if action == "SEARCH_INTERNAL":
                    additions, confirmation = lookup_internal(context)
                else:
                    if confirmation is None:
                        raise ValueError("公网身份查询缺少内部候选 Observation")
                    additions, confirmation = lookup_public(context, confirmation)
            except Exception as exc:
                technical_status = "FAILED"
                error = f"{type(exc).__name__}: {exc}"[:1_000]

            resolutions = self._merge_resolutions(resolutions, additions)
            information_status = (
                "NO_RESULT"
                if technical_status == "FAILED"
                else self._information_status(additions, confirmation)
            )
            attempt = IntakeToolAttempt(
                action=action,
                target_fields=context.target_fields or self._identity_fields(context),
                query=query,
                technical_status=technical_status,
                information_status=information_status,
                observation=error or self._observation_summary(additions, confirmation),
            )
            controlled_attempts = [*context.tool_attempts, attempt]
            context = self._preserve_controlled_state(
                context,
                resolutions=resolutions,
                tool_attempts=controlled_attempts,
            )
            observation = {
                "action": action,
                "target_fields": attempt.target_fields,
                "query": query,
                "technical_status": technical_status,
                "information_status": information_status,
                "resolutions": additions,
                "confirmation": confirmation.model_dump(mode="json")
                if confirmation
                else None,
                "error": error,
                "success_criteria": previous_criteria,
            }
            self._record_observation(iteration, observation)
            controlled_context = context
            context = self._update_after_observation(request, context, observation)
            context = self._preserve_controlled_state(
                context,
                controlled_context=controlled_context,
                resolutions=resolutions,
                tool_attempts=controlled_attempts,
            )
            self._checkpoint(checkpoint, context, confirmation)

        context = self._waiting_context(
            context, "已达到本轮身份确认循环上限，请补充或确认身份信息。"
        )
        return self._finish(
            context,
            resolutions,
            confirmation,
            "MAX_LOOPS",
            tool_calls,
            checkpoint,
        )

    def _initial_context(
        self,
        request: IntakeChatRequest,
        extracted_context: IntakeStructuredContext,
        previous_context: IntakeStructuredContext | None,
    ) -> IntakeStructuredContext:
        if previous_context is None or not self._has_loop_state(previous_context):
            try:
                context = self.agent.initialize_context(request, extracted_context)
            except Exception:
                context = extracted_context.model_copy(
                    update={
                        "target_fields": self._identity_fields(extracted_context),
                        "next_action": "SEARCH_INTERNAL",
                        "success_criteria": ["获得可验证的标准身份候选"],
                    }
                )
            return self._preserve_controlled_state(
                context,
                controlled_context=extracted_context,
                resolutions=list(extracted_context.entity_resolutions),
                tool_attempts=list(extracted_context.tool_attempts),
            )

        try:
            context = self.agent.update_context(
                request,
                previous_context,
                extracted_context=extracted_context,
            )
        except Exception:
            context = previous_context.model_copy(
                update={"next_action": "SEARCH_INTERNAL"}
            )
        return self._preserve_controlled_state(
            context,
            controlled_context=previous_context,
            resolutions=list(previous_context.entity_resolutions),
            tool_attempts=list(previous_context.tool_attempts),
        )

    def _update_after_observation(
        self,
        request: IntakeChatRequest,
        context: IntakeStructuredContext,
        observation: dict,
    ) -> IntakeStructuredContext:
        try:
            return self.agent.update_context(
                request,
                context,
                tool_observation=observation,
            )
        except Exception:
            if observation["information_status"] == "RESOLVED":
                action: IntakeContextNextAction = "READY"
            elif observation["action"] == "SEARCH_INTERNAL":
                action = "SEARCH_PUBLIC"
            else:
                action = "ASK_USER"
            return context.model_copy(update={"next_action": action})

    @staticmethod
    def _preserve_controlled_state(
        context: IntakeStructuredContext,
        *,
        controlled_context: IntakeStructuredContext | None = None,
        resolutions: list[dict],
        tool_attempts: list[IntakeToolAttempt],
    ) -> IntakeStructuredContext:
        data = context.model_dump(mode="json")
        data["entity_resolutions"] = resolutions
        data["tool_attempts"] = [item.model_dump(mode="json") for item in tool_attempts]
        if controlled_context is not None:
            data["final_confirmation"] = (
                controlled_context.final_confirmation.model_dump(mode="json")
                if controlled_context.final_confirmation
                else None
            )
        return IntakeStructuredContext.model_validate(data)

    @staticmethod
    def _has_loop_state(context: IntakeStructuredContext) -> bool:
        return bool(
            context.next_action
            or context.target_fields
            or context.success_criteria
            or context.tool_attempts
        )

    @staticmethod
    def _apply_hard_constraints(
        action: IntakeContextNextAction,
        context: IntakeStructuredContext,
        confirmation: ConfirmationRequest | None,
        hard_gate: Callable[[IntakeStructuredContext], bool],
    ) -> IntakeContextNextAction:
        current_query = IntakeIdentityLoop._controlled_query(context)
        attempts = {
            item.action
            for item in context.tool_attempts
            if item.query == current_query
        }
        ineffective_attempts = {
            item.action
            for item in context.tool_attempts
            if item.query == current_query
            and item.technical_status == "SUCCESS"
            and item.information_status == "NO_RESULT"
        }
        if action == "READY" and not hard_gate(context):
            if "SEARCH_INTERNAL" not in attempts:
                return "SEARCH_INTERNAL"
            if confirmation is not None and "SEARCH_PUBLIC" not in attempts:
                return "SEARCH_PUBLIC"
            return "ASK_USER"
        if action == "SEARCH_PUBLIC" and "SEARCH_INTERNAL" not in attempts:
            return "SEARCH_INTERNAL"
        if action == "SEARCH_PUBLIC" and confirmation is None:
            return "READY" if hard_gate(context) else "ASK_USER"
        if action in ineffective_attempts:
            if (
                action == "SEARCH_INTERNAL"
                and confirmation is not None
                and "SEARCH_PUBLIC" not in ineffective_attempts
            ):
                return "SEARCH_PUBLIC"
            return "ASK_USER"
        return action

    @staticmethod
    def _information_status(
        additions: list[dict], confirmation: ConfirmationRequest | None
    ) -> Literal["RESOLVED", "PARTIAL", "NO_RESULT"]:
        if additions and confirmation is None:
            return "RESOLVED"
        candidate_count = sum(
            len(item.candidates) for item in confirmation.items
        ) if confirmation else 0
        if additions or candidate_count:
            return "PARTIAL"
        return "NO_RESULT"

    @staticmethod
    def _identity_fields(context: IntakeStructuredContext) -> list[str]:
        fields = []
        if context.people:
            fields.append("people")
        if context.organizations:
            fields.append("organizations")
        return fields or ["people", "organizations"]

    @staticmethod
    def _controlled_query(context: IntakeStructuredContext) -> str:
        parts = [
            *(f"人物:{value}" for value in context.people[:3]),
            *(f"企业:{value}" for value in context.organizations[:3]),
        ]
        return "；".join(parts) or "目标人物或企业身份"

    @staticmethod
    def _observation_summary(
        additions: list[dict], confirmation: ConfirmationRequest | None
    ) -> str:
        pending = len(confirmation.items) if confirmation else 0
        candidates = sum(
            len(item.candidates) for item in confirmation.items
        ) if confirmation else 0
        return f"新增确认身份 {len(additions)} 个，待确认 {pending} 项，候选 {candidates} 个。"

    @staticmethod
    def _merge_resolutions(existing: list, additions: list[dict]) -> list[dict]:
        merged = {}
        for item in existing:
            value = item if isinstance(item, dict) else item.model_dump(mode="json")
            merged[(value.get("entity_type"), value.get("mention"))] = value
        for item in additions:
            merged[(item.get("entity_type"), item.get("mention"))] = item
        return list(merged.values())

    @staticmethod
    def _waiting_context(
        context: IntakeStructuredContext, fallback_question: str | None = None
    ) -> IntakeStructuredContext:
        question = context.user_question or fallback_question or (
            "请确认目标人物或企业的完整名称、职位或所属关系。"
        )
        return context.model_copy(
            update={"next_action": "ASK_USER", "user_question": question}
        )

    def _finish(
        self,
        context: IntakeStructuredContext,
        resolutions: list[dict],
        confirmation: ConfirmationRequest | None,
        reason: IntakeLoopStopReason,
        tool_calls: int,
        checkpoint: ContextCheckpoint | None,
    ) -> IntakeIdentityLoopResult:
        self._checkpoint(checkpoint, context, confirmation)
        self._log(
            event_type="AGENT_LOOP_STOP",
            node_name="intake_identity_loop",
            status=reason,
            title="Intake 身份确认 Loop 已停止",
            detail=f"停止原因：{reason}。",
            payload={"reason": reason, "tool_calls": tool_calls},
        )
        return IntakeIdentityLoopResult(
            context=context,
            resolutions=tuple(resolutions),
            confirmation=confirmation,
            stop_reason=reason,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _checkpoint(
        checkpoint: ContextCheckpoint | None,
        context: IntakeStructuredContext,
        confirmation: ConfirmationRequest | None,
    ) -> None:
        if checkpoint is not None:
            checkpoint(context, confirmation)

    def _record_action(
        self,
        iteration: int,
        requested_action: object,
        action: IntakeContextNextAction,
    ) -> None:
        self._log(
            event_type="AGENT_ACTION",
            node_name="intake_identity_loop",
            status=action,
            title=f"Intake 身份动作：{action}",
            detail=f"第 {iteration} 轮身份确认动作。",
            payload={
                "iteration": iteration,
                "requested_action": requested_action,
                "action": action,
            },
        )

    def _record_observation(self, iteration: int, observation: dict) -> None:
        self._log(
            event_type="AGENT_OBSERVATION",
            node_name="intake_identity_loop",
            status=observation["technical_status"],
            title=f"身份工具观察：{observation['action']}",
            detail=(
                f"技术状态 {observation['technical_status']}，"
                f"信息状态 {observation['information_status']}。"
            ),
            payload={"iteration": iteration, "observation": observation},
        )

    def _log(self, **values) -> None:
        logger = getattr(self.event_recorder, "log_execution_event", None)
        if logger is not None:
            logger(self.scope_id, **values)
