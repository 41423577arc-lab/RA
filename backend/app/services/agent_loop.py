from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from app.schemas.task import (
    AgentAction,
    AgentContext,
    AgentPhase,
    ConfirmedContext,
    Observation,
    ProjectResult,
    PublicClaim,
    SearchResult,
    TaskChatMessage,
    WebPage,
    WebVerification,
)
from app.services.agent_context import AgentContextBuilder

if TYPE_CHECKING:
    from app.models.database import ResearchTask


LoopStopReason = Literal[
    "DONE",
    "WAITING_USER",
    "MAX_LOOPS",
    "MAX_TOOL_CALLS",
    "REPEATED_ACTION",
]
AgentTurn = Callable[[str, AgentContext], object]


TRANSITIONS: dict[tuple[AgentPhase, AgentAction], AgentPhase] = {
    ("IDENTITY", "ASK_USER"): "WAITING_USER",
    ("IDENTITY", "SEARCH_PUBLIC"): "PUBLIC_RESEARCH",
    ("WAITING_USER", "ASK_USER"): "WAITING_USER",
    ("WAITING_USER", "SEARCH_PUBLIC"): "PUBLIC_RESEARCH",
    ("PUBLIC_RESEARCH", "SEARCH_PUBLIC"): "PROJECT_RESEARCH",
    ("PUBLIC_RESEARCH", "SEARCH_INTERNAL"): "SYNTHESIS",
    ("PUBLIC_RESEARCH", "SYNTHESIZE"): "DONE",
    ("PUBLIC_RESEARCH", "RESPOND"): "DONE",
    ("PUBLIC_RESEARCH", "FINISH"): "DONE",
    ("PROJECT_RESEARCH", "SEARCH_PUBLIC"): "PROJECT_RESEARCH",
    ("PROJECT_RESEARCH", "SEARCH_INTERNAL"): "SYNTHESIS",
    ("PROJECT_RESEARCH", "SYNTHESIZE"): "DONE",
    ("PROJECT_RESEARCH", "RESPOND"): "DONE",
    ("PROJECT_RESEARCH", "FINISH"): "DONE",
    ("SYNTHESIS", "SEARCH_PUBLIC"): "PROJECT_RESEARCH",
    ("SYNTHESIS", "SEARCH_INTERNAL"): "SYNTHESIS",
    ("SYNTHESIS", "SYNTHESIZE"): "DONE",
    ("SYNTHESIS", "RESPOND"): "DONE",
    ("SYNTHESIS", "FINISH"): "DONE",
    ("DONE", "FINISH"): "DONE",
}

FALLBACK_ACTIONS: dict[AgentPhase, AgentAction] = {
    "IDENTITY": "ASK_USER",
    "PUBLIC_RESEARCH": "SEARCH_PUBLIC",
    "PROJECT_RESEARCH": "SEARCH_INTERNAL",
    "SYNTHESIS": "SYNTHESIZE",
    "WAITING_USER": "ASK_USER",
    "DONE": "FINISH",
}


class ExecutionEventRecorder(Protocol):
    def log_execution_event(self, scope_id: str, **values: object) -> object: ...


@dataclass(frozen=True)
class ToolExecutionResult:
    observation: Observation
    project_results: tuple[ProjectResult, ...] = ()
    search_results: tuple[SearchResult, ...] = ()
    web_pages: tuple[WebPage, ...] = ()
    web_verifications: tuple[WebVerification, ...] = ()
    public_claims: tuple[PublicClaim, ...] = ()
    degraded_nodes: tuple[str, ...] = ()
    web_search_status: str | None = None
    web_fetch_status: str | None = None
    internal_search_status: str | None = None


class AgentToolRunner(Protocol):
    def execute(
        self,
        task_id: str,
        action: AgentAction,
        context: AgentContext,
    ) -> ToolExecutionResult: ...


@dataclass(frozen=True)
class ActionValidation:
    requested_action: str
    action: AgentAction
    used_fallback: bool


@dataclass(frozen=True)
class AgentLoopStep:
    iteration: int
    phase: AgentPhase
    requested_action: str
    action: AgentAction
    next_phase: AgentPhase
    used_fallback: bool
    selection_error: str | None = None


@dataclass(frozen=True)
class AgentLoopResult:
    phase: AgentPhase
    stop_reason: LoopStopReason
    steps: tuple[AgentLoopStep, ...]
    observations: tuple[Observation, ...] = ()
    project_results: tuple[ProjectResult, ...] = ()
    search_results: tuple[SearchResult, ...] = ()
    web_pages: tuple[WebPage, ...] = ()
    web_verifications: tuple[WebVerification, ...] = ()
    public_claims: tuple[PublicClaim, ...] = ()
    degraded_nodes: tuple[str, ...] = ()
    web_search_status: str = "SKIPPED"
    web_fetch_status: str = "SKIPPED"
    internal_search_status: str = "SKIPPED"
    tool_calls: int = 0


class AgentActionValidator:
    def validate(
        self,
        phase: AgentPhase,
        requested_action: object,
    ) -> ActionValidation:
        requested = (
            requested_action
            if isinstance(requested_action, str)
            else repr(requested_action)
        )
        if (phase, requested) in TRANSITIONS:
            return ActionValidation(
                requested_action=requested,
                action=cast(AgentAction, requested),
                used_fallback=False,
            )
        return ActionValidation(
            requested_action=requested[:100],
            action=FALLBACK_ACTIONS[phase],
            used_fallback=True,
        )


class RepeatedActionGuard:
    def __init__(self, max_repeated_actions: int):
        if max_repeated_actions < 1:
            raise ValueError("max_repeated_actions must be at least 1")
        self.max_repeated_actions = max_repeated_actions
        self._last_action: AgentAction | None = None
        self._count = 0

    def observe(self, action: AgentAction) -> bool:
        if action == self._last_action:
            self._count += 1
        else:
            self._last_action = action
            self._count = 1
        return self._count > self.max_repeated_actions


class AgentLoopRunner:
    def __init__(
        self,
        context_builder: AgentContextBuilder,
        event_recorder: ExecutionEventRecorder,
        agent_turn: AgentTurn,
        tool_executor: AgentToolRunner,
        *,
        max_loops: int = 8,
        max_tool_calls: int = 4,
        max_repeated_actions: int = 2,
        checkpoint: Callable[[], None] | None = None,
    ):
        if max_loops < 1:
            raise ValueError("max_loops must be at least 1")
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        self.context_builder = context_builder
        self.event_recorder = event_recorder
        self.agent_turn = agent_turn
        self.tool_executor = tool_executor
        self.max_loops = max_loops
        self.max_tool_calls = max_tool_calls
        self.max_repeated_actions = max_repeated_actions
        self.checkpoint = checkpoint
        self.validator = AgentActionValidator()

    def run(
        self,
        initial_phase: AgentPhase,
        task: "ResearchTask",
        confirmed_context: ConfirmedContext | None,
        evidence: Sequence[PublicClaim | dict[str, Any]],
        project_results: Sequence[ProjectResult | dict[str, Any]],
        recent_messages: Sequence[TaskChatMessage | dict[str, Any]],
    ) -> AgentLoopResult:
        phase = initial_phase
        steps: list[AgentLoopStep] = []
        observations: list[Observation] = []
        current_projects = [ProjectResult.model_validate(item) for item in project_results]
        current_evidence = [PublicClaim.model_validate(item) for item in evidence]
        search_results: list[SearchResult] = []
        web_pages: list[WebPage] = []
        web_verifications: list[WebVerification] = []
        degraded_nodes: list[str] = []
        web_search_status = "SKIPPED"
        web_fetch_status = "SKIPPED"
        internal_search_status = "SKIPPED"
        tool_calls = 0
        guard = RepeatedActionGuard(self.max_repeated_actions)

        if phase == "DONE":
            return self._finish(
                task.id,
                phase,
                "DONE",
                steps,
                observations,
                current_projects,
                search_results,
                web_pages,
                web_verifications,
                current_evidence,
                degraded_nodes,
                web_search_status,
                web_fetch_status,
                internal_search_status,
                tool_calls,
            )

        for iteration in range(1, self.max_loops + 1):
            if self.checkpoint is not None:
                self.checkpoint()
            self._record_phase(task.id, phase, iteration)
            context = self.context_builder.build(
                phase,
                task,
                confirmed_context,
                current_evidence,
                current_projects,
                recent_messages,
                observations,
            )
            selection_error = None
            try:
                requested_action = self.agent_turn(task.id, context)
            except Exception as exc:
                selection_error = type(exc).__name__
                requested_action = f"AGENT_TURN_ERROR:{selection_error}"
                _append_unique(degraded_nodes, "agent_turn")
            validation = self.validator.validate(phase, requested_action)

            if guard.observe(validation.action):
                _append_unique(degraded_nodes, "agent_loop")
                step = AgentLoopStep(
                    iteration=iteration,
                    phase=phase,
                    requested_action=validation.requested_action,
                    action=validation.action,
                    next_phase=phase,
                    used_fallback=validation.used_fallback,
                    selection_error=selection_error,
                )
                steps.append(step)
                self._record_action(task.id, step, repeated=True)
                return self._finish(
                    task.id,
                    phase,
                    "REPEATED_ACTION",
                    steps,
                    observations,
                    current_projects,
                    search_results,
                    web_pages,
                    web_verifications,
                    current_evidence,
                    degraded_nodes,
                    web_search_status,
                    web_fetch_status,
                    internal_search_status,
                    tool_calls,
                )

            next_phase = TRANSITIONS[(phase, validation.action)]
            tool_result = None
            if validation.action in {"SEARCH_PUBLIC", "SEARCH_INTERNAL"}:
                if tool_calls >= self.max_tool_calls:
                    step = AgentLoopStep(
                        iteration=iteration,
                        phase=phase,
                        requested_action=validation.requested_action,
                        action=validation.action,
                        next_phase=phase,
                        used_fallback=validation.used_fallback,
                        selection_error=selection_error,
                    )
                    steps.append(step)
                    self._record_action(task.id, step, blocked_reason="MAX_TOOL_CALLS")
                    _append_unique(degraded_nodes, "agent_loop")
                    return self._finish(
                        task.id,
                        phase,
                        "MAX_TOOL_CALLS",
                        steps,
                        observations,
                        current_projects,
                        search_results,
                        web_pages,
                        web_verifications,
                        current_evidence,
                        degraded_nodes,
                        web_search_status,
                        web_fetch_status,
                        internal_search_status,
                        tool_calls,
                    )

            step = AgentLoopStep(
                iteration=iteration,
                phase=phase,
                requested_action=validation.requested_action,
                action=validation.action,
                next_phase=next_phase,
                used_fallback=validation.used_fallback,
                selection_error=selection_error,
            )
            steps.append(step)
            self._record_action(task.id, step)

            if validation.action in {"SEARCH_PUBLIC", "SEARCH_INTERNAL"}:
                tool_calls += 1
                tool_result = self.tool_executor.execute(
                    task.id,
                    validation.action,
                    context,
                )
                observations.append(tool_result.observation)
                current_projects = self._merge_projects(
                    current_projects,
                    tool_result.project_results,
                )
                search_results = self._merge_by_key(
                    search_results,
                    tool_result.search_results,
                    lambda item: item.web_result_id or item.url,
                )
                web_pages = self._merge_by_key(
                    web_pages,
                    tool_result.web_pages,
                    lambda item: item.web_result_id or item.url,
                )
                web_verifications = self._merge_by_key(
                    web_verifications,
                    tool_result.web_verifications,
                    lambda item: item.web_result_id,
                )
                current_evidence = self._merge_by_key(
                    current_evidence,
                    tool_result.public_claims,
                    lambda item: (
                        item.web_result_id,
                        item.evidence_id,
                        item.source_url,
                    ),
                )
                for node_name in tool_result.degraded_nodes:
                    _append_unique(degraded_nodes, node_name)
                web_search_status = (
                    tool_result.web_search_status or web_search_status
                )
                web_fetch_status = tool_result.web_fetch_status or web_fetch_status
                internal_search_status = (
                    tool_result.internal_search_status or internal_search_status
                )
                if self.checkpoint is not None:
                    self.checkpoint()
            if tool_result is not None:
                self._record_observation(task.id, iteration, tool_result.observation)
            phase = next_phase

            if phase == "WAITING_USER":
                return self._finish(
                    task.id,
                    phase,
                    "WAITING_USER",
                    steps,
                    observations,
                    current_projects,
                    search_results,
                    web_pages,
                    web_verifications,
                    current_evidence,
                    degraded_nodes,
                    web_search_status,
                    web_fetch_status,
                    internal_search_status,
                    tool_calls,
                )
            if phase == "DONE":
                return self._finish(
                    task.id,
                    phase,
                    "DONE",
                    steps,
                    observations,
                    current_projects,
                    search_results,
                    web_pages,
                    web_verifications,
                    current_evidence,
                    degraded_nodes,
                    web_search_status,
                    web_fetch_status,
                    internal_search_status,
                    tool_calls,
                )

        _append_unique(degraded_nodes, "agent_loop")
        return self._finish(
            task.id,
            phase,
            "MAX_LOOPS",
            steps,
            observations,
            current_projects,
            search_results,
            web_pages,
            web_verifications,
            current_evidence,
            degraded_nodes,
            web_search_status,
            web_fetch_status,
            internal_search_status,
            tool_calls,
        )

    def _record_phase(self, task_id: str, phase: AgentPhase, iteration: int) -> None:
        self._log_event(
            task_id,
            event_type="AGENT_PHASE",
            node_name="agent_loop",
            status=phase,
            title=f"Agent Loop 阶段：{phase}",
            detail=f"开始第 {iteration} 轮，仅构建上下文并选择骨架动作。",
            payload={"iteration": iteration, "phase": phase},
        )

    def _record_action(
        self,
        task_id: str,
        step: AgentLoopStep,
        *,
        repeated: bool = False,
        blocked_reason: str | None = None,
    ) -> None:
        self._log_event(
            task_id,
            event_type="AGENT_ACTION",
            node_name="agent_loop",
            status="BLOCKED" if repeated or blocked_reason else step.action,
            title=f"Agent Loop 动作：{step.action}",
            detail=(
                "检测到连续重复动作，阻止阶段转换。"
                if repeated
                else "达到工具调用次数上限，阻止继续调用工具。"
                if blocked_reason
                else f"{step.phase} -> {step.next_phase}"
            ),
            payload={
                "iteration": step.iteration,
                "phase": step.phase,
                "requested_action": step.requested_action,
                "action": step.action,
                "next_phase": step.next_phase,
                "used_fallback": step.used_fallback,
                "selection_error": step.selection_error,
                "repeated": repeated,
                "blocked_reason": blocked_reason,
            },
        )

    def _record_observation(
        self,
        task_id: str,
        iteration: int,
        observation: Observation,
    ) -> None:
        self._log_event(
            task_id,
            event_type="AGENT_OBSERVATION",
            node_name="agent_loop",
            status=observation.status,
            title=f"工具观察：{observation.action}",
            detail=observation.summary,
            payload={
                "iteration": iteration,
                "plan_source": "DETERMINISTIC_RULE",
                "observation": observation.model_dump(mode="json"),
            },
        )

    @staticmethod
    def _merge_projects(
        existing: Sequence[ProjectResult],
        incoming: Sequence[ProjectResult],
    ) -> list[ProjectResult]:
        merged = {item.project_id: item for item in existing}
        for item in incoming:
            merged.setdefault(item.project_id, item)
        return list(merged.values())

    @staticmethod
    def _merge_by_key(existing, incoming, key):
        merged = {key(item): item for item in existing}
        for item in incoming:
            merged.setdefault(key(item), item)
        return list(merged.values())

    def _finish(
        self,
        task_id: str,
        phase: AgentPhase,
        reason: LoopStopReason,
        steps: list[AgentLoopStep],
        observations: list[Observation],
        project_results: list[ProjectResult],
        search_results: list[SearchResult],
        web_pages: list[WebPage],
        web_verifications: list[WebVerification],
        public_claims: list[PublicClaim],
        degraded_nodes: list[str],
        web_search_status: str,
        web_fetch_status: str,
        internal_search_status: str,
        tool_calls: int,
    ) -> AgentLoopResult:
        self._log_event(
            task_id,
            event_type="AGENT_LOOP_STOP",
            node_name="agent_loop",
            status=reason,
            title="Agent Loop 已停止",
            detail=f"停止原因：{reason}，当前阶段：{phase}。",
            payload={"reason": reason, "phase": phase, "iterations": len(steps)},
        )
        return AgentLoopResult(
            phase=phase,
            stop_reason=reason,
            steps=tuple(steps),
            observations=tuple(observations),
            project_results=tuple(project_results),
            search_results=tuple(search_results),
            web_pages=tuple(web_pages),
            web_verifications=tuple(web_verifications),
            public_claims=tuple(public_claims),
            degraded_nodes=tuple(degraded_nodes),
            web_search_status=web_search_status,
            web_fetch_status=web_fetch_status,
            internal_search_status=internal_search_status,
            tool_calls=tool_calls,
        )

    def _log_event(self, task_id: str, **values) -> None:
        logger = getattr(self.event_recorder, "log_execution_event", None)
        if logger is not None:
            logger(task_id, **values)


def scaffold_action(_task_id: str, context: AgentContext) -> AgentAction:
    return FALLBACK_ACTIONS[context.phase]


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
