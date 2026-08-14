from types import SimpleNamespace

from app.schemas.task import ConfirmedContext, Observation
from app.services.agent_context import AgentContextBuilder
from app.services.agent_loop import (
    AgentLoopRunner,
    RepeatedActionGuard,
    ToolExecutionResult,
    scaffold_action,
)


class EventRecorder:
    def __init__(self):
        self.events: list[dict] = []

    def log_execution_event(self, scope_id: str, **values):
        self.events.append({"scope_id": scope_id, **values})


class EmptyToolExecutor:
    def execute(self, _task_id, action, context):
        return ToolExecutionResult(
            observation=Observation(
                phase=context.phase,
                action=action,
                status="EMPTY",
                summary="规则工具未返回结果",
            )
        )


def _task():
    return SimpleNamespace(
        id="task-loop",
        input_text="准备会面",
        confirmation_request=None,
        ranked_internal_results=[],
        association_analysis=None,
    )


def _context() -> ConfirmedContext:
    return ConfirmedContext(
        intents=["MEETING_PREPARATION"],
        entities=[],
        event_type="会议",
    )


def _run(runner: AgentLoopRunner, phase="PUBLIC_RESEARCH"):
    return runner.run(
        phase,
        _task(),
        _context(),
        evidence=[],
        project_results=[],
        recent_messages=[],
    )


def test_runner_expresses_the_scaffold_transition_chain_without_tools() -> None:
    recorder = EventRecorder()
    result = _run(
        AgentLoopRunner(
            AgentContextBuilder(), recorder, scaffold_action, EmptyToolExecutor()
        )
    )

    assert [step.phase for step in result.steps] == [
        "PUBLIC_RESEARCH",
        "PROJECT_RESEARCH",
        "SYNTHESIS",
    ]
    assert [step.action for step in result.steps] == [
        "SEARCH_PUBLIC",
        "SEARCH_INTERNAL",
        "SYNTHESIZE",
    ]
    assert [step.next_phase for step in result.steps] == [
        "PROJECT_RESEARCH",
        "SYNTHESIS",
        "DONE",
    ]
    assert result.phase == "DONE"
    assert result.stop_reason == "DONE"
    assert [event["event_type"] for event in recorder.events] == [
        "AGENT_PHASE",
        "AGENT_ACTION",
        "AGENT_OBSERVATION",
        "AGENT_PHASE",
        "AGENT_ACTION",
        "AGENT_OBSERVATION",
        "AGENT_PHASE",
        "AGENT_ACTION",
        "AGENT_LOOP_STOP",
    ]


def test_illegal_actions_use_phase_fallbacks() -> None:
    recorder = EventRecorder()
    runner = AgentLoopRunner(
        AgentContextBuilder(),
        recorder,
        lambda _task_id, _context: "UNSAFE_TOOL_CALL",
        EmptyToolExecutor(),
    )

    result = _run(runner)

    assert result.phase == "DONE"
    assert all(step.used_fallback for step in result.steps)
    assert [step.action for step in result.steps] == [
        "SEARCH_PUBLIC",
        "SEARCH_INTERNAL",
        "SYNTHESIZE",
    ]
    action_events = [event for event in recorder.events if event["event_type"] == "AGENT_ACTION"]
    assert all(event["payload"]["used_fallback"] for event in action_events)


def test_max_loop_limit_stops_before_another_iteration() -> None:
    recorder = EventRecorder()
    runner = AgentLoopRunner(
        AgentContextBuilder(),
        recorder,
        scaffold_action,
        EmptyToolExecutor(),
        max_loops=2,
    )

    result = _run(runner)

    assert result.phase == "SYNTHESIS"
    assert result.stop_reason == "MAX_LOOPS"
    assert len(result.steps) == 2


def test_repeated_action_guard_blocks_consecutive_duplicates() -> None:
    guard = RepeatedActionGuard(max_repeated_actions=2)

    assert guard.observe("SEARCH_PUBLIC") is False
    assert guard.observe("SEARCH_PUBLIC") is False
    assert guard.observe("SEARCH_PUBLIC") is True
    assert guard.observe("SEARCH_INTERNAL") is False


def test_runner_records_and_stops_a_repeated_action() -> None:
    recorder = EventRecorder()
    runner = AgentLoopRunner(
        AgentContextBuilder(),
        recorder,
        lambda _task_id, _context: "SEARCH_PUBLIC",
        EmptyToolExecutor(),
        max_repeated_actions=1,
    )

    result = _run(runner, phase="IDENTITY")

    assert result.phase == "PUBLIC_RESEARCH"
    assert result.stop_reason == "REPEATED_ACTION"
    assert len(result.steps) == 2
    blocked = [
        event
        for event in recorder.events
        if event["event_type"] == "AGENT_ACTION" and event["status"] == "BLOCKED"
    ]
    assert len(blocked) == 1
    assert blocked[0]["payload"]["repeated"] is True


def test_agent_turn_error_uses_deterministic_fallback() -> None:
    recorder = EventRecorder()

    def fail_agent_turn(_task_id, _context):
        raise RuntimeError("model unavailable")

    runner = AgentLoopRunner(
        AgentContextBuilder(),
        recorder,
        fail_agent_turn,
        EmptyToolExecutor(),
    )

    result = _run(runner)

    assert result.phase == "DONE"
    assert all(step.used_fallback for step in result.steps)
    assert all(step.selection_error == "RuntimeError" for step in result.steps)
    action_events = [
        event for event in recorder.events if event["event_type"] == "AGENT_ACTION"
    ]
    assert all(event["payload"]["selection_error"] == "RuntimeError" for event in action_events)


def test_direct_synthesis_skips_all_tools() -> None:
    recorder = EventRecorder()
    runner = AgentLoopRunner(
        AgentContextBuilder(),
        recorder,
        lambda _task_id, _context: "SYNTHESIZE",
        EmptyToolExecutor(),
    )

    result = _run(runner)

    assert result.stop_reason == "DONE"
    assert result.tool_calls == 0
    assert [step.action for step in result.steps] == ["SYNTHESIZE"]


def test_max_tool_call_limit_degrades_and_stops_the_loop() -> None:
    recorder = EventRecorder()
    runner = AgentLoopRunner(
        AgentContextBuilder(),
        recorder,
        lambda _task_id, _context: "SEARCH_PUBLIC",
        EmptyToolExecutor(),
        max_tool_calls=1,
        max_repeated_actions=3,
    )

    result = _run(runner)

    assert result.stop_reason == "MAX_TOOL_CALLS"
    assert result.tool_calls == 1
    assert "agent_loop" in result.degraded_nodes
    blocked = [
        event
        for event in recorder.events
        if event["event_type"] == "AGENT_ACTION" and event["status"] == "BLOCKED"
    ]
    assert blocked[0]["payload"]["blocked_reason"] == "MAX_TOOL_CALLS"
