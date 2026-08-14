from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.task import AgentContext, AgentTurnDecision, Observation
from app.services.agent_loop import AgentLoopRunner, ToolExecutionResult
from app.services.agent_nodes import AgentNodes
from app.services.agent_context import AgentContextBuilder


ROOT = Path(__file__).resolve().parents[2]


class CapturingLlm:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = []

    def parse(self, task_id, node_name, input_payload, output_model):
        self.calls.append((task_id, node_name, input_payload, output_model))
        return AgentTurnDecision(action=next(self.actions))


class EventRecorder:
    def __init__(self):
        self.events = []

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


def test_agent_turn_decision_allows_only_one_action_field() -> None:
    assert AgentTurnDecision.model_validate({"action": "SEARCH_PUBLIC"}).action == "SEARCH_PUBLIC"

    with pytest.raises(ValidationError):
        AgentTurnDecision.model_validate({"action": "CALL_TAVILY"})
    with pytest.raises(ValidationError):
        AgentTurnDecision.model_validate(
            {
                "action": "SEARCH_PUBLIC",
                "tavily_query": "范玉峰 中建二局",
            }
        )


def test_agent_node_sends_context_and_returns_only_action() -> None:
    llm = CapturingLlm(["SEARCH_PUBLIC"])
    context = AgentContext(phase="PUBLIC_RESEARCH")

    action = AgentNodes(llm).agent_turn("task-1", context)

    assert action == "SEARCH_PUBLIC"
    task_id, node_name, payload, output_model = llm.calls[0]
    assert task_id == "task-1"
    assert node_name == "agent_turn"
    assert payload == {"context": context.model_dump(mode="json")}
    assert output_model is AgentTurnDecision
    assert set(output_model.model_fields) == {"action"}
    assert output_model.model_json_schema()["additionalProperties"] is False


def test_runner_uses_agent_turn_for_each_phase_without_tool_details() -> None:
    llm = CapturingLlm(["SEARCH_PUBLIC", "SEARCH_INTERNAL", "SYNTHESIZE"])
    recorder = EventRecorder()
    runner = AgentLoopRunner(
        AgentContextBuilder(),
        recorder,
        AgentNodes(llm).agent_turn,
        EmptyToolExecutor(),
    )
    task = type(
        "Task",
        (),
        {
            "id": "task-loop",
            "input_text": "准备会面",
            "confirmation_request": None,
            "ranked_internal_results": [],
            "association_analysis": None,
        },
    )()

    result = runner.run(
        "PUBLIC_RESEARCH",
        task,
        confirmed_context=None,
        evidence=[],
        project_results=[],
        recent_messages=[],
    )

    assert result.phase == "DONE"
    assert [call[1] for call in llm.calls] == ["agent_turn", "agent_turn", "agent_turn"]
    assert [call[2]["context"]["phase"] for call in llm.calls] == [
        "PUBLIC_RESEARCH",
        "PROJECT_RESEARCH",
        "SYNTHESIS",
    ]
    assert all(set(call[2]) == {"context"} for call in llm.calls)


def test_agent_turn_prompt_forbids_tool_details_and_facts() -> None:
    prompt = (ROOT / "backend/prompts/agent_turn_v1.txt").read_text(encoding="utf-8")

    assert "只决定动作意图" in prompt
    assert "Tavily 查询词" in prompt
    assert "MCP 工具名" in prompt
    assert "项目排序" in prompt
    assert "新的公开事实" in prompt
    assert "未经验证" in prompt


def test_agent_nodes_exposes_only_active_business_llm_nodes() -> None:
    public_methods = {
        name
        for name, value in AgentNodes.__dict__.items()
        if not name.startswith("_") and callable(value)
    }

    assert public_methods == {"agent_turn", "evidence_verify", "final_synthesis"}
