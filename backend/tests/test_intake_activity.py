from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.api.tasks as task_api
from app.config import Settings
from app.database import SessionLocal, TaskRepository
from app.main import app
from app.schemas.task import WebSearchPlan, WebSearchQuery
from app.services.llm_client import StructuredLLM


class RecordingRepository:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.llm_calls: list[tuple[str, dict]] = []

    def log_execution_event(self, scope_id: str, **values) -> None:
        self.events.append((scope_id, values))

    def log_llm_call(self, task_id: str, **values) -> None:
        self.llm_calls.append((task_id, values))


class FakeChatCompletions:
    def __init__(self, content: str) -> None:
        self.content = content

    def create(self, **_):
        return SimpleNamespace(
            id="chat-execution-log",
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
        )


def test_task_execution_log_endpoint_supports_incremental_reads(monkeypatch) -> None:
    monkeypatch.setattr(task_api.run_research_pipeline, "delay", lambda _: None)

    with TestClient(app) as client:
        created = client.post("/api/v1/tasks/text", json={"text": "测试执行日志"})
        task_id = created.json()["task_id"]

        first = client.get(f"/api/v1/tasks/{task_id}/execution-log")
        assert first.status_code == 200
        first_payload = first.json()
        assert first_payload["events"][0]["event_type"] == "STATUS"
        assert first_payload["events"][0]["title"] == "任务已创建"

        with SessionLocal() as session:
            TaskRepository(session).log_execution_event(
                task_id,
                event_type="TOOL_REQUEST",
                node_name="demo_tool",
                status="RUNNING",
                title="执行测试工具",
                detail="测试增量读取。",
                payload={"query": "完整搜索指令"},
            )

        second = client.get(
            f"/api/v1/tasks/{task_id}/execution-log",
            params={"after_sequence": first_payload["latest_sequence"]},
        )
        assert second.status_code == 200
        assert [item["node_name"] for item in second.json()["events"]] == ["demo_tool"]


def test_llm_execution_events_include_exact_messages_and_parsed_output() -> None:
    repository = RecordingRepository()
    config = Settings(
        openai_api_key="test-key",
        llm_api_mode="chat_completions",
        llm_max_retries=0,
        llm_disable_response_storage=True,
    )
    service = StructuredLLM(config, repository)
    expected = WebSearchPlan(
        queries=[WebSearchQuery(query="范玉峰 中建二局", purpose="身份核验")]
    )
    service.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeChatCompletions(expected.model_dump_json())
        )
    )

    result = service.parse(
        "task-execution-log",
        "agent_turn",
        {"confirmed_name": "范玉峰"},
        WebSearchPlan,
    )

    assert result == expected
    request = next(
        values for _, values in repository.events if values["event_type"] == "LLM_REQUEST"
    )
    response = next(
        values for _, values in repository.events if values["event_type"] == "LLM_RESPONSE"
    )
    raw_response = next(
        values
        for _, values in repository.events
        if values["event_type"] == "LLM_RAW_RESPONSE"
    )
    assert "JSON Schema" in request["payload"]["messages"][0]["content"]
    assert "范玉峰" in request["payload"]["messages"][1]["content"]
    assert response["payload"]["parsed_output"]["queries"][0]["query"] == "范玉峰 中建二局"
    assert "范玉峰 中建二局" in raw_response["payload"]["content"]
    assert repository.llm_calls[0][1]["input_tokens"] == 12
    assert repository.llm_calls[0][1]["output_tokens"] == 8
