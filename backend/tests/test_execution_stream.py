import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import app.api.tasks as task_api
from app.database import SessionLocal, TaskRepository
from app.main import app
from app.models.database import ExecutionEvent, ResearchTask
from app.services.infrastructure.execution_stream import (
    encode_sse,
    map_execution_event,
    stream_execution_events,
)


def _event(event_type: str, *, status: str = "RUNNING", payload=None):
    event = ExecutionEvent(
        scope_id="task-stream",
        event_type=event_type,
        node_name="demo",
        status=status,
        title="Demo event",
        detail="Demo detail",
        payload=payload,
    )
    event.id = 42
    event.created_at = datetime.now(timezone.utc)
    return event


@pytest.mark.parametrize(
    ("source_type", "status", "stream_type"),
    [
        ("AGENT_PHASE", "PUBLIC_RESEARCH", "PHASE_CHANGED"),
        ("AGENT_ACTION", "SEARCH_PUBLIC", "AGENT_ACTION"),
        ("SEARCH_REQUEST", "RUNNING", "TOOL_STARTED"),
        ("TOOL_RESPONSE", "SUCCESS", "TOOL_RESULT"),
        ("AGENT_OBSERVATION", "SUCCESS", "CONTEXT_UPDATED"),
        ("LLM_REQUEST", "RUNNING", "LLM_STARTED"),
        ("LLM_TOKEN", "RUNNING", "LLM_TOKEN"),
        ("FALLBACK", "DEGRADED", "DEGRADED"),
        ("STATUS", "COMPLETED", "DONE"),
    ],
)
def test_execution_events_map_to_the_public_stream_protocol(
    source_type, status, stream_type
) -> None:
    mapped = map_execution_event(
        _event(source_type, status=status, payload={"phase": status})
    )

    assert mapped is not None
    assert mapped.event_type == stream_type
    assert mapped.sequence == 42


def test_stream_hides_llm_messages_and_raw_responses() -> None:
    started = map_execution_event(
        _event(
            "LLM_REQUEST",
            payload={
                "model": "demo-model",
                "messages": [{"role": "user", "content": "secret prompt"}],
            },
        )
    )
    raw = map_execution_event(
        _event("LLM_RAW_RESPONSE", payload={"content": "secret response"})
    )

    assert started is not None
    assert started.payload == {"model": "demo-model"}
    assert "secret" not in encode_sse(started)
    assert raw is None


def test_sse_endpoint_streams_and_resumes_from_last_event_id(monkeypatch) -> None:
    monkeypatch.setattr(task_api.run_research_pipeline, "delay", lambda _: None)

    with TestClient(app) as client:
        created = client.post("/api/v1/tasks/text", json={"text": "stream demo"})
        task_id = created.json()["task_id"]
        with SessionLocal() as session:
            repository = TaskRepository(session)
            started = repository.log_execution_event(
                task_id,
                event_type="TOOL_REQUEST",
                node_name="tavily_search",
                status="RUNNING",
                title="Tavily started",
                detail="Search started",
                payload={"provider": "Tavily", "queries": ["demo"]},
            )
            completed = repository.log_execution_event(
                task_id,
                event_type="SEARCH_RESPONSE",
                node_name="tavily_search",
                status="SUCCESS",
                title="Tavily completed",
                detail="Search completed",
                payload={"result_count": 1},
            )
            repository.update(task_id, status="COMPLETED")

        response = client.get(f"/api/v1/tasks/{task_id}/events")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "event: TOOL_STARTED" in response.text
        assert "event: TOOL_RESULT" in response.text
        assert "event: DONE" in response.text

        resumed = client.get(
            f"/api/v1/tasks/{task_id}/events",
            headers={"Last-Event-ID": str(started.id)},
            params={"after_sequence": max(0, started.id - 1)},
        )
        assert f"id: {started.id}\n" not in resumed.text
        assert f"id: {completed.id}\n" in resumed.text
        data_lines = [
            line.removeprefix("data: ")
            for line in resumed.text.splitlines()
            if line.startswith("data: ")
        ]
        assert all(json.loads(line)["sequence"] > started.id for line in data_lines)


def test_sse_endpoint_returns_404_for_unknown_task() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/tasks/00000000-0000-0000-0000-000000000000/events"
        )

    assert response.status_code == 404


class ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_stream_polls_for_events_created_after_connection() -> None:
    task_id = str(uuid4())
    with SessionLocal() as session:
        TaskRepository(session).add(
            ResearchTask(id=task_id, input_type="text", input_text="live stream")
        )

    stream = stream_execution_events(
        task_id,
        ConnectedRequest(),
        poll_seconds=0.01,
        heartbeat_seconds=60,
    )
    assert await anext(stream) == "retry: 2000\n\n"
    first_event = await anext(stream)
    assert "event: PHASE_CHANGED" in first_event

    with SessionLocal() as session:
        TaskRepository(session).log_execution_event(
            task_id,
            event_type="AGENT_ACTION",
            node_name="intake_identity_loop",
            status="SEARCH_INTERNAL",
            title="Agent action",
            detail="Search internal projects",
            payload={"action": "SEARCH_INTERNAL", "phase": "PUBLIC_RESEARCH"},
        )

    next_event = await anext(stream)
    assert "event: AGENT_ACTION" in next_event
    assert "SEARCH_INTERNAL" in next_event
    await stream.aclose()


@pytest.mark.asyncio
async def test_terminal_snapshot_emits_done_without_execution_history() -> None:
    task_id = str(uuid4())
    with SessionLocal() as session:
        task = ResearchTask(
            id=task_id,
            input_type="text",
            input_text="legacy task",
            status="COMPLETED",
        )
        session.add(task)
        session.commit()

    stream = stream_execution_events(task_id, ConnectedRequest(), poll_seconds=0.01)
    assert await anext(stream) == "retry: 2000\n\n"
    done = await anext(stream)

    assert "event: DONE" in done
    assert '"source_event_type":"TASK_SNAPSHOT"' in done
