import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from typing import Literal

from fastapi import Request
from sqlalchemy.orm import Session

from app.database import SessionLocal, TaskRepository
from app.models.database import ExecutionEvent, ResearchTask
from app.schemas.task import StreamExecutionEvent

#任务终态集合
TERMINAL_TASK_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "NEEDS_CONFIRMATION"}
StreamEventType = Literal[
    "PHASE_CHANGED",
    "AGENT_ACTION",
    "TOOL_STARTED",
    "TOOL_RESULT",
    "CONTEXT_UPDATED",
    "LLM_STARTED",
    "LLM_TOKEN",
    "DEGRADED",
    "DONE",
]


def map_execution_event(event: ExecutionEvent) -> StreamExecutionEvent | None:
    event_type = _stream_event_type(event)
    if event_type is None:
        return None
    return StreamExecutionEvent(
        sequence=event.id,
        event_type=event_type,
        node_name=event.node_name,
        status=event.status,
        title=event.title,
        detail=event.detail,
        payload=_public_payload(event, event_type),
        created_at=event.created_at,
    )


def encode_sse(event: StreamExecutionEvent) -> str:
    return (
        f"id: {event.sequence}\n"
        f"event: {event.event_type}\n"
        f"data: {event.model_dump_json()}\n\n"
    )


async def stream_execution_events(
    task_id: str,
    request: Request,
    *,
    after_sequence: int = 0,
    session_factory: Callable[[], Session] = SessionLocal,
    poll_seconds: float = 0.5,
    heartbeat_seconds: float = 15.0,
) -> AsyncIterator[str]:
    cursor = after_sequence
    heartbeat_elapsed = 0.0
    yield "retry: 2000\n\n"

    while True:
        if await request.is_disconnected():
            return

        with session_factory() as session:
            repository = TaskRepository(session)
            task = repository.get_fresh(task_id)
            if task is None:
                return
            events = repository.list_execution_events(
                _scope_ids(task),
                after_sequence=cursor,
            )
            terminal = task.status in TERMINAL_TASK_STATUSES

        emitted = False
        done_emitted = False
        for stored_event in events:
            cursor = max(cursor, stored_event.id)
            stream_event = map_execution_event(stored_event)
            if stream_event is None:
                continue
            emitted = True
            done_emitted = done_emitted or stream_event.event_type == "DONE"
            yield encode_sse(stream_event)

        if terminal:
            if not done_emitted:
                yield encode_sse(
                    StreamExecutionEvent(
                        sequence=cursor + 1,
                        event_type="DONE",
                        node_name="research_pipeline",
                        status=task.status,
                        title="任务状态已恢复",
                        detail="已从任务快照恢复终态。",
                        payload={
                            "task_status": task.status,
                            "source_event_type": "TASK_SNAPSHOT",
                        },
                        created_at=getattr(task, "updated_at", None)
                        or datetime.now(timezone.utc),
                    )
                )
            return

        if emitted:
            heartbeat_elapsed = 0.0
        else:
            heartbeat_elapsed += poll_seconds
            if heartbeat_elapsed >= heartbeat_seconds:
                yield ": keep-alive\n\n"
                heartbeat_elapsed = 0.0
        await asyncio.sleep(poll_seconds)


def _scope_ids(task: ResearchTask) -> list[str]:
    snapshot = dict(task.input_snapshot or {})
    return list(
        dict.fromkeys(
            item
            for item in (
                task.intake_session_id,
                snapshot.get("cleared_from_intake_session_id"),
                task.id,
            )
            if item
        )
    )


def _stream_event_type(event: ExecutionEvent) -> StreamEventType | None:
    source = event.event_type
    if source == "STATUS":
        return "DONE" if event.status in TERMINAL_TASK_STATUSES else "PHASE_CHANGED"
    if source == "AGENT_PHASE":
        return "PHASE_CHANGED"
    if source == "AGENT_ACTION":
        return "AGENT_ACTION"
    if source in {"SEARCH_REQUEST", "TOOL_REQUEST"}:
        return "TOOL_STARTED"
    if source in {"SEARCH_RESPONSE", "TOOL_RESPONSE", "TOOL_ERROR"}:
        return "TOOL_RESULT"
    if source in {"AGENT_OBSERVATION", "RULE_ROUTING"}:
        return "CONTEXT_UPDATED"
    if source == "LLM_REQUEST":
        return "LLM_STARTED"
    if source == "LLM_TOKEN":
        return "LLM_TOKEN"
    if source in {"FALLBACK", "LLM_ERROR", "PIPELINE_ERROR"}:
        return "DEGRADED"
    if source in {"AGENT_LOOP_STOP", "PIPELINE_CANCELLED"}:
        return "CONTEXT_UPDATED"
    if source in {"RULE_GENERATION", "LLM_RESPONSE", "LLM_RAW_RESPONSE"}:
        return None
    return "CONTEXT_UPDATED"


def _public_payload(
    event: ExecutionEvent,
    stream_type: StreamEventType,
) -> dict | list | str | None:
    payload = event.payload
    if not isinstance(payload, dict):
        return payload

    if stream_type == "PHASE_CHANGED":
        return {
            "phase": payload.get("phase") or event.status,
            "iteration": payload.get("iteration"),
            "source_event_type": event.event_type,
        }
    if stream_type == "AGENT_ACTION":
        return {
            key: payload.get(key)
            for key in (
                "iteration",
                "phase",
                "action",
                "next_phase",
                "used_fallback",
                "selection_error",
                "blocked_reason",
            )
            if key in payload
        }
    if stream_type == "TOOL_STARTED":
        return {
            key: payload.get(key)
            for key in ("provider", "tool", "queries", "urls", "arguments")
            if key in payload
        }
    if stream_type == "TOOL_RESULT":
        return {
            key: payload.get(key)
            for key in (
                "provider",
                "tool",
                "result_count",
                "project_ids",
                "pages",
                "error_type",
            )
            if key in payload
        }
    if stream_type == "LLM_STARTED":
        return {
            key: payload.get(key)
            for key in ("model", "prompt_version", "attempt")
            if key in payload
        }
    if stream_type == "LLM_TOKEN":
        return {
            key: payload.get(key)
            for key in ("token", "delta", "index")
            if key in payload
        }
    if stream_type == "DEGRADED":
        return {
            key: payload.get(key)
            for key in (
                "error_type",
                "ambiguous_candidate_ids",
                "blocked_reason",
            )
            if key in payload
        }
    if stream_type == "DONE":
        return {
            "task_status": event.status,
            "source_event_type": event.event_type,
        }
    if event.event_type == "AGENT_OBSERVATION":
        return {
            "iteration": payload.get("iteration"),
            "observation": payload.get("observation"),
        }
    if event.event_type == "AGENT_LOOP_STOP":
        return {
            key: payload.get(key)
            for key in ("reason", "phase", "iterations")
            if key in payload
        }
    if event.event_type in {"RULE_GENERATION", "RULE_ROUTING"}:
        return {
            key: payload.get(key)
            for key in (
                "generator",
                "action",
                "accepted_candidate_ids",
                "rejected_candidate_ids",
                "ambiguous_candidate_ids",
            )
            if key in payload
        }
    return {"source_event_type": event.event_type}
