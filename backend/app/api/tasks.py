from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal, TaskRepository, get_session
from app.models.database import IntakeSession, ResearchTask
from app.schemas.task import (
    AssociationAnalysis,
    ConfirmationPayload,
    ConfirmationRequest,
    ConfirmedContext,
    ExtractedInfo,
    ExecutionEventResponse,
    ExecutionLogResponse,
    IntentUnderstanding,
    ProjectQueryPlan,
    ProjectRanking,
    ProjectResult,
    PublicClaim,
    TaskCreated,
    TaskChatMessage,
    TaskChatRequest,
    TaskChatResponse,
    TaskClearResponse,
    TaskResponse,
    TextTaskRequest,
    WebSearchPlan,
    WebVerification,
)
from app.services.entity_resolver import EntityResolver
from app.services.analysis_chat import (
    AnalysisChatAgent,
    build_task_chat_context,
    fallback_task_chat_reply,
)
from app.services.llm_client import LLMCallFailed, LLMUnavailable, StructuredLLM
from app.services.execution_stream import stream_execution_events
from app.tasks.pipeline import run_research_pipeline


router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])
MAX_AUDIO_BYTES = 30 * 1024 * 1024


@router.post("/text", response_model=TaskCreated, status_code=status.HTTP_202_ACCEPTED)
def create_text_task(payload: TextTaskRequest, session: Session = Depends(get_session)) -> TaskCreated:
    task = TaskRepository(session).add(
        ResearchTask(id=str(uuid4()), input_type="text", input_text=payload.text.strip())
    )
    run_research_pipeline.delay(task.id)
    return TaskCreated(task_id=UUID(task.id), input_type="text")


@router.post("/audio", response_model=TaskCreated, status_code=status.HTTP_202_ACCEPTED)
async def create_audio_task(
    audio: UploadFile = File(...), session: Session = Depends(get_session)
) -> TaskCreated:
    if audio.content_type != "audio/webm":
        raise HTTPException(status_code=415, detail="仅支持 audio/webm 录音")
    content = await audio.read(MAX_AUDIO_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="录音文件为空")
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="录音文件不能超过 30 MB")

    task_id = str(uuid4())
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = Path(settings.audio_dir) / f"{task_id}.webm"
    audio_path.write_bytes(content)
    task = TaskRepository(session).add(
        ResearchTask(id=task_id, input_type="audio", audio_path=str(audio_path))
    )
    run_research_pipeline.delay(task.id)
    return TaskCreated(task_id=UUID(task.id), input_type="audio")


@router.get("/{task_id}", response_model=TaskResponse)
def get_task(task_id: UUID, session: Session = Depends(get_session)) -> TaskResponse:
    task = TaskRepository(session).get(str(task_id))
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    snapshot = dict(task.input_snapshot or {})
    return TaskResponse(
        task_id=UUID(task.id),
        status=task.status,
        input_type=task.input_type,
        input_text=task.input_text,
        extracted_info=ExtractedInfo.model_validate(task.extracted_info)
        if task.extracted_info
        else None,
        llm_understanding=IntentUnderstanding.model_validate(task.llm_understanding)
        if task.llm_understanding
        else None,
        confirmation_request=ConfirmationRequest.model_validate(task.confirmation_request)
        if task.confirmation_request
        else None,
        confirmed_context=ConfirmedContext.model_validate(task.confirmed_context)
        if task.confirmed_context
        else None,
        web_search_plan=WebSearchPlan.model_validate(task.web_search_plan)
        if task.web_search_plan
        else None,
        web_search_status=task.web_search_status,
        web_fetch_status=task.web_fetch_status,
        verified_web_results=[
            WebVerification.model_validate(item) for item in (task.verified_web_results or [])
        ],
        public_claims=[PublicClaim.model_validate(item) for item in (task.public_claims or [])],
        project_query_plan=ProjectQueryPlan.model_validate(task.project_query_plan)
        if task.project_query_plan
        else None,
        internal_search_status=task.internal_search_status,
        internal_results=[ProjectResult.model_validate(item) for item in (task.internal_results or [])],
        ranked_internal_results=[
            ProjectRanking.model_validate(item) for item in (task.ranked_internal_results or [])
        ],
        association_analysis=AssociationAnalysis.model_validate(task.association_analysis)
        if task.association_analysis
        else None,
        detailed_report_markdown=task.detailed_report_markdown,
        action_brief_markdown=task.action_brief_markdown,
        report_markdown=task.report_markdown,
        degraded_nodes=task.degraded_nodes or [],
        error_message=task.error_message,
        analysis_chat_messages=[
            TaskChatMessage.model_validate(item)
            for item in snapshot.get("analysis_chat_messages", [])
        ],
    )


@router.get("/{task_id}/execution-log", response_model=ExecutionLogResponse)
def get_execution_log(
    task_id: UUID,
    after_sequence: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
) -> ExecutionLogResponse:
    repository = TaskRepository(session)
    task = repository.get(str(task_id))
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    snapshot = dict(task.input_snapshot or {})
    scope_ids = [
        item
        for item in (
            task.intake_session_id,
            snapshot.get("cleared_from_intake_session_id"),
            task.id,
        )
        if item
    ]
    events = repository.list_execution_events(scope_ids, after_sequence=after_sequence)
    return ExecutionLogResponse(
        task_id=task_id,
        latest_sequence=max([after_sequence, *(event.id for event in events)]),
        events=[
            ExecutionEventResponse(
                sequence=event.id,
                event_type=event.event_type,
                node_name=event.node_name,
                status=event.status,
                title=event.title,
                detail=event.detail,
                payload=event.payload,
                created_at=event.created_at,
            )
            for event in events
        ],
    )


@router.get("/{task_id}/events")
async def stream_task_execution_events(
    task_id: UUID,
    request: Request,
    after_sequence: int = Query(default=0, ge=0),
) -> StreamingResponse:
    with SessionLocal() as session:
        if TaskRepository(session).get(str(task_id)) is None:
            raise HTTPException(status_code=404, detail="Task not found")

    last_event_id = request.headers.get("last-event-id")
    if last_event_id:
        try:
            after_sequence = max(after_sequence, int(last_event_id))
        except ValueError:
            pass

    return StreamingResponse(
        stream_execution_events(
            str(task_id),
            request,
            after_sequence=after_sequence,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{task_id}/confirm", response_model=TaskResponse)
def confirm_task(
    task_id: UUID,
    payload: ConfirmationPayload,
    session: Session = Depends(get_session),
) -> TaskResponse:
    repository = TaskRepository(session)
    task = repository.get(str(task_id))
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.status != "NEEDS_CONFIRMATION":
        raise HTTPException(status_code=409, detail="任务当前不需要确认")
    if payload.confirmation_version != task.confirmation_version:
        raise HTTPException(status_code=409, detail="确认版本已过期，请刷新任务后重试")
    try:
        request = ConfirmationRequest.model_validate(task.confirmation_request)
        understanding = IntentUnderstanding.model_validate(task.llm_understanding)
        context = EntityResolver().apply_confirmation(
            request,
            payload.selections,
            understanding,
            task.input_text or "",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    repository.update(
        str(task_id),
        status="PLANNING_PROJECT_SEARCH",
        confirmed_context=context.model_dump(mode="json"),
        confirmed_at=datetime.now(timezone.utc),
        error_message=None,
    )
    run_research_pipeline.delay(str(task_id))
    return get_task(task_id, session)


@router.post("/{task_id}/cancel", response_model=TaskResponse)
def cancel_task(task_id: UUID, session: Session = Depends(get_session)) -> TaskResponse:
    repository = TaskRepository(session)
    task = repository.get(str(task_id))
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.status in {"COMPLETED", "FAILED", "CANCELLED"}:
        raise HTTPException(status_code=409, detail="当前任务已经结束")
    repository.update(str(task_id), status="CANCELLED")
    return get_task(task_id, session)


@router.post("/{task_id}/chat", response_model=TaskChatResponse)
def chat_with_task(
    task_id: UUID,
    payload: TaskChatRequest,
    session: Session = Depends(get_session),
) -> TaskChatResponse:
    repository = TaskRepository(session)
    task = repository.get_fresh(str(task_id))
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    snapshot = dict(task.input_snapshot or {})
    history = [
        TaskChatMessage.model_validate(item).model_dump(mode="json")
        for item in snapshot.get("analysis_chat_messages", [])
    ][-38:]
    context = build_task_chat_context(task)
    try:
        result = AnalysisChatAgent(StructuredLLM(settings, repository)).respond(
            str(task_id), payload.message.strip(), history, context
        )
        assistant_reply = result.assistant_reply
    except (LLMUnavailable, LLMCallFailed):
        assistant_reply = fallback_task_chat_reply(context)
    messages = [
        *history,
        {"role": "user", "content": payload.message.strip()},
        {"role": "assistant", "content": assistant_reply},
    ][-40:]
    snapshot["analysis_chat_messages"] = messages
    repository.update(str(task_id), input_snapshot=snapshot)
    return TaskChatResponse(
        task_id=task_id,
        task_status=task.status,
        messages=[TaskChatMessage.model_validate(item) for item in messages],
    )


@router.post("/{task_id}/clear", response_model=TaskClearResponse)
def clear_task_analysis(
    task_id: UUID, session: Session = Depends(get_session)
) -> TaskClearResponse:
    repository = TaskRepository(session)
    task = repository.get_fresh(str(task_id))
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.status not in {"COMPLETED", "FAILED", "CANCELLED", "NEEDS_CONFIRMATION"}:
        raise HTTPException(status_code=409, detail="请先停止当前分析，再清空")

    intake_version = None
    ready_to_analyze = False
    if task.intake_session_id:
        previous_intake_session_id = task.intake_session_id
        intake_session = session.get(IntakeSession, previous_intake_session_id)
        if intake_session and intake_session.research_task_id == task.id:
            final_confirmation = (
                (intake_session.structured_context or {}).get("final_confirmation") or {}
            )
            confirmed = final_confirmation.get("status") == "CONFIRMED"
            intake_session.research_task_id = None
            intake_session.status = "READY" if confirmed else "COLLECTING"
            intake_session.ready_to_analyze = confirmed
            intake_session.version += 1
            snapshot = dict(task.input_snapshot or {})
            snapshot["cleared_from_intake_session_id"] = previous_intake_session_id
            task.input_snapshot = snapshot
            task.intake_session_id = None
            session.commit()
            session.refresh(intake_session)
            intake_version = intake_session.version
            ready_to_analyze = confirmed
    repository.log_execution_event(
        str(task_id),
        event_type="TASK_CLEARED",
        status=task.status,
        title="当前分析已清空",
        detail="已解除信息采集会话与当前分析任务的关联，任务记录仍保留。",
    )
    return TaskClearResponse(
        task_id=task_id,
        intake_session_version=intake_version,
        ready_to_analyze=ready_to_analyze,
    )
