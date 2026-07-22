from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import TaskRepository, get_session
from app.models.database import ResearchTask
from app.schemas.task import (
    AssociationAnalysis,
    ConfirmationPayload,
    ConfirmationRequest,
    ConfirmedContext,
    ExtractedInfo,
    IntentUnderstanding,
    ProjectQueryPlan,
    ProjectRanking,
    ProjectResult,
    PublicClaim,
    TaskCreated,
    TaskResponse,
    TextTaskRequest,
    WebSearchPlan,
    WebVerification,
)
from app.services.entity_resolver import EntityResolver
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
        status="PLANNING_WEB_SEARCH",
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
    if task.status not in {"PENDING", "NEEDS_CONFIRMATION"}:
        raise HTTPException(status_code=409, detail="当前任务不能取消")
    repository.update(str(task_id), status="CANCELLED")
    return get_task(task_id, session)
