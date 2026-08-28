from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.database import IntakeSessionRepository, TaskRepository, get_session
from app.models.database import IntakeAudioJob, IntakeSession, ResearchTask
from app.schemas.intake import (
    ConfirmIntakeSummaryRequest,
    IntakeActivityResponse,
    IntakeChatRequest,
    IntakeChatResponse,
    IntakeChatResult,
    IntakeAudioJobResponse,
    IntakeSessionResponse,
    IntakeStructuredContext,
    StartAnalysisRequest,
)
from app.schemas.task import TaskCreated
from app.schemas.task import ConfirmationPayload
from app.services.intake.activity import intake_activity
from app.services.intake.agent import IntakeAgent
from app.services.agent_config.service import AgentConfigService
from app.services.agent_config.secrets import SecretStore
from app.services.auth import Principal, get_current_principal
from app.services.conversations import ConversationService
from app.services.intake.completeness import (
    is_intake_ready,
    required_missing_information,
)
from app.services.intake.entity_candidates import (
    IntakeEntityCandidateService,
    user_provided_entity_resolutions,
)
from app.services.intake.defaults import with_default_requester_context
from app.services.integrations.llm_client import StructuredLLM
from app.services.integrations.mcp_client import ProjectMcpClient
from app.services.integrations.tavily_client import TavilyClient
from app.services.intake.runner import (
    IntakeAudioJobNotFound,
    IntakeAudioJobNotReviewable,
    IntakeChatConflict,
    IntakeRunner,
    align_resolution_relationships as _align_resolution_relationships,
    has_resolved_entities as _has_resolved_entities,
    merge_resolutions as _merge_resolutions,
    prepare_final_confirmation as _prepare_final_confirmation,
    source_text as _source_text,
    standardized_analysis_input as _standardized_analysis_input,
    standardized_context as _standardized_context,
    with_field_states as _with_field_states,
)
from app.tasks.pipeline import (
    context_from_intake_snapshot,
    run_research_pipeline,
)
from app.tasks.intake_audio import run_intake_audio_transcription


router = APIRouter(prefix="/api/v1/intake", tags=["intake"])
_default_intake_agent = IntakeAgent(StructuredLLM(settings))
intake_agent = _default_intake_agent
_default_entity_candidates = IntakeEntityCandidateService(
    ProjectMcpClient(settings.mcp_server_url), TavilyClient(settings.tavily_api_key)
)
entity_candidates = _default_entity_candidates
MAX_AUDIO_BYTES = 30 * 1024 * 1024


def _owned_intake(
    session: Session,
    session_id: str,
    principal: Principal,
    *,
    for_update: bool = False,
) -> IntakeSession | None:
    intake_session = IntakeSessionRepository(session).get(
        session_id, for_update=for_update
    )
    if intake_session is None:
        return None
    if intake_session.owner_id and (
        intake_session.owner_id != principal.user_id
        or intake_session.tenant_id != principal.tenant_id
    ):
        return None
    return intake_session


def _agent_for(
    repository: IntakeSessionRepository, resolved_config: dict | None = None
) -> IntakeAgent:
    if intake_agent is not _default_intake_agent:
        return intake_agent
    return IntakeAgent(StructuredLLM(settings, repository, resolved_config))


def _entity_candidates_for(session: Session, resolved_config: dict) -> object:
    if entity_candidates is not _default_entity_candidates:
        return entity_candidates
    projects = ProjectMcpClient.from_snapshot(
        resolved_config,
        caller_node="intake_agent",
        secret_resolver=SecretStore(session, settings).resolve,
    )
    return IntakeEntityCandidateService(
        projects, TavilyClient(settings.tavily_api_key)
    )


def _chat_response(intake_session: IntakeSession) -> IntakeChatResponse:
    messages = intake_session.messages or []
    assistant_reply = next(
        (
            item["content"]
            for item in reversed(messages)
            if item.get("role") == "assistant"
        ),
        "请继续补充本次分析信息。",
    )
    structured_context = IntakeStructuredContext.model_validate(
        intake_session.structured_context or {}
    )
    return IntakeChatResponse(
        session_id=UUID(intake_session.id),
        status=intake_session.status,
        version=intake_session.version,
        assistant_reply=assistant_reply,
        analysis_input=intake_session.analysis_input or "等待补充分析信息。",
        ready_to_analyze=intake_session.ready_to_analyze,
        missing_information=intake_session.missing_information or [],
        structured_context=structured_context,
        next_action="READY"
        if intake_session.status == "READY"
        else "ASK_USER",
        confirmation_request=intake_session.confirmation_request,
        final_confirmation=structured_context.final_confirmation,
    )


def _audio_response(job: IntakeAudioJob) -> IntakeAudioJobResponse:
    return IntakeAudioJobResponse(
        job_id=UUID(job.id),
        session_id=UUID(job.session_id),
        status=job.status,
        transcript=job.transcript,
        corrected_transcript=job.corrected_transcript,
        error_message=job.error_message,
        retry_count=job.retry_count,
    )


def _repair_ready_session(
    intake_session: IntakeSession, repository: IntakeSessionRepository
) -> IntakeSession:
    if intake_session.status != "COLLECTING" or intake_session.confirmation_request:
        return intake_session
    context = IntakeStructuredContext.model_validate(
        intake_session.structured_context or {}
    )
    if not context.final_confirmation or context.final_confirmation.status != "CONFIRMED":
        return intake_session
    result = IntakeChatResult(
        assistant_reply="信息已完整，可以开始分析。",
        analysis_input=intake_session.analysis_input or "等待补充分析信息。",
        ready_to_analyze=True,
        missing_information=[],
        structured_context=context,
    )
    source_text = _source_text(intake_session)
    if not is_intake_ready(result, source_text):
        return intake_session
    structured_context = with_default_requester_context(
        dict(intake_session.structured_context or {})
    )
    existing = structured_context.get("entity_resolutions", [])
    if not _has_resolved_entities(structured_context):
        existing.extend(user_provided_entity_resolutions(context, source_text))
        structured_context["entity_resolutions"] = existing
    if not _has_resolved_entities(structured_context):
        return intake_session
    return repository.update(
        intake_session.id,
        status="READY",
        structured_context=structured_context,
        missing_information=[],
        ready_to_analyze=True,
        version=intake_session.version + 1,
    )


@router.post("/chat", response_model=IntakeChatResponse)
def chat(
    request: IntakeChatRequest,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> IntakeChatResponse:
    repository = IntakeSessionRepository(session)
    existing = _owned_intake(session, str(request.session_id), principal)
    if session.get(IntakeSession, str(request.session_id)) is not None and existing is None:
        raise HTTPException(status_code=404, detail="信息采集会话不存在")
    conversations = ConversationService(session)
    try:
        conversation = conversations.ensure_for_intake(
            principal, str(request.session_id)
        )
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="信息采集会话不存在") from exc
    agent_run = AgentConfigService(session, settings).ensure_intake_run(
        str(request.session_id),
        owner_id=principal.user_id,
        tenant_id=principal.tenant_id,
        conversation_id=conversation.id,
        initiator_role=principal.role,
    )
    runner = IntakeRunner(
        repository=repository,
        session=session,
        agent=_agent_for(repository, agent_run.resolved_config_snapshot),
        entity_candidates=_entity_candidates_for(
            session, agent_run.resolved_config_snapshot
        ),
        activity=intake_activity,
        settings=settings,
    )
    try:
        intake_session = runner.run_chat(request)
    except IntakeAudioJobNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (IntakeChatConflict, IntakeAudioJobNotReviewable) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    conversations.attach_intake(conversation, intake_session)
    conversations.sync_messages(
        conversation,
        intake_session.messages or [],
        channel="intake",
        author_id=principal.user_id,
    )
    return _chat_response(intake_session)


@router.get("/{session_id}/activity", response_model=IntakeActivityResponse)
def get_intake_activity(
    session_id: UUID,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> IntakeActivityResponse:
    if (
        _owned_intake(session, str(session_id), principal) is None
        and principal.auth_enabled
    ):
        raise HTTPException(status_code=404, detail="信息采集会话不存在")
    return IntakeActivityResponse.model_validate(intake_activity.get(str(session_id)))


@router.get("/{session_id}", response_model=IntakeSessionResponse)
def get_intake_session(
    session_id: UUID,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> IntakeSessionResponse:
    intake_session = _owned_intake(session, str(session_id), principal)
    if intake_session is None:
        raise HTTPException(status_code=404, detail="信息采集会话不存在")
    active_audio_job = session.scalar(
        select(IntakeAudioJob)
        .where(
            IntakeAudioJob.session_id == str(session_id),
            IntakeAudioJob.status.in_(("QUEUED", "TRANSCRIBING", "NEEDS_REVIEW", "FAILED")),
        )
        .order_by(IntakeAudioJob.created_at.desc())
        .limit(1)
    )
    if active_audio_job is None:
        intake_session = _repair_ready_session(intake_session, IntakeSessionRepository(session))
    response = _chat_response(intake_session)
    return IntakeSessionResponse(
        **response.model_dump(),
        messages=intake_session.messages or [],
        research_task_id=UUID(intake_session.research_task_id)
        if intake_session.research_task_id
        else None,
        active_audio_job=_audio_response(active_audio_job).model_dump(mode="json")
        if active_audio_job
        else None,
    )


@router.post("/{session_id}/audio", response_model=IntakeAudioJobResponse, status_code=202)
async def upload_intake_audio(
    session_id: UUID,
    audio: UploadFile = File(...),
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> IntakeAudioJobResponse:
    if not settings.intake_audio_enabled:
        raise HTTPException(status_code=503, detail="音频预处理当前已关闭")
    intake_session = _owned_intake(
        session, str(session_id), principal, for_update=True
    )
    if session.get(IntakeSession, str(session_id)) is not None and intake_session is None:
        raise HTTPException(status_code=404, detail="信息采集会话不存在")
    conversations = ConversationService(session)
    try:
        conversation = conversations.ensure_for_intake(principal, str(session_id))
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="信息采集会话不存在") from exc
    if intake_session is None:
        intake_session = IntakeSession(
            id=str(session_id),
            tenant_id=principal.tenant_id,
            owner_id=principal.user_id,
            conversation_id=conversation.id,
            status="COLLECTING",
            messages=[],
            structured_context={},
            missing_information=[],
            analysis_input="",
        )
        session.add(intake_session)
        session.flush()
    conversations.attach_intake(conversation, intake_session)
    if intake_session.status in {"STARTING_ANALYSIS", "ANALYZING"}:
        raise HTTPException(status_code=409, detail="分析任务已创建，不能上传录音")
    if audio.content_type != "audio/webm":
        raise HTTPException(status_code=415, detail="仅支持 audio/webm 录音")
    content = await audio.read(MAX_AUDIO_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="录音文件为空")
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="录音文件不能超过 30 MB")

    job_id = str(uuid4())
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = Path(settings.audio_dir) / f"intake-{job_id}.webm"
    audio_path.write_bytes(content)
    job = IntakeAudioJob(
        id=job_id,
        session_id=str(session_id),
        status="QUEUED",
        audio_path=str(audio_path),
    )
    intake_session.status = "PROCESSING_AUDIO"
    intake_session.ready_to_analyze = False
    intake_session.missing_information = ["等待音频转写和确认"]
    intake_session.version += 1
    session.add(job)
    session.commit()
    run_intake_audio_transcription.delay(job_id)
    return _audio_response(job)


@router.get(
    "/{session_id}/audio/{job_id}", response_model=IntakeAudioJobResponse
)
def get_intake_audio(
    session_id: UUID,
    job_id: UUID,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> IntakeAudioJobResponse:
    if _owned_intake(session, str(session_id), principal) is None:
        raise HTTPException(status_code=404, detail="音频转写任务不存在")
    job = session.get(IntakeAudioJob, str(job_id))
    if job is None or job.session_id != str(session_id):
        raise HTTPException(status_code=404, detail="音频转写任务不存在")
    return _audio_response(job)


@router.post(
    "/{session_id}/audio/{job_id}/retry",
    response_model=IntakeAudioJobResponse,
    status_code=202,
)
def retry_intake_audio(
    session_id: UUID,
    job_id: UUID,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> IntakeAudioJobResponse:
    job = session.get(IntakeAudioJob, str(job_id))
    intake_session = _owned_intake(session, str(session_id), principal)
    if job is None or job.session_id != str(session_id) or intake_session is None:
        raise HTTPException(status_code=404, detail="音频转写任务不存在")
    if job.status != "FAILED" or not job.audio_path:
        raise HTTPException(status_code=409, detail="音频当前不能重试")
    job.status = "QUEUED"
    job.error_message = None
    intake_session.status = "PROCESSING_AUDIO"
    intake_session.missing_information = ["等待音频转写和确认"]
    intake_session.version += 1
    session.commit()
    run_intake_audio_transcription.delay(job.id)
    return _audio_response(job)


@router.post("/{session_id}/confirm", response_model=IntakeSessionResponse)
def confirm_intake_entities(
    session_id: UUID,
    payload: ConfirmationPayload,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> IntakeSessionResponse:
    repository = IntakeSessionRepository(session)
    existing_session = _owned_intake(session, str(session_id), principal)
    if existing_session is None:
        raise HTTPException(status_code=404, detail="信息采集会话不存在")
    conversation = ConversationService(session).ensure_for_intake(
        principal, str(session_id)
    )
    agent_run = AgentConfigService(session, settings).ensure_intake_run(
        str(session_id),
        owner_id=principal.user_id,
        tenant_id=principal.tenant_id,
        conversation_id=conversation.id,
        initiator_role=principal.role,
    )
    intake_session = _owned_intake(
        session, str(session_id), principal, for_update=True
    )
    if intake_session is None:  # pragma: no cover - deleted between reads
        raise HTTPException(status_code=404, detail="信息采集会话不存在")
    request_agent = _agent_for(repository, agent_run.resolved_config_snapshot)
    if intake_session.status != "NEEDS_CONFIRMATION" or not intake_session.confirmation_request:
        raise HTTPException(status_code=409, detail="当前会话不需要身份确认")
    request = intake_session.confirmation_request
    if payload.confirmation_version != request.get("version"):
        raise HTTPException(status_code=409, detail="确认版本已过期，请刷新后重试")

    selections = {item.mention: item for item in payload.selections}
    base_context = IntakeStructuredContext.model_validate(
        intake_session.structured_context or {}
    )
    resolutions = list((intake_session.structured_context or {}).get("entity_resolutions", []))
    confirmed_selections: list[dict] = []
    for item in request.get("items", []):
        selection = selections.get(item["mention"])
        if selection is None:
            raise HTTPException(status_code=422, detail=f"缺少确认项：{item['mention']}")
        candidate = None
        if selection.candidate_id:
            candidate = next(
                (
                    option
                    for option in item.get("candidates", [])
                    if option.get("candidate_id") == selection.candidate_id
                ),
                None,
            )
            if candidate is None:
                raise HTTPException(status_code=422, detail="候选项不属于当前确认请求")
            resolution = {**candidate, "mention": item["mention"], "confirmed_by": "USER"}
        else:
            manual_value = (selection.manual_value or "").strip()
            if len(manual_value) < 2 or len(manual_value) > 100:
                raise HTTPException(status_code=422, detail="手工确认名称长度必须为 2 到 100 个字符")
            resolution = {
                "candidate_id": None,
                "entity_type": item["entity_type"],
                "canonical_name": manual_value,
                "mention": item["mention"],
                "organization": base_context.organizations[0]
                if item["entity_type"] == "PERSON" and base_context.organizations
                else None,
                "title": next(
                    (
                        person.title
                        for person in base_context.people_details
                        if person.name == item["mention"]
                    ),
                    None,
                ),
                "confirmed_by": "USER",
            }
        resolutions = _merge_resolutions(resolutions, [resolution])
        confirmed_selections.append(resolution)

    structured_context = with_default_requester_context(
        dict(intake_session.structured_context or {})
    )
    resolutions = _align_resolution_relationships(resolutions, base_context)
    structured_context = _standardized_context(structured_context, resolutions)
    standardized_input = _standardized_analysis_input(
        intake_session.analysis_input, resolutions
    )
    confirmed_names = list(dict.fromkeys(
        item.get("canonical_name")
        for item in resolutions
        if item.get("canonical_name")
    ))
    selection_descriptions = []
    for item in confirmed_selections:
        details = "、".join(
            value
            for value in (item.get("organization"), item.get("title"))
            if value
        )
        identity = item.get("canonical_name") or item.get("mention")
        selection_descriptions.append(
            f"{item.get('mention')} → {identity}{f'（{details}）' if details else ''}"
        )
    confirmation_message = (
        f"我已在身份确认中选择：{'；'.join(selection_descriptions)}。"
    )
    validation_result = IntakeChatResult(
        assistant_reply=(
            f"已确认标准身份：{'、'.join(confirmed_names)}。可以开始分析。"
            if confirmed_names
            else "身份已确认，可以开始分析。"
        ),
        analysis_input=standardized_input,
        ready_to_analyze=True,
        missing_information=[],
        structured_context=structured_context,
    )
    source_text = "\n".join(
        item.get("content", "")
        for item in (intake_session.messages or [])
        if item.get("role") == "user"
    )
    intake_activity.update(
        str(session_id),
        "PROCESSING_TOOL_RESULT",
        "正在整理确认后的字段并生成最终确认问题",
        tool_name="summarize_intake_confirmation",
    )
    required_missing = required_missing_information(validation_result, source_text)
    all_entities_resolved = _has_resolved_entities(structured_context)
    server_ready = not required_missing and all_entities_resolved
    fallback_missing = required_missing or (
        [] if all_entities_resolved else ["人物或企业身份确认"]
    )
    readiness_reply = (
        f"已复核确认的标准身份：{'、'.join(confirmed_names)}。"
        if server_ready and confirmed_names
        else "身份选择已记录，但还需要补充分析目标或确认关键身份。"
    )
    ready = server_ready
    next_version = intake_session.version + 1
    structured_context = _with_field_states(structured_context)
    final_confirmation = None
    if ready:
        review_messages = [
            *(intake_session.messages or []),
            {"role": "user", "content": confirmation_message},
        ]
        structured_context, final_confirmation = _prepare_final_confirmation(
            request_agent,
            IntakeChatRequest(
                session_id=session_id,
                messages=review_messages[-30:],
            ),
            structured_context,
            standardized_input,
            next_version,
        )
        ready = False
    missing_information = (
        []
        if ready
        else fallback_missing
        or ["身份或分析目标确认"]
    )
    if final_confirmation:
        missing_information = []
    messages = [
        *(intake_session.messages or []),
        {"role": "user", "content": confirmation_message},
        {
            "role": "assistant",
            "content": final_confirmation.question
            if final_confirmation
            else readiness_reply,
        },
    ]
    intake_session = repository.update(
        str(session_id),
        status="AWAITING_FINAL_CONFIRMATION" if final_confirmation else (
            "READY" if ready else "COLLECTING"
        ),
        messages=messages,
        structured_context=structured_context,
        analysis_input=standardized_input,
        missing_information=missing_information,
        confirmation_request=None,
        ready_to_analyze=ready,
        version=next_version,
    )
    intake_activity.update(
        str(session_id),
        "COMPLETED",
        "当前信息已整理，等待最终确认",
        active=False,
        tool_name="summarize_intake_confirmation",
    )
    response = _chat_response(intake_session)
    ConversationService(session).sync_messages(
        conversation,
        intake_session.messages or [],
        channel="intake",
        author_id=principal.user_id,
    )
    return IntakeSessionResponse(
        **response.model_dump(),
        messages=intake_session.messages or [],
        research_task_id=None,
        active_audio_job=None,
    )


@router.post(
    "/{session_id}/confirm-summary",
    response_model=IntakeSessionResponse,
)
def confirm_intake_summary(
    session_id: UUID,
    payload: ConfirmIntakeSummaryRequest,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> IntakeSessionResponse:
    repository = IntakeSessionRepository(session)
    intake_session = _owned_intake(
        session, str(session_id), principal, for_update=True
    )
    if intake_session is None:
        raise HTTPException(status_code=404, detail="信息采集会话不存在")
    context = IntakeStructuredContext.model_validate(
        intake_session.structured_context or {}
    )
    confirmation = context.final_confirmation
    if (
        intake_session.status != "AWAITING_FINAL_CONFIRMATION"
        or confirmation is None
        or confirmation.status != "PENDING"
    ):
        raise HTTPException(status_code=409, detail="当前没有待确认的信息摘要")
    if (
        payload.expected_version != intake_session.version
        or payload.expected_version != confirmation.version
    ):
        raise HTTPException(status_code=409, detail="确认版本已更新，请刷新后重试")

    validation_result = IntakeChatResult(
        assistant_reply=confirmation.question,
        analysis_input=intake_session.analysis_input,
        ready_to_analyze=False,
        structured_context=context,
    )
    if required_missing_information(validation_result, _source_text(intake_session)):
        raise HTTPException(status_code=422, detail="当前信息仍不完整，不能确认")
    if settings.intake_entity_resolution_enabled and not _has_resolved_entities(
        intake_session.structured_context or {}
    ):
        raise HTTPException(status_code=422, detail="人物或企业身份尚未确认")

    structured_context = _with_field_states(
        intake_session.structured_context or {}, final_confirmed=True
    )
    structured_context["final_confirmation"] = confirmation.model_copy(
        update={"status": "CONFIRMED"}
    ).model_dump(mode="json")
    intake_session = repository.update(
        str(session_id),
        status="READY",
        messages=[
            *(intake_session.messages or []),
            {"role": "user", "content": "对，是这样的"},
            {"role": "assistant", "content": "好的，当前信息已经确认，可以立即开始分析。"},
        ],
        structured_context=structured_context,
        missing_information=[],
        ready_to_analyze=True,
        version=intake_session.version + 1,
    )
    intake_activity.update(
        str(session_id),
        "COMPLETED",
        "用户已确认当前信息",
        active=False,
        tool_name="confirm_intake_summary",
    )
    response = _chat_response(intake_session)
    conversation = ConversationService(session).ensure_for_intake(
        principal, str(session_id)
    )
    ConversationService(session).sync_messages(
        conversation,
        intake_session.messages or [],
        channel="intake",
        author_id=principal.user_id,
    )
    return IntakeSessionResponse(
        **response.model_dump(),
        messages=intake_session.messages or [],
        research_task_id=None,
        active_audio_job=None,
    )


@router.post(
    "/{session_id}/start-analysis",
    response_model=TaskCreated,
    status_code=202,
)
def start_analysis(
    session_id: UUID,
    payload: StartAnalysisRequest,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> TaskCreated:
    repository = IntakeSessionRepository(session)
    existing_session = _owned_intake(session, str(session_id), principal)
    if existing_session is None:
        raise HTTPException(status_code=404, detail="信息采集会话不存在")
    conversation = ConversationService(session).ensure_for_intake(
        principal, str(session_id)
    )
    AgentConfigService(session, settings).ensure_intake_run(
        str(session_id),
        owner_id=principal.user_id,
        tenant_id=principal.tenant_id,
        conversation_id=conversation.id,
        initiator_role=principal.role,
    )
    intake_session = _owned_intake(
        session, str(session_id), principal, for_update=True
    )
    if intake_session is None:  # pragma: no cover - deleted between reads
        raise HTTPException(status_code=404, detail="信息采集会话不存在")

    if intake_session.research_task_id:
        task = session.get(ResearchTask, intake_session.research_task_id)
        if task is None:
            raise HTTPException(status_code=409, detail="会话关联的分析任务不存在")
        AgentConfigService(session, settings).link_research_task(
            intake_session.id,
            task.id,
            owner_id=principal.user_id,
            tenant_id=principal.tenant_id,
            conversation_id=conversation.id,
            initiator_role=principal.role,
        )
        return TaskCreated(task_id=UUID(task.id), input_type=task.input_type)
    if payload.expected_version is not None and payload.expected_version != intake_session.version:
        raise HTTPException(status_code=409, detail="会话版本已更新，请刷新后重试")
    if intake_session.status != "READY" or not intake_session.ready_to_analyze:
        raise HTTPException(status_code=409, detail="信息尚未完整，不能开始分析")
    if intake_session.confirmation_request:
        raise HTTPException(status_code=409, detail="仍有待确认的身份信息")
    final_confirmation = IntakeStructuredContext.model_validate(
        intake_session.structured_context or {}
    ).final_confirmation
    if not final_confirmation or final_confirmation.status != "CONFIRMED":
        raise HTTPException(status_code=409, detail="请先确认当前信息摘要")
    if (
        settings.intake_entity_resolution_enabled
        and not _has_resolved_entities(intake_session.structured_context or {})
    ):
        raise HTTPException(status_code=422, detail="人物或企业身份尚未确认")
    audio_jobs = list(
        session.scalars(
            select(IntakeAudioJob).where(IntakeAudioJob.session_id == str(session_id))
        )
    )
    if any(job.status != "TRANSCRIBED" for job in audio_jobs):
        raise HTTPException(status_code=409, detail="仍有未完成或未确认的音频转写")

    validation_result = IntakeChatResult(
        assistant_reply="信息已完整，可以开始分析。",
        analysis_input=intake_session.analysis_input,
        ready_to_analyze=intake_session.ready_to_analyze,
        missing_information=intake_session.missing_information or [],
        structured_context=intake_session.structured_context or {},
    )
    source_text = "\n".join(
        item.get("content", "")
        for item in (intake_session.messages or [])
        if item.get("role") == "user"
    )
    if not is_intake_ready(validation_result, source_text):
        raise HTTPException(status_code=422, detail="会话内容未通过完整性校验")

    task_id = str(uuid4())
    snapshot = {
        "session_id": intake_session.id,
        "session_version": intake_session.version,
        "messages": intake_session.messages or [],
        "structured_context": with_default_requester_context(
            intake_session.structured_context or {}
        ),
        "missing_information": intake_session.missing_information or [],
        "analysis_input": intake_session.analysis_input,
        "audio_transcripts": [job.corrected_transcript for job in audio_jobs],
    }
    confirmed_context = context_from_intake_snapshot(snapshot)
    if confirmed_context is None:
        raise HTTPException(status_code=422, detail="已确认身份无法转换为分析上下文")
    task = ResearchTask(
        id=task_id,
        tenant_id=principal.tenant_id,
        owner_id=principal.user_id,
        conversation_id=conversation.id,
        input_type="audio" if audio_jobs else "text",
        input_text=intake_session.analysis_input.strip(),
        intake_session_id=intake_session.id,
        input_snapshot=snapshot,
        confirmed_context=confirmed_context.model_dump(mode="json"),
        confirmed_at=datetime.now(timezone.utc),
    )
    intake_session.status = "ANALYZING"
    intake_session.research_task_id = task_id
    session.add(task)
    ConversationService(session).attach_task(conversation, task)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = _owned_intake(session, str(session_id), principal)
        if existing is None or not existing.research_task_id:
            raise
        task = session.get(ResearchTask, existing.research_task_id)
        if task is None:
            raise
        AgentConfigService(session, settings).link_research_task(
            intake_session.id,
            task.id,
            owner_id=principal.user_id,
            tenant_id=principal.tenant_id,
            conversation_id=conversation.id,
            initiator_role=principal.role,
        )
        return TaskCreated(task_id=UUID(task.id), input_type=task.input_type)

    AgentConfigService(session, settings).link_research_task(
        intake_session.id,
        task_id,
        owner_id=principal.user_id,
        tenant_id=principal.tenant_id,
        conversation_id=conversation.id,
        initiator_role=principal.role,
    )
    TaskRepository(session).log_execution_event(
        task_id,
        event_type="STATUS",
        status=task.status,
        title="研究任务已创建",
        detail="录入会话已确认，研究任务等待分析服务接管。",
        payload={"intake_session_id": intake_session.id},
    )
    run_research_pipeline.delay(task_id)
    return TaskCreated(task_id=UUID(task.id), input_type=task.input_type)
