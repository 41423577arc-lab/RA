from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.database import IntakeSessionRepository, get_session
from app.models.database import IntakeAudioJob, IntakeSession, ResearchTask
from app.schemas.intake import (
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
from app.services.intake_agent import IntakeAgent
from app.services.intake_completeness import (
    is_intake_ready,
    required_missing_information,
)
from app.services.intake_entity_candidates import (
    IntakeEntityCandidateService,
    user_provided_entity_resolutions,
)
from app.services.intake_defaults import with_default_requester_context
from app.services.llm_client import LLMCallFailed, LLMUnavailable, StructuredLLM
from app.services.mcp_client import ProjectMcpClient
from app.services.tavily_client import TavilyClient
from app.tasks.pipeline import run_research_pipeline
from app.tasks.intake_audio import run_intake_audio_transcription


router = APIRouter(prefix="/api/v1/intake", tags=["intake"])
intake_agent = IntakeAgent(StructuredLLM(settings))
entity_candidates = IntakeEntityCandidateService(
    ProjectMcpClient(settings.mcp_server_url), TavilyClient(settings.tavily_api_key)
)
MAX_AUDIO_BYTES = 30 * 1024 * 1024


def _has_resolved_entities(structured_context: dict) -> bool:
    context = IntakeStructuredContext.model_validate(structured_context)
    targets = {*context.people, *context.organizations}
    if not targets:
        return True
    resolutions = structured_context.get("entity_resolutions", [])
    resolved = {
        value
        for item in resolutions
        if item.get("confirmed_by")
        for value in (item.get("mention"), item.get("canonical_name"))
        if value
    }
    return targets.issubset(resolved)


def _fallback_result(request: IntakeChatRequest) -> IntakeChatResult:
    user_text = "\n".join(
        message.content for message in request.messages if message.role == "user"
    ).strip()
    return IntakeChatResult(
        assistant_reply="信息采集助手暂时不可用。请补充涉及的人物或企业，以及希望分析的事项。",
        analysis_input=(user_text or "请补充本次分析信息。")[-10_000:],
        ready_to_analyze=False,
        missing_information=["人物、企业或项目", "希望分析或推动的事项"],
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
    return IntakeChatResponse(
        session_id=UUID(intake_session.id),
        status=intake_session.status,
        version=intake_session.version,
        assistant_reply=assistant_reply,
        analysis_input=intake_session.analysis_input or "等待补充分析信息。",
        ready_to_analyze=intake_session.ready_to_analyze,
        missing_information=intake_session.missing_information or [],
        structured_context=IntakeStructuredContext.model_validate(
            intake_session.structured_context or {}
        ),
        next_action="PROPOSE_READY"
        if intake_session.status == "READY"
        else "ASK_USER",
        confirmation_request=intake_session.confirmation_request,
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


def _source_text(intake_session: IntakeSession) -> str:
    return "\n".join(
        item.get("content", "")
        for item in (intake_session.messages or [])
        if item.get("role") == "user"
    )


def _standardized_analysis_input(text: str, resolutions: list[dict]) -> str:
    standardized = text
    identities: list[str] = []
    ordered = sorted(
        resolutions,
        key=lambda item: len((item.get("mention") or "").strip()),
        reverse=True,
    )
    for item in ordered:
        mention = (item.get("mention") or "").strip()
        canonical = (item.get("canonical_name") or "").strip()
        if mention and canonical and mention != canonical:
            standardized = standardized.replace(mention, canonical)
        if not canonical:
            continue
        if item.get("entity_type") == "PERSON":
            details = "、".join(
                value
                for value in (item.get("organization"), item.get("title"))
                if value
            )
            identities.append(f"人物：{canonical}{f'（{details}）' if details else ''}")
        elif item.get("entity_type") == "ORGANIZATION":
            identities.append(f"企业：{canonical}")
    if identities:
        standardized = f"{standardized.rstrip()}\n已确认标准身份：{'；'.join(dict.fromkeys(identities))}。"
    return standardized


def _align_resolution_relationships(
    resolutions: list[dict], context: IntakeStructuredContext
) -> list[dict]:
    organizations = [
        item.get("canonical_name")
        for item in resolutions
        if item.get("entity_type") == "ORGANIZATION" and item.get("canonical_name")
    ]
    if len(organizations) != 1:
        return resolutions
    canonical_organization = organizations[0]
    return [
        {
            **item,
            "organization": canonical_organization,
        }
        if item.get("entity_type") == "PERSON"
        and (
            not item.get("organization")
            or item.get("organization") in context.organizations
        )
        else item
        for item in resolutions
    ]


def _standardized_context(context: dict, resolutions: list[dict] | None) -> dict:
    resolutions = resolutions or []
    output = dict(context)
    person_names = {
        item.get("mention"): item.get("canonical_name")
        for item in resolutions
        if item.get("entity_type") == "PERSON" and item.get("canonical_name")
    }
    organization_names = {
        item.get("mention"): item.get("canonical_name")
        for item in resolutions
        if item.get("entity_type") == "ORGANIZATION" and item.get("canonical_name")
    }
    output["people"] = [person_names.get(name, name) for name in output.get("people", [])]
    output["organizations"] = [
        organization_names.get(name, name) for name in output.get("organizations", [])
    ]
    output["people_details"] = [
        {
            **item,
            "name": person_names.get(item.get("name"), item.get("name")),
            "organization": organization_names.get(
                item.get("organization"), item.get("organization")
            ),
            "title": next(
                (
                    resolution.get("title")
                    for resolution in resolutions
                    if resolution.get("entity_type") == "PERSON"
                    and resolution.get("mention") == item.get("name")
                    and resolution.get("title")
                ),
                item.get("title"),
            ),
        }
        for item in output.get("people_details", [])
    ]
    output["entity_resolutions"] = resolutions
    return output


def _merge_resolutions(existing: list[dict], additions: list[dict]) -> list[dict]:
    merged: dict[tuple[str | None, str | None], dict] = {}
    for item in [*existing, *additions]:
        key = (item.get("entity_type"), item.get("mention"))
        merged[key] = item
    return list(merged.values())


def _repair_ready_session(
    intake_session: IntakeSession, repository: IntakeSessionRepository
) -> IntakeSession:
    if intake_session.status != "COLLECTING" or intake_session.confirmation_request:
        return intake_session
    context = IntakeStructuredContext.model_validate(
        intake_session.structured_context or {}
    )
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
    request: IntakeChatRequest, session: Session = Depends(get_session)
) -> IntakeChatResponse:
    repository = IntakeSessionRepository(session)
    session_id = str(request.session_id)
    intake_session = repository.get(session_id)
    incoming_messages = [message.model_dump() for message in request.messages]

    if intake_session is not None:
        if intake_session.status in {"STARTING_ANALYSIS", "ANALYZING"}:
            raise HTTPException(status_code=409, detail="分析任务已创建，不能继续修改采集信息")
        stored_messages = intake_session.messages or []
        if (
            len(stored_messages) == len(incoming_messages) + 1
            and stored_messages[:-1] == incoming_messages
            and stored_messages[-1].get("role") == "assistant"
        ):
            return _chat_response(intake_session)
        if stored_messages and incoming_messages[: len(stored_messages)] != stored_messages:
            raise HTTPException(status_code=409, detail="会话内容已更新，请刷新后重试")

    audio_job = None
    if request.audio_job_id:
        audio_job = session.get(IntakeAudioJob, str(request.audio_job_id))
        if audio_job is None or audio_job.session_id != session_id:
            raise HTTPException(status_code=404, detail="音频转写任务不存在")
        if audio_job.status != "NEEDS_REVIEW":
            raise HTTPException(status_code=409, detail="音频当前不能确认转写")

    try:
        result = intake_agent.respond(request)
    except (LLMUnavailable, LLMCallFailed):
        result = _fallback_result(request)

    source_text = "\n".join(
        message.content for message in request.messages if message.role == "user"
    )
    required_missing = required_missing_information(result, source_text)
    ready = not required_missing
    result.missing_information = required_missing
    next_version = (intake_session.version if intake_session else 0) + 1
    stored_context = with_default_requester_context(
        result.structured_context.model_dump(mode="json")
    )
    confirmation_request = None
    existing_resolutions = (
        (intake_session.structured_context or {}).get("entity_resolutions", [])
        if intake_session
        else []
    )
    if (
        settings.intake_entity_resolution_enabled
        and ready
        and (result.structured_context.people or result.structured_context.organizations)
    ):
        targets = {
            *result.structured_context.people,
            *result.structured_context.organizations,
        }
        confirmed_names = {
            value
            for item in existing_resolutions
            for value in (item.get("mention"), item.get("canonical_name"))
            if value
        }
        if not targets.issubset(confirmed_names):
            unresolved_people = [
                name
                for name in result.structured_context.people
                if name not in confirmed_names
            ]
            unresolved_organizations = [
                name
                for name in result.structured_context.organizations
                if name not in confirmed_names
            ]
            unresolved_targets = {*unresolved_people, *unresolved_organizations}
            candidate_context = result.structured_context.model_copy(
                update={
                    "people": unresolved_people,
                    "organizations": unresolved_organizations,
                    "people_details": [
                        item
                        for item in result.structured_context.people_details
                        if item.name in unresolved_people
                    ],
                    "entity_assessments": [
                        item
                        for item in result.structured_context.entity_assessments
                        if item.mention in unresolved_targets
                    ],
                }
            )
            lookup_internal = getattr(entity_candidates, "lookup_internal", None)
            if callable(lookup_internal):
                resolutions, confirmation = lookup_internal(
                    candidate_context, next_version, source_text
                )
            else:
                resolutions, confirmation = entity_candidates.resolve(
                    candidate_context, next_version, source_text
                )
            apply_automatic = getattr(
                entity_candidates, "apply_automatic_candidates", None
            )
            if callable(apply_automatic):
                resolutions, confirmation = apply_automatic(
                    resolutions,
                    confirmation,
                    settings.llm_web_identity_threshold,
                )
            tool_decision = None
            follow_up = getattr(intake_agent, "follow_up", None)
            if confirmation and settings.intake_react_enabled and callable(follow_up):
                internal_observation = {
                    "tool": "lookup_internal_identity",
                    "resolved_count": len(resolutions),
                    "unresolved": [
                        {
                            "mention": item.mention,
                            "entity_type": item.entity_type,
                            "candidate_count": len(item.candidates),
                        }
                        for item in confirmation.items
                    ],
                    "external_search_allowed": any(
                        len(item.candidates) != 1 for item in confirmation.items
                    ),
                }
                try:
                    tool_decision = follow_up(
                        request, result, internal_observation
                    )
                except (LLMUnavailable, LLMCallFailed):
                    pass

            external_normalizer = getattr(
                intake_agent, "normalize_external_identity", None
            )
            if (
                confirmation
                and tool_decision is not None
                and tool_decision.next_action == "SEARCH_EXTERNAL"
                and any(len(item.candidates) != 1 for item in confirmation.items)
                and callable(external_normalizer)
            ):
                confirmation = entity_candidates.search_key_person_identity_web(
                    candidate_context,
                    confirmation,
                    lambda mentions, pages: external_normalizer(
                        request, mentions, pages
                    ),
                )
                resolutions, confirmation = apply_automatic(
                    resolutions,
                    confirmation,
                    settings.llm_web_identity_threshold,
                )
            resolutions = _merge_resolutions(existing_resolutions, resolutions)
            resolutions = _align_resolution_relationships(
                resolutions, result.structured_context
            )
            stored_context = _standardized_context(stored_context, resolutions)
            follow_up_reply = (
                tool_decision.assistant_reply if tool_decision is not None else None
            )
            if confirmation:
                confirmation_request = confirmation.model_dump(mode="json")
                ready = False
                result.assistant_reply = follow_up_reply or (
                    "请确认人物或企业候选，确认后即可开始分析。"
                )
                result.missing_information = ["人物或企业身份确认"]
            else:
                result.assistant_reply = "关键人身份已经标准化，可以开始分析。"
        else:
            stored_context = _standardized_context(
                stored_context, existing_resolutions
            )

    result.analysis_input = _standardized_analysis_input(
        result.analysis_input,
        stored_context.get("entity_resolutions", []),
    )
    persisted_messages = [
        *incoming_messages,
        {"role": "assistant", "content": result.assistant_reply},
    ]
    values = {
        "status": "READY" if ready else (
            "NEEDS_CONFIRMATION" if confirmation_request else "COLLECTING"
        ),
        "messages": persisted_messages,
        "structured_context": stored_context,
        "missing_information": result.missing_information,
        "confirmation_request": confirmation_request,
        "analysis_input": result.analysis_input,
        "ready_to_analyze": ready,
        "version": next_version,
    }
    if intake_session is None:
        intake_session = repository.add(IntakeSession(id=session_id, **values))
    else:
        intake_session = repository.update(session_id, **values)
    if audio_job is not None:
        audio_job.corrected_transcript = request.messages[-1].content
        audio_job.status = "TRANSCRIBED"
        audio_path = Path(audio_job.audio_path) if audio_job.audio_path else None
        audio_job.audio_path = None
        session.commit()
        if audio_path:
            audio_path.unlink(missing_ok=True)
            audio_path.with_suffix(".wav").unlink(missing_ok=True)
    return _chat_response(intake_session)


@router.get("/{session_id}", response_model=IntakeSessionResponse)
def get_intake_session(
    session_id: UUID, session: Session = Depends(get_session)
) -> IntakeSessionResponse:
    intake_session = IntakeSessionRepository(session).get(str(session_id))
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
    session: Session = Depends(get_session),
) -> IntakeAudioJobResponse:
    if not settings.intake_audio_enabled:
        raise HTTPException(status_code=503, detail="音频预处理当前已关闭")
    intake_session = IntakeSessionRepository(session).get(str(session_id), for_update=True)
    if intake_session is None:
        intake_session = IntakeSession(
            id=str(session_id),
            status="COLLECTING",
            messages=[],
            structured_context={},
            missing_information=[],
            analysis_input="",
        )
        session.add(intake_session)
        session.flush()
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
    session_id: UUID, job_id: UUID, session: Session = Depends(get_session)
) -> IntakeAudioJobResponse:
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
    session_id: UUID, job_id: UUID, session: Session = Depends(get_session)
) -> IntakeAudioJobResponse:
    job = session.get(IntakeAudioJob, str(job_id))
    intake_session = IntakeSessionRepository(session).get(str(session_id))
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
    session: Session = Depends(get_session),
) -> IntakeSessionResponse:
    repository = IntakeSessionRepository(session)
    intake_session = repository.get(str(session_id), for_update=True)
    if intake_session is None:
        raise HTTPException(status_code=404, detail="信息采集会话不存在")
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

    structured_context = with_default_requester_context(
        dict(intake_session.structured_context or {})
    )
    resolutions = _align_resolution_relationships(resolutions, base_context)
    structured_context = _standardized_context(structured_context, resolutions)
    standardized_input = _standardized_analysis_input(
        intake_session.analysis_input, resolutions
    )
    confirmed_names = list(
        dict.fromkeys(
            item.get("canonical_name")
            for item in resolutions
            if item.get("canonical_name")
        )
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
    ready = is_intake_ready(validation_result, source_text) and _has_resolved_entities(
        structured_context
    )
    messages = [
        *(intake_session.messages or []),
        {"role": "assistant", "content": validation_result.assistant_reply},
    ]
    intake_session = repository.update(
        str(session_id),
        status="READY" if ready else "COLLECTING",
        messages=messages,
        structured_context=structured_context,
        analysis_input=standardized_input,
        missing_information=[] if ready else ["分析目标或重点"],
        confirmation_request=None,
        ready_to_analyze=ready,
        version=intake_session.version + 1,
    )
    response = _chat_response(intake_session)
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
    session: Session = Depends(get_session),
) -> TaskCreated:
    repository = IntakeSessionRepository(session)
    intake_session = repository.get(str(session_id), for_update=True)
    if intake_session is None:
        raise HTTPException(status_code=404, detail="信息采集会话不存在")

    if intake_session.research_task_id:
        task = session.get(ResearchTask, intake_session.research_task_id)
        if task is None:
            raise HTTPException(status_code=409, detail="会话关联的分析任务不存在")
        return TaskCreated(task_id=UUID(task.id), input_type=task.input_type)
    if payload.expected_version is not None and payload.expected_version != intake_session.version:
        raise HTTPException(status_code=409, detail="会话版本已更新，请刷新后重试")
    if intake_session.status != "READY" or not intake_session.ready_to_analyze:
        raise HTTPException(status_code=409, detail="信息尚未完整，不能开始分析")
    if intake_session.confirmation_request:
        raise HTTPException(status_code=409, detail="仍有待确认的身份信息")
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
    task = ResearchTask(
        id=task_id,
        input_type="audio" if audio_jobs else "text",
        input_text=intake_session.analysis_input.strip(),
        intake_session_id=intake_session.id,
        input_snapshot=snapshot,
    )
    intake_session.status = "ANALYZING"
    intake_session.research_task_id = task_id
    session.add(task)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = repository.get(str(session_id))
        if existing is None or not existing.research_task_id:
            raise
        task = session.get(ResearchTask, existing.research_task_id)
        if task is None:
            raise
        return TaskCreated(task_id=UUID(task.id), input_type=task.input_type)

    run_research_pipeline.delay(task_id)
    return TaskCreated(task_id=UUID(task.id), input_type=task.input_type)
