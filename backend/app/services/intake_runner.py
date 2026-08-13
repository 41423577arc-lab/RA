import re
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.database import IntakeAudioJob, IntakeSession
from app.schemas.intake import (
    IntakeChatRequest,
    IntakeChatResult,
    IntakeFinalConfirmation,
    IntakeFinalConfirmationResult,
    IntakeStructuredContext,
)
from app.services.intake_agent import IntakeAgent
from app.services.intake_completeness import required_missing_information
from app.services.intake_defaults import with_default_requester_context
from app.services.intake_entity_candidates import IntakeEntityCandidateService
from app.services.intake_field_state import (
    derive_field_states,
    fallback_confirmation_question,
    fields_ready_for_confirmation,
)
from app.services.llm_client import LLMCallFailed, LLMUnavailable
from app.tasks.pipeline import infer_event_type


class IntakeChatConflict(RuntimeError):
    pass


class IntakeAudioJobNotFound(RuntimeError):
    pass


class IntakeAudioJobNotReviewable(RuntimeError):
    pass


def has_resolved_entities(structured_context: dict) -> bool:
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


def source_text(intake_session: IntakeSession) -> str:
    return "\n".join(
        item.get("content", "")
        for item in (intake_session.messages or [])
        if item.get("role") == "user"
    )


def standardized_analysis_input(text: str, resolutions: list[dict]) -> str:
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
            suffix = canonical[len(mention) :] if canonical.startswith(mention) else ""
            pattern = re.escape(mention)
            if suffix:
                pattern += f"(?!{re.escape(suffix)})"
            standardized = re.sub(pattern, canonical, standardized)
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


def align_resolution_relationships(
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
        {**item, "organization": canonical_organization}
        if item.get("entity_type") == "PERSON"
        and (
            not item.get("organization")
            or item.get("organization") in context.organizations
        )
        else item
        for item in resolutions
    ]


def standardized_context(context: dict, resolutions: list[dict] | None) -> dict:
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


def with_field_states(context: dict, *, final_confirmed: bool = False) -> dict:
    output = dict(context)
    structured = IntakeStructuredContext.model_validate(output)
    output["field_states"] = {
        name: state.model_dump(mode="json")
        for name, state in derive_field_states(
            structured, final_confirmed=final_confirmed
        ).items()
    }
    return output


def prepare_final_confirmation(
    agent: IntakeAgent,
    request: IntakeChatRequest,
    context: dict,
    analysis_input: str,
    version: int,
) -> tuple[dict, IntakeFinalConfirmation | None]:
    if not context.get("event_type"):
        context = {**context, "event_type": infer_event_type(analysis_input)}
    context = with_field_states(context)
    structured = IntakeStructuredContext.model_validate(context)
    if not fields_ready_for_confirmation(structured.field_states):
        return context, None
    fallback = IntakeFinalConfirmationResult(
        question=fallback_confirmation_question(structured)
    )
    summarize = getattr(agent, "summarize_for_confirmation", None)
    if callable(summarize):
        try:
            summary = summarize(request, structured, analysis_input)
        except (LLMUnavailable, LLMCallFailed):
            summary = fallback
    else:
        summary = fallback
    final_confirmation = IntakeFinalConfirmation(
        version=version,
        question=summary.question,
        status="PENDING",
    )
    context["final_confirmation"] = final_confirmation.model_dump(mode="json")
    return context, final_confirmation


def merge_resolutions(existing: list[dict], additions: list[dict]) -> list[dict]:
    merged: dict[tuple[str | None, str | None], dict] = {}
    for item in [*existing, *additions]:
        key = (item.get("entity_type"), item.get("mention"))
        merged[key] = item
    return list(merged.values())


class IntakeRunner:
    def __init__(
        self,
        repository,
        session: Session,
        agent: IntakeAgent,
        entity_candidates: IntakeEntityCandidateService,
        activity,
        settings,
    ):
        self.repository = repository
        self.session = session
        self.agent = agent
        self.entity_candidates = entity_candidates
        self.activity = activity
        self.settings = settings

    def run_chat(self, request: IntakeChatRequest) -> IntakeSession:
        session_id = str(request.session_id)
        self.activity.update(session_id, "THINKING", "大模型正在理解当前对话")
        intake_session = self.repository.get(session_id)
        incoming_messages = [message.model_dump() for message in request.messages]

        if intake_session is not None:
            if intake_session.status in {"STARTING_ANALYSIS", "ANALYZING"}:
                raise IntakeChatConflict("分析任务已创建，不能继续修改采集信息")
            stored_messages = intake_session.messages or []
            if (
                len(stored_messages) == len(incoming_messages) + 1
                and stored_messages[:-1] == incoming_messages
                and stored_messages[-1].get("role") == "assistant"
            ):
                self.activity.update(
                    session_id, "COMPLETED", "已返回当前对话结果", active=False
                )
                return intake_session
            if stored_messages and incoming_messages[: len(stored_messages)] != stored_messages:
                raise IntakeChatConflict("会话内容已更新，请刷新后重试")

        audio_job = None
        if request.audio_job_id:
            audio_job = self.session.get(IntakeAudioJob, str(request.audio_job_id))
            if audio_job is None or audio_job.session_id != session_id:
                raise IntakeAudioJobNotFound("音频转写任务不存在")
            if audio_job.status != "NEEDS_REVIEW":
                raise IntakeAudioJobNotReviewable("音频当前不能确认转写")

        try:
            result = self.agent.respond(request)
        except (LLMUnavailable, LLMCallFailed):
            result = self._fallback_result(request)

        self.activity.update(session_id, "CHECKING_CONTEXT", "正在检查关键人信息")
        source = "\n".join(
            message.content for message in request.messages if message.role == "user"
        )
        required_missing = required_missing_information(result, source)
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
            self.settings.intake_entity_resolution_enabled
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
                resolutions, confirmation = self._lookup_internal(
                    session_id, candidate_context, next_version, source
                )
                apply_automatic = getattr(
                    self.entity_candidates, "apply_automatic_candidates", None
                )
                if callable(apply_automatic):
                    resolutions, confirmation = apply_automatic(
                        resolutions,
                        confirmation,
                        self.settings.llm_web_identity_threshold,
                    )
                tool_decision = self._follow_up(
                    request, result, confirmation, resolutions, session_id
                )
                external_attempted = False
                external_normalizer = getattr(
                    self.agent, "normalize_external_identity", None
                )
                if (
                    confirmation
                    and any(len(item.candidates) != 1 for item in confirmation.items)
                    and callable(external_normalizer)
                ):
                    external_attempted = True
                    confirmation = self._lookup_external(
                        request,
                        session_id,
                        candidate_context,
                        confirmation,
                        external_normalizer,
                    )
                    resolutions, confirmation = apply_automatic(
                        resolutions,
                        confirmation,
                        self.settings.llm_web_identity_threshold,
                    )
                    self.activity.update(
                        session_id,
                        "PROCESSING_TOOL_RESULT",
                        "大模型正在整理联网身份候选",
                        tool_name="search_key_person_identity_web",
                    )
                    self._record_external_result(session_id, resolutions, confirmation)
                resolutions = merge_resolutions(existing_resolutions, resolutions)
                resolutions = align_resolution_relationships(
                    resolutions, result.structured_context
                )
                stored_context = standardized_context(stored_context, resolutions)
                follow_up_reply = (
                    tool_decision.assistant_reply if tool_decision is not None else None
                )
                if confirmation:
                    confirmation_request = confirmation.model_dump(mode="json")
                    ready = False
                    candidate_count = sum(
                        len(item.candidates) for item in confirmation.items
                    )
                    if external_attempted and candidate_count:
                        result.assistant_reply = (
                            "内部与联网检索仍未能唯一确定全部关键人身份，"
                            "请确认候选或手工填写缺失信息。"
                        )
                    elif external_attempted:
                        unresolved_mentions = "、".join(
                            item.mention for item in confirmation.items
                        )
                        result.assistant_reply = (
                            f"内部与联网检索后，仍无法可靠确定：{unresolved_mentions}。"
                            "请手工填写以下缺失信息。"
                        )
                    else:
                        result.assistant_reply = follow_up_reply or (
                            "请确认人物或企业候选，确认后即可开始分析。"
                        )
                    result.missing_information = ["人物或企业身份确认"]
                else:
                    result.assistant_reply = "关键人身份已经标准化，可以开始分析。"
            else:
                stored_context = standardized_context(
                    stored_context, existing_resolutions
                )

        result.analysis_input = standardized_analysis_input(
            result.analysis_input,
            stored_context.get("entity_resolutions", []),
        )
        stored_context["final_confirmation"] = None
        stored_context = with_field_states(stored_context)
        final_confirmation = None
        identities_ready = (
            not self.settings.intake_entity_resolution_enabled
            or has_resolved_entities(stored_context)
        )
        if ready and not confirmation_request and identities_ready:
            stored_context, final_confirmation = prepare_final_confirmation(
                self.agent,
                request,
                stored_context,
                result.analysis_input,
                next_version,
            )
            if final_confirmation:
                result.assistant_reply = final_confirmation.question
                result.ready_to_analyze = False
                result.missing_information = []
            else:
                result.assistant_reply = "当前字段仍有待补全，请继续提供相关信息。"
                result.ready_to_analyze = False
            ready = False

        values = {
            "status": (
                "NEEDS_CONFIRMATION"
                if confirmation_request
                else "AWAITING_FINAL_CONFIRMATION"
                if final_confirmation
                else "READY"
                if ready
                else "COLLECTING"
            ),
            "messages": [
                *incoming_messages,
                {"role": "assistant", "content": result.assistant_reply},
            ],
            "structured_context": stored_context,
            "missing_information": result.missing_information,
            "confirmation_request": confirmation_request,
            "analysis_input": result.analysis_input,
            "ready_to_analyze": ready,
            "version": next_version,
        }
        if intake_session is None:
            intake_session = self.repository.add(IntakeSession(id=session_id, **values))
        else:
            intake_session = self.repository.update(session_id, **values)
        self._complete_audio_job(audio_job, request)
        self.activity.update(
            session_id, "COMPLETED", "本轮对话处理完成", active=False
        )
        return intake_session

    @staticmethod
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

    def _lookup_internal(self, session_id, context, version, source):
        lookup_internal = getattr(self.entity_candidates, "lookup_internal", None)
        if not callable(lookup_internal):
            return self.entity_candidates.resolve(context, version, source)
        arguments = {
            "person_mention": context.people[0] if context.people else None,
            "organization_mention": context.organizations[0]
            if context.organizations
            else None,
        }
        self.repository.log_execution_event(
            session_id,
            event_type="TOOL_REQUEST",
            node_name="mcp.find_entity_candidates",
            status="RUNNING",
            title="查询内部身份候选",
            detail="调用 MCP 工具 find_entity_candidates。",
            payload={"tool": "find_entity_candidates", "arguments": arguments},
        )
        self.activity.update(
            session_id,
            "CALLING_TOOL",
            "正在查询内部身份候选",
            tool_name="lookup_internal_identity",
        )
        resolutions, confirmation = lookup_internal(context, version, source)
        self.repository.log_execution_event(
            session_id,
            event_type="TOOL_RESPONSE",
            node_name="mcp.find_entity_candidates",
            status="SUCCESS",
            title="内部身份查询完成",
            detail=f"自动解析 {len(resolutions)} 个身份，仍有 {len(confirmation.items) if confirmation else 0} 个待确认项。",
            payload={
                "resolutions": resolutions,
                "confirmation": confirmation.model_dump(mode="json")
                if confirmation
                else None,
            },
        )
        return resolutions, confirmation

    def _follow_up(self, request, result, confirmation, resolutions, session_id):
        follow_up = getattr(self.agent, "follow_up", None)
        if not (
            confirmation
            and self.settings.intake_react_enabled
            and callable(follow_up)
        ):
            return None
        self.activity.update(
            session_id,
            "PROCESSING_TOOL_RESULT",
            "大模型正在判断内部查询结果",
            tool_name="lookup_internal_identity",
        )
        observation = {
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
            return follow_up(request, result, observation)
        except (LLMUnavailable, LLMCallFailed):
            return None

    def _lookup_external(
        self,
        request,
        session_id,
        context,
        confirmation,
        external_normalizer,
    ):
        person = context.people[0] if context.people else None
        organization = context.organizations[0] if context.organizations else None
        quoted_targets = " ".join(
            f'"{value}"' for value in (person, organization) if value
        )
        identity_query = f"{quoted_targets} 完整姓名 企业全称 职位".strip()
        self.repository.log_execution_event(
            session_id,
            event_type="SEARCH_REQUEST",
            node_name="intake_identity_web",
            status="RUNNING",
            title="联网补全关键人身份",
            detail="向 Tavily 提交身份补全搜索指令。",
            payload={
                "provider": "Tavily",
                "request": {
                    "query": identity_query,
                    "search_depth": "basic",
                    "max_results": 5,
                },
            },
        )
        self.activity.update(
            session_id,
            "CALLING_TOOL",
            "正在联网补全关键人身份",
            tool_name="search_key_person_identity_web",
        )
        confirmation = self.entity_candidates.search_key_person_identity_web(
            context,
            confirmation,
            lambda mentions, pages: external_normalizer(request, mentions, pages),
        )
        return confirmation

    def _record_external_result(self, session_id, resolutions, confirmation):
        self.repository.log_execution_event(
            session_id,
            event_type="SEARCH_RESPONSE",
            node_name="intake_identity_web",
            status="SUCCESS",
            title="联网身份补全完成",
            detail=f"当前仍有 {len(confirmation.items) if confirmation else 0} 个待确认项。",
            payload={
                "resolutions": resolutions,
                "confirmation": confirmation.model_dump(mode="json")
                if confirmation
                else None,
            },
        )

    def _complete_audio_job(self, audio_job, request):
        if audio_job is None:
            return
        audio_job.corrected_transcript = request.messages[-1].content
        audio_job.status = "TRANSCRIBED"
        audio_path = Path(audio_job.audio_path) if audio_job.audio_path else None
        audio_job.audio_path = None
        self.session.commit()
        if audio_path:
            audio_path.unlink(missing_ok=True)
            audio_path.with_suffix(".wav").unlink(missing_ok=True)
