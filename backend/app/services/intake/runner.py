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
from app.schemas.task import ConfirmationRequest
from app.services.intake.agent import IntakeAgent
from app.services.intake.agent_loop import (
    AgentState,
    MechanicalIntakeAgentLoop,
    ToolObservation,
)
from app.services.intake.completeness import required_missing_information
from app.services.intake.defaults import with_default_requester_context
from app.services.intake.entity_candidates import IntakeEntityCandidateService
from app.services.intake.field_state import (
    derive_field_states,
    fallback_confirmation_question,
    fields_ready_for_confirmation,
)
from app.services.intake.identity_loop import IntakeIdentityLoop, IntakeIdentityLoopResult
from app.services.intake.query_executor import IntakeQueryExecutor
from app.services.intake.state_reducer import IntakeStateReducer
from app.services.integrations.llm_client import LLMCallFailed, LLMUnavailable
from app.tasks.pipeline import infer_event_type


class IntakeChatConflict(RuntimeError):
    pass


class IntakeAudioJobNotFound(RuntimeError):
    pass


class IntakeAudioJobNotReviewable(RuntimeError):
    pass


class _RunnerDecisionProvider:
    """把当前请求绑定到统一 Intake Agent，并记录每次决策。"""

    def __init__(self, runner, request: IntakeChatRequest, session_id: str):
        self.runner = runner
        self.request = request
        self.session_id = session_id

    def decide(self, state: AgentState):
        self.runner.activity.update(
            self.session_id,
            "THINKING",
            "大模型正在根据当前状态选择处理 Skill",
        )
        turn = self.runner.agent.decide_turn(self.request, state)
        self.runner.repository.log_execution_event(
            self.session_id,
            event_type="AGENT_ACTION",
            node_name="intake_agent_loop_v2",
            status="SUCCESS",
            title="Intake Agent 已选择下一步动作",
            detail=f"Skill={turn.skill}，Action={turn.next_action}。",
            payload={"turn": turn.model_dump(mode="json")},
        )
        return turn


class _RunnerCandidateBackend:
    """让新执行器复用 Runner 已有的工具事件和 Activity 接线。"""

    def __init__(
        self,
        runner,
        request: IntakeChatRequest,
        session_id: str,
    ):
        self.runner = runner
        self.request = request
        self.session_id = session_id
        self.public_result_pending = False

    def lookup_internal(
        self,
        context,
        version,
        source_text,
        *,
        raise_on_error=False,
    ):
        return self.runner._lookup_internal(
            self.session_id,
            context,
            version,
            source_text,
            raise_on_error=raise_on_error,
        )

    def search_key_person_identity_web(
        self,
        context,
        confirmation,
        external_normalizer,
        *,
        raise_on_error=False,
    ):
        result = self.runner._lookup_external(
            self.request,
            self.session_id,
            context,
            confirmation,
            external_normalizer,
            raise_on_error=raise_on_error,
        )
        self.public_result_pending = True
        return result

    def apply_automatic_candidates(
        self,
        resolutions,
        confirmation,
        threshold=0.8,
    ):
        result = self.runner.entity_candidates.apply_automatic_candidates(
            resolutions,
            confirmation,
            threshold,
        )
        if self.public_result_pending:
            self.runner._record_external_result(
                self.session_id,
                result[0],
                result[1],
            )
            self.public_result_pending = False
        return result


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
    output["organizations"] = list(
        dict.fromkeys(
            organization_names.get(name, name)
            for name in output.get("organizations", [])
        )
    )
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
    structured = IntakeStructuredContext.model_validate(context)
    gate_states = derive_field_states(structured)
    if not fields_ready_for_confirmation(gate_states):
        return context, None
    if structured.next_action is None:
        context = {
            **context,
            "field_states": {
                name: state.model_dump(mode="json")
                for name, state in gate_states.items()
            },
        }
        structured = IntakeStructuredContext.model_validate(context)
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
        try:
            return self._run_chat(request)
        except Exception:
            self.activity.update(
                session_id,
                "FAILED",
                "本轮对话处理失败，请稍后重试。",
                active=False,
            )
            raise

    def _run_chat(self, request: IntakeChatRequest) -> IntakeSession:
        session_id = str(request.session_id)
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

        source = "\n".join(
            message.content for message in request.messages if message.role == "user"
        )
        v2_capable = bool(
            getattr(self.settings, "intake_agent_v2_enabled", False)
        ) and callable(getattr(self.agent, "decide_turn", None))
        if v2_capable:
            result = self._v2_seed_result(
                source,
                previous_context=IntakeStructuredContext.model_validate(
                    intake_session.structured_context or {}
                )
                if intake_session
                else None,
                previous_analysis_input=intake_session.analysis_input
                if intake_session
                else None,
            )
        else:
            try:
                result = self.agent.respond(request)
            except (LLMUnavailable, LLMCallFailed):
                result = self._fallback_result(
                    request,
                    previous_context=IntakeStructuredContext.model_validate(
                        intake_session.structured_context or {}
                    )
                    if intake_session
                    else None,
                    previous_analysis_input=intake_session.analysis_input
                    if intake_session
                    else None,
                )

        self.activity.update(session_id, "CHECKING_CONTEXT", "正在检查关键人信息")
        required_missing = required_missing_information(result, source)
        ready = v2_capable or not required_missing
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
        identity_loop_used = False
        legacy_capable = callable(
            getattr(self.agent, "initialize_context", None)
        ) and callable(getattr(self.agent, "update_context", None))

        if (
            self.settings.intake_entity_resolution_enabled
            and self.settings.intake_react_enabled
            and (
                v2_capable
                or (
                    ready
                    and (
                        result.structured_context.people
                        or result.structured_context.organizations
                    )
                    and legacy_capable
                )
            )
        ):
            if intake_session is None:
                intake_session = self.repository.add(
                    IntakeSession(
                        id=session_id,
                        status="COLLECTING",
                        messages=incoming_messages,
                        structured_context=stored_context,
                        missing_information=result.missing_information,
                        confirmation_request=None,
                        analysis_input=result.analysis_input,
                        ready_to_analyze=False,
                        version=0,
                    )
                )
            loop_result = self._run_identity_loop(
                request,
                result.structured_context,
                intake_session,
                next_version,
                source,
            )
            identity_loop_used = True
            resolutions = align_resolution_relationships(
                [item.model_dump(mode="json") for item in loop_result.resolutions],
                loop_result.context,
            )
            stored_context = standardized_context(
                with_default_requester_context(
                    loop_result.context.model_dump(mode="json")
                ),
                resolutions,
            )
            confirmation_request = (
                loop_result.confirmation.model_dump(mode="json")
                if loop_result.confirmation
                else None
            )
            ready = (
                loop_result.stop_reason == "READY"
                and bool(
                    stored_context.get("people")
                    or stored_context.get("organizations")
                )
                and has_resolved_entities(stored_context)
            )
            if ready:
                result.assistant_reply = "关键人身份已经标准化，可以继续确认本次分析信息。"
                result.missing_information = []
            else:
                result.assistant_reply = loop_result.context.user_question or (
                    "请确认目标人物或企业的完整名称、职位或所属关系。"
                )
                result.missing_information = ["人物或企业身份确认"]

        if (
            self.settings.intake_entity_resolution_enabled
            and ready
            and (result.structured_context.people or result.structured_context.organizations)
            and not identity_loop_used
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
        if not identity_loop_used:
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

    def _run_identity_loop(
        self,
        request: IntakeChatRequest,
        extracted_context: IntakeStructuredContext,
        intake_session: IntakeSession | None,
        version: int,
        source: str,
    ):
        if (
            getattr(self.settings, "intake_agent_v2_enabled", False)
            and callable(getattr(self.agent, "decide_turn", None))
        ):
            return self._run_agent_loop_v2(
                request,
                extracted_context,
                intake_session,
                version,
                source,
            )
        return self._run_legacy_identity_loop(
            request,
            extracted_context,
            intake_session,
            version,
            source,
        )

    def _run_agent_loop_v2(
        self,
        request: IntakeChatRequest,
        extracted_context: IntakeStructuredContext,
        intake_session: IntakeSession | None,
        version: int,
        source: str,
    ) -> IntakeIdentityLoopResult:
        session_id = str(request.session_id)
        previous_context = (
            IntakeStructuredContext.model_validate(
                intake_session.structured_context or {}
            )
            if intake_session and intake_session.version > 0
            else None
        )
        initial_context = previous_context or extracted_context.model_copy(
            update={
                "entity_resolutions": [],
                "tool_attempts": [],
                "final_confirmation": None,
            },
            deep=True,
        )
        previous_confirmation = (
            ConfirmationRequest.model_validate(intake_session.confirmation_request)
            if intake_session and intake_session.confirmation_request
            else None
        )

        def checkpoint(
            context: IntakeStructuredContext,
            confirmation: ConfirmationRequest | None,
        ) -> None:
            if intake_session is None:
                return
            self.repository.update(
                session_id,
                structured_context=with_default_requester_context(
                    context.model_dump(mode="json")
                ),
                confirmation_request=confirmation.model_dump(mode="json")
                if confirmation
                else None,
            )

        def record_observation(observation: ToolObservation) -> None:
            self.activity.update(
                session_id,
                "PROCESSING_TOOL_RESULT",
                "大模型正在根据工具结果更新身份状态",
                tool_name=(
                    "lookup_internal_identity"
                    if observation.action == "SEARCH_INTERNAL"
                    else "search_key_person_identity_web"
                ),
            )
            self.repository.log_execution_event(
                session_id,
                event_type="AGENT_OBSERVATION",
                node_name="intake_agent_loop_v2",
                status=observation.technical_status,
                title="Intake Agent 已接收工具 Observation",
                detail=observation.error or observation.summary,
                payload={
                    "observation": observation.model_dump(mode="json")
                },
            )

        candidate_backend = _RunnerCandidateBackend(self, request, session_id)
        loop = MechanicalIntakeAgentLoop(
            _RunnerDecisionProvider(self, request, session_id),
            IntakeQueryExecutor(
                candidate_backend,
                automatic_threshold=getattr(
                    self.settings,
                    "llm_web_identity_threshold",
                    0.8,
                ),
            ),
            IntakeStateReducer,
            max_loops=getattr(self.settings, "agent_max_loops", 8),
            max_tool_calls=getattr(self.settings, "agent_max_tool_calls", 4),
            max_repeated_actions=getattr(
                self.settings,
                "agent_max_repeated_actions",
                2,
            ),
            on_observation=record_observation,
        )
        result = loop.run(
            initial_context,
            version=version,
            source_text=source,
            hard_gate=lambda context: bool(
                context.people or context.organizations
            )
            and has_resolved_entities(context.model_dump(mode="json")),
            confirmation=previous_confirmation,
            external_normalizer=getattr(
                self.agent,
                "normalize_external_identity",
                None,
            ),
            checkpoint=checkpoint,
        )
        return IntakeIdentityLoopResult(
            context=result.state.context,
            resolutions=tuple(result.state.context.entity_resolutions),
            confirmation=result.confirmation,
            stop_reason=result.stop_reason,
            tool_calls=result.tool_calls,
        )

    def _run_legacy_identity_loop(
        self,
        request: IntakeChatRequest,
        extracted_context: IntakeStructuredContext,
        intake_session: IntakeSession | None,
        version: int,
        source: str,
    ):
        session_id = str(request.session_id)
        previous_context = (
            IntakeStructuredContext.model_validate(
                intake_session.structured_context or {}
            )
            if intake_session
            else None
        )
        apply_automatic = getattr(
            self.entity_candidates, "apply_automatic_candidates", None
        )
        threshold = getattr(self.settings, "llm_web_identity_threshold", 0.8)

        def lookup_internal(context: IntakeStructuredContext):
            resolutions, confirmation = self._lookup_internal(
                session_id, context, version, source, raise_on_error=True
            )
            if callable(apply_automatic):
                return apply_automatic(resolutions, confirmation, threshold)
            return resolutions, confirmation

        def lookup_public(
            context: IntakeStructuredContext, confirmation
        ):
            external_normalizer = getattr(
                self.agent, "normalize_external_identity", None
            )
            if not callable(external_normalizer):
                raise RuntimeError("Intake Agent 不支持公网身份候选标准化")
            updated_confirmation = self._lookup_external(
                request,
                session_id,
                context,
                confirmation,
                external_normalizer,
                raise_on_error=True,
            )
            if callable(apply_automatic):
                resolutions, updated_confirmation = apply_automatic(
                    [], updated_confirmation, threshold
                )
            else:
                resolutions = []
            self._record_external_result(
                session_id, resolutions, updated_confirmation
            )
            return resolutions, updated_confirmation

        def checkpoint(
            context: IntakeStructuredContext, confirmation
        ) -> None:
            if intake_session is None:
                return
            self.repository.update(
                session_id,
                structured_context=with_default_requester_context(
                    context.model_dump(mode="json")
                ),
                confirmation_request=confirmation.model_dump(mode="json")
                if confirmation
                else None,
            )

        loop = IntakeIdentityLoop(
            self.agent,
            self.repository,
            session_id,
            max_loops=getattr(self.settings, "agent_max_loops", 8),
            max_tool_calls=getattr(self.settings, "agent_max_tool_calls", 4),
            max_repeated_actions=getattr(
                self.settings, "agent_max_repeated_actions", 2
            ),
        )
        return loop.run(
            request,
            extracted_context,
            previous_context,
            ConfirmationRequest.model_validate(intake_session.confirmation_request)
            if intake_session and intake_session.confirmation_request
            else None,
            lookup_internal=lookup_internal,
            lookup_public=lookup_public,
            hard_gate=lambda context: bool(context.people or context.organizations)
            and has_resolved_entities(context.model_dump(mode="json")),
            checkpoint=checkpoint,
        )

    @staticmethod
    def _v2_seed_result(
        source: str,
        *,
        previous_context: IntakeStructuredContext | None = None,
        previous_analysis_input: str | None = None,
    ) -> IntakeChatResult:
        return IntakeChatResult(
            assistant_reply="正在处理本轮信息。",
            analysis_input=(previous_analysis_input or source or "请补充本次分析信息。")[
                -10_000:
            ],
            ready_to_analyze=False,
            missing_information=[],
            structured_context=(
                previous_context.model_copy(deep=True)
                if previous_context is not None
                else IntakeStructuredContext()
            ),
            next_action="ASK_USER",
        )

    @staticmethod
    def _fallback_result(
        request: IntakeChatRequest,
        *,
        previous_context: IntakeStructuredContext | None = None,
        previous_analysis_input: str | None = None,
    ) -> IntakeChatResult:
        user_text = "\n".join(
            message.content for message in request.messages if message.role == "user"
        ).strip()
        return IntakeChatResult(
            assistant_reply="信息采集助手暂时不可用。请补充涉及的人物或企业，以及希望分析的事项。",
            analysis_input=(
                previous_analysis_input or user_text or "请补充本次分析信息。"
            )[-10_000:],
            ready_to_analyze=False,
            missing_information=["人物、企业或项目", "希望分析或推动的事项"],
            structured_context=(
                previous_context.model_copy(deep=True)
                if previous_context is not None
                else IntakeStructuredContext()
            ),
        )

    def _lookup_internal(
        self, session_id, context, version, source, *, raise_on_error=False
    ):
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
        resolutions, confirmation = (
            lookup_internal(
                context, version, source, raise_on_error=True
            )
            if raise_on_error
            else lookup_internal(context, version, source)
        )
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
        *,
        raise_on_error=False,
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

        def normalize(mentions, pages):
            return external_normalizer(request, mentions, pages)

        confirmation = (
            self.entity_candidates.search_key_person_identity_web(
                context,
                confirmation,
                normalize,
                raise_on_error=True,
            )
            if raise_on_error
            else self.entity_candidates.search_key_person_identity_web(
                context,
                confirmation,
                normalize,
            )
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
