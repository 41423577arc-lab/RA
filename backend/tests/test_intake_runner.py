from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models.database import IntakeSession
from app.schemas.intake import (
    IntakeChatRequest,
    IntakeChatResult,
    IntakeEntityResolution,
    IntakeStructuredContext,
)
from app.services.intake.identity_loop import IntakeIdentityLoopResult
from app.services.intake.runner import IntakeChatConflict, IntakeRunner
from app.services.integrations.llm_client import LLMCallFailed


class RecordingRepository:
    def __init__(self, intake_session=None):
        self.intake_session = intake_session
        self.added = []
        self.updated = []

    def get(self, session_id):
        if self.intake_session and self.intake_session.id == session_id:
            return self.intake_session
        return None

    def add(self, intake_session):
        self.intake_session = intake_session
        self.added.append(intake_session)
        return intake_session

    def update(self, session_id, **values):
        self.updated.append((session_id, values))
        for key, value in values.items():
            setattr(self.intake_session, key, value)
        return self.intake_session

    def log_execution_event(self, *_args, **_kwargs):
        raise AssertionError("This scenario must not call an identity tool")


class RecordingActivity:
    def __init__(self):
        self.events = []

    def update(self, session_id, phase, detail, **values):
        self.events.append((session_id, phase, detail, values))


class FailingAgent:
    def __init__(self):
        self.calls = 0

    def respond(self, _request):
        self.calls += 1
        raise LLMCallFailed("unavailable")


class MustNotRunAgent:
    def respond(self, _request):
        raise AssertionError("A replay must not invoke the model")


class StructuredAgent:
    def respond(self, _request):
        context = IntakeStructuredContext(
            people=["刘希川"],
            organizations=["中国建筑第二工程局有限公司"],
            next_action="READY",
        )
        return IntakeChatResult(
            assistant_reply="信息已记录。",
            analysis_input="今晚和中国建筑第二工程局有限公司刘希川吃饭",
            ready_to_analyze=True,
            structured_context=context,
            next_action="READY",
        )

    def initialize_context(self, *_args, **_kwargs):
        raise AssertionError("The test replaces the identity loop")

    def update_context(self, *_args, **_kwargs):
        raise AssertionError("The test replaces the identity loop")


def build_runner(repository, agent, activity):
    return IntakeRunner(
        repository=repository,
        session=SimpleNamespace(get=lambda *_: None, commit=lambda: None),
        agent=agent,
        entity_candidates=SimpleNamespace(),
        activity=activity,
        settings=SimpleNamespace(
            intake_entity_resolution_enabled=True,
            intake_react_enabled=True,
            llm_web_identity_threshold=0.8,
        ),
    )


def test_runner_preserves_llm_failure_fallback_version_and_activity() -> None:
    repository = RecordingRepository()
    activity = RecordingActivity()
    agent = FailingAgent()
    request = IntakeChatRequest(
        session_id=uuid4(),
        messages=[{"role": "user", "content": "请帮我准备一次客户会面"}],
    )

    result = build_runner(repository, agent, activity).run_chat(request)

    assert agent.calls == 1
    assert result.status == "COLLECTING"
    assert result.version == 1
    assert result.ready_to_analyze is False
    assert result.missing_information == ["候选人姓名或候选企业"]
    assert result.messages[-1]["content"].startswith("信息采集助手暂时不可用")
    assert [event[1] for event in activity.events] == [
        "THINKING",
        "CHECKING_CONTEXT",
        "COMPLETED",
    ]


def test_runner_preserves_existing_context_when_llm_fails() -> None:
    session_id = str(uuid4())
    existing_context = IntakeStructuredContext(
        people=["刘希川"],
        organizations=["中国建筑第二工程局有限公司"],
        entity_resolutions=[
            IntakeEntityResolution(
                entity_type="PERSON",
                canonical_name="刘希川",
                mention="刘希川",
                confirmed_by="USER_INPUT",
            ),
            IntakeEntityResolution(
                entity_type="ORGANIZATION",
                canonical_name="中国建筑第二工程局有限公司",
                mention="中国建筑第二工程局有限公司",
                confirmed_by="USER_INPUT",
            ),
        ],
    )
    previous_messages = [
        {"role": "user", "content": "今晚和中国建筑第二工程局有限公司刘希川吃饭"},
        {"role": "assistant", "content": "信息已记录，请继续。"},
    ]
    intake_session = IntakeSession(
        id=session_id,
        status="COLLECTING",
        messages=previous_messages,
        structured_context=existing_context.model_dump(mode="json"),
        missing_information=[],
        analysis_input="今晚和中国建筑第二工程局有限公司刘希川吃饭",
        ready_to_analyze=False,
        version=2,
    )
    repository = RecordingRepository(intake_session)
    activity = RecordingActivity()
    request = IntakeChatRequest(
        session_id=session_id,
        messages=[*previous_messages, {"role": "user", "content": "请继续处理"}],
    )

    result = build_runner(repository, FailingAgent(), activity).run_chat(request)

    restored = IntakeStructuredContext.model_validate(result.structured_context)
    assert restored.people == ["刘希川"]
    assert restored.organizations == ["中国建筑第二工程局有限公司"]
    assert {item.canonical_name for item in restored.entity_resolutions} == {
        "刘希川",
        "中国建筑第二工程局有限公司",
    }
    assert result.version == 3
    assert activity.events[-1][1] == "COMPLETED"
    assert activity.events[-1][3]["active"] is False


def test_runner_marks_activity_failed_when_identity_loop_raises() -> None:
    activity = RecordingActivity()
    runner = build_runner(RecordingRepository(), StructuredAgent(), activity)

    def fail_loop(*_args, **_kwargs):
        raise RuntimeError("unexpected loop failure")

    runner._run_identity_loop = fail_loop
    request = IntakeChatRequest(
        session_id=uuid4(),
        messages=[
            {
                "role": "user",
                "content": "今晚和中国建筑第二工程局有限公司刘希川吃饭",
            }
        ],
    )

    with pytest.raises(RuntimeError, match="unexpected loop failure"):
        runner.run_chat(request)

    assert [event[1] for event in activity.events] == [
        "THINKING",
        "CHECKING_CONTEXT",
        "FAILED",
    ]
    assert activity.events[-1][3]["active"] is False


def test_runner_serializes_typed_loop_resolutions_at_json_boundary() -> None:
    repository = RecordingRepository()
    activity = RecordingActivity()
    agent = StructuredAgent()
    runner = build_runner(repository, agent, activity)
    context = agent.respond(None).structured_context
    resolutions = (
        IntakeEntityResolution(
            entity_type="PERSON",
            canonical_name="刘希川",
            mention="刘希川",
            organization="中国建筑第二工程局有限公司",
            confirmed_by="USER_INPUT",
        ),
        IntakeEntityResolution(
            entity_type="ORGANIZATION",
            canonical_name="中国建筑第二工程局有限公司",
            mention="中国建筑第二工程局有限公司",
            confirmed_by="USER_INPUT",
        ),
    )
    runner._run_identity_loop = lambda *_args, **_kwargs: IntakeIdentityLoopResult(
        context=context,
        resolutions=resolutions,
        confirmation=None,
        stop_reason="READY",
        tool_calls=0,
    )
    request = IntakeChatRequest(
        session_id=uuid4(),
        messages=[
            {
                "role": "user",
                "content": "今晚和中国建筑第二工程局有限公司刘希川吃饭",
            }
        ],
    )

    result = runner.run_chat(request)

    assert result.status == "AWAITING_FINAL_CONFIRMATION"
    assert all(
        isinstance(item, dict)
        for item in result.structured_context["entity_resolutions"]
    )


def test_runner_replays_duplicate_without_model_or_version_change() -> None:
    session_id = str(uuid4())
    incoming = [{"role": "user", "content": "请帮我准备一次客户会面"}]
    intake_session = IntakeSession(
        id=session_id,
        status="COLLECTING",
        messages=[*incoming, {"role": "assistant", "content": "请提供目标人物或企业。"}],
        structured_context={},
        missing_information=["候选人姓名或候选企业"],
        analysis_input="请帮我准备一次客户会面",
        ready_to_analyze=False,
        version=3,
    )
    repository = RecordingRepository(intake_session)
    activity = RecordingActivity()
    request = IntakeChatRequest(session_id=session_id, messages=incoming)

    result = build_runner(repository, MustNotRunAgent(), activity).run_chat(request)

    assert result is intake_session
    assert result.version == 3
    assert repository.updated == []
    assert [event[1] for event in activity.events] == ["THINKING", "COMPLETED"]
    assert activity.events[-1][2] == "已返回当前对话结果"


def test_runner_rejects_message_prefix_conflict_before_model_call() -> None:
    session_id = str(uuid4())
    intake_session = IntakeSession(
        id=session_id,
        status="COLLECTING",
        messages=[
            {"role": "user", "content": "原始消息"},
            {"role": "assistant", "content": "请继续补充。"},
        ],
        structured_context={},
        missing_information=[],
        analysis_input="原始消息",
        ready_to_analyze=False,
        version=2,
    )
    repository = RecordingRepository(intake_session)
    request = IntakeChatRequest(
        session_id=session_id,
        messages=[{"role": "user", "content": "被修改的消息"}],
    )

    with pytest.raises(IntakeChatConflict, match="会话内容已更新"):
        build_runner(repository, MustNotRunAgent(), RecordingActivity()).run_chat(
            request
        )
