from pathlib import Path
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

import app.api.intake as intake_api
from app.database import SessionLocal
from app.config import settings
from app.main import app
from app.models.database import IntakeAudioJob, ResearchTask
from app.schemas.intake import (
    IntakeChatResult,
    IntakeFollowupResult,
    IntakeStructuredContext,
)
from app.schemas.task import (
    CandidateOption,
    ConfirmationItem,
    ConfirmationRequest,
    IntentUnderstanding,
    SearchResult,
    WebPage,
)
from app.services.intake_completeness import is_intake_ready
from app.services.intake_entity_candidates import IntakeEntityCandidateService
from app.services.intake_agent import IntakeAgent
from app.tasks.pipeline import context_from_intake_snapshot


class FakeIntakeAgent:
    def __init__(self):
        self.request = None

    def respond(self, request):
        self.request = request
        return IntakeChatResult(
            assistant_reply="这次主要准备讨论什么事情？",
            analysis_input="用户将与新城水务的方正会面。",
            ready_to_analyze=False,
            missing_information=["讨论事项"],
        )


class ReadyIntakeAgent:
    def respond(self, request):
        return IntakeChatResult(
            assistant_reply="信息已经完整，可以开始分析。",
            analysis_input="与比亚迪股份有限公司的王传福讨论储能项目推进。",
            ready_to_analyze=True,
            missing_information=[],
            structured_context=IntakeStructuredContext(
                people=["王传福"],
                organizations=["比亚迪股份有限公司"],
                projects=["储能项目"],
                focus_questions=["储能项目如何推进"],
            ),
        )


class AutoConfirmEntityCandidates:
    def resolve(self, context, version, source_text=None):
        return (
            [
                {
                    "candidate_id": "internal:contact:C001",
                    "entity_type": "PERSON",
                    "canonical_name": context.people[0],
                    "mention": context.people[0],
                    "confirmed_by": "INTERNAL",
                },
                {
                    "candidate_id": "internal:customer:CU001",
                    "entity_type": "ORGANIZATION",
                    "canonical_name": context.organizations[0],
                    "mention": context.organizations[0],
                    "confirmed_by": "INTERNAL",
                },
            ],
            None,
        )


class ExternalConfirmationCandidates:
    def resolve(self, context, version, source_text=None):
        return (
            [
                {
                    "candidate_id": "internal:customer:CU001",
                    "entity_type": "ORGANIZATION",
                    "canonical_name": context.organizations[0],
                    "mention": context.organizations[0],
                    "confirmed_by": "INTERNAL",
                }
            ],
            ConfirmationRequest(
                version=version,
                items=[
                    ConfirmationItem(
                        mention=context.people[0],
                        entity_type="PERSON",
                        candidates=[
                            CandidateOption(
                                candidate_id="external:person-1",
                                entity_type="PERSON",
                                canonical_name=context.people[0],
                                organization=context.organizations[0],
                                reason="公开网页候选，必须由用户确认",
                                confidence=0.6,
                                source_url="https://example.com/person",
                                evidence_quote="人物与企业同时出现",
                            )
                        ],
                    )
                ],
            ),
        )


def test_intake_chat_collects_information_without_creating_task(monkeypatch) -> None:
    fake = FakeIntakeAgent()
    monkeypatch.setattr(intake_api, "intake_agent", fake)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/intake/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "我晚上去和新城水务的方正赴宴。",
                    }
                ]
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert UUID(payload["session_id"])
    assert payload["ready_to_analyze"] is False
    assert payload["missing_information"] == ["人物、企业或项目", "希望分析或推动的事项"]
    assert fake.request.messages[0].content.endswith("赴宴。")


def test_intake_chat_rejects_empty_messages() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/intake/chat", json={"messages": []})

    assert response.status_code == 422


def test_ready_requires_server_side_context_validation() -> None:
    result = IntakeChatResult(
        assistant_reply="可以开始分析。",
        analysis_input="准备一次会面。",
        ready_to_analyze=True,
        missing_information=[],
    )

    assert is_intake_ready(result) is False


def test_ready_rejects_target_not_supported_by_user_message() -> None:
    result = ReadyIntakeAgent().respond(None)

    assert is_intake_ready(result, "请帮我准备一次客户会面") is False


def test_optional_missing_information_does_not_block_ready() -> None:
    result = IntakeChatResult(
        assistant_reply="还可以补充具体物资类别。",
        analysis_input="与中建二局总经理张伟讨论上游物资供应。",
        ready_to_analyze=False,
        missing_information=["用户自身角色", "具体物资类别", "会面期望成果"],
        structured_context=IntakeStructuredContext(
            people=["张伟"],
            organizations=["中建二局"],
            projects=["上游物资供应"],
            focus_questions=["上游物资供应合作机会"],
        ),
    )

    assert is_intake_ready(result, result.analysis_input) is True


def test_intake_session_can_be_restored(monkeypatch) -> None:
    monkeypatch.setattr(intake_api, "intake_agent", FakeIntakeAgent())

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/intake/chat",
            json={"messages": [{"role": "user", "content": "我要准备一次客户会面"}]},
        )
        session_id = created.json()["session_id"]
        restored = client.get(f"/api/v1/intake/{session_id}")

    assert restored.status_code == 200
    payload = restored.json()
    assert payload["version"] == 1
    assert payload["status"] == "COLLECTING"
    assert payload["messages"][-1]["role"] == "assistant"


def test_get_repairs_existing_complete_session_to_ready() -> None:
    session_id = str(uuid4())
    with SessionLocal() as session:
        from app.models.database import IntakeSession

        session.add(
            IntakeSession(
                id=session_id,
                status="COLLECTING",
                messages=[
                    {
                        "role": "user",
                        "content": "我要和中建二局总经理张伟聊上游物资供应",
                    },
                    {
                        "role": "assistant",
                        "content": "还可以补充具体物资类别。",
                    },
                ],
                structured_context={
                    "people": ["张伟"],
                    "organizations": ["中建二局"],
                    "projects": ["上游物资供应"],
                    "focus_questions": ["上游物资供应合作机会"],
                },
                missing_information=["具体物资类别"],
                analysis_input="与中建二局总经理张伟讨论上游物资供应。",
                ready_to_analyze=False,
                version=3,
            )
        )
        session.commit()

    with TestClient(app) as client:
        restored = client.get(f"/api/v1/intake/{session_id}")

    assert restored.status_code == 200
    assert restored.json()["status"] == "READY"
    assert restored.json()["ready_to_analyze"] is True
    assert restored.json()["missing_information"] == []
    assert restored.json()["next_action"] == "PROPOSE_READY"
    with SessionLocal() as session:
        repaired = session.get(IntakeSession, session_id)
        assert repaired.structured_context["requester_context"]["organization"] == (
            "澄岳产业发展有限公司"
        )


def test_start_analysis_rejects_session_that_is_not_ready(monkeypatch) -> None:
    monkeypatch.setattr(intake_api, "intake_agent", FakeIntakeAgent())

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/intake/chat",
            json={"messages": [{"role": "user", "content": "我要准备一次客户会面"}]},
        )
        payload = created.json()
        started = client.post(
            f"/api/v1/intake/{payload['session_id']}/start-analysis",
            json={"expected_version": payload["version"]},
        )

    assert started.status_code == 409


def test_start_analysis_is_idempotent_and_freezes_snapshot(monkeypatch) -> None:
    dispatched: list[str] = []
    monkeypatch.setattr(intake_api, "intake_agent", ReadyIntakeAgent())
    monkeypatch.setattr(intake_api, "entity_candidates", AutoConfirmEntityCandidates())
    monkeypatch.setattr(intake_api.run_research_pipeline, "delay", dispatched.append)

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/intake/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "和比亚迪的王传福讨论储能项目如何推进",
                    }
                ]
            },
        )
        intake = created.json()
        url = f"/api/v1/intake/{intake['session_id']}/start-analysis"
        first = client.post(url, json={"expected_version": intake["version"]})
        second = client.post(url, json={"expected_version": intake["version"]})

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["task_id"] == second.json()["task_id"]
    assert dispatched == [first.json()["task_id"]]
    with SessionLocal() as session:
        task = session.get(ResearchTask, first.json()["task_id"])
        assert task is not None
        assert task.intake_session_id == intake["session_id"]
        assert task.input_snapshot["session_version"] == intake["version"]
        assert task.input_snapshot["structured_context"]["people"] == ["王传福"]
        assert task.input_snapshot["structured_context"]["requester_context"] == {
            "name": "林致远",
            "organization": "澄岳产业发展有限公司",
            "title": "副总经理",
            "role_type": "企业高层领导",
        }


def test_external_candidate_requires_user_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(intake_api, "intake_agent", ReadyIntakeAgent())
    monkeypatch.setattr(intake_api, "entity_candidates", ExternalConfirmationCandidates())

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/intake/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "和比亚迪的王传福讨论储能项目如何推进",
                    }
                ]
            },
        )
        intake = created.json()
        assert intake["status"] == "NEEDS_CONFIRMATION"
        assert intake["ready_to_analyze"] is False
        candidate = intake["confirmation_request"]["items"][0]["candidates"][0]
        confirmed = client.post(
            f"/api/v1/intake/{intake['session_id']}/confirm",
            json={
                "confirmation_version": intake["confirmation_request"]["version"],
                "selections": [
                    {
                        "mention": "王传福",
                        "candidate_id": candidate["candidate_id"],
                    }
                ],
            },
        )

    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "READY"
    assert confirmed.json()["ready_to_analyze"] is True


class NoInternalCandidates:
    async def find_entity_candidates(self, *_):
        return []


class IdentityWeb:
    async def search(self, queries):
        return [
            SearchResult(
                title="人物介绍",
                url="https://example.com/identity",
                query=queries[0],
                rank=0,
            )
        ]

    async def extract(self, _):
        return [
            WebPage(
                title="人物介绍",
                url="https://example.com/identity",
                raw_content="王传福是比亚迪股份有限公司负责人。",
                rank=0,
            )
        ]


class WebMustNotRun:
    async def search(self, _):
        raise AssertionError("Explicit user identity must not require web confirmation")

    async def extract(self, _):
        raise AssertionError


def test_explicit_full_user_identity_does_not_require_web_confirmation() -> None:
    service = IntakeEntityCandidateService(NoInternalCandidates(), WebMustNotRun())
    resolutions, confirmation = service.resolve(
        IntakeStructuredContext(
            people=["张伟"],
            organizations=["中建二局"],
            focus_questions=["上游物资供应合作机会"],
        ),
        1,
        "我要和中建二局总经理张伟聊上游物资供应",
    )

    assert confirmation is None
    assert {item["canonical_name"] for item in resolutions} == {"张伟", "中建二局"}
    assert {item["confirmed_by"] for item in resolutions} == {"USER_INPUT"}

def test_entity_service_keeps_external_results_as_candidates() -> None:
    service = IntakeEntityCandidateService(NoInternalCandidates(), IdentityWeb())
    resolutions, confirmation = service.resolve(
        IntakeStructuredContext(
            people=["王传福"],
            organizations=["比亚迪股份有限公司"],
            focus_questions=["储能项目如何推进"],
        ),
        1,
    )

    assert resolutions == []
    assert confirmation is not None
    assert confirmation.items[0].candidates[0].candidate_id.startswith("external:")


def test_audio_is_transcribed_and_reviewed_before_analysis(monkeypatch, tmp_path: Path) -> None:
    transcription_jobs: list[str] = []
    research_jobs: list[str] = []
    session_id = str(uuid4())
    monkeypatch.setattr(settings, "audio_dir", tmp_path)
    monkeypatch.setattr(
        intake_api.run_intake_audio_transcription, "delay", transcription_jobs.append
    )
    monkeypatch.setattr(intake_api.run_research_pipeline, "delay", research_jobs.append)
    monkeypatch.setattr(intake_api, "intake_agent", ReadyIntakeAgent())
    monkeypatch.setattr(intake_api, "entity_candidates", AutoConfirmEntityCandidates())

    with TestClient(app) as client:
        uploaded = client.post(
            f"/api/v1/intake/{session_id}/audio",
            files={"audio": ("recording.webm", b"demo-webm", "audio/webm")},
        )
        assert uploaded.status_code == 202
        job_id = uploaded.json()["job_id"]
        assert transcription_jobs == [job_id]
        audio_path = tmp_path / f"intake-{job_id}.webm"
        assert audio_path.exists()

        with SessionLocal() as session:
            job = session.get(IntakeAudioJob, job_id)
            job.status = "NEEDS_REVIEW"
            job.transcript = "和比亚迪的王传福讨论储能项目如何推进"
            session.commit()

        reviewed = client.post(
            "/api/v1/intake/chat",
            json={
                "session_id": session_id,
                "audio_job_id": job_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "和比亚迪的王传福讨论储能项目如何推进",
                    }
                ],
            },
        )
        assert reviewed.status_code == 200
        assert reviewed.json()["status"] == "READY"
        started = client.post(
            f"/api/v1/intake/{session_id}/start-analysis",
            json={"expected_version": reviewed.json()["version"]},
        )

    assert started.status_code == 202
    assert started.json()["input_type"] == "audio"
    assert research_jobs == [started.json()["task_id"]]
    assert not audio_path.exists()
    with SessionLocal() as session:
        job = session.get(IntakeAudioJob, job_id)
        assert job.status == "TRANSCRIBED"
        assert job.corrected_transcript.startswith("和比亚迪")


class RecordingLlm:
    def __init__(self):
        self.nodes: list[str] = []
        self.payloads: list[dict] = []

    def parse(self, task_id, node_name, payload, output_model):
        self.nodes.append(node_name)
        self.payloads.append(payload)
        if node_name == "intake_chat":
            return ReadyIntakeAgent().respond(None)
        return IntakeFollowupResult(assistant_reply="请确认身份候选。")


def test_controlled_intake_agent_has_only_two_model_steps() -> None:
    llm = RecordingLlm()
    agent = IntakeAgent(llm)
    request = intake_api.IntakeChatRequest(
        messages=[{"role": "user", "content": "和比亚迪的王传福讨论储能项目"}]
    )

    decision = agent.respond(request)
    follow_up = agent.follow_up(
        request,
        decision,
        {"resolved_count": 0, "needs_confirmation": True, "candidate_count": 1},
    )

    assert follow_up.assistant_reply == "请确认身份候选。"
    assert llm.nodes == ["intake_chat", "intake_followup"]
    assert llm.payloads[0]["default_requester_context"]["organization"] == (
        "澄岳产业发展有限公司"
    )


def test_pipeline_reuses_confirmed_entities_from_input_snapshot() -> None:
    understanding = IntentUnderstanding(
        intents=["MEETING_PREPARATION"],
        event_type="会议",
        overall_confidence=0.9,
    )
    context = context_from_intake_snapshot(
        {
            "structured_context": {
                "entity_resolutions": [
                    {
                        "candidate_id": "internal:contact:C001",
                        "entity_type": "PERSON",
                        "canonical_name": "王传福",
                        "organization": "比亚迪股份有限公司",
                        "confirmed_by": "INTERNAL",
                    }
                ]
            }
        },
        understanding,
    )

    assert context is not None
    assert context.entities[0].canonical_name == "王传福"
    assert context.entities[0].confirmed_by == "AUTO"
