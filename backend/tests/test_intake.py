from uuid import UUID

from fastapi.testclient import TestClient

import app.api.intake as intake_api
from app.main import app
from app.schemas.intake import IntakeChatResult


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
    assert payload["missing_information"] == ["讨论事项"]
    assert fake.request.messages[0].content.endswith("赴宴。")


def test_intake_chat_rejects_empty_messages() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/intake/chat", json={"messages": []})

    assert response.status_code == 422
