from pathlib import Path

from fastapi.testclient import TestClient

import app.api.tasks as task_api
from app.config import settings
from app.main import app


def test_text_task_endpoint_and_status_endpoint(monkeypatch) -> None:
    dispatched: list[str] = []
    monkeypatch.setattr(task_api.run_research_pipeline, "delay", dispatched.append)

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/tasks/text",
            json={"text": "老板周五要和比亚迪股份有限公司的王传福董事长兼总裁吃饭。"},
        )
        assert created.status_code == 202
        payload = created.json()
        assert payload["status"] == "PENDING"
        assert dispatched == [payload["task_id"]]

        fetched = client.get(f"/api/v1/tasks/{payload['task_id']}")
        assert fetched.status_code == 200
        assert fetched.json()["input_text"].startswith("老板周五")


def test_audio_endpoint_rejects_wrong_mime_type(monkeypatch) -> None:
    monkeypatch.setattr(task_api.run_research_pipeline, "delay", lambda _: None)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tasks/audio",
            files={"audio": ("recording.mp3", b"not-webm", "audio/mpeg")},
        )
    assert response.status_code == 415


def test_audio_endpoint_saves_webm_and_dispatches(monkeypatch, tmp_path: Path) -> None:
    dispatched: list[str] = []
    monkeypatch.setattr(task_api.run_research_pipeline, "delay", dispatched.append)
    monkeypatch.setattr(settings, "audio_dir", tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tasks/audio",
            files={"audio": ("recording.webm", b"demo-webm", "audio/webm")},
        )

    assert response.status_code == 202
    task_id = response.json()["task_id"]
    assert dispatched == [task_id]
    assert (tmp_path / f"{task_id}.webm").read_bytes() == b"demo-webm"
