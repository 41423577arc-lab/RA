from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.services.mcp_client as mcp_module
from app.services.mcp_client import ProjectMcpClient


class FakeTransport:
    async def __aenter__(self):
        return "read", "write", lambda: None

    async def __aexit__(self, *_):
        return False


class FakeSession:
    def __init__(self, read, write):
        assert (read, write) == ("read", "write")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def initialize(self):
        return None

    async def call_tool(self, name, arguments):
        assert name == "search_projects"
        assert arguments["person_names"] == ["王传福"]
        return SimpleNamespace(
            structuredContent={
                "result": [
                    {
                        "project_id": "P001",
                        "project_name": "储能平台",
                        "customer_name": "比亚迪股份有限公司",
                        "contact_name": "王传福",
                        "status": "ACTIVE",
                        "owner_name": "张伟",
                        "start_date": "2026-01-10",
                        "end_date": None,
                        "description": "园区储能",
                        "match_type": "PERSON_EXACT",
                        "similarity": None,
                    }
                ]
            },
            content=[],
        )


@pytest.mark.asyncio
async def test_mcp_client_calls_only_search_projects(monkeypatch) -> None:
    captured = {}

    def fake_transport(url, *, http_client):
        captured["url"] = url
        captured["http_client"] = http_client
        return FakeTransport()

    monkeypatch.setattr(mcp_module, "streamable_http_client", fake_transport)
    monkeypatch.setattr(mcp_module, "ClientSession", FakeSession)

    projects = await ProjectMcpClient("http://mcp:8001/mcp").search_projects(
        ["王传福"], ["比亚迪股份有限公司"], ["储能"]
    )

    assert captured["url"] == "http://mcp:8001/mcp"
    assert projects[0].project_id == "P001"


@pytest.mark.asyncio
async def test_mcp_client_retries_twice_before_succeeding(monkeypatch) -> None:
    client = ProjectMcpClient("http://mcp:8001/mcp")
    search_once = AsyncMock(side_effect=[RuntimeError("one"), RuntimeError("two"), []])
    sleep = AsyncMock()
    monkeypatch.setattr(client, "_search_once", search_once)
    monkeypatch.setattr(mcp_module.asyncio, "sleep", sleep)

    projects = await client.search_projects([], [], ["储能"])

    assert projects == []
    assert search_once.await_count == 3
    assert [call.args[0] for call in sleep.await_args_list] == [1, 2]
