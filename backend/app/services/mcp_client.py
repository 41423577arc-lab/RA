import asyncio
import json

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.schemas.task import CandidateOption, ProjectResult


class ProjectMcpClient:
    def __init__(self, server_url: str):
        self.server_url = server_url

    async def search_projects(
        self,
        person_names: list[str],
        organization_names: list[str],
        keywords: list[str],
    ) -> list[ProjectResult]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return await self._search_once(person_names, organization_names, keywords)
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(f"MCP search_projects failed: {last_error}") from last_error

    async def resolve_entities(
        self, mentions: list[str], organization_names: list[str]
    ) -> list[CandidateOption]:
        payload = await self._call_tool(
            "resolve_entities",
            {"mentions": mentions, "organization_names": organization_names},
        )
        return [CandidateOption.model_validate(item) for item in payload]

    async def _search_once(
        self,
        person_names: list[str],
        organization_names: list[str],
        keywords: list[str],
    ) -> list[ProjectResult]:
        payload = await self._call_tool(
            "search_projects",
            {
                "person_names": person_names,
                "organization_names": organization_names,
                "keywords": keywords,
            },
        )
        return [ProjectResult.model_validate(item) for item in payload]

    async def _call_tool(self, name: str, arguments: dict) -> list[dict]:
        async with httpx.AsyncClient(timeout=10) as http_client:
            async with streamable_http_client(
                self.server_url, http_client=http_client
            ) as streams:
                read_stream, write_stream, _ = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments=arguments)
        payload = result.structuredContent
        if isinstance(payload, dict) and "result" in payload:
            payload = payload["result"]
        if payload is None and result.content:
            text = getattr(result.content[0], "text", "[]")
            payload = json.loads(text)
        if not isinstance(payload, list):
            raise RuntimeError("MCP search_projects returned an invalid payload")
        return payload
