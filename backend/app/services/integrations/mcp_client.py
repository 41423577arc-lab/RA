import asyncio
import json

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.schemas.task import ProjectResult


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

    async def find_entity_candidates(
        self,
        person_mention: str | None = None,
        organization_mention: str | None = None,
    ) -> list[dict]:
        payload = await self._call_with_retry(
            "find_entity_candidates",
            {
                "person_mention": person_mention,
                "organization_mention": organization_mention,
            },
        )
        if not isinstance(payload, list):
            raise RuntimeError("MCP find_entity_candidates returned an invalid payload")
        return [item for item in payload if isinstance(item, dict)]

    async def _call_with_retry(self, name: str, arguments: dict) -> object:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return await self._call_tool(name, arguments)
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(f"MCP {name} failed: {last_error}") from last_error

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
        if not isinstance(payload, list):
            raise RuntimeError("MCP search_projects returned an invalid payload")
        return [ProjectResult.model_validate(item) for item in payload]

    async def get_project_details(self, project_id: str) -> dict:
        payload = await self._call_tool("get_project_details", {"project_id": project_id})
        if not isinstance(payload, dict):
            raise RuntimeError("MCP get_project_details returned an invalid payload")
        return payload

    async def get_sales_portfolio(
        self, manager_name: str | None = None, sales_rep_name: str | None = None
    ) -> list[dict]:
        payload = await self._call_tool(
            "get_sales_portfolio",
            {"manager_name": manager_name, "sales_rep_name": sales_rep_name},
        )
        if not isinstance(payload, list):
            raise RuntimeError("MCP get_sales_portfolio returned an invalid payload")
        return payload

    async def _call_tool(self, name: str, arguments: dict) -> object:
        async with httpx.AsyncClient(timeout=10) as http_client:
            async with streamable_http_client(
                self.server_url, http_client=http_client
            ) as streams:
                read_stream, write_stream, _ = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments=arguments)
        if getattr(result, "isError", False):
            message = (
                getattr(result.content[0], "text", "unknown MCP error")
                if result.content
                else "unknown MCP error"
            )
            raise RuntimeError(f"MCP {name} failed: {message}")
        payload = result.structuredContent
        if isinstance(payload, dict) and "result" in payload:
            payload = payload["result"]
        if payload is None and result.content:
            text = getattr(result.content[0], "text", "[]")
            payload = json.loads(text)
        return payload
