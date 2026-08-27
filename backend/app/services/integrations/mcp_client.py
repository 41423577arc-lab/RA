import asyncio
import json
from collections.abc import Callable

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.schemas.task import ProjectResult


class ProjectMcpClient:
    def __init__(
        self,
        server_url: str,
        *,
        resolved_config: dict | None = None,
        caller_node: str | None = None,
        secret_resolver: Callable[[str], str] | None = None,
    ):
        self.server_url = server_url
        self.resolved_config = resolved_config
        self.caller_node = caller_node
        self.secret_resolver = secret_resolver

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict,
        *,
        caller_node: str,
        secret_resolver: Callable[[str], str] | None = None,
    ) -> "ProjectMcpClient":
        servers = snapshot.get("mcp_server_revisions") or []
        fallback_url = servers[0].get("url", "") if servers else ""
        return cls(
            fallback_url,
            resolved_config=snapshot,
            caller_node=caller_node,
            secret_resolver=secret_resolver,
        )

    @staticmethod
    async def discover_tools(
        server: dict,
        *,
        secret_resolver: Callable[[str], str] | None = None,
    ) -> list[dict]:
        headers: dict[str, str] = {}
        authentication_type = server.get("authentication_type", "none")
        if authentication_type == "bearer":
            secret_ref = server.get("secret_ref")
            if not secret_ref or secret_resolver is None:
                raise RuntimeError("MCP bearer authentication secret cannot be resolved")
            headers["Authorization"] = f"Bearer {secret_resolver(secret_ref)}"
        elif authentication_type != "none":
            raise RuntimeError(f"Unsupported MCP authentication type: {authentication_type}")
        async with httpx.AsyncClient(
            timeout=int(server.get("timeout_seconds") or 10), headers=headers
        ) as http_client:
            async with streamable_http_client(
                server["url"], http_client=http_client
            ) as streams:
                read_stream, write_stream, _ = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema or {},
            }
            for tool in result.tools
        ]

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
            "identity.find_candidates" if self.resolved_config else "find_entity_candidates",
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
            "projects.search" if self.resolved_config else "search_projects",
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
        server_url = self.server_url
        remote_name = name
        timeout_seconds = 10
        headers: dict[str, str] = {}
        output_mapping: dict = {}
        if self.resolved_config is not None:
            mapping, server = self._resolve_logical_tool(name)
            remote_name = mapping["remote_tool_name"]
            arguments = DeclarativeToolAdapter.map_input(
                arguments, mapping.get("input_mapping") or {}
            )
            output_mapping = mapping.get("output_mapping") or {}
            server_url = server["url"]
            timeout_seconds = int(
                mapping.get("timeout_seconds") or server.get("timeout_seconds") or 10
            )
            authentication_type = server.get("authentication_type") or (
                server.get("authentication") or {}
            ).get("type", "none")
            secret_ref = server.get("secret_ref") or (
                server.get("authentication") or {}
            ).get("secret_ref")
            if authentication_type == "bearer":
                if not secret_ref or self.secret_resolver is None:
                    raise RuntimeError("MCP bearer authentication secret cannot be resolved")
                headers["Authorization"] = f"Bearer {self.secret_resolver(secret_ref)}"
            elif authentication_type != "none":
                raise RuntimeError(f"Unsupported MCP authentication type: {authentication_type}")
        async with httpx.AsyncClient(timeout=timeout_seconds, headers=headers) as http_client:
            async with streamable_http_client(
                server_url, http_client=http_client
            ) as streams:
                read_stream, write_stream, _ = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(remote_name, arguments=arguments)
        if getattr(result, "isError", False):
            message = (
                getattr(result.content[0], "text", "unknown MCP error")
                if result.content
                else "unknown MCP error"
            )
            raise RuntimeError(f"MCP {remote_name} failed: {message}")
        payload = result.structuredContent
        if isinstance(payload, dict) and "result" in payload:
            payload = payload["result"]
        if payload is None and result.content:
            text = getattr(result.content[0], "text", "[]")
            payload = json.loads(text)
        return DeclarativeToolAdapter.map_output(payload, output_mapping)

    def _resolve_logical_tool(self, logical_tool_key: str) -> tuple[dict, dict]:
        mappings = {
            item.get("logical_tool_key"): item
            for item in self.resolved_config.get("tool_mappings", [])
            if item.get("provider") == "mcp"
        }
        mapping = mappings.get(logical_tool_key)
        if mapping is None:
            raise PermissionError(f"Logical tool is not enabled: {logical_tool_key}")
        if not self.caller_node or self.caller_node not in mapping.get("allowed_nodes", []):
            raise PermissionError(
                f"Node {self.caller_node or '<unknown>'} cannot call {logical_tool_key}"
            )
        if mapping.get("adapter_key", "declarative") != "declarative":
            raise RuntimeError(f"Code adapter is not registered: {mapping.get('adapter_key')}")
        servers = {
            item.get("revision_id"): item
            for item in self.resolved_config.get("mcp_server_revisions", [])
        }
        server = servers.get(mapping.get("server_revision_id"))
        if server is None:
            raise RuntimeError(f"MCP server revision is missing for {logical_tool_key}")
        return mapping, server


class DeclarativeToolAdapter:
    @staticmethod
    def map_input(arguments: dict, mapping: dict) -> dict:
        rename = mapping.get("rename") or {}
        result = {rename.get(key, key): value for key, value in arguments.items()}
        result.update(mapping.get("constants") or {})
        return result

    @classmethod
    def map_output(cls, payload: object, mapping: dict) -> object:
        if not mapping:
            return payload
        if isinstance(payload, list):
            return [cls.map_output(item, mapping) for item in payload]
        if not isinstance(payload, dict):
            return payload
        rename = mapping.get("rename") or {}
        result = {rename.get(key, key): value for key, value in payload.items()}
        result.update(mapping.get("constants") or {})
        return result
