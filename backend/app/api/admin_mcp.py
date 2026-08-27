from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.admin_models import require_agent_admin
from app.config import settings
from app.database import get_session
from app.models.database import (
    McpServerDefinition,
    McpServerRevision,
    ToolMappingDefinition,
    ToolMappingRevision,
)
from app.schemas.admin import AgentVersionResponse
from app.schemas.admin_mcp import (
    AgentToolBindingRequest,
    DiscoveredToolResponse,
    McpServerCreate,
    McpServerResponse,
    McpServerRevisionCreate,
    ToolMappingCreate,
    ToolMappingResponse,
    ToolMappingRevisionCreate,
)
from app.services.agent_config.mcp import McpConfigService
from app.services.agent_config.secrets import SecretStore
from app.services.agent_config.service import AgentConfigService, SYSTEM_TENANT_ID
from app.services.integrations.mcp_client import ProjectMcpClient


router = APIRouter(
    prefix="/api/v1/admin",
    tags=["agent-admin-mcp"],
    dependencies=[Depends(require_agent_admin)],
)


@router.post("/mcp-servers", response_model=McpServerResponse)
def create_mcp_server(
    payload: McpServerCreate, session: Session = Depends(get_session)
) -> McpServerResponse:
    try:
        definition, revision = McpConfigService(session, settings).create_server(
            tenant_id=SYSTEM_TENANT_ID, **payload.model_dump()
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _server_response(definition, revision)


@router.get("/mcp-servers", response_model=list[McpServerResponse])
def list_mcp_servers(session: Session = Depends(get_session)) -> list[McpServerResponse]:
    AgentConfigService(session, settings).ensure_default_agent()
    definitions = session.scalars(
        select(McpServerDefinition)
        .where(McpServerDefinition.tenant_id == SYSTEM_TENANT_ID)
        .order_by(McpServerDefinition.name)
    )
    return [
        _server_response(item, session.get(McpServerRevision, item.active_revision_id))
        for item in definitions
        if item.active_revision_id
    ]


@router.post("/mcp-servers/{server_id}/revisions", response_model=McpServerResponse)
def revise_mcp_server(
    server_id: str,
    payload: McpServerRevisionCreate,
    session: Session = Depends(get_session),
) -> McpServerResponse:
    try:
        revision = McpConfigService(session, settings).revise_server(
            server_id, **payload.model_dump()
        )
        definition = session.get(McpServerDefinition, server_id)
    except (KeyError, ValueError, RuntimeError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _server_response(definition, revision)


@router.post(
    "/mcp-servers/{server_id}/discover-tools",
    response_model=list[DiscoveredToolResponse],
)
async def discover_mcp_tools(
    server_id: str, session: Session = Depends(get_session)
) -> list[DiscoveredToolResponse]:
    try:
        definition = session.get(McpServerDefinition, server_id)
        if definition is None or not definition.active_revision_id:
            raise KeyError(f"MCP server not found: {server_id}")
        server = McpConfigService(session, settings).resolve_server_revision(
            definition.active_revision_id
        )
        tools = await ProjectMcpClient.discover_tools(
            server, secret_resolver=SecretStore(session, settings).resolve
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return [DiscoveredToolResponse.model_validate(item) for item in tools]


@router.post("/tool-mappings", response_model=ToolMappingResponse)
def create_tool_mapping(
    payload: ToolMappingCreate, session: Session = Depends(get_session)
) -> ToolMappingResponse:
    try:
        definition, revision = McpConfigService(session, settings).create_mapping(
            tenant_id=SYSTEM_TENANT_ID, **payload.model_dump()
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _mapping_response(definition, revision)


@router.get("/tool-mappings", response_model=list[ToolMappingResponse])
def list_tool_mappings(session: Session = Depends(get_session)) -> list[ToolMappingResponse]:
    AgentConfigService(session, settings).ensure_default_agent()
    definitions = session.scalars(
        select(ToolMappingDefinition)
        .where(ToolMappingDefinition.tenant_id == SYSTEM_TENANT_ID)
        .order_by(ToolMappingDefinition.logical_tool_key)
    )
    return [
        _mapping_response(item, session.get(ToolMappingRevision, item.active_revision_id))
        for item in definitions
        if item.active_revision_id
    ]


@router.post(
    "/tool-mappings/{mapping_id}/revisions", response_model=ToolMappingResponse
)
def revise_tool_mapping(
    mapping_id: str,
    payload: ToolMappingRevisionCreate,
    session: Session = Depends(get_session),
) -> ToolMappingResponse:
    try:
        revision = McpConfigService(session, settings).revise_mapping(
            mapping_id, **payload.model_dump()
        )
        definition = session.get(ToolMappingDefinition, mapping_id)
    except (KeyError, ValueError, RuntimeError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _mapping_response(definition, revision)


@router.post(
    "/agent-versions/{agent_version_id}/tools/{logical_tool_key}/binding",
    response_model=AgentVersionResponse,
)
def bind_agent_tool(
    agent_version_id: str,
    logical_tool_key: str,
    payload: AgentToolBindingRequest,
    session: Session = Depends(get_session),
) -> AgentVersionResponse:
    try:
        version = AgentConfigService(session, settings).set_draft_tool_mapping(
            agent_version_id,
            logical_tool_key,
            payload.tool_mapping_revision_id,
            payload.allowed_nodes,
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return AgentVersionResponse(
        id=version.id,
        agent_definition_id=version.agent_definition_id,
        version=version.version,
        status=version.status,
        config_hash=version.config_hash,
    )


def _server_response(
    definition: McpServerDefinition | None, revision: McpServerRevision | None
) -> McpServerResponse:
    if definition is None or revision is None:
        raise HTTPException(status_code=500, detail="MCP server revision is missing")
    return McpServerResponse(
        id=definition.id,
        name=definition.name,
        slug=definition.slug,
        status=definition.status,
        active_revision_id=revision.id,
        revision_version=revision.version,
        transport=revision.transport,
        url=revision.url,
        authentication_type=revision.authentication_type,
        secret_ref=revision.secret_ref,
        timeout_seconds=revision.timeout_seconds,
    )


def _mapping_response(
    definition: ToolMappingDefinition | None, revision: ToolMappingRevision | None
) -> ToolMappingResponse:
    if definition is None or revision is None:
        raise HTTPException(status_code=500, detail="Tool mapping revision is missing")
    return ToolMappingResponse(
        id=definition.id,
        name=definition.name,
        logical_tool_key=definition.logical_tool_key,
        status=definition.status,
        active_revision_id=revision.id,
        revision_version=revision.version,
        mcp_server_revision_id=revision.mcp_server_revision_id,
        remote_tool_name=revision.remote_tool_name,
        adapter_key=revision.adapter_key,
        input_mapping=revision.input_mapping,
        output_mapping=revision.output_mapping,
        timeout_seconds=revision.timeout_seconds,
    )
