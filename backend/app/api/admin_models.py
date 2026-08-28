from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_session
from app.models.database import (
    AgentDefinition,
    AgentNodeBinding,
    AgentToolBinding,
    AgentVersion,
    ModelConnection,
    ModelConnectionRevision,
    ModelProfile,
    ModelProfileRevision,
    PromptRevision,
)
from app.schemas.admin import (
    AgentDefinitionDetailResponse,
    AgentConfigDiffResponse,
    AgentNodeBindingResponse,
    AgentToolBindingResponse,
    AgentVersionDetailResponse,
    AgentVersionResponse,
    ConnectionTestResponse,
    ModelConnectionCreate,
    ModelConnectionResponse,
    ModelConnectionRevisionCreate,
    ModelProfileCreate,
    ModelProfileResponse,
    ModelProfileRevisionCreate,
    NodeModelBindingRequest,
    PublishAgentVersionRequest,
    RestoreAgentVersionRequest,
    SecretRotateRequest,
)
from app.services.agent_config.models import ModelConfigService
from app.services.agent_config.mcp import McpConfigService
from app.services.agent_config.registry import NODE_REGISTRY
from app.services.agent_config.service import (
    DEFAULT_AGENT_DEFINITION_ID,
    AgentConfigService,
    SYSTEM_TENANT_ID,
)
from app.services.auth import Principal, get_current_principal


def require_agent_admin(
    principal: Principal = Depends(get_current_principal),
) -> None:
    if not settings.agent_admin_enabled:
        raise HTTPException(status_code=404, detail="Agent administration is disabled")
    if settings.auth_enabled and principal.role not in {"ADMIN", "SYSTEM"}:
        raise HTTPException(status_code=403, detail="Administrator access required")


router = APIRouter(
    prefix="/api/v1/admin",
    tags=["agent-admin"],
    dependencies=[Depends(require_agent_admin)],
)


@router.get("/agent", response_model=AgentDefinitionDetailResponse)
def get_default_agent(
    session: Session = Depends(get_session),
) -> AgentDefinitionDetailResponse:
    AgentConfigService(session, settings).ensure_default_agent()
    definition = session.get(AgentDefinition, DEFAULT_AGENT_DEFINITION_ID)
    if definition is None or definition.tenant_id != SYSTEM_TENANT_ID:
        raise HTTPException(status_code=404, detail="Agent definition not found")
    published = session.get(AgentVersion, definition.published_version_id)
    if published is None:
        raise HTTPException(status_code=409, detail="Agent has no published version")
    draft = _latest_draft(session, definition.id)
    return AgentDefinitionDetailResponse(
        id=definition.id,
        name=definition.name,
        slug=definition.slug,
        status=definition.status,
        published_version=_version_detail(session, published),
        draft_version=_version_detail(session, draft) if draft else None,
    )


@router.get("/agent/versions", response_model=list[AgentVersionResponse])
def list_default_agent_versions(
    session: Session = Depends(get_session),
) -> list[AgentVersionResponse]:
    AgentConfigService(session, settings).ensure_default_agent()
    return [
        _version_response(version)
        for version in AgentConfigService(
            session, settings
        ).list_published_versions()
    ]


@router.get("/agent/diff", response_model=AgentConfigDiffResponse)
def diff_default_agent(
    session: Session = Depends(get_session),
) -> AgentConfigDiffResponse:
    try:
        AgentConfigService(session, settings).ensure_default_agent()
        return AgentConfigDiffResponse.model_validate(
            AgentConfigService(session, settings).diff_draft_to_published()
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/agent/versions/{agent_version_id}/restore",
    response_model=AgentVersionResponse,
)
def restore_default_agent_version(
    agent_version_id: str,
    payload: RestoreAgentVersionRequest,
    session: Session = Depends(get_session),
) -> AgentVersionResponse:
    try:
        version = AgentConfigService(
            session, settings
        ).restore_published_version_to_draft(
            agent_version_id,
            confirm_overwrite=payload.confirm_overwrite,
        )
    except (KeyError, ValueError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _version_response(version)


@router.post("/model-connections", response_model=ModelConnectionResponse)
def create_model_connection(
    payload: ModelConnectionCreate,
    session: Session = Depends(get_session),
) -> ModelConnectionResponse:
    try:
        connection, revision = ModelConfigService(session, settings).create_connection(
            **payload.model_dump()
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _connection_response(connection, revision)


@router.get("/model-connections", response_model=list[ModelConnectionResponse])
def list_model_connections(
    session: Session = Depends(get_session),
) -> list[ModelConnectionResponse]:
    connections = list(
        session.scalars(
            select(ModelConnection)
            .where(ModelConnection.tenant_id == SYSTEM_TENANT_ID)
            .order_by(ModelConnection.name)
        )
    )
    return [
        _connection_response(
            connection,
            session.get(ModelConnectionRevision, connection.active_revision_id),
        )
        for connection in connections
        if connection.active_revision_id
    ]


@router.post(
    "/model-connections/{connection_id}/revisions",
    response_model=ModelConnectionResponse,
)
def revise_model_connection(
    connection_id: str,
    payload: ModelConnectionRevisionCreate,
    session: Session = Depends(get_session),
) -> ModelConnectionResponse:
    service = ModelConfigService(session, settings)
    try:
        revision = service.revise_connection(connection_id, **payload.model_dump())
        connection = session.get(ModelConnection, connection_id)
    except (KeyError, ValueError, RuntimeError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _connection_response(connection, revision)


@router.post("/model-connections/{connection_id}/rotate-secret")
def rotate_model_connection_secret(
    connection_id: str,
    payload: SecretRotateRequest,
    session: Session = Depends(get_session),
) -> dict[str, str]:
    try:
        fingerprint = ModelConfigService(session, settings).rotate_connection_secret(
            connection_id, payload.api_key
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "rotated", "fingerprint": fingerprint}


@router.post(
    "/model-connections/{connection_id}/test",
    response_model=ConnectionTestResponse,
)
def test_model_connection(
    connection_id: str,
    session: Session = Depends(get_session),
) -> ConnectionTestResponse:
    try:
        return ConnectionTestResponse.model_validate(
            ModelConfigService(session, settings).test_connection(connection_id)
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/model-profiles", response_model=ModelProfileResponse)
def create_model_profile(
    payload: ModelProfileCreate,
    session: Session = Depends(get_session),
) -> ModelProfileResponse:
    try:
        profile, revision = ModelConfigService(session, settings).create_profile(
            **payload.model_dump()
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _profile_response(profile, revision)


@router.get("/model-profiles", response_model=list[ModelProfileResponse])
def list_model_profiles(
    session: Session = Depends(get_session),
) -> list[ModelProfileResponse]:
    profiles = list(
        session.scalars(
            select(ModelProfile)
            .where(ModelProfile.tenant_id == SYSTEM_TENANT_ID)
            .order_by(ModelProfile.name)
        )
    )
    return [
        _profile_response(
            profile,
            session.get(ModelProfileRevision, profile.active_revision_id),
        )
        for profile in profiles
        if profile.active_revision_id
    ]


@router.post("/model-profiles/{profile_id}/revisions", response_model=ModelProfileResponse)
def revise_model_profile(
    profile_id: str,
    payload: ModelProfileRevisionCreate,
    session: Session = Depends(get_session),
) -> ModelProfileResponse:
    try:
        revision = ModelConfigService(session, settings).revise_profile(
            profile_id, **payload.model_dump()
        )
        profile = session.get(ModelProfile, profile_id)
    except (KeyError, ValueError, RuntimeError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _profile_response(profile, revision)


@router.post(
    "/agents/{agent_definition_id}/drafts",
    response_model=AgentVersionResponse,
)
def create_agent_draft(
    agent_definition_id: str,
    session: Session = Depends(get_session),
) -> AgentVersionResponse:
    try:
        version = AgentConfigService(session, settings).create_draft(agent_definition_id)
    except (KeyError, ValueError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _version_response(version)


@router.post(
    "/agent-versions/{agent_version_id}/nodes/{node_key}/model",
    response_model=AgentVersionResponse,
)
def bind_node_model(
    agent_version_id: str,
    node_key: str,
    payload: NodeModelBindingRequest,
    session: Session = Depends(get_session),
) -> AgentVersionResponse:
    try:
        version = AgentConfigService(session, settings).set_draft_node_model(
            agent_version_id,
            node_key,
            payload.model_profile_revision_id,
        )
    except (KeyError, ValueError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _version_response(version)


@router.post(
    "/agent-versions/{agent_version_id}/publish",
    response_model=AgentVersionResponse,
)
def publish_agent_version(
    agent_version_id: str,
    payload: PublishAgentVersionRequest | None = Body(default=None),
    session: Session = Depends(get_session),
) -> AgentVersionResponse:
    try:
        version = AgentConfigService(session, settings).publish_draft(
            agent_version_id,
            release_note=payload.release_note if payload else None,
        )
    except (KeyError, ValueError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _version_response(version)


def _connection_response(
    connection: ModelConnection | None,
    revision: ModelConnectionRevision | None,
) -> ModelConnectionResponse:
    if connection is None or revision is None:
        raise HTTPException(status_code=500, detail="Model connection revision is missing")
    return ModelConnectionResponse(
        id=connection.id,
        name=connection.name,
        slug=connection.slug,
        status=connection.status,
        active_revision_id=revision.id,
        revision_version=revision.version,
        provider=revision.provider,
        base_url=revision.base_url,
        authentication_type=revision.authentication_type,
        secret_ref=revision.secret_ref,
    )


def _profile_response(
    profile: ModelProfile | None,
    revision: ModelProfileRevision | None,
) -> ModelProfileResponse:
    if profile is None or revision is None:
        raise HTTPException(status_code=500, detail="Model profile revision is missing")
    return ModelProfileResponse(
        id=profile.id,
        name=profile.name,
        slug=profile.slug,
        status=profile.status,
        active_revision_id=revision.id,
        revision_version=revision.version,
        connection_revision_id=revision.connection_revision_id,
        model_id=revision.model_id,
        api_mode=revision.api_mode,
        parameters=revision.parameters,
    )


def _version_response(version: AgentVersion) -> AgentVersionResponse:
    return AgentVersionResponse(
        id=version.id,
        agent_definition_id=version.agent_definition_id,
        version=version.version,
        status=version.status,
        config_hash=version.config_hash,
        release_note=version.release_note,
        published_at=version.published_at,
    )


def _latest_draft(session: Session, agent_definition_id: str) -> AgentVersion | None:
    drafts = list(
        session.scalars(
            select(AgentVersion)
            .where(
                AgentVersion.agent_definition_id == agent_definition_id,
                AgentVersion.status == "DRAFT",
            )
            .order_by(AgentVersion.version)
        )
    )
    if len(drafts) > 1:
        raise HTTPException(
            status_code=409,
            detail="Agent configuration has multiple draft versions",
        )
    return drafts[0] if drafts else None


def _version_detail(
    session: Session, version: AgentVersion
) -> AgentVersionDetailResponse:
    service = AgentConfigService(session, settings)
    behavior = service.behavior_for_version(version.id)
    bindings = list(
        session.scalars(
            select(AgentNodeBinding)
            .where(AgentNodeBinding.agent_version_id == version.id)
            .order_by(AgentNodeBinding.node_key)
        )
    )
    nodes = []
    for binding in bindings:
        node = behavior["nodes"][binding.node_key]
        prompt_revision = (
            session.get(PromptRevision, binding.prompt_revision_id)
            if binding.prompt_revision_id
            else None
        )
        spec = NODE_REGISTRY[binding.node_key]
        nodes.append(
            AgentNodeBindingResponse(
                node_key=binding.node_key,
                output_schema=spec.output_schema,
                conditional=spec.conditional,
                allows_tools=spec.allows_tools,
                model_profile_revision_id=binding.model_profile_revision_id,
                model_id=node["model"].get("model_id", ""),
                provider=node["model"].get("provider", ""),
                prompt_definition_id=(
                    prompt_revision.prompt_definition_id if prompt_revision else None
                ),
                prompt_revision_id=binding.prompt_revision_id,
                prompt_version=prompt_revision.version if prompt_revision else None,
                prompt_source=node["prompt"].get("source"),
                prompt_config=node["prompt"],
                allowed_tools=node.get("allowed_tools", []),
            )
        )
    tools = []
    mcp_service = McpConfigService(session, settings)
    for binding in session.scalars(
        select(AgentToolBinding)
        .where(AgentToolBinding.agent_version_id == version.id)
        .order_by(AgentToolBinding.logical_tool_key)
    ):
        mapping = mcp_service.resolve_mapping_revision(
            binding.tool_mapping_revision_id
        )
        tools.append(
            AgentToolBindingResponse(
                logical_tool_key=binding.logical_tool_key,
                tool_mapping_revision_id=binding.tool_mapping_revision_id,
                remote_tool_name=mapping["remote_tool_name"],
                adapter_key=mapping["adapter_key"],
                allowed_nodes=binding.allowed_nodes,
            )
        )
    return AgentVersionDetailResponse(
        **_version_response(version).model_dump(),
        config_schema_version=version.config_schema_version,
        nodes=nodes,
        tools=tools,
    )
