from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_session
from app.models.database import (
    AgentVersion,
    ModelConnection,
    ModelConnectionRevision,
    ModelProfile,
    ModelProfileRevision,
)
from app.schemas.admin import (
    AgentVersionResponse,
    ConnectionTestResponse,
    ModelConnectionCreate,
    ModelConnectionResponse,
    ModelConnectionRevisionCreate,
    ModelProfileCreate,
    ModelProfileResponse,
    ModelProfileRevisionCreate,
    NodeModelBindingRequest,
    SecretRotateRequest,
)
from app.services.agent_config.models import ModelConfigService
from app.services.agent_config.service import AgentConfigService, SYSTEM_TENANT_ID


def require_agent_admin() -> None:
    if not settings.agent_admin_enabled:
        raise HTTPException(status_code=404, detail="Agent administration is disabled")


router = APIRouter(
    prefix="/api/v1/admin",
    tags=["agent-admin"],
    dependencies=[Depends(require_agent_admin)],
)


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
    session: Session = Depends(get_session),
) -> AgentVersionResponse:
    try:
        version = AgentConfigService(session, settings).publish_draft(agent_version_id)
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
    )
