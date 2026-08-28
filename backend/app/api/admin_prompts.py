from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.admin_models import require_agent_admin
from app.config import settings
from app.database import get_session
from app.models.database import PromptDefinition, PromptRevision
from app.schemas.admin import AgentVersionResponse
from app.schemas.admin_prompts import (
    NodePromptBindingRequest,
    PromptDefinitionCreate,
    PromptDefinitionResponse,
    PromptRevisionCreate,
    PromptRevisionResponse,
    PromptValidationResponse,
    PromptWorkingCopyUpdate,
)
from app.services.agent_config.prompts import PromptConfigService
from app.services.agent_config.service import (
    SYSTEM_TENANT_ID,
    AgentConfigService,
)


router = APIRouter(
    prefix="/api/v1/admin",
    tags=["agent-admin-prompts"],
    dependencies=[Depends(require_agent_admin)],
)


@router.post("/prompts", response_model=PromptDefinitionResponse)
def create_prompt_definition(
    payload: PromptDefinitionCreate,
    session: Session = Depends(get_session),
) -> PromptDefinitionResponse:
    try:
        AgentConfigService(session, settings).ensure_default_agent()
        definition, revision = PromptConfigService(
            session, settings
        ).create_definition(
            tenant_id=SYSTEM_TENANT_ID,
            **payload.model_dump(),
        )
    except (KeyError, ValueError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _definition_response(definition, revision)


@router.get("/prompts", response_model=list[PromptDefinitionResponse])
def list_prompt_definitions(
    node_key: str | None = Query(default=None),
    session: Session = Depends(get_session),
) -> list[PromptDefinitionResponse]:
    try:
        AgentConfigService(session, settings).ensure_default_agent()
        items = PromptConfigService(session, settings).list_definitions(
            SYSTEM_TENANT_ID,
            node_key=node_key,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [_definition_response(definition, revision) for definition, revision in items]


@router.get(
    "/prompts/{prompt_definition_id}/revisions",
    response_model=list[PromptRevisionResponse],
)
def list_prompt_revisions(
    prompt_definition_id: str,
    session: Session = Depends(get_session),
) -> list[PromptRevisionResponse]:
    try:
        revisions = PromptConfigService(session, settings).list_revisions(
            prompt_definition_id
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [_revision_response(revision) for revision in revisions]


@router.post(
    "/prompts/{prompt_definition_id}/revisions",
    response_model=PromptRevisionResponse,
)
def revise_prompt_definition(
    prompt_definition_id: str,
    payload: PromptRevisionCreate,
    session: Session = Depends(get_session),
) -> PromptRevisionResponse:
    try:
        revision = PromptConfigService(session, settings).revise_definition(
            prompt_definition_id,
            **payload.model_dump(),
        )
    except (KeyError, ValueError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _revision_response(revision)


@router.post(
    "/prompt-revisions/{prompt_revision_id}/validate",
    response_model=PromptValidationResponse,
)
def validate_prompt_revision(
    prompt_revision_id: str,
    session: Session = Depends(get_session),
) -> PromptValidationResponse:
    try:
        report = PromptConfigService(session, settings).validate_revision(
            prompt_revision_id
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PromptValidationResponse.model_validate(report)


@router.post(
    "/agent-versions/{agent_version_id}/nodes/{node_key}/prompt",
    response_model=AgentVersionResponse,
)
def bind_node_prompt(
    agent_version_id: str,
    node_key: str,
    payload: NodePromptBindingRequest,
    session: Session = Depends(get_session),
) -> AgentVersionResponse:
    try:
        version = AgentConfigService(session, settings).set_draft_node_prompt(
            agent_version_id,
            node_key,
            payload.prompt_revision_id,
        )
    except (KeyError, ValueError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return AgentVersionResponse(
        id=version.id,
        agent_definition_id=version.agent_definition_id,
        version=version.version,
        status=version.status,
        config_hash=version.config_hash,
    )


@router.put(
    "/agent-versions/{agent_version_id}/nodes/{node_key}/prompt-working-copy",
    response_model=AgentVersionResponse,
)
def save_node_prompt_working_copy(
    agent_version_id: str,
    node_key: str,
    payload: PromptWorkingCopyUpdate,
    session: Session = Depends(get_session),
) -> AgentVersionResponse:
    try:
        version = AgentConfigService(
            session, settings
        ).save_draft_node_prompt_working_copy(
            agent_version_id,
            node_key,
            **payload.model_dump(),
        )
    except (KeyError, ValueError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return AgentVersionResponse(
        id=version.id,
        agent_definition_id=version.agent_definition_id,
        version=version.version,
        status=version.status,
        config_hash=version.config_hash,
    )


@router.post(
    "/agent-versions/{agent_version_id}/nodes/{node_key}/prompt-working-copy/discard",
    response_model=AgentVersionResponse,
)
def discard_node_prompt_working_copy(
    agent_version_id: str,
    node_key: str,
    session: Session = Depends(get_session),
) -> AgentVersionResponse:
    try:
        version = AgentConfigService(
            session, settings
        ).discard_draft_node_prompt_working_copy(agent_version_id, node_key)
    except (KeyError, ValueError) as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return AgentVersionResponse(
        id=version.id,
        agent_definition_id=version.agent_definition_id,
        version=version.version,
        status=version.status,
        config_hash=version.config_hash,
    )


def _definition_response(
    definition: PromptDefinition,
    revision: PromptRevision,
) -> PromptDefinitionResponse:
    return PromptDefinitionResponse(
        id=definition.id,
        name=definition.name,
        slug=definition.slug,
        node_key=definition.node_key,
        status=definition.status,
        active_revision=_revision_response(revision),
    )


def _revision_response(revision: PromptRevision) -> PromptRevisionResponse:
    return PromptRevisionResponse(
        id=revision.id,
        prompt_definition_id=revision.prompt_definition_id,
        version=revision.version,
        content=revision.content,
        content_hash=revision.content_hash,
        required_variables=revision.required_variables,
        skills=revision.skill_bundle,
        validation_report=revision.validation_report,
        smoke_test_status=revision.smoke_test_status,
        source=revision.source,
        status=revision.status,
    )
