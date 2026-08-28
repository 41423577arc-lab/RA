from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ModelConnectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    slug: str = Field(min_length=3, max_length=64)
    provider: Literal["openai", "openai_compatible"]
    base_url: str = Field(min_length=8, max_length=500)
    api_key: str | None = Field(default=None, min_length=1, max_length=1000)
    secret_ref: str | None = Field(default=None, min_length=4, max_length=255)


class ModelConnectionRevisionCreate(BaseModel):
    provider: Literal["openai", "openai_compatible"]
    base_url: str = Field(min_length=8, max_length=500)
    secret_ref: str = Field(min_length=4, max_length=255)


class SecretRotateRequest(BaseModel):
    api_key: str = Field(min_length=1, max_length=1000)


class ModelConnectionResponse(BaseModel):
    id: str
    name: str
    slug: str
    status: str
    active_revision_id: str
    revision_version: int
    provider: str
    base_url: str
    authentication_type: str
    secret_ref: str


class ModelProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    slug: str = Field(min_length=3, max_length=64)
    connection_revision_id: str
    model_id: str = Field(min_length=1, max_length=200)
    api_mode: Literal["chat_completions", "responses"]
    parameters: dict = Field(default_factory=dict)


class ModelProfileRevisionCreate(BaseModel):
    connection_revision_id: str
    model_id: str = Field(min_length=1, max_length=200)
    api_mode: Literal["chat_completions", "responses"]
    parameters: dict = Field(default_factory=dict)


class ModelProfileResponse(BaseModel):
    id: str
    name: str
    slug: str
    status: str
    active_revision_id: str
    revision_version: int
    connection_revision_id: str
    model_id: str
    api_mode: str
    parameters: dict


class ConnectionTestResponse(BaseModel):
    ok: bool
    models: list[str] = Field(default_factory=list)


class AgentVersionResponse(BaseModel):
    id: str
    agent_definition_id: str
    version: int
    status: str
    config_hash: str
    release_note: str | None = None
    published_at: datetime | None = None


class PublishAgentVersionRequest(BaseModel):
    release_note: str | None = Field(default=None, max_length=2000)


class RestoreAgentVersionRequest(BaseModel):
    confirm_overwrite: bool = False


class AgentNodeDiffResponse(BaseModel):
    node_key: str
    prompt_changed: bool
    draft_prompt_revision_id: str | None = None
    published_prompt_revision_id: str | None = None
    prompt_content_hash_changed: bool
    model_changed: bool
    draft_model_name: str
    published_model_name: str
    draft_model_revision_id: str | None = None
    published_model_revision_id: str | None = None


class AgentToolDiffResponse(BaseModel):
    logical_tool_key: str
    changed: bool


class AgentConfigDiffResponse(BaseModel):
    has_draft: bool
    has_changes: bool
    draft_version: int | None = None
    published_version: int
    nodes: list[AgentNodeDiffResponse] = Field(default_factory=list)
    tools: list[AgentToolDiffResponse] = Field(default_factory=list)


class NodeModelBindingRequest(BaseModel):
    model_profile_revision_id: str


class AgentNodeBindingResponse(BaseModel):
    node_key: str
    output_schema: str
    conditional: bool
    allows_tools: bool
    model_profile_revision_id: str | None = None
    model_id: str
    provider: str
    prompt_definition_id: str | None = None
    prompt_revision_id: str | None = None
    prompt_version: int | None = None
    prompt_source: str | None = None
    prompt_config: dict = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)


class AgentToolBindingResponse(BaseModel):
    logical_tool_key: str
    tool_mapping_revision_id: str
    remote_tool_name: str
    adapter_key: str
    allowed_nodes: list[str] = Field(default_factory=list)


class AgentVersionDetailResponse(AgentVersionResponse):
    config_schema_version: int
    nodes: list[AgentNodeBindingResponse]
    tools: list[AgentToolBindingResponse]


class AgentDefinitionDetailResponse(BaseModel):
    id: str
    name: str
    slug: str
    status: str
    published_version: AgentVersionDetailResponse
    draft_version: AgentVersionDetailResponse | None = None
