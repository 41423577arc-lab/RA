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


class NodeModelBindingRequest(BaseModel):
    model_profile_revision_id: str
