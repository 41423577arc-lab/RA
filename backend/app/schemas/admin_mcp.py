from typing import Literal

from pydantic import BaseModel, Field


class McpServerCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    slug: str = Field(min_length=3, max_length=64)
    url: str = Field(min_length=8, max_length=500)
    authentication_type: Literal["none", "bearer"] = "none"
    api_token: str | None = Field(default=None, min_length=1, max_length=2000)
    secret_ref: str | None = Field(default=None, min_length=4, max_length=255)
    timeout_seconds: int = Field(default=10, ge=1, le=120)


class McpServerRevisionCreate(BaseModel):
    url: str = Field(min_length=8, max_length=500)
    authentication_type: Literal["none", "bearer"] = "none"
    api_token: str | None = Field(default=None, min_length=1, max_length=2000)
    secret_ref: str | None = Field(default=None, min_length=4, max_length=255)
    timeout_seconds: int = Field(default=10, ge=1, le=120)


class McpServerResponse(BaseModel):
    id: str
    name: str
    slug: str
    status: str
    active_revision_id: str
    revision_version: int
    transport: str
    url: str
    authentication_type: str
    secret_ref: str | None
    timeout_seconds: int


class DiscoveredToolResponse(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict = Field(default_factory=dict)


class ToolMappingCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    logical_tool_key: str = Field(min_length=3, max_length=100)
    mcp_server_revision_id: str
    remote_tool_name: str = Field(min_length=1, max_length=100)
    adapter_key: str = "declarative"
    input_mapping: dict = Field(default_factory=dict)
    output_mapping: dict = Field(default_factory=dict)
    timeout_seconds: int = Field(default=10, ge=1, le=120)


class ToolMappingRevisionCreate(BaseModel):
    mcp_server_revision_id: str
    remote_tool_name: str = Field(min_length=1, max_length=100)
    adapter_key: str = "declarative"
    input_mapping: dict = Field(default_factory=dict)
    output_mapping: dict = Field(default_factory=dict)
    timeout_seconds: int = Field(default=10, ge=1, le=120)


class ToolMappingResponse(BaseModel):
    id: str
    name: str
    logical_tool_key: str
    status: str
    active_revision_id: str
    revision_version: int
    mcp_server_revision_id: str
    remote_tool_name: str
    adapter_key: str
    input_mapping: dict
    output_mapping: dict
    timeout_seconds: int


class AgentToolBindingRequest(BaseModel):
    tool_mapping_revision_id: str
    allowed_nodes: list[str] = Field(min_length=1)
