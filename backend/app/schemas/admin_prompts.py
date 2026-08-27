from pydantic import BaseModel, Field


class PromptSkillInput(BaseModel):
    name: str = Field(min_length=2, max_length=64)
    content: str = Field(min_length=1, max_length=50_000)


class PromptDefinitionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    slug: str = Field(min_length=3, max_length=64)
    node_key: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=200_000)
    skills: list[PromptSkillInput] = Field(default_factory=list, max_length=20)


class PromptRevisionCreate(BaseModel):
    content: str = Field(min_length=1, max_length=200_000)
    skills: list[PromptSkillInput] | None = Field(default=None, max_length=20)


class PromptRevisionResponse(BaseModel):
    id: str
    prompt_definition_id: str
    version: int
    content: str
    content_hash: str
    required_variables: list[str]
    skills: list[dict]
    validation_report: dict
    smoke_test_status: str
    source: str
    status: str


class PromptDefinitionResponse(BaseModel):
    id: str
    name: str
    slug: str
    node_key: str
    status: str
    active_revision: PromptRevisionResponse


class PromptValidationResponse(BaseModel):
    valid: bool
    node_key: str
    output_schema: str
    output_schema_boundary: str
    required_variables: list[str]
    skill_names: list[str]


class NodePromptBindingRequest(BaseModel):
    prompt_revision_id: str
