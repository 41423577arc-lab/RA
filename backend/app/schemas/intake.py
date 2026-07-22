from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class IntakeMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2_000)


class IntakeChatRequest(BaseModel):
    session_id: UUID = Field(default_factory=uuid4)
    messages: list[IntakeMessage] = Field(min_length=1, max_length=30)


class IntakeChatResult(BaseModel):
    assistant_reply: str = Field(min_length=1, max_length=1_000)
    analysis_input: str = Field(min_length=1, max_length=10_000)
    ready_to_analyze: bool
    missing_information: list[str] = Field(default_factory=list, max_length=8)


class IntakeChatResponse(IntakeChatResult):
    session_id: UUID
