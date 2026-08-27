from datetime import datetime

from pydantic import BaseModel, Field


class ConversationMessageResponse(BaseModel):
    id: str
    sequence: int
    role: str
    channel: str
    content: str
    created_at: datetime


class ConversationSummaryResponse(BaseModel):
    id: str
    title: str
    status: str
    intake_session_id: str | None = None
    latest_task_id: str | None = None
    task_status: str | None = None
    last_message: str | None = None
    created_at: datetime
    updated_at: datetime


class ConversationDetailResponse(ConversationSummaryResponse):
    messages: list[ConversationMessageResponse] = Field(default_factory=list)
