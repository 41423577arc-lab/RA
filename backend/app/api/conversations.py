from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_session
from app.models.database import Conversation, ConversationMessage, ResearchTask
from app.schemas.conversations import (
    ConversationDetailResponse,
    ConversationMessageResponse,
    ConversationSummaryResponse,
)
from app.services.auth import Principal, get_current_principal
from app.services.conversations import ConversationService


router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationSummaryResponse])
def list_conversations(
    limit: int = Query(default=30, ge=1, le=100),
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> list[ConversationSummaryResponse]:
    return [
        _summary(item, session)
        for item in ConversationService(session).list_owned(principal, limit=limit)
    ]


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(
    conversation_id: str,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> ConversationDetailResponse:
    conversation = ConversationService(session).get_owned(principal, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    summary = _summary(conversation, session)
    messages = list(
        session.scalars(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation.id)
            .order_by(ConversationMessage.sequence)
        )
    )
    return ConversationDetailResponse(
        **summary.model_dump(),
        messages=[
            ConversationMessageResponse(
                id=item.id,
                sequence=item.sequence,
                role=item.role,
                channel=item.channel,
                content=item.content,
                created_at=item.created_at,
            )
            for item in messages
        ],
    )


def _summary(
    conversation: Conversation, session: Session
) -> ConversationSummaryResponse:
    task = (
        session.get(ResearchTask, conversation.latest_task_id)
        if conversation.latest_task_id
        else None
    )
    last_message = session.scalar(
        select(ConversationMessage.content)
        .where(ConversationMessage.conversation_id == conversation.id)
        .order_by(ConversationMessage.sequence.desc())
        .limit(1)
    )
    return ConversationSummaryResponse(
        id=conversation.id,
        title=conversation.title,
        status=conversation.status,
        intake_session_id=conversation.intake_session_id,
        latest_task_id=conversation.latest_task_id,
        task_status=task.status if task else None,
        last_message=last_message,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )
