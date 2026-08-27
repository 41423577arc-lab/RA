from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.database import Conversation, ConversationMessage, IntakeSession, ResearchTask
from app.services.auth import Principal


class ConversationService:
    def __init__(self, session: Session):
        self.session = session

    def ensure_for_intake(
        self, principal: Principal, intake_session_id: str
    ) -> Conversation:
        conversation = self.session.scalar(
            select(Conversation).where(
                Conversation.intake_session_id == intake_session_id,
                Conversation.tenant_id == principal.tenant_id,
                Conversation.owner_id == principal.user_id,
            )
        )
        if conversation is not None:
            return conversation
        conflicting = self.session.scalar(
            select(Conversation).where(
                Conversation.intake_session_id == intake_session_id
            )
        )
        if conflicting is not None:
            raise PermissionError("Conversation is owned by another user")
        conversation = Conversation(
            tenant_id=principal.tenant_id,
            owner_id=principal.user_id,
            title="新调查",
            intake_session_id=intake_session_id,
        )
        self.session.add(conversation)
        self.session.flush()
        return conversation

    def create(self, principal: Principal, *, title: str = "新调查") -> Conversation:
        conversation = Conversation(
            tenant_id=principal.tenant_id,
            owner_id=principal.user_id,
            title=title.strip()[:80] or "新调查",
        )
        self.session.add(conversation)
        self.session.flush()
        return conversation

    def get_owned(self, principal: Principal, conversation_id: str) -> Conversation | None:
        return self.session.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.tenant_id == principal.tenant_id,
                Conversation.owner_id == principal.user_id,
            )
        )

    def list_owned(self, principal: Principal, *, limit: int = 50) -> list[Conversation]:
        return list(
            self.session.scalars(
                select(Conversation)
                .where(
                    Conversation.tenant_id == principal.tenant_id,
                    Conversation.owner_id == principal.user_id,
                    Conversation.status != "DELETED",
                )
                .order_by(Conversation.updated_at.desc())
                .limit(limit)
            )
        )

    def sync_messages(
        self,
        conversation: Conversation,
        messages: list[dict],
        *,
        channel: str,
        author_id: str,
    ) -> None:
        existing = list(
            self.session.scalars(
                select(ConversationMessage)
                .where(
                    ConversationMessage.conversation_id == conversation.id,
                    ConversationMessage.channel == channel,
                )
                .order_by(ConversationMessage.sequence)
            )
        )
        existing_pairs = [(item.role, item.content) for item in existing]
        incoming_pairs = [
            (str(item.get("role") or ""), str(item.get("content") or ""))
            for item in messages
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        prefix = 0
        while (
            prefix < len(existing_pairs)
            and prefix < len(incoming_pairs)
            and existing_pairs[prefix] == incoming_pairs[prefix]
        ):
            prefix += 1
        if prefix < len(existing_pairs):
            return
        next_sequence = self.session.scalar(
            select(func.max(ConversationMessage.sequence)).where(
                ConversationMessage.conversation_id == conversation.id
            )
        ) or 0
        for role, content in incoming_pairs[prefix:]:
            next_sequence += 1
            self.session.add(
                ConversationMessage(
                    tenant_id=conversation.tenant_id,
                    conversation_id=conversation.id,
                    author_id=author_id if role == "user" else None,
                    sequence=next_sequence,
                    role=role,
                    channel=channel,
                    content=content,
                )
            )
        first_user_message = next(
            (content for role, content in incoming_pairs if role == "user"), None
        )
        if conversation.title == "新调查" and first_user_message:
            conversation.title = first_user_message.strip()[:80]
        conversation.updated_at = datetime.now(timezone.utc)
        self.session.commit()

    def attach_intake(
        self, conversation: Conversation, intake_session: IntakeSession
    ) -> None:
        intake_session.tenant_id = conversation.tenant_id
        intake_session.owner_id = conversation.owner_id
        intake_session.conversation_id = conversation.id
        conversation.intake_session_id = intake_session.id
        self.session.flush()

    def attach_task(self, conversation: Conversation, task: ResearchTask) -> None:
        task.tenant_id = conversation.tenant_id
        task.owner_id = conversation.owner_id
        task.conversation_id = conversation.id
        conversation.latest_task_id = task.id
        conversation.updated_at = datetime.now(timezone.utc)
        self.session.flush()
