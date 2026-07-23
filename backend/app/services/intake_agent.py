from app.schemas.intake import (
    ExternalIdentityNormalizationResult,
    IntakeChatRequest,
    IntakeChatResult,
    IntakeFollowupResult,
)
from app.services.intake_defaults import DEFAULT_REQUESTER_CONTEXT
from app.services.llm_client import StructuredLLM


class IntakeAgent:
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def respond(self, request: IntakeChatRequest) -> IntakeChatResult:
        return self.llm.parse(
            str(request.session_id),
            "intake_chat",
            {
                "messages": [message.model_dump() for message in request.messages],
                "default_requester_context": DEFAULT_REQUESTER_CONTEXT,
            },
            IntakeChatResult,
        )

    def follow_up(
        self,
        request: IntakeChatRequest,
        decision: IntakeChatResult,
        tool_observation: dict,
    ) -> IntakeFollowupResult:
        return self.llm.parse(
            str(request.session_id),
            "intake_followup",
            {
                "messages": [message.model_dump() for message in request.messages],
                "decision": decision.model_dump(mode="json"),
                "tool_observation": tool_observation,
                "default_requester_context": DEFAULT_REQUESTER_CONTEXT,
            },
            IntakeFollowupResult,
        )

    def normalize_external_identity(
        self,
        request: IntakeChatRequest,
        mentions: list[dict],
        pages: list[dict],
    ) -> ExternalIdentityNormalizationResult:
        return self.llm.parse(
            str(request.session_id),
            "intake_identity_normalize",
            {
                "mentions": mentions,
                "pages": pages,
                "default_requester_context": DEFAULT_REQUESTER_CONTEXT,
            },
            ExternalIdentityNormalizationResult,
        )
