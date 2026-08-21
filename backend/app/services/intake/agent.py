from app.schemas.intake import (
    ExternalIdentityNormalizationResult,
    IntakeFinalConfirmationResult,
    IntakeChatRequest,
    IntakeChatResult,
    IntakeFollowupResult,
    IntakeReadinessResult,
    IntakeStructuredContext,
)
from app.services.intake.defaults import DEFAULT_REQUESTER_CONTEXT
from app.services.integrations.llm_client import StructuredLLM


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

    def initialize_context(
        self,
        request: IntakeChatRequest,
        extracted_context: IntakeStructuredContext,
    ) -> IntakeStructuredContext:
        return self.llm.parse(
            str(request.session_id),
            "intake_identity_initialize",
            {
                "messages": [message.model_dump() for message in request.messages],
                "latest_user_reply": self._latest_user_reply(request),
                "extracted_context": extracted_context.model_dump(mode="json"),
                "default_requester_context": DEFAULT_REQUESTER_CONTEXT,
            },
            IntakeStructuredContext,
        )

    def update_context(
        self,
        request: IntakeChatRequest,
        old_context: IntakeStructuredContext,
        *,
        extracted_context: IntakeStructuredContext | None = None,
        tool_observation: dict | None = None,
    ) -> IntakeStructuredContext:
        return self.llm.parse(
            str(request.session_id),
            "intake_identity_update",
            {
                "messages": [message.model_dump() for message in request.messages],
                "latest_user_reply": self._latest_user_reply(request),
                "old_context": old_context.model_dump(mode="json"),
                "extracted_context": extracted_context.model_dump(mode="json")
                if extracted_context
                else None,
                "tool_observation": tool_observation,
                "previous_success_criteria": old_context.success_criteria,
                "default_requester_context": DEFAULT_REQUESTER_CONTEXT,
            },
            IntakeStructuredContext,
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

    def assess_readiness(
        self,
        request: IntakeChatRequest,
        structured_context: IntakeStructuredContext,
        tool_observation: dict,
    ) -> IntakeReadinessResult:
        return self.llm.parse(
            str(request.session_id),
            "intake_readiness",
            {
                "messages": [message.model_dump() for message in request.messages],
                "structured_context": structured_context.model_dump(mode="json"),
                "tool_observation": tool_observation,
                "default_requester_context": DEFAULT_REQUESTER_CONTEXT,
            },
            IntakeReadinessResult,
        )

    def summarize_for_confirmation(
        self,
        request: IntakeChatRequest,
        structured_context: IntakeStructuredContext,
        analysis_input: str,
    ) -> IntakeFinalConfirmationResult:
        return self.llm.parse(
            str(request.session_id),
            "intake_final_confirmation",
            {
                "messages": [message.model_dump() for message in request.messages],
                "structured_context": structured_context.model_dump(mode="json"),
                "analysis_input": analysis_input,
                "default_requester_context": DEFAULT_REQUESTER_CONTEXT,
            },
            IntakeFinalConfirmationResult,
        )

    @staticmethod
    def _latest_user_reply(request: IntakeChatRequest) -> str:
        return next(
            (
                message.content
                for message in reversed(request.messages)
                if message.role == "user"
            ),
            "",
        )
