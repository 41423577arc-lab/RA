from app.schemas.intake import IntakeChatRequest, IntakeChatResult
from app.services.llm_client import StructuredLLM


class IntakeAgent:
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def respond(self, request: IntakeChatRequest) -> IntakeChatResult:
        return self.llm.parse(
            str(request.session_id),
            "intake_chat",
            {"messages": [message.model_dump() for message in request.messages]},
            IntakeChatResult,
        )
