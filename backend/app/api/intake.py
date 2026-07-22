from fastapi import APIRouter, HTTPException

from app.config import settings
from app.schemas.intake import IntakeChatRequest, IntakeChatResponse
from app.services.intake_agent import IntakeAgent
from app.services.llm_client import LLMCallFailed, LLMUnavailable, StructuredLLM


router = APIRouter(prefix="/api/v1/intake", tags=["intake"])
intake_agent = IntakeAgent(StructuredLLM(settings))


@router.post("/chat", response_model=IntakeChatResponse)
def chat(request: IntakeChatRequest) -> IntakeChatResponse:
    try:
        result = intake_agent.respond(request)
    except (LLMUnavailable, LLMCallFailed) as exc:
        raise HTTPException(status_code=503, detail="信息采集助手暂时不可用，请稍后重试") from exc
    return IntakeChatResponse(session_id=request.session_id, **result.model_dump())
