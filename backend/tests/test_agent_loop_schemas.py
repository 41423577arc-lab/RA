from datetime import date

import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.task import (
    AgentAction,
    AgentContext,
    AgentPhase,
    ConfirmedContext,
    Observation,
    ProjectResult,
    PublicClaim,
)


def test_agent_phase_and_action_are_closed_contracts() -> None:
    assert TypeAdapter(AgentPhase).validate_python("PUBLIC_RESEARCH") == "PUBLIC_RESEARCH"
    assert TypeAdapter(AgentAction).validate_python("SEARCH_PUBLIC") == "SEARCH_PUBLIC"

    with pytest.raises(ValidationError):
        TypeAdapter(AgentPhase).validate_python("WEB_SEARCHING")
    with pytest.raises(ValidationError):
        TypeAdapter(AgentAction).validate_python("CALL_TOOL")


def test_agent_context_reuses_confirmed_evidence_and_project_schemas() -> None:
    confirmed = ConfirmedContext(
        intents=["MEETING_PREPARATION"],
        entities=[],
        event_type="会议",
    )
    claim = PublicClaim(
        web_result_id="W001",
        evidence_id="E001",
        subject="示例企业",
        claim="示例企业发布了公开信息",
        evidence_quote="示例企业发布了公开信息。",
        source_title="公开页面",
        source_url="https://example.com/source",
    )
    project = ProjectResult(
        project_id="P001",
        project_name="示例项目",
        customer_name="示例企业",
        status="ACTIVE",
        owner_name="张伟",
        start_date=date(2026, 1, 1),
        description="示例项目描述",
        match_type="ORG_EXACT",
    )
    observation = Observation(
        phase="PUBLIC_RESEARCH",
        action="SEARCH_PUBLIC",
        status="SUCCESS",
        summary="发现一条已核验公开事实",
        evidence_refs=["WEB:W001:E001"],
    )

    context = AgentContext(
        phase="PROJECT_RESEARCH",
        user_input="准备与示例企业开会",
        confirmed_context=confirmed,
        public_evidence=[claim],
        project_results=[project],
        observations=[observation],
    )

    assert context.confirmed_context is confirmed
    assert context.public_evidence[0] is claim
    assert context.project_results[0] is project
    assert context.observations[0] is observation
    assert set(AgentContext.model_fields) == {
        "phase",
        "user_input",
        "confirmed_context",
        "identity_candidates",
        "public_evidence",
        "project_results",
        "information_gaps",
        "recent_messages",
        "observations",
    }
    assert set(Observation.model_fields) == {
        "phase",
        "action",
        "status",
        "summary",
        "result_refs",
        "evidence_refs",
        "project_ids",
    }
