from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from app.schemas.task import (
    AgentContext,
    AgentPhase,
    AssociationAnalysis,
    CandidateOption,
    ConfirmationItem,
    ConfirmationRequest,
    ConfirmedContext,
    EvidenceBackedItem,
    Observation,
    ProjectResult,
    PublicClaim,
    TaskChatMessage,
)

if TYPE_CHECKING:
    from app.models.database import ResearchTask


IDENTITY_MESSAGE_LIMIT = 8
IDENTITY_ITEM_LIMIT = 10
IDENTITY_CANDIDATE_LIMIT = 5
PUBLIC_EVIDENCE_LIMIT = 12
PROJECT_EVIDENCE_LIMIT = 8
SYNTHESIS_EVIDENCE_LIMIT = 20
PROJECT_RESULT_LIMIT = 20
SYNTHESIS_PROJECT_LIMIT = 10
INFORMATION_GAP_LIMIT = 10
OBSERVATION_LIMIT = 6


class AgentContextBuilder:
    def build(
        self,
        phase: AgentPhase,
        task: "ResearchTask",
        confirmed_context: ConfirmedContext | None,
        evidence: Sequence[PublicClaim | dict[str, Any]],
        project_results: Sequence[ProjectResult | dict[str, Any]],
        recent_messages: Sequence[TaskChatMessage | dict[str, Any]],
        observations: Sequence[Observation | dict[str, Any]] = (),
    ) -> AgentContext:
        context = _project_confirmed_context(phase, confirmed_context)
        recent_observations = _observations(observations)

        if phase in {"IDENTITY", "WAITING_USER"}:
            return AgentContext(
                phase=phase,
                user_input=_truncate(getattr(task, "input_text", None), 10_000),
                confirmed_context=context,
                identity_candidates=_identity_candidates(task),
                recent_messages=_recent_messages(recent_messages),
                observations=recent_observations,
            )

        if phase == "PUBLIC_RESEARCH":
            return AgentContext(
                phase=phase,
                confirmed_context=context,
                public_evidence=_public_evidence(
                    evidence,
                    limit=PUBLIC_EVIDENCE_LIMIT,
                    quote_limit=600,
                ),
                observations=recent_observations,
            )

        if phase == "PROJECT_RESEARCH":
            return AgentContext(
                phase=phase,
                confirmed_context=context,
                public_evidence=_public_evidence(
                    evidence,
                    limit=PROJECT_EVIDENCE_LIMIT,
                    quote_limit=0,
                ),
                project_results=_projects(project_results, limit=PROJECT_RESULT_LIMIT),
                observations=recent_observations,
            )

        if phase == "SYNTHESIS":
            return AgentContext(
                phase=phase,
                confirmed_context=context,
                public_evidence=_public_evidence(
                    evidence,
                    limit=SYNTHESIS_EVIDENCE_LIMIT,
                    quote_limit=800,
                ),
                project_results=_projects(
                    project_results,
                    limit=SYNTHESIS_PROJECT_LIMIT,
                    rank_order=_project_rank_order(task),
                ),
                information_gaps=_information_gaps(task),
                observations=recent_observations,
            )

        return AgentContext(
            phase=phase,
            confirmed_context=context,
            observations=recent_observations,
        )


def _project_confirmed_context(
    phase: AgentPhase, context: ConfirmedContext | None
) -> ConfirmedContext | None:
    if context is None or phase == "DONE":
        return None
    if phase == "SYNTHESIS":
        return context.model_copy(deep=True)

    allowed_entity_types = {"PERSON", "ORGANIZATION"}
    if phase == "PROJECT_RESEARCH":
        allowed_entity_types.add("PROJECT")
    identities = [
        entity
        for entity in context.entities
        if entity.entity_type in allowed_entity_types
    ]
    updates: dict[str, Any] = {
        "entities": identities,
        "event_time": None,
        "event_location": None,
        "business_directions": [],
        "focus_questions": [],
    }
    if phase == "PUBLIC_RESEARCH":
        updates["focus_questions"] = list(context.focus_questions)
    elif phase == "PROJECT_RESEARCH":
        updates["business_directions"] = list(context.business_directions)
    return context.model_copy(update=updates, deep=True)


def _identity_candidates(task: "ResearchTask") -> ConfirmationRequest | None:
    raw = getattr(task, "confirmation_request", None)
    if not raw:
        return None
    request = ConfirmationRequest.model_validate(raw)
    items = []
    for item in request.items[:IDENTITY_ITEM_LIMIT]:
        candidates = [_trim_candidate(candidate) for candidate in item.candidates]
        items.append(
            ConfirmationItem(
                mention=_truncate(item.mention, 200),
                entity_type=item.entity_type,
                candidates=candidates[:IDENTITY_CANDIDATE_LIMIT],
                required=item.required,
            )
        )
    return request.model_copy(update={"items": items})


def _trim_candidate(candidate: CandidateOption) -> CandidateOption:
    return candidate.model_copy(
        update={
            "reason": _truncate(candidate.reason, 300),
            "evidence_quote": (
                _truncate(candidate.evidence_quote, 500)
                if candidate.evidence_quote
                else None
            ),
        }
    )


def _recent_messages(
    messages: Sequence[TaskChatMessage | dict[str, Any]],
) -> list[TaskChatMessage]:
    output = []
    for raw in messages[-IDENTITY_MESSAGE_LIMIT:]:
        message = TaskChatMessage.model_validate(raw)
        output.append(message.model_copy(update={"content": _truncate(message.content, 2_000)}))
    return output


def _public_evidence(
    evidence: Sequence[PublicClaim | dict[str, Any]],
    *,
    limit: int,
    quote_limit: int,
) -> list[PublicClaim]:
    output = []
    seen = set()
    for raw in evidence:
        claim = PublicClaim.model_validate(raw)
        key = (claim.web_result_id, claim.evidence_id, claim.source_url, claim.claim)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            claim.model_copy(
                update={
                    "subject": _truncate(claim.subject, 200),
                    "claim": _truncate(claim.claim, 500),
                    "evidence_quote": (
                        _truncate(claim.evidence_quote, quote_limit) if quote_limit else ""
                    ),
                    "source_title": _truncate(claim.source_title, 200),
                    "matched_keywords": claim.matched_keywords[:8],
                }
            )
        )
        if len(output) == limit:
            break
    return output


def _projects(
    projects: Sequence[ProjectResult | dict[str, Any]],
    *,
    limit: int,
    rank_order: dict[str, int] | None = None,
) -> list[ProjectResult]:
    output = []
    seen = set()
    for raw in projects:
        project = ProjectResult.model_validate(raw)
        if project.project_id in seen:
            continue
        seen.add(project.project_id)
        output.append(
            project.model_copy(
                update={
                    "description": _truncate(project.description, 600),
                    "project_aliases": project.project_aliases[:10],
                }
            )
        )
    if rank_order is not None:
        output.sort(
            key=lambda item: (
                rank_order.get(item.project_id, len(rank_order)),
                item.project_id,
            )
        )
    return output[:limit]


def _project_rank_order(task: "ResearchTask") -> dict[str, int]:
    order = {}
    for item in getattr(task, "ranked_internal_results", None) or []:
        project_id = item.get("project_id") if isinstance(item, dict) else item.project_id
        if project_id and project_id not in order:
            order[project_id] = len(order)
    return order


def _information_gaps(task: "ResearchTask") -> list[EvidenceBackedItem]:
    raw = getattr(task, "association_analysis", None)
    if not raw:
        return []
    analysis = AssociationAnalysis.model_validate(raw)
    return [
        item.model_copy(update={"text": _truncate(item.text, 500)})
        for item in analysis.information_gaps[:INFORMATION_GAP_LIMIT]
    ]


def _observations(
    observations: Sequence[Observation | dict[str, Any]],
) -> list[Observation]:
    output = []
    for raw in observations[-OBSERVATION_LIMIT:]:
        observation = Observation.model_validate(raw)
        output.append(
            observation.model_copy(
                update={
                    "summary": _truncate(observation.summary, 2_000),
                    "result_refs": observation.result_refs[:20],
                    "evidence_refs": observation.evidence_refs[:20],
                    "project_ids": observation.project_ids[:20],
                }
            )
        )
    return output


def _truncate(value: str | None, limit: int) -> str:
    if not value:
        return ""
    return value[:limit]
