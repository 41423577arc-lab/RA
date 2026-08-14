from app.schemas.task import (
    ActionBrief,
    AssociationAnalysis,
    ConfirmedContext,
    EvidenceBackedItem,
    GeneratedReportContent,
    ProjectRanking,
    ProjectResult,
    PublicClaim,
)
from app.services.agent_nodes import (
    build_person_identity_summaries,
    businessize_text,
    validate_report_content,
)


def ordered_ranked_projects(
    projects: list[ProjectResult],
    rankings: list[ProjectRanking],
) -> list[tuple[ProjectResult, ProjectRanking]]:
    project_by_id = {}
    for raw in projects:
        project = ProjectResult.model_validate(raw)
        project_by_id.setdefault(project.project_id, project)
    output = []
    seen = set()
    for raw in sorted(
        (ProjectRanking.model_validate(item) for item in rankings),
        key=lambda item: (
            item.rank if item.rank is not None else 1_000_000,
            -item.relevance_score,
            item.project_id,
        ),
    ):
        project = project_by_id.get(raw.project_id)
        if project is None or raw.project_id in seen:
            continue
        seen.add(raw.project_id)
        output.append((project, raw))
    return output


def build_final_synthesis_input(
    context: ConfirmedContext,
    evidence: list[PublicClaim],
    ranked_projects: list[tuple[ProjectResult, ProjectRanking]],
    association: AssociationAnalysis,
) -> dict:
    return {
        "confirmed_context": context.model_dump(mode="json"),
        "verified_evidence": [item.model_dump(mode="json") for item in evidence],
        "ranked_projects": [
            {
                "rank": ranking.rank,
                "score": ranking.score,
                "reason_codes": ranking.reason_codes,
                "project": project.model_dump(mode="json"),
            }
            for project, ranking in ranked_projects
        ],
        "deterministic_association": {
            "related_projects": [
                item.model_dump(mode="json") for item in association.related_projects
            ],
            "customer_and_internal_resources": [
                item.model_dump(mode="json")
                for item in association.available_resources
            ],
            "risk_flags": [
                item.model_dump(mode="json") for item in association.risks
            ],
        },
        "information_gaps": [
            item.model_dump(mode="json") for item in association.information_gaps
        ],
    }


def validate_final_synthesis(
    content: GeneratedReportContent,
    context: ConfirmedContext,
    evidence: list[PublicClaim],
    ranked_projects: list[tuple[ProjectResult, ProjectRanking]],
    association: AssociationAnalysis,
) -> GeneratedReportContent:
    projects = [project for project, _ in ranked_projects]
    validated = validate_report_content(content, evidence, projects, context)
    priority_by_ref = {
        item.evidence_refs[0]: _businessize_item(item)
        for item in association.related_projects
        if item.evidence_refs
    }
    ordered_priority = []
    for project, _ in ranked_projects:
        item = priority_by_ref.get(f"PROJECT:{project.project_id}")
        if item is not None:
            ordered_priority.append(item)
        if len(ordered_priority) == 3:
            break

    people = {
        item.canonical_name
        for item in context.entities
        if item.entity_type == "PERSON"
    }
    owners = {project.owner_name for project in projects}
    brief = validated.action_brief
    grounded_brief = brief.model_copy(
        update={
            "destination": context.event_location,
            "meeting_people": [item for item in brief.meeting_people if item in people],
            "internal_contacts": [
                item for item in brief.internal_contacts if item in owners
            ],
            "risks": [businessize_text(item.text) for item in association.risks],
        }
    )
    return validated.model_copy(
        update={
            "person_and_company_summary": build_person_identity_summaries(
                context,
                evidence,
                [],
            ),
            "public_information_summary": [
                EvidenceBackedItem(
                    text=businessize_text(claim.claim),
                    statement_type="FACT",
                    evidence_refs=[
                        f"WEB:{claim.web_result_id}:{claim.evidence_id}"
                    ],
                    confidence=claim.confidence,
                )
                for claim in evidence[:6]
            ],
            "priority_projects": ordered_priority,
            "resource_analysis": [
                _businessize_item(item) for item in association.available_resources
            ],
            "recommended_topics": _as_recommendations(
                validated.recommended_topics
            ),
            "advancement_advice": _as_recommendations(
                validated.advancement_advice
            ),
            "preparation_items": _as_recommendations(
                validated.preparation_items
            ),
            "gaps_and_risks": [
                *(_businessize_item(item) for item in association.risks),
                *(
                    _businessize_item(item)
                    for item in association.information_gaps
                ),
            ],
            "action_brief": grounded_brief,
        }
    )


def _as_recommendations(
    items: list[EvidenceBackedItem],
) -> list[EvidenceBackedItem]:
    return [
        item.model_copy(update={"statement_type": "RECOMMENDATION"})
        for item in items
    ]


def _businessize_item(item: EvidenceBackedItem) -> EvidenceBackedItem:
    return item.model_copy(update={"text": businessize_text(item.text)})
