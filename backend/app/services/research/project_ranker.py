import math
import re
from datetime import date, datetime

from app.schemas.task import ConfirmedContext, ProjectRanking, ProjectResult
from app.services.research.agent_nodes import organization_aliases


MATCH_TYPE_WEIGHTS = {
    "PERSON_EXACT": 35,
    "PROJECT_EXACT": 32,
    "ORG_EXACT": 28,
    "TEXT_MATCH": 12,
    "VECTOR_MATCH": 5,
}
STATUS_WEIGHTS = {"ACTIVE": 8, "COMPLETED": 2}
STAGE_WEIGHTS = {
    "NEGOTIATION": 7,
    "PROPOSAL": 6,
    "SOLUTION": 5,
    "QUALIFICATION": 4,
    "DISCOVERY": 3,
    "DELIVERY": 3,
    "CLOSED_WON": 1,
}
PRIORITY_WEIGHTS = {"P0": 6, "P1": 4, "P2": 2, "P3": 0}
HEALTH_WEIGHTS = {"GREEN": 3, "AMBER": 2, "RED": 0}
CONTEXT_PERSON_WEIGHT = 12
CONTEXT_ORGANIZATION_WEIGHT = 10
CONTEXT_PROJECT_WEIGHT = 12
BUSINESS_TERM_WEIGHT = 4
BUSINESS_TERM_LIMIT = 2
SIMILARITY_WEIGHT = 10


class ProjectRanker:
    version = "project_ranker_v1"

    def rank(
        self,
        projects: list[ProjectResult],
        context: ConfirmedContext,
        *,
        reference_date: date | datetime | None = None,
    ) -> list[ProjectRanking]:
        unique_projects = _unique_projects(projects)
        effective_date = _reference_date(unique_projects, reference_date)
        scored = [
            self._score(project, context, effective_date)
            for project in unique_projects
        ]
        activity_by_id = {
            project.project_id: project.last_activity_date
            for project in unique_projects
        }
        scored.sort(
            key=lambda item: (
                -item.relevance_score,
                -_date_ordinal(activity_by_id[item.project_id]),
                item.project_id,
            )
        )
        return [item.model_copy(update={"rank": index}) for index, item in enumerate(scored, 1)]

    def _score(
        self,
        project: ProjectResult,
        context: ConfirmedContext,
        reference_date: date,
    ) -> ProjectRanking:
        components: list[tuple[str, int]] = []
        _add(
            components,
            f"MATCH_{project.match_type}",
            MATCH_TYPE_WEIGHTS[project.match_type],
        )

        people, organizations, project_names = _context_entities(context)
        if _normalize(project.contact_name) in people:
            _add(components, "CONTEXT_PERSON_EXACT", CONTEXT_PERSON_WEIGHT)
        if _organization_matches(project.customer_name, organizations):
            _add(
                components,
                "CONTEXT_ORGANIZATION_EXACT",
                CONTEXT_ORGANIZATION_WEIGHT,
            )
        if _project_matches(project, project_names):
            _add(components, "CONTEXT_PROJECT_EXACT", CONTEXT_PROJECT_WEIGHT)

        searchable = _normalize(
            " ".join(
                [
                    project.project_name,
                    *project.project_aliases,
                    project.customer_name,
                    project.description,
                ]
            )
        )
        matched_terms = []
        for term in context.business_directions:
            normalized = _normalize(term)
            if normalized and normalized in searchable and normalized not in matched_terms:
                matched_terms.append(normalized)
            if len(matched_terms) == BUSINESS_TERM_LIMIT:
                break
        for term in matched_terms:
            _add(components, f"CONTEXT_BUSINESS_TERM:{term}", BUSINESS_TERM_WEIGHT)

        similarity = project.similarity
        if similarity is not None and math.isfinite(similarity):
            bounded_similarity = min(1.0, max(0.0, similarity))
            similarity_points = int(bounded_similarity * SIMILARITY_WEIGHT + 0.5)
            _add(
                components,
                f"SIMILARITY_{bounded_similarity:.2f}",
                similarity_points,
            )

        _add(components, f"STATUS_{project.status}", STATUS_WEIGHTS[project.status])
        if project.project_stage:
            _add(
                components,
                f"STAGE_{project.project_stage}",
                STAGE_WEIGHTS.get(project.project_stage, 0),
            )
        if project.priority:
            _add(
                components,
                f"PRIORITY_{project.priority}",
                PRIORITY_WEIGHTS[project.priority],
            )
        if project.health_status:
            _add(
                components,
                f"HEALTH_{project.health_status}",
                HEALTH_WEIGHTS[project.health_status],
            )
        recency_code, recency_points = _recency_score(
            project.last_activity_date,
            reference_date,
        )
        if recency_code:
            _add(components, recency_code, recency_points)

        raw_score = sum(points for _, points in components)
        score = min(100, max(0, raw_score))
        reason_codes = [f"{code}:+{points}" for code, points in components]
        if raw_score > 100:
            reason_codes.append("CAP_100")
        return ProjectRanking(
            project_id=project.project_id,
            relevance_score=score,
            score=score,
            reason_codes=reason_codes,
            rank=1,
            relevance_reason="；".join(reason_codes),
            recommended_use="",
            related_internal_resource=project.owner_name,
            confidence=1.0,
            evidence_refs=[f"PROJECT:{project.project_id}"],
        )


def _context_entities(
    context: ConfirmedContext,
) -> tuple[set[str], set[str], set[str]]:
    people = set()
    organizations = set()
    projects = set()
    for entity in context.entities:
        names = {_normalize(entity.canonical_name), *map(_normalize, entity.aliases)}
        if entity.entity_type == "PERSON":
            people.update(names)
            organizations.update(
                _normalize(item) for item in organization_aliases(entity.organization)
            )
        elif entity.entity_type == "ORGANIZATION":
            for name in names:
                organizations.update(
                    _normalize(item) for item in organization_aliases(name)
                )
        elif entity.entity_type == "PROJECT":
            projects.update(names)
    return people - {""}, organizations - {""}, projects - {""}


def _organization_matches(customer_name: str, organizations: set[str]) -> bool:
    customer_aliases = {
        _normalize(item) for item in organization_aliases(customer_name)
    }
    return bool(customer_aliases & organizations)


def _project_matches(project: ProjectResult, project_names: set[str]) -> bool:
    names = {
        _normalize(project.project_name),
        *map(_normalize, project.project_aliases),
    }
    return bool((names - {""}) & project_names)


def _recency_score(
    last_activity_date: date | None,
    reference_date: date,
) -> tuple[str | None, int]:
    if last_activity_date is None:
        return None, 0
    age_days = max(0, (reference_date - last_activity_date).days)
    if age_days <= 30:
        return "ACTIVITY_WITHIN_30D", 6
    if age_days <= 90:
        return "ACTIVITY_WITHIN_90D", 4
    if age_days <= 180:
        return "ACTIVITY_WITHIN_180D", 2
    return None, 0


def _reference_date(
    projects: list[ProjectResult],
    value: date | datetime | None,
) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    project_dates = [
        item.last_activity_date or item.start_date
        for item in projects
        if item.last_activity_date or item.start_date
    ]
    return max(project_dates, default=date.min)


def _unique_projects(projects: list[ProjectResult]) -> list[ProjectResult]:
    output = []
    seen = set()
    for raw in projects:
        project = ProjectResult.model_validate(raw)
        if project.project_id in seen:
            continue
        seen.add(project.project_id)
        output.append(project)
    return output


def _date_ordinal(value: date | None) -> int:
    return value.toordinal() if value else 0


def _normalize(value: str | None) -> str:
    return re.sub(r"\s+", "", value or "").casefold()


def _add(components: list[tuple[str, int]], code: str, points: int) -> None:
    if points:
        components.append((code, points))
