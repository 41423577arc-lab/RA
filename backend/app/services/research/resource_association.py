from datetime import date, datetime

from app.schemas.task import (
    AssociationAnalysis,
    ConfirmedContext,
    EvidenceBackedItem,
    ProjectRanking,
    ProjectResult,
    PublicClaim,
)


LOW_RELEVANCE_THRESHOLD = 40
STALE_ACTIVITY_DAYS = 180


class ResourceAssociationBuilder:
    version = "resource_association_v1"

    def build(
        self,
        context: ConfirmedContext,
        evidence: list[PublicClaim],
        projects: list[ProjectResult],
        rankings: list[ProjectRanking],
        *,
        reference_date: date | datetime | None = None,
    ) -> AssociationAnalysis:
        ordered = _ranked_projects(projects, rankings)
        effective_date = _reference_date(ordered, reference_date)
        related_projects = [
            _related_project(project, ranking) for project, ranking in ordered
        ]
        customer_contacts = _customer_contacts(ordered)
        internal_owners = _internal_owners(ordered)
        gaps = _information_gaps(context, evidence, ordered)
        risks = _risk_flags(ordered, effective_date)
        return AssociationAnalysis(
            key_findings=[],
            related_projects=related_projects,
            available_resources=[*customer_contacts, *internal_owners],
            recommended_topics=[],
            risks=risks,
            information_gaps=gaps,
            next_actions=[],
        )


def _ranked_projects(
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


def _related_project(
    project: ProjectResult,
    ranking: ProjectRanking,
) -> EvidenceBackedItem:
    details = [
        f"项目：{project.project_name}",
        f"客户：{project.customer_name}",
        f"状态：{project.status}",
        f"排序：{ranking.rank}",
        f"评分：{ranking.score}",
    ]
    if project.project_stage:
        details.append(f"阶段：{project.project_stage}")
    return EvidenceBackedItem(
        text="；".join(details),
        statement_type="FACT",
        evidence_refs=[f"PROJECT:{project.project_id}"],
        confidence=1.0,
    )


def _customer_contacts(
    ordered: list[tuple[ProjectResult, ProjectRanking]],
) -> list[EvidenceBackedItem]:
    contacts: dict[tuple[str, str, str, str], dict] = {}
    for project, _ in ordered:
        if not project.contact_name:
            continue
        key = (
            project.contact_name,
            project.customer_name,
            project.customer_contact_title or "",
            project.customer_contact_phone or "",
        )
        entry = contacts.setdefault(
            key,
            {
                "name": project.contact_name,
                "title": project.customer_contact_title,
                "phone": project.customer_contact_phone,
                "customers": [],
                "refs": [],
            },
        )
        entry["title"] = entry["title"] or project.customer_contact_title
        entry["phone"] = entry["phone"] or project.customer_contact_phone
        _append_unique(entry["customers"], project.customer_name)
        _append_unique(entry["refs"], f"PROJECT:{project.project_id}")

    output = []
    for entry in contacts.values():
        contact = f"客户联系人为{entry['name']}"
        if entry["title"]:
            contact += f"（{entry['title']}）"
        if entry["phone"]:
            contact += f"，联系电话 {entry['phone']}"
        details = [contact, f"所属客户：{'、'.join(entry['customers'])}"]
        output.append(
            EvidenceBackedItem(
                text="；".join(details),
                statement_type="FACT",
                evidence_refs=entry["refs"],
                confidence=1.0,
            )
        )
    return output


def _internal_owners(
    ordered: list[tuple[ProjectResult, ProjectRanking]],
) -> list[EvidenceBackedItem]:
    owners: dict[tuple[str, ...], dict] = {}
    for project, _ in ordered:
        key = (
            ("SALES_REP_ID", project.sales_rep_id)
            if project.sales_rep_id
            else (
                "CONTACT_FIELDS",
                project.owner_name,
                project.owner_phone or "",
                project.owner_email or "",
            )
        )
        entry = owners.setdefault(
            key,
            {
                "name": project.owner_name,
                "phone": project.owner_phone,
                "email": project.owner_email,
                "manager": project.owner_manager_name,
                "region": project.owner_region,
                "refs": [],
            },
        )
        for field, value in (
            ("phone", project.owner_phone),
            ("email", project.owner_email),
            ("manager", project.owner_manager_name),
            ("region", project.owner_region),
        ):
            entry[field] = entry[field] or value
        _append_unique(entry["refs"], f"PROJECT:{project.project_id}")

    output = []
    for entry in owners.values():
        details = [f"我方项目销售员为{entry['name']}"]
        if entry["phone"]:
            details.append(f"联系电话 {entry['phone']}")
        if entry["email"]:
            details.append(f"邮箱 {entry['email']}")
        if entry["manager"] and entry["region"]:
            details.append(f"上级为{entry['manager']}，负责 {entry['region']}")
        elif entry["manager"]:
            details.append(f"上级为{entry['manager']}")
        elif entry["region"]:
            details.append(f"负责区域 {entry['region']}")
        output.append(
            EvidenceBackedItem(
                text="；".join(details),
                statement_type="FACT",
                evidence_refs=entry["refs"],
                confidence=1.0,
            )
        )
    return output


def _information_gaps(
    context: ConfirmedContext,
    evidence: list[PublicClaim],
    ordered: list[tuple[ProjectResult, ProjectRanking]],
) -> list[EvidenceBackedItem]:
    output = []
    verified_subjects = {_normalize(item.subject) for item in evidence}
    for entity in context.entities:
        if (
            entity.entity_type == "PERSON"
            and _normalize(entity.canonical_name) not in verified_subjects
        ):
            output.append(
                _item(
                    f"MISSING_VERIFIED_PERSON_EVIDENCE:{entity.canonical_name}",
                    ["INPUT:ORIGINAL"],
                )
            )
    if not ordered:
        output.append(_item("NO_RANKED_PROJECTS", ["INPUT:ORIGINAL"]))
    for project, _ in ordered:
        project_ref = [f"PROJECT:{project.project_id}"]
        if not project.contact_name:
            output.append(
                _item(
                    f"MISSING_CUSTOMER_CONTACT:{project.project_id}",
                    project_ref,
                )
            )
        if not project.owner_phone and not project.owner_email:
            output.append(
                _item(
                    f"MISSING_INTERNAL_OWNER_CONTACT:{project.project_id}",
                    project_ref,
                )
            )
    return output


def _risk_flags(
    ordered: list[tuple[ProjectResult, ProjectRanking]],
    reference_date: date,
) -> list[EvidenceBackedItem]:
    output = []
    for project, ranking in ordered:
        refs = [f"PROJECT:{project.project_id}"]
        if project.match_type in {"TEXT_MATCH", "VECTOR_MATCH"}:
            output.append(_item(f"FUZZY_PROJECT_MATCH:{project.project_id}", refs))
        if ranking.relevance_score < LOW_RELEVANCE_THRESHOLD:
            output.append(_item(f"LOW_RELEVANCE_SCORE:{project.project_id}", refs))
        if project.health_status in {"AMBER", "RED"}:
            output.append(
                _item(
                    f"PROJECT_HEALTH_{project.health_status}:{project.project_id}",
                    refs,
                )
            )
        if (
            project.status == "ACTIVE"
            and project.last_activity_date is not None
            and (reference_date - project.last_activity_date).days
            > STALE_ACTIVITY_DAYS
        ):
            output.append(_item(f"STALE_PROJECT_ACTIVITY:{project.project_id}", refs))
    return output


def _item(code: str, refs: list[str]) -> EvidenceBackedItem:
    return EvidenceBackedItem(
        text=code,
        statement_type="FACT",
        evidence_refs=refs,
        confidence=1.0,
    )


def _reference_date(
    ordered: list[tuple[ProjectResult, ProjectRanking]],
    value: date | datetime | None,
) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    dates = [
        project.last_activity_date or project.start_date
        for project, _ in ordered
        if project.last_activity_date or project.start_date
    ]
    return max(dates, default=date.min)


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _normalize(value: str | None) -> str:
    return "".join((value or "").split()).casefold()
