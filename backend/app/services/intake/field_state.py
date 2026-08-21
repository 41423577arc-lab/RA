from app.schemas.intake import IntakeFieldState, IntakeStructuredContext


def derive_field_states(
    context: IntakeStructuredContext, *, final_confirmed: bool = False
) -> dict[str, IntakeFieldState]:
    states = {
        "people": _identity_state(
            context, "PERSON", context.people, required=not context.organizations
        ),
        "organizations": _identity_state(
            context, "ORGANIZATION", context.organizations, required=not context.people
        ),
        "people_details": _optional_state(
            any(item.title or item.organization for item in context.people_details)
        ),
        "event_type": _required_state(context.event_type),
        "event_time": _optional_state(context.event_time),
        "event_location": _optional_state(context.event_location),
        "projects": _optional_state(context.projects),
        "business_directions": _optional_state(context.business_directions),
        "focus_questions": _optional_state(context.focus_questions),
    }
    if final_confirmed:
        states = {
            name: state.model_copy(update={"status": "USER_CONFIRMED"})
            if state.status == "STANDARD_COMPLETE"
            else state
            for name, state in states.items()
        }
    return states


def fields_ready_for_confirmation(states: dict[str, IntakeFieldState]) -> bool:
    return all(
        state.status not in {"MISSING", "NEEDS_COMPLETION"}
        for state in states.values()
    )


def fallback_confirmation_question(context: IntakeStructuredContext) -> str:
    people = _resolved_people(context)
    organizations = _resolved_organizations(context)
    targets: list[str] = []
    used_organizations: set[str] = set()
    for person in people:
        organization = person.get("organization")
        title = person.get("title")
        name = person["name"]
        if organization:
            used_organizations.add(organization)
            targets.append(f"{organization}的{title or ''}{name}")
        else:
            targets.append(f"{title or ''}{name}")
    targets.extend(name for name in organizations if name not in used_organizations)

    event_phrase = {
        "宴请": "见面吃饭",
        "拜访": "进行拜访",
        "会议": "开会沟通",
        "其他": "沟通相关事项",
    }.get(context.event_type, "沟通相关事项")
    when = f"{context.event_time}" if context.event_time else ""
    where = f"在{context.event_location}" if context.event_location else ""
    target_text = "、".join(targets) or "上述对象"
    focus = [*context.business_directions, *context.projects, *context.focus_questions]
    focus_text = f"，重点关注{'、'.join(dict.fromkeys(focus))}" if focus else ""
    return f"请确认一下：您{when}{where}要与{target_text}{event_phrase}{focus_text}，对吗？"


def _identity_state(
    context: IntakeStructuredContext,
    entity_type: str,
    values: list[str],
    *,
    required: bool,
) -> IntakeFieldState:
    if not values:
        return IntakeFieldState(
            status="MISSING" if required else "NOT_PROVIDED",
            required=required,
            reason="尚未提供目标身份" if required else "本次未提供",
        )
    resolved = {
        value
        for item in context.entity_resolutions
        if item.entity_type == entity_type and item.confirmed_by
        for value in (item.mention, item.canonical_name)
        if value
    }
    assessments = {
        item.mention: item.is_standard
        for item in context.entity_assessments
        if item.entity_type == entity_type
    }
    incomplete = [
        value
        for value in values
        if value not in resolved and not assessments.get(value, False)
    ]
    if incomplete:
        return IntakeFieldState(
            status="NEEDS_COMPLETION",
            required=required,
            reason=f"待补全或确认：{'、'.join(incomplete)}",
        )
    return IntakeFieldState(
        status="STANDARD_COMPLETE",
        required=required,
        reason="名称已标准化",
    )


def _required_state(value: object) -> IntakeFieldState:
    return IntakeFieldState(
        status="STANDARD_COMPLETE" if value else "MISSING",
        required=True,
        reason="已结构化" if value else "尚未明确活动类型",
    )


def _optional_state(value: object) -> IntakeFieldState:
    return IntakeFieldState(
        status="STANDARD_COMPLETE" if value else "NOT_PROVIDED",
        required=False,
        reason="已提供" if value else "可选字段未提供",
    )


def _resolved_people(context: IntakeStructuredContext) -> list[dict[str, str | None]]:
    resolutions = [
        item
        for item in context.entity_resolutions
        if item.entity_type == "PERSON"
    ]
    if resolutions:
        return [
            {
                "name": item.canonical_name,
                "organization": item.organization,
                "title": item.title,
            }
            for item in resolutions
        ]
    details = {item.name: item for item in context.people_details if item.name}
    return [
        {
            "name": name,
            "organization": details[name].organization if name in details else None,
            "title": details[name].title if name in details else None,
        }
        for name in context.people
    ]


def _resolved_organizations(context: IntakeStructuredContext) -> list[str]:
    resolved = [
        item.canonical_name
        for item in context.entity_resolutions
        if item.entity_type == "ORGANIZATION"
    ]
    return list(dict.fromkeys(resolved or context.organizations))
