from app.schemas.intake import (
    IntakeEntityResolution,
    IntakeStructuredContext,
    IntakeToolAttempt,
)
from app.services.intake.agent_loop import AgentState, AgentTurn, ToolObservation


class IntakeStateReducer:
    """用经过 Schema 校验的输入生成新的不可变运行态。"""

    _PRESERVED_COLLECTION_FIELDS = {
        "people",
        "people_details",
        "organizations",
        "projects",
        "business_directions",
        "focus_questions",
        "entity_assessments",
        "field_states",
    }
    _PRESERVED_NULLABLE_FIELDS = {
        "event_type",
        "event_time",
        "event_location",
        "resolution_result",
    }

    @staticmethod
    def apply_turn(state: AgentState, turn: AgentTurn) -> AgentState:
        context_data = state.context.model_dump(mode="python")
        updates = dict(turn.context_patch.updates)
        for field_name in IntakeStateReducer._PRESERVED_COLLECTION_FIELDS:
            if not updates.get(field_name) and context_data.get(field_name):
                updates.pop(field_name, None)
        for field_name in IntakeStateReducer._PRESERVED_NULLABLE_FIELDS:
            if (
                updates.get(field_name) is None
                and context_data.get(field_name) is not None
            ):
                updates.pop(field_name, None)
        context_data.update(updates)
        context_data["next_action"] = turn.next_action
        context_data["user_question"] = (
            turn.user_message if turn.next_action == "ASK_USER" else None
        )
        context = IntakeStructuredContext.model_validate(context_data)
        return AgentState(
            context=context,
            latest_observation=state.latest_observation,
            loop_count=state.loop_count + 1,
            llm_turn_count=state.llm_turn_count + 1,
        )

    @staticmethod
    def apply_observation(
        state: AgentState,
        observation: ToolObservation,
    ) -> AgentState:
        additions = IntakeStateReducer._expand_relationship_resolutions(
            observation.resolutions
        )
        resolutions = IntakeStateReducer._merge_resolutions(
            state.context.entity_resolutions,
            additions,
        )
        linked_organizations = [
            item.organization
            for item in additions
            if item.entity_type == "PERSON" and item.organization
        ]
        resolved_people = [
            item.canonical_name
            for item in additions
            if item.entity_type == "PERSON"
        ]
        attempt = IntakeToolAttempt(
            action=observation.action,
            target_fields=observation.target_fields,
            query=observation.executed_query,
            technical_status=observation.technical_status,
            information_status=observation.information_status,
            observation=observation.error or observation.summary,
        )
        context_data = state.context.model_dump(mode="python")
        context_data.update(
            {
                "entity_resolutions": resolutions[-40:],
                "people": IntakeStateReducer._replace_resolved_aliases(
                    state.context.people,
                    resolved_people,
                ),
                "organizations": list(
                    dict.fromkeys(
                        [*state.context.organizations, *linked_organizations]
                    )
                ),
                "tool_attempts": [*state.context.tool_attempts, attempt][-20:],
            }
        )
        context = IntakeStructuredContext.model_validate(context_data)
        return AgentState(
            context=context,
            latest_observation=observation,
            loop_count=state.loop_count,
            llm_turn_count=state.llm_turn_count,
        )

    @staticmethod
    def preserve_after_llm_failure(state: AgentState) -> AgentState:
        return AgentState(
            context=state.context.model_copy(deep=True),
            latest_observation=state.latest_observation,
            loop_count=state.loop_count,
            llm_turn_count=state.llm_turn_count + 1,
        )

    @staticmethod
    def _merge_resolutions(
        existing: list[IntakeEntityResolution],
        additions: list[IntakeEntityResolution],
    ) -> list[IntakeEntityResolution]:
        merged = {
            (item.entity_type, item.canonical_name, item.organization): item
            for item in existing
        }
        for item in additions:
            merged[(item.entity_type, item.canonical_name, item.organization)] = item
        return list(merged.values())

    @staticmethod
    def _expand_relationship_resolutions(
        resolutions: list[IntakeEntityResolution],
    ) -> list[IntakeEntityResolution]:
        expanded = list(resolutions)
        existing_organizations = {
            item.canonical_name
            for item in resolutions
            if item.entity_type == "ORGANIZATION"
        }
        for item in resolutions:
            if (
                item.entity_type != "PERSON"
                or not item.organization
                or item.organization in existing_organizations
            ):
                continue
            # 受控人物候选中的所属关系同时构成企业身份依据。
            expanded.append(
                IntakeEntityResolution(
                    entity_type="ORGANIZATION",
                    canonical_name=item.organization,
                    mention=item.organization,
                    confidence=item.confidence,
                    confirmed_by=item.confirmed_by,
                    source_url=item.source_url,
                    evidence_quote=item.evidence_quote,
                )
            )
            existing_organizations.add(item.organization)
        return expanded

    @staticmethod
    def _replace_resolved_aliases(
        people: list[str],
        resolved_people: list[str],
    ) -> list[str]:
        title_suffixes = (
            "副董事长",
            "董事长",
            "副总经理",
            "总经理",
            "副总裁",
            "总裁",
            "负责人",
            "经理",
            "主任",
            "领导",
            "总",
            "董",
        )
        output = list(people)
        for canonical_name in resolved_people:
            matching_aliases: list[str] = []
            for mention in output:
                for title in title_suffixes:
                    if not mention.endswith(title):
                        continue
                    fragment = mention[: -len(title)]
                    if fragment and canonical_name.startswith(fragment):
                        matching_aliases.append(mention)
                    break
            if len(matching_aliases) == 1:
                output = [item for item in output if item != matching_aliases[0]]
            if canonical_name not in output:
                output.append(canonical_name)
        return output
