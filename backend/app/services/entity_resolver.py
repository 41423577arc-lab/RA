import csv
from pathlib import Path

from app.schemas.task import (
    CandidateOption,
    ConfirmationItem,
    ConfirmationRequest,
    ConfirmedContext,
    ConfirmedEntity,
    ExtractedInfo,
    IntentUnderstanding,
)


class EntityResolver:
    def __init__(self, seed_dir: Path, confirm_threshold: float = 0.85):
        with (seed_dir / "entities.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        self.confirm_threshold = confirm_threshold

    def resolve(
        self,
        input_text: str,
        understanding: IntentUnderstanding,
        extracted: ExtractedInfo,
        version: int,
        external_candidates: list[CandidateOption] | None = None,
    ) -> tuple[ConfirmedContext | None, ConfirmationRequest | None]:
        external_groups: dict[str, list[CandidateOption]] = {}
        for candidate in external_candidates or []:
            mention = next(
                (
                    token
                    for token in [candidate.canonical_name, *candidate.aliases]
                    if token and token in input_text
                ),
                None,
            )
            if mention:
                external_groups.setdefault(mention, []).append(candidate)
        groups: dict[str, list[dict[str, str]]] = {}
        for row in self.rows:
            person_tokens = [row.get("person_name", ""), row.get("person_alias", "")]
            organization_tokens = [
                row.get("organization_name", ""),
                row.get("organization_alias", ""),
            ]
            person_match = next((token for token in person_tokens if token and token in input_text), None)
            organization_match = next(
                (token for token in organization_tokens if token and token in input_text), None
            )
            if person_match and (not organization_match or row.get("organization_name")):
                groups.setdefault(person_match, []).append(row)

        confirmation_items: list[ConfirmationItem] = []
        confirmed: list[ConfirmedEntity] = []
        seen_ids: set[str] = set()
        for mention, candidates in external_groups.items():
            unique_candidates = {item.candidate_id: item for item in candidates}
            if len(unique_candidates) > 1:
                confirmation_items.append(
                    ConfirmationItem(
                        mention=mention,
                        entity_type="PERSON",
                        candidates=list(unique_candidates.values()),
                    )
                )
                continue
            candidate = next(iter(unique_candidates.values()))
            seen_ids.add(candidate.candidate_id)
            confirmed.append(
                ConfirmedEntity(
                    candidate_id=candidate.candidate_id,
                    entity_type=candidate.entity_type,
                    canonical_name=candidate.canonical_name,
                    aliases=candidate.aliases,
                    organization=candidate.organization,
                    title=candidate.title,
                    region=candidate.region,
                    confirmed_by="AUTO",
                )
            )
        for mention, rows in groups.items():
            if mention in external_groups:
                continue
            unique_rows = {row["candidate_id"]: row for row in rows}
            if len(unique_rows) > 1:
                confirmation_items.append(
                    ConfirmationItem(
                        mention=mention,
                        entity_type="PERSON",
                        candidates=[self._candidate(row, mention) for row in unique_rows.values()],
                    )
                )
                continue
            row = next(iter(unique_rows.values()))
            seen_ids.add(row["candidate_id"])
            confirmed.append(self._confirmed(row, "AUTO"))

        if confirmation_items:
            return None, ConfirmationRequest(version=version, items=confirmation_items)

        for person in understanding.people:
            if person.needs_confirmation or person.confidence < self.confirm_threshold:
                candidates = self._candidates_for_mention(person.mention, input_text)
                if len(candidates) != 1:
                    confirmation_items.append(
                        ConfirmationItem(
                            mention=person.mention,
                            entity_type="PERSON",
                            candidates=candidates,
                        )
                    )
                    continue
            if person.canonical_name and not any(
                entity.canonical_name == person.canonical_name for entity in confirmed
            ):
                confirmed.append(
                    ConfirmedEntity(
                        entity_type="PERSON",
                        canonical_name=person.canonical_name,
                        aliases=person.aliases,
                        organization=person.organization,
                        title=person.title,
                        region=person.region,
                        confirmed_by="AUTO",
                    )
                )

        if confirmation_items:
            return None, ConfirmationRequest(version=version, items=confirmation_items)

        if not confirmed:
            for person in extracted.people:
                canonical = person.name or person.organization
                if not canonical:
                    continue
                confirmed.append(
                    ConfirmedEntity(
                        entity_type="PERSON" if person.name else "ORGANIZATION",
                        canonical_name=canonical,
                        organization=person.organization,
                        title=person.title,
                        confirmed_by="AUTO",
                    )
                )

        return (
            ConfirmedContext(
                intents=understanding.intents,
                entities=confirmed,
                event_type=understanding.event_type,
                event_time=understanding.event_time,
                event_location=understanding.event_location,
                business_directions=understanding.business_directions,
                focus_questions=understanding.focus_questions,
            ),
            None,
        )

    def apply_confirmation(
        self,
        request: ConfirmationRequest,
        selections,
        understanding: IntentUnderstanding,
    ) -> ConfirmedContext:
        by_mention = {selection.mention: selection for selection in selections}
        entities: list[ConfirmedEntity] = []
        for item in request.items:
            selection = by_mention.get(item.mention)
            if selection is None:
                raise ValueError(f"缺少确认项: {item.mention}")
            if selection.candidate_id:
                candidate = next(
                    (item for item in item.candidates if item.candidate_id == selection.candidate_id),
                    None,
                )
                if candidate is None:
                    raise ValueError(f"候选项不属于当前确认请求: {selection.candidate_id}")
                entities.append(
                    ConfirmedEntity(
                        candidate_id=candidate.candidate_id,
                        entity_type=candidate.entity_type,
                        canonical_name=candidate.canonical_name,
                        aliases=candidate.aliases,
                        organization=candidate.organization,
                        title=candidate.title,
                        region=candidate.region,
                        confirmed_by="USER",
                    )
                )
            elif selection.manual_value:
                entities.append(
                    ConfirmedEntity(
                        entity_type=item.entity_type,
                        canonical_name=selection.manual_value.strip(),
                        confirmed_by="USER",
                    )
                )
            else:
                raise ValueError(f"确认项必须选择候选人或填写名称: {item.mention}")
        return ConfirmedContext(
            intents=understanding.intents,
            entities=entities,
            event_type=understanding.event_type,
            event_time=understanding.event_time,
            event_location=understanding.event_location,
            business_directions=understanding.business_directions,
            focus_questions=understanding.focus_questions,
        )

    def _candidates_for_mention(self, mention: str, input_text: str) -> list[CandidateOption]:
        output = []
        for row in self.rows:
            if not any(
                token and (token in mention or token in input_text)
                for token in (row.get("person_name"), row.get("person_alias"))
            ):
                continue
            output.append(self._candidate(row, mention))
        return output

    @staticmethod
    def _candidate(row: dict[str, str], mention: str) -> CandidateOption:
        return CandidateOption(
            candidate_id=row["candidate_id"],
            entity_type="PERSON",
            canonical_name=row["person_name"],
            aliases=[row["person_alias"]] if row.get("person_alias") else [],
            organization=row.get("organization_name") or None,
            title=row.get("title") or None,
            region=row.get("region") or None,
            reason=f"输入称呼“{mention}”与人物姓名或别名匹配",
            confidence=0.95 if mention == row.get("person_name") else 0.80,
        )

    @staticmethod
    def _confirmed(row: dict[str, str], source: str) -> ConfirmedEntity:
        return ConfirmedEntity(
            candidate_id=row["candidate_id"],
            entity_type="PERSON",
            canonical_name=row["person_name"],
            aliases=[row["person_alias"]] if row.get("person_alias") else [],
            organization=row.get("organization_name") or None,
            title=row.get("title") or None,
            region=row.get("region") or None,
            confirmed_by=source,
        )
