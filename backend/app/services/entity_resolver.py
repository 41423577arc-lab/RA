import hashlib
import re
import unicodedata

from app.schemas.task import (
    CandidateOption,
    ConfirmationItem,
    ConfirmationRequest,
    ConfirmedContext,
    ConfirmedEntity,
    IntentUnderstanding,
    WebPage,
)


class InsufficientContextError(ValueError):
    pass


class EntityResolver:
    """Use model decisions after verifying that their evidence exists in the input."""

    def resolve(
        self,
        input_text: str,
        understanding: IntentUnderstanding,
        version: int,
        external_candidates: list[CandidateOption] | None = None,
    ) -> tuple[ConfirmedContext | None, ConfirmationRequest | None]:
        confirmed, uncertain_people, uncertain_organizations = self._supported_entities(
            input_text, understanding
        )
        if external_candidates:
            uncertain_people.extend(
                item for item in external_candidates if item.entity_type == "PERSON"
            )
            uncertain_organizations.extend(
                item for item in external_candidates if item.entity_type == "ORGANIZATION"
            )
        has_person = any(item.entity_type == "PERSON" for item in confirmed)
        has_organization = any(item.entity_type == "ORGANIZATION" for item in confirmed)

        if not has_person and not has_organization and not uncertain_people and not uncertain_organizations:
            raise InsufficientContextError(
                "未识别到明确的人物姓名和企业名称，请补充后重新提交"
            )

        items: list[ConfirmationItem] = []
        if not has_person:
            items.append(
                ConfirmationItem(
                    mention="人物姓名",
                    entity_type="PERSON",
                    candidates=uncertain_people,
                )
            )
        if not has_organization:
            items.append(
                ConfirmationItem(
                    mention="企业名称",
                    entity_type="ORGANIZATION",
                    candidates=uncertain_organizations,
                )
            )
        if items:
            return None, ConfirmationRequest(version=version, items=items)

        return self._context(understanding, confirmed), None

    def candidate_lookup(
        self,
        input_text: str,
        understanding: IntentUnderstanding,
    ) -> tuple[str, str] | None:
        confirmed, _, _ = self._supported_entities(input_text, understanding)
        if any(item.entity_type == "PERSON" for item in confirmed):
            return None
        organizations = [
            item.canonical_name for item in confirmed if item.entity_type == "ORGANIZATION"
        ]
        if len(organizations) != 1:
            return None
        for person in understanding.people:
            mention = person.mention.strip()
            match = re.fullmatch(r"([\u4e00-\u9fff]{1,2})(?:总|董|经理|主任|书记|院长|校长)", mention)
            if match and self._normalize(mention) in self._normalize(input_text):
                return mention, organizations[0]
        return None

    def candidates_from_web(
        self,
        mention: str,
        organization: str,
        pages: list[WebPage],
    ) -> list[CandidateOption]:
        surname_match = re.fullmatch(
            r"([\u4e00-\u9fff]{1,2})(?:总|董|经理|主任|书记|院长|校长)", mention
        )
        if not surname_match:
            return []
        surname = surname_match.group(1)
        title_pattern = "董事长|副董事长|总经理|副总经理|总裁|副总裁|负责人|法定代表人"
        patterns = [
            re.compile(
                rf"(?P<name>{re.escape(surname)}[\u4e00-\u9fff]{{1,2}})"
                rf"[^。！？!?；;\n]{{0,14}}?(?P<title>{title_pattern})"
            ),
            re.compile(
                rf"(?P<title>{title_pattern})[^。！？!?；;\n]{{0,14}}?"
                rf"(?P<name>{re.escape(surname)}[\u4e00-\u9fff]{{1,2}})"
                rf"(?=负责|现任|任职|[，,。；;、\s（(]|$)"
            ),
        ]
        invalid_name_terms = ("总", "董", "经理", "公司", "集团", "股份", "有限", "先生", "女士")
        output: list[CandidateOption] = []
        for page in pages[:10]:
            if self._normalize(organization) not in self._normalize(page.raw_content):
                continue
            sentences = [
                item.strip()
                for item in re.split(r"(?<=[。！？!?；;\n])", page.raw_content)
                if organization in item
            ]
            for sentence in sentences:
                for pattern in patterns:
                    for match in pattern.finditer(sentence):
                        name = match.group("name")
                        if name == mention or any(term in name for term in invalid_name_terms):
                            continue
                        output.append(
                            self._input_candidate(
                                "PERSON",
                                name,
                                mention,
                                organization,
                                match.group("title"),
                                "联网公开资料在同一段文字中同时出现该人物、企业和职务",
                                0.82,
                                source_url=page.url,
                                evidence_quote=sentence[:500],
                            )
                        )
        return self._unique_candidates(output)[:5]

    def apply_confirmation(
        self,
        request: ConfirmationRequest,
        selections,
        understanding: IntentUnderstanding,
        input_text: str,
    ) -> ConfirmedContext:
        confirmed, _, _ = self._supported_entities(input_text, understanding)
        by_mention = {selection.mention: selection for selection in selections}
        for item in request.items:
            selection = by_mention.get(item.mention)
            if selection is None:
                raise ValueError(f"缺少确认项: {item.mention}")
            candidate = None
            if selection.candidate_id:
                candidate = next(
                    (
                        option
                        for option in item.candidates
                        if option.candidate_id == selection.candidate_id
                    ),
                    None,
                )
                if candidate is None:
                    raise ValueError(f"候选项不属于当前确认请求: {selection.candidate_id}")
                value = candidate.canonical_name
            else:
                value = self._validate_manual_value(selection.manual_value, item.entity_type)
            confirmed.append(
                ConfirmedEntity(
                    candidate_id=candidate.candidate_id if candidate else None,
                    entity_type=item.entity_type,
                    canonical_name=value,
                    aliases=candidate.aliases if candidate else [],
                    organization=candidate.organization if candidate else None,
                    title=candidate.title if candidate else None,
                    region=candidate.region if candidate else None,
                    confirmed_by="USER",
                )
            )

        confirmed = self._deduplicate(confirmed)
        people = [item for item in confirmed if item.entity_type == "PERSON"]
        organizations = [item for item in confirmed if item.entity_type == "ORGANIZATION"]
        if not people or not organizations:
            raise ValueError("人物姓名和企业名称均为必填项")
        if len(people) == 1 and len(organizations) == 1 and not people[0].organization:
            organization_name = organizations[0].canonical_name
            confirmed = [
                item.model_copy(update={"organization": organization_name})
                if item.entity_type == "PERSON"
                else item
                for item in confirmed
            ]
        return self._context(understanding, confirmed)

    def _supported_entities(
        self,
        input_text: str,
        understanding: IntentUnderstanding,
    ) -> tuple[list[ConfirmedEntity], list[CandidateOption], list[CandidateOption]]:
        source = self._normalize(input_text)
        confirmed: list[ConfirmedEntity] = []
        uncertain_people: list[CandidateOption] = []
        uncertain_organizations: list[CandidateOption] = []

        for person in understanding.people:
            canonical = (person.canonical_name or "").strip()
            mention = person.mention.strip()
            full_name_in_input = bool(canonical and self._normalize(canonical) in source)
            supported = self._has_source_evidence(input_text, mention, person.evidence_text)
            source_organization = self._source_value(person.organization, source)
            source_title = self._source_value(person.title, source)
            if (
                person.resolution == "CONFIRMED"
                and full_name_in_input
                and supported
            ):
                confirmed.append(
                    ConfirmedEntity(
                        entity_type="PERSON",
                        canonical_name=canonical,
                        aliases=[mention] if mention != canonical else [],
                        organization=source_organization,
                        title=source_title,
                        region=person.region,
                        confirmed_by="AUTO",
                    )
                )
            elif (
                person.resolution == "NEEDS_CONFIRMATION"
                and supported
                and canonical
                and full_name_in_input
            ):
                uncertain_people.append(
                    self._input_candidate(
                        "PERSON", canonical, mention, source_organization, source_title,
                        "大模型识别到人物，但明确要求用户确认",
                        person.confidence,
                    )
                )

        for organization in understanding.organizations:
            canonical = (organization.canonical_name or "").strip()
            mention = organization.mention.strip()
            supported = self._has_source_evidence(
                input_text, mention, organization.evidence_text
            )
            source_name = canonical if canonical and self._normalize(canonical) in source else mention
            if (
                organization.resolution == "CONFIRMED"
                and source_name
                and supported
            ):
                confirmed.append(
                    ConfirmedEntity(
                        entity_type="ORGANIZATION",
                        canonical_name=source_name,
                        aliases=[mention] if mention != source_name else [],
                        region=organization.region,
                        confirmed_by="AUTO",
                    )
                )
            elif (
                organization.resolution == "NEEDS_CONFIRMATION"
                and supported
                and (canonical or mention)
            ):
                uncertain_organizations.append(
                    self._input_candidate(
                        "ORGANIZATION", source_name, mention, None, None,
                        "大模型识别到企业，但明确要求用户确认",
                        organization.confidence,
                    )
                )

        return (
            self._deduplicate(confirmed),
            self._unique_candidates(uncertain_people),
            self._unique_candidates(uncertain_organizations),
        )

    @classmethod
    def _has_source_evidence(cls, input_text: str, mention: str, evidence_text: str) -> bool:
        source = cls._normalize(input_text)
        mention_text = cls._normalize(mention)
        evidence = cls._normalize(evidence_text)
        return bool(mention_text and evidence and mention_text in source and evidence in source)

    @classmethod
    def _source_value(cls, value: str | None, normalized_source: str) -> str | None:
        candidate = (value or "").strip()
        return candidate if candidate and cls._normalize(candidate) in normalized_source else None

    @staticmethod
    def _input_candidate(
        entity_type: str,
        canonical_name: str,
        mention: str,
        organization: str | None,
        title: str | None,
        reason: str,
        confidence: float,
        source_url: str | None = None,
        evidence_quote: str | None = None,
    ) -> CandidateOption:
        digest = hashlib.sha256(f"{entity_type}:{canonical_name}".encode("utf-8")).hexdigest()[:12]
        return CandidateOption(
            candidate_id=f"INPUT-{entity_type}-{digest}",
            entity_type=entity_type,
            canonical_name=canonical_name,
            aliases=[mention] if mention != canonical_name else [],
            organization=organization,
            title=title,
            reason=reason,
            confidence=confidence,
            source_url=source_url,
            evidence_quote=evidence_quote,
        )

    @staticmethod
    def _validate_manual_value(value: str | None, entity_type: str) -> str:
        normalized = re.sub(r"\s+", " ", (value or "").strip())
        label = "人物姓名" if entity_type == "PERSON" else "企业名称"
        if not 2 <= len(normalized) <= 100:
            raise ValueError(f"{label}长度必须为 2 到 100 个字符")
        if re.search(r"[\x00-\x1f<>]", normalized):
            raise ValueError(f"{label}包含无效字符")
        return normalized

    @staticmethod
    def _deduplicate(entities: list[ConfirmedEntity]) -> list[ConfirmedEntity]:
        output = {}
        for entity in entities:
            output[(entity.entity_type, entity.canonical_name)] = entity
        return list(output.values())

    @staticmethod
    def _unique_candidates(candidates: list[CandidateOption]) -> list[CandidateOption]:
        output = {}
        for candidate in candidates:
            output[candidate.canonical_name] = candidate
        return list(output.values())

    @staticmethod
    def _context(
        understanding: IntentUnderstanding, entities: list[ConfirmedEntity]
    ) -> ConfirmedContext:
        return ConfirmedContext(
            intents=understanding.intents,
            entities=entities,
            event_type=understanding.event_type,
            event_time=understanding.event_time,
            event_location=understanding.event_location,
            business_directions=understanding.business_directions,
            focus_questions=understanding.focus_questions,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()
