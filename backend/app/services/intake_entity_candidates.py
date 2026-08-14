import asyncio
import hashlib
import re
from collections.abc import Callable
from datetime import date

from app.schemas.intake import (
    ExternalIdentityNormalizationResult,
    IntakeEntityResolution,
    IntakeStructuredContext,
)
from app.schemas.task import CandidateOption, ConfirmationItem, ConfirmationRequest
from app.services.mcp_client import ProjectMcpClient
from app.services.entity_resolver import EntityResolver
from app.services.tavily_client import TavilyClient


class IntakeEntityCandidateService:
    def __init__(
        self,
        projects: ProjectMcpClient,
        web: TavilyClient,
        today_provider: Callable[[], date] | None = None,
    ):
        self.projects = projects
        self.web = web
        self.today_provider = today_provider or date.today

    def resolve(
        self,
        context: IntakeStructuredContext,
        version: int,
        source_text: str | None = None,
        external_normalizer: Callable[
            [list[dict], list[dict]], ExternalIdentityNormalizationResult
        ]
        | None = None,
    ) -> tuple[list[dict], ConfirmationRequest | None]:
        resolutions, confirmation = self.lookup_internal(context, version, source_text)
        if confirmation and external_normalizer and any(
            len(item.candidates) != 1 for item in confirmation.items
        ):
            confirmation = self.search_key_person_identity_web(
                context, confirmation, external_normalizer
            )
        return self.apply_automatic_candidates(resolutions, confirmation)

    def lookup_internal(
        self,
        context: IntakeStructuredContext,
        version: int,
        source_text: str | None = None,
    ) -> tuple[list[dict], ConfirmationRequest | None]:
        if not context.people and not context.organizations:
            return [], None

        resolutions: list[dict] = []
        unresolved: dict[str, list[str]] = {"PERSON": [], "ORGANIZATION": []}
        for entity_type, mentions in (
            ("PERSON", context.people),
            ("ORGANIZATION", context.organizations),
        ):
            for mention in mentions:
                if source_text and _is_standard_user_entity(
                    context, mention, entity_type, source_text
                ):
                    detail = next(
                        (item for item in context.people_details if item.name == mention),
                        None,
                    )
                    resolutions.append(
                        {
                            "candidate_id": None,
                            "entity_type": entity_type,
                            "canonical_name": mention,
                            "mention": mention,
                            "organization": detail.organization if detail else None,
                            "title": detail.title if detail else None,
                            "confirmed_by": "USER_INPUT",
                        }
                    )
                else:
                    unresolved[entity_type].append(mention)

        if not unresolved["PERSON"] and not unresolved["ORGANIZATION"]:
            return resolutions, None

        person = unresolved["PERSON"][0] if unresolved["PERSON"] else None
        organization = (
            unresolved["ORGANIZATION"][0]
            if unresolved["ORGANIZATION"]
            else (context.organizations[0] if context.organizations else None)
        )

        try:
            internal = asyncio.run(
                self.projects.find_entity_candidates(person, organization)
            )
        except Exception:
            internal = []

        pending: list[tuple[str, str, list[CandidateOption]]] = []
        for entity_type, mentions in (
            ("PERSON", unresolved["PERSON"]),
            ("ORGANIZATION", unresolved["ORGANIZATION"]),
        ):
            for index, mention in enumerate(mentions):
                candidates = [
                    item
                    for item in internal
                    if item.get("entity_type") == entity_type
                    and (index == 0 or item.get("canonical_name") == mention)
                ]
                pending.append(
                    (
                        entity_type,
                        mention,
                        [self._internal_option(item) for item in candidates],
                    )
                )

        items = [
            ConfirmationItem(
                mention=mention,
                entity_type=entity_type,
                candidates=self._unique(options)[:5],
            )
            for entity_type, mention, options in pending
        ]
        return resolutions, ConfirmationRequest(version=version, items=items)

    def search_key_person_identity_web(
        self,
        context: IntakeStructuredContext,
        confirmation: ConfirmationRequest,
        external_normalizer: Callable[
            [list[dict], list[dict]], ExternalIdentityNormalizationResult
        ],
    ) -> ConfirmationRequest:
        external_pending = [
            item for item in confirmation.items if len(item.candidates) != 1
        ]
        if not external_pending:
            return confirmation
        person = context.people[0] if context.people else None
        organization = context.organizations[0] if context.organizations else None
        as_of_date = self.today_provider()
        pages = self._external_pages(person, organization, as_of_date)
        normalized_external: list[tuple[str, CandidateOption]] = []
        if pages:
            try:
                normalized = external_normalizer(
                    [
                        {
                            "entity_type": item.entity_type,
                            "mention": item.mention,
                            "known_organization": organization,
                        }
                        for item in external_pending
                    ],
                    [page.model_dump(mode="json") for page in pages],
                )
                normalized_external = self._validated_normalized_candidates(
                    normalized, pages, as_of_date
                )
            except Exception:
                normalized_external = []
        items: list[ConfirmationItem] = []
        for item in confirmation.items:
            options = list(item.candidates)
            if len(options) != 1:
                if item.entity_type == "PERSON" and organization:
                    options.extend(
                        _explicit_person_options(
                            pages, item.mention, organization, as_of_date
                        )
                    )
                options.extend(
                    option
                    for normalized_mention, option in normalized_external
                    if normalized_mention == item.mention
                    and option.entity_type == item.entity_type
                )
                options.extend(
                    self._rule_external_options(
                        pages,
                        item.mention,
                        item.entity_type,
                        organization if item.entity_type == "PERSON" else None,
                    )
                )
            items.append(
                ConfirmationItem(
                    mention=item.mention,
                    entity_type=item.entity_type,
                    candidates=self._unique(options)[:5],
                )
            )
        return ConfirmationRequest(version=confirmation.version, items=items)

    @staticmethod
    def apply_automatic_candidates(
        resolutions: list[dict],
        confirmation: ConfirmationRequest | None,
        threshold: float = 0.80,
    ) -> tuple[list[dict], ConfirmationRequest | None]:
        if confirmation is None:
            return resolutions, None
        pending: list[ConfirmationItem] = []
        for item in confirmation.items:
            eligible = [
                option
                for option in item.candidates
                if option.confidence >= threshold
                and (
                    not option.candidate_id.startswith("internal:")
                    or option.canonical_name == item.mention
                )
            ]
            external_eligible = [
                option
                for option in eligible
                if option.source_url
            ]
            if len(external_eligible) == 1:
                eligible = external_eligible
            canonical_names = {option.canonical_name for option in eligible}
            if len(eligible) == 1 and len(canonical_names) == 1:
                option = eligible[0]
                resolutions.append(
                    IntakeEntityResolution(
                        **option.model_dump(mode="json"),
                        mention=item.mention,
                        confirmed_by=(
                            "EXTERNAL_AUTO"
                            if option.source_url
                            else "INTERNAL"
                        ),
                    ).model_dump(mode="json")
                )
            else:
                pending.append(item)
        if pending:
            return resolutions, ConfirmationRequest(
                version=confirmation.version, items=pending
            )
        return resolutions, None

    def _external_pages(
        self,
        person: str | None,
        organization: str | None,
        as_of_date: date,
    ):
        try:
            search_identity = getattr(self.web, "search_identity", None)
            extract_identity = getattr(self.web, "extract_identity", None)
            if callable(search_identity):
                queries = _identity_queries(person, organization)
                start_date = date(as_of_date.year - 1, 1, 1).isoformat()
                results = asyncio.run(
                    search_identity(
                        queries,
                        start_date=start_date,
                        end_date=as_of_date.isoformat(),
                    )
                )
            else:
                query = " ".join(
                    f'"{value}"' for value in (person, organization) if value
                )
                results = asyncio.run(
                    self.web.search([f"{query} 完整姓名 企业全称 职位"])
                )
            if not results:
                return []
            if callable(extract_identity):
                return asyncio.run(extract_identity(results))
            return asyncio.run(self.web.extract(results))
        except Exception:
            return []

    @staticmethod
    def _rule_external_options(
        pages,
        mention: str,
        entity_type: str,
        organization: str | None,
    ) -> list[CandidateOption]:
        output: list[CandidateOption] = []
        if entity_type == "PERSON" and organization:
            output.extend(
                EntityResolver().candidates_from_web(mention, organization, pages)
            )
        elif entity_type == "ORGANIZATION":
            suffix = (
                r"股份有限公司|有限责任公司|集团有限公司|有限公司|集团公司|"
                r"工程局|研究院|委员会|人民政府|大学|银行"
            )
            alias = re.escape(mention)
            pattern = re.compile(
                rf"(?P<name>[\u4e00-\u9fff]{{2,40}}(?:{suffix}))"
                rf"[^。！？!?；;\n]{{0,20}}?(?:以下简称|简称)[“\"']?{alias}"
            )
            for page in pages[:10]:
                for sentence in re.split(r"(?<=[。！？!?；;\n])", page.raw_content):
                    match = pattern.search(sentence)
                    if match is None:
                        continue
                    canonical_name = match.group("name")
                    candidate_id = hashlib.sha256(
                        f"ORGANIZATION|{mention}|{canonical_name}|{page.url}".encode(
                            "utf-8"
                        )
                    ).hexdigest()[:24]
                    output.append(
                        CandidateOption(
                            candidate_id=f"external:{candidate_id}",
                            entity_type="ORGANIZATION",
                            canonical_name=canonical_name,
                            reason="网页原文明确标注该企业全称及简称",
                            confidence=0.92,
                            source_url=page.url,
                            evidence_quote=sentence.strip()[:500],
                        )
                    )
        return output

    @staticmethod
    def _validated_normalized_candidates(
        result: ExternalIdentityNormalizationResult,
        pages,
        as_of_date: date,
    ) -> list[tuple[str, CandidateOption]]:
        by_url = {page.url: page for page in pages}
        output: list[tuple[str, CandidateOption]] = []
        for item in result.candidates:
            page = by_url.get(item.source_url)
            if (
                page is None
                or item.evidence_quote not in page.raw_content
                or item.canonical_name not in page.raw_content
            ):
                continue
            candidate_id = hashlib.sha256(
                (
                    f"{item.entity_type}|{item.mention}|{item.canonical_name}|"
                    f"{item.source_url}"
                ).encode("utf-8")
            ).hexdigest()[:24]
            output.append(
                (
                    item.mention,
                    CandidateOption(
                        candidate_id=f"external:{candidate_id}",
                        entity_type=item.entity_type,
                        canonical_name=item.canonical_name,
                        organization=item.organization,
                        title=item.title,
                        reason=_temporal_reason(page, as_of_date),
                        confidence=item.confidence,
                        source_url=item.source_url,
                        evidence_quote=item.evidence_quote,
                    ),
                )
            )
        return output

    @staticmethod
    def _internal_option(item: dict) -> CandidateOption:
        return CandidateOption(
            candidate_id=item["candidate_id"],
            entity_type=item["entity_type"],
            canonical_name=item["canonical_name"],
            organization=item.get("organization"),
            title=item.get("title"),
            region=item.get("region"),
            reason="内部客户或联系人候选",
            confidence=1.0 if item.get("match_type") == "EXACT" else 0.8,
        )

    @staticmethod
    def _unique(items: list[CandidateOption]) -> list[CandidateOption]:
        output: list[CandidateOption] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            key = (item.entity_type, item.canonical_name)
            if key not in seen:
                seen.add(key)
                output.append(item)
        return output


_PERSON_TITLE_PATTERN = (
    r"党委书记、董事长|党委副书记、总经理|党委书记|党委副书记|"
    r"董事长兼总经理|董事长|总经理|常务副总经理|副总经理|"
    r"总工程师|法定代表人"
)


def _identity_queries(
    person: str | None, organization: str | None
) -> list[str]:
    quoted_person = f'"{person}"' if person else ""
    aliases = _organization_search_aliases(organization) if organization else []
    queries: list[str] = []
    if person and organization:
        queries.append(f'{quoted_person} "{organization}"')
        role_organization = aliases[1] if len(aliases) > 1 else organization
        queries.append(f'{quoted_person} "{role_organization}" 职务 现任')
    elif person:
        queries.append(f"{quoted_person} 现任 职务")
    elif organization:
        queries.append(f'"{organization}" 企业全称 简称')
    return list(dict.fromkeys(queries))


def _organization_search_aliases(organization: str) -> list[str]:
    aliases = [organization]
    replacements = (
        ("工程有限公司", "公司"),
        ("有限责任公司", "公司"),
        ("股份有限公司", "公司"),
    )
    for suffix, replacement in replacements:
        if organization.endswith(suffix):
            aliases.append(organization[: -len(suffix)] + replacement)
            break
    if "中国建筑第二工程局" in organization:
        aliases.append(organization.replace("中国建筑第二工程局", "中建二局"))
    return list(dict.fromkeys(alias for alias in aliases if len(alias) >= 4))


def _explicit_person_options(
    pages,
    mention: str,
    organization: str,
    as_of_date: date,
) -> list[CandidateOption]:
    aliases = _organization_search_aliases(organization)
    alias_pattern = "|".join(re.escape(alias) for alias in aliases)
    person = re.escape(mention)
    patterns = (
        re.compile(
            rf"姓名\s*[:：]\s*{person}\s*.{{0,20}}?职位\s*[:：]\s*"
            rf"(?:(?:{alias_pattern}))?(?P<title>{_PERSON_TITLE_PATTERN})"
        ),
        re.compile(
            rf"(?P<title>(?:公司)?(?:{_PERSON_TITLE_PATTERN}))\s*{person}"
        ),
        re.compile(
            rf"{person}.{{0,20}}?(?:现任|任|担任|职位\s*[:：])\s*"
            rf"(?P<title>{_PERSON_TITLE_PATTERN})"
        ),
    )
    output: list[CandidateOption] = []
    for page in pages[:20]:
        text = page.raw_content
        if mention not in text or not any(alias in text for alias in aliases):
            continue
        for pattern in patterns:
            for match in pattern.finditer(text):
                start = max(0, match.start() - 120)
                end = min(len(text), match.end() + 120)
                evidence = text[start:end].strip()
                if not any(alias in evidence for alias in aliases):
                    continue
                title = match.group("title").removeprefix("公司")
                candidate_id = hashlib.sha256(
                    f"PERSON|{mention}|{organization}|{title}|{page.url}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:24]
                output.append(
                    CandidateOption(
                        candidate_id=f"external:{candidate_id}",
                        entity_type="PERSON",
                        canonical_name=mention,
                        organization=organization,
                        title=title,
                        reason=_temporal_reason(page, as_of_date),
                        confidence=0.93,
                        source_url=page.url,
                        evidence_quote=evidence[:500],
                    )
                )
    return output


def _temporal_reason(page, as_of_date: date) -> str:
    evidence_time = _page_evidence_time(page)
    if evidence_time:
        return (
            f"公开资料时间为 {evidence_time}，"
            f"身份检索截止 {as_of_date.isoformat()}"
        )
    return f"无发布日期的公开页面，身份检索截止 {as_of_date.isoformat()}"


def _page_evidence_time(page) -> str | None:
    if page.published_at is not None:
        return page.published_at.date().isoformat()
    match = re.search(r"/(20\d{2})(0[1-9]|1[0-2])(?:/|\d{2}/)", page.url)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    match = re.search(
        r"(?:发布日期|发布时间)\s*[:：]?\s*(20\d{2})[-年/]"
        r"(0?[1-9]|1[0-2])(?:[-月/](0?[1-9]|[12]\d|3[01]))?",
        page.raw_content[:1000],
    )
    if match:
        suffix = f"-{int(match.group(3)):02d}" if match.group(3) else ""
        return f"{match.group(1)}-{int(match.group(2)):02d}{suffix}"
    return None


def verify_identity_evidence(
    page_text: str, mention: str, organization: str | None = None
) -> str | None:
    normalized = "".join(page_text.split())
    if "".join(mention.split()) not in normalized:
        return None
    if organization and "".join(organization.split()) not in normalized:
        return None
    position = page_text.find(mention)
    if position < 0:
        return page_text[:300]
    return page_text[max(0, position - 100) : position + len(mention) + 200]


def user_provided_entity_resolutions(
    context: IntakeStructuredContext, source_text: str
) -> list[dict]:
    organization = context.organizations[0] if context.organizations else None
    output: list[dict] = []
    for entity_type, mentions in (
        ("PERSON", context.people),
        ("ORGANIZATION", context.organizations),
    ):
        for mention in mentions:
            if _is_standard_user_entity(
                context, mention, entity_type, source_text
            ):
                detail = next(
                    (item for item in context.people_details if item.name == mention),
                    None,
                )
                output.append(
                    {
                        "candidate_id": None,
                        "entity_type": entity_type,
                        "canonical_name": mention,
                        "mention": mention,
                        "organization": organization
                        if entity_type == "PERSON"
                        else None,
                        "title": detail.title if detail else None,
                        "confidence": 1.0,
                        "confirmed_by": "USER_INPUT",
                    }
                )
    return output


def _is_standard_user_entity(
    context: IntakeStructuredContext,
    mention: str,
    entity_type: str,
    source_text: str,
) -> bool:
    normalized_mention = "".join(mention.split())
    normalized_source = "".join(source_text.split())
    if not normalized_mention or normalized_mention not in normalized_source:
        return False
    assessment = next(
        (
            item
            for item in context.entity_assessments
            if item.entity_type == entity_type and item.mention == mention
        ),
        None,
    )
    if assessment is not None and not assessment.is_standard:
        return False
    if entity_type == "ORGANIZATION":
        standard_suffixes = (
            "有限公司",
            "股份有限公司",
            "集团有限公司",
            "集团",
            "大学",
            "银行",
            "委员会",
            "人民政府",
        )
        return normalized_mention.endswith(standard_suffixes)
    title_suffixes = ("总", "经理", "主任", "董事长", "负责人", "领导")
    if normalized_mention.endswith(title_suffixes):
        return False
    return bool(
        re.fullmatch(r"[\u4e00-\u9fff]{2,4}", normalized_mention)
        or len(mention.split()) >= 2
    )
