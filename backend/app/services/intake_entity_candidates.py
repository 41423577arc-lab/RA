import asyncio
import hashlib
import re
from collections.abc import Callable

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
    def __init__(self, projects: ProjectMcpClient, web: TavilyClient):
        self.projects = projects
        self.web = web

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
        pages = self._external_pages(person, organization)
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
                    normalized, pages
                )
            except Exception:
                normalized_external = []
        items: list[ConfirmationItem] = []
        for item in confirmation.items:
            options = list(item.candidates)
            if len(options) != 1:
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
                option for option in item.candidates if option.confidence >= threshold
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

    def _external_pages(self, person: str | None, organization: str | None):
        query = " ".join(f'"{value}"' for value in (person, organization) if value)
        try:
            results = asyncio.run(
                self.web.search([f"{query} 完整姓名 企业全称 职位"])
            )
            return asyncio.run(self.web.extract(results)) if results else []
        except Exception:
            return []

    @staticmethod
    def _rule_external_options(
        pages, mention: str, entity_type: str, organization: str | None
    ) -> list[CandidateOption]:
        output: list[CandidateOption] = []
        if entity_type == "PERSON" and organization:
            output.extend(
                EntityResolver().candidates_from_web(mention, organization, pages)
            )
        return output

    @staticmethod
    def _validated_normalized_candidates(
        result: ExternalIdentityNormalizationResult, pages
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
                        reason="联网公开资料补充的关键人标准身份候选",
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
