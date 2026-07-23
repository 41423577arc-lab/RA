import asyncio
import hashlib
import re

from app.schemas.intake import IntakeStructuredContext
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
    ) -> tuple[list[dict], ConfirmationRequest | None]:
        person = context.people[0] if context.people else None
        organization = context.organizations[0] if context.organizations else None
        if not person and not organization:
            return [], None

        try:
            internal = asyncio.run(
                self.projects.find_entity_candidates(person, organization)
            )
        except Exception:
            internal = []

        resolutions: list[dict] = []
        pending: list[tuple[str, str, list[CandidateOption]]] = []
        for entity_type, mentions in (
            ("PERSON", context.people),
            ("ORGANIZATION", context.organizations),
        ):
            for index, mention in enumerate(mentions):
                candidates = [
                    item
                    for item in internal
                    if item.get("entity_type") == entity_type
                    and (index == 0 or item.get("canonical_name") == mention)
                ]
                exact = [
                    item
                    for item in candidates
                    if item.get("match_type") == "EXACT"
                    and item.get("canonical_name") == mention
                ]
                if len(exact) == 1:
                    resolutions.append({**exact[0], "confirmed_by": "INTERNAL"})
                    continue

                if source_text and _is_explicit_user_entity(
                    mention, entity_type, source_text
                ):
                    resolutions.append(
                        {
                            "candidate_id": None,
                            "entity_type": entity_type,
                            "canonical_name": mention,
                            "mention": mention,
                            "organization": organization
                            if entity_type == "PERSON"
                            else None,
                            "confirmed_by": "USER_INPUT",
                        }
                    )
                    continue

                pending.append(
                    (
                        entity_type,
                        mention,
                        [self._internal_option(item) for item in candidates],
                    )
                )

        pages = self._external_pages(person, organization) if pending else []
        items: list[ConfirmationItem] = []
        for entity_type, mention, options in pending:
            options.extend(
                self._external_options(
                    pages,
                    mention,
                    entity_type,
                    organization if entity_type == "PERSON" else None,
                )
            )
            items.append(
                ConfirmationItem(
                    mention=mention,
                    entity_type=entity_type,
                    candidates=self._unique(options)[:5],
                )
            )

        if items:
            return resolutions, ConfirmationRequest(version=version, items=items)
        return resolutions, None

    def _external_pages(self, person: str | None, organization: str | None):
        query = " ".join(f'"{value}"' for value in (person, organization) if value)
        try:
            results = asyncio.run(self.web.search([f"{query} 身份 职务"]))
            return asyncio.run(self.web.extract(results)) if results else []
        except Exception:
            return []

    @staticmethod
    def _external_options(
        pages, mention: str, entity_type: str, organization: str | None
    ) -> list[CandidateOption]:
        output: list[CandidateOption] = []
        if entity_type == "PERSON" and organization:
            output.extend(
                EntityResolver().candidates_from_web(mention, organization, pages)
            )
        for page in pages[:5]:
            evidence = verify_identity_evidence(
                page.raw_content, mention, organization
            )
            if evidence is None:
                continue
            candidate_id = hashlib.sha256(
                f"{entity_type}|{mention}|{page.url}".encode("utf-8")
            ).hexdigest()[:24]
            output.append(
                CandidateOption(
                    candidate_id=f"external:{candidate_id}",
                    entity_type=entity_type,
                    canonical_name=mention,
                    organization=organization,
                    reason="公开网页中出现该身份信息，必须由用户确认",
                    confidence=0.6,
                    source_url=page.url,
                    evidence_quote=evidence,
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
            if _is_explicit_user_entity(mention, entity_type, source_text):
                output.append(
                    {
                        "candidate_id": None,
                        "entity_type": entity_type,
                        "canonical_name": mention,
                        "mention": mention,
                        "organization": organization
                        if entity_type == "PERSON"
                        else None,
                        "confirmed_by": "USER_INPUT",
                    }
                )
    return output


def _is_explicit_user_entity(
    mention: str, entity_type: str, source_text: str
) -> bool:
    normalized_mention = "".join(mention.split())
    normalized_source = "".join(source_text.split())
    if not normalized_mention or normalized_mention not in normalized_source:
        return False
    if entity_type == "ORGANIZATION":
        return len(normalized_mention) >= 2
    title_suffixes = ("总", "经理", "主任", "董事长", "负责人", "领导")
    if normalized_mention.endswith(title_suffixes):
        return False
    return bool(
        re.fullmatch(r"[\u4e00-\u9fff]{2,4}", normalized_mention)
        or len(mention.split()) >= 2
    )
