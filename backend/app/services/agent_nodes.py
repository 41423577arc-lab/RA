import re
from app.schemas.task import (
    ActionBrief,
    AgentAction,
    AgentContext,
    AgentTurnDecision,
    AssociationAnalysis,
    ConfirmedContext,
    EntityMention,
    EvidenceBackedItem,
    ExtractedInfo,
    GeneratedReportContent,
    IntentUnderstanding,
    ProjectQueryPlan,
    ProjectRanking,
    ProjectResult,
    PublicClaim,
    SupportedWebEvidence,
    WebEvidenceCandidate,
    WebEvidenceDecision,
    WebEvidence,
    WebPage,
    WebSearchPlan,
    WebSearchQuery,
    WebVerification,
)
from app.services.llm_client import StructuredLLM


class AgentNodes:
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def agent_turn(self, task_id: str, context: AgentContext) -> AgentAction:
        decision = self.llm.parse(
            task_id,
            "agent_turn",
            {"context": context.model_dump(mode="json")},
            AgentTurnDecision,
        )
        return decision.action

    def evidence_verify(
        self, task_id: str, candidates: list[WebEvidenceCandidate]
    ) -> WebEvidenceDecision:
        return self.llm.parse(
            task_id,
            "evidence_verify",
            {
                "candidates": [
                    candidate.model_dump(mode="json", exclude={"web_result_id"})
                    for candidate in candidates
                ],
            },
            WebEvidenceDecision,
        )

    def final_synthesis(
        self,
        task_id: str,
        context: ConfirmedContext,
        evidence: list[PublicClaim],
        ranked_projects: list[tuple[ProjectResult, ProjectRanking]],
        association: AssociationAnalysis,
    ) -> GeneratedReportContent:
        from app.services.final_synthesis import build_final_synthesis_input

        return self.llm.parse(
            task_id,
            "final_synthesis",
            build_final_synthesis_input(
                context,
                evidence,
                ranked_projects,
                association,
            ),
            GeneratedReportContent,
        )


def fallback_understanding(extracted: ExtractedInfo) -> IntentUnderstanding:
    intents = ["REPORT_GENERATION"]
    if extracted.event_type != "其他":
        intents = ["MEETING_PREPARATION", "PERSON_BACKGROUND_RESEARCH", "INTERNAL_PROJECT_QUERY", "REPORT_GENERATION"]
    return IntentUnderstanding(
        intents=intents,
        people=[
            EntityMention(
                mention=person.name,
                canonical_name=person.name,
                organization=person.organization,
                title=person.title,
                evidence_text=person.name,
                confidence=0.95,
                resolution="CONFIRMED",
            )
            for person in extracted.people
            if person.name
        ],
        organizations=[
            EntityMention(
                mention=person.organization,
                canonical_name=person.organization,
                evidence_text=person.organization,
                confidence=0.95,
                resolution="CONFIRMED",
            )
            for person in extracted.people
            if person.organization
        ],
        projects=[],
        event_type=extracted.event_type,
        event_time=extracted.event_time,
        event_location=extracted.event_location,
        business_directions=extracted.keywords,
        focus_questions=[],
        overall_confidence=0.75,
    )


def fallback_web_plan(context: ConfirmedContext) -> WebSearchPlan:
    queries: list[WebSearchQuery] = []
    for entity in context.entities:
        if entity.entity_type == "PERSON":
            terms = [entity.canonical_name, entity.organization, entity.title]
            for focus in ("负责业务", "近期动态"):
                query = " ".join(item for item in [*terms, focus] if item)
                queries.append(
                    WebSearchQuery(
                        query=query,
                        purpose=f"核验人物身份并了解{focus}",
                        target_person=entity.canonical_name,
                        target_organization=entity.organization,
                        required_terms=[item for item in [entity.canonical_name, entity.organization] if item],
                    )
                )
        elif entity.entity_type == "ORGANIZATION":
            queries.append(
                WebSearchQuery(
                    query=f"{entity.canonical_name} 主营业务 近期项目",
                    purpose="了解单位业务范围和近期项目",
                    target_organization=entity.canonical_name,
                    required_terms=[entity.canonical_name],
                )
            )
    if not queries and context.business_directions:
        queries.append(
            WebSearchQuery(
                query=" ".join(context.business_directions[:3]),
                purpose="补充用户关注的业务信息",
                required_terms=context.business_directions[:3],
            )
        )
    return WebSearchPlan(queries=queries[:6] or [WebSearchQuery(query="资源调查", purpose="基础检索")])


def fallback_project_query(context: ConfirmedContext) -> ProjectQueryPlan:
    people = [item for item in context.entities if item.entity_type == "PERSON"]
    organizations = [item for item in context.entities if item.entity_type == "ORGANIZATION"]
    return ProjectQueryPlan(
        person_names=unique([name for item in people for name in [item.canonical_name, *item.aliases]]),
        organization_names=unique(
            [name for item in people for name in [item.organization] if name]
            + [name for item in organizations for name in [item.canonical_name, *item.aliases]]
        ),
        project_names=[item.canonical_name for item in context.entities if item.entity_type == "PROJECT"],
        business_terms=context.business_directions,
        statuses=["ACTIVE", "COMPLETED"],
        purpose="、".join(context.intents),
    )


WEB_VERIFY_MAX_SEGMENTS_PER_PAGE = 3
WEB_VERIFY_MAX_SEGMENT_CHARS = 1000
WEB_VERIFY_MAX_BATCH_CHARS = 20_000

IDENTITY_QUERY_MARKERS = (
    "身份",
    "职位",
    "任职",
    "履历",
    "简历",
    "背景",
    "董事长",
    "总裁",
    "负责人",
    "管理范围",
    "分管",
    "公开活动",
    "演讲",
    "采访",
    "发言",
)
ORGANIZATION_TOPIC_MARKERS = (
    "主营业务",
    "业务布局",
    "战略",
    "项目",
    "销量",
    "产能",
    "近期动态",
    "新能源",
    "储能",
    "组织架构",
    "子公司",
    "事业部",
    "市场",
)
POSITION_MARKERS = (
    "董事长",
    "总裁",
    "经理",
    "主任",
    "书记",
    "委员",
    "创始人",
    "负责人",
    "任职",
    "担任",
    "现任",
    "职位",
    "履历",
    "管理",
)


def build_web_verification_candidates(
    pages: list[WebPage],
    context: ConfirmedContext,
    queries: list[WebSearchQuery],
    *,
    max_segments_per_page: int = WEB_VERIFY_MAX_SEGMENTS_PER_PAGE,
    max_segment_chars: int = WEB_VERIFY_MAX_SEGMENT_CHARS,
    max_batch_chars: int = WEB_VERIFY_MAX_BATCH_CHARS,
) -> list[WebEvidenceCandidate]:
    """Build a bounded, source-backed input for the web verification model."""
    query_by_text = {item.query: item for item in queries}
    output: list[WebEvidenceCandidate] = []
    used_chars = 0

    for page in sorted(pages, key=lambda item: (item.rank, item.web_result_id)):
        query = query_by_text.get(page.query)
        kind = classify_web_evidence_kind(page, query)
        person = page.target_person or (query.target_person if query else None)
        target_organization = page.target_organization or (
            query.target_organization if query else None
        )
        organization_terms = _organization_terms(context, target_organization)
        source_text = _clean_candidate_text(page.raw_content or page.search_snippet)
        if len(source_text) < 10 or not organization_terms:
            continue

        required_terms = unique(
            [
                *(query.required_terms if query else []),
                *context.business_directions,
            ]
        )
        if kind == "IDENTITY":
            if not person:
                continue
            anchors = [person]
            relevant_terms = unique([*organization_terms, *POSITION_MARKERS])
        else:
            anchors = organization_terms
            excluded = {person, *organization_terms, None}
            topic_terms = [term for term in required_terms if term not in excluded]
            topic_terms.extend(
                marker
                for marker in ORGANIZATION_TOPIC_MARKERS
                if marker in f"{page.query} {query.purpose if query else ''}"
            )
            relevant_terms = unique(topic_terms)

        ranked: list[tuple[int, str, list[str]]] = []
        for window in _bounded_candidate_windows(
            source_text, anchors, max_segment_chars=max_segment_chars
        ):
            if kind == "IDENTITY":
                if person not in window or not any(
                    term in window for term in organization_terms
                ):
                    continue
            else:
                if not any(term in window for term in organization_terms):
                    continue
            matched_terms = unique(
                [term for term in relevant_terms if term and term in window]
            )[:8]
            score = 100 if kind == "IDENTITY" else 50
            score += 10 * len(matched_terms)
            score += sum(2 for term in required_terms if term and term in window)
            ranked.append((score, window, matched_terms))

        seen_text: set[str] = set()
        page_candidates: list[tuple[str, list[str]]] = []
        for _, window, matched_terms in sorted(
            ranked, key=lambda item: (-item[0], len(item[1]))
        ):
            key = window.casefold()
            if key in seen_text:
                continue
            seen_text.add(key)
            page_candidates.append((window, matched_terms))
            if len(page_candidates) >= max_segments_per_page:
                break

        for window, matched_terms in page_candidates:
            if used_chars + len(window) > max_batch_chars:
                return output
            candidate_id = f"{page.web_result_id[:56]}-C{len(output) + 1:02d}"
            output.append(
                WebEvidenceCandidate(
                    candidate_id=candidate_id,
                    web_result_id=page.web_result_id,
                    kind=kind,
                    text=window,
                    target_person=person if kind == "IDENTITY" else None,
                    target_organization=target_organization
                    or next(iter(organization_terms), None),
                    matched_terms=matched_terms,
                )
            )
            used_chars += len(window)
    return output


def classify_web_evidence_kind(
    page: WebPage, query: WebSearchQuery | None
) -> str:
    target_person = page.target_person or (query.target_person if query else None)
    descriptor = " ".join(
        [
            page.query,
            query.purpose if query else "",
            *(query.required_terms if query else []),
        ]
    )
    if target_person and any(marker in descriptor for marker in IDENTITY_QUERY_MARKERS):
        return "IDENTITY"
    if any(marker in descriptor for marker in ORGANIZATION_TOPIC_MARKERS):
        return "ORGANIZATION_TOPIC"
    return "IDENTITY" if target_person else "ORGANIZATION_TOPIC"


def materialize_web_verifications(
    decision: WebEvidenceDecision,
    candidates: list[WebEvidenceCandidate],
) -> list[WebVerification]:
    supported_by_id: dict[str, SupportedWebEvidence] = {}
    for item in decision.supported:
        supported_by_id.setdefault(item.candidate_id, item)
    ambiguous_ids = set(decision.ambiguous_candidate_ids)
    by_page: dict[str, list[WebEvidenceCandidate]] = {}
    for candidate in candidates:
        by_page.setdefault(candidate.web_result_id, []).append(candidate)

    output: list[WebVerification] = []
    for web_result_id, page_candidates in by_page.items():
        evidence: list[WebEvidence] = []
        positions: list[str] = []
        matched_person = None
        matched_organization = None
        has_ambiguous = False
        for candidate in page_candidates:
            supported = supported_by_id.get(candidate.candidate_id)
            if candidate.candidate_id in ambiguous_ids:
                has_ambiguous = True
            if supported is None:
                continue
            position = normalize(supported.position or "")
            if candidate.kind == "IDENTITY":
                if not position or position not in normalize(candidate.text):
                    has_ambiguous = True
                    continue
                positions.append(position)
                matched_person = candidate.target_person
            matched_organization = candidate.target_organization
            if candidate.kind == "IDENTITY" and position:
                claim = (
                    f"{candidate.target_person}在"
                    f"{candidate.target_organization}担任{position}"
                )
            else:
                claim = candidate.text[:500]
            evidence.append(
                WebEvidence(
                    evidence_id=f"E{len(evidence) + 1}",
                    quote=candidate.text,
                    claim=claim,
                    matched_terms=candidate.matched_terms,
                )
            )

        keep = bool(evidence)
        if positions:
            reason = f"候选原文明确支持目标人物任职：{'、'.join(unique(positions))}"
        elif keep:
            reason = "候选原文明确支持目标企业相关事实"
        elif has_ambiguous:
            reason = "候选原文存在歧义或未给出可逐字核验的职位"
        else:
            reason = "候选原文不足以支持目标事实"
        output.append(
            WebVerification(
                web_result_id=web_result_id,
                keep=keep,
                matched_person=matched_person,
                matched_organization=matched_organization,
                identity_reason=reason,
                confidence=0.9 if keep else 0.2,
                same_name_risk=bool(
                    not keep
                    and any(item.kind == "IDENTITY" for item in page_candidates)
                ),
                conflicts=["候选原文存在歧义"] if has_ambiguous else [],
                evidence=evidence,
            )
        )
    return output


def _clean_candidate_text(value: str) -> str:
    without_controls = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", value or ""
    )
    return normalize(without_controls)


def _bounded_candidate_windows(
    text: str, anchors: list[str], *, max_segment_chars: int
) -> list[str]:
    windows: list[str] = []
    radius = max_segment_chars // 2
    for anchor in unique(anchors):
        start = 0
        while True:
            position = text.find(anchor, start)
            if position < 0:
                break
            left = max(0, position - radius)
            right = min(len(text), left + max_segment_chars)
            left = max(0, right - max_segment_chars)
            window = text[left:right].strip()
            if len(window) >= 10:
                windows.append(window)
            start = position + len(anchor)
    return unique(windows)


def claims_from_verifications(
    verifications: list[WebVerification], pages: list[WebPage]
) -> list[PublicClaim]:
    by_id = {page.web_result_id: page for page in pages}
    claims: list[PublicClaim] = []
    for verification in verifications:
        if not verification.keep:
            continue
        page = by_id.get(verification.web_result_id)
        if page is None:
            continue
        for evidence in verification.evidence:
            claims.append(
                PublicClaim(
                    web_result_id=verification.web_result_id,
                    evidence_id=evidence.evidence_id,
                    subject=verification.matched_person or verification.matched_organization or "目标实体",
                    claim=evidence.claim,
                    evidence_quote=evidence.quote,
                    source_title=page.title,
                    source_url=page.url,
                    evidence_source=_evidence_source(page, evidence.quote),
                    published_at=page.published_at,
                    matched_keywords=evidence.matched_terms,
                    confidence=verification.confidence,
                )
            )
    return claims


def fallback_report_content(
    input_text: str,
    context: ConfirmedContext,
    analysis: AssociationAnalysis,
    claims: list[PublicClaim],
    projects: list[ProjectResult],
) -> GeneratedReportContent:
    people = [entity.canonical_name for entity in context.entities if entity.entity_type == "PERSON"]
    return GeneratedReportContent(
        task_overview=build_task_overview(context),
        person_and_company_summary=build_person_identity_summaries(
            context, claims, []
        ),
        public_information_summary=[
            EvidenceBackedItem(
                text=claim.claim,
                statement_type="FACT",
                evidence_refs=[f"WEB:{claim.web_result_id}:{claim.evidence_id}"],
                confidence=claim.confidence,
            )
            for claim in claims[:6]
        ],
        priority_projects=analysis.related_projects[:3],
        resource_analysis=analysis.available_resources,
        recommended_topics=analysis.recommended_topics,
        advancement_advice=analysis.next_actions,
        preparation_items=analysis.next_actions,
        gaps_and_risks=[*analysis.risks, *analysis.information_gaps],
        action_brief=ActionBrief(
            destination=context.event_location,
            meeting_people=people,
            objective="围绕用户关注的业务方向了解合作机会",
            discussion_topics=context.business_directions,
            internal_contacts=unique([project.owner_name for project in projects]),
            preparation_items=["核对关键人身份", "联系相关项目负责人了解最新进展"],
            risks=[item.text for item in analysis.risks],
            evidence_refs=unique(
                [ref for section in analysis.model_dump().values() if isinstance(section, list) for item in section for ref in item.get("evidence_refs", [])]
            ),
        ),
    )


def validate_report_content(
    content: GeneratedReportContent,
    claims: list[PublicClaim],
    projects: list[ProjectResult],
    context: ConfirmedContext,
) -> GeneratedReportContent:
    web_refs = {f"WEB:{item.web_result_id}:{item.evidence_id}" for item in claims}
    project_refs = {f"PROJECT:{item.project_id}" for item in projects}
    context_refs = {"INPUT:ORIGINAL", "RULE:EXTRACTED", "CONFIRMATION:1"}
    allowed = web_refs | project_refs | context_refs

    def clean(items, section_refs):
        output = []
        seen = set()
        for item in items:
            refs = [ref for ref in item.evidence_refs if ref in section_refs]
            text = businessize_text(item.text)
            dedupe_key = normalize(text).casefold()
            if not refs or not text or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            output.append(item.model_copy(update={"text": text, "evidence_refs": refs}))
            if len(output) >= 6:
                break
        return output

    person_items = clean(content.person_and_company_summary, allowed)
    updates = {
        "task_overview": build_task_overview(context),
        "person_and_company_summary": build_person_identity_summaries(
            context, claims, person_items
        ),
        "public_information_summary": clean(
            content.public_information_summary, web_refs
        ),
        "priority_projects": clean(content.priority_projects, project_refs)[:3],
        "resource_analysis": clean(content.resource_analysis, web_refs | project_refs),
        "recommended_topics": clean(content.recommended_topics, allowed),
        "advancement_advice": clean(content.advancement_advice, allowed),
        "preparation_items": clean(content.preparation_items, allowed),
        "gaps_and_risks": clean(content.gaps_and_risks, allowed),
    }
    brief = content.action_brief
    updates["action_brief"] = brief.model_copy(
        update={
            "destination": businessize_text(brief.destination) if brief.destination else None,
            "objective": businessize_text(brief.objective),
            "discussion_topics": unique(businessize_text(item) for item in brief.discussion_topics),
            "internal_contacts": unique(businessize_text(item) for item in brief.internal_contacts),
            "preparation_items": unique(businessize_text(item) for item in brief.preparation_items),
            "risks": unique(businessize_text(item) for item in brief.risks),
            "evidence_refs": [ref for ref in brief.evidence_refs if ref in allowed],
        }
    )
    return content.model_copy(update=updates)


def build_person_identity_summaries(
    context: ConfirmedContext,
    claims: list[PublicClaim],
    candidates: list[EvidenceBackedItem],
) -> list[EvidenceBackedItem]:
    summaries: list[EvidenceBackedItem] = []
    role_terms = ("现任", "担任", "职位", "负责", "分管", "书记", "董事", "总经理", "总裁")
    activity_terms = ("参加", "出席", "调研", "会议", "学习", "检查", "慰问", "拜会")

    for entity in context.entities:
        if entity.entity_type != "PERSON":
            continue
        name = normalize(entity.canonical_name).casefold()
        ranked: list[tuple[int, int, EvidenceBackedItem]] = []
        for index, item in enumerate(candidates):
            text = normalize(item.text).casefold()
            if item.statement_type != "FACT" or name not in text:
                continue
            score = 4
            if entity.organization and normalize(entity.organization).casefold() in text:
                score += 3
            if entity.title and normalize(entity.title).casefold() in text:
                score += 3
            score += 2 * sum(term in item.text for term in role_terms)
            score -= 4 * sum(term in item.text for term in activity_terms)
            ranked.append((score, -index, item))
        best = max(ranked, key=lambda entry: (entry[0], entry[1])) if ranked else None
        if best and best[0] >= 7:
            summaries.append(best[2])
            continue

        refs = unique(
            f"WEB:{claim.web_result_id}:{claim.evidence_id}"
            for claim in claims
            if normalize(claim.subject).casefold() == name
            and (
                not entity.organization
                or normalize(entity.organization).casefold()
                in normalize(claim.claim).casefold()
            )
            and (
                not entity.title
                or normalize(entity.title).casefold()
                in normalize(claim.claim).casefold()
            )
        ) or ["CONFIRMATION:1"]
        if entity.organization and entity.title:
            text = f"{entity.canonical_name}现任{entity.organization}{entity.title}。"
        elif entity.organization:
            text = f"{entity.canonical_name}所属企业为{entity.organization}。"
        elif entity.title:
            text = f"{entity.canonical_name}现任{entity.title}。"
        else:
            text = f"已确认目标人物为{entity.canonical_name}。"
        summaries.append(
            EvidenceBackedItem(
                text=text,
                statement_type="FACT",
                evidence_refs=refs,
                confidence=1,
            )
        )
    return summaries


def complete_report_content(
    primary: GeneratedReportContent, fallback: GeneratedReportContent
) -> GeneratedReportContent:
    fields = (
        "task_overview",
        "person_and_company_summary",
        "public_information_summary",
        "priority_projects",
        "resource_analysis",
        "recommended_topics",
        "advancement_advice",
        "preparation_items",
        "gaps_and_risks",
    )
    primary_brief = primary.action_brief
    fallback_brief = fallback.action_brief
    updates = {
        field: merge_evidence_items(
            getattr(primary, field), getattr(fallback, field)
        )
        for field in fields
    }
    updates["priority_projects"] = updates["priority_projects"][:3]
    updates["action_brief"] = primary_brief.model_copy(
        update={
            "destination": primary_brief.destination or fallback_brief.destination,
            "meeting_people": primary_brief.meeting_people
            or fallback_brief.meeting_people,
            "objective": primary_brief.objective.strip()
            or fallback_brief.objective,
            "discussion_topics": unique(
                [*primary_brief.discussion_topics, *fallback_brief.discussion_topics]
            ),
            "internal_contacts": unique(
                [*primary_brief.internal_contacts, *fallback_brief.internal_contacts]
            ),
            "preparation_items": unique(
                [*primary_brief.preparation_items, *fallback_brief.preparation_items]
            ),
            "risks": unique([*primary_brief.risks, *fallback_brief.risks]),
            "evidence_refs": unique(
                [*primary_brief.evidence_refs, *fallback_brief.evidence_refs]
            ),
        }
    )
    return primary.model_copy(update=updates)


def merge_evidence_items(
    primary: list[EvidenceBackedItem], fallback: list[EvidenceBackedItem]
) -> list[EvidenceBackedItem]:
    output: list[EvidenceBackedItem] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for item in [*primary, *fallback]:
        key = (normalize(item.text).casefold(), tuple(sorted(item.evidence_refs)))
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def build_task_overview(context: ConfirmedContext) -> list[EvidenceBackedItem]:
    people = [
        entity.canonical_name
        for entity in context.entities
        if entity.entity_type == "PERSON"
    ]
    meeting_target = "、".join(people) if people else "相关人员"
    time = context.event_time or "时间未确认"
    location = context.event_location or "地点未确认"
    overview = [
        EvidenceBackedItem(
            text=f"{time}在{location}与{meeting_target}进行{context.event_type}。",
            statement_type="FACT",
            evidence_refs=["INPUT:ORIGINAL"],
            confidence=1,
        )
    ]
    focus = unique([*context.business_directions, *context.focus_questions])
    if focus:
        overview.append(
            EvidenceBackedItem(
                text=f"本次重点关注：{'；'.join(focus[:5])}。",
                statement_type="FACT",
                evidence_refs=["INPUT:ORIGINAL"],
                confidence=1,
            )
        )
    return overview


INTERNAL_TERMS = {
    "ACTIVE": "在建",
    "COMPLETED": "已结项",
    "start_date": "开始日期",
    "end_date": "结束日期",
    "owner_name": "项目负责人",
    "contact_name": "项目联系人",
    "project_id": "项目编号",
}


def businessize_text(value: str) -> str:
    output = value.strip()
    for internal, display in INTERNAL_TERMS.items():
        output = re.sub(rf"(?<![A-Za-z_]){re.escape(internal)}(?![A-Za-z_])", display, output)
    output = re.sub(r"结束日期\s*(?:为|=)?\s*(?:空|None|null|未填写)", "尚未记录结束日期", output, flags=re.IGNORECASE)
    output = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", output)
    return output



def unique(values) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _organization_terms(
    context: ConfirmedContext, target_organization: str | None = None
) -> list[str]:
    values = organization_aliases(target_organization)
    for entity in context.entities:
        if entity.entity_type == "PERSON":
            values.extend(organization_aliases(entity.organization))
        elif entity.entity_type == "ORGANIZATION":
            for name in [entity.canonical_name, *entity.aliases]:
                values.extend(organization_aliases(name))
    return unique(values)


def organization_aliases(value: str | None) -> list[str]:
    """Return conservative legal-name variants suitable for identity evidence."""
    name = normalize(value or "")
    if not name:
        return []
    aliases = [name]
    if name.endswith("工程有限公司"):
        aliases.append(f"{name[:-len('工程有限公司')]}公司")
    if name.endswith("股份有限公司"):
        aliases.append(name[: -len("有限公司")])
    if name.endswith("有限公司"):
        aliases.append(name[: -len("有限公司")])
    return unique(aliases)


def _evidence_source(page: WebPage, quote: str) -> str:
    if normalize(quote) in normalize(page.raw_content):
        return page.content_source
    return "SEARCH_SNIPPET"
