import re
from collections.abc import Callable

from app.schemas.task import (
    ActionBrief,
    AssociationAnalysis,
    ConfirmedContext,
    EntityMention,
    EvidenceBackedItem,
    ExtractedInfo,
    GeneratedReportContent,
    IntentUnderstanding,
    ProjectQueryPlan,
    ProjectRanking,
    ProjectRankingBatch,
    ProjectResult,
    PublicClaim,
    WebEvidence,
    WebPage,
    WebSearchPlan,
    WebSearchQuery,
    WebVerification,
    WebVerificationBatch,
)
from app.services.llm_client import StructuredLLM


class AgentNodes:
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def understanding(
        self, task_id: str, input_text: str, extracted: ExtractedInfo
    ) -> IntentUnderstanding:
        return self.llm.parse(
            task_id,
            "understanding",
            {"input_text": input_text, "rule_extraction": extracted.model_dump(mode="json")},
            IntentUnderstanding,
        )

    def web_plan(self, task_id: str, context: ConfirmedContext) -> WebSearchPlan:
        return self.llm.parse(
            task_id, "web_plan", {"confirmed_context": context.model_dump(mode="json")}, WebSearchPlan
        )

    def web_verify(
        self, task_id: str, context: ConfirmedContext, pages: list[WebPage]
    ) -> WebVerificationBatch:
        return self.llm.parse(
            task_id,
            "web_verify",
            {
                "confirmed_context": context.model_dump(mode="json"),
                "pages": [page.model_dump(mode="json") for page in pages],
            },
            WebVerificationBatch,
        )

    def project_query(self, task_id: str, context: ConfirmedContext) -> ProjectQueryPlan:
        return self.llm.parse(
            task_id,
            "project_query",
            {"confirmed_context": context.model_dump(mode="json")},
            ProjectQueryPlan,
        )

    def project_rerank(
        self,
        task_id: str,
        context: ConfirmedContext,
        projects: list[ProjectResult],
    ) -> ProjectRankingBatch:
        return self.llm.parse(
            task_id,
            "project_rerank",
            {
                "confirmed_context": context.model_dump(mode="json"),
                "projects": [project.model_dump(mode="json") for project in projects],
            },
            ProjectRankingBatch,
        )

    def association(
        self,
        task_id: str,
        context: ConfirmedContext,
        claims: list[PublicClaim],
        projects: list[ProjectResult],
        rankings: list[ProjectRanking],
    ) -> AssociationAnalysis:
        return self.llm.parse(
            task_id,
            "association",
            {
                "confirmed_context": context.model_dump(mode="json"),
                "public_claims": [claim.model_dump(mode="json") for claim in claims],
                "projects": [project.model_dump(mode="json") for project in projects],
                "rankings": [ranking.model_dump(mode="json") for ranking in rankings],
            },
            AssociationAnalysis,
        )

    def report_content(
        self,
        task_id: str,
        input_text: str,
        context: ConfirmedContext,
        claims: list[PublicClaim],
        projects: list[ProjectResult],
        analysis: AssociationAnalysis,
    ) -> GeneratedReportContent:
        return self.llm.parse(
            task_id,
            "report_content",
            {
                "input_text": input_text,
                "confirmed_context": context.model_dump(mode="json"),
                "public_claims": [claim.model_dump(mode="json") for claim in claims],
                "projects": [project.model_dump(mode="json") for project in projects],
                "association_analysis": analysis.model_dump(mode="json"),
            },
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


def validate_web_results(
    batch: WebVerificationBatch,
    pages: list[WebPage],
    context: ConfirmedContext,
    threshold: float,
) -> list[WebVerification]:
    by_id = {page.web_result_id: page for page in pages}
    people = {
        entity.canonical_name
        for entity in context.entities
        if entity.entity_type == "PERSON"
    }
    output: list[WebVerification] = []
    for result in batch.results:
        page = by_id.get(result.web_result_id)
        if page is None:
            continue
        body = normalize(_web_evidence_text(page))
        target_person = page.target_person or (
            result.matched_person if result.matched_person in people else None
        )
        organization_terms = _organization_terms(context, page.target_organization)
        organization_present = bool(
            not organization_terms
            or any(name in body for name in organization_terms)
        )
        matched_person_valid = bool(
            target_person
            and result.matched_person == target_person
            and target_person in body
            and organization_present
        )
        matched_organization_valid = bool(
            not target_person
            and not people
            and result.matched_organization
            and result.matched_organization in organization_terms
            and organization_present
        )
        identity_present = matched_person_valid or matched_organization_valid
        evidence = []
        for item in result.evidence:
            quote = normalize(item.quote)
            if quote not in body:
                continue
            if target_person:
                if target_person not in quote or (
                    organization_terms
                    and not any(name in quote for name in organization_terms)
                ):
                    continue
            elif not any(name in quote for name in organization_terms):
                continue
            evidence.append(item)
        keep = result.keep and result.confidence >= threshold and identity_present and bool(evidence)
        output.append(result.model_copy(update={"keep": keep, "evidence": evidence if keep else []}))
    return output


def strict_rule_verifications(
    pages: list[WebPage], context: ConfirmedContext, keywords: list[str]
) -> list[WebVerification]:
    people = [item.canonical_name for item in context.entities if item.entity_type == "PERSON"]
    output = []
    for page in pages:
        source_text = _web_evidence_text(page)
        plain = normalize(source_text)
        target_people = [page.target_person] if page.target_person else people
        person = next((name for name in target_people if name and name in plain), None)
        organization_terms = _organization_terms(context, page.target_organization)
        organization = next(
            (name for name in organization_terms if name in plain), None
        )
        keep = bool(person and (organization or not organization_terms))
        if not target_people:
            keep = bool(organization)
        evidence: list[WebEvidence] = []
        if keep:
            candidates = (
                _person_evidence_windows(source_text, person)
                if person
                else [
                    item.strip()
                    for item in re.split(r"[。！？!?；;\n]+", plain)
                    if item.strip()
                ]
            )
            for sentence in candidates:
                if person and person not in sentence:
                    continue
                if person and organization_terms and not any(
                    name in sentence for name in organization_terms
                ):
                    continue
                if not person and organization and organization not in sentence:
                    continue
                matched = [word for word in keywords if word in sentence]
                if len(sentence) >= 10:
                    evidence.append(
                        WebEvidence(
                            evidence_id=f"E{len(evidence) + 1}",
                            quote=sentence[:500],
                            claim=sentence[:500],
                            matched_terms=matched,
                        )
                    )
                if len(evidence) >= 3:
                    break
            keep = bool(evidence)
        output.append(
            WebVerification(
                web_result_id=page.web_result_id,
                keep=keep,
                matched_person=person,
                matched_organization=organization,
                identity_reason="规则确认正文命中已确认人物与单位或目标企业" if keep else "正文未命中已确认身份",
                confidence=0.85 if keep else 0.2,
                same_name_risk=bool(target_people and not organization),
                evidence=evidence,
            )
        )
    return output


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


def deterministic_rankings(projects: list[ProjectResult], context: ConfirmedContext) -> list[ProjectRanking]:
    match_scores = {"PERSON_EXACT": 95, "ORG_EXACT": 85, "PROJECT_EXACT": 90, "TEXT_MATCH": 65, "VECTOR_MATCH": 50}
    output = []
    for project in projects:
        score = match_scores[project.match_type]
        if project.status == "ACTIVE":
            score = min(100, score + 3)
        output.append(
            ProjectRanking(
                project_id=project.project_id,
                relevance_score=score,
                relevance_reason=f"确定性匹配依据为 {project.match_type}，项目状态为 {project.status}",
                recommended_use="会面中了解当前进展" if project.status == "ACTIVE" else "作为历史合作案例",
                related_internal_resource=project.owner_name,
                confidence=0.9 if project.match_type in {"PERSON_EXACT", "ORG_EXACT", "PROJECT_EXACT"} else 0.65,
                evidence_refs=[f"PROJECT:{project.project_id}"],
            )
        )
    return sorted(output, key=lambda item: (-item.relevance_score, item.project_id))


def validate_rankings(
    rankings: list[ProjectRanking], projects: list[ProjectResult], threshold: float
) -> list[ProjectRanking]:
    by_id = {project.project_id: project for project in projects}
    fallback = {item.project_id: item for item in deterministic_rankings(projects, _empty_context())}
    output = []
    seen = set()
    for ranking in rankings:
        if ranking.project_id not in by_id or ranking.project_id in seen:
            continue
        seen.add(ranking.project_id)
        valid_ref = f"PROJECT:{ranking.project_id}"
        if ranking.confidence < threshold:
            output.append(fallback[ranking.project_id])
        else:
            output.append(ranking.model_copy(update={"evidence_refs": [valid_ref]}))
    for project_id, ranking in fallback.items():
        if project_id not in seen:
            output.append(ranking)
    return sorted(output, key=lambda item: (-item.relevance_score, item.project_id))


def fallback_association(
    claims: list[PublicClaim], projects: list[ProjectResult], rankings: list[ProjectRanking]
) -> AssociationAnalysis:
    findings = [
        EvidenceBackedItem(
            text=claim.claim,
            statement_type="FACT",
            evidence_refs=[f"WEB:{claim.web_result_id}:{claim.evidence_id}"],
            confidence=claim.confidence,
        )
        for claim in claims[:5]
    ]
    related = [
        EvidenceBackedItem(
            text=f"{project.project_name}，状态：{project.status}，负责人：{project.owner_name}",
            statement_type="FACT",
            evidence_refs=[f"PROJECT:{project.project_id}"],
            confidence=next((item.confidence for item in rankings if item.project_id == project.project_id), 0.8),
        )
        for project in projects[:5]
    ]
    resources = [
        EvidenceBackedItem(
            text="；".join(
                filter(
                    None,
                    (
                        f"{project.project_name} 的我方项目销售员为 {project.owner_name}",
                        f"联系电话 {project.owner_phone}" if project.owner_phone else None,
                        f"邮箱 {project.owner_email}" if project.owner_email else None,
                        (
                            f"客户联系人为 {project.contact_name}"
                            + (
                                f"（{project.customer_contact_title}）"
                                if project.customer_contact_title
                                else ""
                            )
                            + (
                                f"，联系电话 {project.customer_contact_phone}"
                                if project.customer_contact_phone
                                else ""
                            )
                            if project.contact_name
                            else None
                        ),
                        (
                            f"上级为 {project.owner_manager_name}，负责 {project.owner_region}"
                            if project.owner_manager_name and project.owner_region
                            else None
                        ),
                    ),
                )
            ),
            statement_type="FACT",
            evidence_refs=[f"PROJECT:{project.project_id}"],
            confidence=0.8,
        )
        for project in projects[:3]
    ]
    gaps = []
    if not claims:
        gaps.append(EvidenceBackedItem(text="本次未使用联网身份来源", statement_type="FACT", evidence_refs=["INPUT:ORIGINAL"], confidence=1))
    if not projects:
        gaps.append(EvidenceBackedItem(text="未检索到相关内部项目", statement_type="FACT", evidence_refs=["INPUT:ORIGINAL"], confidence=1))
    return AssociationAnalysis(
        key_findings=findings,
        related_projects=related,
        available_resources=resources,
        recommended_topics=[
            EvidenceBackedItem(
                text=f"可围绕 {project.project_name} 的当前进展、客户需求和下一步安排展开沟通",
                statement_type="RECOMMENDATION",
                evidence_refs=[f"PROJECT:{project.project_id}"],
                confidence=0.8,
            )
            for project in projects[:3]
        ],
        risks=[],
        information_gaps=gaps,
        next_actions=resources,
    )


def validate_analysis(
    analysis: AssociationAnalysis,
    claims: list[PublicClaim],
    projects: list[ProjectResult],
    threshold: float,
) -> AssociationAnalysis:
    allowed = {f"WEB:{item.web_result_id}:{item.evidence_id}" for item in claims}
    allowed.update(f"PROJECT:{item.project_id}" for item in projects)
    allowed.update({"INPUT:ORIGINAL", "RULE:EXTRACTED", "CONFIRMATION:1"})

    def clean(items: list[EvidenceBackedItem]) -> list[EvidenceBackedItem]:
        return [
            item.model_copy(update={"evidence_refs": [ref for ref in item.evidence_refs if ref in allowed]})
            for item in items
            if item.confidence >= threshold and any(ref in allowed for ref in item.evidence_refs)
        ]

    return AssociationAnalysis(**{name: clean(getattr(analysis, name)) for name in AssociationAnalysis.model_fields})


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


def complete_analysis(
    primary: AssociationAnalysis, fallback: AssociationAnalysis
) -> AssociationAnalysis:
    fields = (
        "key_findings",
        "related_projects",
        "available_resources",
        "recommended_topics",
        "risks",
        "information_gaps",
        "next_actions",
    )
    return primary.model_copy(
        update={
            field: merge_evidence_items(
                getattr(primary, field), getattr(fallback, field)
            )
            for field in fields
        }
    )


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


def _empty_context() -> ConfirmedContext:
    return ConfirmedContext(intents=["REPORT_GENERATION"], entities=[], event_type="其他")


def unique(values) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _web_evidence_text(page: WebPage) -> str:
    if page.search_snippet and page.search_snippet not in page.raw_content:
        return f"{page.raw_content}\n{page.search_snippet}"
    return page.raw_content or page.search_snippet


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


def _person_evidence_windows(text: str, person: str, radius: int = 250) -> list[str]:
    lines = [normalize(line) for line in text.splitlines() if normalize(line)]
    windows: list[str] = []
    for index, line in enumerate(lines):
        if person not in line:
            continue
        block = [line]
        for following in lines[index + 1 : index + 4]:
            if following.startswith("姓名："):
                break
            block.append(following)
        windows.append(" ".join(block))

    if windows:
        distinct = sorted(unique(windows), key=len, reverse=True)
        return [
            window
            for index, window in enumerate(distinct)
            if not any(window in longer for longer in distinct[:index])
        ]

    plain = normalize(text)
    start = 0
    while True:
        position = plain.find(person, start)
        if position < 0:
            break
        windows.append(
            plain[max(0, position - radius) : position + len(person) + radius].strip()
        )
        start = position + len(person)
    return unique(windows)


def _evidence_source(page: WebPage, quote: str) -> str:
    if normalize(quote) in normalize(page.raw_content):
        return page.content_source
    return "SEARCH_SNIPPET"
