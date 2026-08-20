import re
from dataclasses import dataclass
from typing import Protocol

from app.schemas.task import (
    ConfirmedContext,
    PublicClaim,
    SupportedWebEvidence,
    WebEvidenceCandidate,
    WebEvidenceDecision,
    WebPage,
    WebSearchQuery,
    WebVerification,
)
from app.services.research.agent_nodes import (
    build_web_verification_candidates,
    claims_from_verifications,
    materialize_web_verifications,
    organization_aliases,
)


CURRENT_RELATION_MARKERS = (
    "现任",
    "现担任",
    "担任",
    "出任",
    "就任",
    "任命为",
    "被任命为",
)
HISTORICAL_RELATION_MARKERS = (
    "曾任",
    "原任",
    "前任",
    "卸任",
    "辞任",
    "离任",
    "不再担任",
    "不再任职",
)
NEGATIVE_RELATION_MARKERS = ("并非", "不是", "未担任", "从未担任", "否认担任")
ORGANIZATION_FACT_MARKERS = (
    "发布",
    "建设",
    "推进",
    "开展",
    "布局",
    "投资",
    "签署",
    "中标",
    "实现",
    "拥有",
    "主营",
    "聚焦",
    "扩大",
    "新增",
    "投产",
    "启动",
    "完成",
    "合作",
    "研发",
    "生产",
    "销售",
)
ROLE_PATTERN = re.compile(
    r"(?:党委书记|党委副书记|党支部书记|董事长|副董事长|"
    r"总经理|副总经理|总裁|副总裁|首席执行官|CEO|CFO|CTO|"
    r"主任|副主任|负责人|经理)"
    r"(?:(?:兼任?|、|和|及)"
    r"(?:党委书记|党委副书记|党支部书记|董事长|副董事长|"
    r"总经理|副总经理|总裁|副总裁|首席执行官|CEO|CFO|CTO|"
    r"主任|副主任|负责人|经理)){0,3}"
)


class EvidenceAgent(Protocol):
    def evidence_verify(
        self,
        task_id: str,
        candidates: list[WebEvidenceCandidate],
    ) -> WebEvidenceDecision: ...


class EvidenceEventRecorder(Protocol):
    def log_execution_event(self, scope_id: str, **values: object) -> object: ...


@dataclass(frozen=True)
class EvidenceProcessingResult:
    verifications: tuple[WebVerification, ...]
    claims: tuple[PublicClaim, ...]
    degraded_nodes: tuple[str, ...] = ()


class AgentEvidenceProcessor:
    def __init__(
        self,
        agent: EvidenceAgent,
        event_recorder: EvidenceEventRecorder,
    ):
        self.agent = agent
        self.event_recorder = event_recorder

    def process(
        self,
        task_id: str,
        pages: list[WebPage],
        context: ConfirmedContext,
        queries: list[WebSearchQuery],
    ) -> EvidenceProcessingResult:
        candidates = build_web_verification_candidates(pages, context, queries)
        routing = route_web_evidence_candidates(candidates, context)
        self._record(
            task_id,
            event_type="RULE_ROUTING",
            node_name="evidence_verify",
            status="SUCCESS",
            title="规则分流公开证据候选",
            detail=(
                f"规则接受 {len(routing.accepted)} 条，拒绝 "
                f"{len(routing.rejected)} 条，待模型核验 "
                f"{len(routing.ambiguous)} 条。"
            ),
            payload={
                "generator": "strict_evidence_rules",
                "accepted_candidate_ids": [
                    item.candidate_id for item in routing.accepted
                ],
                "rejected_candidate_ids": [
                    item.candidate_id for item in routing.rejected
                ],
                "ambiguous_candidate_ids": [
                    item.candidate_id for item in routing.ambiguous
                ],
            },
        )
        decision = None
        failed = False
        if routing.ambiguous:
            try:
                decision = self.agent.evidence_verify(
                    task_id,
                    list(routing.ambiguous),
                )
            except Exception as exc:
                failed = True
                self._record(
                    task_id,
                    event_type="FALLBACK",
                    node_name="evidence_verify",
                    status="DEGRADED",
                    title="公开证据模型核验已降级",
                    detail="歧义候选保持未核验，研究流水线继续。",
                    payload={
                        "error_type": type(exc).__name__,
                        "ambiguous_candidate_ids": [
                            item.candidate_id for item in routing.ambiguous
                        ],
                    },
                )
        verifications = materialize_routed_web_verifications(
            routing,
            decision,
            llm_failed=failed,
        )
        claims = claims_from_verifications(verifications, pages)
        return EvidenceProcessingResult(
            verifications=tuple(verifications),
            claims=tuple(claims),
            degraded_nodes=("evidence_verify",) if failed else (),
        )

    def _record(self, task_id: str, **values) -> None:
        logger = getattr(self.event_recorder, "log_execution_event", None)
        if logger is not None:
            logger(task_id, **values)


@dataclass(frozen=True)
class EvidenceRouting:
    candidates: tuple[WebEvidenceCandidate, ...]
    accepted: tuple[WebEvidenceCandidate, ...]
    rejected: tuple[WebEvidenceCandidate, ...]
    ambiguous: tuple[WebEvidenceCandidate, ...]
    accepted_support: tuple[SupportedWebEvidence, ...]


def route_web_evidence_candidates(
    candidates: list[WebEvidenceCandidate],
    context: ConfirmedContext,
) -> EvidenceRouting:
    routed_candidates = []
    accepted = []
    rejected = []
    ambiguous = []
    accepted_support = []

    for candidate in candidates:
        if candidate.kind == "IDENTITY":
            route, routed, position = _route_identity(candidate, context)
        else:
            route, routed, position = _route_organization_topic(candidate)
        routed_candidates.append(routed)
        if route == "accepted":
            accepted.append(routed)
            accepted_support.append(
                SupportedWebEvidence(
                    candidate_id=routed.candidate_id,
                    position=position,
                )
            )
        elif route == "rejected":
            rejected.append(routed)
        else:
            ambiguous.append(routed)

    return EvidenceRouting(
        candidates=tuple(routed_candidates),
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        ambiguous=tuple(ambiguous),
        accepted_support=tuple(accepted_support),
    )


def materialize_routed_web_verifications(
    routing: EvidenceRouting,
    llm_decision: WebEvidenceDecision | None = None,
    *,
    llm_failed: bool = False,
) -> list[WebVerification]:
    ambiguous_ids = {item.candidate_id for item in routing.ambiguous}
    supported = list(routing.accepted_support)
    unresolved = set(ambiguous_ids) if llm_failed else set()

    if llm_decision is not None and not llm_failed:
        seen = {item.candidate_id for item in supported}
        for item in llm_decision.supported:
            if item.candidate_id in ambiguous_ids and item.candidate_id not in seen:
                supported.append(item)
                seen.add(item.candidate_id)
        unresolved.update(
            candidate_id
            for candidate_id in llm_decision.ambiguous_candidate_ids
            if candidate_id in ambiguous_ids and candidate_id not in seen
        )

    decision = WebEvidenceDecision(
        supported=supported,
        ambiguous_candidate_ids=sorted(unresolved),
    )
    verifications = materialize_web_verifications(
        decision,
        list(routing.candidates),
    )
    if not llm_failed:
        return verifications

    unresolved_pages = {
        item.web_result_id
        for item in routing.ambiguous
        if item.candidate_id in unresolved
    }
    return [
        item.model_copy(
            update={
                "identity_reason": (
                    f"{item.identity_reason}；模型不可用，歧义候选未核验"
                ),
                "same_name_risk": item.same_name_risk
                or item.web_result_id in unresolved_pages,
                "conflicts": list(
                    dict.fromkeys([*item.conflicts, "歧义候选未核验"])
                ),
            }
        )
        if item.web_result_id in unresolved_pages
        else item
        for item in verifications
    ]


def _route_identity(
    candidate: WebEvidenceCandidate,
    context: ConfirmedContext,
) -> tuple[str, WebEvidenceCandidate, str | None]:
    person = candidate.target_person or ""
    organizations = organization_aliases(candidate.target_organization)
    text = _normalize(candidate.text)
    if not person or person not in text or not any(item in text for item in organizations):
        return "rejected", candidate, None
    if any(marker in text for marker in NEGATIVE_RELATION_MARKERS):
        return "rejected", candidate, None
    if any(marker in text for marker in HISTORICAL_RELATION_MARKERS):
        return "ambiguous", candidate, None

    confirmed_title = next(
        (
            item.title
            for item in context.entities
            if item.entity_type == "PERSON"
            and item.canonical_name == person
            and item.title
        ),
        None,
    )
    for sentence in _sentences(text):
        organization = next((item for item in organizations if item in sentence), None)
        if person not in sentence or organization is None:
            continue
        for position in _positions(sentence, confirmed_title):
            if _has_direct_identity_relation(
                sentence,
                person,
                organization,
                position,
            ):
                return (
                    "accepted",
                    candidate.model_copy(update={"text": sentence[:1_000]}),
                    position,
                )
    return "ambiguous", candidate, None


def _route_organization_topic(
    candidate: WebEvidenceCandidate,
) -> tuple[str, WebEvidenceCandidate, None]:
    organizations = organization_aliases(candidate.target_organization)
    text = _normalize(candidate.text)
    if not organizations or not any(item in text for item in organizations):
        return "rejected", candidate, None
    topic_terms = [item for item in candidate.matched_terms if item in text]
    if candidate.matched_terms and not topic_terms:
        return "rejected", candidate, None
    for sentence in _sentences(text):
        if not any(item in sentence for item in organizations):
            continue
        if topic_terms and not any(item in sentence for item in topic_terms):
            continue
        if any(marker in sentence for marker in ORGANIZATION_FACT_MARKERS):
            return (
                "accepted",
                candidate.model_copy(update={"text": sentence[:1_000]}),
                None,
            )
    return "ambiguous", candidate, None


def _positions(sentence: str, confirmed_title: str | None) -> list[str]:
    output = []
    if confirmed_title and confirmed_title in sentence:
        output.append(confirmed_title)
    output.extend(match.group(0) for match in ROLE_PATTERN.finditer(sentence))
    return list(dict.fromkeys(output))


def _has_direct_identity_relation(
    sentence: str,
    person: str,
    organization: str,
    position: str,
) -> bool:
    relation = "(?:" + "|".join(
        re.escape(marker) for marker in CURRENT_RELATION_MARKERS
    ) + ")"
    gap = r"[^，,。；;！？!?]"
    compact_patterns = (
        rf"{re.escape(person)}{gap}{{0,12}}{relation}{gap}{{0,30}}"
        rf"{re.escape(organization)}{gap}{{0,20}}{re.escape(position)}",
        rf"{re.escape(person)}{gap}{{0,12}}{re.escape(organization)}"
        rf"{gap}{{0,12}}{relation}{gap}{{0,20}}{re.escape(position)}",
        rf"{re.escape(organization)}{gap}{{0,20}}{re.escape(person)}"
        rf"{gap}{{0,8}}{relation}{gap}{{0,20}}{re.escape(position)}",
        rf"{re.escape(person)}[，,:： ]{{0,3}}{re.escape(organization)}"
        rf"[，,:： ]{{0,3}}{re.escape(position)}",
        rf"{re.escape(organization)}[，,:： ]{{0,3}}{re.escape(position)}"
        rf"[，,:： ]{{0,3}}{re.escape(person)}",
    )
    return any(re.search(pattern, sentence) for pattern in compact_patterns)


def _sentences(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"[。！？!?；;\n]+", text)
        if item.strip()
    ]


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()
