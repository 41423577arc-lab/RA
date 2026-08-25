import re
import unicodedata
from collections.abc import Callable
from typing import Protocol

from app.schemas.intake import (
    ExternalIdentityNormalizationResult,
    IntakeEntityResolution,
    IntakeStructuredContext,
)
from app.schemas.task import ConfirmationItem, ConfirmationRequest
from app.services.intake.agent_loop import QueryPlan, ToolObservation


ExternalNormalizer = Callable[
    [list[dict], list[dict]], ExternalIdentityNormalizationResult
]


class EntityCandidateBackend(Protocol):
    def lookup_internal(
        self,
        context: IntakeStructuredContext,
        version: int,
        source_text: str | None = None,
        *,
        raise_on_error: bool = False,
    ) -> tuple[list[dict], ConfirmationRequest | None]: ...

    def search_key_person_identity_web(
        self,
        context: IntakeStructuredContext,
        confirmation: ConfirmationRequest,
        external_normalizer: ExternalNormalizer,
        *,
        raise_on_error: bool = False,
    ) -> ConfirmationRequest: ...

    def apply_automatic_candidates(
        self,
        resolutions: list[dict],
        confirmation: ConfirmationRequest | None,
        threshold: float = 0.8,
    ) -> tuple[list[dict], ConfirmationRequest | None]: ...


class IntakeQueryExecutor:
    """把模型查询计划编译为现有身份候选服务的受控调用。"""

    def __init__(
        self,
        candidates: EntityCandidateBackend,
        *,
        automatic_threshold: float = 0.8,
    ):
        self.candidates = candidates
        self.automatic_threshold = automatic_threshold

    def execute(
        self,
        plan: QueryPlan,
        context: IntakeStructuredContext,
        *,
        version: int,
        source_text: str | None = None,
        confirmation: ConfirmationRequest | None = None,
        external_normalizer: ExternalNormalizer | None = None,
    ) -> ToolObservation:
        controlled_context = self._controlled_context(plan, context)
        executed_query = self._executed_query(controlled_context)
        try:
            if plan.action == "SEARCH_INTERNAL":
                resolutions, pending = self.candidates.lookup_internal(
                    controlled_context,
                    version,
                    source_text,
                    raise_on_error=True,
                )
            else:
                if confirmation is None:
                    raise ValueError("公网身份查询缺少内部候选 Observation")
                if external_normalizer is None:
                    raise ValueError("公网身份查询缺少候选标准化能力")
                pending = self.candidates.search_key_person_identity_web(
                    controlled_context,
                    confirmation,
                    external_normalizer,
                    raise_on_error=True,
                )
                resolutions = []

            resolutions, pending = self.candidates.apply_automatic_candidates(
                resolutions,
                pending,
                self.automatic_threshold,
            )

            typed_resolutions = [
                IntakeEntityResolution.model_validate(item)
                for item in resolutions[: plan.result_limit]
            ]
            limited_confirmation = self._limit_confirmation(
                pending,
                plan.result_limit,
            )
            information_status = self._information_status(
                typed_resolutions,
                limited_confirmation,
            )
            return ToolObservation(
                action=plan.action,
                target_fields=plan.target_fields,
                executed_query=executed_query,
                technical_status="SUCCESS",
                information_status=information_status,
                resolutions=typed_resolutions,
                confirmation=limited_confirmation,
                summary=self._summary(
                    typed_resolutions,
                    limited_confirmation,
                    information_status,
                ),
            )
        except Exception as exc:
            return ToolObservation(
                action=plan.action,
                target_fields=plan.target_fields,
                executed_query=executed_query,
                technical_status="FAILED",
                information_status="NO_RESULT",
                error=f"{type(exc).__name__}: {exc}"[:1_000],
            )

    def controlled_query(
        self,
        plan: QueryPlan,
        context: IntakeStructuredContext,
    ) -> str:
        return self._executed_query(self._controlled_context(plan, context))

    @staticmethod
    def _controlled_context(
        plan: QueryPlan,
        context: IntakeStructuredContext,
    ) -> IntakeStructuredContext:
        people = IntakeQueryExecutor._allowed_mentions(
            plan.person_mentions,
            context.people,
        )
        organizations = IntakeQueryExecutor._allowed_mentions(
            plan.organization_mentions,
            context.organizations,
        )
        return context.model_copy(
            update={
                "people": people,
                "people_details": [
                    item for item in context.people_details if item.name in people
                ],
                "organizations": organizations,
            },
            deep=True,
        )

    @staticmethod
    def _allowed_mentions(requested: list[str], known: list[str]) -> list[str]:
        if not requested:
            return list(known)
        requested_keys = {IntakeQueryExecutor._normalize(item) for item in requested}
        selected = [
            item
            for item in known
            if IntakeQueryExecutor._normalize(item) in requested_keys
        ]
        return selected or list(known)

    @staticmethod
    def _executed_query(context: IntakeStructuredContext) -> str:
        parts = [
            *(f"人物:{item}" for item in context.people),
            *(f"企业:{item}" for item in context.organizations),
        ]
        return "；".join(parts) or "身份候选"

    @staticmethod
    def _limit_confirmation(
        confirmation: ConfirmationRequest | None,
        result_limit: int,
    ) -> ConfirmationRequest | None:
        if confirmation is None:
            return None
        return ConfirmationRequest(
            version=confirmation.version,
            items=[
                ConfirmationItem(
                    mention=item.mention,
                    entity_type=item.entity_type,
                    candidates=item.candidates[:result_limit],
                    required=item.required,
                )
                for item in confirmation.items[:result_limit]
            ],
        )

    @staticmethod
    def _information_status(
        resolutions: list[IntakeEntityResolution],
        confirmation: ConfirmationRequest | None,
    ) -> str:
        if resolutions and confirmation is None:
            return "RESOLVED"
        if resolutions or (
            confirmation
            and any(item.candidates for item in confirmation.items)
        ):
            return "PARTIAL"
        return "NO_RESULT"

    @staticmethod
    def _summary(
        resolutions: list[IntakeEntityResolution],
        confirmation: ConfirmationRequest | None,
        information_status: str,
    ) -> str:
        pending_count = len(confirmation.items) if confirmation else 0
        if information_status == "NO_RESULT":
            return "查询执行成功，但未获得可用身份信息。"
        return (
            f"确认身份 {len(resolutions)} 个，"
            f"仍有 {pending_count} 个身份项需要处理。"
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(
            r"\s+",
            "",
            unicodedata.normalize("NFKC", value),
        ).casefold()
