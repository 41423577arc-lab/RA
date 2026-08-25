from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.intake import (
    IntakeAction,
    IntakeEntityResolution,
    IntakeEntityType,
    IntakeStructuredContext,
    IntakeToolAction,
)
from app.schemas.task import ConfirmationRequest


IntakeSkillName = Literal[
    "identity_resolution",
    "internal_lookup",
    "public_lookup",
    "intake_readiness",
]
TechnicalStatus = Literal["SUCCESS", "FAILED"]
InformationStatus = Literal["RESOLVED", "PARTIAL", "NO_RESULT"]

_CONTROLLED_CONTEXT_FIELDS = {
    "entity_resolutions",
    "tool_attempts",
    "final_confirmation",
}
_PATCHABLE_CONTEXT_FIELDS = (
    set(IntakeStructuredContext.model_fields) - _CONTROLLED_CONTEXT_FIELDS
)


class IntakeContextPatch(BaseModel):
    """表示单次模型推理建议更新的业务字段。"""

    model_config = ConfigDict(frozen=True)

    updates: dict[str, Any] = Field(default_factory=dict, max_length=30)

    @field_validator("updates")
    @classmethod
    def validate_updates(cls, updates: dict[str, Any]) -> dict[str, Any]:
        unknown_fields = set(updates) - _PATCHABLE_CONTEXT_FIELDS
        if unknown_fields:
            fields = "、".join(sorted(unknown_fields))
            raise ValueError(f"Context Patch 包含不可修改字段：{fields}")

        normalized: dict[str, Any] = {}
        for field_name, value in updates.items():
            context = IntakeStructuredContext.model_validate({field_name: value})
            normalized[field_name] = getattr(context, field_name)
        return normalized


class QueryPlan(BaseModel):
    """模型只描述查询意图，实际工具参数由 Python 执行器生成。"""

    model_config = ConfigDict(frozen=True)

    action: IntakeToolAction
    target_fields: list[str] = Field(default_factory=list, max_length=20)
    entity_types: list[IntakeEntityType] = Field(default_factory=list, max_length=2)
    person_mentions: list[str] = Field(default_factory=list, max_length=20)
    organization_mentions: list[str] = Field(default_factory=list, max_length=20)
    relationship_hints: list[str] = Field(default_factory=list, max_length=20)
    result_limit: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def require_query_target(self):
        if not (
            self.target_fields
            or self.person_mentions
            or self.organization_mentions
            or self.relationship_hints
        ):
            raise ValueError("QueryPlan 至少需要一个查询目标")
        return self


class ToolObservation(BaseModel):
    """统一表示工具技术状态和信息补充状态。"""

    model_config = ConfigDict(frozen=True)

    action: IntakeToolAction
    target_fields: list[str] = Field(default_factory=list, max_length=20)
    executed_query: str = Field(min_length=1, max_length=500)
    technical_status: TechnicalStatus
    information_status: InformationStatus
    resolutions: list[IntakeEntityResolution] = Field(default_factory=list, max_length=40)
    confirmation: ConfirmationRequest | None = None
    summary: str = Field(default="", max_length=4_000)
    error: str | None = Field(default=None, max_length=1_000)

    @model_validator(mode="after")
    def validate_statuses(self):
        if self.technical_status == "FAILED":
            if self.information_status != "NO_RESULT":
                raise ValueError("工具技术失败时信息状态必须为 NO_RESULT")
            if not self.error:
                raise ValueError("工具技术失败时必须记录错误信息")
        elif self.error:
            raise ValueError("工具技术成功时不能携带错误信息")
        return self


class AgentTurn(BaseModel):
    """表示模型在一次决策中输出的受控动作。"""

    model_config = ConfigDict(frozen=True)

    context_patch: IntakeContextPatch = Field(default_factory=IntakeContextPatch)
    skill: IntakeSkillName
    next_action: IntakeAction
    query_plan: QueryPlan | None = None
    user_message: str | None = Field(default=None, min_length=1, max_length=1_000)
    reason: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_action_payload(self):
        if self.next_action in ("SEARCH_INTERNAL", "SEARCH_PUBLIC"):
            if self.query_plan is None:
                raise ValueError("查询动作必须提供 QueryPlan")
            if self.query_plan.action != self.next_action:
                raise ValueError("QueryPlan 动作必须与 next_action 一致")
            if self.user_message is not None:
                raise ValueError("查询动作不能同时向用户提问")
        elif self.next_action == "ASK_USER":
            if self.user_message is None:
                raise ValueError("ASK_USER 必须提供用户问题")
            if self.query_plan is not None:
                raise ValueError("ASK_USER 不能携带 QueryPlan")
        elif self.query_plan is not None or self.user_message is not None:
            raise ValueError("READY 不能携带 QueryPlan 或用户问题")
        return self


class AgentState(BaseModel):
    """Intake Agent 的请求内运行态，业务事实只保存在 context 中。"""

    model_config = ConfigDict(frozen=True)

    context: IntakeStructuredContext
    latest_observation: ToolObservation | None = None
    loop_count: int = Field(default=0, ge=0)
    llm_turn_count: int = Field(default=0, ge=0)
