import re

from app.schemas.intake import IntakeChatResult


def is_intake_ready(result: IntakeChatResult, source_text: str | None = None) -> bool:
    return not required_missing_information(result, source_text)


def required_missing_information(
    result: IntakeChatResult, source_text: str | None = None
) -> list[str]:
    context = result.structured_context
    targets = [*context.people, *context.organizations, *context.projects]
    has_target = bool(targets)
    if source_text is not None:
        normalized_source = _normalize(source_text)
        has_target = any(_normalize(target) in normalized_source for target in targets)
    has_focus = bool(context.focus_questions)
    missing: list[str] = []
    if not has_target:
        missing.append("人物、企业或项目")
    if not has_focus:
        missing.append("希望分析或推动的事项")
    return missing


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()
