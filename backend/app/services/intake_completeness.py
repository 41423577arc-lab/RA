import re

from app.schemas.intake import IntakeChatResult


def is_intake_ready(result: IntakeChatResult, source_text: str | None = None) -> bool:
    return not required_missing_information(result, source_text)


def required_missing_information(
    result: IntakeChatResult, source_text: str | None = None
) -> list[str]:
    context = result.structured_context
    targets = [*context.people, *context.organizations]
    has_target = bool(targets)
    if source_text is not None:
        normalized_source = _normalize(source_text)
        source_candidates = [
            *targets,
            *(
                value
                for item in context.entity_resolutions
                for value in (item.get("mention"), item.get("canonical_name"))
                if value
            ),
        ]
        has_target = any(
            _normalize(target) in normalized_source for target in source_candidates
        )
    missing: list[str] = []
    if not has_target:
        missing.append("候选人姓名或候选企业")
    return missing


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()
