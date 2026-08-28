import json
import re
import tomllib
from dataclasses import dataclass
from typing import Any


_MARKDOWN_LINK = re.compile(r"^\[([^\]]+)]\(([^)]+)\)$")
_ENV_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SLUG_PART = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ImportedModelProfile:
    role: str
    name: str
    slug: str
    model_id: str


@dataclass(frozen=True)
class ModelImportPreview:
    source_format: str
    connection_name: str
    connection_slug: str
    provider: str
    base_url: str
    api_mode: str
    parameters: dict[str, Any]
    profiles: tuple[ImportedModelProfile, ...]
    ignored_fields: tuple[str, ...]


def parse_model_configuration(content: str) -> ModelImportPreview:
    text = content.strip()
    if not text:
        raise ValueError("Model configuration cannot be empty")
    data, source_format = _parse_document(text)
    normalized = _normalize_mapping(data)

    # 厂商名称只用于定位配置；最终执行仍落到现有协议 Adapter。
    provider_name = _first(normalized, "model_provider", "provider")
    providers = normalized.get("model_providers")
    provider_config: dict[str, Any] = {}
    if isinstance(providers, dict):
        if provider_name:
            provider_config = _mapping_value(providers, str(provider_name)) or {}
        elif len(providers) == 1:
            provider_name, provider_config = next(iter(providers.items()))
    if not isinstance(provider_config, dict):
        raise ValueError("Selected model provider configuration must be an object")
    provider_name = str(
        provider_name or provider_config.get("name") or "openai_compatible"
    ).strip()
    provider = provider_name.lower().replace("-", "_").replace(" ", "_")
    if provider not in {"openai", "openai_compatible"}:
        raise ValueError(f"Unsupported model provider protocol: {provider_name}")

    base_url = _first(
        provider_config,
        "base_url",
    ) or _first(normalized, "base_url", "openai_base_url")
    if not base_url:
        raise ValueError("Model configuration is missing Base URL")
    base_url = _unwrap_markdown_url(str(base_url).strip())

    model_id = _first(normalized, "model", "model_id", "llm_model")
    if not model_id:
        raise ValueError("Model configuration is missing Model ID")
    model_id = str(model_id).strip()
    review_model = _first(
        normalized, "review_model", "review_model_id", "llm_review_model"
    )
    review_model = str(review_model).strip() if review_model else None

    raw_api_mode = _first(
        provider_config, "wire_api", "api_mode"
    ) or _first(normalized, "wire_api", "api_mode", "llm_api_mode")
    api_mode = _normalize_api_mode(raw_api_mode or "chat_completions")
    parameters = _parameters(normalized)

    display_name = str(provider_config.get("name") or provider_name).strip()
    profiles = [
        ImportedModelProfile(
            role="primary",
            name=model_id,
            slug=_slug(model_id, "model"),
            model_id=model_id,
        )
    ]
    if review_model and review_model != model_id:
        profiles.append(
            ImportedModelProfile(
                role="review",
                name=f"{review_model} Review",
                slug=_slug(f"{review_model}-review", "review-model"),
                model_id=review_model,
            )
        )

    return ModelImportPreview(
        source_format=source_format,
        connection_name=display_name,
        connection_slug=_slug(f"{display_name}-connection", "model-connection"),
        provider=provider,
        base_url=base_url,
        api_mode=api_mode,
        parameters=parameters,
        profiles=tuple(profiles),
        ignored_fields=tuple(sorted(_ignored_fields(normalized, provider_name))),
    )


def _parse_document(text: str) -> tuple[dict[str, Any], str]:
    # 先识别结构特征，避免把合法 ENV 当成宽松 TOML 后丢失来源信息。
    if text.lstrip().startswith("{"):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON model configuration: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError("JSON model configuration must be an object")
        return value, "json"

    significant = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", ";"))
    ]
    if significant and all(
        "=" in line
        and _ENV_KEY.fullmatch(line.removeprefix("export ").split("=", 1)[0].strip())
        for line in significant
    ):
        return _parse_env(significant), "env"
    try:
        value = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML model configuration: {exc}") from exc
    return value, "toml"


def _parse_env(lines: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for line in lines:
        key, raw_value = line.removeprefix("export ").split("=", 1)
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        output[key.strip()] = value
    return output


def _normalize_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key).strip().lower(): _normalize_mapping(item)
        if isinstance(item, dict)
        else item
        for key, item in value.items()
    }


def _mapping_value(mapping: dict[str, Any], key: str) -> Any:
    target = key.strip().lower()
    return next((value for name, value in mapping.items() if name.lower() == target), None)


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def _normalize_api_mode(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "chat": "chat_completions",
        "chat_completion": "chat_completions",
        "chat_completions": "chat_completions",
        "response": "responses",
        "responses": "responses",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported model API mode: {value}")
    return aliases[normalized]


def _parameters(data: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "model_reasoning_effort": "reasoning_effort",
        "llm_reasoning_effort": "reasoning_effort",
        "reasoning_effort": "reasoning_effort",
        "max_output_tokens": "max_output_tokens",
        "llm_max_output_tokens": "max_output_tokens",
        "max_retries": "max_retries",
        "llm_max_retries": "max_retries",
        "timeout_seconds": "timeout_seconds",
        "llm_timeout_seconds": "timeout_seconds",
        "temperature": "temperature",
        "top_p": "top_p",
    }
    output: dict[str, Any] = {"store": False}
    nested = data.get("parameters") or data.get("model_parameters")
    if nested is not None:
        if not isinstance(nested, dict):
            raise ValueError("Model parameters must be an object")
        output.update({str(key): _scalar(value) for key, value in nested.items()})
    for source, target in aliases.items():
        if source in data and data[source] not in (None, ""):
            output[target] = _scalar(data[source])
    disabled = _first(
        data,
        "disable_response_storage",
        "llm_disable_response_storage",
        "response_storage_disabled",
    )
    if disabled is not None:
        output["response_storage_disabled"] = _boolean(disabled)
    else:
        output["response_storage_disabled"] = True
    return output


def _scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if re.fullmatch(r"-?\d+", lowered):
        return int(lowered)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", lowered):
        return float(lowered)
    return value.strip()


def _boolean(value: Any) -> bool:
    parsed = _scalar(value)
    if isinstance(parsed, bool):
        return parsed
    if parsed in (1, 0):
        return bool(parsed)
    raise ValueError(f"Expected a boolean model configuration value, got: {value}")


def _unwrap_markdown_url(value: str) -> str:
    match = _MARKDOWN_LINK.fullmatch(value)
    return match.group(2).strip() if match else value


def _slug(value: str, fallback: str) -> str:
    slug = _SLUG_PART.sub("-", value.lower()).strip("-")[:64].rstrip("-")
    if len(slug) < 3:
        slug = fallback
    return slug


def _ignored_fields(data: dict[str, Any], provider_name: str) -> set[str]:
    # 未消费字段只返回字段名，API Key 等粘贴值绝不进入预览响应。
    consumed = {
        "model_provider",
        "provider",
        "model",
        "model_id",
        "llm_model",
        "review_model",
        "review_model_id",
        "llm_review_model",
        "base_url",
        "openai_base_url",
        "wire_api",
        "api_mode",
        "llm_api_mode",
        "model_reasoning_effort",
        "llm_reasoning_effort",
        "reasoning_effort",
        "disable_response_storage",
        "llm_disable_response_storage",
        "response_storage_disabled",
        "max_output_tokens",
        "llm_max_output_tokens",
        "max_retries",
        "llm_max_retries",
        "timeout_seconds",
        "llm_timeout_seconds",
        "temperature",
        "top_p",
        "parameters",
        "model_parameters",
    }
    ignored = {key for key in data if key not in consumed | {"model_providers"}}
    providers = data.get("model_providers")
    if isinstance(providers, dict):
        selected = _mapping_value(providers, provider_name)
        if isinstance(selected, dict):
            for key in selected:
                if key not in {"name", "base_url", "wire_api", "api_mode", "requires_openai_auth"}:
                    ignored.add(f"model_providers.{provider_name}.{key}")
        for key in providers:
            if key.lower() != provider_name.lower():
                ignored.add(f"model_providers.{key}")
    expanded: set[str] = set()
    for key in ignored:
        value = data.get(key)
        if isinstance(value, dict):
            expanded.update(f"{key}.{child}" for child in value)
        else:
            expanded.add(key)
    return expanded
