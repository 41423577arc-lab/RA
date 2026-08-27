import hashlib
import json
from pathlib import Path

from app.config import Settings
from app.services.agent_config.registry import NODE_REGISTRY


CONFIG_SCHEMA_VERSION = 1
REVIEW_NODES = {
    "evidence_verify",
    "intake_identity_normalize",
    "intake_readiness",
}
LONG_NODES = {"final_synthesis"}


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_legacy_behavior_config(settings: Settings) -> dict:
    prompt_dir = Path(settings.prompt_dir)
    nodes = {
        node_key: _legacy_node_config(settings, prompt_dir, node_key)
        for node_key in NODE_REGISTRY
    }
    mcp_server = {
        "transport": "streamable_http",
        "url": settings.mcp_server_url,
        "authentication": {"type": "none", "secret_ref": None},
    }
    mcp_revision_id = f"legacy-mcp:{canonical_hash(mcp_server)}"
    tool_mappings = [
        {
            "logical_tool_key": "identity.find_candidates",
            "mapping_revision_id": "legacy-mapping:identity.find_candidates:v1",
            "provider": "mcp",
            "server_revision_id": mcp_revision_id,
            "remote_tool_name": "find_entity_candidates",
            "allowed_nodes": ["intake_agent", "intake_identity_update"],
            "timeout_seconds": 10,
            "input_mapping": {},
            "output_mapping": {},
        },
        {
            "logical_tool_key": "projects.search",
            "mapping_revision_id": "legacy-mapping:projects.search:v1",
            "provider": "mcp",
            "server_revision_id": mcp_revision_id,
            "remote_tool_name": "search_projects",
            "allowed_nodes": ["research_pipeline"],
            "timeout_seconds": 10,
            "input_mapping": {},
            "output_mapping": {},
        },
        {
            "logical_tool_key": "identity.search_public",
            "mapping_revision_id": "legacy-mapping:identity.search_public:v1",
            "provider": "tavily",
            "secret_ref": "env:TAVILY_API_KEY",
            "allowed_nodes": ["intake_agent", "intake_identity_normalize"],
        },
        {
            "logical_tool_key": "public.search",
            "mapping_revision_id": "legacy-mapping:public.search:v1",
            "provider": "tavily",
            "secret_ref": "env:TAVILY_API_KEY",
            "allowed_nodes": ["research_pipeline"],
        },
        {
            "logical_tool_key": "public.extract_pages",
            "mapping_revision_id": "legacy-mapping:public.extract_pages:v1",
            "provider": "tavily",
            "secret_ref": "env:TAVILY_API_KEY",
            "allowed_nodes": ["research_pipeline"],
        },
    ]
    return {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "management": {"source": "legacy_bootstrap"},
        "nodes": nodes,
        "mcp_server_revisions": [
            {"revision_id": mcp_revision_id, **mcp_server}
        ],
        "tool_mappings": tool_mappings,
        "loop": {
            "max_loops": settings.agent_max_loops,
            "max_tool_calls": settings.agent_max_tool_calls,
            "max_repeated_actions": settings.agent_max_repeated_actions,
            "identity_auto_accept_threshold": settings.llm_web_identity_threshold,
            "intake_agent_v2_enabled": settings.intake_agent_v2_enabled,
            "intake_entity_resolution_enabled": settings.intake_entity_resolution_enabled,
            "intake_react_enabled": settings.intake_react_enabled,
        },
        "output": {
            "formats": ["detailed_markdown", "action_brief_markdown"],
            "templates": [
                _template_config("report", Path(settings.report_template)),
                _template_config("detailed_report", Path(settings.detailed_report_template)),
                _template_config("action_brief", Path(settings.action_brief_template)),
            ],
            "evidence_validation_required": True,
        },
    }


def resolved_snapshot(
    behavior_config: dict,
    *,
    agent_definition_id: str,
    agent_version_id: str,
) -> dict:
    return {
        "config_schema_version": behavior_config["config_schema_version"],
        "agent_definition_id": agent_definition_id,
        "agent_version_id": agent_version_id,
        "nodes": behavior_config["nodes"],
        "mcp_server_revisions": behavior_config["mcp_server_revisions"],
        "tool_mappings": behavior_config["tool_mappings"],
        "loop": behavior_config["loop"],
        "output": behavior_config["output"],
    }


def _legacy_node_config(settings: Settings, prompt_dir: Path, node_key: str) -> dict:
    prompt_path = prompt_dir / f"{node_key}_v1.txt"
    content = prompt_path.read_text(encoding="utf-8")
    prompt = {
        "revision_id": f"legacy-prompt:{node_key}:{canonical_hash(content)}",
        "content_hash": canonical_hash(content),
        "content": content,
        "source": f"backend/prompts/{prompt_path.name}",
        "skills": [],
    }
    if node_key == "intake_agent":
        prompt["skills"] = [
            _prompt_asset(path) for path in sorted((prompt_dir / "intake_skills").glob("*.txt"))
        ]
    return {
        "model": {
            "provider": settings.model_provider,
            "base_url": settings.openai_base_url,
            "secret_ref": "env:OPENAI_API_KEY",
            "safety_identifier_salt_ref": "env:LLM_SAFETY_SALT",
            "model_id": settings.llm_review_model
            if node_key in REVIEW_NODES
            else settings.llm_model,
            "api_mode": settings.llm_api_mode,
            "reasoning_effort": settings.llm_reasoning_effort,
            "timeout_seconds": settings.llm_timeout_seconds,
            "max_retries": settings.llm_max_retries,
            "max_output_tokens": 16000 if node_key in LONG_NODES else 8000,
            "store": False,
            "enabled": settings.llm_enabled,
            "response_storage_disabled": settings.llm_disable_response_storage,
        },
        "prompt": prompt,
        "allowed_tools": _allowed_tools(node_key),
    }


def _allowed_tools(node_key: str) -> list[str]:
    if node_key == "intake_agent":
        return ["identity.find_candidates", "identity.search_public"]
    if node_key == "intake_identity_update":
        return ["identity.find_candidates"]
    if node_key == "intake_identity_normalize":
        return ["identity.search_public"]
    return []


def _prompt_asset(path: Path) -> dict:
    content = path.read_text(encoding="utf-8")
    digest = canonical_hash(content)
    return {
        "revision_id": f"legacy-skill:{path.stem}:{digest}",
        "name": path.stem,
        "content_hash": digest,
        "content": content,
        "source": f"backend/prompts/intake_skills/{path.name}",
    }


def _template_config(name: str, path: Path) -> dict:
    content = path.read_text(encoding="utf-8")
    digest = canonical_hash(content)
    return {
        "name": name,
        "revision_id": f"legacy-template:{name}:{digest}",
        "content_hash": digest,
        "content": content,
        "source": f"backend/templates/{path.name}",
    }
