from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tavily_api_key: str = ""
    database_url: str = "sqlite:///./resource_agent.db"
    database_readonly_url: str = "postgresql+psycopg://resource_reader:resource_reader@localhost:5432/resource_agent"
    redis_url: str = "redis://localhost:6379/0"
    mcp_server_url: str = "http://localhost:8001/mcp"
    audio_dir: Path = Path("./shared/audio")
    whisper_model_path: str = "small"
    vector_similarity_threshold: float = 0.18
    seed_dir: Path = Path("./seed")
    report_template: Path = Path("./backend/templates/report.md.j2")
    celery_task_always_eager: bool = False
    model_provider: str = "OpenAI"
    openai_api_key: str = ""
    openai_base_url: str = "https://vftllmapi.vf-tech.cn/v1"
    llm_enabled: bool = True
    llm_model: str = "MiniMax-M3"
    llm_review_model: str = "MiniMax-M3"
    llm_api_mode: Literal["chat_completions", "responses"] = "chat_completions"
    llm_reasoning_effort: str = "xhigh"
    llm_timeout_seconds: float = 120
    llm_max_retries: int = 1
    llm_disable_response_storage: bool = True
    llm_web_identity_threshold: float = 0.80
    llm_project_confidence_threshold: float = 0.60
    llm_analysis_confidence_threshold: float = 0.60
    llm_safety_salt: str = "resource-agent-demo"
    prompt_dir: Path = Path("./backend/prompts")
    detailed_report_template: Path = Path("./backend/templates/detailed_report.md.j2")
    action_brief_template: Path = Path("./backend/templates/action_brief.md.j2")


settings = Settings()
