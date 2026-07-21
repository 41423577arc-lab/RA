import hashlib
import json
import time
from pathlib import Path
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from app.config import Settings


OutputT = TypeVar("OutputT", bound=BaseModel)
REVIEW_NODES = {"web_verify", "project_rerank", "association"}
LONG_NODES = {"association", "report_content"}


class LLMUnavailable(RuntimeError):
    pass


class LLMCallFailed(RuntimeError):
    pass


class StructuredLLM:
    def __init__(self, settings: Settings, repository=None):
        self.settings = settings
        self.repository = repository
        self.client = None
        if self.enabled:
            self.client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=settings.llm_timeout_seconds,
                max_retries=0,
            )

    @property
    def enabled(self) -> bool:
        return bool(
            self.settings.llm_enabled
            and self.settings.openai_api_key
            and self.settings.llm_disable_response_storage
        )

    def parse(
        self,
        task_id: str,
        node_name: str,
        input_payload: dict,
        output_model: type[OutputT],
    ) -> OutputT:
        if not self.enabled or self.client is None:
            raise LLMUnavailable("大模型未启用、密钥为空或响应存储未关闭")

        model = (
            self.settings.llm_review_model
            if node_name in REVIEW_NODES
            else self.settings.llm_model
        )
        prompt_path = Path(self.settings.prompt_dir) / f"{node_name}_v1.txt"
        system_prompt = prompt_path.read_text(encoding="utf-8")
        safety_identifier = hashlib.sha256(
            f"{task_id}{self.settings.llm_safety_salt}".encode("utf-8")
        ).hexdigest()
        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                response = self.client.responses.parse(
                    model=model,
                    reasoning={"effort": self.settings.llm_reasoning_effort},
                    instructions=system_prompt,
                    input=json.dumps(
                        {"UNTRUSTED_DATA": input_payload}, ensure_ascii=False, default=str
                    ),
                    text_format=output_model,
                    max_output_tokens=16000 if node_name in LONG_NODES else 8000,
                    store=False,
                    safety_identifier=safety_identifier,
                    timeout=self.settings.llm_timeout_seconds,
                )
                parsed = response.output_parsed
                if parsed is None:
                    raise ValueError("模型未返回可解析的结构化结果")
                usage = getattr(response, "usage", None)
                self._log(
                    task_id,
                    node_name,
                    model,
                    "SUCCESS",
                    started,
                    response_id=getattr(response, "id", None),
                    input_tokens=getattr(usage, "input_tokens", None),
                    output_tokens=getattr(usage, "output_tokens", None),
                )
                return parsed
            except Exception as exc:
                last_error = exc
                if attempt < self.settings.llm_max_retries:
                    time.sleep(2)

        self._log(
            task_id,
            node_name,
            model,
            "DEGRADED",
            started,
            error_type=type(last_error).__name__ if last_error else "UnknownError",
            error_message=str(last_error)[:1000] if last_error else "未知错误",
        )
        raise LLMCallFailed(f"{node_name} 调用失败: {last_error}") from last_error

    def _log(
        self,
        task_id: str,
        node_name: str,
        model: str,
        status: str,
        started: float,
        **extra,
    ) -> None:
        logger = getattr(self.repository, "log_llm_call", None)
        if logger is None:
            return
        logger(
            task_id,
            node_name=node_name,
            model=model,
            status=status,
            prompt_version="v1",
            latency_ms=int((time.perf_counter() - started) * 1000),
            **extra,
        )
