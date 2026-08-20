from typing import TYPE_CHECKING

from app.schemas.task import TaskChatResult

if TYPE_CHECKING:
    from app.services.integrations.llm_client import StructuredLLM


STATUS_SUMMARIES = {
    "PENDING": "任务等待执行",
    "TRANSCRIBING": "正在转写语音",
    "CONTEXT_EXTRACTING": "正在整理输入上下文",
    "PLANNING_WEB_SEARCH": "正在规划公开网络检索",
    "WEB_SEARCHING": "正在检索公开网络信息",
    "WEB_FETCHING": "正在抓取候选网页正文",
    "VERIFYING_WEB_RESULTS": "正在核验网页证据",
    "PLANNING_PROJECT_SEARCH": "正在规划内部项目查询",
    "PROJECT_SEARCHING": "正在查询内部项目",
    "RERANKING_PROJECTS": "正在排序内部项目",
    "ANALYZING_ASSOCIATIONS": "正在综合公开信息和内部项目",
    "GENERATING_REPORT_CONTENT": "正在生成报告内容",
    "RENDERING_REPORT": "正在渲染报告",
    "COMPLETED": "分析已经完成",
    "FAILED": "分析执行失败",
    "CANCELLED": "分析已由用户停止",
}


class AnalysisChatAgent:
    def __init__(self, llm: "StructuredLLM"):
        self.llm = llm

    def respond(
        self,
        task_id: str,
        message: str,
        history: list[dict],
        task_context: dict,
    ) -> TaskChatResult:
        return self.llm.parse(
            task_id,
            "analysis_chat",
            {
                "message": message,
                "history": history[-20:],
                "task_context": task_context,
            },
            TaskChatResult,
        )


def build_task_chat_context(task) -> dict:
    public_claims = list(task.public_claims or [])[:20]
    internal_projects = list(task.internal_results or [])[:20]
    return {
        "task_status": task.status,
        "progress_summary": STATUS_SUMMARIES.get(task.status, f"当前阶段：{task.status}"),
        "confirmed_context": task.confirmed_context,
        "web": {
            "search_status": task.web_search_status,
            "fetch_status": task.web_fetch_status,
            "verified_claim_count": len(task.public_claims or []),
            "verified_claims": public_claims,
        },
        "internal_projects": {
            "search_status": task.internal_search_status,
            "result_count": len(task.internal_results or []),
            "results": internal_projects,
            "rankings": list(task.ranked_internal_results or [])[:20],
        },
        "association_analysis": task.association_analysis,
        "report_available": bool(task.detailed_report_markdown or task.report_markdown),
        "degraded_nodes": list(task.degraded_nodes or []),
        "error_message": task.error_message,
    }


def fallback_task_chat_reply(task_context: dict) -> str:
    web = task_context["web"]
    projects = task_context["internal_projects"]
    return (
        f"{task_context['progress_summary']}。"
        f"当前已核验 {web['verified_claim_count']} 条公开信息，"
        f"已找到 {projects['result_count']} 个内部项目。"
        "详细问答暂时不可用，请稍后重试。"
    )
