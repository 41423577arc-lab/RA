from types import SimpleNamespace

from app.services.reporting.analysis_chat import (
    build_task_chat_context,
    fallback_task_chat_reply,
)


def test_analysis_chat_context_uses_verified_claims_and_internal_results() -> None:
    task = SimpleNamespace(
        status="VERIFYING_WEB_RESULTS",
        confirmed_context={"entities": [{"canonical_name": "王伟"}]},
        web_search_status="SUCCESS",
        web_fetch_status="SUCCESS",
        public_claims=[
            {
                "claim": "王伟负责示例项目",
                "source_title": "企业官网",
                "source_url": "https://example.com/profile",
            }
        ],
        internal_search_status="SUCCESS",
        internal_results=[
            {
                "project_id": "P001",
                "project_name": "示例内部项目",
                "status": "ACTIVE",
            }
        ],
        ranked_internal_results=[{"project_id": "P001", "relevance_score": 95}],
        association_analysis=None,
        detailed_report_markdown=None,
        report_markdown=None,
        degraded_nodes=[],
        error_message=None,
        web_results=[{"title": "未经核验的搜索结果"}],
        web_pages=[{"raw_content": "未经核验的网页正文"}],
    )

    context = build_task_chat_context(task)

    assert context["progress_summary"] == "正在核验网页证据"
    assert context["web"]["verified_claim_count"] == 1
    assert context["web"]["verified_claims"][0]["source_title"] == "企业官网"
    assert context["internal_projects"]["result_count"] == 1
    assert context["internal_projects"]["results"][0]["project_name"] == "示例内部项目"
    assert "web_results" not in context
    assert "web_pages" not in context


def test_analysis_chat_fallback_reports_current_counts() -> None:
    context = {
        "progress_summary": "正在查询内部项目",
        "web": {"verified_claim_count": 3},
        "internal_projects": {"result_count": 2},
    }

    reply = fallback_task_chat_reply(context)

    assert "正在查询内部项目" in reply
    assert "3 条公开信息" in reply
    assert "2 个内部项目" in reply
