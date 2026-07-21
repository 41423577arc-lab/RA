from pathlib import Path

from datetime import date

from app.schemas.task import (
    ActionBrief,
    ConfirmedContext,
    ConfirmedEntity,
    EvidenceBackedItem,
    ExtractedInfo,
    GeneratedReportContent,
    Person,
    ProjectResult,
    PublicClaim,
)
from app.services.agent_nodes import validate_report_content
from app.services.report_renderer import ReportRenderer


ROOT = Path(__file__).resolve().parents[2]


def test_report_escapes_raw_html_from_external_content() -> None:
    renderer = ReportRenderer(ROOT / "backend/templates/report.md.j2")
    report = renderer.render(
        "测试输入",
        ExtractedInfo(event_type="会议", people=[Person(name="王传福")]),
        [
            PublicClaim(
                subject="王传福",
                claim="王传福参与新能源业务<script>alert(1)</script>",
                source_title="<b>来源</b>",
                source_url="https://example.com",
                matched_keywords=["新能源"],
            )
        ],
        [],
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
    )

    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert "<b>" not in report


def test_generated_report_enforces_section_meaning_and_business_labels() -> None:
    context = ConfirmedContext(
        intents=["MEETING_PREPARATION"],
        entities=[
            ConfirmedEntity(
                entity_type="PERSON",
                canonical_name="王传福",
                organization="比亚迪股份有限公司",
                confirmed_by="AUTO",
            )
        ],
        event_type="宴请",
        event_time="今晚",
        event_location="深圳",
        business_directions=["储能"],
        focus_questions=["园区储能管理平台目前的建设进展"],
    )
    project = ProjectResult(
        project_id="P001",
        project_name="比亚迪园区储能管理平台",
        customer_name="比亚迪股份有限公司",
        contact_name="王传福",
        status="ACTIVE",
        owner_name="张伟",
        start_date=date(2026, 1, 10),
        description="储能管理",
        match_type="PERSON_EXACT",
    )
    project_fact = EvidenceBackedItem(
        text="P001 状态为 ACTIVE，end_date 为空。",
        statement_type="FACT",
        evidence_refs=["PROJECT:P001"],
        confidence=0.99,
    )
    content = GeneratedReportContent(
        task_overview=[project_fact, project_fact],
        person_and_company_summary=[],
        public_information_summary=[project_fact],
        priority_projects=[project_fact, project_fact],
        resource_analysis=[],
        recommended_topics=[],
        advancement_advice=[],
        preparation_items=[],
        gaps_and_risks=[],
        action_brief=ActionBrief(
            destination="深圳",
            meeting_people=["王传福"],
            objective="确认 P001 的 ACTIVE 状态和 end_date",
            evidence_refs=["PROJECT:P001"],
        ),
    )

    validated = validate_report_content(content, [], [project], context)

    assert len(validated.task_overview) == 2
    assert validated.task_overview[0].text == "今晚在深圳与王传福进行宴请。"
    assert all("P001" not in item.text for item in validated.task_overview)
    assert validated.public_information_summary == []
    assert len(validated.priority_projects) == 1
    assert validated.priority_projects[0].text == "P001 状态为在建，尚未记录结束日期。"
    assert validated.action_brief.objective == "确认 P001 的在建状态和结束日期"

    renderer = ReportRenderer(
        ROOT / "backend/templates/report.md.j2",
        ROOT / "backend/templates/detailed_report.md.j2",
        ROOT / "backend/templates/action_brief.md.j2",
    )
    report, _ = renderer.render_generated(
        validated, [], [project], "SUCCESS", "SUCCESS", "SUCCESS"
    )

    assert "## 活动或任务概况" in report
    assert "今晚在深圳与王传福进行宴请" in report
    assert "P001`-" not in report
    assert "ACTIVE" not in report
    assert "end_date" not in report
    assert "状态：在建" in report
