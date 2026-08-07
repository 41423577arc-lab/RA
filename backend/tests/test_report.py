from pathlib import Path

from datetime import date

from app.schemas.task import (
    ActionBrief,
    AssociationAnalysis,
    ConfirmedContext,
    ConfirmedEntity,
    EvidenceBackedItem,
    ExtractedInfo,
    GeneratedReportContent,
    Person,
    ProjectResult,
    PublicClaim,
)
from app.services.agent_nodes import (
    build_person_identity_summaries,
    complete_report_content,
    fallback_report_content,
    validate_report_content,
)
from app.services.report_renderer import ReportRenderer


ROOT = Path(__file__).resolve().parents[2]


def _project_items(count: int) -> list[EvidenceBackedItem]:
    return [
        EvidenceBackedItem(
            text=f"Project {index}",
            statement_type="FACT",
            evidence_refs=[f"PROJECT:P{index:03d}"],
            confidence=1,
        )
        for index in range(count)
    ]


def test_fallback_report_content_caps_priority_projects() -> None:
    items = _project_items(6)
    context = ConfirmedContext(
        intents=["REPORT_GENERATION"],
        entities=[],
        event_type="其他",
    )

    content = fallback_report_content(
        "input",
        context,
        AssociationAnalysis(related_projects=items),
        [],
        [],
    )

    assert content.priority_projects == items[:3]


def test_complete_report_content_caps_merged_priority_projects() -> None:
    items = _project_items(6)
    primary = GeneratedReportContent(
        priority_projects=items[:3],
        action_brief=ActionBrief(objective="Primary"),
    )
    fallback = GeneratedReportContent(
        priority_projects=items[3:],
        action_brief=ActionBrief(objective="Fallback"),
    )

    completed = complete_report_content(primary, fallback)

    assert completed.priority_projects == items[:3]


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
    identity_fact = EvidenceBackedItem(
        text="王传福现任比亚迪股份有限公司董事长兼总裁，负责公司经营管理。",
        statement_type="FACT",
        evidence_refs=["CONFIRMATION:1"],
        confidence=1,
    )
    content = GeneratedReportContent(
        task_overview=[project_fact, project_fact],
        person_and_company_summary=[
            EvidenceBackedItem(
                text="王传福近期参加公司会议。",
                statement_type="FACT",
                evidence_refs=["CONFIRMATION:1"],
                confidence=1,
            ),
            identity_fact,
            EvidenceBackedItem(
                text="比亚迪股份有限公司管理层还包括其他成员。",
                statement_type="FACT",
                evidence_refs=["CONFIRMATION:1"],
                confidence=1,
            ),
        ],
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
    assert validated.person_and_company_summary == [identity_fact]
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

    assert "## 会面概况" in report
    assert "## 关键人及企业" in report
    assert "## 公开信息" in report
    assert "## 重点项目与可用资源" in report
    assert "## 会谈行动建议" in report
    assert "### 建议讨论" in report
    assert "### 推动动作" in report
    assert "### 会前准备" in report
    assert "## 风险与信息缺口" in report
    assert "今晚在深圳与王传福进行宴请" in report
    assert "| 类型 | 内容 | 来源 |" in report
    assert "| 重点项目 | P001 状态为在建" in report
    assert "内部项目 `P001`" in report
    assert "P001`-" not in report
    assert "ACTIVE" not in report
    assert "end_date" not in report
    assert "P001 状态为在建" in report


def test_generated_report_numbers_and_deduplicates_web_sources() -> None:
    context = ConfirmedContext(
        intents=["MEETING_PREPARATION"],
        entities=[
            ConfirmedEntity(
                entity_type="PERSON",
                canonical_name="范玉峰",
                organization="中建二局安装公司",
                title="党委书记、董事长",
                confirmed_by="AUTO",
            )
        ],
        event_type="会议",
    )
    claims = [
        PublicClaim(
            web_result_id="W001",
            evidence_id="E001",
            subject="范玉峰",
            claim="范玉峰现任中建二局安装公司党委书记、董事长。",
            source_title="公司领导",
            source_url="https://example.com/leader",
        ),
        PublicClaim(
            web_result_id="W001",
            evidence_id="E002",
            subject="范玉峰",
            claim="范玉峰负责生产经营。",
            source_title="公司领导",
            source_url="https://example.com/leader",
        ),
        PublicClaim(
            web_result_id="W002",
            evidence_id="E001",
            subject="范玉峰",
            claim="范玉峰负责对外商务拓展。",
            source_title="公司要闻",
            source_url="https://example.com/news",
        ),
    ]
    item = EvidenceBackedItem(
        text="范玉峰现任中建二局安装公司党委书记、董事长。",
        statement_type="FACT",
        evidence_refs=["WEB:W001:E001", "WEB:W001:E002", "WEB:W002:E001"],
        confidence=1,
    )
    content = GeneratedReportContent(
        task_overview=[],
        person_and_company_summary=[item],
        public_information_summary=[],
        priority_projects=[],
        resource_analysis=[],
        recommended_topics=[],
        advancement_advice=[],
        preparation_items=[],
        gaps_and_risks=[],
        action_brief=ActionBrief(objective="确认会面事项"),
    )
    renderer = ReportRenderer(
        ROOT / "backend/templates/report.md.j2",
        ROOT / "backend/templates/detailed_report.md.j2",
        ROOT / "backend/templates/action_brief.md.j2",
    )

    report, _ = renderer.render_generated(
        content, claims, [], "SUCCESS", "SUCCESS", "SUCCESS"
    )

    assert "[¹](https://example.com/leader)" in report
    assert "[²](https://example.com/news)" in report
    assert "[来源1](https://example.com/leader)" in report
    assert "[来源2](https://example.com/news)" in report
    assert "来源3" not in report


def test_person_identity_summary_rejects_activity_only_candidate() -> None:
    context = ConfirmedContext(
        intents=["PERSON_BACKGROUND_RESEARCH"],
        entities=[
            ConfirmedEntity(
                entity_type="PERSON",
                canonical_name="范玉峰",
                organization="中建二局安装公司",
                title="党委书记、董事长",
                confirmed_by="AUTO",
            )
        ],
        event_type="会议",
    )
    activity = EvidenceBackedItem(
        text="范玉峰近期参加公司会议和节前检查。",
        statement_type="FACT",
        evidence_refs=["WEB:W001:E001"],
        confidence=1,
    )
    claims = [
        PublicClaim(
            web_result_id="W001",
            evidence_id="E001",
            subject="范玉峰",
            claim=activity.text,
            source_title="公司要闻",
            source_url="https://example.com/news",
        )
    ]

    summaries = build_person_identity_summaries(context, claims, [activity])

    assert len(summaries) == 1
    assert summaries[0].text == "范玉峰现任中建二局安装公司党委书记、董事长。"
    assert summaries[0].evidence_refs == ["CONFIRMATION:1"]
