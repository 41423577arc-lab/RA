from datetime import date
from pathlib import Path
from types import SimpleNamespace

from app.schemas.task import ProjectResult, SearchResult, WebPage
from app.services.extractor import RuleExtractor
from app.services.report_renderer import ReportRenderer
from app.tasks.pipeline import (
    ResearchPipeline,
    identity_claims_from_intake_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]


class FakeRepository:
    def __init__(self, task):
        self.task = task
        self.statuses: list[str] = []

    def get(self, task_id: str):
        return self.task if task_id == self.task.id else None

    def update(self, task_id: str, **values):
        assert task_id == self.task.id
        for key, value in values.items():
            setattr(self.task, key, value)
        if "status" in values:
            self.statuses.append(values["status"])
        return self.task


class NoopTranscriber:
    def transcribe(self, _):
        raise AssertionError("Text tasks must not call the transcriber")


class FakeTranscriber:
    def transcribe(self, path):
        assert path.read_bytes() == b"webm"
        return "老板周五要和比亚迪股份有限公司的王传福董事长兼总裁吃饭，主要聊新能源和储能项目。"


class FakeWeb:
    async def search(self, queries):
        assert queries[0].startswith("王传福 比亚迪股份有限公司")
        return [
            SearchResult(
                title="比亚迪公司简介",
                url="https://example.com/byd",
                content="摘要",
                query=queries[0],
                rank=0,
            )
        ]

    async def extract(self, _):
        return [
            WebPage(
                title="比亚迪公司简介",
                url="https://example.com/byd",
                raw_content="王传福表示比亚迪股份有限公司持续发展新能源汽车和储能业务。",
                rank=0,
            )
        ]


class FailedWeb:
    async def search(self, _):
        raise AssertionError("Pipeline must not search the public web")

    async def extract(self, _):
        raise AssertionError("Extract must be skipped after search failure")


class FakeProjects:
    async def search_projects(self, person_names, organization_names, keywords):
        assert person_names == ["王传福"]
        assert organization_names == ["比亚迪股份有限公司"]
        assert keywords == ["新能源", "储能"]
        return [
            ProjectResult(
                project_id="P001",
                project_name="比亚迪园区储能管理平台",
                customer_name="比亚迪股份有限公司",
                contact_name="王传福",
                status="ACTIVE",
                owner_name="张伟",
                start_date=date(2026, 1, 10),
                description="建设园区储能监控与能源管理平台",
                match_type="PERSON_EXACT",
            ),
            ProjectResult(
                project_id="P002",
                project_name="比亚迪新能源汽车供应链分析",
                customer_name="比亚迪股份有限公司",
                contact_name="王传福",
                status="COMPLETED",
                owner_name="刘芳",
                start_date=date(2024, 3, 1),
                end_date=date(2024, 11, 30),
                description="完成新能源汽车供应链分析",
                match_type="PERSON_EXACT",
            ),
        ]


class FailedProjects:
    async def search_projects(self, person_names, organization_names, keywords):
        raise RuntimeError("mcp unavailable")


def make_pipeline(repository, web):
    return ResearchPipeline(
        repository=repository,
        transcriber=NoopTranscriber(),
        extractor=RuleExtractor(ROOT / "seed"),
        web=web,
        projects=FakeProjects(),
        renderer=ReportRenderer(ROOT / "backend/templates/report.md.j2"),
    )


def test_pipeline_reuses_only_intake_identity_evidence() -> None:
    claims = identity_claims_from_intake_snapshot(
        {
            "structured_context": {
                "entity_resolutions": [
                    {
                        "entity_type": "PERSON",
                        "mention": "王总",
                        "canonical_name": "王传福",
                        "organization": "比亚迪股份有限公司",
                        "title": "董事长兼总裁",
                        "confidence": 0.9,
                        "confirmed_by": "EXTERNAL_AUTO",
                        "source_url": "https://example.com/identity",
                        "evidence_quote": "王传福任比亚迪股份有限公司董事长兼总裁。",
                    }
                ]
            }
        }
    )

    assert len(claims) == 1
    assert claims[0].claim == "王传福（比亚迪股份有限公司、董事长兼总裁）"
    assert claims[0].matched_keywords == []
    assert "业务" not in claims[0].claim


def test_full_text_pipeline_generates_report_and_all_states() -> None:
    task = SimpleNamespace(
        id="task-1",
        input_type="text",
        input_text="老板周五要和比亚迪股份有限公司的王传福董事长兼总裁吃饭，主要聊新能源和储能项目。",
        audio_path=None,
    )
    repository = FakeRepository(task)

    make_pipeline(repository, FailedWeb()).run(task.id)

    assert repository.statuses == [
        "EXTRACTING",
        "PROJECT_SEARCHING",
        "GENERATING",
        "COMPLETED",
    ]
    assert "比亚迪园区储能管理平台" in task.report_markdown
    assert "比亚迪新能源汽车供应链分析" in task.report_markdown
    assert task.web_search_status == "SKIPPED"
    assert task.internal_search_status == "SUCCESS"


def test_search_failure_is_partial_and_internal_projects_continue() -> None:
    task = SimpleNamespace(
        id="task-2",
        input_type="text",
        input_text="老板周五要和比亚迪股份有限公司的王传福董事长兼总裁吃饭，主要聊新能源和储能项目。",
        audio_path=None,
    )
    repository = FakeRepository(task)

    make_pipeline(repository, FailedWeb()).run(task.id)

    assert task.status == "COMPLETED"
    assert task.web_search_status == "SKIPPED"
    assert task.web_fetch_status == "SKIPPED"
    assert "本次没有可复用的联网身份来源" in task.report_markdown
    assert "比亚迪园区储能管理平台" in task.report_markdown


def test_audio_pipeline_transcribes_and_deletes_shared_file(tmp_path) -> None:
    audio_path = tmp_path / "task-audio.webm"
    audio_path.write_bytes(b"webm")
    task = SimpleNamespace(
        id="task-audio",
        input_type="audio",
        input_text=None,
        audio_path=str(audio_path),
    )
    repository = FakeRepository(task)
    pipeline = ResearchPipeline(
        repository=repository,
        transcriber=FakeTranscriber(),
        extractor=RuleExtractor(ROOT / "seed"),
        web=FailedWeb(),
        projects=FakeProjects(),
        renderer=ReportRenderer(ROOT / "backend/templates/report.md.j2"),
    )

    pipeline.run(task.id)

    assert repository.statuses[0] == "TRANSCRIBING"
    assert task.status == "COMPLETED"
    assert task.input_text.startswith("老板周五")
    assert not audio_path.exists()


def test_mcp_failure_is_partial_and_public_information_continues() -> None:
    task = SimpleNamespace(
        id="task-mcp-failed",
        input_type="text",
        input_text="老板周五要和比亚迪股份有限公司的王传福董事长兼总裁吃饭，主要聊新能源和储能项目。",
        audio_path=None,
    )
    repository = FakeRepository(task)
    pipeline = ResearchPipeline(
        repository=repository,
        transcriber=NoopTranscriber(),
        extractor=RuleExtractor(ROOT / "seed"),
        web=FailedWeb(),
        projects=FailedProjects(),
        renderer=ReportRenderer(ROOT / "backend/templates/report.md.j2"),
    )

    pipeline.run(task.id)

    assert task.status == "COMPLETED"
    assert task.internal_search_status == "FAILED"
    assert "公司内部项目信息检索失败" in task.report_markdown
    assert task.web_search_status == "SKIPPED"
