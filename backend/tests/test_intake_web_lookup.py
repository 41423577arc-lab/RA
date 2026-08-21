from datetime import date

import pytest

import app.services.integrations.tavily_client as tavily_module
from app.schemas.intake import (
    ExternalIdentityNormalizationResult,
    IntakeStructuredContext,
)
from app.schemas.task import ConfirmationItem, ConfirmationRequest, SearchResult, WebPage
from app.services.intake.entity_candidates import IntakeEntityCandidateService
from app.services.integrations.tavily_client import TavilyClient


class NoInternalCandidates:
    async def find_entity_candidates(self, *_):
        return []


@pytest.mark.parametrize(
    ("entity_type", "mention", "canonical_name"),
    [
        ("ORGANIZATION", "中建二局", "中建二局安装工程有限公司"),
        ("PERSON", "王总", "王志远"),
    ],
)
def test_partial_internal_candidate_always_requires_user_confirmation(
    entity_type, mention, canonical_name
) -> None:
    option = IntakeEntityCandidateService._internal_option(
        {
            "candidate_id": "internal:test:001",
            "entity_type": entity_type,
            "canonical_name": canonical_name,
            "match_type": "PARTIAL",
        }
    )
    confirmation = ConfirmationRequest(
        version=1,
        items=[
            ConfirmationItem(
                mention=mention,
                entity_type=entity_type,
                candidates=[option],
            )
        ],
    )

    resolutions, pending = IntakeEntityCandidateService.apply_automatic_candidates(
        [], confirmation, threshold=0
    )

    assert option.confidence == 0.8
    assert resolutions == []
    assert pending is not None
    assert pending.items[0].candidates == [option]


def test_exact_internal_candidate_can_still_be_automatically_confirmed() -> None:
    option = IntakeEntityCandidateService._internal_option(
        {
            "candidate_id": "internal:customer:C001",
            "entity_type": "ORGANIZATION",
            "canonical_name": "比亚迪股份有限公司",
            "match_type": "EXACT",
        }
    )
    confirmation = ConfirmationRequest(
        version=1,
        items=[
            ConfirmationItem(
                mention="比亚迪股份有限公司",
                entity_type="ORGANIZATION",
                candidates=[option],
            )
        ],
    )

    resolutions, pending = IntakeEntityCandidateService.apply_automatic_candidates(
        [], confirmation
    )

    assert option.confidence == 1.0
    assert pending is None
    assert resolutions[0]["canonical_name"] == "比亚迪股份有限公司"
    assert resolutions[0]["confirmed_by"] == "INTERNAL"


class FanYufengIdentityWeb:
    def __init__(self):
        self.queries: list[str] = []
        self.window: tuple[str, str] | None = None

    async def search_identity(self, queries, *, start_date, end_date):
        self.queries = queries
        self.window = (start_date, end_date)
        return [
            SearchResult(
                web_result_id="W001",
                title="公司领导",
                url="https://2bmep.cscec.com/gygs/gsld",
                content=(
                    "姓名：范玉峰  职位：中建二局安装公司党委书记、董事长 "
                    "姓名：黄巢  职位：公司总经理、党委副书记"
                ),
                query=queries[0],
                rank=0,
            )
        ]

    async def extract_identity(self, results):
        result = results[0]
        return [
            WebPage(
                web_result_id=result.web_result_id,
                title=result.title,
                url=result.url,
                raw_content=result.content,
                rank=result.rank,
            )
        ]


def test_identity_lookup_uses_company_alias_and_recent_time_window() -> None:
    web = FanYufengIdentityWeb()
    service = IntakeEntityCandidateService(
        NoInternalCandidates(),
        web,
        today_provider=lambda: date(2026, 7, 28),
    )
    context = IntakeStructuredContext(
        people=["范玉峰"], organizations=["中建二局安装工程有限公司"]
    )
    confirmation = ConfirmationRequest(
        version=1,
        items=[
            ConfirmationItem(
                mention="范玉峰", entity_type="PERSON", candidates=[]
            )
        ],
    )

    result = service.search_key_person_identity_web(
        context,
        confirmation,
        lambda *_: ExternalIdentityNormalizationResult(candidates=[]),
    )

    assert web.queries == [
        '"范玉峰" "中建二局安装工程有限公司"',
        '"范玉峰" "中建二局安装公司" 职务 现任',
    ]
    assert web.window == ("2025-01-01", "2026-07-28")
    candidate = result.items[0].candidates[0]
    assert candidate.canonical_name == "范玉峰"
    assert candidate.organization == "中建二局安装工程有限公司"
    assert candidate.title == "党委书记、董事长"
    assert candidate.source_url == "https://2bmep.cscec.com/gygs/gsld"
    assert "身份检索截止 2026-07-28" in candidate.reason
    assert candidate.evidence_quote in (
        "姓名：范玉峰  职位：中建二局安装公司党委书记、董事长 "
        "姓名：黄巢  职位：公司总经理、党委副书记"
    )


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class CapturingHttpClient:
    search_payloads: list[dict] = []

    def __init__(self, *_, **__):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def post(self, url, json, headers):
        assert headers["Authorization"] == "Bearer test-key"
        if url.endswith("/search"):
            self.search_payloads.append(json)
            index = len(self.search_payloads)
            return FakeResponse(
                {
                    "results": [
                        {
                            "title": "公司领导",
                            "url": f"https://example.com/leader-{index}",
                            "content": "范玉峰，党委书记、董事长",
                        }
                    ]
                }
            )
        return FakeResponse({"results": []})


@pytest.mark.asyncio
async def test_tavily_identity_search_uses_advanced_china_filters(
    monkeypatch,
) -> None:
    CapturingHttpClient.search_payloads = []
    monkeypatch.setattr(tavily_module.httpx, "AsyncClient", CapturingHttpClient)
    client = TavilyClient("test-key")

    results = await client.search_identity(
        ["query one", "query two"],
        start_date="2025-01-01",
        end_date="2026-07-28",
    )

    assert len(results) == 2
    assert [payload["query"] for payload in CapturingHttpClient.search_payloads] == [
        "query one",
        "query two",
    ]
    assert all(
        payload
        == {
            "query": payload["query"],
            "search_depth": "advanced",
            "max_results": 10,
            "country": "china",
            "start_date": "2025-01-01",
            "end_date": "2026-07-28",
        }
        for payload in CapturingHttpClient.search_payloads
    )


@pytest.mark.asyncio
async def test_identity_extract_keeps_search_snippet_when_extract_is_empty(
    monkeypatch,
) -> None:
    monkeypatch.setattr(tavily_module.httpx, "AsyncClient", CapturingHttpClient)
    client = TavilyClient("test-key")
    result = SearchResult(
        web_result_id="W001",
        title="公司领导",
        url="https://example.com/leader",
        content="姓名：范玉峰 职位：中建二局安装公司党委书记、董事长",
        query="范玉峰",
        rank=0,
    )

    pages = await client.extract_identity([result])

    assert pages[0].title == "公司领导"
    assert pages[0].raw_content == result.content


def test_dated_role_evidence_records_source_month_and_lookup_cutoff() -> None:
    web = FanYufengIdentityWeb()

    async def dated_extract(results):
        return [
            WebPage(
                web_result_id="W001",
                title="公司会议",
                url=(
                    "https://2bmep.cscec.com/xwzx_new_41667/"
                    "gsyw/202504/3868784.html"
                ),
                raw_content=(
                    "中建二局安装公司党委书记、董事长范玉峰出席会议并讲话。"
                ),
                rank=0,
            )
        ]

    web.extract_identity = dated_extract
    service = IntakeEntityCandidateService(
        NoInternalCandidates(),
        web,
        today_provider=lambda: date(2026, 7, 28),
    )
    confirmation = ConfirmationRequest(
        version=1,
        items=[
            ConfirmationItem(
                mention="范玉峰", entity_type="PERSON", candidates=[]
            )
        ],
    )

    result = service.search_key_person_identity_web(
        IntakeStructuredContext(
            people=["范玉峰"], organizations=["中建二局安装工程有限公司"]
        ),
        confirmation,
        lambda *_: ExternalIdentityNormalizationResult(candidates=[]),
    )

    assert result.items[0].candidates[0].reason == (
        "公开资料时间为 2025-04，身份检索截止 2026-07-28"
    )
