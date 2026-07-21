import pytest

import app.services.tavily_client as tavily_module
from app.services.tavily_client import TavilyClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeHttpClient:
    def __init__(self, *_, **__):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def post(self, url, json, headers):
        assert headers["Authorization"] == "Bearer test-key"
        if url.endswith("/search"):
            suffix = "a" if "人物" in json["query"] else "b"
            results = [
                {
                    "title": f"页面-{index}",
                    "url": f"https://example.com/{suffix}/{index}",
                    "content": "摘要",
                }
                for index in range(5)
            ]
            results.append(results[0])
            return FakeResponse({"results": results})
        return FakeResponse(
            {
                "results": [
                    {"url": target, "raw_content": f"正文 {target}"}
                    for target in json["urls"]
                ]
            }
        )


@pytest.mark.asyncio
async def test_search_deduplicates_and_extract_preserves_titles(monkeypatch) -> None:
    monkeypatch.setattr(tavily_module.httpx, "AsyncClient", FakeHttpClient)
    client = TavilyClient("test-key")

    results = await client.search(["人物 负责业务", "单位 主营业务"])
    pages = await client.extract(results)

    assert len(results) == 10
    assert len({item.url for item in results}) == 10
    assert len(pages) == 10
    assert pages[0].title == results[0].title
    assert pages[0].raw_content.startswith("正文 https://")
