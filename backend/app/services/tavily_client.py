import asyncio
from datetime import datetime

import httpx

from app.schemas.task import SearchResult, WebPage


class TavilyClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com"

    async def search(self, queries: list[str]) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is empty")
        output: list[SearchResult] = []
        seen: set[str] = set()
        async with httpx.AsyncClient(timeout=15) as client:
            for query in queries:
                payload = {"query": query, "search_depth": "basic", "max_results": 5}
                response = await self._post_with_retry(client, "/search", payload)
                for item in response.get("results", []):
                    url = item.get("url", "")
                    if not url or url in seen or len(output) >= 10:
                        continue
                    seen.add(url)
                    output.append(
                        SearchResult(
                            web_result_id=f"W{len(output) + 1:03d}",
                            title=item.get("title") or url,
                            url=url,
                            content=item.get("content", ""),
                            query=query,
                            rank=len(output),
                            published_at=parse_datetime(item.get("published_date")),
                        )
                    )
        return output

    async def extract(self, results: list[SearchResult]) -> list[WebPage]:
        if not results:
            return []
        async with httpx.AsyncClient(timeout=30) as client:
            payload = {"urls": [item.url for item in results], "extract_depth": "basic"}
            response = await self._post_with_retry(client, "/extract", payload)
        by_url = {item.url: item for item in results}
        pages: list[WebPage] = []
        for item in response.get("results", []):
            url = item.get("url", "")
            source = by_url.get(url)
            content = item.get("raw_content") or ""
            if source and content:
                pages.append(
                    WebPage(
                        web_result_id=source.web_result_id,
                        title=source.title,
                        url=url,
                        raw_content=content[:20_000],
                        rank=source.rank,
                        published_at=source.published_at,
                    )
                )
        return pages

    async def _post_with_retry(
        self, client: httpx.AsyncClient, path: str, payload: dict[str, object]
    ) -> dict:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await client.post(
                    f"{self.base_url}{path}",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(f"Tavily request failed: {last_error}")


def parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
