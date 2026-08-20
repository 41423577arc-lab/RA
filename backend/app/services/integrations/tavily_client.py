import asyncio
from datetime import datetime

import httpx

from app.schemas.task import SearchResult, WebPage


class TavilyClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com"

    async def search(self, queries: list[str]) -> list[SearchResult]:
        return await self._search(
            queries,
            search_depth="basic",
            max_results=5,
            max_total_results=10,
        )

    async def search_identity(
        self,
        queries: list[str],
        *,
        start_date: str,
        end_date: str,
    ) -> list[SearchResult]:
        return await self._search(
            queries,
            search_depth="advanced",
            max_results=10,
            max_total_results=20,
            country="china",
            start_date=start_date,
            end_date=end_date,
        )

    async def _search(
        self,
        queries: list[str],
        *,
        search_depth: str,
        max_results: int,
        max_total_results: int,
        country: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is empty")
        output: list[SearchResult] = []
        seen: set[str] = set()
        async with httpx.AsyncClient(timeout=15) as client:
            for query in queries:
                payload: dict[str, object] = {
                    "query": query,
                    "search_depth": search_depth,
                    "max_results": max_results,
                }
                if country:
                    payload["country"] = country
                if start_date:
                    payload["start_date"] = start_date
                if end_date:
                    payload["end_date"] = end_date
                response = await self._post_with_retry(client, "/search", payload)
                for item in response.get("results", []):
                    url = item.get("url", "")
                    if not url or url in seen or len(output) >= max_total_results:
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
        return await self._extract(results, extract_depth="basic")

    async def extract_identity(self, results: list[SearchResult]) -> list[WebPage]:
        return await self._extract(results, extract_depth="advanced")

    async def _extract(
        self, results: list[SearchResult], *, extract_depth: str
    ) -> list[WebPage]:
        if not results:
            return []
        async with httpx.AsyncClient(timeout=30) as client:
            payload = {
                "urls": [item.url for item in results],
                "extract_depth": extract_depth,
            }
            response = await self._post_with_retry(client, "/extract", payload)
        extracted_by_url = {
            item.get("url", ""): item.get("raw_content") or ""
            for item in response.get("results", [])
        }
        pages: list[WebPage] = []
        for source in results:
            content = extracted_by_url.get(source.url) or source.content
            if content:
                pages.append(
                    WebPage(
                        web_result_id=source.web_result_id,
                        title=source.title,
                        url=source.url,
                        raw_content=content[:20_000],
                        rank=source.rank,
                        query=source.query,
                        target_person=source.target_person,
                        target_organization=source.target_organization,
                        search_snippet=source.content,
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
