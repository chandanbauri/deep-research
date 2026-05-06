from __future__ import annotations

import asyncio

from deep_research.config import MAX_SCRAPE_URLS, MODELS
from deep_research.llm.client import OllamaClient
from deep_research.models.research_state import RawResult, ResearchState
from deep_research.tools import get_tool
from deep_research.utils.logger import get_logger

log = get_logger(__name__)


class ResearcherAgent:
    """Gathers raw content for a single sub-question.

    Tools used:
      - web_search: find relevant URLs and snippets
      - deduplicator: skip URLs already visited in this research session
      - web_scraper: fetch full page text from the top URLs

    Why this agent doesn't inherit BaseAgent: its workflow is fully deterministic
    (search → dedup → scrape). No LLM reasoning is needed to decide which tools to
    call — the sequence is always the same. Using a fixed async flow is simpler,
    faster, and cheaper than adding an LLM planning loop here.
    """

    def __init__(self, llm_client: OllamaClient | None = None) -> None:
        self.name = "Researcher"
        self._search = get_tool("web_search")
        self._scraper = get_tool("web_scraper")
        self._dedup = get_tool("deduplicator")

    async def research(self, state: ResearchState, sub_question: str) -> list[RawResult]:
        log.info(f"[Researcher] researching: {sub_question!r}")

        search_result = await self._search.run(query=sub_question, max_results=5)
        if not search_result.success:
            log.warning(f"[Researcher] search failed: {search_result.error}")
            return []

        all_items: list[dict] = search_result.data or []

        # Deduplicate against already-visited URLs
        dedup_result = await self._dedup.run(items=all_items, mode="url")
        candidates = [
            item for item in (dedup_result.data or [])
            if not state.is_visited(item.get("url", ""))
        ][:MAX_SCRAPE_URLS]

        # Scrape all candidate URLs concurrently
        scrape_tasks = [self._scraper.run(url=item["url"]) for item in candidates]
        scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

        raw_results: list[RawResult] = []
        for item, scrape in zip(candidates, scrape_results):
            url = item["url"]
            if isinstance(scrape, Exception) or not scrape.success:
                log.warning(f"[Researcher] scrape failed for {url}")
                continue
            data = scrape.data
            if not data or not data.get("text"):
                continue
            state.mark_visited(url)
            raw_results.append(RawResult(
                url=url,
                title=data.get("title", item.get("title", "")),
                raw_text=data["text"],
                char_count=data.get("char_count", len(data["text"])),
                source_query=sub_question,
            ))
            log.info(f"[Researcher] scraped {url} ({data.get('char_count', 0)} chars)")

        return raw_results
