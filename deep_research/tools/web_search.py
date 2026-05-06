from __future__ import annotations

"""
WebSearchTool
─────────────
Purpose : Retrieve a ranked list of URLs and snippets for a natural-language query.
Library : duckduckgo-search (pip install duckduckgo-search)

Why duckduckgo-search:
  - Completely free — no API key or account required.
  - Pure Python implementation; no Selenium or browser dependency.
  - Actively maintained (v6+ has stable async-compatible interface).
  - Returns structured dicts with title, href, and body fields.

Input  : query (str), max_results (int, default 5)
Output : list of dicts — [{title, url, snippet}, ...]
"""

import asyncio
from duckduckgo_search import DDGS

from deep_research.tools.base import BaseTool, ToolResult
from deep_research.utils.logger import get_logger

log = get_logger(__name__)


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        'web_search(query: str, max_results: int = 5) -> list[{title, url, snippet}]\n'
        "Search the web for a query. Returns ranked URLs with titles and snippets."
    )

    async def run(self, query: str, max_results: int = 5) -> ToolResult:
        log.info(f"[web_search] query={query!r} max_results={max_results}")
        try:
            results = await asyncio.to_thread(
                self._search_sync, query, max_results
            )
            log.info(f"[web_search] found {len(results)} results")
            return ToolResult(success=True, data=results)
        except Exception as exc:
            log.warning(f"[web_search] failed: {exc}")
            return ToolResult(success=False, error=str(exc))

    @staticmethod
    def _search_sync(query: str, max_results: int) -> list[dict]:
        # duckduckgo_search is synchronous; we run it in a thread via asyncio.to_thread
        with DDGS() as ddgs:
            raw = ddgs.text(query, max_results=max_results)
            return [
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                for r in raw
            ]
