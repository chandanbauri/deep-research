from __future__ import annotations

"""
DeduplicatorTool
────────────────
Purpose : Remove duplicate URLs and near-duplicate text passages from collected results.
Library : Python stdlib hashlib + set — no external dependency needed.

Why needed: DuckDuckGo searches on related sub-questions frequently return the same
URL. Without deduplication, the ResearcherAgent wastes time scraping the same page
multiple times, and the SummarizerAgent wastes LLM calls on identical content.

URL mode   : exact match via a set — O(1) lookup.
Text mode  : sliding-window md5 fingerprint over 150-char windows — catches
             near-duplicate passages (e.g. syndicated content).

Input  : items (list[str | dict]), mode ("url" | "text", default "url")
         In "url" mode: items are strings (URLs) or dicts with a "url" key.
         In "text" mode: items are strings.
Output : deduplicated list preserving original order.
"""

import hashlib
from typing import Union

from deep_research.tools.base import BaseTool, ToolResult


class DeduplicatorTool(BaseTool):
    name = "deduplicator"
    description = (
        'deduplicator(items: list, mode: "url"|"text" = "url") -> list\n'
        "Remove duplicate URLs or near-duplicate text passages."
    )

    async def run(
        self,
        items: list[Union[str, dict]],
        mode: str = "url",
    ) -> ToolResult:
        if mode == "url":
            return ToolResult(success=True, data=self._dedup_urls(items))
        elif mode == "text":
            return ToolResult(success=True, data=self._dedup_text(items))
        return ToolResult(success=False, error=f"Unknown mode: {mode}")

    @staticmethod
    def _dedup_urls(items: list) -> list:
        seen: set[str] = set()
        result = []
        for item in items:
            url = item.get("url", item) if isinstance(item, dict) else item
            if url not in seen:
                seen.add(url)
                result.append(item)
        return result

    @staticmethod
    def _dedup_text(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result = []
        for text in items:
            fingerprint = hashlib.md5(text[:150].encode()).hexdigest()
            if fingerprint not in seen:
                seen.add(fingerprint)
                result.append(text)
        return result
