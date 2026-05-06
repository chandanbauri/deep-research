from __future__ import annotations

"""
WebScraperTool
──────────────
Purpose : Fetch the full textual content of a web page, stripping navigation,
          ads, scripts, and HTML boilerplate.
Libraries:
  - httpx (pip install httpx) — async HTTP client.
    Why: already installed; async-native; handles redirects, timeouts, and
    connection pooling cleanly.
  - beautifulsoup4 + lxml — HTML parsing and text extraction.
    Why: BS4 is the industry standard for reliable HTML parsing; lxml is a
    fast C-based backend that handles malformed HTML better than html.parser.

Input  : url (str), timeout (int, default 15 seconds)
Output : {url, title, text, char_count}
"""

import httpx
from bs4 import BeautifulSoup

from deep_research.config import SCRAPE_TIMEOUT
from deep_research.tools.base import BaseTool, ToolResult
from deep_research.utils.logger import get_logger

log = get_logger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; DeepResearchBot/1.0; +research-tool)"
    )
}


class WebScraperTool(BaseTool):
    name = "web_scraper"
    description = (
        "web_scraper(url: str) -> {url, title, text, char_count}\n"
        "Fetch and extract clean text from a web page URL."
    )

    async def run(self, url: str, timeout: int = SCRAPE_TIMEOUT) -> ToolResult:
        log.info(f"[web_scraper] fetching {url}")
        try:
            async with httpx.AsyncClient(
                headers=HEADERS,
                timeout=timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Remove noise elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            title = soup.title.string.strip() if soup.title else ""
            paragraphs = [t.get_text(" ", strip=True) for t in soup.find_all(["p", "h1", "h2", "h3", "h4", "li"])]
            text = "\n".join(p for p in paragraphs if len(p) > 30)

            log.info(f"[web_scraper] extracted {len(text)} chars from {url}")
            return ToolResult(
                success=True,
                data={"url": url, "title": title, "text": text, "char_count": len(text)},
            )
        except httpx.HTTPStatusError as exc:
            log.warning(f"[web_scraper] HTTP {exc.response.status_code} for {url}")
            return ToolResult(success=False, error=f"HTTP {exc.response.status_code}")
        except Exception as exc:
            log.warning(f"[web_scraper] failed for {url}: {exc}")
            return ToolResult(success=False, error=str(exc))
