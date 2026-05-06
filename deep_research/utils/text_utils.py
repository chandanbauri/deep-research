from __future__ import annotations

import re


def strip_html_boilerplate(html: str) -> str:
    """Remove tags, scripts, styles, and collapse whitespace."""
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.S | re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"&[a-z]+;", " ", html)
    html = re.sub(r"\s+", " ", html)
    return html.strip()


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Rough truncation: ~4 chars per token."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def slugify(text: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = text[:60]
    return text.strip("-")
