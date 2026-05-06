from __future__ import annotations

"""
TextChunkerTool
───────────────
Purpose : Split a long document into overlapping chunks suitable for LLM processing.
Library : Python stdlib re — sentence boundary detection.

Why needed: Scraped web pages can exceed 100k characters. Available Ollama models
have a ~32k token context window (~128k chars). Without chunking, the summarizer
would either truncate content or exceed the context limit. Overlapping chunks
prevent facts that span chunk boundaries from being lost.

Input  : text (str), chunk_size (int, default 1500 chars), overlap (int, default 150)
Output : list[str] — ordered list of text chunks
"""

import re

from deep_research.config import CHUNK_SIZE, CHUNK_OVERLAP
from deep_research.tools.base import BaseTool, ToolResult


class TextChunkerTool(BaseTool):
    name = "text_chunker"
    description = (
        "text_chunker(text: str, chunk_size: int = 1500, overlap: int = 150) -> list[str]\n"
        "Split long text into overlapping chunks for LLM processing."
    )

    async def run(
        self,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> ToolResult:
        if not text.strip():
            return ToolResult(success=True, data=[])

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= chunk_size:
                current = (current + " " + sentence).strip()
            else:
                if current:
                    chunks.append(current)
                # Start new chunk with overlap from end of previous
                current = (current[-overlap:] + " " + sentence).strip() if current else sentence

        if current:
            chunks.append(current)

        return ToolResult(success=True, data=chunks)
