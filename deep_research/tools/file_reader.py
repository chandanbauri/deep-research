from __future__ import annotations

"""
FileReaderTool
──────────────
Purpose : Read a local text or Markdown file from the outputs directory.
Library : Python stdlib pathlib — no external dependency needed.

Why needed: Agents (SynthesizerAgent, WriterAgent) need to re-read previously
saved summaries and drafts from disk without keeping them all in memory.

Security: The resolved path must stay within OUTPUT_DIR to prevent path traversal.

Input  : path (str) — relative to OUTPUT_DIR, e.g. "my-topic/summaries/page1.md"
Output : {path, content, lines}
"""

import asyncio
from pathlib import Path

import deep_research.config as _cfg
from deep_research.tools.base import BaseTool, ToolResult
from deep_research.utils.logger import get_logger

log = get_logger(__name__)


class FileReaderTool(BaseTool):
    name = "file_reader"
    description = (
        "file_reader(path: str) -> {path, content, lines}\n"
        "Read a file from the outputs directory. Path is relative to outputs/."
    )

    async def run(self, path: str) -> ToolResult:
        OUTPUT_DIR = _cfg.OUTPUT_DIR
        target = (OUTPUT_DIR / path).resolve()
        if not str(target).startswith(str(OUTPUT_DIR.resolve())):
            return ToolResult(success=False, error="Path traversal denied.")
        if not target.exists():
            return ToolResult(success=False, error=f"File not found: {path}")
        try:
            content = await asyncio.to_thread(target.read_text, encoding="utf-8")
            log.info(f"[file_reader] read {target} ({len(content)} chars)")
            return ToolResult(success=True, data={"path": str(target), "content": content, "lines": content.count("\n")})
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))
