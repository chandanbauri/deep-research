from __future__ import annotations

"""
FileWriterTool
──────────────
Purpose : Write or append text content to a file within the outputs directory.
Library : Python stdlib pathlib — no external dependency needed.

Why needed: Agents must persist intermediate work (summaries, section drafts,
final report) between pipeline steps. This enables resumable runs and allows
agents to read back earlier work via FileReaderTool.

Security: Path is validated against OUTPUT_DIR via Path.resolve() to prevent
directory traversal attacks.

Input  : path (str), content (str), mode ("write" | "append", default "write")
Output : {path, bytes_written}
"""

import asyncio
from pathlib import Path
from typing import Literal

import deep_research.config as _cfg
from deep_research.tools.base import BaseTool, ToolResult
from deep_research.utils.logger import get_logger

log = get_logger(__name__)


class FileWriterTool(BaseTool):
    name = "file_writer"
    description = (
        'file_writer(path: str, content: str, mode: "write"|"append" = "write") -> {path, bytes_written}\n'
        "Write or append content to a file in the outputs directory."
    )

    async def run(
        self,
        path: str,
        content: str,
        mode: Literal["write", "append"] = "write",
    ) -> ToolResult:
        OUTPUT_DIR = _cfg.OUTPUT_DIR
        target = (OUTPUT_DIR / path).resolve()
        if not str(target).startswith(str(OUTPUT_DIR.resolve())):
            return ToolResult(success=False, error="Path traversal denied.")
        try:
            await asyncio.to_thread(self._write, target, content, mode)
            log.info(f"[file_writer] {mode} {target} ({len(content)} chars)")
            return ToolResult(success=True, data={"path": str(target), "bytes_written": len(content.encode())})
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    @staticmethod
    def _write(target: Path, content: str, mode: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_mode = "a" if mode == "append" else "w"
        target.write_text(content, encoding="utf-8") if write_mode == "w" else \
            target.open("a", encoding="utf-8").write(content)
