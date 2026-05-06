from __future__ import annotations

"""
ToolRegistry
────────────
Central registry of all available tools. Agents query this registry at startup
to get tool instances and inject tool descriptions into their system prompts.

Why a registry: It decouples agent code from specific tool implementations.
Adding a new tool only requires registering it here — no agent code changes.
"""

from deep_research.tools.base import BaseTool
from deep_research.tools.deduplicator import DeduplicatorTool
from deep_research.tools.file_reader import FileReaderTool
from deep_research.tools.file_writer import FileWriterTool
from deep_research.tools.text_chunker import TextChunkerTool
from deep_research.tools.web_scraper import WebScraperTool
from deep_research.tools.web_search import WebSearchTool

_REGISTRY: dict[str, BaseTool] = {
    "web_search": WebSearchTool(),
    "web_scraper": WebScraperTool(),
    "file_reader": FileReaderTool(),
    "file_writer": FileWriterTool(),
    "text_chunker": TextChunkerTool(),
    "deduplicator": DeduplicatorTool(),
}


def get_tool(name: str) -> BaseTool:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown tool: {name!r}. Available: {list_tools()}")
    return _REGISTRY[name]


def list_tools() -> list[str]:
    return list(_REGISTRY.keys())


def describe_tools(tool_names: list[str] | None = None) -> str:
    """Return a formatted description block suitable for injection into a system prompt."""
    names = tool_names if tool_names is not None else list_tools()
    lines = []
    for name in names:
        if name in _REGISTRY:
            lines.append(f"- {_REGISTRY[name].description}")
    return "\n".join(lines)
