from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: str | None = None


class BaseTool(ABC):
    """Abstract base class for all tools.

    Every tool exposes a single async `run(**kwargs)` method and a human-readable
    `description` string that is injected into agent system prompts so the LLM
    knows what tools are available and how to call them.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        ...
