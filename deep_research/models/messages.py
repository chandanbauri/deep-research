from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = {}


class ToolResult(BaseModel):
    tool: str
    success: bool
    data: Any = None
    error: str | None = None


class AgentMessage(BaseModel):
    agent: str
    role: str          # "assistant" or "user"
    content: str
    tool_calls: list[ToolCall] = []
    tool_results: list[ToolResult] = []
