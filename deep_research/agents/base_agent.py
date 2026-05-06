from __future__ import annotations

"""
BaseAgent
─────────
The shared think → parse → act loop inherited by all six agents.

Design decisions:
  - Tool calls are expressed as JSON blocks inside ```json ... ``` fences in the
    LLM's text output. This works for all available Ollama models (qwen2.5, deepseek)
    regardless of whether they support Ollama's native tool-use API.
  - MAX_TOOL_ITERATIONS caps runaway loops. If the LLM keeps calling tools without
    producing a final answer, we return the last LLM output after N iterations.
  - Message history is kept per-invocation (not persisted across calls) to avoid
    unbounded context growth.
"""

import json
import re
from typing import Any

from deep_research.config import MAX_TOOL_ITERATIONS
from deep_research.llm.client import OllamaClient
from deep_research.llm.prompts import build_messages, format_system_prompt
from deep_research.models.messages import AgentMessage, ToolCall, ToolResult
from deep_research.tools import describe_tools, get_tool
from deep_research.utils.logger import get_logger

log = get_logger(__name__)

_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.S)


class BaseAgent:
    def __init__(
        self,
        name: str,
        role_description: str,
        model: str,
        tool_names: list[str] | None = None,
        llm_client: OllamaClient | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.tool_names: list[str] = tool_names or []
        self._client = llm_client or OllamaClient()
        self._system_prompt = format_system_prompt(
            role_description,
            describe_tools(self.tool_names) if self.tool_names else "",
        )

    async def think(self, history: list[dict], user_message: str) -> str:
        messages = build_messages(self._system_prompt, history, user_message)
        return await self._client.chat(messages, model=self.model)

    async def act(self, tool_call: ToolCall) -> ToolResult:
        try:
            tool = get_tool(tool_call.tool)
            result = await tool.run(**tool_call.args)
            return ToolResult(tool=tool_call.tool, success=result.success, data=result.data, error=result.error)
        except KeyError as exc:
            return ToolResult(tool=tool_call.tool, success=False, error=str(exc))
        except Exception as exc:
            return ToolResult(tool=tool_call.tool, success=False, error=str(exc))

    async def run(self, user_message: str, extra_context: str = "") -> AgentMessage:
        history: list[dict] = []
        message = user_message
        if extra_context:
            message = f"{extra_context}\n\n{user_message}"

        tool_calls_made: list[ToolCall] = []
        tool_results_received: list[ToolResult] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            log.info(f"[{self.name}] thinking…")
            response = await self.think(history, message)
            log.debug(f"[{self.name}] raw response: {response[:200]}…")

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})

            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                # No more tool calls — this is the final answer
                return AgentMessage(
                    agent=self.name,
                    role="assistant",
                    content=response,
                    tool_calls=tool_calls_made,
                    tool_results=tool_results_received,
                )

            # Execute each tool call and build the next user message
            result_lines: list[str] = []
            for tc in tool_calls:
                log.info(f"[{self.name}] calling tool {tc.tool!r} with {tc.args}")
                result = await self.act(tc)
                tool_calls_made.append(tc)
                tool_results_received.append(result)
                if result.success:
                    result_lines.append(f"Tool `{tc.tool}` result:\n{json.dumps(result.data, default=str)[:2000]}")
                else:
                    result_lines.append(f"Tool `{tc.tool}` error: {result.error}")

            message = "\n\n".join(result_lines)

        # Exhausted iterations — return last response
        log.warning(f"[{self.name}] hit MAX_TOOL_ITERATIONS, returning last response")
        return AgentMessage(
            agent=self.name,
            role="assistant",
            content=response,  # type: ignore[possibly-undefined]
            tool_calls=tool_calls_made,
            tool_results=tool_results_received,
        )

    @staticmethod
    def _parse_tool_calls(text: str) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for match in _JSON_BLOCK_RE.finditer(text):
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "tool" in data:
                    calls.append(ToolCall(tool=data["tool"], args=data.get("args", {})))
            except json.JSONDecodeError:
                pass
        return calls
