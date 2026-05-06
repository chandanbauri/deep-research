from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from deep_research.agents.base_agent import BaseAgent
from deep_research.llm.client import OllamaClient


class ConcreteAgent(BaseAgent):
    pass


@pytest.mark.asyncio
async def test_agent_returns_plain_response_when_no_tool_calls():
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.chat = AsyncMock(return_value="This is my final answer with no tools.")

    agent = ConcreteAgent(
        name="TestAgent",
        role_description="You are a helpful test agent.",
        model="test-model",
        tool_names=[],
        llm_client=mock_client,
    )
    msg = await agent.run("What is the capital of France?")
    assert msg.content == "This is my final answer with no tools."
    assert msg.tool_calls == []


@pytest.mark.asyncio
async def test_agent_parses_and_executes_tool_call():
    tool_response = '```json\n{"tool": "web_search", "args": {"query": "test"}}\n```'
    final_response = "Based on the search results, here is my answer."

    call_count = 0

    async def fake_chat(messages, model, options=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return final_response

    mock_client = MagicMock(spec=OllamaClient)
    mock_client.chat = fake_chat

    # Patch the web_search tool to avoid real HTTP calls
    from unittest.mock import patch
    from deep_research.tools.base import ToolResult as TR

    mock_tool = MagicMock()
    mock_tool.run = AsyncMock(return_value=TR(success=True, data=[{"url": "x", "title": "T", "snippet": "s"}]))

    with patch("deep_research.agents.base_agent.get_tool", return_value=mock_tool):
        agent = ConcreteAgent(
            name="TestAgent",
            role_description="You are a helpful test agent.",
            model="test-model",
            tool_names=["web_search"],
            llm_client=mock_client,
        )
        msg = await agent.run("Search for something.")

    assert msg.content == final_response
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].tool == "web_search"
