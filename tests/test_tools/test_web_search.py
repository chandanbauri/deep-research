from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from deep_research.tools.web_search import WebSearchTool


@pytest.mark.asyncio
async def test_web_search_returns_results():
    mock_results = [
        {"title": "Test", "href": "https://example.com", "body": "A test page."}
    ]
    with patch("deep_research.tools.web_search.DDGS") as mock_ddgs:
        instance = MagicMock()
        instance.__enter__ = MagicMock(return_value=instance)
        instance.__exit__ = MagicMock(return_value=False)
        instance.text = MagicMock(return_value=mock_results)
        mock_ddgs.return_value = instance

        tool = WebSearchTool()
        result = await tool.run(query="test query", max_results=1)

    assert result.success is True
    assert len(result.data) == 1
    assert result.data[0]["url"] == "https://example.com"
    assert result.data[0]["title"] == "Test"


@pytest.mark.asyncio
async def test_web_search_handles_error():
    with patch("deep_research.tools.web_search.DDGS") as mock_ddgs:
        mock_ddgs.side_effect = RuntimeError("network error")
        tool = WebSearchTool()
        result = await tool.run(query="failing query")

    assert result.success is False
    assert "network error" in result.error
