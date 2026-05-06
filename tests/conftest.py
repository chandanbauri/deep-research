from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from deep_research.llm.client import OllamaClient
from deep_research.tools.base import ToolResult


@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=OllamaClient)
    client.chat = AsyncMock(return_value='{"result": "test response"}')
    return client


@pytest.fixture
def search_results():
    return [
        {"title": "Article A", "url": "https://example.com/a", "snippet": "About lithium mining."},
        {"title": "Article B", "url": "https://example.com/b", "snippet": "Environmental impact."},
    ]


@pytest.fixture
def scrape_result():
    return ToolResult(
        success=True,
        data={
            "url": "https://example.com/a",
            "title": "Article A",
            "text": "Lithium mining has significant environmental impacts including water usage and land degradation.",
            "char_count": 90,
        },
    )
