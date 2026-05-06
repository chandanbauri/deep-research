from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from deep_research.tools.web_scraper import WebScraperTool


@pytest.mark.asyncio
async def test_scraper_extracts_text():
    html = "<html><body><p>This is a meaningful paragraph about research topics.</p></body></html>"
    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("deep_research.tools.web_scraper.httpx.AsyncClient", return_value=mock_client):
        tool = WebScraperTool()
        result = await tool.run(url="https://example.com")

    assert result.success is True
    assert "meaningful paragraph" in result.data["text"]
    assert result.data["url"] == "https://example.com"


@pytest.mark.asyncio
async def test_scraper_handles_http_error():
    import httpx
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_client.get = AsyncMock(
        side_effect=httpx.HTTPStatusError("Not Found", request=MagicMock(), response=mock_response)
    )

    with patch("deep_research.tools.web_scraper.httpx.AsyncClient", return_value=mock_client):
        tool = WebScraperTool()
        result = await tool.run(url="https://example.com/missing")

    assert result.success is False
    assert "404" in result.error
