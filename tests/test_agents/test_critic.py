from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from deep_research.agents.critic import CriticAgent
from deep_research.llm.client import OllamaClient
from deep_research.models.research_state import ResearchState, Summary


def make_state() -> ResearchState:
    state = ResearchState(query="test query")
    state.summaries = [
        Summary(
            source_url="https://example.com",
            source_query="test query",
            main_points=["Point A", "Point B"],
            key_facts=["Fact 1"],
            relevance_score=7,
            raw_text="sample text",
        )
    ]
    return state


@pytest.mark.asyncio
async def test_critic_parses_sufficient_verdict():
    mock_response = '{"quality_score": 8, "gaps": [], "contradictions": [], "verdict": "sufficient"}'
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.chat = AsyncMock(return_value=mock_response)

    critic = CriticAgent(llm_client=mock_client)
    result = await critic.critique(make_state())

    assert result.quality_score == 8
    assert result.verdict == "sufficient"
    assert result.gaps == []


@pytest.mark.asyncio
async def test_critic_parses_needs_more_verdict():
    mock_response = '{"quality_score": 4, "gaps": ["missing economic data"], "contradictions": [], "verdict": "needs_more_research"}'
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.chat = AsyncMock(return_value=mock_response)

    critic = CriticAgent(llm_client=mock_client)
    result = await critic.critique(make_state())

    assert result.quality_score == 4
    assert result.verdict == "needs_more_research"
    assert "missing economic data" in result.gaps


@pytest.mark.asyncio
async def test_critic_falls_back_on_malformed_json():
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.chat = AsyncMock(return_value="I think this is sufficient research. No JSON here.")

    critic = CriticAgent(llm_client=mock_client)
    result = await critic.critique(make_state())

    # Should not raise; verdict falls back to needs_more_research (score < 7)
    assert result.verdict in ("sufficient", "needs_more_research")
