from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from deep_research.models.report import FinalReport, ReportSection
from deep_research.models.research_state import CritiqueResult, RawResult, Summary
from deep_research.pipeline.runner import ResearchPipeline


def _make_report() -> FinalReport:
    return FinalReport(
        title="Test Report",
        query="test query",
        executive_summary="Summary text.",
        sections=[ReportSection(title="Section 1", content="Content.", sources=[])],
        conclusion="Conclusion.",
        sources=["https://example.com"],
    )


def _make_raw() -> RawResult:
    return RawResult(url="https://example.com", title="T", raw_text="text", char_count=4, source_query="q")


def _make_summary() -> Summary:
    return Summary(
        source_url="https://example.com",
        source_query="q",
        main_points=["p"],
        key_facts=["f"],
        relevance_score=8,
        raw_text="t",
    )


@pytest.mark.asyncio
async def test_pipeline_runs_end_to_end():
    """
    Integration test: replace all six agents with mocks and verify the
    LangGraph pipeline calls each node in the correct order and returns a FinalReport.

    The pipeline._agents dict is replaced AFTER __init__ because the LangGraph
    node closures capture `pipeline` (not `pipeline._agents`), so they always
    look up agents at call time — making post-construction injection work.
    """
    report = _make_report()
    raw = _make_raw()
    summary = _make_summary()
    critique_sufficient = CritiqueResult(
        quality_score=8, gaps=[], contradictions=[], verdict="sufficient"
    )
    sections = [ReportSection(title="S1", content="C1", sources=[])]

    mock_orchestrator = MagicMock()
    mock_orchestrator.plan = AsyncMock(return_value=["sub-question 1"])
    mock_orchestrator.new_sub_questions = AsyncMock(return_value=[])

    mock_researcher = MagicMock()
    mock_researcher.research = AsyncMock(return_value=[raw])

    mock_summarizer = MagicMock()
    mock_summarizer.summarize = AsyncMock(return_value=summary)

    mock_critic = MagicMock()
    mock_critic.critique = AsyncMock(return_value=critique_sufficient)

    mock_synthesizer = MagicMock()
    mock_synthesizer.synthesize = AsyncMock(return_value=sections)

    mock_writer = MagicMock()
    mock_writer.write = AsyncMock(return_value=report)

    mock_state_manager = MagicMock()
    mock_state_manager.checkpoint = AsyncMock(
        return_value=MagicMock(parent=MagicMock(__truediv__=lambda s, o: "outputs/test/report.md"))
    )

    pipeline = ResearchPipeline(max_iterations=1)

    # Inject mocked agents AFTER init — works because node closures
    # reference pipeline._agents at call time, not at graph-build time.
    pipeline._agents = {
        "orchestrator": mock_orchestrator,
        "researcher":   mock_researcher,
        "summarizer":   mock_summarizer,
        "critic":       mock_critic,
        "synthesizer":  mock_synthesizer,
        "writer":       mock_writer,
    }
    pipeline._state_manager = mock_state_manager

    result = await pipeline.run("test query")

    assert result.title == "Test Report"
    mock_orchestrator.plan.assert_called_once()
    mock_researcher.research.assert_called_once()
    mock_summarizer.summarize.assert_called_once()
    mock_critic.critique.assert_called_once()
    mock_synthesizer.synthesize.assert_called_once()
    mock_writer.write.assert_called_once()
