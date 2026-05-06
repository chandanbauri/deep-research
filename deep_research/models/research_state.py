from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class RawResult(BaseModel):
    url: str
    title: str
    raw_text: str
    char_count: int
    source_query: str


class Summary(BaseModel):
    source_url: str
    source_query: str
    main_points: list[str]
    key_facts: list[str]
    relevance_score: int
    raw_text: str


class CritiqueResult(BaseModel):
    quality_score: int = Field(ge=1, le=10)
    gaps: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    verdict: str = Field(pattern="^(sufficient|needs_more_research)$")


class ResearchState(BaseModel):
    query: str
    plan: list[str] = Field(default_factory=list)
    raw_results: list[RawResult] = Field(default_factory=list)
    summaries: list[Summary] = Field(default_factory=list)
    critique: CritiqueResult | None = None
    synthesis_sections: list[Any] = Field(default_factory=list)
    final_report: Any | None = None
    iteration_count: int = 0
    max_iterations: int = 3
    visited_urls: set[str] = Field(default_factory=set)

    model_config = {"arbitrary_types_allowed": True}

    def mark_visited(self, url: str) -> None:
        self.visited_urls.add(url)

    def is_visited(self, url: str) -> bool:
        return url in self.visited_urls
