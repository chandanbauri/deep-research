from __future__ import annotations

import json
import re

from deep_research.config import MODELS
from deep_research.llm.client import OllamaClient
from deep_research.llm.prompts import build_messages, format_system_prompt
from deep_research.models.research_state import CritiqueResult, ResearchState
from deep_research.utils.logger import get_logger

log = get_logger(__name__)

_ROLE = """You are a rigorous academic research critic. Your job is to evaluate whether
the collected research summaries sufficiently answer the original query.

Assess:
1. Are all major aspects of the query addressed?
2. Are there contradictions between sources?
3. Are important perspectives or data points missing?

Respond ONLY with valid JSON:
{
  "quality_score": 7,
  "gaps": ["missing topic 1", "missing data point 2"],
  "contradictions": ["source A says X but source B says Y"],
  "verdict": "sufficient" or "needs_more_research"
}

Use verdict "sufficient" if quality_score >= 7 and no critical gaps remain.
Use verdict "needs_more_research" if quality_score < 7 or critical gaps exist."""


class CriticAgent:
    """Evaluates research quality and identifies gaps.

    Pure reasoning — no tool calls. Uses the most capable available model
    (qwen2.5-coder:3b) because multi-document gap analysis is a complex task.
    """

    def __init__(self, llm_client: OllamaClient | None = None) -> None:
        self.name = "Critic"
        self.model = MODELS["critic"]
        self._client = llm_client or OllamaClient()
        self._system = format_system_prompt(_ROLE)

    async def critique(self, state: ResearchState) -> CritiqueResult:
        log.info(f"[Critic] evaluating {len(state.summaries)} summaries")

        summaries_text = self._format_summaries(state)
        prompt = (
            f'Original research query: "{state.query}"\n\n'
            f"Research collected so far:\n\n{summaries_text}"
        )
        messages = build_messages(self._system, [], prompt)
        response = await self._client.chat(messages, model=self.model)

        result = self._parse_critique(response)
        log.info(f"[Critic] score={result.quality_score}, verdict={result.verdict}, gaps={len(result.gaps)}")
        return result

    @staticmethod
    def _format_summaries(state: ResearchState) -> str:
        lines = []
        for i, s in enumerate(state.summaries, 1):
            lines.append(f"--- Source {i}: {s.source_url} ---")
            lines.append(f"Query answered: {s.source_query}")
            lines.append("Main points: " + "; ".join(s.main_points[:5]))
            lines.append("Key facts: " + "; ".join(s.key_facts[:5]))
            lines.append("")
        return "\n".join(lines) or "No summaries collected yet."

    @staticmethod
    def _parse_critique(text: str) -> CritiqueResult:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        raw = match.group(1) if match else text
        data: dict = {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            brace = re.search(r"\{.*\}", raw, re.S)
            if brace:
                try:
                    data = json.loads(brace.group())
                except json.JSONDecodeError:
                    pass

        score = max(1, min(10, int(data.get("quality_score", 5))))
        verdict = data.get("verdict", "needs_more_research")
        if verdict not in ("sufficient", "needs_more_research"):
            verdict = "sufficient" if score >= 7 else "needs_more_research"

        return CritiqueResult(
            quality_score=score,
            gaps=data.get("gaps", []),
            contradictions=data.get("contradictions", []),
            verdict=verdict,
        )
