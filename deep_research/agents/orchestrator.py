from __future__ import annotations

import json
import re

from deep_research.agents.base_agent import BaseAgent
from deep_research.config import MODELS
from deep_research.llm.client import OllamaClient
from deep_research.models.research_state import ResearchState
from deep_research.utils.logger import get_logger

log = get_logger(__name__)

_ROLE = """You are the Orchestrator of a deep research team. Your job is to:
1. Break a research query into 3-5 focused sub-questions that, when answered together,
   fully address the original query.
2. After reviewing research results, decide whether the team has gathered sufficient
   information or needs to research further gaps.

Always respond with valid JSON. No prose outside the JSON block."""


class OrchestratorAgent(BaseAgent):
    def __init__(self, llm_client: OllamaClient | None = None) -> None:
        super().__init__(
            name="Orchestrator",
            role_description=_ROLE,
            model=MODELS["orchestrator"],
            tool_names=[],
            llm_client=llm_client,
        )

    async def plan(self, state: ResearchState) -> list[str]:
        prompt = (
            f'Research query: "{state.query}"\n\n'
            "Generate 3-5 focused sub-questions that will help answer this query comprehensively.\n"
            'Respond with JSON: {"sub_questions": ["Q1", "Q2", ...]}'
        )
        msg = await self.run(prompt)
        return self._extract_list(msg.content, "sub_questions")

    async def decide_next_step(self, state: ResearchState) -> str:
        gaps = state.critique.gaps if state.critique else []
        prompt = (
            f'Original query: "{state.query}"\n'
            f"Iteration: {state.iteration_count}/{state.max_iterations}\n"
            f"Critique score: {state.critique.quality_score if state.critique else 'N/A'}\n"
            f"Gaps identified: {gaps}\n\n"
            "Decide the next action. If there are important gaps AND iterations remain, "
            "return new sub-questions to research. Otherwise signal completion.\n"
            'Respond with JSON: {"action": "research"|"done", "sub_questions": [...] or []}'
        )
        msg = await self.run(prompt)
        data = self._parse_json(msg.content)
        if data.get("action") == "research" and data.get("sub_questions"):
            return "research"
        return "done"

    async def new_sub_questions(self, state: ResearchState) -> list[str]:
        gaps = state.critique.gaps if state.critique else []
        prompt = (
            f'Research gaps identified: {gaps}\n'
            f'Original query: "{state.query}"\n\n'
            "Generate 2-3 sub-questions targeting these specific gaps.\n"
            'Respond with JSON: {"sub_questions": ["Q1", "Q2"]}'
        )
        msg = await self.run(prompt)
        return self._extract_list(msg.content, "sub_questions")

    @staticmethod
    def _extract_list(text: str, key: str) -> list[str]:
        data = OrchestratorAgent._parse_json(text)
        result = data.get(key, [])
        if isinstance(result, list):
            return [str(q) for q in result if q]
        return []

    @staticmethod
    def _parse_json(text: str) -> dict:
        # Try JSON block first, then raw parse
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        raw = match.group(1) if match else text
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Last resort: find first {...} block
            brace = re.search(r"\{.*\}", raw, re.S)
            if brace:
                try:
                    return json.loads(brace.group())
                except json.JSONDecodeError:
                    pass
        return {}
