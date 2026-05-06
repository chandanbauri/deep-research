from __future__ import annotations

import json
import re

from deep_research.config import CHUNK_SIZE, MODELS
from deep_research.llm.client import OllamaClient
from deep_research.llm.prompts import build_messages, format_system_prompt
from deep_research.models.research_state import RawResult, ResearchState, Summary
from deep_research.tools import get_tool
from deep_research.utils.logger import get_logger
from deep_research.utils.text_utils import slugify

log = get_logger(__name__)

_ROLE = """You are a research summarizer. Given a block of text from a web page and a
research question, extract structured information relevant to answering the question.

Respond ONLY with valid JSON in this exact format:
{
  "main_points": ["point 1", "point 2"],
  "key_facts": ["fact 1", "fact 2"],
  "relevance_score": 7
}

Do not add prose outside the JSON. Do not invent information not present in the source."""


class SummarizerAgent:
    """Distills raw scraped content into structured notes.

    Uses text_chunker to handle pages longer than the LLM context window,
    then summarizes each chunk and merges the results.
    """

    def __init__(self, llm_client: OllamaClient | None = None) -> None:
        self.name = "Summarizer"
        self.model = MODELS["summarizer"]
        self._client = llm_client or OllamaClient()
        self._chunker = get_tool("text_chunker")
        self._writer = get_tool("file_writer")
        self._system = format_system_prompt(_ROLE)

    async def summarize(self, state: ResearchState, raw: RawResult) -> Summary | None:
        log.info(f"[Summarizer] summarizing {raw.url} ({raw.char_count} chars)")

        text = raw.raw_text
        chunks_result = await self._chunker.run(text=text, chunk_size=CHUNK_SIZE)
        chunks: list[str] = chunks_result.data or [text]

        all_points: list[str] = []
        all_facts: list[str] = []
        scores: list[int] = []

        for i, chunk in enumerate(chunks[:6]):  # cap at 6 chunks per page
            prompt = (
                f'Research question: "{raw.source_query}"\n\n'
                f"Source: {raw.url}\n\n"
                f"Text (chunk {i+1}/{len(chunks)}):\n{chunk}"
            )
            messages = build_messages(self._system, [], prompt)
            try:
                response = await self._client.chat(messages, model=self.model)
                data = self._parse_json(response)
                all_points.extend(data.get("main_points", []))
                all_facts.extend(data.get("key_facts", []))
                scores.append(int(data.get("relevance_score", 5)))
            except Exception as exc:
                log.warning(f"[Summarizer] chunk {i+1} failed: {exc}")

        if not all_points and not all_facts:
            return None

        summary = Summary(
            source_url=raw.url,
            source_query=raw.source_query,
            main_points=list(dict.fromkeys(all_points))[:10],
            key_facts=list(dict.fromkeys(all_facts))[:10],
            relevance_score=round(sum(scores) / len(scores)) if scores else 5,
            raw_text=text[:500],
        )

        # Persist summary to disk so SynthesizerAgent can re-read it
        slug = slugify(state.query)
        fname = f"{slug}/summaries/{slugify(raw.url[:50])}.md"
        content = f"# Summary: {raw.title or raw.url}\n\nSource: {raw.url}\nQuery: {raw.source_query}\n\n"
        content += "## Main Points\n" + "\n".join(f"- {p}" for p in summary.main_points)
        content += "\n\n## Key Facts\n" + "\n".join(f"- {f}" for f in summary.key_facts)
        await self._writer.run(path=fname, content=content)

        log.info(f"[Summarizer] done — {len(summary.main_points)} points, score={summary.relevance_score}")
        return summary

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        raw = match.group(1) if match else text
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            brace = re.search(r"\{.*\}", raw, re.S)
            if brace:
                try:
                    return json.loads(brace.group())
                except json.JSONDecodeError:
                    pass
        return {}
