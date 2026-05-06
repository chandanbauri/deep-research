from __future__ import annotations

from deep_research.config import MODELS
from deep_research.llm.client import OllamaClient
from deep_research.llm.prompts import build_messages, format_system_prompt
from deep_research.models.report import ReportSection
from deep_research.models.research_state import ResearchState
from deep_research.tools import get_tool
from deep_research.utils.logger import get_logger
from deep_research.utils.text_utils import slugify, truncate_to_tokens

log = get_logger(__name__)

_ROLE = """You are a research synthesizer. Given a collection of research notes on a topic,
organize and combine them into coherent thematic sections for a research report.

For each section provide:
- A clear section title
- 2-3 paragraphs of synthesized content that integrates insights from multiple sources
- Do not simply list bullet points — write flowing, analytical paragraphs

Respond with the section title on its own line starting with '## ', followed by the content."""


class SynthesizerAgent:
    """Groups all collected summaries into 3-5 thematic report sections."""

    def __init__(self, llm_client: OllamaClient | None = None) -> None:
        self.name = "Synthesizer"
        self.model = MODELS["synthesizer"]
        self._client = llm_client or OllamaClient()
        self._writer = get_tool("file_writer")
        self._system = format_system_prompt(_ROLE)

    async def synthesize(self, state: ResearchState) -> list[ReportSection]:
        log.info(f"[Synthesizer] synthesizing {len(state.summaries)} summaries")

        notes = self._build_notes(state)
        prompt = (
            f'Research query: "{state.query}"\n\n'
            f"Research notes:\n\n{truncate_to_tokens(notes, 6000)}\n\n"
            "Write 3-5 thematic sections that together answer the research query. "
            "Use '## Section Title' to start each section."
        )
        messages = build_messages(self._system, [], prompt)
        response = await self._client.chat(messages, model=self.model)

        sections = self._parse_sections(response, state)

        # Save section drafts
        slug = slugify(state.query)
        for i, sec in enumerate(sections):
            await self._writer.run(
                path=f"{slug}/sections/section_{i+1}.md",
                content=f"# {sec.title}\n\n{sec.content}\n",
            )

        log.info(f"[Synthesizer] produced {len(sections)} sections")
        return sections

    @staticmethod
    def _build_notes(state: ResearchState) -> str:
        lines = []
        for s in state.summaries:
            lines.append(f"Source: {s.source_url}")
            lines.append("Points: " + " | ".join(s.main_points[:5]))
            lines.append("Facts: " + " | ".join(s.key_facts[:5]))
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _parse_sections(text: str, state: ResearchState) -> list[ReportSection]:
        import re
        sections: list[ReportSection] = []
        parts = re.split(r"^##\s+", text, flags=re.M)
        sources = list({s.source_url for s in state.summaries})
        for part in parts:
            part = part.strip()
            if not part:
                continue
            lines = part.split("\n", 1)
            title = lines[0].strip()
            content = lines[1].strip() if len(lines) > 1 else ""
            if title and content:
                sections.append(ReportSection(title=title, content=content, sources=sources))
        return sections or [ReportSection(title="Research Findings", content=text, sources=sources)]
