from __future__ import annotations

from deep_research.config import MODELS
from deep_research.llm.client import OllamaClient
from deep_research.llm.prompts import build_messages, format_system_prompt
from deep_research.models.report import FinalReport, ReportSection
from deep_research.models.research_state import ResearchState
from deep_research.tools import get_tool
from deep_research.utils.logger import get_logger
from deep_research.utils.text_utils import slugify, truncate_to_tokens

log = get_logger(__name__)

_ROLE = """You are a professional research writer producing a formal research report.

Write in an analytical, informative tone. Use clear headings and smooth transitions.
Avoid bullet lists in the executive summary and conclusion — use prose paragraphs.

Your output will be used verbatim in the final report."""


class WriterAgent:
    """Produces the final polished Markdown research report."""

    def __init__(self, llm_client: OllamaClient | None = None) -> None:
        self.name = "Writer"
        self.model = MODELS["writer"]
        self._client = llm_client or OllamaClient()
        self._writer_tool = get_tool("file_writer")
        self._system = format_system_prompt(_ROLE)

    async def write(self, state: ResearchState, sections: list[ReportSection]) -> FinalReport:
        log.info(f"[Writer] writing final report for {state.query!r}")

        section_text = "\n\n".join(f"## {s.title}\n{s.content}" for s in sections)
        sources = list({url for s in state.summaries for url in [s.source_url]})

        # Executive summary
        exec_prompt = (
            f'Research query: "{state.query}"\n\n'
            f"Report sections:\n{truncate_to_tokens(section_text, 3000)}\n\n"
            "Write a 150-200 word executive summary of this research. Prose only, no bullets."
        )
        exec_messages = build_messages(self._system, [], exec_prompt)
        exec_summary = await self._client.chat(exec_messages, model=self.model)

        # Conclusion
        conc_prompt = (
            f'Research query: "{state.query}"\n\n'
            f"Report sections:\n{truncate_to_tokens(section_text, 3000)}\n\n"
            "Write a 100-150 word conclusion with key takeaways. Prose only."
        )
        conc_messages = build_messages(self._system, [], conc_prompt)
        conclusion = await self._client.chat(conc_messages, model=self.model)

        title = f"Research Report: {state.query}"
        report = FinalReport(
            title=title,
            query=state.query,
            executive_summary=exec_summary.strip(),
            sections=sections,
            conclusion=conclusion.strip(),
            sources=sources,
        )

        # Save final report
        slug = slugify(state.query)
        await self._writer_tool.run(
            path=f"{slug}/report.md",
            content=report.to_markdown(),
        )
        log.info(f"[Writer] report saved to outputs/{slug}/report.md")
        return report
