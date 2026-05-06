from __future__ import annotations

from pydantic import BaseModel


class ReportSection(BaseModel):
    title: str
    content: str
    sources: list[str] = []


class FinalReport(BaseModel):
    title: str
    query: str
    executive_summary: str
    sections: list[ReportSection]
    conclusion: str
    sources: list[str]

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"# {self.title}",
            "",
            "## Executive Summary",
            "",
            self.executive_summary,
            "",
        ]
        for section in self.sections:
            lines += [f"## {section.title}", "", section.content, ""]

        lines += ["## Conclusion", "", self.conclusion, "", "## Sources", ""]
        for i, src in enumerate(self.sources, 1):
            lines.append(f"{i}. {src}")
        return "\n".join(lines)
