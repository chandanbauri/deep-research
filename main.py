#!/usr/bin/env python3
"""
Deep Research CLI
─────────────────
Usage:
  python main.py "What are the environmental impacts of lithium mining?" --depth 2

Arguments:
  query       The research topic or question.
  --depth     Research depth: 1=quick (1 iteration), 2=standard (3), 3=thorough (5).
  --model     Override the default Ollama model for all agents.
"""

from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from deep_research.config import MODELS, OUTPUT_DIR
from deep_research.llm.client import OllamaConnectionError, OllamaClient
from deep_research.pipeline.runner import ResearchPipeline
from deep_research.utils.text_utils import slugify

console = Console()

DEPTH_TO_ITERATIONS = {1: 1, 2: 3, 3: 5}


@click.command()
@click.argument("query")
@click.option("--depth", type=click.IntRange(1, 3), default=2, show_default=True, help="Research depth (1=quick, 2=standard, 3=thorough)")
@click.option("--model", default=None, help="Override Ollama model for all agents (e.g. qwen2.5-coder:3b)")
def main(query: str, depth: int, model: str | None) -> None:
    """Run a deep research pipeline on QUERY and produce a Markdown report."""

    if model:
        for key in MODELS:
            MODELS[key] = model

    max_iterations = DEPTH_TO_ITERATIONS[depth]
    console.print(f"\n[bold cyan]Deep Research[/bold cyan]")
    console.print(f"Query   : [yellow]{query}[/yellow]")
    console.print(f"Depth   : {depth} ({max_iterations} iteration(s))")
    console.print(f"Output  : {OUTPUT_DIR / slugify(query)}/\n")

    try:
        asyncio.run(_run(query, max_iterations))
    except OllamaConnectionError as exc:
        console.print(f"[red]Connection error:[/red] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)


async def _run(query: str, max_iterations: int) -> None:
    pipeline = ResearchPipeline(max_iterations=max_iterations)
    report = await pipeline.run(query)
    slug = slugify(query)
    report_path = OUTPUT_DIR / slug / "report.md"
    console.print(f"\n[bold]Report:[/bold] {report_path}")


if __name__ == "__main__":
    main()
