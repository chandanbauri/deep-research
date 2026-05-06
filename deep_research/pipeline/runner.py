from __future__ import annotations

"""
ResearchPipeline — powered by LangGraph
────────────────────────────────────────
Why LangGraph:
  LangGraph models the research pipeline as a directed state graph — nodes are
  processing steps, edges are transitions, and conditional edges handle the
  critique feedback loop. This replaces an imperative while-loop with a
  declarative graph that is:
    - Visualisable: graph.get_graph().draw_mermaid() shows the full flow.
    - Resumable: LangGraph supports checkpointing at every node boundary.
    - Extensible: adding a new agent means adding one node and a few edges,
      not touching the orchestration loop.

Graph structure:
  START → plan → research → summarize → critique ──(sufficient)──→ synthesize → write → END
                    ↑                        │
                    └───── replan ←──────────┘ (needs_more_research AND iterations < max)

State (GraphState TypedDict):
  - raw_results and summaries use Annotated[list, operator.add] so each node
    appends new items rather than overwriting the accumulated list.
  - All other fields are plain overwrites.
"""

import asyncio
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from deep_research.agents import build_agents
from deep_research.config import MAX_ITERATIONS
from deep_research.llm.client import OllamaClient
from deep_research.models.report import FinalReport, ReportSection
from deep_research.models.research_state import (
    CritiqueResult,
    RawResult,
    ResearchState,
    Summary,
)
from deep_research.pipeline.event_bus import EventBus
from deep_research.pipeline.state_manager import StateManager
from deep_research.utils.logger import get_logger

log = get_logger(__name__)
console = Console()


# ── LangGraph shared state ────────────────────────────────────────────────────

class GraphState(TypedDict):
    query: str
    plan: list[str]                                    # current sub-questions
    pending_raw: list[dict]                            # new raw results awaiting summarise
    raw_results: Annotated[list[dict], operator.add]   # accumulated across all iterations
    summaries: Annotated[list[dict], operator.add]     # accumulated across all iterations
    critique: dict | None                              # latest CritiqueResult dict
    synthesis_sections: list[dict]
    final_report: dict | None
    iteration_count: int
    max_iterations: int
    visited_urls: list[str]                            # full set, overwritten each iteration
    verdict: str                                       # routing signal from critique node


# ── Node functions ────────────────────────────────────────────────────────────

def _make_nodes(pipeline: "ResearchPipeline"):
    """Return async node functions as closures over pipeline._agents.

    Closures capture `pipeline` (not a snapshot of _agents), so replacing
    pipeline._agents in tests after construction still works correctly.
    """

    async def plan_node(state: GraphState) -> dict:
        log.info("[graph:plan] planning sub-questions")
        temp = ResearchState(query=state["query"])
        questions = await pipeline._agents["orchestrator"].plan(temp)
        await pipeline.event_bus.emit("plan_ready", questions)
        return {"plan": questions}

    async def research_node(state: GraphState) -> dict:
        log.info(f"[graph:research] iteration {state['iteration_count'] + 1}, {len(state['plan'])} questions")
        temp = ResearchState(
            query=state["query"],
            visited_urls=set(state.get("visited_urls", [])),
        )
        tasks = [
            pipeline._agents["researcher"].research(temp, q)
            for q in state.get("plan", [])
        ]
        results_per_q = await asyncio.gather(*tasks, return_exceptions=True)

        new_raw: list[RawResult] = []
        for res in results_per_q:
            if isinstance(res, Exception):
                log.warning(f"[graph:research] task failed: {res}")
            else:
                new_raw.extend(res)

        await pipeline.event_bus.emit("research_done", {"new_pages": len(new_raw)})
        return {
            "raw_results": [r.model_dump() for r in new_raw],   # appended via operator.add
            "pending_raw": [r.model_dump() for r in new_raw],   # batch for summarise node
            "visited_urls": list(temp.visited_urls),             # full updated set
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    async def summarize_node(state: GraphState) -> dict:
        pending = state.get("pending_raw", [])
        log.info(f"[graph:summarize] {len(pending)} pages")
        temp = ResearchState(query=state["query"])

        tasks = [
            pipeline._agents["summarizer"].summarize(temp, RawResult(**r))
            for r in pending
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_summaries = []
        for r in results:
            if r and not isinstance(r, Exception):
                new_summaries.append(r.model_dump())

        return {
            "summaries": new_summaries,   # appended via operator.add
            "pending_raw": [],            # clear the pending batch
        }

    async def critique_node(state: GraphState) -> dict:
        log.info(f"[graph:critique] evaluating {len(state.get('summaries', []))} summaries")
        temp = ResearchState(
            query=state["query"],
            summaries=[Summary(**s) for s in state.get("summaries", [])],
        )
        critique = await pipeline._agents["critic"].critique(temp)
        await pipeline.event_bus.emit("critique_done", critique)

        console.print(Panel(
            f"Iteration {state['iteration_count']} | Score: {critique.quality_score}/10 | "
            f"Verdict: [bold]{critique.verdict}[/bold] | Gaps: {len(critique.gaps)}",
            title="Critic Report",
            style="yellow",
        ))
        return {"critique": critique.model_dump(), "verdict": critique.verdict}

    async def replan_node(state: GraphState) -> dict:
        log.info("[graph:replan] generating gap-targeting sub-questions")
        critique_data = state.get("critique") or {}
        temp = ResearchState(
            query=state["query"],
            critique=CritiqueResult(**critique_data) if critique_data else None,
        )
        new_questions = await pipeline._agents["orchestrator"].new_sub_questions(temp)
        return {"plan": new_questions or state.get("plan", [])}

    async def synthesize_node(state: GraphState) -> dict:
        log.info("[graph:synthesize] grouping notes into sections")
        temp = ResearchState(
            query=state["query"],
            summaries=[Summary(**s) for s in state.get("summaries", [])],
        )
        sections = await pipeline._agents["synthesizer"].synthesize(temp)
        return {"synthesis_sections": [s.model_dump() for s in sections]}

    async def write_node(state: GraphState) -> dict:
        log.info("[graph:write] producing final report")
        temp = ResearchState(
            query=state["query"],
            summaries=[Summary(**s) for s in state.get("summaries", [])],
        )
        sections = [ReportSection(**s) for s in state.get("synthesis_sections", [])]
        report = await pipeline._agents["writer"].write(temp, sections)
        await pipeline.event_bus.emit("report_ready", report)
        return {"final_report": report.model_dump()}

    def route_after_critique(state: GraphState) -> str:
        """Conditional edge: loop back for more research or proceed to synthesize."""
        verdict = state.get("verdict", "needs_more_research")
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", MAX_ITERATIONS)
        if verdict == "sufficient" or iteration >= max_iter:
            return "synthesize"
        return "replan"

    return (
        plan_node,
        research_node,
        summarize_node,
        critique_node,
        replan_node,
        synthesize_node,
        write_node,
        route_after_critique,
    )


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_graph(pipeline: "ResearchPipeline"):
    """
    Construct and compile the LangGraph StateGraph.

    Why StateGraph over a custom loop:
      - Declarative: the graph edges ARE the control flow — no imperative if/while.
      - add_conditional_edges() makes the critique→replan→research feedback loop
        explicit and visible rather than buried in Python logic.
      - Future extensibility: new agents are new nodes; routing changes are new edges.
    """
    (
        plan_node, research_node, summarize_node, critique_node,
        replan_node, synthesize_node, write_node, route_after_critique,
    ) = _make_nodes(pipeline)

    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("plan",       plan_node)
    graph.add_node("research",   research_node)
    graph.add_node("summarize",  summarize_node)
    graph.add_node("critique",   critique_node)
    graph.add_node("replan",     replan_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("write",      write_node)

    # Linear edges
    graph.add_edge(START,       "plan")
    graph.add_edge("plan",      "research")
    graph.add_edge("research",  "summarize")
    graph.add_edge("summarize", "critique")
    graph.add_edge("replan",    "research")       # feedback loop re-enters research
    graph.add_edge("synthesize","write")
    graph.add_edge("write",     END)

    # Conditional edge: critique decides whether to loop or finish
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {"replan": "replan", "synthesize": "synthesize"},
    )

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

class ResearchPipeline:
    def __init__(
        self,
        max_iterations: int = MAX_ITERATIONS,
        llm_client: OllamaClient | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.event_bus = EventBus()
        self._agents = build_agents(llm_client=llm_client)
        self._state_manager = StateManager()
        self._app = _build_graph(self)   # compile graph once at startup

    async def run(self, query: str) -> FinalReport:
        initial: GraphState = {
            "query": query,
            "plan": [],
            "pending_raw": [],
            "raw_results": [],
            "summaries": [],
            "critique": None,
            "synthesis_sections": [],
            "final_report": None,
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
            "visited_urls": [],
            "verdict": "",
        }

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("[cyan]Starting deep research…", total=None)

            def _on_event(event_type: str):
                # Each label is a lambda so it is only evaluated for its own event type,
                # avoiding AttributeError when payload types differ across events.
                label_fns = {
                    "plan_ready":    lambda p: f"[cyan]Plan ready — {len(p)} sub-questions",
                    "research_done": lambda p: f"[yellow]Scraped {p.get('new_pages', 0)} pages",
                    "critique_done": lambda p: "[magenta]Critique complete",
                    "report_ready":  lambda p: "[green]Report written",
                }
                async def handler(payload):
                    label = label_fns.get(event_type, lambda p: event_type)(payload)
                    progress.update(task, description=label)
                return handler

            for event in ("plan_ready", "research_done", "critique_done", "report_ready"):
                self.event_bus.subscribe(event, _on_event(event))

            # Run the LangGraph graph asynchronously
            final_state: GraphState = await self._app.ainvoke(initial)

        # Persist full state to disk
        report_data = final_state.get("final_report") or {}
        report = FinalReport(**report_data)

        checkpoint_state = ResearchState(
            query=query,
            raw_results=[RawResult(**r) for r in final_state.get("raw_results", [])],
            summaries=[Summary(**s) for s in final_state.get("summaries", [])],
            iteration_count=final_state.get("iteration_count", 0),
        )
        checkpoint_path = await self._state_manager.checkpoint(checkpoint_state)

        console.print(Panel(
            f"Report: [bold green]{checkpoint_path.parent / 'report.md'}[/bold green]",
            title="Research Complete",
            style="green",
        ))
        return report
