from __future__ import annotations

from deep_research.agents.critic import CriticAgent
from deep_research.agents.orchestrator import OrchestratorAgent
from deep_research.agents.researcher import ResearcherAgent
from deep_research.agents.summarizer import SummarizerAgent
from deep_research.agents.synthesizer import SynthesizerAgent
from deep_research.agents.writer import WriterAgent
from deep_research.llm.client import OllamaClient


def build_agents(llm_client: OllamaClient | None = None) -> dict:
    client = llm_client or OllamaClient()
    return {
        "orchestrator": OrchestratorAgent(llm_client=client),
        "researcher": ResearcherAgent(llm_client=client),
        "summarizer": SummarizerAgent(llm_client=client),
        "critic": CriticAgent(llm_client=client),
        "synthesizer": SynthesizerAgent(llm_client=client),
        "writer": WriterAgent(llm_client=client),
    }
