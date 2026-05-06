from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Model assignments — heavier reasoning agents use the 3B model;
# repetitive extraction agents use lighter 1.5B models to reduce latency.
MODELS = {
    "orchestrator": os.getenv("ORCHESTRATOR_MODEL", "qwen2.5-coder:3b"),
    "researcher": os.getenv("RESEARCHER_MODEL", "deepseek-r1:1.5b"),
    "summarizer": os.getenv("SUMMARIZER_MODEL", "qwen2.5-coder:1.5b"),
    "critic": os.getenv("CRITIC_MODEL", "qwen2.5-coder:3b"),
    "synthesizer": os.getenv("SYNTHESIZER_MODEL", "qwen2.5-coder:3b"),
    "writer": os.getenv("WRITER_MODEL", "qwen2.5-coder:3b"),
    "default": os.getenv("DEFAULT_MODEL", "qwen2.5-coder:3b"),
}

# Pipeline behaviour
MAX_ITERATIONS = 3          # Max critique-feedback loops before forcing completion
MAX_SEARCH_RESULTS = 5      # URLs returned per DuckDuckGo query
MAX_SCRAPE_URLS = 3         # Pages scraped per sub-question (top-N from search results)
SCRAPE_TIMEOUT = 15         # HTTP timeout in seconds for web scraping
CHUNK_SIZE = 1500           # Characters per text chunk sent to summarizer
CHUNK_OVERLAP = 150         # Overlap between adjacent chunks
LLM_TIMEOUT = 120           # Seconds to wait for an Ollama response
MAX_TOOL_ITERATIONS = 5     # Max tool-call rounds inside a single agent invocation
