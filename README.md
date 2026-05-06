# Deep Research — Multi-Agent Research System

A team of six AI agents that collaborate to produce a comprehensive, cited research report on any topic. Runs entirely locally using [Ollama](https://ollama.com) — no cloud APIs or API keys required.

---

## Architecture

```
User Query
    │
    ▼
OrchestratorAgent   breaks query into 3-5 focused sub-questions
    │
    ▼
ResearcherAgent     web search + parallel page scraping per sub-question
    │
    ▼
SummarizerAgent     extracts structured notes (main points, key facts) per page
    │
    ▼
CriticAgent         scores quality (1-10), identifies gaps, verdicts: sufficient | needs_more
    │  (loops back if needs_more AND iterations < max)
    ▼
SynthesizerAgent    groups notes into 3-5 thematic sections
    │
    ▼
WriterAgent         produces final polished Markdown report with citations
    │
    ▼
outputs/{slug}/report.md
```

### Agents

| Agent | Model | Tools | Role |
|-------|-------|-------|------|
| OrchestratorAgent | qwen2.5-coder:3b | none | Plan sub-questions; decide when done |
| ResearcherAgent | deepseek-r1:1.5b | web_search, web_scraper, deduplicator | Gather raw content |
| SummarizerAgent | qwen2.5-coder:1.5b | text_chunker, file_writer | Extract structured notes |
| CriticAgent | qwen2.5-coder:3b | none | Score quality; identify gaps |
| SynthesizerAgent | qwen2.5-coder:3b | file_reader, file_writer | Group notes into sections |
| WriterAgent | qwen2.5-coder:3b | file_writer | Produce final Markdown report |

### Tools

| Tool | Library | Purpose |
|------|---------|---------|
| `web_search` | `duckduckgo-search` | Retrieve URLs + snippets — free, no API key |
| `web_scraper` | `httpx` + `beautifulsoup4` + `lxml` | Fetch and clean full page text |
| `file_reader` | stdlib `pathlib` | Read saved summaries/drafts from `outputs/` |
| `file_writer` | stdlib `pathlib` | Persist intermediate work; path-traversal safe |
| `text_chunker` | stdlib `re` | Split long pages to fit LLM context windows |
| `deduplicator` | stdlib `hashlib` | Remove duplicate URLs and near-duplicate passages |

---

## Setup

**Requirements:** Python 3.11+, [Ollama](https://ollama.com) installed and running.

```bash
# 1. Clone and enter the project
cd Deep-Research

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment config
cp .env.example .env
# Edit .env if Ollama runs on a non-default host/port

# 4. Ensure Ollama is running and models are available
ollama serve &          # if not already running
ollama pull qwen2.5-coder:3b
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5-coder:1.5b

# 5. Verify connectivity
python -c "import ollama; print([m.model for m in ollama.list().models])"
```

---

## Usage

```bash
# Quick research (1 iteration)
python main.py "What causes inflation?" --depth 1

# Standard research (3 iterations, default)
python main.py "Environmental impacts of lithium mining" --depth 2

# Thorough research (5 iterations)
python main.py "History and future of quantum computing" --depth 3

# Override all models
python main.py "AI safety risks" --depth 2 --model qwen2.5-coder:3b
```

Output is saved to `outputs/{topic-slug}/`:
- `report.md` — the final research report
- `state.json` — full pipeline state (for debugging/inspection)
- `summaries/` — per-page summary notes
- `sections/` — synthesized section drafts

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
Deep-Research/
├── main.py                  # CLI entry point
├── requirements.txt
├── TASKS.md                 # Implementation checklist
├── deep_research/
│   ├── config.py            # Model assignments, paths, timeouts
│   ├── models/              # Pydantic data models
│   ├── llm/                 # Ollama async client + prompt helpers
│   ├── tools/               # 6 discrete tools (search, scrape, file I/O, etc.)
│   ├── agents/              # 6 specialized agents
│   ├── pipeline/            # Orchestration loop, state management, event bus
│   └── utils/               # Logger, retry decorator, text utilities
├── outputs/                 # Research outputs (git-ignored)
└── tests/                   # Pytest test suite
```

---

## Design Decisions

**Custom BaseAgent, not LangChain** — Full control over tool dispatch, logging, and prompt format. LangChain adds import overhead and opaque abstractions that complicate debugging with local models.

**Shared ResearchState** — All agents read/write one Pydantic object. Simple, debuggable, and easily serialisable to JSON for checkpointing.

**JSON-in-markdown tool calls** — Agents express tool calls as ` ```json {"tool": "...", "args": {...}} ``` ` blocks in their responses. Works for all 4 available Ollama models without requiring native tool-use API support.

**Async throughout** — `asyncio.gather` parallelises URL scraping within each iteration. The Researcher scrapes 3 URLs per sub-question concurrently, cutting wall time by ~60% vs. sequential scraping.

**Model-by-complexity assignment** — The SummarizerAgent runs once per scraped page (15+ times per deep run). Assigning it the 1.5B model cuts cost and latency. Reasoning agents (Critic, Synthesizer, Writer) use the 3B model for better output quality.
