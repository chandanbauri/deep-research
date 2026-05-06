# Tools & Libraries — Deep Research Agent System

A complete reference for every library and standard module used in this project: what it does, where it is used, and why it was chosen. LangChain and LangGraph are now actively used — their roles, integration points, and design rationale are documented in detail.

---

## Table of Contents

1. [LangChain & LangGraph](#1-langchain--langgraph)
2. [Other External Libraries](#2-other-external-libraries)
3. [Python Standard Library Modules](#3-python-standard-library-modules)
4. [Quick Reference Table](#4-quick-reference-table)

---

## 1. LangChain & LangGraph

### `langchain-ollama` (ChatOllama)
- **Version:** `>=1.1.0`
- **Installed via:** `pip install langchain-ollama`
- **Import:** `from langchain_ollama import ChatOllama`
- **What it is:** LangChain's official integration with the Ollama local model server. `ChatOllama` is a LangChain `Runnable` — the standard LangChain interface for chat models.
- **Purpose in this project:** Powers every LLM call across all six agents. `OllamaClient` (`llm/client.py`) wraps `ChatOllama` and exposes a simple `chat(messages, model)` interface used uniformly by every agent.
- **Where it is used:** `deep_research/llm/client.py` — `OllamaClient._llm(model)` instantiates a `ChatOllama` object. `await llm.ainvoke(messages)` is called inside `OllamaClient.chat()`. `llm.astream(messages)` is used by `stream_chat()`.
- **Why LangChain ChatOllama over raw `ollama.AsyncClient`:**
  - **LangChain Runnable interface** — `ChatOllama` implements `.ainvoke()` and `.astream()` which are the standard async LangChain interfaces, meaning the LLM layer is compatible with LangGraph nodes out of the box.
  - **Typed message objects** — `ChatOllama` accepts `SystemMessage`, `HumanMessage`, `AIMessage` from `langchain_core.messages`. This is a richer, validated format compared to raw dicts.
  - **Provider portability** — swapping `ChatOllama` for `ChatAnthropic`, `ChatOpenAI`, or any other LangChain chat model requires changing one line in `client.py`. All agent code stays identical.
  - **Consistent error handling** — LangChain wraps underlying HTTP errors in a consistent exception hierarchy, making retry logic simpler.

**How the message conversion works:**

```
Agent calls:  client.chat([{"role": "system", "content": "..."}, ...], model="qwen2.5-coder:3b")
                          ↓
OllamaClient._to_lc_messages() converts to:
              [SystemMessage("..."), HumanMessage("..."), ...]
                          ↓
ChatOllama.ainvoke([SystemMessage, HumanMessage, ...])
                          ↓
Returns: AIMessage(content="LLM response text")
                          ↓
OllamaClient.chat() returns: response.content  (plain string)
```

---

### `langgraph`
- **Version:** `>=0.3.0`
- **Installed via:** `pip install langgraph`
- **Import:** `from langgraph.graph import StateGraph, START, END`
- **What it is:** A library for building stateful, graph-structured agent workflows. Agents are nodes, transitions are edges, and the shared state flows through the graph.
- **Purpose in this project:** Powers the entire pipeline orchestration in `ResearchPipeline` (`pipeline/runner.py`). The custom while-loop has been replaced with a LangGraph `StateGraph` where each processing step is a node and the critique feedback loop is an explicit conditional edge.
- **Where it is used:** `deep_research/pipeline/runner.py` — the entire `_build_graph()` function and `GraphState` TypedDict.

**The graph structure:**

```
START
  │
  ▼
plan ──────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  ▼                                                                        │
research ──(parallel asyncio.gather per sub-question)──────────────────── │
  │                                                                        │
  ▼                                                                        │
summarize ──(parallel asyncio.gather per page)─────────────────────────── │
  │                                                                        │
  ▼                                                                        │
critique                                                                   │
  │                                                                        │
  ├──(verdict="sufficient" OR iteration >= max)──→ synthesize ──→ write ──→ END
  │
  └──(verdict="needs_more_research")──→ replan ──→ research (loop back)
```

**Why LangGraph's `StateGraph` over a custom `while` loop:**
  - **Declarative control flow** — `add_edge()` and `add_conditional_edges()` make the pipeline structure immediately visible. The critique → replan → research feedback loop is a first-class graph concept, not an implicit Python control structure.
  - **`Annotated[list, operator.add]` reducers** — `raw_results` and `summaries` are declared with accumulator reducers. Each node returns only new items; LangGraph appends them to the running list automatically. No manual `state.extend()` needed.
  - **State isolation** — each node receives the full current state and returns only the fields it modifies. Nodes cannot accidentally overwrite unrelated state fields.
  - **Visualisable** — `pipeline._app.get_graph().draw_mermaid()` generates a Mermaid diagram of the exact graph at runtime.
  - **Extensible** — adding a new agent step is `graph.add_node("validate", validate_node)` plus two edge calls. No refactoring of the existing loop body.

**How `GraphState` works:**

```python
class GraphState(TypedDict):
    query: str
    plan: list[str]                                    # overwritten each iteration
    pending_raw: list[dict]                            # batch for summarise, cleared after
    raw_results: Annotated[list[dict], operator.add]   # accumulated via operator.add
    summaries:   Annotated[list[dict], operator.add]   # accumulated via operator.add
    critique: dict | None                              # overwritten by critique node
    synthesis_sections: list[dict]                     # written once by synthesize node
    final_report: dict | None                          # written once by write node
    iteration_count: int                               # incremented by research node
    max_iterations: int                                # constant
    visited_urls: list[str]                            # overwritten with full set
    verdict: str                                       # "sufficient" | "needs_more_research"
```

**The conditional edge routing function:**
```python
def route_after_critique(state: GraphState) -> str:
    if state["verdict"] == "sufficient" or state["iteration_count"] >= state["max_iterations"]:
        return "synthesize"
    return "replan"
```

---

### `langchain-core`
- **Version:** `>=1.3.3` (installed as a dependency of `langchain-ollama`)
- **Import:** `from langchain_core.messages import SystemMessage, HumanMessage, AIMessage`
- **What it is:** The foundational message types and Runnable interface shared across all LangChain integrations.
- **Purpose in this project:** `OllamaClient._to_lc_messages()` converts agent message dicts into typed `SystemMessage`, `HumanMessage`, and `AIMessage` objects before passing them to `ChatOllama`.
- **Where it is used:** `deep_research/llm/client.py` — imported in `_to_lc_messages()`.

---

## 2. Other External Libraries

### `duckduckgo-search`
- **Version:** `>=6.3.7`
- **What it is:** Pure-Python DuckDuckGo search client — no API key or account needed.
- **Purpose:** Powers `WebSearchTool`. The Researcher calls this for every sub-question to get a ranked list of URLs and snippets.
- **Where used:** `deep_research/tools/web_search.py` — `DDGS().text(query, max_results=N)` run via `asyncio.to_thread()` (the library is synchronous).
- **Why:** Completely free, no API key, actively maintained. Alternatives (Tavily, SerpAPI, Bing) all require paid keys, which conflicts with the fully-local design goal.

---

### `httpx`
- **Version:** `>=0.27.0` (pre-installed)
- **What it is:** Async-native HTTP client — the async successor to `requests`.
- **Purpose:** Powers `WebScraperTool`. After `WebSearchTool` returns URLs, `httpx.AsyncClient` fetches the full HTML of each page.
- **Where used:** `deep_research/tools/web_scraper.py` — `async with httpx.AsyncClient(...) as client`.
- **Why:** Async-native means `await client.get(url)` doesn't block the event loop, so the Researcher can scrape multiple URLs via `asyncio.gather()` concurrently. `requests` would block; `aiohttp` has a more complex API.

---

### `beautifulsoup4`
- **Version:** `>=4.12.3`
- **What it is:** HTML and XML document parser and navigator.
- **Purpose:** Inside `WebScraperTool`, strips scripts, navigation, and footer tags from raw HTML, then extracts text from `<p>`, `<h1>`–`<h4>`, and `<li>` elements.
- **Where used:** `deep_research/tools/web_scraper.py` — `BeautifulSoup(response.text, "lxml")`.
- **Why:** Industry standard for HTML extraction. Handles malformed HTML that would break `html.parser`.

---

### `lxml`
- **Version:** `>=5.2.0`
- **What it is:** Fast C-based XML/HTML parser. Used as the BeautifulSoup backend.
- **Purpose:** Speeds up page parsing 5–10× vs the pure-Python fallback. Since scraping runs on every page for every sub-question, parsing speed directly affects total pipeline time.
- **Where used:** Implicitly via `BeautifulSoup(html, "lxml")` in `web_scraper.py`.

---

### `pydantic`
- **Version:** `>=2.9.0` (pre-installed)
- **What it is:** Data validation using Python type annotations.
- **Purpose:** All shared data objects — `ResearchState`, `RawResult`, `Summary`, `CritiqueResult`, `FinalReport`, `AgentMessage` — are Pydantic models. `.model_dump()` serialises them to dicts for the LangGraph state; `Model(**dict)` reconstructs them inside node functions.
- **Where used:** `deep_research/models/` — all three model files.
- **Why:** Automatic type validation catches bad LLM output (e.g., `quality_score` outside 1–10) immediately. `.model_dump(mode="json")` handles serialisation to LangGraph's dict-based state.

---

### `click`
- **Version:** `>=8.1.7` (pre-installed)
- **What it is:** Decorator-based CLI framework.
- **Purpose:** Parses `QUERY`, `--depth`, and `--model` arguments in `main.py`. Auto-generates `--help` output.
- **Where used:** `main.py`.

---

### `rich`
- **Version:** `>=13.8.0` (pre-installed)
- **What it is:** Terminal formatting library.
- **Purpose:**
  1. **Logging** (`utils/logger.py`) — `RichHandler` colours log output by severity.
  2. **Progress display** (`pipeline/runner.py`) — `Progress` spinner shows the active LangGraph node; `Panel` renders the Critic's score after each iteration.
- **Where used:** `utils/logger.py`, `pipeline/runner.py`, `main.py`.

---

### `python-dotenv`
- **Version:** `>=1.0.1` (pre-installed)
- **What it is:** Loads `.env` files into `os.environ` at startup.
- **Purpose:** Lets users configure `OLLAMA_HOST`, `DEFAULT_MODEL`, `OUTPUT_DIR` in a `.env` file without shell exports.
- **Where used:** `deep_research/config.py` — `load_dotenv()` at module import time.

---

### `pytest` / `pytest-asyncio` / `pytest-mock`
- **Purpose:** Test runner, async test support, and mock fixtures. All 13 tests use `@pytest.mark.asyncio` and mock LLM + HTTP calls so the suite runs in under 1 second with no external dependencies.
- **Where used:** All files under `tests/`.

---

## 3. Python Standard Library Modules

| Module | Where Used | Purpose |
|--------|-----------|---------|
| `asyncio` | `runner.py`, `researcher.py`, `tools/web_search.py`, `tools/file_*.py`, `utils/retry.py`, `main.py` | `gather()` for parallel scraping; `to_thread()` for sync blocking calls; `run()` to start the event loop |
| `pathlib` | `tools/file_*.py`, `pipeline/state_manager.py`, `config.py` | Typed file paths; `resolve()` for path-traversal safety; `mkdir(parents=True)` |
| `json` | `agents/base_agent.py`, `agents/*.py`, `pipeline/state_manager.py` | Parse LLM tool-call JSON blocks; serialize/deserialize checkpoint files |
| `re` | `agents/base_agent.py`, `agents/*.py`, `tools/text_chunker.py`, `utils/text_utils.py` | Extract JSON blocks from LLM output; sentence splitting; HTML stripping |
| `hashlib` | `tools/deduplicator.py` | MD5 fingerprints for near-duplicate text detection |
| `abc` | `tools/base.py` | `BaseTool` abstract contract — enforces that every tool implements `run()` |
| `dataclasses` | `tools/base.py` | Lightweight `ToolResult` container (no Pydantic validation needed for internal use) |
| `operator` | `pipeline/runner.py` | `operator.add` as LangGraph state reducer for accumulated lists |
| `typing` | Throughout | `TypedDict` for `GraphState`; `Annotated` for LangGraph reducers; `AsyncIterator` for streaming |
| `logging` | `utils/logger.py` | Standard logging configured with `RichHandler` per module |
| `os` | `config.py` | `os.getenv()` reads environment variables |
| `functools` | `utils/retry.py` | `@functools.wraps` preserves function metadata in the retry decorator |
| `random` | `utils/retry.py` | `uniform(0, 0.5)` jitter prevents thundering-herd retries |
| `collections` | `pipeline/event_bus.py` | `defaultdict(list)` for the subscriber map |

---

## 4. Quick Reference Table

| Library | Used? | Role |
|---------|-------|------|
| `langchain-ollama` (ChatOllama) | ✅ **Yes** | LLM engine for all six agents — replaces raw `ollama.AsyncClient` |
| `langgraph` (StateGraph) | ✅ **Yes** | Pipeline orchestration — nodes, edges, conditional feedback loop |
| `langchain-core` (messages) | ✅ **Yes** | `SystemMessage`, `HumanMessage`, `AIMessage` for ChatOllama |
| `duckduckgo-search` | ✅ Yes | Free web search in `WebSearchTool` |
| `httpx` | ✅ Yes | Async page fetching in `WebScraperTool` |
| `beautifulsoup4` | ✅ Yes | HTML text extraction |
| `lxml` | ✅ Yes | Fast BS4 parser backend |
| `pydantic` | ✅ Yes | Type-validated models; serialisation to/from LangGraph state |
| `click` | ✅ Yes | CLI argument parsing |
| `rich` | ✅ Yes | Coloured logs, progress spinner, critic panel |
| `python-dotenv` | ✅ Yes | Load `.env` config at startup |
| `pytest` + plugins | ✅ Yes | Test suite (13 tests, <1 second) |
| `ollama` | ✅ transitive | Pulled in by `langchain-ollama`; no longer called directly |
| `langchain` (base) | ✅ transitive | Pulled in as dependency; no direct imports |
| `langchain-google-genai` | ❌ Not used | Requires Google API key; project is fully local |
