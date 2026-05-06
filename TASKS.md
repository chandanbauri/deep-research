# Deep Research — Implementation Checklist

Track progress by changing `[ ]` to `[x]` as each task is completed.

---

## PHASE 0 — Environment Setup
- [x] 0.1  Create full directory skeleton + all `__init__.py` files
- [x] 0.2  Write `requirements.txt` (new deps only; already-installed packages annotated)
- [x] 0.3  Write `.env.example` (OLLAMA_HOST, DEFAULT_MODEL, OUTPUT_DIR)
- [x] 0.4  Write `TASKS.md` (this checklist)

---

## PHASE 1 — Data Models
- [x] 1.1  `deep_research/models/research_state.py` — `ResearchState` Pydantic model
           Fields: query, plan, raw_results, summaries, critique, synthesis_sections,
           final_report, iteration_count, max_iterations
- [x] 1.2  `deep_research/models/messages.py` — `AgentMessage`, `ToolCall`, `ToolResult`
- [x] 1.3  `deep_research/models/report.py` — `ReportSection`, `FinalReport` + `.to_markdown()`

---

## PHASE 2 — LLM Layer
- [x] 2.1  `deep_research/llm/client.py` — `OllamaClient` (async chat + stream_chat)
- [x] 2.2  `deep_research/llm/prompts.py` — `format_system_prompt`, `build_messages`, `estimate_tokens`

---

## PHASE 3 — Tools

Each tool is a self-contained class inheriting `BaseTool`. All methods are `async`.

- [x] 3.1  `deep_research/tools/base.py` — `BaseTool` (abstract) + `ToolResult` dataclass
- [x] 3.2  `deep_research/tools/web_search.py` — `WebSearchTool`
           Library: `duckduckgo-search` | Input: query, max_results | Output: list of {title,url,snippet}
- [x] 3.3  `deep_research/tools/web_scraper.py` — `WebScraperTool`
           Libraries: `httpx` + `beautifulsoup4` + `lxml` | Input: url | Output: {url,title,text,char_count}
- [x] 3.4  `deep_research/tools/file_reader.py` — `FileReaderTool`
           Library: stdlib `pathlib` | Input: path (within outputs/) | Output: {path,content,lines}
- [x] 3.5  `deep_research/tools/file_writer.py` — `FileWriterTool`
           Library: stdlib `pathlib` | Input: path, content, mode | Output: {path,bytes_written}
           Security: validates path stays within outputs/ via Path.resolve()
- [x] 3.6  `deep_research/tools/text_chunker.py` — `TextChunkerTool`
           Library: stdlib `re` | Input: text, chunk_size, overlap | Output: list[str]
- [x] 3.7  `deep_research/tools/deduplicator.py` — `DeduplicatorTool`
           Library: stdlib `hashlib` | Input: items, mode (url|text) | Output: deduplicated list
- [x] 3.8  `deep_research/tools/__init__.py` — `ToolRegistry`
           Methods: `get_tool(name)`, `list_tools()`, `describe_tools()` (for system prompt injection)

---

## PHASE 4 — Agents

All agents inherit `BaseAgent`. Tool calls are expressed as JSON blocks in LLM output.

- [x] 4.1  `deep_research/agents/base_agent.py` — `BaseAgent` think → parse → act loop
           MAX_TOOL_ITERATIONS = 5 per invocation to prevent runaway loops
- [x] 4.2  `deep_research/agents/orchestrator.py` — `OrchestratorAgent`
           Model: qwen2.5-coder:3b | Tools: none | Methods: plan(), decide_next_step()
- [x] 4.3  `deep_research/agents/researcher.py` — `ResearcherAgent`
           Model: deepseek-r1:1.5b | Tools: web_search, web_scraper, deduplicator
- [x] 4.4  `deep_research/agents/summarizer.py` — `SummarizerAgent`
           Model: qwen2.5-coder:1.5b | Tools: text_chunker, file_writer
- [x] 4.5  `deep_research/agents/critic.py` — `CriticAgent`
           Model: qwen2.5-coder:3b | Tools: none | Output: CritiqueResult
- [x] 4.6  `deep_research/agents/synthesizer.py` — `SynthesizerAgent`
           Model: qwen2.5-coder:3b | Tools: file_reader, file_writer
- [x] 4.7  `deep_research/agents/writer.py` — `WriterAgent`
           Model: qwen2.5-coder:3b | Tools: file_writer
- [x] 4.8  `deep_research/agents/__init__.py` — `AgentFactory.build_agent(role)`

---

## PHASE 5 — Pipeline

- [x] 5.1  `deep_research/pipeline/event_bus.py` — `EventBus` (subscribe/emit, no external deps)
- [x] 5.2  `deep_research/pipeline/state_manager.py` — JSON checkpoint save/load
- [x] 5.3  `deep_research/pipeline/runner.py` — `ResearchPipeline.run(query) -> FinalReport`
           Main loop: plan → [research+summarize]×N → critique → (loop or continue) → synthesize → write

---

## PHASE 6 — Utilities & Config

- [x] 6.1  `deep_research/utils/logger.py` — `rich`-based coloured structured logger
- [x] 6.2  `deep_research/utils/retry.py` — `@retry` exponential backoff decorator
- [x] 6.3  `deep_research/utils/text_utils.py` — `strip_html`, `truncate_to_tokens`, `slugify`
- [x] 6.4  `deep_research/config.py` — `Settings` class (reads from .env)

---

## PHASE 7 — CLI & Docs

- [x] 7.1  `main.py` — `click` CLI: `python main.py "topic" [--model] [--depth 1-3]`
- [x] 7.2  `README.md` — Setup, usage examples, architecture overview, model table

---

## PHASE 8 — Tests

- [x] 8.1  `tests/conftest.py` — shared fixtures (mock Ollama client, mock HTTP responses)
- [x] 8.2  `tests/test_tools/test_web_search.py` — WebSearchTool unit tests
- [x] 8.3  `tests/test_tools/test_web_scraper.py` — WebScraperTool unit tests
- [x] 8.4  `tests/test_tools/test_file_io.py` — FileReaderTool + FileWriterTool unit tests
- [x] 8.5  `tests/test_agents/test_base_agent.py` — BaseAgent think-act loop tests
- [x] 8.6  `tests/test_agents/test_critic.py` — CriticAgent output parsing tests
- [x] 8.7  `tests/test_pipeline/test_runner.py` — Integration test with fully mocked agents

---

## Tool Catalog Reference

| Tool | File | Library | Purpose | Why This Library |
|------|------|---------|---------|-----------------|
| WebSearchTool | `tools/web_search.py` | `duckduckgo-search` | Retrieve URLs + snippets for a query | Free, no API key, actively maintained |
| WebScraperTool | `tools/web_scraper.py` | `httpx` + `beautifulsoup4` + `lxml` | Fetch and clean full page text | httpx already installed; BS4+lxml is the standard |
| FileReaderTool | `tools/file_reader.py` | stdlib `pathlib` | Read saved summaries/drafts | No extra dep; outputs dir only |
| FileWriterTool | `tools/file_writer.py` | stdlib `pathlib` | Persist intermediate work to disk | Path-traversal validated via `Path.resolve()` |
| TextChunkerTool | `tools/text_chunker.py` | stdlib `re` | Split long pages to fit LLM context | Pages can exceed 100k chars; LLM context ~32k tokens |
| DeduplicatorTool | `tools/deduplicator.py` | stdlib `hashlib` | Remove duplicate URLs and passages | Same URL often returned by multiple queries |

---

## Agent Roster Reference

| Agent | Model | Tools | Role |
|-------|-------|-------|------|
| OrchestratorAgent | qwen2.5-coder:3b | none | Plan sub-questions; decide when done |
| ResearcherAgent | deepseek-r1:1.5b | web_search, web_scraper, deduplicator | Gather raw content |
| SummarizerAgent | qwen2.5-coder:1.5b | text_chunker, file_writer | Extract structured notes |
| CriticAgent | qwen2.5-coder:3b | none | Score quality; identify gaps |
| SynthesizerAgent | qwen2.5-coder:3b | file_reader, file_writer | Group notes into sections |
| WriterAgent | qwen2.5-coder:3b | file_writer | Produce final Markdown report |
