from __future__ import annotations

"""
OllamaClient
────────────
LLM abstraction layer powered by LangChain's ChatOllama.

Why LangChain ChatOllama (langchain-ollama):
  - Provides a unified LangChain message interface (SystemMessage, HumanMessage,
    AIMessage) that works identically across every LangChain-compatible model.
  - Exposes ainvoke() and astream() — fully async, no event-loop blocking.
  - Integrates natively with LangGraph nodes, which expect LangChain Runnables.
  - Switching to a different model provider later (e.g., Anthropic, Groq) only
    requires swapping the ChatOllama constructor — all agent code stays the same.

External interface is unchanged: agents still call client.chat(messages, model)
with plain dicts. The conversion to LangChain message objects happens here.
"""

from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from deep_research.config import OLLAMA_HOST
from deep_research.utils.retry import retry


class OllamaConnectionError(Exception):
    pass


class OllamaClient:
    """LangChain-backed async LLM client for Ollama models.

    Accepts the same dict-based message format used throughout the codebase
    and converts it to LangChain message objects before calling ChatOllama.
    """

    def __init__(self, host: str = OLLAMA_HOST) -> None:
        self._host = host

    def _llm(self, model: str) -> ChatOllama:
        # ChatOllama is stateless — create a fresh instance per call.
        # Why: allows per-call model overrides without managing shared state.
        return ChatOllama(model=model, base_url=self._host, num_predict=4096)

    @retry(max_attempts=3, base_delay=2.0, exceptions=(Exception,))
    async def chat(
        self,
        messages: list[dict],
        model: str,
        options: dict | None = None,
    ) -> str:
        lc_messages = self._to_lc_messages(messages)
        try:
            response = await self._llm(model).ainvoke(lc_messages)
            return response.content
        except Exception as exc:
            err = str(exc).lower()
            if "connection" in err or "refused" in err or "connect" in err:
                raise OllamaConnectionError(
                    f"Cannot reach Ollama at {self._host}. Is `ollama serve` running?"
                ) from exc
            raise

    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
    ) -> AsyncIterator[str]:
        lc_messages = self._to_lc_messages(messages)
        try:
            async for chunk in self._llm(model).astream(lc_messages):
                if chunk.content:
                    yield chunk.content
        except Exception as exc:
            err = str(exc).lower()
            if "connection" in err or "refused" in err:
                raise OllamaConnectionError(
                    f"Cannot reach Ollama at {self._host}. Is `ollama serve` running?"
                ) from exc
            raise

    @staticmethod
    def _to_lc_messages(messages: list[dict]) -> list:
        """Convert dict-based messages to LangChain message objects.

        Why: LangChain's ChatOllama.ainvoke() requires typed message objects.
        Keeping this conversion here means agents never import langchain_core directly.
        """
        result = []
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "user":
                result.append(HumanMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
        return result
