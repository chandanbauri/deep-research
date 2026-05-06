from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable


class EventBus:
    """Lightweight async pub/sub event bus.

    Why: Decouples pipeline stages. Agents emit events (e.g. "iteration_complete")
    without knowing who listens. The CLI uses this for live progress display.
    No external dependencies — pure asyncio.
    """

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Callable) -> None:
        self._listeners[event_type].append(callback)

    async def emit(self, event_type: str, payload: Any = None) -> None:
        for callback in self._listeners.get(event_type, []):
            if asyncio.iscoroutinefunction(callback):
                await callback(payload)
            else:
                callback(payload)
