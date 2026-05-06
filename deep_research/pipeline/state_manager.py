from __future__ import annotations

import json
from pathlib import Path

from deep_research.config import OUTPUT_DIR
from deep_research.models.research_state import ResearchState
from deep_research.utils.logger import get_logger
from deep_research.utils.text_utils import slugify

log = get_logger(__name__)


class StateManager:
    """Saves and loads ResearchState to/from disk as JSON.

    Why: Enables run inspection, debugging, and future resumability.
    The state file at outputs/{slug}/state.json captures the full pipeline
    state including all raw results, summaries, and critique scores.
    """

    @staticmethod
    def checkpoint_path(query: str) -> Path:
        return OUTPUT_DIR / slugify(query) / "state.json"

    async def checkpoint(self, state: ResearchState) -> Path:
        path = self.checkpoint_path(state.query)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = state.model_dump(mode="json")
        # sets are not JSON-serialisable
        data["visited_urls"] = list(state.visited_urls)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        log.info(f"[StateManager] checkpoint saved → {path}")
        return path

    @staticmethod
    def load(query: str) -> ResearchState | None:
        path = StateManager.checkpoint_path(query)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["visited_urls"] = set(data.get("visited_urls", []))
            return ResearchState(**data)
        except Exception as exc:
            log.warning(f"[StateManager] failed to load state: {exc}")
            return None
