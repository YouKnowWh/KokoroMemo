"""Per-conversation cache for precomputed context embeddings.

After each turn, the recent conversation context embedding is precomputed
in the background and stored here. On the next retrieval, it can be used
directly, skipping the embedding API call.

Keyed by conversation_id. TTL 300s — unused conversations age out.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("kokoromemo.context_embedding_cache")


class ContextEmbeddingCache:
    """Stores one precomputed embedding per conversation, plus gate metadata."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[list[float], str, float, str, int]] = {}
        # conv_id -> (vector, model, cached_at, last_context_text, last_refresh_turn)

    def put(self, conversation_id: str, model: str, vector: list[float],
            context_text: str = "", turn_index: int = 0) -> None:
        self._store[conversation_id] = (list(vector), model, time.monotonic(), context_text, turn_index)

    def get(self, conversation_id: str) -> list[float] | None:
        entry = self._store.get(conversation_id)
        if entry is None:
            return None
        vector, _model, cached_at, _ctx, _turn = entry
        if time.monotonic() - cached_at > self.ttl_seconds:
            del self._store[conversation_id]
            return None
        return list(vector)

    def get_meta(self, conversation_id: str) -> tuple[str | None, int | None]:
        """Return (last_context_text, last_refresh_turn) for gate decisions."""
        entry = self._store.get(conversation_id)
        if entry is None:
            return None, None
        _vector, _model, cached_at, ctx_text, turn = entry
        if time.monotonic() - cached_at > self.ttl_seconds:
            del self._store[conversation_id]
            return None, None
        return ctx_text, turn

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


_context_cache: ContextEmbeddingCache | None = None


def get_context_cache() -> ContextEmbeddingCache:
    global _context_cache
    if _context_cache is None:
        _context_cache = ContextEmbeddingCache()
    return _context_cache


def init_context_cache(ttl_seconds: float = 300.0) -> ContextEmbeddingCache:
    global _context_cache
    _context_cache = ContextEmbeddingCache(ttl_seconds=ttl_seconds)
    return _context_cache
