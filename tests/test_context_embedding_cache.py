"""Tests for ContextEmbeddingCache and precomputation integration."""

import time

import pytest

from app.memory.context_embedding_cache import (
    ContextEmbeddingCache,
    get_context_cache,
    init_context_cache,
)


class TestContextEmbeddingCache:
    def test_put_and_get(self):
        cache = ContextEmbeddingCache(ttl_seconds=60.0)
        vec = [0.1, 0.2, 0.3]
        cache.put("conv1", "model-a", vec)
        result = cache.get("conv1")
        assert result == vec

    def test_miss_on_unknown_conversation(self):
        cache = ContextEmbeddingCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        cache = ContextEmbeddingCache(ttl_seconds=0.01)
        cache.put("conv1", "model-a", [1.0])
        assert cache.get("conv1") == [1.0]
        time.sleep(0.02)
        assert cache.get("conv1") is None

    def test_put_overwrites_previous(self):
        cache = ContextEmbeddingCache(ttl_seconds=60.0)
        cache.put("conv1", "model-a", [1.0, 2.0])
        cache.put("conv1", "model-b", [3.0, 4.0])
        assert cache.get("conv1") == [3.0, 4.0]

    def test_returns_copy_not_reference(self):
        cache = ContextEmbeddingCache()
        vec = [1.0, 2.0]
        cache.put("conv1", "m", vec)
        result = cache.get("conv1")
        result[0] = 99.0
        assert cache.get("conv1") == [1.0, 2.0]

    def test_clear(self):
        cache = ContextEmbeddingCache()
        cache.put("a", "m", [1.0])
        cache.put("b", "m", [2.0])
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_conversations_are_independent(self):
        cache = ContextEmbeddingCache()
        cache.put("conv1", "m", [1.0])
        cache.put("conv2", "m", [2.0])
        assert cache.get("conv1") == [1.0]
        assert cache.get("conv2") == [2.0]


class TestSingleton:
    def test_get_context_cache_returns_same_instance(self):
        init_context_cache(ttl_seconds=60.0)
        c1 = get_context_cache()
        c2 = get_context_cache()
        assert c1 is c2


class TestResolveQueryVectorPriority:
    @pytest.mark.asyncio
    async def test_context_cache_hit_skips_embedding(self):
        """When context cache has a precomputed embedding, use it directly."""
        from app.memory.card_retriever import _resolve_query_vector
        from app.memory.query_builder import RetrievalQuery
        from app.memory.context_embedding_cache import init_context_cache, get_context_cache
        from app.providers.embedding_dummy import DummyEmbeddingProvider

        init_context_cache(ttl_seconds=60.0)
        ctx_cache = get_context_cache()
        ctx_cache.put("conv1", "dummy", [7.0, 7.0])

        ep = DummyEmbeddingProvider(dimension=8)
        query = RetrievalQuery(
            query_text="test query",
            latest_user_text="test",
            recent_context_text="test",
            scope_filter={"user_id": "u1"},
        )

        vector = await _resolve_query_vector(ep, query, "conv1")
        assert vector == [7.0, 7.0]

    @pytest.mark.asyncio
    async def test_falls_back_to_retrieval_cache_when_no_context_cache(self):
        """When context cache misses, try retrieval cache."""
        from app.memory.card_retriever import _resolve_query_vector
        from app.memory.query_builder import RetrievalQuery
        from app.memory.retrieval_embedding_cache import init_retrieval_cache, get_retrieval_cache
        from app.memory.context_embedding_cache import init_context_cache
        from app.providers.embedding_dummy import DummyEmbeddingProvider

        init_context_cache(ttl_seconds=60.0)
        init_retrieval_cache(ttl_seconds=60.0, capacity=16)
        ret_cache = get_retrieval_cache()
        ret_cache.put("dummy", "test query", [3.0, 3.0])

        ep = DummyEmbeddingProvider(dimension=8)
        query = RetrievalQuery(
            query_text="test query",
            latest_user_text="test",
            recent_context_text="test",
            scope_filter={"user_id": "u1"},
        )

        vector = await _resolve_query_vector(ep, query, "conv_no_cache")
        assert vector == [3.0, 3.0]

    @pytest.mark.asyncio
    async def test_falls_back_to_live_embedding(self):
        """When both caches miss, call the embedding API."""
        from app.memory.card_retriever import _resolve_query_vector
        from app.memory.query_builder import RetrievalQuery
        from app.memory.retrieval_embedding_cache import init_retrieval_cache
        from app.memory.context_embedding_cache import init_context_cache
        from app.providers.embedding_dummy import DummyEmbeddingProvider

        init_context_cache(ttl_seconds=60.0)
        init_retrieval_cache(ttl_seconds=60.0, capacity=16)

        ep = DummyEmbeddingProvider(dimension=8)
        query = RetrievalQuery(
            query_text="new unique query",
            latest_user_text="new",
            recent_context_text="new",
            scope_filter={"user_id": "u1"},
        )

        vector = await _resolve_query_vector(ep, query, "conv_new")
        assert len(vector) == 8


class TestBuildRecentContextText:
    def test_builds_context_from_messages_and_assistant_response(self):
        from app.pipeline.chat import _build_recent_context_text

        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "我叫小明"},
        ]
        context = _build_recent_context_text(messages, "好的小明！", max_turns=3)

        assert "user: 你好" in context
        assert "assistant: 你好！" in context
        assert "user: 我叫小明" in context
        assert "assistant: 好的小明！" in context

    def test_truncates_long_messages(self):
        from app.pipeline.chat import _build_recent_context_text

        long_msg = "x" * 300
        messages = [{"role": "user", "content": long_msg}]
        context = _build_recent_context_text(messages, "ok", max_turns=1)

        # Each message truncated to 120 chars
        for line in context.split("\n"):
            content_part = line.split(": ", 1)[1] if ": " in line else line
            assert len(content_part) <= 120
