#!/usr/bin/env python3
"""Benchmark memory retrieval in an isolated KokoroMemo sandbox.

This script never touches the default data/ directory unless you explicitly
point it there. By default it creates a temporary storage root, builds a small
benchmark library, syncs vectors, runs retrieval queries, prints hit metrics
and latency summaries, then removes the sandbox.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from app.core.config import load_config
from app.core.services import ServiceRegistry
from app.memory.card_retriever import retrieve_cards
from app.memory.query_builder import build_retrieval_query
from app.memory.retrieval_embedding_cache import init_retrieval_cache, get_retrieval_cache
from app.memory.context_embedding_cache import init_context_cache, get_context_cache
from app.storage.sqlite_cards import (
    create_memory_library,
    init_cards_db,
    insert_card,
    set_conversation_mounts,
)
from app.storage.vector_sync import sync_card_vector


USER_ID = "bench_user"
CHARACTER_ID = "bench_character"
CONVERSATION_ID = "bench_conversation"


class TimedEmbeddingProvider:
    def __init__(self, inner):
        self._inner = inner
        self.provider_name = inner.provider_name
        self.model = inner.model
        self.dimension = inner.dimension
        self.last_embed_ms = 0.0

    async def embed_text(self, text: str) -> list[float]:
        started_at = time.perf_counter()
        vector = await self._inner.embed_text(text)
        self.last_embed_ms = (time.perf_counter() - started_at) * 1000
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        started_at = time.perf_counter()
        vectors = await self._inner.embed_batch(texts)
        self.last_embed_ms = (time.perf_counter() - started_at) * 1000
        return vectors

    async def health_check(self) -> dict:
        return await self._inner.health_check()


class TimedVectorStore:
    def __init__(self, inner):
        self._inner = inner
        self.last_search_ms = 0.0

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def search(self, query_vector, where=None, top_k=30, select_columns=None):
        started_at = time.perf_counter()
        results = self._inner.search(query_vector, where=where, top_k=top_k, select_columns=select_columns)
        self.last_search_ms = (time.perf_counter() - started_at) * 1000
        return results


@dataclass
class BenchmarkCard:
    card_id: str
    content: str
    card_type: str = "preference"
    importance: float = 0.85
    confidence: float = 0.9
    scope: str = "character"


@dataclass
class BenchmarkQuery:
    name: str
    messages: list[dict]
    expected_card_ids: list[str]


BENCHMARK_CARDS = [
    BenchmarkCard("card_name_pref", "用户希望被称呼为老师。"),
    BenchmarkCard("card_city", "用户目前住在上海。", card_type="fact"),
    BenchmarkCard("card_boundary", "正式场合不要用轻浮玩笑。", card_type="boundary"),
    BenchmarkCard("card_exam", "用户这周正在准备答辩。", card_type="state"),
    BenchmarkCard("card_drink", "用户喜欢无糖美式。", card_type="preference"),
    # Topic-shift test cards
    BenchmarkCard("card_spicy", "用户喜欢吃辣，尤其是川菜。"),
    BenchmarkCard("card_cilantro", "用户不吃香菜。", card_type="boundary"),
    BenchmarkCard("card_company", "用户在字节跳动做后端开发。", card_type="fact"),
    BenchmarkCard("card_lang", "用户主要用Go和Rust编程。", card_type="fact"),
    BenchmarkCard("card_cat", "用户养了一只叫豆包的橘猫。", card_type="fact"),
    BenchmarkCard("card_game", "用户最近在玩黑神话悟空。", card_type="state"),
]


BENCHMARK_QUERIES = [
    BenchmarkQuery(
        name="name_preference",
        messages=[
            {"role": "assistant", "content": "我们继续。"},
            {"role": "user", "content": "你还记得该怎么称呼我吗？"},
        ],
        expected_card_ids=["card_name_pref"],
    ),
    BenchmarkQuery(
        name="city_fact",
        messages=[
            {"role": "assistant", "content": "当然。"},
            {"role": "user", "content": "我住在哪来着？"},
        ],
        expected_card_ids=["card_city"],
    ),
    BenchmarkQuery(
        name="boundary",
        messages=[
            {"role": "assistant", "content": "明白。"},
            {"role": "user", "content": "正式回复时别太油。"},
        ],
        expected_card_ids=["card_boundary"],
    ),
    BenchmarkQuery(
        name="current_state",
        messages=[
            {"role": "assistant", "content": "继续说。"},
            {"role": "user", "content": "你记得我这周在忙什么吗？"},
        ],
        expected_card_ids=["card_exam"],
    ),
    BenchmarkQuery(
        name="drink_preference",
        messages=[
            {"role": "assistant", "content": "点单前确认一下。"},
            {"role": "user", "content": "咖啡还是按我平时喜欢的来。"},
        ],
        expected_card_ids=["card_drink"],
    ),
    # Topic-shift queries: accumulated context + changing topics
    BenchmarkQuery(
        name="ts_turn1_food",
        messages=[
            {"role": "user", "content": "帮我推荐个外卖"},
            {"role": "assistant", "content": "喜欢什么口味？"},
            {"role": "user", "content": "要辣的不要香菜"},
        ],
        expected_card_ids=["card_spicy", "card_cilantro"],
    ),
    BenchmarkQuery(
        name="ts_turn2_work",
        messages=[
            {"role": "user", "content": "帮我推荐个外卖"},
            {"role": "assistant", "content": "喜欢什么口味？"},
            {"role": "user", "content": "要辣的不要香菜"},
            {"role": "assistant", "content": "川菜很适合"},
            {"role": "user", "content": "话说我在哪里工作？"},
        ],
        expected_card_ids=["card_company", "card_lang"],
    ),
    BenchmarkQuery(
        name="ts_turn3_life",
        messages=[
            {"role": "user", "content": "帮我推荐个外卖"},
            {"role": "assistant", "content": "喜欢什么口味？"},
            {"role": "user", "content": "要辣的不要香菜"},
            {"role": "assistant", "content": "川菜很适合"},
            {"role": "user", "content": "我家猫叫什么名字？"},
        ],
        expected_card_ids=["card_cat"],
    ),
    BenchmarkQuery(
        name="ts_turn4_game",
        messages=[
            {"role": "user", "content": "帮我推荐个外卖"},
            {"role": "assistant", "content": "喜欢什么口味？"},
            {"role": "user", "content": "要辣的不要香菜"},
            {"role": "assistant", "content": "川菜很适合"},
            {"role": "user", "content": "有什么好玩的游戏？"},
        ],
        expected_card_ids=["card_game"],
    ),
    BenchmarkQuery(
        name="ts_turn5_food_again",
        messages=[
            {"role": "user", "content": "帮我推荐个外卖"},
            {"role": "assistant", "content": "喜欢什么口味？"},
            {"role": "user", "content": "要辣的不要香菜"},
            {"role": "assistant", "content": "川菜很适合"},
            {"role": "user", "content": "我还是想吃川菜外卖"},
        ],
        expected_card_ids=["card_spicy", "card_cilantro"],
    ),
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * pct)))
    return ordered[index]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark KokoroMemo memory retrieval in an isolated sandbox.")
    parser.add_argument("--config", default=None, help="Optional config.yaml path. Defaults to repository config resolution.")
    parser.add_argument("--root-dir", default=None, help="Optional sandbox root dir. Defaults to a temporary directory.")
    parser.add_argument("--keep", action="store_true", help="Keep the sandbox directory after the run.")
    parser.add_argument("--queries", type=int, default=3, help="How many times to run each benchmark query.")
    parser.add_argument("--top-k", type=int, default=6, help="Final retrieval top-k.")
    parser.add_argument("--vector-top-k", type=int, default=20, help="Vector search top-k.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    parser.add_argument("--cache-ttl-seconds", type=float, default=60.0, help="Retrieval embedding cache TTL in seconds (default 60).")
    parser.add_argument("--cache-capacity", type=int, default=128, help="Retrieval embedding cache capacity (default 128).")
    parser.add_argument("--warmup-runs", type=int, default=0, help="Warmup runs before timed runs (populates cache).")
    parser.add_argument("--repeat-same-query", action="store_true", help="Repeat only the first benchmark query instead of all queries.")
    parser.add_argument("--output-path", default=None, help="Write JSON results to a file.")
    parser.add_argument("--precompute", action="store_true", help="Precompute context embedding before timed runs (simulates post-processing).")
    return parser


async def prepare_sandbox(root_dir: Path, config_path: str | None) -> tuple[object, ServiceRegistry, str]:
    cfg = load_config(config_path)
    cfg.storage.root_dir = str(root_dir)
    cfg.storage.sqlite.app_db = str(root_dir / "app.sqlite")
    cfg.storage.sqlite.memory_db = str(root_dir / "memory.sqlite")
    cfg.storage.lancedb.path = str(root_dir / "vector_indexes" / "benchmark" / "lancedb")

    await init_cards_db(cfg.storage.sqlite.memory_db)

    library_id = await create_memory_library(
        cfg.storage.sqlite.memory_db,
        name="Benchmark Library",
        description="Isolated benchmark memory library.",
        library_id="lib_benchmark",
    )
    await set_conversation_mounts(
        cfg.storage.sqlite.memory_db,
        CONVERSATION_ID,
        [library_id],
        write_library_id=library_id,
        user_id=USER_ID,
        character_id=CHARACTER_ID,
    )

    registry = ServiceRegistry()
    return cfg, registry, library_id


async def seed_cards(cfg, registry: ServiceRegistry, library_id: str) -> list[dict]:
    db_path = cfg.storage.sqlite.memory_db
    embedding_provider = registry.get_embedding_provider(cfg)
    vector_store = registry.get_lancedb_store(cfg)
    seeded: list[dict] = []

    for card in BENCHMARK_CARDS:
        await insert_card(
            db_path,
            card_id=card.card_id,
            library_id=library_id,
            user_id=USER_ID,
            character_id=CHARACTER_ID,
            conversation_id=CONVERSATION_ID,
            scope=card.scope,
            card_type=card.card_type,
            content=card.content,
            importance=card.importance,
            confidence=card.confidence,
            status="approved",
        )
        sync_started_at = time.perf_counter()
        if embedding_provider and vector_store:
            await sync_card_vector(db_path, card.card_id, embedding_provider, vector_store)
        seeded.append(
            {
                "card_id": card.card_id,
                "content": card.content,
                "sync_ms": (time.perf_counter() - sync_started_at) * 1000,
            }
        )
    return seeded


def _snapshot_cache_stats() -> dict:
    cache = get_retrieval_cache()
    return {"hits": cache.stats.hits, "misses": cache.stats.misses, "evictions": cache.stats.evictions, "expirations": cache.stats.expirations}


async def _run_one_query(
    bench_query: BenchmarkQuery,
    cfg,
    embedding_provider,
    vector_store,
    top_k: int,
    vector_top_k: int,
    run_index: int,
    prev_cache_stats: dict | None,
) -> dict:
    build_started_at = time.perf_counter()
    retrieval_query = build_retrieval_query(
        messages=bench_query.messages,
        user_id=USER_ID,
        character_id=CHARACTER_ID,
        conversation_id=CONVERSATION_ID,
        max_recent_turns=cfg.memory.max_recent_turns_for_query,
    )
    query_build_ms = (time.perf_counter() - build_started_at) * 1000
    started_at = time.perf_counter()
    candidates = await retrieve_cards(
        retrieval_query,
        embedding_provider,
        vector_store,
        cfg.storage.sqlite.memory_db,
        vector_top_k=vector_top_k,
        final_top_k=top_k,
    )
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    returned_ids = [candidate.card_id for candidate in candidates]
    matched = [card_id for card_id in bench_query.expected_card_ids if card_id in returned_ids]

    current_stats = _snapshot_cache_stats()
    cache_hit = False
    if prev_cache_stats:
        cache_hit = current_stats["hits"] > prev_cache_stats["hits"]

    return {
        "query": bench_query.name,
        "run": run_index + 1,
        "elapsed_ms": elapsed_ms,
        "query_build_ms": query_build_ms,
        "embedding_ms": embedding_provider.last_embed_ms,
        "vector_search_ms": vector_store.last_search_ms,
        "expected": bench_query.expected_card_ids,
        "returned": returned_ids,
        "matched": matched,
        "hit": bool(matched),
        "cache_hit": cache_hit,
    }


async def run_queries(
    cfg, registry: ServiceRegistry, query_runs: int, top_k: int, vector_top_k: int,
    warmup_runs: int = 0, repeat_same_query: bool = False, precompute: bool = False,
) -> list[dict]:
    raw_embedding_provider = registry.get_embedding_provider(cfg)
    raw_vector_store = registry.get_lancedb_store(cfg)
    embedding_provider = TimedEmbeddingProvider(raw_embedding_provider) if raw_embedding_provider else None
    vector_store = TimedVectorStore(raw_vector_store) if raw_vector_store else None
    if not embedding_provider or not vector_store:
        raise RuntimeError("Embedding provider or vector store unavailable; cannot run retrieval benchmark.")

    queries = BENCHMARK_QUERIES[:1] if repeat_same_query else BENCHMARK_QUERIES

    # Precompute context embedding (simulates _post_process_turn background work)
    if precompute:
        ctx_cache = get_context_cache()
        from app.pipeline.chat import _build_recent_context_text
        # Use the first query's messages as "previous turn" context
        first_query = queries[0]
        context_text = _build_recent_context_text(
            first_query.messages, "", cfg.memory.max_recent_turns_for_query,
        )
        if context_text.strip():
            ctx_vector = await embedding_provider.embed_text(context_text)
            ctx_cache.put(CONVERSATION_ID, embedding_provider.model, ctx_vector)

    # Warmup runs (not timed/recorded)
    for _ in range(warmup_runs):
        for bench_query in queries:
            retrieval_query = build_retrieval_query(
                messages=bench_query.messages,
                user_id=USER_ID,
                character_id=CHARACTER_ID,
                conversation_id=CONVERSATION_ID,
                max_recent_turns=cfg.memory.max_recent_turns_for_query,
            )
            await retrieve_cards(
                retrieval_query, embedding_provider, vector_store,
                cfg.storage.sqlite.memory_db,
                vector_top_k=vector_top_k, final_top_k=top_k,
            )

    results: list[dict] = []
    for bench_query in queries:
        for run_index in range(query_runs):
            prev_stats = _snapshot_cache_stats()
            entry = await _run_one_query(bench_query, cfg, embedding_provider, vector_store, top_k, vector_top_k, run_index, prev_stats)
            results.append(entry)
    return results


def summarize(seed_results: list[dict], query_results: list[dict], root_dir: Path) -> dict:
    sync_times = [item["sync_ms"] for item in seed_results]
    retrieval_times = [item["elapsed_ms"] for item in query_results]
    query_build_times = [item["query_build_ms"] for item in query_results]
    embedding_times = [item["embedding_ms"] for item in query_results]
    vector_search_times = [item["vector_search_ms"] for item in query_results]
    hit_count = sum(1 for item in query_results if item["hit"])
    cache_hit_count = sum(1 for item in query_results if item.get("cache_hit"))
    cache = get_retrieval_cache()
    ctx_cache = get_context_cache()

    per_query: dict[str, dict] = {}
    for item in query_results:
        bucket = per_query.setdefault(
            item["query"],
            {"runs": 0, "hits": 0, "cache_hits": 0, "elapsed_ms": [], "embedding_ms": [], "vector_search_ms": []},
        )
        bucket["runs"] += 1
        bucket["hits"] += 1 if item["hit"] else 0
        bucket["cache_hits"] += 1 if item.get("cache_hit") else 0
        bucket["elapsed_ms"].append(item["elapsed_ms"])
        bucket["embedding_ms"].append(item["embedding_ms"])
        bucket["vector_search_ms"].append(item["vector_search_ms"])

    for name, bucket in per_query.items():
        bucket["hit_rate"] = bucket["hits"] / bucket["runs"] if bucket["runs"] else 0.0
        bucket["cache_hit_rate"] = bucket["cache_hits"] / bucket["runs"] if bucket["runs"] else 0.0
        bucket["avg_ms"] = statistics.fmean(bucket["elapsed_ms"]) if bucket["elapsed_ms"] else 0.0
        bucket["p95_ms"] = percentile(bucket["elapsed_ms"], 0.95)
        bucket["embedding_avg_ms"] = statistics.fmean(bucket["embedding_ms"]) if bucket["embedding_ms"] else 0.0
        bucket["vector_search_avg_ms"] = statistics.fmean(bucket["vector_search_ms"]) if bucket["vector_search_ms"] else 0.0
        del bucket["elapsed_ms"]
        del bucket["embedding_ms"]
        del bucket["vector_search_ms"]

    return {
        "sandbox_root": str(root_dir),
        "seed_card_count": len(seed_results),
        "query_count": len(query_results),
        "retrieval_hit_rate": hit_count / len(query_results) if query_results else 0.0,
        "seed_sync_avg_ms": statistics.fmean(sync_times) if sync_times else 0.0,
        "retrieval_avg_ms": statistics.fmean(retrieval_times) if retrieval_times else 0.0,
        "retrieval_p50_ms": percentile(retrieval_times, 0.50),
        "retrieval_p95_ms": percentile(retrieval_times, 0.95),
        "query_build_avg_ms": statistics.fmean(query_build_times) if query_build_times else 0.0,
        "embedding_avg_ms": statistics.fmean(embedding_times) if embedding_times else 0.0,
        "embedding_p95_ms": percentile(embedding_times, 0.95),
        "vector_search_avg_ms": statistics.fmean(vector_search_times) if vector_search_times else 0.0,
        "cache_hit_rate": cache_hit_count / len(query_results) if query_results else 0.0,
        "cache_stats": {
            "hits": cache.stats.hits,
            "misses": cache.stats.misses,
            "evictions": cache.stats.evictions,
            "expirations": cache.stats.expirations,
        },
        "context_cache_entries": len(ctx_cache),
        "per_query": per_query,
        "seed_results": seed_results,
        "query_results": query_results,
    }


def print_summary(summary: dict) -> None:
    print("Memory benchmark summary")
    print(f"- Sandbox root: {summary['sandbox_root']}")
    print(f"- Seed cards: {summary['seed_card_count']}")
    print(f"- Query runs: {summary['query_count']}")
    print(f"- Retrieval hit rate: {summary['retrieval_hit_rate']:.1%}")
    print(f"- Cache hit rate: {summary['cache_hit_rate']:.1%}")
    print(f"- Seed vector sync avg: {summary['seed_sync_avg_ms']:.1f} ms")
    print(f"- Retrieval avg: {summary['retrieval_avg_ms']:.1f} ms")
    print(f"- Retrieval p50: {summary['retrieval_p50_ms']:.1f} ms")
    print(f"- Retrieval p95: {summary['retrieval_p95_ms']:.1f} ms")
    print(f"- Query build avg: {summary['query_build_avg_ms']:.1f} ms")
    print(f"- Embedding avg: {summary['embedding_avg_ms']:.1f} ms")
    print(f"- Embedding p95: {summary['embedding_p95_ms']:.1f} ms")
    print(f"- Vector search avg: {summary['vector_search_avg_ms']:.1f} ms")
    cs = summary.get("cache_stats", {})
    if cs:
        print(f"- Cache: hits={cs['hits']} misses={cs['misses']} evictions={cs['evictions']} expirations={cs['expirations']}")
    print("")
    print("Per-query")
    for name, bucket in summary["per_query"].items():
        print(
            f"- {name}: hit_rate={bucket['hit_rate']:.1%} "
            f"cache_hit_rate={bucket.get('cache_hit_rate', 0):.1%} "
            f"avg={bucket['avg_ms']:.1f} ms p95={bucket['p95_ms']:.1f} ms "
            f"embed_avg={bucket['embedding_avg_ms']:.1f} ms "
            f"search_avg={bucket['vector_search_avg_ms']:.1f} ms"
        )


async def async_main(args: argparse.Namespace) -> int:
    root_dir = Path(args.root_dir) if args.root_dir else Path(tempfile.mkdtemp(prefix="kokoromemo-memory-bench-"))
    root_dir.mkdir(parents=True, exist_ok=True)

    init_retrieval_cache(ttl_seconds=args.cache_ttl_seconds, capacity=args.cache_capacity)
    init_context_cache(ttl_seconds=args.cache_ttl_seconds)

    try:
        cfg, registry, library_id = await prepare_sandbox(root_dir, args.config)
        seed_results = await seed_cards(cfg, registry, library_id)
        query_results = await run_queries(
            cfg, registry, args.queries, args.top_k, args.vector_top_k,
            warmup_runs=args.warmup_runs, repeat_same_query=args.repeat_same_query,
            precompute=args.precompute,
        )
        summary = summarize(seed_results, query_results, root_dir)
        if args.json:
            output = json.dumps(summary, ensure_ascii=False, indent=2)
            print(output)
            if args.output_path:
                Path(args.output_path).write_text(output, encoding="utf-8")
        else:
            print_summary(summary)
            if args.output_path:
                Path(args.output_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(root_dir, ignore_errors=True)


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
