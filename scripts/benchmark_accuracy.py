#!/usr/bin/env python3
"""Compare retrieval accuracy: baseline (live embedding) vs precompute (cached context).

Creates an isolated sandbox, seeds more diverse cards, runs both modes, and
reports per-query recall overlap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
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
    create_memory_library, init_cards_db, insert_card, set_conversation_mounts,
)
from app.storage.vector_sync import sync_card_vector

USER_ID = "acc_user"
CHARACTER_ID = "acc_char"
CONVERSATION_ID = "acc_conv"


@dataclass
class Card:
    card_id: str
    content: str
    scope: str = "character"
    card_type: str = "preference"
    importance: float = 0.85
    confidence: float = 0.9


@dataclass
class TestQuery:
    name: str
    messages: list[dict]
    expected_ids: list[str]


# More diverse card set
CARDS = [
    Card("card_food", "用户不吃香菜和芹菜。", card_type="boundary"),
    Card("card_drink", "用户喜欢无糖冰美式咖啡。"),
    Card("card_name", "用户叫张伟，英文名David。", card_type="fact"),
    Card("card_city", "用户住在上海浦东新区。", card_type="fact"),
    Card("card_job", "用户是一名后端工程师，主要用Go和Rust。", card_type="fact"),
    Card("card_company", "用户在字节跳动工作。", card_type="fact"),
    Card("card_hobby", "用户周末喜欢攀岩和徒步。"),
    Card("card_pet", "用户养了一只叫豆包的橘猫。", card_type="fact"),
    Card("card_formal", "正式工作场合不要用网络流行语。", card_type="boundary"),
    Card("card_health", "用户对花生过敏，食物中不能含花生。", card_type="boundary"),
    Card("card_project", "用户最近在做Kubernetes集群迁移项目。", card_type="state"),
    Card("card_team", "用户的团队有5个人，每周一早上站会。", card_type="state"),
    Card("card_tool", "用户偏好用Neovim编辑器，主题是tokyonight。"),
    Card("card_sleep", "用户习惯凌晨1点睡，早上9点起。", card_type="fact"),
    Card("card_learn", "用户最近在学Rust的async编程。", card_type="state"),
]

QUERIES = [
    TestQuery(name="food_restriction", messages=[
        {"role": "user", "content": "帮我点个外卖"},
        {"role": "assistant", "content": "好的，有什么忌口吗？"},
        {"role": "user", "content": "不要香菜和芹菜"},
    ], expected_ids=["card_food"]),

    TestQuery(name="coffee_pref", messages=[
        {"role": "assistant", "content": "旁边有家咖啡店"},
        {"role": "user", "content": "帮我买杯咖啡，老样子"},
    ], expected_ids=["card_drink"]),

    TestQuery(name="name_lookup", messages=[
        {"role": "assistant", "content": "收到你的邮件了"},
        {"role": "user", "content": "我的英文名是什么来着？"},
    ], expected_ids=["card_name"]),

    TestQuery(name="allergy_check", messages=[
        {"role": "user", "content": "我们去吃火锅吧"},
        {"role": "assistant", "content": "好的，你选锅底"},
        {"role": "user", "content": "有没有什么我不能吃的？"},
    ], expected_ids=["card_health", "card_food"]),

    TestQuery(name="work_context", messages=[
        {"role": "assistant", "content": "技术方案写好了"},
        {"role": "user", "content": "我在哪家公司来着？"},
    ], expected_ids=["card_company", "card_job"]),

    TestQuery(name="project_status", messages=[
        {"role": "assistant", "content": "我们继续上次的话题"},
        {"role": "user", "content": "最近工作上在做什么项目？"},
    ], expected_ids=["card_project", "card_job"]),

    TestQuery(name="weekend_plan", messages=[
        {"role": "user", "content": "周末有什么安排？"},
    ], expected_ids=["card_hobby"]),

    TestQuery(name="formal_setting", messages=[
        {"role": "user", "content": "帮我看一下这个方案合不合理"},
        {"role": "assistant", "content": "我来分析"},
        {"role": "user", "content": "写封正式的邮件回复"},
    ], expected_ids=["card_formal", "card_company"]),

    TestQuery(name="pet_name", messages=[
        {"role": "assistant", "content": "你家很温馨"},
        {"role": "user", "content": "还记得我的猫叫什么吗？"},
    ], expected_ids=["card_pet"]),

    TestQuery(name="editor_pref", messages=[
        {"role": "assistant", "content": "新IDE发布了"},
        {"role": "user", "content": "我写代码喜欢用什么工具？"},
    ], expected_ids=["card_tool", "card_job"]),
]


class TimedEmbeddingProvider:
    def __init__(self, inner):
        self._inner = inner
        self.provider_name = inner.provider_name
        self.model = inner.model
        self.dimension = inner.dimension

    async def embed_text(self, text: str) -> list[float]:
        return await self._inner.embed_text(text)

    async def embed_batch(self, texts):
        return await self._inner.embed_batch(texts)

    async def health_check(self):
        return await self._inner.health_check()


async def prepare_sandbox(root_dir: Path, config_path: str | None):
    cfg = load_config(config_path)
    cfg.storage.root_dir = str(root_dir)
    cfg.storage.sqlite.app_db = str(root_dir / "app.sqlite")
    cfg.storage.sqlite.memory_db = str(root_dir / "memory.sqlite")
    cfg.storage.lancedb.path = str(root_dir / "vector_indexes" / "benchmark" / "lancedb")

    await init_cards_db(cfg.storage.sqlite.memory_db)
    lib_id = await create_memory_library(cfg.storage.sqlite.memory_db, name="AccBench", library_id="lib_acc")
    await set_conversation_mounts(cfg.storage.sqlite.memory_db, CONVERSATION_ID, [lib_id],
                                   write_library_id=lib_id, user_id=USER_ID, character_id=CHARACTER_ID)
    registry = ServiceRegistry()
    return cfg, registry, lib_id


async def seed_cards(cfg, registry, lib_id):
    db_path = cfg.storage.sqlite.memory_db
    ep = registry.get_embedding_provider(cfg)
    store = registry.get_lancedb_store(cfg)
    for card in CARDS:
        await insert_card(db_path, card.card_id, lib_id, USER_ID, CHARACTER_ID, CONVERSATION_ID,
                          card.scope, card.card_type, card.content, card.importance, card.confidence, "approved")
        if ep and store:
            await sync_card_vector(db_path, card.card_id, ep, store)


async def run_retrieval(cfg, registry, top_k, vector_top_k):
    ep_raw = registry.get_embedding_provider(cfg)
    store_raw = registry.get_lancedb_store(cfg)
    ep = TimedEmbeddingProvider(ep_raw)
    store = type("TimedStore", (), {
        "__getattr__": lambda s, x: getattr(store_raw, x),
        "search": lambda s, qv, where, top_k, **kw: store_raw.search(qv, where=where, top_k=top_k, **kw),
    })()

    results = {}
    for q in QUERIES:
        rq = build_retrieval_query(q.messages, USER_ID, CHARACTER_ID, CONVERSATION_ID,
                                    max_recent_turns=cfg.memory.max_recent_turns_for_query)
        candidates = await retrieve_cards(rq, ep, store, cfg.storage.sqlite.memory_db,
                                           vector_top_k=vector_top_k, final_top_k=top_k)
        results[q.name] = [c.card_id for c in candidates]
    return results


def compare(name, baseline_ids, precompute_ids):
    baseline_set = set(baseline_ids)
    precompute_set = set(precompute_ids)
    overlap = baseline_set & precompute_set
    only_baseline = baseline_set - precompute_set
    only_precompute = precompute_set - baseline_set
    jaccard = len(overlap) / len(baseline_set | precompute_set) if (baseline_set | precompute_set) else 1.0
    return {
        "query": name,
        "baseline_count": len(baseline_ids),
        "precompute_count": len(precompute_ids),
        "overlap_count": len(overlap),
        "only_baseline": sorted(only_baseline),
        "only_precompute": sorted(only_precompute),
        "jaccard": round(jaccard, 3),
        "identical": baseline_ids == precompute_ids,
    }


async def async_main(args):
    root_dir = Path(args.root_dir) if args.root_dir else Path(tempfile.mkdtemp(prefix="kokoromemo-acc-"))
    root_dir.mkdir(parents=True, exist_ok=True)

    init_retrieval_cache(ttl_seconds=0, capacity=1)  # disable exact-match cache
    init_context_cache(ttl_seconds=300)

    try:
        cfg, registry, lib_id = await prepare_sandbox(root_dir, args.config)
        await seed_cards(cfg, registry, lib_id)

        # --- Baseline: no precompute ---
        ctx_cache = get_context_cache()
        ctx_cache.clear()
        baseline = await run_retrieval(cfg, registry, args.top_k, args.vector_top_k)

        # --- Precompute: populate context cache for each query ---
        from app.pipeline.chat import _build_recent_context_text
        ep = registry.get_embedding_provider(cfg)
        for q in QUERIES:
            ctx_text = _build_recent_context_text(q.messages, "", cfg.memory.max_recent_turns_for_query)
            if ctx_text.strip():
                vec = await ep.embed_text(ctx_text)
                ctx_cache.put(CONVERSATION_ID, ep.model, vec)
        precompute = await run_retrieval(cfg, registry, args.top_k, args.vector_top_k)

        # --- Compare ---
        comparisons = []
        identical_count = 0
        total_jaccard = 0.0

        for q in QUERIES:
            cmp = compare(q.name, baseline[q.name], precompute[q.name])
            comparisons.append(cmp)
            if cmp["identical"]:
                identical_count += 1
            total_jaccard += cmp["jaccard"]

        summary = {
            "total_queries": len(QUERIES),
            "identical_count": identical_count,
            "identical_rate": identical_count / len(QUERIES),
            "avg_jaccard": round(total_jaccard / len(QUERIES), 3),
            "per_query": comparisons,
        }

        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(f"Accuracy comparison ({len(QUERIES)} queries)")
            print(f"  Identical results: {identical_count}/{len(QUERIES)} ({identical_count/len(QUERIES):.0%})")
            print(f"  Avg Jaccard similarity: {summary['avg_jaccard']}")
            for c in comparisons:
                status = "✓" if c["identical"] else f"✗ diff={c['only_baseline']} vs {c['only_precompute']}"
                print(f"  {c['query']}: J={c['jaccard']} {status}")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(root_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--vector-top-k", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    main()
