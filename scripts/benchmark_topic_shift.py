#!/usr/bin/env python3
"""Test accuracy with stale context embedding across topic shifts.

BASELINE: fresh embedding per turn
STALE:   precompute context embedding from turn-1, never refresh
EVERY-N: refresh context embedding every N turns
"""

from __future__ import annotations

import argparse, asyncio, json, shutil, sys, tempfile
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

USER_ID = "u"
CHARACTER_ID = "c"
CONVERSATION_ID = "conv"


class TimedEmbeddingProvider:
    def __init__(self, inner):
        self._inner = inner
        self.provider_name = inner.provider_name
        self.model = inner.model
        self.dimension = inner.dimension

    async def embed_text(self, text):
        return await self._inner.embed_text(text)

    async def embed_batch(self, texts):
        return await self._inner.embed_batch(texts)

    async def health_check(self):
        return await self._inner.health_check()


@dataclass
class CardSpec:
    card_id: str; content: str; scope: str = "character"
    card_type: str = "preference"; importance: float = 0.85; confidence: float = 0.9


CARDS = [
    CardSpec("card_spicy", "用户喜欢吃辣，尤其是川菜。"),
    CardSpec("card_cilantro", "用户不吃香菜。", card_type="boundary"),
    CardSpec("card_company", "用户在字节跳动做后端开发。", card_type="fact"),
    CardSpec("card_lang", "用户主要用Go和Rust编程。", card_type="fact"),
    CardSpec("card_cat", "用户养了一只叫豆包的橘猫。", card_type="fact"),
    CardSpec("card_sleep", "用户习惯凌晨1点睡，早上9点起。", card_type="fact"),
    CardSpec("card_game", "用户最近在玩黑神话悟空。", card_type="state"),
    CardSpec("card_anime", "用户喜欢看进击的巨人。"),
]


# Each turn: more accumulated context + a new user message targeting a different topic
TURNS = [
    {"name": "turn1_food", "topic": "饮食", "messages": [
        {"role": "user", "content": "帮我推荐个外卖"},
        {"role": "assistant", "content": "喜欢什么口味？"},
        {"role": "user", "content": "要辣的不要香菜"},
    ], "expected": ["card_spicy", "card_cilantro"]},

    {"name": "turn2_food2", "topic": "饮食", "messages": [
        {"role": "user", "content": "帮我推荐个外卖"},
        {"role": "assistant", "content": "喜欢什么口味？"},
        {"role": "user", "content": "要辣的不要香菜"},
        {"role": "assistant", "content": "川菜很适合"},
        {"role": "user", "content": "那就川菜吧"},
    ], "expected": ["card_spicy", "card_cilantro"]},

    {"name": "turn3_work", "topic": "工作", "messages": [
        {"role": "user", "content": "帮我推荐个外卖"},
        {"role": "assistant", "content": "喜欢什么口味？"},
        {"role": "user", "content": "要辣的不要香菜"},
        {"role": "assistant", "content": "川菜很适合"},
        {"role": "user", "content": "话说我在哪里工作来着？"},
    ], "expected": ["card_company", "card_lang"]},

    {"name": "turn4_life", "topic": "生活", "messages": [
        {"role": "user", "content": "帮我推荐个外卖"},
        {"role": "assistant", "content": "喜欢什么口味？"},
        {"role": "user", "content": "要辣的不要香菜"},
        {"role": "assistant", "content": "川菜很适合"},
        {"role": "user", "content": "我家猫总半夜闹怎么办"},
    ], "expected": ["card_cat", "card_sleep"]},

    {"name": "turn5_game", "topic": "娱乐", "messages": [
        {"role": "user", "content": "帮我推荐个外卖"},
        {"role": "assistant", "content": "喜欢什么口味？"},
        {"role": "user", "content": "要辣的不要香菜"},
        {"role": "assistant", "content": "川菜很适合"},
        {"role": "user", "content": "有什么好玩的游戏吗"},
    ], "expected": ["card_game", "card_anime"]},

    {"name": "turn6_food", "topic": "饮食", "messages": [
        {"role": "user", "content": "帮我推荐个外卖"},
        {"role": "assistant", "content": "喜欢什么口味？"},
        {"role": "user", "content": "要辣的不要香菜"},
        {"role": "assistant", "content": "川菜很适合"},
        {"role": "user", "content": "我还是想点外卖，川菜"},
    ], "expected": ["card_spicy", "card_cilantro"]},
]


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--root-dir", default=None)
    p.add_argument("--keep", action="store_true")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--vector-top-k", type=int, default=30)
    p.add_argument("--json", action="store_true")
    return p


async def prepare_sandbox(root_dir, config_path):
    cfg = load_config(config_path)
    cfg.storage.root_dir = str(root_dir)
    cfg.storage.sqlite.app_db = str(root_dir / "app.sqlite")
    cfg.storage.sqlite.memory_db = str(root_dir / "memory.sqlite")
    cfg.storage.lancedb.path = str(root_dir / "vector_indexes" / "benchmark" / "lancedb")
    await init_cards_db(cfg.storage.sqlite.memory_db)
    lib = await create_memory_library(cfg.storage.sqlite.memory_db, name="TS", library_id="lib_ts")
    await set_conversation_mounts(cfg.storage.sqlite.memory_db, CONVERSATION_ID, [lib],
                                   write_library_id=lib, user_id=USER_ID, character_id=CHARACTER_ID)
    return cfg, ServiceRegistry(), lib


async def seed(cfg, reg, lib):
    db = cfg.storage.sqlite.memory_db
    ep = reg.get_embedding_provider(cfg)
    store = reg.get_lancedb_store(cfg)
    for c in CARDS:
        await insert_card(db, c.card_id, lib, USER_ID, CHARACTER_ID, CONVERSATION_ID,
                          c.scope, c.card_type, c.content, c.importance, c.confidence, "approved")
        if ep and store:
            await sync_card_vector(db, c.card_id, ep, store)


async def run_one(cfg, turn, ep, store, top_k, vk):
    rq = build_retrieval_query(turn["messages"], USER_ID, CHARACTER_ID, CONVERSATION_ID,
                                max_recent_turns=cfg.memory.max_recent_turns_for_query)
    candidates = await retrieve_cards(rq, ep, store, cfg.storage.sqlite.memory_db,
                                       vector_top_k=vk, final_top_k=top_k)
    returned = [c.card_id for c in candidates]
    matched = [e for e in turn["expected"] if e in returned]
    return {"name": turn["name"], "topic": turn["topic"],
            "expected": turn["expected"], "returned": returned, "matched": matched,
            "hit_rate": len(matched)/len(turn["expected"]) if turn["expected"] else 0}


def summarize(label, results):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    th, te = 0, 0
    for r in results:
        th += len(r["matched"]); te += len(r["expected"])
        ok = "OK" if r["hit_rate"]==1.0 else f"MISS {r['hit_rate']:.0%}"
        print(f"  {r['name']} [{r['topic']}] expected={r['expected']} matched={r['matched']} {ok}")
    print(f"  OVERALL: {th}/{te} ({th/te:.0%})" if te else "  OVERALL: N/A")


async def async_main(args):
    root_dir = Path(args.root_dir) if args.root_dir else Path(tempfile.mkdtemp(prefix="ts-"))
    root_dir.mkdir(parents=True, exist_ok=True)

    try:
        cfg, reg, lib = await prepare_sandbox(root_dir, args.config)
        await seed(cfg, reg, lib)

        ep_raw = reg.get_embedding_provider(cfg)
        store_raw = reg.get_lancedb_store(cfg)
        ep = TimedEmbeddingProvider(ep_raw)

        # --- BASELINE: fresh embedding per turn ---
        init_retrieval_cache(ttl_seconds=300, capacity=16)
        init_context_cache(ttl_seconds=0)  # disable context cache
        fresh = []
        for turn in TURNS:
            fresh.append(await run_one(cfg, turn, ep, store_raw, args.top_k, args.vector_top_k))

        # --- STALE: precompute from turn-1 context, never refresh ---
        init_context_cache(ttl_seconds=300)
        get_context_cache().clear()
        from app.pipeline.chat import _build_recent_context_text
        ctx_text = _build_recent_context_text(TURNS[0]["messages"], "", cfg.memory.max_recent_turns_for_query)
        if ctx_text.strip():
            vec = await ep_raw.embed_text(ctx_text)
            get_context_cache().put(CONVERSATION_ID, ep_raw.model, vec)
        stale = []
        for turn in TURNS:
            stale.append(await run_one(cfg, turn, ep, store_raw, args.top_k, args.vector_top_k))

        # --- EVERY-3: refresh every 3 turns ---
        get_context_cache().clear()
        n3 = []
        for i, turn in enumerate(TURNS):
            if i % 3 == 0:
                ctx_text = _build_recent_context_text(turn["messages"], "", cfg.memory.max_recent_turns_for_query)
                if ctx_text.strip():
                    vec = await ep_raw.embed_text(ctx_text)
                    get_context_cache().put(CONVERSATION_ID, ep_raw.model, vec)
            n3.append(await run_one(cfg, turn, ep, store_raw, args.top_k, args.vector_top_k))

        summarize("BASELINE: fresh embedding per turn", fresh)
        summarize("STALE:   context from turn-1, never refreshed", stale)
        summarize("N=3:     refresh context every 3 turns", n3)

        if args.json:
            print(json.dumps({"fresh": fresh, "stale": stale, "n3": n3}, ensure_ascii=False, indent=2))
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(root_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main(make_parser().parse_args())))
