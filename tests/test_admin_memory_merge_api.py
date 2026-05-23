"""Tests for admin memory dedup/merge API endpoints (Phase 2)."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import AppConfig
from app.core.state import set_config
from app.main import app
from app.storage.sqlite_cards import init_cards_db, insert_card


def make_test_dir() -> Path:
    root = Path(".test_tmp") / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def cleanup_test_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def make_config(test_dir: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.storage.root_dir = str(test_dir)
    cfg.storage.sqlite.memory_db = str(test_dir / "memory.sqlite")
    cfg.server.allow_remote_access = True
    cfg.embedding.enabled = False
    set_config(cfg)
    return cfg


async def seed_test_cards(db_path: str) -> dict[str, dict]:
    """Insert test cards and return their dicts keyed by label."""
    cards: dict[str, dict] = {}

    specs = [
        ("card_a", "u1", "ch1", "conv1", "character", "preference", "Likes quiet evenings at home", 0.6, 0.7),
        ("card_b", "u1", "ch1", "conv1", "character", "preference", "Likes quiet evenings at home", 0.8, 0.9),
        ("card_c", "u1", "ch1", "conv2", "character", "preference", "Likes quiet evenings at home mostly", 0.5, 0.6),
        ("card_d", "u1", "ch1", "conv3", "character", "preference", "Prefers morning walks", 0.7, 0.8),
        ("card_e", "u1", "ch1", "conv4", "character", "preference", "Prefers morning walks", 0.3, 0.4),
    ]

    for card_id, uid, chid, cvid, scope, ctype, content, imp, conf in specs:
        await insert_card(
            db_path, card_id=card_id, user_id=uid, character_id=chid,
            conversation_id=cvid, scope=scope, card_type=ctype, content=content,
            importance=imp, confidence=conf, status="approved",
            evidence_text=f"evidence for {card_id}",
        )
        cards[card_id] = {
            "card_id": card_id, "user_id": uid, "character_id": chid,
            "scope": scope, "card_type": ctype, "content": content,
            "importance": imp, "confidence": conf,
        }

    return cards


# ---------------------------------------------------------------------------
# POST /admin/memories/merge-preview
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_preview_returns_dry_run_preview():
    test_dir = make_test_dir()
    try:
        cfg = make_config(test_dir)
        await init_cards_db(cfg.storage.sqlite.memory_db)
        await seed_test_cards(cfg.storage.sqlite.memory_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/memories/merge-preview", json={
                "survivor_card_id": "card_a",
                "superseded_card_ids": ["card_b"],
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        preview = data["preview"]
        assert preview["dry_run"] is True
        assert preview["survivor_card_id"] == "card_a"
        assert "card_b" in preview["superseded_card_ids"]
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_merge_preview_requires_at_least_one_superseded():
    test_dir = make_test_dir()
    try:
        cfg = make_config(test_dir)
        await init_cards_db(cfg.storage.sqlite.memory_db)
        await seed_test_cards(cfg.storage.sqlite.memory_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/memories/merge-preview", json={
                "survivor_card_id": "card_a",
                "superseded_card_ids": [],
            })

        assert resp.status_code == 400
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_merge_preview_rejects_missing_survivor():
    test_dir = make_test_dir()
    try:
        cfg = make_config(test_dir)
        await init_cards_db(cfg.storage.sqlite.memory_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/memories/merge-preview", json={
                "survivor_card_id": "nonexistent",
                "superseded_card_ids": ["card_a"],
            })

        assert resp.status_code == 400
    finally:
        cleanup_test_dir(test_dir)


# ---------------------------------------------------------------------------
# POST /admin/memories/merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_executes_and_soft_deletes_superseded():
    test_dir = make_test_dir()
    try:
        cfg = make_config(test_dir)
        await init_cards_db(cfg.storage.sqlite.memory_db)
        await seed_test_cards(cfg.storage.sqlite.memory_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/memories/merge", json={
                "survivor_card_id": "card_a",
                "superseded_card_ids": ["card_b"],
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        merge_result = data["merge"]
        assert merge_result["dry_run"] is False
        assert merge_result["superseded_count"] == 1

        # Verify DB state
        import sqlite3
        with sqlite3.connect(cfg.storage.sqlite.memory_db) as conn:
            rows = conn.execute(
                "SELECT card_id, status, merged_into_card_id FROM memory_cards WHERE card_id IN ('card_a', 'card_b')"
            ).fetchall()
        rows_by_id = {r[0]: r for r in rows}
        assert rows_by_id["card_a"][1] == "approved"
        assert rows_by_id["card_b"][1] == "deleted"
        assert rows_by_id["card_b"][2] == "card_a"
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_merge_raises_for_missing_survivor():
    test_dir = make_test_dir()
    try:
        cfg = make_config(test_dir)
        await init_cards_db(cfg.storage.sqlite.memory_db)
        await seed_test_cards(cfg.storage.sqlite.memory_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/memories/merge", json={
                "survivor_card_id": "nonexistent",
                "superseded_card_ids": ["card_a"],
            })

        assert resp.status_code == 400
    finally:
        cleanup_test_dir(test_dir)


# ---------------------------------------------------------------------------
# POST /admin/memories/batch-merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_merge_processes_multiple_merges():
    test_dir = make_test_dir()
    try:
        cfg = make_config(test_dir)
        await init_cards_db(cfg.storage.sqlite.memory_db)
        await seed_test_cards(cfg.storage.sqlite.memory_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/memories/batch-merge", json={
                "merges": [
                    {"survivor_card_id": "card_a", "superseded_card_ids": ["card_b"]},
                    {"survivor_card_id": "card_d", "superseded_card_ids": ["card_e"]},
                ],
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["total"] == 2
        assert data["succeeded"] == 2
        assert data["failed"] == 0
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_batch_merge_reports_errors_for_invalid_merges():
    test_dir = make_test_dir()
    try:
        cfg = make_config(test_dir)
        await init_cards_db(cfg.storage.sqlite.memory_db)
        await seed_test_cards(cfg.storage.sqlite.memory_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/memories/batch-merge", json={
                "merges": [
                    {"survivor_card_id": "card_a", "superseded_card_ids": []},
                    {"survivor_card_id": "nonexistent", "superseded_card_ids": ["card_a"]},
                ],
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["succeeded"] == 0
        assert data["failed"] >= 1
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_batch_merge_requires_non_empty_merges():
    test_dir = make_test_dir()
    try:
        cfg = make_config(test_dir)
        await init_cards_db(cfg.storage.sqlite.memory_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/memories/batch-merge", json={
                "merges": [],
            })

        assert resp.status_code == 400
    finally:
        cleanup_test_dir(test_dir)


# ---------------------------------------------------------------------------
# GET /admin/memories/duplicates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicates_exact_mode_finds_identical_content():
    test_dir = make_test_dir()
    try:
        cfg = make_config(test_dir)
        await init_cards_db(cfg.storage.sqlite.memory_db)
        cards = await seed_test_cards(cfg.storage.sqlite.memory_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/admin/memories/duplicates", params={"mode": "exact"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "exact"
        assert data["total_groups"] >= 1

        # card_a and card_b have identical content; card_d and card_e too
        contents_found = set()
        for group in data["groups"]:
            for card in group["cards"]:
                contents_found.add(card["card_id"])
        assert "card_a" in contents_found
        assert "card_b" in contents_found
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_duplicates_near_mode_finds_similar_content():
    test_dir = make_test_dir()
    try:
        cfg = make_config(test_dir)
        await init_cards_db(cfg.storage.sqlite.memory_db)
        await seed_test_cards(cfg.storage.sqlite.memory_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/admin/memories/duplicates", params={
                "mode": "near",
                "min_similarity": 0.7,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "near"
        # card_a, card_b, card_c are all similar
        assert data["total_pairs"] >= 1
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_duplicates_rejects_bad_mode():
    test_dir = make_test_dir()
    try:
        cfg = make_config(test_dir)
        await init_cards_db(cfg.storage.sqlite.memory_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/admin/memories/duplicates", params={"mode": "invalid"})

        assert resp.status_code == 400
    finally:
        cleanup_test_dir(test_dir)


# ---------------------------------------------------------------------------
# POST /admin/memories/dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_dry_run_returns_merge_plan():
    test_dir = make_test_dir()
    try:
        cfg = make_config(test_dir)
        await init_cards_db(cfg.storage.sqlite.memory_db)
        await seed_test_cards(cfg.storage.sqlite.memory_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/memories/dedup", json={
                "mode": "dry_run",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["mode"] == "dry_run"
        assert "merge_plan" in data
        assert data["total_merges"] >= 1
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_dedup_apply_executes_merges():
    test_dir = make_test_dir()
    try:
        cfg = make_config(test_dir)
        await init_cards_db(cfg.storage.sqlite.memory_db)
        await seed_test_cards(cfg.storage.sqlite.memory_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/memories/dedup", json={
                "mode": "apply",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["mode"] == "apply"
        assert data["succeeded"] >= 1
        assert data["failed"] == 0
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_dedup_rejects_bad_mode():
    test_dir = make_test_dir()
    try:
        cfg = make_config(test_dir)
        await init_cards_db(cfg.storage.sqlite.memory_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/memories/dedup", json={
                "mode": "invalid",
            })

        assert resp.status_code == 400
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_dedup_dry_run_with_filters():
    test_dir = make_test_dir()
    try:
        cfg = make_config(test_dir)
        await init_cards_db(cfg.storage.sqlite.memory_db)
        await seed_test_cards(cfg.storage.sqlite.memory_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/admin/memories/dedup", json={
                "mode": "dry_run",
                "user_id": "u1",
                "character_id": "ch1",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
    finally:
        cleanup_test_dir(test_dir)
