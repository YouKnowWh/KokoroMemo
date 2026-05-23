"""Tests for memory card deduplication and merge logic (Phase 1)."""

from __future__ import annotations

import sqlite3
import shutil
import uuid
from pathlib import Path

import pytest

from app.memory.dedup import (
    DuplicateCandidate,
    MergePreview,
    content_similarity,
    merge_field_rules,
    normalize_card_content,
    normalize_title,
    same_scope_bucket,
    sync_after_card_merge,
)
from app.storage.sqlite_cards import (
    find_exact_duplicate_groups,
    insert_card,
    insert_card_event,
    init_cards_db,
    merge_memory_cards,
    get_cards_by_ids,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_dir() -> Path:
    root = Path(".test_tmp") / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def cleanup_test_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


async def _seed_cards(db_path: str, user_id: str = "u1") -> dict[str, dict]:
    """Insert several test cards and return them keyed by a short label."""
    cards: dict[str, dict] = {}

    specs = [
        ("card_a", "u1", "ch1", "conv1", "character", "preference", "Likes quiet", 0.6, 0.7, 0.5),
        ("card_b", "u1", "ch1", "conv1", "character", "preference", "Likes quiet", 0.8, 0.9, 0.6),  # exact dup of A
        ("card_c", "u1", "ch1", "conv2", "character", "preference", "Likes quiet environments", 0.7, 0.8, 0.4),  # near dup
        ("card_d", "u1", "ch1", "conv3", "character", "fact", "Born in Tokyo", 0.5, 0.6, 0.5),  # different type
        ("card_e", "u2", "ch1", "conv4", "character", "preference", "Likes quiet", 0.9, 0.95, 0.8),  # different user
        ("card_f", "u1", "ch1", "conv5", "global", "preference", "Prefers morning", 0.5, 0.5, 0.5),  # different scope
    ]

    for card_id, uid, chid, cvid, scope, ctype, content, imp, conf, stab in specs:
        await insert_card(
            db_path, card_id=card_id, user_id=uid, character_id=chid,
            conversation_id=cvid, scope=scope, card_type=ctype, content=content,
            importance=imp, confidence=conf, status="approved",
            evidence_text=f"evidence for {card_id}",
        )
        cards[card_id] = {
            "card_id": card_id, "user_id": uid, "character_id": chid,
            "scope": scope, "card_type": ctype, "content": content,
            "importance": imp, "confidence": conf, "stability": stab,
            "status": "approved", "evidence_text": f"evidence for {card_id}",
        }

    return cards


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------


def test_normalize_card_content_lowercases_and_strips():
    assert normalize_card_content("  Hello WORLD  ") == "hello world"


def test_normalize_card_content_collapses_whitespace():
    assert normalize_card_content("hello   \t\n  world") == "hello world"


def test_normalize_card_content_removes_common_punctuation():
    result = normalize_card_content("Hello, world! How's it going?")
    assert "," not in result
    assert "!" not in result
    assert "?" not in result
    assert "'" not in result
    assert result == "hello world hows it going"


def test_normalize_card_content_empty():
    assert normalize_card_content("") == ""
    assert normalize_card_content(None) == ""


def test_normalize_title():
    assert normalize_title("  My Title  ") == "my title"
    assert normalize_title("UPPER CASE") == "upper case"
    assert normalize_title("") == ""
    assert normalize_title(None) == ""


# ---------------------------------------------------------------------------
# Similarity tests
# ---------------------------------------------------------------------------


def test_content_similarity_identical():
    assert content_similarity("hello world", "hello world") == 1.0


def test_content_similarity_normalized_match():
    assert content_similarity("Hello World!", "hello world") == 1.0


def test_content_similarity_mostly_different():
    assert content_similarity("hello world", "goodbye mars") < 0.3


def test_content_similarity_empty():
    assert content_similarity("", "") == 1.0
    assert content_similarity("hello", "") == 0.0
    assert content_similarity("", "world") == 0.0


def test_content_similarity_single_char():
    assert content_similarity("a", "a") == 1.0
    assert content_similarity("a", "b") == 0.0


# ---------------------------------------------------------------------------
# Scope bucketing tests
# ---------------------------------------------------------------------------


def test_same_scope_bucket_match():
    a = {"user_id": "u1", "character_id": "ch1", "scope": "character"}
    b = {"user_id": "u1", "character_id": "ch1", "scope": "character"}
    assert same_scope_bucket(a, b)


def test_same_scope_bucket_different_user():
    a = {"user_id": "u1", "character_id": "ch1", "scope": "character"}
    b = {"user_id": "u2", "character_id": "ch1", "scope": "character"}
    assert not same_scope_bucket(a, b)


def test_same_scope_bucket_different_scope():
    a = {"user_id": "u1", "character_id": "ch1", "scope": "character"}
    b = {"user_id": "u1", "character_id": "ch1", "scope": "global"}
    assert not same_scope_bucket(a, b)


def test_same_scope_bucket_both_none_character():
    a = {"user_id": "u1", "character_id": None, "scope": "global"}
    b = {"user_id": "u1", "character_id": None, "scope": "global"}
    assert same_scope_bucket(a, b)


# ---------------------------------------------------------------------------
# Merge field rules tests
# ---------------------------------------------------------------------------


def test_merge_field_rules_max_importance_confidence_stability():
    survivor = {"importance": 0.5, "confidence": 0.6, "stability": 0.4, "content": "a"}
    superseded = [
        {"importance": 0.8, "confidence": 0.3, "stability": 0.7, "content": "b"},
        {"importance": 0.3, "confidence": 0.9, "stability": 0.5, "content": "c"},
    ]
    merged = merge_field_rules(survivor, superseded)
    assert merged["importance"] == 0.8
    assert merged["confidence"] == 0.9
    assert merged["stability"] == 0.7


def test_merge_field_rules_survivor_content_wins():
    survivor = {"content": "survivor content", "title": "survivor title"}
    superseded = [{"content": "other content", "title": "other title"}]
    merged = merge_field_rules(survivor, superseded)
    assert merged["content"] == "survivor content"
    assert merged["title"] == "survivor title"


def test_merge_field_rules_union_source_turn_ids():
    survivor = {"source_turn_ids_json": '["t1","t2"]'}
    superseded = [
        {"source_turn_ids_json": '["t2","t3"]'},
        {"source_turn_ids_json": '["t4"]'},
    ]
    merged = merge_field_rules(survivor, superseded)
    assert merged["source_turn_ids_json"] == '["t1","t2","t3","t4"]'


def test_merge_field_rules_handles_null_source_turn_ids():
    survivor = {"source_turn_ids_json": None}
    superseded = [{"source_turn_ids_json": '["t1"]'}, {}]
    merged = merge_field_rules(survivor, superseded)
    assert merged["source_turn_ids_json"] == '["t1"]'


def test_merge_field_rules_all_null_source_turn_ids():
    merged = merge_field_rules({}, [{}, {}])
    assert merged.get("source_turn_ids_json") is None


def test_merge_field_rules_concatenates_evidence():
    survivor = {"evidence_text": "ev1"}
    superseded = [{"evidence_text": "ev2"}, {"evidence_text": None}]
    merged = merge_field_rules(survivor, superseded)
    assert "ev1" in merged["evidence_text"]
    assert "ev2" in merged["evidence_text"]
    assert len(merged["evidence_text"]) <= 1000


def test_merge_field_rules_sums_access_count():
    survivor = {"access_count": 3}
    superseded = [{"access_count": 5}, {"access_count": 2}]
    merged = merge_field_rules(survivor, superseded)
    assert merged["access_count"] == 10


def test_merge_field_rules_missing_fields_default_zero():
    survivor = {}
    superseded = [{}]
    merged = merge_field_rules(survivor, superseded)
    assert merged.get("importance") == 0.0
    assert merged.get("confidence") == 0.0
    assert merged.get("stability") == 0.0
    assert merged.get("access_count") == 0


# ---------------------------------------------------------------------------
# Exact duplicate group tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_exact_duplicate_groups_finds_exact_matches():
    test_dir = make_test_dir()
    db_path = str(test_dir / "memory.sqlite")
    try:
        await init_cards_db(db_path)
        await _seed_cards(db_path)

        groups = await find_exact_duplicate_groups(db_path)
        # card_a and card_b have identical content "Likes quiet"
        exact_groups = [g for g in groups if len(g) >= 2]
        assert len(exact_groups) >= 1

        ab_group = next(g for g in exact_groups if g[0]["content"] == "Likes quiet")
        card_ids = {c["card_id"] for c in ab_group}
        assert "card_a" in card_ids
        assert "card_b" in card_ids
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_find_exact_duplicate_groups_respects_min_group_size():
    test_dir = make_test_dir()
    db_path = str(test_dir / "memory.sqlite")
    try:
        await init_cards_db(db_path)
        await _seed_cards(db_path)

        groups = await find_exact_duplicate_groups(db_path, min_group_size=99)
        assert len(groups) == 0
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_find_exact_duplicate_groups_empty_db():
    test_dir = make_test_dir()
    db_path = str(test_dir / "memory.sqlite")
    try:
        await init_cards_db(db_path)
        groups = await find_exact_duplicate_groups(db_path)
        assert groups == []
    finally:
        cleanup_test_dir(test_dir)


# ---------------------------------------------------------------------------
# merge_memory_cards tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_memory_cards_dry_run():
    test_dir = make_test_dir()
    db_path = str(test_dir / "memory.sqlite")
    try:
        await init_cards_db(db_path)
        await _seed_cards(db_path)

        result = await merge_memory_cards(
            db_path,
            survivor_card_id="card_a",
            superseded_card_ids=["card_b"],
            merge_reason="exact_duplicate",
            dry_run=True,
        )

        assert result["dry_run"] is True
        assert result["survivor_card_id"] == "card_a"
        assert result["superseded_card_ids"] == ["card_b"]
        assert "changes" in result
        assert "merged_card" in result
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_merge_memory_cards_executes_merge():
    test_dir = make_test_dir()
    db_path = str(test_dir / "memory.sqlite")
    try:
        await init_cards_db(db_path)
        await _seed_cards(db_path)

        result = await merge_memory_cards(
            db_path,
            survivor_card_id="card_a",
            superseded_card_ids=["card_b"],
            merge_reason="exact_duplicate",
            dry_run=False,
        )

        assert result["dry_run"] is False
        assert result["survivor_card_id"] == "card_a"
        assert result["superseded_count"] == 1

        # Verify survivor updated with max importance/confidence
        cards = await get_cards_by_ids(db_path, ["card_a", "card_b"])
        assert cards["card_a"]["status"] == "approved"
        assert cards["card_a"]["importance"] == 0.8  # max(0.6, 0.8)
        assert cards["card_a"]["confidence"] == 0.9  # max(0.7, 0.9)

        # Verify superseded card is soft-deleted
        assert cards["card_b"]["status"] == "deleted"
        assert cards["card_b"]["merged_into_card_id"] == "card_a"
        assert cards["card_b"]["merge_reason"] == "exact_duplicate"
        assert cards["card_b"]["deleted_at"] is not None
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_merge_memory_cards_writes_audit_history():
    test_dir = make_test_dir()
    db_path = str(test_dir / "memory.sqlite")
    try:
        await init_cards_db(db_path)
        await _seed_cards(db_path)

        await merge_memory_cards(
            db_path,
            survivor_card_id="card_a",
            superseded_card_ids=["card_b"],
            merge_reason="exact_duplicate",
        )

        # Check card_versions
        with sqlite3.connect(db_path) as conn:
            versions = conn.execute(
                "SELECT * FROM memory_card_versions WHERE card_id = ?",
                ("card_a",),
            ).fetchall()
            assert len(versions) >= 1

            # Check review_actions
            actions = conn.execute(
                "SELECT * FROM review_actions WHERE card_id = ? AND action = ?",
                ("card_a", "merge"),
            ).fetchall()
            assert len(actions) >= 1
            assert "exact_duplicate" in actions[0][5]  # note field

            # Check card_events for survivor
            events_survivor = conn.execute(
                "SELECT * FROM memory_card_events WHERE card_id = ? AND event_type = ?",
                ("card_a", "card_merged_into"),
            ).fetchall()
            assert len(events_survivor) >= 1

            # Check card_events for superseded
            events_superseded = conn.execute(
                "SELECT * FROM memory_card_events WHERE card_id = ? AND event_type = ?",
                ("card_b", "card_merged_out"),
            ).fetchall()
            assert len(events_superseded) >= 1
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_merge_memory_cards_raises_for_missing_survivor():
    test_dir = make_test_dir()
    db_path = str(test_dir / "memory.sqlite")
    try:
        await init_cards_db(db_path)
        await _seed_cards(db_path)

        with pytest.raises(ValueError, match="Survivor card not found"):
            await merge_memory_cards(
                db_path,
                survivor_card_id="nonexistent",
                superseded_card_ids=["card_a"],
            )
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_merge_memory_cards_raises_for_already_deleted():
    test_dir = make_test_dir()
    db_path = str(test_dir / "memory.sqlite")
    try:
        await init_cards_db(db_path)
        await _seed_cards(db_path)

        # Merge first to delete card_b
        await merge_memory_cards(
            db_path, survivor_card_id="card_a", superseded_card_ids=["card_b"],
        )

        # Try to merge card_b again
        with pytest.raises(ValueError, match="already deleted"):
            await merge_memory_cards(
                db_path, survivor_card_id="card_c", superseded_card_ids=["card_b"],
            )
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_merge_memory_cards_multiple_superseded():
    test_dir = make_test_dir()
    db_path = str(test_dir / "memory.sqlite")
    try:
        await init_cards_db(db_path)
        await _seed_cards(db_path)

        result = await merge_memory_cards(
            db_path,
            survivor_card_id="card_d",
            superseded_card_ids=["card_a", "card_b", "card_c"],
            merge_reason="batch_merge",
        )

        assert result["superseded_count"] == 3
        assert result["versions_created"] == 1
        assert result["events_created"] == 4  # 1 survivor + 3 superseded

        cards = await get_cards_by_ids(db_path, ["card_a", "card_b", "card_c", "card_d"])
        assert cards["card_d"]["status"] == "approved"
        for cid in ("card_a", "card_b", "card_c"):
            assert cards[cid]["status"] == "deleted"
            assert cards[cid]["merged_into_card_id"] == "card_d"
    finally:
        cleanup_test_dir(test_dir)


# ---------------------------------------------------------------------------
# Dataclass smoke tests
# ---------------------------------------------------------------------------


def test_duplicate_candidate_creation():
    c = DuplicateCandidate(
        card_a={"card_id": "a", "content": "hello"},
        card_b={"card_id": "b", "content": "hello"},
        similarity=0.95,
        match_type="normalized_content",
    )
    assert c.similarity == 0.95
    assert c.match_type == "normalized_content"


def test_merge_preview_creation():
    p = MergePreview(
        survivor_card_id="card_a",
        superseded_card_ids=["card_b"],
        merged_card={"importance": 0.8},
        changes={"importance": (0.5, 0.8)},
        superseded_updates=[{"card_id": "card_b", "merged_into_card_id": "card_a"}],
    )
    assert len(p.superseded_card_ids) == 1


# ---------------------------------------------------------------------------
# Vector sync after merge tests
# ---------------------------------------------------------------------------


class FakeVectorStore:
    """Records delete calls for verification."""
    def __init__(self):
        self.deleted: list[str] = []
        self.upserted: list = []

    def delete(self, where: str) -> None:
        self.deleted.append(where)

    def upsert(self, rows):
        self.upserted.extend(rows)
        return None


class FakeEmbeddingProvider:
    dimension = 128
    model = "fake"

    async def embed_text(self, text: str):
        return [0.1] * self.dimension

    async def embed_texts(self, texts: list[str]):
        return [[0.1] * self.dimension for _ in texts]


@pytest.mark.asyncio
async def test_sync_after_card_merge_deletes_superseded_vectors():
    test_dir = make_test_dir()
    db_path = str(test_dir / "memory.sqlite")
    try:
        await init_cards_db(db_path)
        await _seed_cards(db_path)

        fake_store = FakeVectorStore()
        fake_embed = FakeEmbeddingProvider()

        result = await sync_after_card_merge(
            db_path, "card_a", ["card_b", "card_c"],
            embedding_provider=fake_embed, lancedb_store=fake_store,
        )

        assert result["vectors_deleted"] == 2
        assert result["delete_errors"] == 0
        assert result["survivor_resynced"] is True
        assert len(fake_store.deleted) == 2
        assert "card_b" in fake_store.deleted[0]
        assert "card_c" in fake_store.deleted[1]
    finally:
        cleanup_test_dir(test_dir)


@pytest.mark.asyncio
async def test_sync_after_card_merge_handles_none_embedding_provider():
    test_dir = make_test_dir()
    db_path = str(test_dir / "memory.sqlite")
    try:
        await init_cards_db(db_path)
        await _seed_cards(db_path)

        fake_store = FakeVectorStore()

        result = await sync_after_card_merge(
            db_path, "card_a", ["card_b"],
            embedding_provider=None, lancedb_store=fake_store,
        )

        assert result["survivor_resynced"] is False
        assert result["vectors_deleted"] == 1
    finally:
        cleanup_test_dir(test_dir)
