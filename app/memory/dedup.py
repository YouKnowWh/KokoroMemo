"""Memory card deduplication and merge logic.

Types, normalization helpers, and similarity functions for detecting
and merging duplicate memory cards.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DuplicateCandidate:
    """A pair of cards that are likely duplicates."""

    card_a: dict
    card_b: dict
    similarity: float
    match_type: str  # "exact_content", "normalized_content", "near_title"


@dataclass
class MergePreview:
    """Dry-run preview of what a merge would produce."""

    survivor_card_id: str
    superseded_card_ids: list[str]
    merged_card: dict
    changes: dict  # field -> (old_value, new_value) for each changed field
    superseded_updates: list[dict]  # per-superseded-card update records


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALPHANUM_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_card_content(text: str) -> str:
    """Normalize card content for fuzzy comparison.

    Lowercases, collapses whitespace, strips surrounding whitespace,
    and removes punctuation that rarely carries semantic weight.
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NON_ALPHANUM_RE.sub("", text)
    return text


def normalize_title(title: str) -> str:
    """Normalize a card title for comparison.

    Lowercases, collapses whitespace, strips.
    """
    if not title:
        return ""
    return _WHITESPACE_RE.sub(" ", title.lower().strip())


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def content_similarity(a: str, b: str) -> float:
    """Compute a simple token-overlap similarity in [0, 1].

    Uses character bigrams for efficiency and language-agnostic matching.
    Returns 1.0 for identical normalized content, 0.0 for no overlap.
    """
    na = normalize_card_content(a)
    nb = normalize_card_content(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    bigrams_a = _bigrams(na)
    bigrams_b = _bigrams(nb)
    if not bigrams_a or not bigrams_b:
        return 0.0

    intersection = len(bigrams_a & bigrams_b)
    union = len(bigrams_a | bigrams_b)
    return intersection / union if union else 0.0


def _bigrams(text: str) -> set[str]:
    """Extract set of character bigrams from text."""
    return {text[i : i + 2] for i in range(len(text) - 1)}


# ---------------------------------------------------------------------------
# Scope bucketing
# ---------------------------------------------------------------------------


def same_scope_bucket(card_a: dict, card_b: dict) -> bool:
    """Return True if two cards belong to the same scope bucket.

    Two cards are in the same bucket when they share user_id AND
    (character_id or both None) AND scope.
    """
    return (
        card_a.get("user_id") == card_b.get("user_id")
        and card_a.get("character_id") == card_b.get("character_id")
        and card_a.get("scope") == card_b.get("scope")
    )


# ---------------------------------------------------------------------------
# Merge field helpers
# ---------------------------------------------------------------------------


def merge_field_rules(survivor: dict, superseded: list[dict]) -> dict:
    """Apply merge field rules to produce a merged card dict.

    Rules:
    - importance / confidence / stability: max across all cards
    - source_turn_ids_json: union of all non-null source turn ids
    - evidence_text: concatenate (survivor first, then each superseded, max 1000 chars)
    - content / title / summary / scope / card_type / status: survivor wins
    - access_count: sum across all cards
    """
    merged = dict(survivor)

    for field in ("importance", "confidence", "stability"):
        merged[field] = max(
            survivor.get(field, 0.0),
            *(card.get(field, 0.0) for card in superseded),
        )

    merged["source_turn_ids_json"] = _union_source_turn_ids(
        [survivor] + superseded
    )

    evidence_parts = []
    for card in [survivor] + superseded:
        t = card.get("evidence_text")
        if t and t.strip():
            evidence_parts.append(t.strip())
    merged["evidence_text"] = "\n---\n".join(evidence_parts)[:1000]

    merged["access_count"] = sum(
        card.get("access_count", 0) for card in [survivor] + superseded
    )

    return merged


def _union_source_turn_ids(cards: list[dict]) -> str | None:
    """Merge source_turn_ids_json fields from multiple cards into a sorted, deduped JSON array."""
    ids: set[str] = set()
    for card in cards:
        raw = card.get("source_turn_ids_json")
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                ids.update(str(t) for t in parsed)
        except (json.JSONDecodeError, TypeError):
            pass
    if not ids:
        return None
    return json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":"))


def _changes_summary(survivor: dict, merged: dict) -> dict:
    """Return a dict of field -> (old_value, new_value) for fields that changed."""
    changes: dict = {}
    for key, new_val in merged.items():
        old_val = survivor.get(key)
        if old_val != new_val:
            changes[key] = (old_val, new_val)
    return changes


# ---------------------------------------------------------------------------
# Vector index consistency
# ---------------------------------------------------------------------------


async def sync_after_card_merge(
    db_path: str,
    survivor_card_id: str,
    superseded_card_ids: list[str],
    embedding_provider,
    lancedb_store,
) -> dict:
    """Ensure vector index consistency after a card merge.

    Deletes superseded card vectors from the vector store, then re-syncs
    the survivor card to pick up any merged content changes.

    Returns a summary dict with success/failure counts.
    """
    import logging

    logger = logging.getLogger("kokoromemo.dedup")

    deleted = 0
    delete_errors = 0
    for cid in superseded_card_ids:
        try:
            lancedb_store.delete(f"memory_id = '{cid}'")
            deleted += 1
        except Exception as exc:
            logger.warning("Failed to delete vector for superseded card %s: %s", cid, exc)
            delete_errors += 1

    synced = False
    sync_error: str | None = None
    if embedding_provider and lancedb_store:
        try:
            from app.storage.vector_sync import sync_card_vector

            await sync_card_vector(db_path, survivor_card_id, embedding_provider, lancedb_store)
            synced = True
        except Exception as exc:
            sync_error = str(exc)
            logger.warning("Failed to re-sync survivor card %s after merge: %s", survivor_card_id, exc)
            try:
                from app.storage.vector_sync import enqueue_card_vector_sync

                await enqueue_card_vector_sync(db_path, survivor_card_id, sync_error)
            except Exception:
                pass

    return {
        "survivor_card_id": survivor_card_id,
        "vectors_deleted": deleted,
        "delete_errors": delete_errors,
        "survivor_resynced": synced,
        "resync_error": sync_error,
    }
