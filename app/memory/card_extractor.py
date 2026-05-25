"""Card-based memory extraction. Produces candidates -> inbox or direct approve."""

from __future__ import annotations

import json
import logging

from aiosqlite import connect as aiosqlite_connect

from app.core.ids import generate_id
from app.memory.judge import MemoryJudgeConfigView, judge_memories_with_llm
from app.memory.review_policy import auto_review, determine_risk_level
from app.storage.sqlite_cards import (
    find_duplicate_card,
    insert_card,
    insert_card_version,
    insert_inbox_item,
    get_write_library_id,
    trim_discarded_inbox,
)
from app.storage.vector_sync import enqueue_card_vector_sync, sync_card_vector

logger = logging.getLogger("kokoromemo.card_extractor")

# Card types that represent volatile task progress — route as pending to avoid
# long-term memory pollution from rapidly-changing task state.
_TASK_STATE_TYPES = frozenset({"task_state", "task_progress", "quest_progress"})

# Card types that should never be stored — they're noise written by legacy plugin paths.
BLOCKED_TYPES = frozenset({"task_state", "tool_result", "artifact", "checkpoint", "summary", "event", "system_prompt"})


def default_card_title(content: str, card_type: str, max_len: int = 80) -> str:
    """Generate a fallback card title when the LLM doesn't provide one.

    Uses the first sentence or first max_len chars of content, with
    the card_type as a prefix if the content is very short.
    """
    text = content.strip()
    if not text:
        return card_type

    for sep in ("。", "！", "？", ".", "!", "?", "\n"):
        idx = text.find(sep)
        if idx > 0:
            return text[:idx + 1].strip()[:max_len]

    if len(text) > max_len:
        return text[:max_len - 1] + "…"
    return text


async def enrich_existing_card_from_duplicate_candidate(
    db_path: str,
    existing_card: dict,
    new_importance: float,
    new_confidence: float,
    new_turn_id: str | None,
) -> None:
    """Update an existing card with enriched fields from a duplicate candidate.

    Merge rules:
    - importance: max(existing, new)
    - confidence: max(existing, new)
    - source_turn_ids: union of existing and new turn id

    Writes a card_version for audit trail.
    """
    import json as _json

    new_imp = max(existing_card.get("importance", 0.0), new_importance)
    new_conf = max(existing_card.get("confidence", 0.0), new_confidence)

    turn_ids: set[str] = set()
    existing_turns = existing_card.get("source_turn_ids_json")
    if existing_turns:
        try:
            parsed = _json.loads(existing_turns)
            if isinstance(parsed, list):
                turn_ids.update(str(t) for t in parsed)
        except (_json.JSONDecodeError, TypeError):
            pass
    if new_turn_id:
        turn_ids.add(new_turn_id)
    merged_turns = _json.dumps(sorted(turn_ids), ensure_ascii=False, separators=(",", ":")) if turn_ids else None

    card_id = existing_card["card_id"]
    async with aiosqlite_connect(db_path, timeout=10.0) as db:
        await db.execute(
            """UPDATE memory_cards
               SET importance = ?, confidence = ?, source_turn_ids_json = ?,
                   updated_at = datetime('now', 'localtime')
               WHERE card_id = ?""",
            (new_imp, new_conf, merged_turns, card_id),
        )
        await db.commit()

    await insert_card_version(
        db_path,
        card_id=card_id,
        content=existing_card.get("content", ""),
        card_type=existing_card.get("card_type", "preference"),
        importance=new_imp,
        confidence=new_conf,
    )

    logger.info(
        "memory_extract route=enriched_existing card_id=%s old_imp=%.2f new_imp=%.2f old_conf=%.2f new_conf=%.2f turn_ids=%d",
        card_id,
        existing_card.get("importance", 0.0),
        new_imp,
        existing_card.get("confidence", 0.0),
        new_conf,
        len(turn_ids),
    )


async def extract_and_route(
    db_path: str,
    user_message: str,
    assistant_message: str,
    user_id: str,
    character_id: str | None,
    conversation_id: str,
    embedding_provider=None,
    lancedb_store=None,
    min_importance: float = 0.45,
    min_confidence: float = 0.55,
    judge_config: MemoryJudgeConfigView | None = None,
    lang: str = "zh",
    discarded_keep_limit: int = 200,
    turn_id: str | None = None,
) -> None:
    """Extract candidate memory cards and route through review policy.

    Flow:
    1. Memory judge model → candidate cards
    2. Scoped dedup check (library/user/character/scope/type)
    3. If duplicate found → enrich existing card (max fields, union turn_ids)
    4. task_state types → always route to pending (volatile, not long-term)
    5. review_policy.auto_review() for remaining
    6. auto_approve → write card(approved) + embed + LanceDB
    7. pending → write inbox item
    8. reject / semantic dup → write inbox item with status='discarded'
    """
    if not judge_config:
        logger.info("memory_extract skipped reason=no_judge_config conv=%s", conversation_id)
        return

    try:
        extracted = await judge_memories_with_llm(
            user_message,
            assistant_message,
            character_id,
            judge_config,
            min_importance=min_importance,
            min_confidence=min_confidence,
            lang=lang,
        )
    except Exception as e:
        logger.warning("Memory judge failed: %s", e)
        return

    if not extracted:
        logger.info("memory_extract no_candidates conv=%s", conversation_id)
        return

    library_id = await get_write_library_id(db_path, conversation_id)
    discarded_written = False
    source_turns = json.dumps([turn_id], ensure_ascii=False, separators=(",", ":")) if turn_id else None
    logger.info("memory_extract candidates=%d conv=%s library_id=%s turn_id=%s", len(extracted), conversation_id, library_id, turn_id)

    for mem in extracted:
        risk_level = _risk_level_from_tags(mem.tags) or determine_risk_level(mem.memory_type, mem.confidence)
        title = mem.title or default_card_title(mem.content, mem.memory_type)
        card_payload = {
            "library_id": library_id,
            "user_id": user_id,
            "character_id": character_id,
            "conversation_id": conversation_id,
            "scope": mem.scope,
            "card_type": mem.memory_type,
            "title": title,
            "content": mem.content,
            "importance": mem.importance,
            "confidence": mem.confidence,
            "tags": mem.tags,
            "evidence_text": user_message[:300],
        }

        # Scoped dedup: check for existing card in same bucket with similar content
        from app.memory.dedup import normalize_card_content
        dup_card = await find_duplicate_card(
            db_path,
            library_id=library_id,
            user_id=user_id,
            character_id=character_id,
            scope=mem.scope,
            card_type=mem.memory_type,
            content=mem.content,
            normalized_content=normalize_card_content(mem.content),
        )
        if dup_card:
            await enrich_existing_card_from_duplicate_candidate(
                db_path,
                existing_card=dup_card,
                new_importance=mem.importance,
                new_confidence=mem.confidence,
                new_turn_id=turn_id,
            )
            logger.debug("Enriched existing card %s from duplicate candidate", dup_card["card_id"])
            continue

        # Semantic dedup via vector similarity
        if embedding_provider and lancedb_store:
            sem_match = await _find_semantic_duplicate(embedding_provider, lancedb_store, user_id, mem.content)
            if sem_match:
                logger.info(
                    "memory_extract route=discarded_semantic_duplicate conv=%s type=%s importance=%.2f confidence=%.2f related_card_id=%s similarity=%.2f",
                    conversation_id,
                    mem.memory_type,
                    mem.importance,
                    mem.confidence,
                    sem_match[0],
                    sem_match[1],
                )
                await _write_discarded(
                    db_path,
                    card_payload=card_payload,
                    user_id=user_id,
                    character_id=character_id,
                    conversation_id=conversation_id,
                    risk_level=risk_level,
                    library_id=library_id,
                    discard_reason="semantic_duplicate",
                    reason=f"与已有卡片语义近似（相似度 {sem_match[1]:.2f}）",
                    related_card_id=sem_match[0],
                )
                discarded_written = True
                logger.debug("Discarded semantic near-duplicate: %s", mem.content[:50])
                continue

        # Block noise card types written by legacy plugin paths
        if mem.memory_type in BLOCKED_TYPES:
            logger.debug("Blocked noise card type %s, skipping", mem.memory_type)
            continue

        # Volatile task_state routing: always pending to avoid long-term clutter
        if mem.memory_type in _TASK_STATE_TYPES:
            decision = "pending"
        else:
            decision = auto_review(
                card_type=mem.memory_type,
                importance=mem.importance,
                confidence=mem.confidence,
                risk_level=risk_level,
                tags=mem.tags,
            )
        logger.info(
            "memory_extract decision=%s conv=%s type=%s importance=%.2f confidence=%.2f risk=%s tags=%s content=%s",
            decision,
            conversation_id,
            mem.memory_type,
            mem.importance,
            mem.confidence,
            risk_level,
            ",".join(mem.tags),
            mem.content[:80],
        )

        if decision == "approve":
            card_id = generate_id("card_")
            await insert_card(
                db_path,
                card_id=card_id,
                library_id=library_id,
                user_id=user_id,
                character_id=character_id,
                conversation_id=conversation_id,
                scope=mem.scope,
                card_type=mem.memory_type,
                title=title,
                content=mem.content,
                importance=mem.importance,
                confidence=mem.confidence,
                status="approved",
                evidence_text=user_message[:300],
                source_turn_ids_json=source_turns,
            )
            await insert_card_version(
                db_path,
                card_id=card_id,
                content=mem.content,
                card_type=mem.memory_type,
                importance=mem.importance,
                confidence=mem.confidence,
            )

            if embedding_provider and lancedb_store:
                try:
                    await sync_card_vector(db_path, card_id, embedding_provider, lancedb_store)
                    logger.info("Auto-approved card: %s (type=%s)", card_id, mem.memory_type)
                    await _emit_card_event("card_approved", card_id, mem)
                except Exception as e:
                    await enqueue_card_vector_sync(db_path, card_id, str(e))
                    logger.warning("Vector sync failed for card %s: %s", card_id, e)
            else:
                logger.info("Auto-approved card (no vector): %s", card_id)

        elif decision == "pending":
            inbox_id = generate_id("inbox_")
            await insert_inbox_item(
                db_path,
                inbox_id=inbox_id,
                candidate_type="card",
                payload_json=json.dumps(card_payload, ensure_ascii=False),
                user_id=user_id,
                character_id=character_id,
                conversation_id=conversation_id,
                suggested_action="approve",
                risk_level=risk_level,
                reason=f"记忆判断模型: {mem.memory_type}",
                status="pending",
                library_id=library_id,
            )
            logger.info("Card sent to inbox: %s (type=%s, risk=%s)", inbox_id, mem.memory_type, risk_level)
            await _emit_card_event("inbox_new", inbox_id, mem)

        else:
            await _write_discarded(
                db_path,
                card_payload=card_payload,
                user_id=user_id,
                character_id=character_id,
                conversation_id=conversation_id,
                risk_level=risk_level,
                library_id=library_id,
                discard_reason="auto_rejected",
                reason=f"自动审核拒绝: type={mem.memory_type}, importance={mem.importance:.2f}",
                related_card_id=None,
            )
            discarded_written = True
            logger.debug("Card rejected by policy: type=%s, importance=%.2f", mem.memory_type, mem.importance)

    if discarded_written and discarded_keep_limit > 0:
        try:
            removed = await trim_discarded_inbox(db_path, discarded_keep_limit)
            if removed:
                logger.debug("Trimmed %s old discarded inbox items", removed)
        except Exception as e:
            logger.warning("Failed to trim discarded inbox: %s", e)


async def _write_discarded(
    db_path: str,
    *,
    card_payload: dict,
    user_id: str,
    character_id: str | None,
    conversation_id: str,
    risk_level: str,
    library_id: str,
    discard_reason: str,
    reason: str,
    related_card_id: str | None,
) -> None:
    inbox_id = generate_id("inbox_")
    await insert_inbox_item(
        db_path,
        inbox_id=inbox_id,
        candidate_type="card",
        payload_json=json.dumps(card_payload, ensure_ascii=False),
        user_id=user_id,
        character_id=character_id,
        conversation_id=conversation_id,
        suggested_action="reject",
        risk_level=risk_level,
        reason=reason,
        status="discarded",
        library_id=library_id,
        discard_reason=discard_reason,
        related_card_id=related_card_id,
    )


def _risk_level_from_tags(tags: list[str]) -> str | None:
    for tag in tags:
        if tag in {"risk:low", "risk:medium", "risk:high"}:
            return tag.split(":", 1)[1]
    return None


async def _find_semantic_duplicate(
    embedding_provider,
    lancedb_store,
    user_id: str,
    content: str,
    threshold: float = 0.92,
) -> tuple[str, float] | None:
    """Return (card_id, similarity) of the most similar existing card if above threshold."""
    try:
        vectors = await embedding_provider.embed_texts([content])
        if not vectors or not vectors[0]:
            return None
        results = await lancedb_store.search(
            vectors[0],
            top_k=3,
            where=f"user_id = '{user_id}' AND status = 'approved'",
        )
        if not results:
            return None
        best: tuple[str, float] | None = None
        for r in results:
            distance = r.get("_distance", 1.0)
            similarity = 1.0 - distance
            if similarity >= threshold:
                card_id = r.get("card_id") or r.get("id") or ""
                if best is None or similarity > best[1]:
                    best = (card_id, similarity)
        return best
    except Exception:
        return None


async def _emit_card_event(event_type: str, card_id: str, mem) -> None:
    """Emit a WebSocket event for card extraction activity."""
    try:
        from app.core.events import emit
        await emit(event_type, {
            "card_id": card_id,
            "content": mem.content[:100],
            "memory_type": mem.memory_type,
            "importance": mem.importance,
        })
    except Exception:
        pass
