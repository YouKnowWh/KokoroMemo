"""Import stable system prompts as compact persona memory cards.

Raw system prompts should not be stored as memory: they often contain tool
instructions, runtime policy, dates, and client scaffolding. This module keeps
only short persona-like facts and writes them as normal card types.
"""

from __future__ import annotations

import hashlib
import re

import aiosqlite

from app.core.ids import generate_id
from app.storage.sqlite_cards import DEFAULT_MEMORY_LIBRARY_ID, insert_card

_IDENTITY_PATTERN = re.compile(r"^You are Hermes Agent,.*?\n\n", re.DOTALL)
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+")
_BULLET_PATTERN = re.compile(r"^[-*+•\d.)、\s]+")

_NOISE_PATTERNS = (
    "tool", "function", "api", "json", "schema", "http", "sqlite", "database",
    "current date", "timezone", "knowledge cutoff", "system prompt", "developer",
    "sandbox", "approval", "token", "cache", "memory injection", "retrieval",
    "工具", "函数", "数据库", "当前日期", "时区", "知识截止", "系统提示",
    "开发者", "沙盒", "审批", "令牌", "缓存", "记忆注入", "召回",
)

_RULE_HINTS = (
    "must", "never", "always", "should", "do not", "forbid", "avoid",
    "必须", "禁止", "永远", "始终", "不要", "不能", "避免", "优先",
)
_PERSONA_HINTS = (
    "you are", "name", "persona", "identity", "character", "role",
    "你是", "名字", "人设", "身份", "角色", "性格", "说话", "口吻",
)
_USER_HINTS = (
    "user", "用户", "主人", "称呼", "偏好", "likes", "prefers",
)


def _clean_prompt(system_prompt: str) -> str:
    cleaned = _IDENTITY_PATTERN.sub("", system_prompt, count=1).strip()
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    return cleaned


def _candidate_lines(cleaned: str) -> list[str]:
    parts: list[str] = []
    for raw in cleaned.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = _HEADING_PATTERN.sub("", line).strip()
        line = _BULLET_PATTERN.sub("", line).strip()
        if len(line) < 12 or len(line) > 260:
            continue
        lowered = line.lower()
        if any(p in lowered for p in _NOISE_PATTERNS):
            continue
        if line.count("{") or line.count("}") or line.count("=") > 2:
            continue
        parts.append(line)
    return parts


def _classify(line: str) -> str | None:
    lowered = line.lower()
    if any(h in lowered for h in _PERSONA_HINTS):
        return "character_state"
    if any(h in lowered for h in _USER_HINTS):
        return "preference"
    if any(h in lowered for h in _RULE_HINTS):
        return "boundary"
    return None


def extract_persona_cards(system_prompt: str, max_cards: int = 12) -> list[dict[str, str]]:
    cleaned = _clean_prompt(system_prompt)
    if len(cleaned) < 50:
        return []

    cards: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in _candidate_lines(cleaned):
        card_type = _classify(line)
        if not card_type:
            continue
        norm = re.sub(r"\s+", " ", line).strip().lower()
        if norm in seen:
            continue
        seen.add(norm)
        title_hash = hashlib.sha256(norm.encode()).hexdigest()[:12]
        cards.append({
            "card_type": card_type,
            "title": f"initial_persona:{title_hash}",
            "content": line,
        })
        if len(cards) >= max_cards:
            break
    return cards


async def import_system_prompt_persona_cards(
    db_path: str,
    system_prompt: str,
    *,
    user_id: str,
    character_id: str | None,
    conversation_id: str | None,
) -> int:
    if not character_id:
        return 0
    cards = extract_persona_cards(system_prompt)
    if not cards:
        return 0

    inserted = 0
    async with aiosqlite.connect(db_path, timeout=10.0) as db:
        for card in cards:
            cursor = await db.execute(
                """SELECT 1 FROM memory_cards
                   WHERE user_id = ? AND character_id = ? AND title = ?
                     AND status != 'deleted'
                   LIMIT 1""",
                (user_id, character_id, card["title"]),
            )
            if await cursor.fetchone():
                continue
            card_id = generate_id("card_")
            await db.commit()
            await insert_card(
                db_path,
                card_id=card_id,
                user_id=user_id,
                character_id=character_id,
                conversation_id=conversation_id,
                scope="character",
                card_type=card["card_type"],
                title=card["title"],
                content=card["content"],
                summary="Derived from initial character system prompt.",
                importance=0.85,
                confidence=0.75,
                status="approved",
                is_pinned=1,
                evidence_text="system_prompt_persona_import",
                library_id=DEFAULT_MEMORY_LIBRARY_ID,
            )
            inserted += 1
    return inserted
