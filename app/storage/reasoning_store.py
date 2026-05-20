"""Persistent reasoning_content store for DeepSeek multi-turn tool calls.

Survives server restarts. Uses sync sqlite3 (independent of the async DB).
Auto-cleans entries older than 1 hour.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("kokoromemo.reasoning_store")

# DB path: same directory as the main data dir
_db_path: str | None = None
_db: sqlite3.Connection | None = None


def init_reasoning_store(data_dir: str) -> None:
    """Initialize the reasoning store database."""
    global _db_path, _db
    _db_path = str(Path(data_dir) / "reasoning_store.sqlite")
    _db = sqlite3.connect(_db_path, check_same_thread=False)
    _db.execute("""
        CREATE TABLE IF NOT EXISTS reasoning_store (
            tool_call_id TEXT PRIMARY KEY,
            reasoning TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    _db.commit()
    logger.info("Reasoning store initialized at %s", _db_path)


def _ensure_db() -> sqlite3.Connection | None:
    global _db
    if _db is None:
        logger.warning("Reasoning store not initialized")
        return None
    return _db


def save_reasoning(tool_call_id: str, reasoning: str) -> None:
    """Save reasoning_content for a tool_call_id."""
    if not tool_call_id or not reasoning:
        return
    db = _ensure_db()
    if db is None:
        return
    try:
        db.execute(
            "INSERT OR REPLACE INTO reasoning_store (tool_call_id, reasoning, created_at) VALUES (?, ?, ?)",
            (tool_call_id, reasoning, time.time()),
        )
        db.commit()
    except Exception as e:
        logger.warning("Failed to save reasoning for %s: %s", tool_call_id[:20], e)


def get_reasoning(tool_call_id: str) -> str:
    """Get reasoning_content for a tool_call_id."""
    if not tool_call_id:
        return ""
    db = _ensure_db()
    if db is None:
        return ""
    try:
        row = db.execute(
            "SELECT reasoning FROM reasoning_store WHERE tool_call_id = ?",
            (tool_call_id,),
        ).fetchone()
        return row[0] if row else ""
    except Exception as e:
        logger.warning("Failed to get reasoning for %s: %s", tool_call_id[:20], e)
        return ""


def cleanup_old_reasoning(max_age_seconds: float = 3600.0) -> None:
    """Remove reasoning entries older than max_age_seconds."""
    db = _ensure_db()
    if db is None:
        return
    cutoff = time.time() - max_age_seconds
    try:
        db.execute("DELETE FROM reasoning_store WHERE created_at < ?", (cutoff,))
        db.commit()
    except Exception as e:
        logger.warning("Failed to cleanup old reasoning: %s", e)
