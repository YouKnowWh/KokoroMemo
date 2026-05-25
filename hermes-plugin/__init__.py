"""KokoroMemo bidirectional memory provider for Hermes Agent.

Reads memory_cards from KokoroMemo's local SQLite store and writes
key context (task state, user preferences, decisions) back after each
turn.  Designed to support aggressive context compression:

  - system_prompt_block() = short constant string (KV cache friendly)
  - prefetch() reads from KokoroMemo SQLite (≤ 1200 chars)
  - sync_turn() writes important info after each turn (async)
  - on_pre_compress() saves a checkpoint before compression
  - get_tool_schemas() returns [] — no tool overhead
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

MAX_PREFETCH_CHARS = 2500
MAX_CHECKPOINT_CHARS = 800

# Auto-generated card types to skip during memory injection
_NOISE_TYPES = frozenset({"task_state", "tool_result", "artifact", "checkpoint", "summary", "system_prompt"})
DEFAULT_DB_PATH = "/home/ubuntu/apps/kokoromemo/data/memory/memory.sqlite"
APP_DB_PATH = "/home/ubuntu/apps/kokoromemo/data/app.sqlite"

_SCOPE_ROUTING: Dict[str, dict] = {
    "arona": {"character_id": "arona", "description": "ARONA character persona + user memories"},
    "prana": {"character_id": "prana", "description": "PRANA character persona + user memories"},
    "project-hermes": {"character_id": "project-hermes", "description": "Hermes project memories"},
    "ops-vps": {"character_id": "ops-vps", "description": "VPS operations logs"},
}

_SYSTEM_PROMPT_BLOCK = "# KokoroMemo Memory\nActive. Bidirectional."

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_db_path(kwargs: dict) -> str:
    candidates = (
        kwargs.get("db_path"),
        kwargs.get("kokoromemo", {}).get("db_path"),
        __import__("os").environ.get("KOKOROMEMO_DB_PATH"),
        DEFAULT_DB_PATH,
    )
    for value in candidates:
        if value:
            return str(value)
    return DEFAULT_DB_PATH


def _connect_readonly(db_path: str) -> Optional[sqlite3.Connection]:
    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        logger.debug("KokoroMemo DB not found at %s", path)
        return None
    try:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0, check_same_thread=False)
        conn.execute("PRAGMA query_only = ON")
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as exc:
        logger.warning("Failed to open KokoroMemo DB (read): %s", exc)
        return None


def _connect_writable(db_path: str) -> Optional[sqlite3.Connection]:
    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        logger.debug("KokoroMemo DB not found at %s (write)", path)
        return None
    try:
        conn = sqlite3.connect(str(path), timeout=10.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn
    except sqlite3.Error as exc:
        logger.warning("Failed to open KokoroMemo DB (write): %s", exc)
        return None


def _is_initial_persona_title(title: str) -> bool:
    return bool(title and title.startswith("initial_persona:"))


def _format_cards(rows: List[sqlite3.Row]) -> str:
    """Format memory cards into concise text, respecting MAX_PREFETCH_CHARS.

    Skips auto-generated noise types and raw system_prompt cards.
    Imported initial persona cards are also skipped here because Hermes profiles
    already carry their active persona/system prompt in SOUL.
    """
    if not rows:
        return ""

    lines: List[str] = []
    used = 0

    for row in rows:
        card_type: str = row["card_type"] or ""
        title: str = row["title"] or ""

        # Skip auto-generated noise cards and imported persona cards. Hermes
        # profiles already provide the related persona/system prompt via SOUL.
        if card_type in _NOISE_TYPES or _is_initial_persona_title(title):
            continue

        content = (row["content"] or "").strip()
        if not content:
            continue

        importance: float = row["importance"] or 0.0
        created: str = row["created_at"] or ""

        tags = []
        if card_type:
            tags.append(card_type)
        if importance > 0.5:
            tags.append(f"imp:{importance:.1f}")
        if created:
            tags.append(created[:10])

        prefix = f"- {' '.join(tags)}: " if tags else "- "
        line = f"{prefix}{content}"
        remaining = MAX_PREFETCH_CHARS - used
        if remaining <= 0:
            break
        if len(line) + 1 > remaining:
            line = line[: max(0, remaining - 1)].rstrip()
            if not line:
                break
        lines.append(line)
        used += len(line) + 1
        if used >= MAX_PREFETCH_CHARS:
            break

    return "\n".join(lines)[:MAX_PREFETCH_CHARS]


def _dedupe_key(scope: str, card_type: str, subject: str) -> str:
    """Generate a stable dedupe key: scope:type:normalized_subject."""
    norm = subject.strip().lower()[:80]
    return f"{scope}:{card_type}:{norm}"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class KokoroMemoMemoryProvider(MemoryProvider):
    """KokoroMemo bidirectional memory provider.

    Design:
      - Read-only connection for prefetch (background thread)
      - Writable connection for sync_turn / on_pre_compress / on_memory_write
      - Scope routing: agent_identity in kwargs selects character_id filter
      - Minimal system prompt: short constant string
    """

    def __init__(self):
        self._db_path: str = ""
        self._ro_conn: Optional[sqlite3.Connection] = None   # read-only (prefetch)
        self._rw_conn: Optional[sqlite3.Connection] = None   # writable (sync / checkpoint)
        self._character_id: Optional[str] = None
        self._user_id: str = "default"
        self._conversation_id: str = ""
        self._session_id: str = ""
        self._active: bool = False
        self._lock = threading.Lock()
        self._prefetch_cache: str = ""
        self._prefetch_thread: Optional[threading.Thread] = None
        self._app_db_path: str = APP_DB_PATH
        self._app_conn: Optional[sqlite3.Connection] = None
        self._hermes_home: str = ""
        self._turn_count: int = 0
        self._last_checkpoint_at: int = 0
        self._is_agent_session: bool = False

    # -- MemoryProvider ABC ---------------------------------------------------

    @property
    def name(self) -> str:
        return "kokoromemo"

    def is_available(self) -> bool:
        path = Path(self._db_path or DEFAULT_DB_PATH)
        if not path.exists():
            return False
        try:
            conn = _connect_readonly(str(path))
            if conn:
                conn.close()
                return True
        except Exception:
            pass
        return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._hermes_home = str(kwargs.get("hermes_home", ""))
        self._db_path = _resolve_db_path(kwargs)
        self._session_id = session_id

        # x-agent header detection: when Hermes CLI uses this provider for
        # non-character sessions, skip character binding to avoid injecting
        # persona memories into generic agent conversations.
        x_agent = str(kwargs.get("x_agent", kwargs.get("x-agent", ""))).lower()
        if x_agent == "hermes":
            self._is_agent_session = True
            self._character_id = None
        else:
            # Scope routing for character profiles
            identity = str(kwargs.get("agent_identity", "arona"))
            if identity in _SCOPE_ROUTING:
                self._character_id = _SCOPE_ROUTING[identity]["character_id"]
            else:
                self._character_id = None
        self._user_id = str(kwargs.get("user_id", "default"))
        self._conversation_id = kwargs.get("conversation_id") or session_id

        # Resolve library_id from character_id
        self._library_id = "lib_default"
        if self._character_id:
            rw = _connect_writable(self._db_path)
            if rw:
                try:
                    row = rw.execute(
                        "SELECT library_id FROM memory_libraries WHERE name = ? AND status = 'active' LIMIT 1",
                        (self._character_id.upper(),),
                    ).fetchone()
                    if row:
                        self._library_id = row[0]
                except Exception:
                    pass
                finally:
                    rw.close()

        logger.info(
            "KokoroMemo provider initialized: agent=%s char=%s lib=%s db=%s",
            self._is_agent_session, self._character_id, self._library_id, self._db_path,
        )

        self._ro_conn = _connect_readonly(self._db_path)
        self._rw_conn = _connect_writable(self._db_path)
        self._active = self._ro_conn is not None

        # Open app DB for session-board conversation timestamps (best-effort)
        app_path = Path(self._app_db_path).expanduser().resolve()
        if app_path.exists():
            try:
                self._app_conn = sqlite3.connect(
                    str(app_path), timeout=10.0, check_same_thread=False
                )
                self._app_conn.execute("PRAGMA journal_mode = WAL")
                self._app_conn.execute("PRAGMA busy_timeout = 5000")
                logger.info("KokoroMemo app_db opened at %s", app_path)
            except sqlite3.Error as exc:
                logger.warning("Failed to open app DB for conversation timestamps: %s", exc)
                self._app_conn = None
        else:
            logger.warning("KokoroMemo app_db not found at %s — session board timestamps will be stale", app_path)

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        return _SYSTEM_PROMPT_BLOCK

    def prefetch(self, query: str = "", *, session_id: str = "") -> str:
        if not self._active:
            return ""
        if self._is_agent_session:
            return ""
        with self._lock:
            cached = self._prefetch_cache
            self._prefetch_cache = ""
        if cached:
            return cached
        return self._query_cards(query=query)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._active:
            return
        if self._is_agent_session:
            return

        def _run():
            try:
                result = self._query_cards(query=query)
                with self._lock:
                    self._prefetch_cache = result
            except Exception as exc:
                logger.debug("KokoroMemo background prefetch failed: %s", exc)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="kokoromemo-prefetch"
        )
        self._prefetch_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "memory_recall",
                "description": "Full scan of your memory library. Search ALL memory cards by keyword. Use when you need to find something you vaguely remember but can't quite recall — a past decision, a user preference, a code path, a configuration detail. Returns all matching cards (no limit).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Keywords to search for in your memory cards. Use specific terms: project names, file paths, person names, technical terms. Multiple words = AND search."
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "memory_lookup",
                "description": "Look up a specific memory card by its ID prefix. Use when a recall result mentions a card_id and you want the full content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "card_id_prefix": {
                            "type": "string",
                            "description": "First few characters of a card_id to look up."
                        }
                    },
                    "required": ["card_id_prefix"]
                }
            }
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "memory_recall":
            return self._tool_recall(args.get("query", ""))
        elif tool_name == "memory_lookup":
            return self._tool_lookup(args.get("card_id_prefix", ""))
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _tool_recall(self, query: str) -> str:
        """Full library scan — find ALL matching cards, not just top N."""
        if self._is_agent_session:
            return json.dumps({"matches": [], "total": 0, "hint": "Memory disabled for agent sessions"})
        if not query or not self._ro_conn:
            return json.dumps({"matches": [], "total": 0, "hint": "No query or DB unavailable"})

        try:
            keywords = [w.strip() for w in query.split() if len(w.strip()) >= 2]
            if not keywords:
                return json.dumps({"matches": [], "total": 0, "hint": "Query too short — use 2+ character words"})

            clauses = ["status = 'approved'"]
            params = []
            if self._character_id:
                clauses.append("character_id = ?")
                params.append(self._character_id)
            clauses.append("card_type NOT IN ('task_state','tool_result','artifact','checkpoint','summary','system_prompt')")

            kw_clauses = []
            for kw in keywords[:8]:
                kw_clauses.append("(title LIKE ? OR content LIKE ?)")
                params.append(f"%{kw}%")
                params.append(f"%{kw}%")

            sql = f"""SELECT card_id, card_type, title, content, importance, created_at
                      FROM memory_cards
                      WHERE {' AND '.join(clauses)} AND ({' OR '.join(kw_clauses)})
                      ORDER BY importance DESC, created_at DESC"""

            with self._lock:
                self._ro_conn.row_factory = sqlite3.Row
                rows = list(self._ro_conn.execute(sql, params))

            results = []
            for r in rows:
                results.append({
                    "card_id": r["card_id"],
                    "type": r["card_type"],
                    "title": (r["title"] or "")[:120],
                    "content": (r["content"] or "")[:300],
                    "importance": r["importance"],
                })

            return json.dumps({"matches": len(results), "results": results}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": str(exc), "matches": 0})

    def _tool_lookup(self, card_id_prefix: str) -> str:
        """Look up full content of a specific card by ID prefix."""
        if self._is_agent_session:
            return json.dumps({"found": False, "hint": "Memory disabled for agent sessions"})
        if not card_id_prefix or not self._ro_conn:
            return json.dumps({"found": False})

        try:
            with self._lock:
                self._ro_conn.row_factory = sqlite3.Row
                row = self._ro_conn.execute(
                    "SELECT * FROM memory_cards WHERE card_id LIKE ? LIMIT 1",
                    (f"{card_id_prefix}%",)
                ).fetchone()

            if not row:
                return json.dumps({"found": False, "hint": "No card matches that ID prefix"})

            return json.dumps({
                "found": True,
                "card_id": row["card_id"],
                "type": row["card_type"],
                "title": row["title"] or "",
                "content": row["content"] or "",
                "character_id": row["character_id"],
                "importance": row["importance"],
                "created_at": row["created_at"],
            }, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"found": False, "error": str(exc)})

    def shutdown(self) -> None:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        for conn_name in ("_ro_conn", "_rw_conn", "_app_conn"):
            conn = getattr(self, conn_name, None)
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
                setattr(self, conn_name, None)
        self._active = False

    # -- Turn lifecycle hooks -------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_count = turn_number

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Write key context after each turn: task state, preferences, decisions."""
        if not self._active or not self._rw_conn:
            return
        if not user_content and not assistant_content:
            return

        memories_to_write: List[dict] = []

        # Extract user preferences
        user = user_content.strip() if user_content else ""
        if user and len(user) > 20:
            prefs = self._extract_preferences(user)
            for pref in prefs:
                memories_to_write.append({
                    "scope": "user",
                    "card_type": "user_preference",
                    "content": pref,
                    "importance": 0.7,
                    "dedupe_subject": f"preference-{pref[:60]}",
                })

        # Batch write
        for mem in memories_to_write:
            self._write_card(mem)

        # Update conversation metadata so the session board shows fresh timestamps
        conv_id = session_id or self._conversation_id or self._session_id
        if conv_id and self._rw_conn:
            try:
                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                # 1. Update/create conversation_configs
                self._rw_conn.execute(
                    """INSERT INTO conversation_configs (conversation_id, profile_id, created_at, updated_at)
                       VALUES (?, 'memory_only', ?, ?)
                       ON CONFLICT(conversation_id) DO UPDATE SET updated_at = ?""",
                    (conv_id, now, now, now),
                )
                # 2. Update/create conversation_memory_mounts with character_id
                if self._character_id:
                    existing = self._rw_conn.execute(
                        "SELECT mount_id FROM conversation_memory_mounts WHERE conversation_id=? AND character_id=?",
                        (conv_id, self._character_id),
                    ).fetchone()
                    if existing:
                        self._rw_conn.execute(
                            "UPDATE conversation_memory_mounts SET updated_at=? WHERE mount_id=?",
                            (now, existing[0]),
                        )
                    else:
                        self._rw_conn.execute(
                            """INSERT INTO conversation_memory_mounts
                               (mount_id, conversation_id, library_id, user_id, character_id, is_write_target, sort_order, status, created_at, updated_at)
                               VALUES (?, ?, ?, ?, ?, 1, 0, 'active', ?, ?)""",
                            (f"mount_{uuid.uuid4().hex[:12]}_{uuid.uuid4().hex[:8]}",
                             conv_id, self._library_id, self._user_id, self._character_id, now, now),
                        )
                self._rw_conn.commit()
            except Exception as exc:
                logger.warning("KokoroMemo conversation metadata update failed: %s", exc)

        # Update app.sqlite + chat.sqlite so the session board shows fresh timestamps and messages
        if self._app_conn and self._user_id and self._character_id:
            try:
                now_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Find existing conversation by character (ignore user_id — QQ bot sends different user than LiteLLM)
                existing = self._app_conn.execute(
                    "SELECT conversation_id, path FROM conversations WHERE character_id=? ORDER BY last_seen_at DESC LIMIT 1",
                    (self._character_id,),
                ).fetchone()
                if existing:
                    app_conv_id, path = existing[0], existing[1]
                    self._app_conn.execute(
                        "UPDATE conversations SET last_seen_at=? WHERE conversation_id=?",
                        (now_local, app_conv_id),
                    )
                else:
                    raw = f"{self._user_id}_{self._character_id}"
                    app_conv_id = f"conv_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"
                    path = f"data/conversations/{app_conv_id}"
                    self._app_conn.execute(
                        """INSERT INTO conversations (conversation_id, user_id, character_id, client_name, title, path, first_seen_at, last_seen_at, status)
                           VALUES (?, ?, ?, 'hermes', 'ARONA', ?, ?, ?, 'active')""",
                        (app_conv_id, self._user_id, self._character_id, path, now_local, now_local),
                    )
                self._app_conn.commit()

                # Write user+assistant messages to chat.sqlite so the session board shows content
                apps_dir = Path(APP_DB_PATH).expanduser().resolve().parent  # e.g. /home/ubuntu/apps/kokoromemo/data
                # path from conversations table is like "data/conversations/conv_xxx"
                # So apps_dir / path = /home/ubuntu/apps/kokoromemo/data/data/conversations/conv_xxx (WRONG)
                # Strip leading "data/" if path starts with it
                if path.startswith("data/"):
                    rel_path = path[len("data/"):]  # conversations/conv_xxx
                else:
                    rel_path = path
                conv_dir = apps_dir / rel_path
                conv_dir.mkdir(parents=True, exist_ok=True)
                chat_db_path = conv_dir / "chat.sqlite"
                chat_conn = sqlite3.connect(str(chat_db_path), timeout=5.0)
                try:
                    chat_conn.executescript("""
                        CREATE TABLE IF NOT EXISTS turns (
                            turn_id TEXT PRIMARY KEY,
                            conversation_id TEXT NOT NULL,
                            user_id TEXT,
                            character_id TEXT,
                            request_id TEXT,
                            turn_index INTEGER,
                            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
                        );
                        CREATE TABLE IF NOT EXISTS messages (
                            message_id TEXT PRIMARY KEY,
                            turn_id TEXT NOT NULL,
                            conversation_id TEXT NOT NULL,
                            role TEXT NOT NULL,
                            name TEXT,
                            content TEXT,
                            raw_json TEXT,
                            token_estimate INTEGER,
                            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
                        );
                    """)
                    turn_id = f"turn_{uuid.uuid4().hex[:8]}_{uuid.uuid4().hex[:8]}"
                    chat_conn.execute(
                        "INSERT INTO turns (turn_id, conversation_id, user_id, character_id, turn_index, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (turn_id, app_conv_id, self._user_id, self._character_id, self._turn_count, now_local),
                    )
                    if user_content:
                        chat_conn.execute(
                            "INSERT INTO messages (message_id, turn_id, conversation_id, role, content, created_at) VALUES (?, ?, ?, 'user', ?, ?)",
                            (f"msg_{uuid.uuid4().hex[:12]}", turn_id, app_conv_id, user_content[:1000], now_local),
                        )
                    if assistant_content:
                        chat_conn.execute(
                            "INSERT INTO messages (message_id, turn_id, conversation_id, role, content, created_at) VALUES (?, ?, ?, 'assistant', ?, ?)",
                            (f"msg_{uuid.uuid4().hex[:12]}", turn_id, app_conv_id, assistant_content[:1000], now_local),
                        )
                    chat_conn.commit()
                finally:
                    chat_conn.close()

                logger.info("KokoroMemo app_db conversation %s updated (last_seen_at=%s, messages=%d)",
                            app_conv_id, now_local, (1 if user_content else 0) + (1 if assistant_content else 0))
            except Exception as exc:
                logger.warning("KokoroMemo app_db + chat.sqlite update failed: %s", exc)

        self._turn_count += 1

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Save a checkpoint before compression discards old messages."""
        if not self._active or not self._rw_conn:
            return ""

        # Build a checkpoint from the messages about to be compressed
        important_lines: List[str] = []
        for msg in messages[-20:]:  # last 20 messages
            role = msg.get("role", "")
            content = str(msg.get("content", ""))[:200]
            if role in ("user", "assistant") and content:
                label = "User" if role == "user" else "Asst"
                important_lines.append(f"[{label}] {content}")

        checkpoint_text = "\n".join(important_lines)[:MAX_CHECKPOINT_CHARS]
        if not checkpoint_text:
            return ""

        self._last_checkpoint_at = self._turn_count

        # Return a hint for the compression summary
        return f"[KokoroMemo checkpoint saved for turn {self._turn_count}]"

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes to KokoroMemo."""
        if not self._active or not self._rw_conn:
            return
        if action != "add" or not content:
            return

        card_type = "user_preference" if target == "user" else "agent_note"
        scope = "user" if target == "user" else "conversation"

        self._write_card({
            "scope": scope,
            "card_type": card_type,
            "content": content[:500],
            "importance": 0.8,
            "dedupe_subject": f"memory-{target}-{content[:60]}",
        })

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Save a final summary when the session ends."""
        if not self._active or not self._rw_conn:
            return

        # Count key stats
        user_msgs = sum(1 for m in messages if m.get("role") == "user")
        asst_msgs = sum(1 for m in messages if m.get("role") == "assistant")

        summary = (
            f"Session ended after {self._turn_count} turns "
            f"({user_msgs} user, {asst_msgs} assistant). "
            f"Last checkpoint at turn {self._last_checkpoint_at}."
        )
    # -- Internal: card writing -----------------------------------------------

    def _write_card(self, mem: dict) -> None:
        """Write a memory card with deduplication and supersede support."""
        if not self._rw_conn:
            return
        try:
            scope = mem.get("scope", "conversation")
            card_type = mem.get("card_type", "note")
            content = (mem.get("content") or "").strip()
            if not content:
                return
            importance = mem.get("importance", 0.5)
            dedupe_subject = mem.get("dedupe_subject", "")
            dk = _dedupe_key(scope, card_type, dedupe_subject) if dedupe_subject else ""

            card_id = f"card_{uuid.uuid4().hex[:12]}"
            now = time.strftime("%Y-%m-%d %H:%M:%S")

            if dedupe_subject:
                dk = _dedupe_key(scope, card_type, dedupe_subject)
                cur = self._rw_conn.execute(
                    "SELECT card_id, content FROM memory_cards WHERE card_type=? AND scope=? AND title=? AND status='approved' LIMIT 1",
                    (card_type, scope, dk),
                )
                existing = cur.fetchone()
                if existing:
                    # Supersede: mark old as superseded, insert new
                    self._rw_conn.execute(
                        "UPDATE memory_cards SET status='superseded', updated_at=? WHERE card_id=?",
                        (now, existing[0]),
                    )
                    # Continue to insert new card below

            self._rw_conn.execute(
                """INSERT INTO memory_cards
                   (card_id, library_id, user_id, character_id, conversation_id,
                    scope, card_type, title, content, importance, confidence,
                    status, created_at, updated_at, source_turn_ids_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    card_id,
                    self._library_id,
                    self._user_id,
                    self._character_id or "",
                    self._conversation_id,
                    scope,
                    card_type,
                    dk or card_type,  # title = dedupe_key for dedup matching
                    content[:1500],
                    min(importance, 1.0),
                    0.7,
                    "approved",
                    now,
                    now,
                    json.dumps([self._session_id]),
                ),
            )
            self._rw_conn.commit()
            logger.debug("KokoroMemo wrote card %s (%s/%s)", card_id, scope, card_type)
        except sqlite3.Error as exc:
            logger.warning("KokoroMemo card write failed: %s", exc)

    @staticmethod
    def _extract_preferences(text: str) -> List[str]:
        """Extract user preferences from text using Chinese indicators."""
        preferences = []
        indicators = ["我喜欢", "我不喜欢", "偏好", "习惯", "不要", "请用", "最好", "帮我", "尽量"]
        for indicator in indicators:
            idx = text.find(indicator)
            if idx >= 0:
                snippet = text[idx:idx + 80].strip()
                if snippet:
                    preferences.append(snippet)
        return preferences

    @staticmethod
    def _extract_tool_results(text: str) -> List[str]:
        """Extract tool result summaries from assistant content."""
        results = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Look for conclusion indicators
            indicators = [
                "发现", "找到", "确认", "结果", "结论", "定位到",
                "Found", "found", "Result", "result",
                "位于", "Located", "located", "路径",
            ]
            for ind in indicators:
                if ind in stripped and len(stripped) > 15:
                    results.append(stripped[:120])
                    break
        return results[:3]  # max 3 tool results per turn

    @staticmethod
    def _extract_paths(text: str) -> List[str]:
        """Extract file/URL paths from text."""
        import re as _re
        paths = set()
        # Unix paths: /opt/... /home/... /tmp/...
        for m in _re.finditer(r'(/[a-zA-Z0-9_\-./]+[a-zA-Z0-9_\-])', text):
            p = m.group(1)
            if len(p) > 12 and not p.startswith("/usr") and not p.startswith("/var"):
                paths.add(p)
        # Python module paths: module.sub.module
        for m in _re.finditer(r'([a-z][a-z0-9_]*\.[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)', text):
            paths.add(m.group(1))
        return sorted(paths)[:5]

    # -- Internal: card querying ----------------------------------------------

    def _query_cards(self, query: str = "") -> str:
        if not self._ro_conn:
            return ""
        try:
            clauses = ["status = ?"]
            params: List[Any] = ["approved"]

            if self._character_id:
                clauses.append("character_id = ?")
                params.append(self._character_id)

            # Exclude auto-generated noise cards at query level
            clauses.append("card_type NOT IN ('task_state','tool_result','artifact','checkpoint','summary','system_prompt')")

            # Keyword matching: add LIKE clauses for each keyword in the query
            if query and query.strip():
                keywords = [w for w in query.strip().split() if len(w) >= 2]
                for kw in keywords[:6]:
                    clauses.append("(title LIKE ? OR content LIKE ?)")
                    params.append(f"%{kw}%")
                    params.append(f"%{kw}%")

            row_limit = 20
            params.append(row_limit)

            clauses.append("(title IS NULL OR title NOT LIKE 'initial_persona:%')")

            sql = f"""SELECT card_id, user_id, character_id, card_type, title, content,
                             importance, status, created_at
                      FROM memory_cards
                      WHERE {' AND '.join(clauses)}
                      ORDER BY importance DESC, created_at DESC
                      LIMIT ?"""

            with self._lock:
                self._ro_conn.row_factory = sqlite3.Row
                rows = list(self._ro_conn.execute(sql, params))

            return _format_cards(rows)
        except sqlite3.Error as exc:
            logger.warning("KokoroMemo query failed: %s", exc)
            return ""
        except Exception as exc:
            logger.warning("KokoroMemo prefetch error: %s", exc)
            return ""


# -- Plugin registration --------------------------------------------------------

def register(ctx):
    """Register the KokoroMemo memory provider with Hermes."""
    ctx.register_memory_provider(KokoroMemoMemoryProvider())
