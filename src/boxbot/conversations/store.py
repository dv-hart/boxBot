"""SQLite-backed persistence for conversation threads.

Schema:
    conversations
        conversation_id        TEXT PRIMARY KEY
        channel                TEXT       — "whatsapp" today; voice/trigger
                                           may opt in later
        channel_key            TEXT       — "whatsapp:+15551234567"
        started_at_iso         TEXT
        last_activity_at_iso   TEXT
        participants_json      TEXT       — JSON array of names
        state                  TEXT       — "active" | "extracted"
        extracted_at_iso       TEXT NULL
        summary                TEXT NULL  — populated by extraction

    conversation_turns
        id               INTEGER PK AUTOINCREMENT
        conversation_id  TEXT  REFERENCES conversations
        turn_index       INTEGER          — monotonic per conversation
        role             TEXT             — "user" | "assistant"
        content_json     TEXT             — the raw Anthropic message dict
        metadata_json    TEXT NULL        — speaker_name, attachment_path, …
        created_at_iso   TEXT
        UNIQUE(conversation_id, turn_index)

Two access patterns drive the indexes:
    1. Lookup-or-create on inbound:
         WHERE channel_key = ? AND state = 'active'
                AND last_activity_at_iso > <now - window>
    2. Sweep extractable:
         WHERE state = 'active' AND last_activity_at_iso < <now - window>

Usage:
    store = ConversationStore()
    await store.initialize()

    rec = await store.get_active("whatsapp:+15551234567",
                                 max_inactive_seconds=14400)
    if rec is None:
        rec = await store.create(channel="whatsapp",
                                 channel_key="whatsapp:+15551234567",
                                 participants={"Jacob"})

    await store.append_turn(rec.conversation_id, role="user",
                            content={"role": "user", "content": "hi"})

    # Periodically:
    expired = await store.list_extractable(max_inactive_seconds=14400)
    for r in expired:
        thread = await store.get_thread(r.conversation_id)
        # ... extract ...
        await store.mark_extracted(r.conversation_id, summary="…")
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from boxbot.core.paths import CONVERSATIONS_DIR

logger = logging.getLogger(__name__)


DB_PATH = CONVERSATIONS_DIR / "conversations.db"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id      TEXT PRIMARY KEY,
    channel              TEXT NOT NULL,
    channel_key          TEXT NOT NULL,
    started_at_iso       TEXT NOT NULL,
    last_activity_at_iso TEXT NOT NULL,
    participants_json    TEXT NOT NULL DEFAULT '[]',
    state                TEXT NOT NULL DEFAULT 'active',
    extracted_at_iso     TEXT,
    summary              TEXT
);

CREATE INDEX IF NOT EXISTS idx_conv_lookup
    ON conversations(channel_key, state, last_activity_at_iso);

CREATE INDEX IF NOT EXISTS idx_conv_extractable
    ON conversations(state, last_activity_at_iso);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE,
    turn_index      INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content_json    TEXT NOT NULL,
    metadata_json   TEXT,
    created_at_iso  TEXT NOT NULL,
    UNIQUE(conversation_id, turn_index)
);

CREATE INDEX IF NOT EXISTS idx_turns_conv
    ON conversation_turns(conversation_id, turn_index);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_conversation_id() -> str:
    return f"conv_{uuid.uuid4().hex[:12]}"


@dataclass
class ConversationRecord:
    """Persisted metadata for one conversation."""

    conversation_id: str
    channel: str
    channel_key: str
    started_at_iso: str
    last_activity_at_iso: str
    participants: list[str]
    state: str
    extracted_at_iso: str | None
    summary: str | None

    @classmethod
    def _from_row(cls, row: aiosqlite.Row) -> ConversationRecord:
        return cls(
            conversation_id=row["conversation_id"],
            channel=row["channel"],
            channel_key=row["channel_key"],
            started_at_iso=row["started_at_iso"],
            last_activity_at_iso=row["last_activity_at_iso"],
            participants=json.loads(row["participants_json"] or "[]"),
            state=row["state"],
            extracted_at_iso=row["extracted_at_iso"],
            summary=row["summary"],
        )


@dataclass
class TurnRecord:
    """One persisted turn — wraps the raw Anthropic message dict."""

    turn_index: int
    role: str
    content: dict[str, Any]
    metadata: dict[str, Any] | None
    created_at_iso: str

    @classmethod
    def _from_row(cls, row: aiosqlite.Row) -> TurnRecord:
        return cls(
            turn_index=row["turn_index"],
            role=row["role"],
            content=json.loads(row["content_json"]),
            metadata=(
                json.loads(row["metadata_json"])
                if row["metadata_json"] else None
            ),
            created_at_iso=row["created_at_iso"],
        )


class ConversationStore:
    """SQLite-backed conversation persistence.

    Single-writer (the agent process) — no cross-process coordination.
    Open one connection at startup, hold it for the lifetime of the
    process. WAL mode enabled for crash safety.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or DB_PATH
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create the directory, open the connection, and apply schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        logger.info("Conversation store initialized at %s", self._db_path)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("ConversationStore not initialized")
        return self._db

    # ------------------------------------------------------------------
    # Conversation CRUD
    # ------------------------------------------------------------------

    async def create(
        self,
        *,
        channel: str,
        channel_key: str,
        participants: set[str] | None = None,
        conversation_id: str | None = None,
    ) -> ConversationRecord:
        """Create a fresh active conversation row."""
        db = self._require_db()
        cid = conversation_id or _generate_conversation_id()
        now = _now_iso()
        participants_json = json.dumps(sorted(participants or []))
        await db.execute(
            "INSERT INTO conversations "
            "(conversation_id, channel, channel_key, "
            " started_at_iso, last_activity_at_iso, "
            " participants_json, state) "
            "VALUES (?, ?, ?, ?, ?, ?, 'active')",
            (cid, channel, channel_key, now, now, participants_json),
        )
        await db.commit()
        return ConversationRecord(
            conversation_id=cid,
            channel=channel,
            channel_key=channel_key,
            started_at_iso=now,
            last_activity_at_iso=now,
            participants=sorted(participants or []),
            state="active",
            extracted_at_iso=None,
            summary=None,
        )

    async def get(self, conversation_id: str) -> ConversationRecord | None:
        db = self._require_db()
        cursor = await db.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )
        row = await cursor.fetchone()
        return ConversationRecord._from_row(row) if row else None

    async def get_active(
        self,
        channel_key: str,
        *,
        max_inactive_seconds: float,
    ) -> ConversationRecord | None:
        """Return the active conversation for ``channel_key`` if it's
        still inside the rolling window, else None.

        "Active" means ``state='active'`` AND ``last_activity_at`` is
        within ``max_inactive_seconds`` of now. A conversation that has
        gone quiet long enough to be eligible for extraction is treated
        as no-longer-active here so a new inbound starts a fresh thread.
        """
        db = self._require_db()
        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(seconds=max_inactive_seconds)
        ).isoformat()
        cursor = await db.execute(
            "SELECT * FROM conversations "
            "WHERE channel_key = ? AND state = 'active' "
            "  AND last_activity_at_iso > ? "
            "ORDER BY last_activity_at_iso DESC LIMIT 1",
            (channel_key, cutoff),
        )
        row = await cursor.fetchone()
        return ConversationRecord._from_row(row) if row else None

    async def list_active(
        self,
        *,
        channel: str | None = None,
        max_inactive_seconds: float | None = None,
    ) -> list[ConversationRecord]:
        """Return all rows that are still considered active.

        Used at agent startup to warm-load conversations whose threads
        survived the restart. Pass ``channel="whatsapp"`` to scope.
        """
        db = self._require_db()
        sql = "SELECT * FROM conversations WHERE state = 'active'"
        args: list[Any] = []
        if channel is not None:
            sql += " AND channel = ?"
            args.append(channel)
        if max_inactive_seconds is not None:
            cutoff = (
                datetime.now(timezone.utc)
                - timedelta(seconds=max_inactive_seconds)
            ).isoformat()
            sql += " AND last_activity_at_iso > ?"
            args.append(cutoff)
        sql += " ORDER BY last_activity_at_iso DESC"
        cursor = await db.execute(sql, args)
        rows = await cursor.fetchall()
        return [ConversationRecord._from_row(r) for r in rows]

    async def list_extractable(
        self,
        *,
        max_inactive_seconds: float,
        channel: str | None = None,
    ) -> list[ConversationRecord]:
        """Return active conversations whose window has expired.

        These are the rows the sweep should extract. Caller is
        responsible for racing safely with new inbound (the row
        transitions to 'extracted' atomically via ``mark_extracted``).
        """
        db = self._require_db()
        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(seconds=max_inactive_seconds)
        ).isoformat()
        sql = (
            "SELECT * FROM conversations "
            "WHERE state = 'active' AND last_activity_at_iso <= ?"
        )
        args: list[Any] = [cutoff]
        if channel is not None:
            sql += " AND channel = ?"
            args.append(channel)
        cursor = await db.execute(sql, args)
        rows = await cursor.fetchall()
        return [ConversationRecord._from_row(r) for r in rows]

    async def update_participants(
        self,
        conversation_id: str,
        participants: set[str],
    ) -> None:
        db = self._require_db()
        await db.execute(
            "UPDATE conversations SET participants_json = ? "
            "WHERE conversation_id = ?",
            (json.dumps(sorted(participants)), conversation_id),
        )
        await db.commit()

    async def touch(self, conversation_id: str) -> None:
        """Bump ``last_activity_at_iso`` to now without writing a turn.

        Useful when the agent advances state without producing a new
        message (rare). Most callers will get this for free via
        ``append_turn``.
        """
        db = self._require_db()
        await db.execute(
            "UPDATE conversations SET last_activity_at_iso = ? "
            "WHERE conversation_id = ? AND state = 'active'",
            (_now_iso(), conversation_id),
        )
        await db.commit()

    async def mark_extracted(
        self,
        conversation_id: str,
        *,
        summary: str | None = None,
    ) -> bool:
        """Atomically transition active → extracted.

        Returns True if the row flipped, False if it was already
        extracted (so concurrent sweepers don't double-extract).
        """
        db = self._require_db()
        cursor = await db.execute(
            "UPDATE conversations "
            "SET state = 'extracted', extracted_at_iso = ?, summary = ? "
            "WHERE conversation_id = ? AND state = 'active'",
            (_now_iso(), summary, conversation_id),
        )
        await db.commit()
        return cursor.rowcount > 0

    async def reactivate(self, conversation_id: str) -> None:
        """Flip extracted → active. Only used by tests right now."""
        db = self._require_db()
        await db.execute(
            "UPDATE conversations SET state = 'active', "
            "extracted_at_iso = NULL, last_activity_at_iso = ? "
            "WHERE conversation_id = ?",
            (_now_iso(), conversation_id),
        )
        await db.commit()

    async def delete(self, conversation_id: str) -> None:
        """Hard-delete a conversation and all its turns."""
        db = self._require_db()
        await db.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )
        await db.commit()

    # ------------------------------------------------------------------
    # Turn CRUD
    # ------------------------------------------------------------------

    async def append_turn(
        self,
        conversation_id: str,
        *,
        role: str,
        content: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Append a turn and bump ``last_activity_at_iso``.

        Returns the new ``turn_index``. Wrapped in a transaction so the
        next-index lookup, insert, and parent-update commit together.
        """
        db = self._require_db()
        async with db.execute("BEGIN"):
            pass
        try:
            cursor = await db.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_idx "
                "FROM conversation_turns WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            next_idx = int(row["next_idx"]) if row else 0
            now = _now_iso()
            await db.execute(
                "INSERT INTO conversation_turns "
                "(conversation_id, turn_index, role, "
                " content_json, metadata_json, created_at_iso) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    conversation_id, next_idx, role,
                    json.dumps(content),
                    json.dumps(metadata) if metadata else None,
                    now,
                ),
            )
            await db.execute(
                "UPDATE conversations SET last_activity_at_iso = ? "
                "WHERE conversation_id = ?",
                (now, conversation_id),
            )
            await db.commit()
            return next_idx
        except Exception:
            await db.rollback()
            raise

    async def append_turns(
        self,
        conversation_id: str,
        turns: list[dict[str, Any]],
    ) -> int:
        """Bulk-append assistant/tool turns produced by one generation.

        ``turns`` is a list of raw Anthropic message dicts. Returns the
        number of turns written.
        """
        if not turns:
            return 0
        db = self._require_db()
        try:
            cursor = await db.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_idx "
                "FROM conversation_turns WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            next_idx = int(row["next_idx"]) if row else 0
            now = _now_iso()
            payload = []
            for offset, msg in enumerate(turns):
                payload.append((
                    conversation_id,
                    next_idx + offset,
                    msg.get("role", "assistant"),
                    json.dumps(msg),
                    None,
                    now,
                ))
            await db.executemany(
                "INSERT INTO conversation_turns "
                "(conversation_id, turn_index, role, "
                " content_json, metadata_json, created_at_iso) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                payload,
            )
            await db.execute(
                "UPDATE conversations SET last_activity_at_iso = ? "
                "WHERE conversation_id = ?",
                (now, conversation_id),
            )
            await db.commit()
            return len(payload)
        except Exception:
            await db.rollback()
            raise

    async def get_thread(
        self,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        """Return every turn's raw content dict, ordered by turn_index.

        This is the rehydration path — the returned list is suitable
        for assigning straight to ``Conversation._thread``.
        """
        db = self._require_db()
        cursor = await db.execute(
            "SELECT content_json FROM conversation_turns "
            "WHERE conversation_id = ? ORDER BY turn_index ASC",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        return [json.loads(r["content_json"]) for r in rows]

    async def get_turns(
        self,
        conversation_id: str,
    ) -> list[TurnRecord]:
        """Return the full TurnRecord list — used by tests + diagnostics."""
        db = self._require_db()
        cursor = await db.execute(
            "SELECT * FROM conversation_turns "
            "WHERE conversation_id = ? ORDER BY turn_index ASC",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        return [TurnRecord._from_row(r) for r in rows]
