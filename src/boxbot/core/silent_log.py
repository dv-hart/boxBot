"""Silent-turn logging for multi-speaker conversations.

When boxBot hears an utterance and decides NOT to respond, the decision
(reason, addressed_to, urgency, optional silent_context_note) is persisted
here alongside the transcript snippet and the speakers present at the time.

Purpose: feed post-conversation memory extraction with ambient observations
— things BB noticed while listening silently that may be worth remembering.

Schema lives in ``data/memory/memory.db`` (shared with the fact memory
store for locality; the extraction pass reads both tables).

Usage:
    from boxbot.core.silent_log import log_silent_turn, get_silent_turns

    await log_silent_turn(
        conversation_id="conv_abc123",
        decision={
            "respond": False,
            "reason": "Jacob was talking to Sarah, not me.",
            "addressed_to": "other_person",
            "urgency": "none",
            "silent_context_note": "Sarah mentioned a 3pm dentist appt Thursday.",
        },
        transcript_snippet="[Jacob]: Can you grab the kids at 3?",
        speakers=["Jacob", "Sarah"],
    )

    turns = await get_silent_turns("conv_abc123")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# Reuse the memory DB for locality with extraction.
# Matches boxbot.memory.store.DB_PATH.
_DB_DIR = Path("data/memory")
_DB_PATH = _DB_DIR / "memory.db"


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS silent_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    reason TEXT NOT NULL,
    addressed_to TEXT NOT NULL,
    urgency TEXT NOT NULL,
    silent_context_note TEXT,
    transcript_snippet TEXT,
    speakers_present TEXT
);
CREATE INDEX IF NOT EXISTS idx_silent_turns_conv ON silent_turns(conversation_id);
"""


def _resolve_db_path(db_path: str | Path | None) -> Path:
    """Return the effective DB path (explicit arg, module override, or default)."""
    if db_path is not None:
        return Path(db_path)
    return _DB_PATH


async def _ensure_schema(db_path: Path) -> None:
    """Create the silent_turns table/index if they do not exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(db_path)) as db:
        await db.executescript(_SCHEMA_SQL)
        await db.commit()


async def log_silent_turn(
    conversation_id: str,
    decision: dict[str, Any],
    transcript_snippet: str | None = None,
    speakers: list[str] | None = None,
    *,
    db_path: str | Path | None = None,
) -> int:
    """Persist a silent-turn decision.

    Args:
        conversation_id: The current conversation ID.
        decision: The parsed decision dict from the model. Must contain
            ``reason``, ``addressed_to``, ``urgency``; may contain
            ``silent_context_note``.
        transcript_snippet: The utterance (or a window of utterances) that
            led to the silent decision.
        speakers: List of speaker names present when the decision was made.
        db_path: Override DB path (for tests). Defaults to
            ``data/memory/memory.db``.

    Returns:
        The autoincrement ``id`` of the inserted row.
    """
    path = _resolve_db_path(db_path)
    await _ensure_schema(path)

    reason = str(decision.get("reason", "")) or "(no reason provided)"
    addressed_to = str(decision.get("addressed_to", "ambiguous"))
    urgency = str(decision.get("urgency", "none"))
    note = decision.get("silent_context_note")
    note_str = str(note) if note else None
    speakers_json = json.dumps(speakers) if speakers else None

    async with aiosqlite.connect(str(path)) as db:
        cursor = await db.execute(
            """
            INSERT INTO silent_turns (
                conversation_id,
                timestamp,
                reason,
                addressed_to,
                urgency,
                silent_context_note,
                transcript_snippet,
                speakers_present
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                time.time(),
                reason,
                addressed_to,
                urgency,
                note_str,
                transcript_snippet,
                speakers_json,
            ),
        )
        await db.commit()
        row_id = cursor.lastrowid or 0

    logger.debug(
        "silent_turn logged conv=%s addressed_to=%s urgency=%s id=%d",
        conversation_id,
        addressed_to,
        urgency,
        row_id,
    )
    return row_id


async def get_silent_turns(
    conversation_id: str,
    *,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return all silent-turn records for a conversation, oldest first.

    Args:
        conversation_id: Conversation ID to query.
        db_path: Override DB path (for tests).

    Returns:
        List of dicts with all columns. ``speakers_present`` is decoded
        from JSON back into a list.
    """
    path = _resolve_db_path(db_path)
    await _ensure_schema(path)

    async with aiosqlite.connect(str(path)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT
                id, conversation_id, timestamp, reason, addressed_to,
                urgency, silent_context_note, transcript_snippet,
                speakers_present
            FROM silent_turns
            WHERE conversation_id = ?
            ORDER BY timestamp ASC, id ASC
            """,
            (conversation_id,),
        ) as cursor:
            rows = await cursor.fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        speakers_raw = row["speakers_present"]
        try:
            speakers = json.loads(speakers_raw) if speakers_raw else None
        except (json.JSONDecodeError, TypeError):
            speakers = None
        results.append(
            {
                "id": row["id"],
                "conversation_id": row["conversation_id"],
                "timestamp": row["timestamp"],
                "reason": row["reason"],
                "addressed_to": row["addressed_to"],
                "urgency": row["urgency"],
                "silent_context_note": row["silent_context_note"],
                "transcript_snippet": row["transcript_snippet"],
                "speakers_present": speakers,
            }
        )
    return results
