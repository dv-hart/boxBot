"""Memory persistence layer using SQLite via aiosqlite.

Handles CRUD for memories, conversations, and system memory versions.
Manages FTS5 indexes for keyword search. Database at data/memory/memory.db.
System memory at data/memory/system.md.

Usage:
    from boxbot.memory.store import MemoryStore

    store = MemoryStore()
    await store.initialize()
    memory_id = await store.create_memory(...)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite
import numpy as np

from boxbot.core.paths import MEMORY_DIR
from boxbot.memory.embeddings import EMBEDDING_DIM, embed

logger = logging.getLogger(__name__)

# Anchored to project root via paths module
DB_DIR = MEMORY_DIR
DB_PATH = DB_DIR / "memory.db"
SYSTEM_MEMORY_PATH = DB_DIR / "system.md"

# Constraints
SYSTEM_MEMORY_MAX_BYTES = 4096
SYSTEM_MEMORY_MAX_VERSIONS = 20

# Pending-extraction transcript retention. After this many days the
# transcript column is nulled out (the row stays as provenance for the
# memories produced from it). Configurable via the memory config; this
# is the default applied when stamping ``transcript_purge_at`` at
# insert time.
TRANSCRIPT_RETENTION_DAYS = 14

# Valid values
# Legitimate memory types for newly-created rows. Legacy rows in the
# DB may still carry ``operational`` from before lifecycle step 7 —
# reads tolerate that, but new memories must use one of these three.
MEMORY_TYPES = {"person", "household", "methodology"}
MEMORY_STATUSES = {"active", "archived", "invalidated"}
PENDING_STATUSES = {"queued", "submitted", "applied", "failed"}

# Default system memory template
DEFAULT_SYSTEM_MEMORY = """## Household
- (no entries yet)

## Standing Instructions
- (no entries yet)

## Operational Notes
- (no entries yet)
"""

# Section names allowed in system memory
SYSTEM_MEMORY_SECTIONS = {"Household", "Standing Instructions", "Operational Notes"}

# Patterns that suggest secrets — block from system memory
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password|credential)\s*[:=]"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"(?i)bearer\s+[a-zA-Z0-9._\-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),                     # AWS access keys
    re.compile(r"gh[ps]_[a-zA-Z0-9]{36,}"),               # GitHub tokens
    re.compile(r"-----BEGIN [A-Z ]*?KEY-----"),            # PEM private keys
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Memory:
    """A fact memory record."""

    id: str
    type: str
    content: str
    summary: str
    person: str | None
    people: list[str]
    tags: list[str]
    source_conversation: str | None
    created_at: str
    last_relevant_at: str
    status: str
    invalidated_by: str | None
    superseded_by: str | None
    embedding: np.ndarray | None
    # Dream-phase fields (added 2026-04 in PR1). Optional for backwards
    # compatibility with code that constructs Memory without them.
    consolidated_by: str | None = None
    supporting_for: str | None = None
    dream_created_by: str | None = None


@dataclass
class Conversation:
    """A conversation log entry."""

    id: str
    channel: str
    participants: list[str]
    started_at: str
    summary: str
    topics: list[str]
    accessed_memories: list[str]
    embedding: np.ndarray | None


@dataclass
class SystemMemoryVersion:
    """A version of the system memory file."""

    version: int
    content: str
    updated_at: str
    updated_by: str
    change_summary: str


@dataclass
class PendingExtraction:
    """A queued or in-flight post-conversation extraction job."""

    conversation_id: str
    transcript: str | None  # nulled after transcript_purge_at
    accessed_memory_ids: list[str]
    channel: str
    participants: list[str]
    started_at: str
    status: str
    batch_id: str | None
    submitted_at: str | None
    completed_at: str | None
    error: str | None
    attempts: int
    transcript_purge_at: str
    # Rendered [Active Memories] block from the conversation start —
    # full memory summaries, not just IDs. Persisted so the extraction
    # model can apply the "ONLY invalidate memories listed here" rule
    # against real content instead of opaque IDs. Empty string for
    # legacy rows; the batch poller falls back to ID-only rendering.
    injected_memories_block: str = ""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id                TEXT PRIMARY KEY,
    type              TEXT NOT NULL,
    content           TEXT NOT NULL,
    summary           TEXT NOT NULL,
    person            TEXT,
    people            TEXT NOT NULL,
    tags              TEXT NOT NULL,
    source_conversation TEXT,
    created_at        TEXT NOT NULL,
    last_relevant_at  TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'active',
    invalidated_by    TEXT,
    superseded_by     TEXT,
    embedding         BLOB,

    FOREIGN KEY (source_conversation) REFERENCES conversations(id) ON DELETE SET NULL,
    FOREIGN KEY (superseded_by) REFERENCES memories(id)
);

CREATE TABLE IF NOT EXISTS conversations (
    id                TEXT PRIMARY KEY,
    channel           TEXT NOT NULL,
    participants      TEXT NOT NULL,
    started_at        TEXT NOT NULL,
    summary           TEXT NOT NULL DEFAULT '',
    topics            TEXT NOT NULL DEFAULT '[]',
    accessed_memories TEXT NOT NULL DEFAULT '[]',
    embedding         BLOB
);

CREATE TABLE IF NOT EXISTS system_memory_versions (
    version           INTEGER PRIMARY KEY AUTOINCREMENT,
    content           TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    updated_by        TEXT NOT NULL,
    change_summary    TEXT NOT NULL
);

-- Durable queue for post-conversation extraction. One row per
-- conversation; survives boxbot restarts so the batch poller can
-- resume polling Anthropic's batch API after a crash.
--
-- The transcript is retained for ``TRANSCRIPT_RETENTION_DAYS`` (14 by
-- default) past creation so the agent can recover full conversation
-- text via search_memory(mode="transcript"). After that, the row is
-- kept (provenance for the memories produced) but the transcript
-- column is nulled out by maintenance.
CREATE TABLE IF NOT EXISTS pending_extractions (
    conversation_id         TEXT PRIMARY KEY,
    transcript              TEXT,                  -- nulled after transcript_purge_at
    accessed_memory_ids     TEXT NOT NULL,         -- JSON list
    channel                 TEXT NOT NULL,
    participants            TEXT NOT NULL,         -- JSON list
    started_at              TEXT NOT NULL,
    status                  TEXT NOT NULL,         -- queued|submitted|applied|failed
    batch_id                TEXT,
    submitted_at            TEXT,
    completed_at            TEXT,
    error                   TEXT,
    attempts                INTEGER NOT NULL DEFAULT 0,
    transcript_purge_at     TEXT NOT NULL,
    injected_memories_block TEXT NOT NULL DEFAULT ''  -- rendered [Active Memories] block
);

-- Durable log of dream-phase consolidation cycles. One row per nightly
-- run. Holds the batch_id (used to undo the cycle) plus a JSON list of
-- candidate memory IDs and a counts-per-request-type breakdown for
-- audit/observability. Status mirrors pending_extractions:
-- submitted -> applied/failed.
CREATE TABLE IF NOT EXISTS pending_dreams (
    batch_id        TEXT PRIMARY KEY,
    submitted_at    TEXT NOT NULL,
    candidate_ids   TEXT NOT NULL,         -- JSON
    request_types   TEXT NOT NULL,         -- JSON: {dedup: N, ...}
    status          TEXT NOT NULL,         -- submitted | applied | failed
    completed_at    TEXT,
    summary         TEXT
);

-- Append-only log of every paid external API call. Used for cost
-- attribution and analysis. cost_usd is captured at write time so
-- historical totals don't drift if rates change. Provider-neutral:
-- Anthropic-shaped fields (tokens, is_batch) and ElevenLabs-shaped
-- fields (character_count, audio_seconds) coexist; unused columns
-- default to 0.
--
-- ``cache_write_tokens`` is the legacy column from the original
-- schema; new writes dual-populate it (= 5m + 1h) so older queries
-- keep working. New code should read the split columns instead.
CREATE TABLE IF NOT EXISTS cost_log (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                TEXT NOT NULL,
    purpose                  TEXT NOT NULL,         -- conversation|extraction|rerank|summary|dream|web_search|tts|stt|...
    provider                 TEXT NOT NULL DEFAULT 'anthropic',
    model                    TEXT NOT NULL,
    input_tokens             INTEGER NOT NULL DEFAULT 0,
    output_tokens            INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens        INTEGER NOT NULL DEFAULT 0,
    cache_write_5m_tokens    INTEGER NOT NULL DEFAULT 0,
    cache_write_1h_tokens    INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens       INTEGER NOT NULL DEFAULT 0,    -- legacy mirror of 5m+1h
    is_batch                 INTEGER NOT NULL DEFAULT 0,
    character_count          INTEGER NOT NULL DEFAULT 0,    -- elevenlabs TTS billed chars
    audio_seconds            REAL    NOT NULL DEFAULT 0,    -- elevenlabs STT measured input
    iterations               INTEGER NOT NULL DEFAULT 0,    -- agentic-loop turn count
    correlation_id           TEXT,                          -- groups rows belonging to one user turn
    cost_usd                 REAL NOT NULL,
    metadata                 TEXT                           -- JSON: conversation_id, batch_id, etc.
);

-- Append-only log of every tool call the agent makes, one row per
-- invocation, across every channel (voice included). Ground truth for
-- the prefetch layer: which searches/loads happened, in what turn, and
-- whether a repeat could have been avoided. Joins to cost_log and the
-- conversation stores on conversation_id. prefetch_attribution is null
-- until an active-mode matcher or the offline harness fills it in.
CREATE TABLE IF NOT EXISTS tool_invocations (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp              TEXT NOT NULL,
    conversation_id        TEXT,                    -- join key (= cost_log.correlation_id)
    channel                TEXT,                    -- voice|whatsapp|signal|trigger
    turn_number            INTEGER,                 -- agentic-loop iteration index
    tool_name              TEXT NOT NULL,
    tool_input_redacted    TEXT,                    -- secrets stripped, truncated ~200 chars
    result_status          TEXT,                    -- ok|error|unknown_tool|dispatched
    latency_ms             INTEGER,
    prefetch_attribution   TEXT,                    -- hit|miss|satisfiable (filled later)
    metadata               TEXT                     -- JSON: backend, sdk actions, etc.
);

-- One row per prefetch run (shadow or active). Records what the
-- prefetcher PREDICTED the agent would need. The offline harness joins
-- this to tool_invocations on conversation_id to compute hit-rate
-- (did the agent fetch what we predicted?) and precision (did the
-- agent use what we prefetched?). For scheduled triggers the run is
-- keyed by trigger_id at T-minus-N; the minted conversation_id is
-- back-filled via prefetch_cache at fire time.
CREATE TABLE IF NOT EXISTS prefetch_events (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                   TEXT NOT NULL,
    key                         TEXT NOT NULL,       -- conversation_id or trigger_id
    key_kind                    TEXT,                -- conversation | trigger
    channel                     TEXT,
    mode                        TEXT NOT NULL,       -- shadow | active
    predicted_memory_ids        TEXT,                -- JSON list
    predicted_skills            TEXT,                -- JSON list of skill names inlined
    predicted_workspace_paths   TEXT,                -- JSON list
    predicted_integration_calls TEXT,               -- JSON: [{source, action, pulled_at}]
    bundle_token_estimate       INTEGER,
    prefetch_latency_ms         INTEGER,
    prefetch_cost_usd           REAL,
    note                        TEXT,                -- 'what you'll likely need' (truncated)
    pulled_at                   TEXT
);

-- Precomputed bundles for scheduled triggers, produced T-minus-N before
-- fire_at and consumed by _on_trigger_fired. Survives restart (the
-- lookahead may straddle a deploy). TTL via expires_at; conversation_id
-- is stamped at fire time so the offline harness can bridge
-- trigger_id -> conversation_id.
CREATE TABLE IF NOT EXISTS prefetch_cache (
    trigger_id       TEXT PRIMARY KEY,
    bundle_json      TEXT NOT NULL,
    pulled_at        TEXT NOT NULL,
    expires_at       TEXT NOT NULL,
    conversation_id  TEXT
);

-- Small key/value scratch for cross-run dream-phase state (e.g. the
-- watermark of the last processed memory window). Avoids a misaligned
-- "since midnight" window that left a daily blind spot.
CREATE TABLE IF NOT EXISTS dream_state (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_person ON memories(person);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_last_relevant ON memories(last_relevant_at);
CREATE INDEX IF NOT EXISTS idx_conversations_started ON conversations(started_at);
CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_extractions(status);
CREATE INDEX IF NOT EXISTS idx_pending_purge ON pending_extractions(transcript_purge_at);
CREATE INDEX IF NOT EXISTS idx_pending_dreams_status ON pending_dreams(status);
CREATE INDEX IF NOT EXISTS idx_cost_log_timestamp ON cost_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_cost_log_purpose ON cost_log(purpose);
CREATE INDEX IF NOT EXISTS idx_tool_inv_conv ON tool_invocations(conversation_id);
CREATE INDEX IF NOT EXISTS idx_tool_inv_ts ON tool_invocations(timestamp);
CREATE INDEX IF NOT EXISTS idx_prefetch_events_key ON prefetch_events(key);
CREATE INDEX IF NOT EXISTS idx_prefetch_cache_conv ON prefetch_cache(conversation_id);
"""

FTS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, summary, tags, person, people,
    content='memories', content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
    summary, topics, participants,
    content='conversations', content_rowid='rowid'
);
"""

# Triggers to keep FTS in sync with content tables
FTS_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, summary, tags, person, people)
    VALUES (new.rowid, new.content, new.summary, new.tags, new.person, new.people);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary, tags, person, people)
    VALUES ('delete', old.rowid, old.content, old.summary, old.tags, old.person, old.people);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary, tags, person, people)
    VALUES ('delete', old.rowid, old.content, old.summary, old.tags, old.person, old.people);
    INSERT INTO memories_fts(rowid, content, summary, tags, person, people)
    VALUES (new.rowid, new.content, new.summary, new.tags, new.person, new.people);
END;

CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations BEGIN
    INSERT INTO conversations_fts(rowid, summary, topics, participants)
    VALUES (new.rowid, new.summary, new.topics, new.participants);
END;

CREATE TRIGGER IF NOT EXISTS conversations_ad AFTER DELETE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, summary, topics, participants)
    VALUES ('delete', old.rowid, old.summary, old.topics, old.participants);
END;

CREATE TRIGGER IF NOT EXISTS conversations_au AFTER UPDATE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, summary, topics, participants)
    VALUES ('delete', old.rowid, old.summary, old.topics, old.participants);
    INSERT INTO conversations_fts(rowid, summary, topics, participants)
    VALUES (new.rowid, new.summary, new.topics, new.participants);
END;
"""


def _embedding_to_blob(arr: np.ndarray) -> bytes:
    """Convert a numpy array to bytes for SQLite BLOB storage."""
    return arr.astype(np.float32).tobytes()


def _blob_to_embedding(blob: bytes | None) -> np.ndarray | None:
    """Convert SQLite BLOB bytes back to a numpy array."""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.utcnow().isoformat()


def _generate_id() -> str:
    """Generate a UUID string for memory/conversation IDs."""
    return str(uuid.uuid4())


def _contains_secret(text: str) -> bool:
    """Check if text contains patterns that look like secrets."""
    return any(p.search(text) for p in SECRET_PATTERNS)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class MemoryStore:
    """Async SQLite-backed persistence for memories, conversations, and
    system memory.

    Usage:
        store = MemoryStore()
        await store.initialize()
        # ... use store ...
        await store.close()
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or DB_PATH
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create database directory, open connection, and apply schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # isolation_level=None puts sqlite3 in autocommit mode: each
        # statement is its own implicit BEGIN/COMMIT at the SQLite layer
        # so a raised statement (FK violation, NOT NULL, etc.) cannot
        # leave a Python-level implicit transaction open and pin the
        # connection. Multi-statement atomicity, where actually needed,
        # is wrapped in explicit BEGIN/COMMIT below.
        self._db = await aiosqlite.connect(
            str(self._db_path), isolation_level=None,
        )
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        # NORMAL is the documented pairing for WAL — durable across
        # crashes, faster commits than the FULL default that's only
        # meaningful in rollback-journal mode.
        await self._db.execute("PRAGMA synchronous=NORMAL")
        # Wait inside SQLite's C layer instead of bouncing SQLITE_BUSY
        # to Python on every contended write. 5s is generous given that
        # all of our txns commit in milliseconds on a healthy DB.
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("PRAGMA foreign_keys=ON")

        # Create tables and indexes
        await self._db.executescript(SCHEMA_SQL)
        await self._db.executescript(FTS_SCHEMA_SQL)
        await self._db.executescript(FTS_TRIGGERS_SQL)
        await self._db.commit()

        # Idempotent column migrations for fields added after the original
        # schema shipped. Each ALTER is wrapped in try/except so a fresh DB
        # (where the column already exists) and an upgraded DB (where the
        # column does not yet exist) both succeed.
        await self._migrate_columns()

        # Ensure system memory file exists
        if not SYSTEM_MEMORY_PATH.exists():
            SYSTEM_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            SYSTEM_MEMORY_PATH.write_text(DEFAULT_SYSTEM_MEMORY, encoding="utf-8")

        logger.info("Memory store initialized at %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _migrate_columns(self) -> None:
        """Apply additive ALTER TABLE migrations idempotently.

        SQLite raises ``OperationalError`` if a column already exists; we
        catch and ignore that case so re-running on a current DB is a
        no-op while upgraded DBs receive the new columns.
        """
        # Each entry: (table, column, type)
        migrations = [
            ("memories", "consolidated_by", "TEXT"),
            ("memories", "supporting_for", "TEXT"),
            ("memories", "dream_created_by", "TEXT"),
            # cost_log columns added with the unified cost-tracker rollout.
            # Existing rows backfill cleanly with the column defaults.
            ("cost_log", "provider", "TEXT NOT NULL DEFAULT 'anthropic'"),
            ("cost_log", "cache_write_5m_tokens", "INTEGER NOT NULL DEFAULT 0"),
            ("cost_log", "cache_write_1h_tokens", "INTEGER NOT NULL DEFAULT 0"),
            ("cost_log", "character_count", "INTEGER NOT NULL DEFAULT 0"),
            ("cost_log", "audio_seconds", "REAL NOT NULL DEFAULT 0"),
            ("cost_log", "iterations", "INTEGER NOT NULL DEFAULT 0"),
            ("cost_log", "correlation_id", "TEXT"),
            # Lifecycle plan step 4 — persist the rendered [Active
            # Memories] block so post-conversation extraction can
            # invalidate stale memories with full content visibility.
            (
                "pending_extractions",
                "injected_memories_block",
                "TEXT NOT NULL DEFAULT ''",
            ),
        ]
        for table, col, col_type in migrations:
            try:
                await self._db.execute(
                    f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"
                )
            except aiosqlite.OperationalError as e:
                # "duplicate column name" is the expected idempotent case.
                if "duplicate column name" not in str(e).lower():
                    raise
        await self._db.commit()

    @property
    def db(self) -> aiosqlite.Connection:
        """Return the active database connection."""
        if self._db is None:
            raise RuntimeError("MemoryStore not initialized. Call initialize() first.")
        return self._db

    # -------------------------------------------------------------------
    # Memory CRUD
    # -------------------------------------------------------------------

    async def create_memory(
        self,
        *,
        type: str,
        content: str,
        summary: str,
        person: str | None = None,
        people: list[str] | None = None,
        tags: list[str] | None = None,
        source_conversation: str | None = None,
    ) -> str:
        """Create a new fact memory with auto-generated embedding.

        Args:
            type: One of person, household, methodology.
            content: Full memory content.
            summary: One-line summary for injection.
            person: Primary person (null for household/methodology).
            people: All people involved/mentioned.
            tags: Topic tags.
            source_conversation: FK to conversations.id.

        Returns:
            The generated memory ID.
        """
        if type not in MEMORY_TYPES:
            raise ValueError(f"Invalid memory type: {type}. Must be one of {MEMORY_TYPES}")

        memory_id = _generate_id()
        now = _now_iso()
        people_list = people or ([person] if person else [])
        tags_list = tags or []

        # Generate embedding from content + summary
        embedding_text = f"{summary} {content}"
        embedding_vec = embed(embedding_text)

        await self.db.execute(
            """INSERT INTO memories
               (id, type, content, summary, person, people, tags,
                source_conversation, created_at, last_relevant_at,
                status, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)""",
            (
                memory_id,
                type,
                content,
                summary,
                person,
                json.dumps(people_list),
                json.dumps(tags_list),
                source_conversation,
                now,
                now,
                _embedding_to_blob(embedding_vec),
            ),
        )
        await self.db.commit()

        logger.debug("Created memory %s (type=%s)", memory_id, type)
        return memory_id

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a single memory by ID.

        Also updates last_relevant_at (access-based retention).

        Returns:
            The Memory record, or None if not found.
        """
        cursor = await self.db.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        # Update last_relevant_at on access
        now = _now_iso()
        await self.db.execute(
            "UPDATE memories SET last_relevant_at = ? WHERE id = ?",
            (now, memory_id),
        )
        await self.db.commit()

        return _row_to_memory(row)

    async def get_memory_no_touch(self, memory_id: str) -> Memory | None:
        """Retrieve a memory without updating last_relevant_at.

        Used by maintenance and search internals where access should not
        extend retention.
        """
        cursor = await self.db.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_memory(row)

    async def update_memory_relevance(self, memory_id: str) -> None:
        """Update last_relevant_at to now (memory passed a filter or was injected)."""
        now = _now_iso()
        await self.db.execute(
            "UPDATE memories SET last_relevant_at = ? WHERE id = ?",
            (now, memory_id),
        )
        await self.db.commit()

    async def update_memory_content(
        self,
        memory_id: str,
        *,
        content: str,
        summary: str,
        tags: list[str] | None = None,
        people: list[str] | None = None,
    ) -> None:
        """Update a memory's content and re-generate its embedding.

        Used by extraction when merging new info into an existing memory.
        """
        embedding_text = f"{summary} {content}"
        embedding_vec = embed(embedding_text)
        now = _now_iso()

        sets = [
            "content = ?", "summary = ?", "embedding = ?", "last_relevant_at = ?",
        ]
        params: list = [content, summary, _embedding_to_blob(embedding_vec), now]

        if tags is not None:
            sets.append("tags = ?")
            params.append(json.dumps(tags))
        if people is not None:
            sets.append("people = ?")
            params.append(json.dumps(people))

        params.append(memory_id)
        await self.db.execute(
            f"UPDATE memories SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        await self.db.commit()

    async def archive_memory(self, memory_id: str) -> None:
        """Set a memory's status to archived."""
        await self.db.execute(
            "UPDATE memories SET status = 'archived' WHERE id = ?",
            (memory_id,),
        )
        await self.db.commit()

    async def unarchive_memory(self, memory_id: str) -> None:
        """Restore an archived memory to active status."""
        now = _now_iso()
        await self.db.execute(
            "UPDATE memories SET status = 'active', last_relevant_at = ? WHERE id = ?",
            (now, memory_id),
        )
        await self.db.commit()

    async def resolve_memory_id(self, id_or_prefix: str) -> list[str]:
        """Resolve a full id or an id *prefix* to matching memory ids.

        Memories are injected into context with an 8-char id prefix
        (``retrieval.py`` → ``#{id[:8]}``), so the agent only ever sees the
        prefix. Callers (delete/invalidate) need to turn that back into a
        real id without silently no-op'ing.

        An exact id match short-circuits to a single id regardless of
        status (idempotent re-invalidation is fine). Otherwise we
        prefix-match against **active** memories only, so a prefix never
        resolves to an already-invalidated duplicate.

        Returns:
            ``[]`` for no match, ``[id]`` for a unique match, or several
            ids when the prefix is ambiguous.
        """
        cursor = await self.db.execute(
            "SELECT id FROM memories WHERE id = ?", (id_or_prefix,)
        )
        row = await cursor.fetchone()
        if row is not None:
            return [row[0]]

        cursor = await self.db.execute(
            "SELECT id FROM memories WHERE id LIKE ? AND status = 'active'",
            (id_or_prefix + "%",),
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def invalidate_memory(
        self,
        memory_id: str,
        *,
        invalidated_by: str,
        superseded_by: str | None = None,
    ) -> int:
        """Mark a memory as invalidated (soft delete).

        Args:
            memory_id: The memory to invalidate. Must be a full id —
                callers pass prefixes through ``resolve_memory_id`` first.
            invalidated_by: Conversation ID that caused the invalidation.
            superseded_by: ID of the replacement memory, if any.

        Returns:
            Number of rows affected (0 if no memory had that id).
        """
        cursor = await self.db.execute(
            """UPDATE memories
               SET status = 'invalidated', invalidated_by = ?, superseded_by = ?
               WHERE id = ?""",
            (invalidated_by, superseded_by, memory_id),
        )
        await self.db.commit()
        return cursor.rowcount

    async def delete_memory(self, memory_id: str) -> None:
        """Permanently delete a memory (used by storage cap eviction)."""
        await self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        await self.db.commit()

    async def repoint_person_name(self, old_name: str, new_name: str) -> int:
        """Re-point person-keyed memory fields after an identity rename/merge.

        Retrieval filters and boosts on the ``person``/``people`` name
        columns (see ``memory.search``), so when a person record is renamed
        or merged away, memories tagged with the old name become unreachable
        for the surviving person — the memory still exists, but a search for
        the new name never matches it. This rewrites the structured name
        fields to close that gap. The free-text ``content``/``summary`` prose
        is deliberately left alone (the nightly dream cycle reconciles stale
        wording over time); invalidated rows are skipped. Returns the number
        of field updates applied.
        """
        if not old_name or not new_name or old_name == new_name:
            return 0
        c1 = await self.db.execute(
            "UPDATE memories SET person = ? "
            "WHERE person = ? AND status != 'invalidated'",
            (new_name, old_name),
        )
        # ``people`` is a JSON array of quoted names; match the quoted token
        # (e.g. '"Eric"') so 'Eric' does not clobber 'Erica'.
        c2 = await self.db.execute(
            "UPDATE memories SET people = replace(people, ?, ?) "
            "WHERE people LIKE ? AND status != 'invalidated'",
            (f'"{old_name}"', f'"{new_name}"', f'%"{old_name}"%'),
        )
        await self.db.commit()
        updated = c1.rowcount + c2.rowcount
        if updated:
            logger.info(
                "Repointed %d memory field(s): person %r -> %r",
                updated, old_name, new_name,
            )
        return updated

    async def get_dream_state(self, key: str) -> str | None:
        """Read a dream-phase scratch value (e.g. the run watermark)."""
        cursor = await self.db.execute(
            "SELECT value FROM dream_state WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        return row[0] if row is not None else None

    async def set_dream_state(self, key: str, value: str) -> None:
        """Write a dream-phase scratch value (upsert)."""
        await self.db.execute(
            "INSERT INTO dream_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        await self.db.commit()

    async def set_dream_audit_fields(
        self,
        memory_id: str,
        *,
        consolidated_by: str | None = None,
        supporting_for: str | None = None,
        dream_created_by: str | None = None,
    ) -> None:
        """Set one or more dream-phase audit columns on a memory.

        Soft-delete only — never DELETE; never bypass the status field.
        Pass an explicit None to leave a field untouched (we only update
        non-None args). Use ``unset_dream_audit_fields`` to clear them.
        """
        sets: list[str] = []
        params: list = []
        if consolidated_by is not None:
            sets.append("consolidated_by = ?")
            params.append(consolidated_by)
        if supporting_for is not None:
            sets.append("supporting_for = ?")
            params.append(supporting_for)
        if dream_created_by is not None:
            sets.append("dream_created_by = ?")
            params.append(dream_created_by)
        if not sets:
            return
        params.append(memory_id)
        await self.db.execute(
            f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", params
        )
        await self.db.commit()

    async def unset_dream_audit_fields(
        self,
        memory_id: str,
        *,
        clear_consolidated_by: bool = False,
        clear_supporting_for: bool = False,
        clear_dream_created_by: bool = False,
    ) -> None:
        """Clear one or more dream-phase audit columns. Used by undo."""
        sets: list[str] = []
        if clear_consolidated_by:
            sets.append("consolidated_by = NULL")
        if clear_supporting_for:
            sets.append("supporting_for = NULL")
        if clear_dream_created_by:
            sets.append("dream_created_by = NULL")
        if not sets:
            return
        await self.db.execute(
            f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", (memory_id,)
        )
        await self.db.commit()

    async def list_memories_by_dream(
        self,
        batch_id: str,
        *,
        field: str = "consolidated_by",
    ) -> list[Memory]:
        """List memories whose dream audit field matches ``batch_id``.

        Used by the undo script. ``field`` must be one of
        ``consolidated_by``, ``supporting_for``, ``dream_created_by``.
        """
        if field not in {"consolidated_by", "supporting_for", "dream_created_by"}:
            raise ValueError(f"Invalid dream audit field: {field}")
        cursor = await self.db.execute(
            f"SELECT * FROM memories WHERE {field} = ?", (batch_id,)
        )
        rows = await cursor.fetchall()
        return [_row_to_memory(r) for r in rows]

    async def reactivate_invalidated_by_dream(
        self,
        batch_id: str,
    ) -> int:
        """Restore status='active' for memories invalidated by a dream batch.

        Specifically: memories whose ``consolidated_by = batch_id`` AND
        whose status is 'invalidated' get their status flipped back to
        'active', and ``invalidated_by`` / ``superseded_by`` are cleared.
        Returns count of rows updated.
        """
        cursor = await self.db.execute(
            """UPDATE memories
               SET status = 'active',
                   invalidated_by = NULL,
                   superseded_by = NULL
               WHERE consolidated_by = ? AND status = 'invalidated'""",
            (batch_id,),
        )
        await self.db.commit()
        return cursor.rowcount

    async def list_memories(
        self,
        *,
        status: str | None = None,
        type: str | None = None,
        person: str | None = None,
        limit: int = 100,
        order_by: str = "last_relevant_at DESC",
    ) -> list[Memory]:
        """List memories with optional filters.

        Args:
            status: Filter by status (active, archived, invalidated).
            type: Filter by memory type.
            person: Filter by primary person.
            limit: Maximum number of results.
            order_by: SQL ORDER BY clause.

        Returns:
            List of Memory records.
        """
        conditions = []
        params: list = []

        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if type is not None:
            conditions.append("type = ?")
            params.append(type)
        if person is not None:
            conditions.append("person = ?")
            params.append(person)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cursor = await self.db.execute(
            f"SELECT * FROM memories {where} ORDER BY {order_by} LIMIT ?",
            params + [limit],
        )
        rows = await cursor.fetchall()
        return [_row_to_memory(r) for r in rows]

    async def list_memories_created_since(
        self,
        iso_timestamp: str,
        *,
        status: str | None = "active",
        limit: int = 1000,
    ) -> list[Memory]:
        """Return memories whose created_at >= ``iso_timestamp``.

        Used by the dream phase to gather "today's" candidate memories.
        Defaults to active memories only, since archived/invalidated
        records are not consolidation candidates.
        """
        conditions = ["created_at >= ?"]
        params: list = [iso_timestamp]
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        where = "WHERE " + " AND ".join(conditions)
        cursor = await self.db.execute(
            f"SELECT * FROM memories {where} "
            f"ORDER BY created_at ASC LIMIT ?",
            params + [limit],
        )
        rows = await cursor.fetchall()
        return [_row_to_memory(r) for r in rows]

    async def list_memories_relevant_since(
        self,
        iso_timestamp: str,
        *,
        status: str = "active",
        limit: int = 1000,
    ) -> list[Memory]:
        """Return memories whose last_relevant_at >= ``iso_timestamp``.

        Used by the dream phase for the "used today" pool — any memory
        that was injected into a conversation today gets its
        last_relevant_at bumped to the inject time.
        """
        cursor = await self.db.execute(
            "SELECT * FROM memories "
            "WHERE last_relevant_at >= ? AND status = ? "
            "ORDER BY last_relevant_at DESC LIMIT ?",
            (iso_timestamp, status, limit),
        )
        rows = await cursor.fetchall()
        return [_row_to_memory(r) for r in rows]

    async def count_memories(self, *, status: str | None = None) -> int:
        """Count memories, optionally filtered by status."""
        if status:
            cursor = await self.db.execute(
                "SELECT COUNT(*) FROM memories WHERE status = ?", (status,)
            )
        else:
            cursor = await self.db.execute("SELECT COUNT(*) FROM memories")
        row = await cursor.fetchone()
        return row[0]

    # -------------------------------------------------------------------
    # Conversation CRUD
    # -------------------------------------------------------------------

    async def create_conversation(
        self,
        *,
        channel: str,
        participants: list[str],
        summary: str,
        topics: list[str] | None = None,
        accessed_memories: list[str] | None = None,
        started_at: str | None = None,
        conversation_id: str | None = None,
    ) -> str:
        """Create a conversation log entry.

        Args:
            conversation_id: If provided, use this id instead of generating
                one. Lets callers (extraction, stub creation) reuse the
                live ``Conversation.conversation_id`` so memories created
                during the conversation can FK-resolve cleanly.

        Returns:
            The conversation ID (generated or passed-through).
        """
        conv_id = conversation_id or _generate_id()
        now = started_at or _now_iso()
        topics_list = topics or []
        accessed_list = accessed_memories or []

        embedding_vec = embed(summary) if summary else None

        await self.db.execute(
            """INSERT INTO conversations
               (id, channel, participants, started_at, summary, topics,
                accessed_memories, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                conv_id,
                channel,
                json.dumps(participants),
                now,
                summary,
                json.dumps(topics_list),
                json.dumps(accessed_list),
                _embedding_to_blob(embedding_vec) if embedding_vec is not None else None,
            ),
        )
        await self.db.commit()

        logger.debug("Created conversation %s (channel=%s)", conv_id, channel)
        return conv_id

    async def create_conversation_stub(
        self,
        *,
        conversation_id: str,
        channel: str,
        participants: list[str],
        started_at: str | None = None,
    ) -> None:
        """Insert a placeholder conversations row when a Conversation begins.

        Idempotent (``INSERT OR IGNORE``) so warm-load / replay paths do
        not error if the row was already created. Summary, topics, and
        accessed_memories start empty; extraction fills them in via
        :meth:`update_conversation` when the conversation closes.

        This exists so memories saved during a live conversation can
        stamp ``source_conversation = conv.conversation_id`` and have
        the FK resolve, without waiting for the post-conversation
        extraction batch to land.
        """
        now = started_at or _now_iso()
        await self.db.execute(
            """INSERT OR IGNORE INTO conversations
               (id, channel, participants, started_at, summary, topics,
                accessed_memories, embedding)
               VALUES (?, ?, ?, ?, '', '[]', '[]', NULL)""",
            (
                conversation_id,
                channel,
                json.dumps(participants),
                now,
            ),
        )
        await self.db.commit()

    async def update_conversation(
        self,
        conversation_id: str,
        *,
        summary: str,
        topics: list[str] | None = None,
        accessed_memories: list[str] | None = None,
    ) -> None:
        """Fill in summary/topics/embedding for a previously stubbed row.

        Used by extraction to upgrade an active stub into a finalized
        log entry without changing the conversation_id (which downstream
        memories already FK against).
        """
        topics_list = topics or []
        accessed_list = accessed_memories or []
        embedding_vec = embed(summary) if summary else None
        await self.db.execute(
            """UPDATE conversations
               SET summary = ?,
                   topics = ?,
                   accessed_memories = ?,
                   embedding = ?
               WHERE id = ?""",
            (
                summary,
                json.dumps(topics_list),
                json.dumps(accessed_list),
                _embedding_to_blob(embedding_vec) if embedding_vec is not None else None,
                conversation_id,
            ),
        )
        await self.db.commit()

    async def get_conversation(self, conv_id: str) -> Conversation | None:
        """Retrieve a single conversation by ID."""
        cursor = await self.db.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_conversation(row)

    async def list_conversations(
        self,
        *,
        limit: int = 50,
        person: str | None = None,
    ) -> list[Conversation]:
        """List recent conversations, newest first.

        Args:
            limit: Maximum number of results.
            person: Filter to conversations involving this person.
        """
        if person:
            # Search in the JSON participants array
            cursor = await self.db.execute(
                """SELECT * FROM conversations
                   WHERE participants LIKE ?
                   ORDER BY started_at DESC LIMIT ?""",
                (f'%"{person}"%', limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM conversations ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [_row_to_conversation(r) for r in rows]

    async def delete_conversation(self, conv_id: str) -> None:
        """Permanently delete a conversation (retention pruning + cap eviction).

        Extracted facts persist independently of their source conversation,
        so we detach any memory that still points at this conversation before
        deleting it. Without this, ``memories.source_conversation`` (an
        enforced FK with no ``ON DELETE`` action) blocks the delete with
        ``FOREIGN KEY constraint failed`` whenever a memory outlives the
        conversation it came from — which is the common case, since person/
        household memories are retained far longer (180d) than conversations
        (60d). Detaching matches the FK's intended ``ON DELETE SET NULL``
        semantics; provenance lookups already tolerate a null source. Both
        statements share one transaction so the detach and delete are atomic.
        """
        await self.db.execute(
            "UPDATE memories SET source_conversation = NULL "
            "WHERE source_conversation = ?",
            (conv_id,),
        )
        await self.db.execute(
            "DELETE FROM conversations WHERE id = ?", (conv_id,)
        )
        await self.db.commit()

    # -------------------------------------------------------------------
    # System memory
    # -------------------------------------------------------------------

    async def read_system_memory(self) -> str:
        """Read the current system memory file content."""
        if SYSTEM_MEMORY_PATH.exists():
            return SYSTEM_MEMORY_PATH.read_text(encoding="utf-8")
        return DEFAULT_SYSTEM_MEMORY

    async def update_system_memory(
        self,
        *,
        section: str,
        action: str,
        content: str,
        updated_by: str,
    ) -> None:
        """Update a section of system memory.

        Args:
            section: Section name (Household, Standing Instructions, Operational Notes).
            action: One of set, add_entry, remove_entry.
            content: The content to set/add/remove.
            updated_by: Who made the change (extraction_agent or conversation ID).

        Raises:
            ValueError: If section is invalid, content contains secrets,
                        or size cap would be exceeded.
        """
        if section not in SYSTEM_MEMORY_SECTIONS:
            raise ValueError(
                f"Invalid section: {section}. "
                f"Must be one of {SYSTEM_MEMORY_SECTIONS}"
            )

        if action not in {"set", "add_entry", "remove_entry"}:
            raise ValueError(f"Invalid action: {action}. Must be set, add_entry, or remove_entry")

        if _contains_secret(content):
            raise ValueError(
                "Content appears to contain secrets (API keys, tokens, etc.). "
                "Use boxbot_sdk.secrets instead."
            )

        current = await self.read_system_memory()
        updated = _apply_section_update(current, section, action, content)

        if len(updated.encode("utf-8")) > SYSTEM_MEMORY_MAX_BYTES:
            raise ValueError(
                f"System memory would exceed {SYSTEM_MEMORY_MAX_BYTES} byte cap "
                f"({len(updated.encode('utf-8'))} bytes after update)"
            )

        # Write the updated file
        SYSTEM_MEMORY_PATH.write_text(updated, encoding="utf-8")

        # Save version history
        change_desc = f"{action} in '{section}': {content[:80]}"
        await self.db.execute(
            """INSERT INTO system_memory_versions
               (content, updated_at, updated_by, change_summary)
               VALUES (?, ?, ?, ?)""",
            (updated, _now_iso(), updated_by, change_desc),
        )
        await self.db.commit()

        # Trim old versions beyond the limit
        await self._trim_system_memory_versions()

        logger.info("System memory updated: %s", change_desc)

    async def get_system_memory_versions(
        self, limit: int = 20
    ) -> list[SystemMemoryVersion]:
        """Get recent system memory versions, newest first."""
        cursor = await self.db.execute(
            """SELECT version, content, updated_at, updated_by, change_summary
               FROM system_memory_versions
               ORDER BY version DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            SystemMemoryVersion(
                version=r["version"],
                content=r["content"],
                updated_at=r["updated_at"],
                updated_by=r["updated_by"],
                change_summary=r["change_summary"],
            )
            for r in rows
        ]

    async def rollback_system_memory(self, version: int) -> None:
        """Rollback system memory to a specific version number.

        Args:
            version: The version number to restore.

        Raises:
            ValueError: If the version does not exist.
        """
        cursor = await self.db.execute(
            "SELECT content FROM system_memory_versions WHERE version = ?",
            (version,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise ValueError(f"System memory version {version} not found")

        SYSTEM_MEMORY_PATH.write_text(row["content"], encoding="utf-8")

        # Record the rollback as a new version
        await self.db.execute(
            """INSERT INTO system_memory_versions
               (content, updated_at, updated_by, change_summary)
               VALUES (?, ?, ?, ?)""",
            (row["content"], _now_iso(), "admin_rollback", f"Rollback to version {version}"),
        )
        await self.db.commit()
        await self._trim_system_memory_versions()

        logger.info("System memory rolled back to version %d", version)

    async def _trim_system_memory_versions(self) -> None:
        """Keep only the most recent SYSTEM_MEMORY_MAX_VERSIONS versions."""
        await self.db.execute(
            """DELETE FROM system_memory_versions
               WHERE version NOT IN (
                   SELECT version FROM system_memory_versions
                   ORDER BY version DESC LIMIT ?
               )""",
            (SYSTEM_MEMORY_MAX_VERSIONS,),
        )
        await self.db.commit()

    # -------------------------------------------------------------------
    # Pending extractions (durable batch queue)
    # -------------------------------------------------------------------

    async def create_pending_extraction(
        self,
        *,
        conversation_id: str,
        transcript: str,
        accessed_memory_ids: list[str],
        channel: str,
        participants: list[str],
        started_at: str,
        injected_memories_block: str = "",
        retention_days: int = TRANSCRIPT_RETENTION_DAYS,
    ) -> None:
        """Insert a new pending-extraction row in queued status."""
        purge_at = (
            datetime.utcnow() + timedelta(days=retention_days)
        ).isoformat()
        await self.db.execute(
            """INSERT OR REPLACE INTO pending_extractions (
                   conversation_id, transcript, accessed_memory_ids,
                   channel, participants, started_at, status,
                   transcript_purge_at, attempts,
                   injected_memories_block
               ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, 0, ?)""",
            (
                conversation_id,
                transcript,
                json.dumps(accessed_memory_ids),
                channel,
                json.dumps(participants),
                started_at,
                purge_at,
                injected_memories_block,
            ),
        )
        await self.db.commit()

    async def mark_pending_submitted(
        self,
        conversation_id: str,
        batch_id: str,
    ) -> None:
        """Move a pending row from queued/failed to submitted."""
        await self.db.execute(
            """UPDATE pending_extractions
               SET status = 'submitted',
                   batch_id = ?,
                   submitted_at = ?,
                   error = NULL,
                   attempts = attempts + 1
               WHERE conversation_id = ?""",
            (batch_id, _now_iso(), conversation_id),
        )
        await self.db.commit()

    async def mark_pending_applied(self, conversation_id: str) -> None:
        await self.db.execute(
            """UPDATE pending_extractions
               SET status = 'applied', completed_at = ?, error = NULL
               WHERE conversation_id = ?""",
            (_now_iso(), conversation_id),
        )
        await self.db.commit()

    async def mark_pending_failed(
        self,
        conversation_id: str,
        error: str,
    ) -> None:
        await self.db.execute(
            """UPDATE pending_extractions
               SET status = 'failed', completed_at = ?, error = ?
               WHERE conversation_id = ?""",
            (_now_iso(), error, conversation_id),
        )
        await self.db.commit()

    async def get_pending_extraction(
        self,
        conversation_id: str,
    ) -> PendingExtraction | None:
        cursor = await self.db.execute(
            "SELECT * FROM pending_extractions WHERE conversation_id = ?",
            (conversation_id,),
        )
        row = await cursor.fetchone()
        return _row_to_pending(row) if row else None

    async def list_pending_extractions(
        self,
        *,
        status: str | None = None,
        limit: int = 200,
    ) -> list[PendingExtraction]:
        if status:
            cursor = await self.db.execute(
                """SELECT * FROM pending_extractions
                   WHERE status = ?
                   ORDER BY started_at ASC LIMIT ?""",
                (status, limit),
            )
        else:
            cursor = await self.db.execute(
                """SELECT * FROM pending_extractions
                   ORDER BY started_at DESC LIMIT ?""",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [_row_to_pending(r) for r in rows]

    async def purge_expired_transcripts(self) -> int:
        """Null out transcript text for rows past their retention window.

        Rows themselves are preserved as provenance — only the raw
        transcript column is cleared. Returns the number of rows affected.
        """
        cursor = await self.db.execute(
            """UPDATE pending_extractions
               SET transcript = NULL
               WHERE transcript IS NOT NULL
                 AND transcript_purge_at < ?""",
            (_now_iso(),),
        )
        await self.db.commit()
        return cursor.rowcount

    async def get_transcript(self, conversation_id: str) -> str | None:
        """Return the raw transcript for a conversation, or None if
        already purged or never recorded."""
        cursor = await self.db.execute(
            "SELECT transcript FROM pending_extractions WHERE conversation_id = ?",
            (conversation_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return row["transcript"]

    async def search_transcripts(
        self,
        query: str,
        *,
        limit: int = 5,
        within_days: int = TRANSCRIPT_RETENTION_DAYS,
    ) -> list[tuple[str, str, str]]:
        """Substring search across recent retained transcripts.

        Cheap LIKE-based scan; the volume is bounded by the retention
        window (default 14 days × ~30 conversations/day worst case).
        Returns list of (conversation_id, started_at, snippet) where
        snippet is ~300 chars of context around the first match.
        """
        cutoff = (
            datetime.utcnow() - timedelta(days=within_days)
        ).isoformat()
        cursor = await self.db.execute(
            """SELECT conversation_id, started_at, transcript
               FROM pending_extractions
               WHERE transcript IS NOT NULL
                 AND started_at >= ?
                 AND transcript LIKE ?
               ORDER BY started_at DESC
               LIMIT ?""",
            (cutoff, f"%{query}%", limit),
        )
        rows = await cursor.fetchall()
        results: list[tuple[str, str, str]] = []
        for r in rows:
            transcript = r["transcript"] or ""
            idx = transcript.lower().find(query.lower())
            if idx < 0:
                snippet = transcript[:300]
            else:
                start = max(0, idx - 120)
                end = min(len(transcript), idx + 180)
                snippet = ("..." if start > 0 else "") + transcript[start:end] + (
                    "..." if end < len(transcript) else ""
                )
            results.append((r["conversation_id"], r["started_at"], snippet))
        return results

    # -------------------------------------------------------------------
    # Pending dreams (durable nightly consolidation queue)
    # -------------------------------------------------------------------

    async def create_pending_dream(
        self,
        *,
        batch_id: str,
        candidate_ids: list[str],
        request_types: dict[str, int],
        summary: str | None = None,
    ) -> None:
        """Insert a new pending-dream row in submitted status.

        Mirrors ``create_pending_extraction``: written as soon as the
        batch is submitted to Anthropic, so a crash mid-cycle leaves a
        durable record the DreamPoller can resume from on next boot.
        """
        await self.db.execute(
            """INSERT OR REPLACE INTO pending_dreams (
                   batch_id, submitted_at, candidate_ids,
                   request_types, status, summary
               ) VALUES (?, ?, ?, ?, 'submitted', ?)""",
            (
                batch_id,
                _now_iso(),
                json.dumps(candidate_ids),
                json.dumps(request_types),
                summary,
            ),
        )
        await self.db.commit()

    async def mark_dream_applied(
        self,
        batch_id: str,
        *,
        summary: str | None = None,
    ) -> None:
        """Mark a dream batch as applied; optionally update its summary."""
        if summary is None:
            await self.db.execute(
                """UPDATE pending_dreams
                   SET status = 'applied', completed_at = ?
                   WHERE batch_id = ?""",
                (_now_iso(), batch_id),
            )
        else:
            await self.db.execute(
                """UPDATE pending_dreams
                   SET status = 'applied', completed_at = ?, summary = ?
                   WHERE batch_id = ?""",
                (_now_iso(), summary, batch_id),
            )
        await self.db.commit()

    async def mark_dream_failed(
        self,
        batch_id: str,
        error: str,
    ) -> None:
        """Mark a dream batch as failed with an error annotation."""
        await self.db.execute(
            """UPDATE pending_dreams
               SET status = 'failed', completed_at = ?, summary = ?
               WHERE batch_id = ?""",
            (_now_iso(), error, batch_id),
        )
        await self.db.commit()

    async def get_pending_dream(self, batch_id: str) -> dict | None:
        """Return one pending-dream row as a dict, or None."""
        cursor = await self.db.execute(
            "SELECT * FROM pending_dreams WHERE batch_id = ?", (batch_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_pending_dream(row)

    async def list_pending_dreams(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """List pending-dream rows, newest first."""
        if status:
            cursor = await self.db.execute(
                """SELECT * FROM pending_dreams
                   WHERE status = ?
                   ORDER BY submitted_at DESC LIMIT ?""",
                (status, limit),
            )
        else:
            cursor = await self.db.execute(
                """SELECT * FROM pending_dreams
                   ORDER BY submitted_at DESC LIMIT ?""",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [_row_to_pending_dream(r) for r in rows]

    # -------------------------------------------------------------------
    # Cost log
    # -------------------------------------------------------------------

    async def record_cost(
        self,
        *,
        purpose: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        is_batch: bool = False,
        cost_usd: float,
        metadata: dict | None = None,
    ) -> None:
        """Append a row to the cost log.

        Legacy entry point. New code should build a
        :class:`boxbot.cost.CostEvent` and call
        :func:`boxbot.cost.record` directly. This shim builds an
        Anthropic-shaped event from the keyword arguments and treats
        ``cache_write_tokens`` as 1h-TTL (matching the historical
        assumption baked into the prior ``compute_cost`` helper) so
        existing callers continue to record correct numbers without
        change.
        """
        # Local import to avoid a circular load at module import time
        # (cost.record reaches back into anything importing the store).
        from boxbot.cost import CostEvent, record

        event = CostEvent(
            purpose=purpose,
            provider="anthropic",
            model=model,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_1h_tokens=cache_write_tokens,
            is_batch=is_batch,
            metadata=metadata,
        )
        await record(self, event)

    async def cost_summary(self, *, days: int = 7) -> dict[str, float]:
        """Return total cost (USD) per purpose over the last N days."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        cursor = await self.db.execute(
            """SELECT purpose, SUM(cost_usd) AS total
               FROM cost_log
               WHERE timestamp >= ?
               GROUP BY purpose""",
            (cutoff,),
        )
        rows = await cursor.fetchall()
        return {r["purpose"]: float(r["total"] or 0.0) for r in rows}

    # -------------------------------------------------------------------
    # Database size
    # -------------------------------------------------------------------

    async def get_db_size_bytes(self) -> int:
        """Return the current database file size in bytes."""
        if self._db_path.exists():
            return self._db_path.stat().st_size
        return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_memory(row: aiosqlite.Row) -> Memory:
    """Convert a database row to a Memory dataclass."""
    # The dream-phase columns are added via _migrate_columns. Reading
    # them safely covers both fresh and upgraded databases plus any
    # caller that selects a subset of columns.
    keys = row.keys() if hasattr(row, "keys") else []
    consolidated_by = row["consolidated_by"] if "consolidated_by" in keys else None
    supporting_for = row["supporting_for"] if "supporting_for" in keys else None
    dream_created_by = (
        row["dream_created_by"] if "dream_created_by" in keys else None
    )
    return Memory(
        id=row["id"],
        type=row["type"],
        content=row["content"],
        summary=row["summary"],
        person=row["person"],
        people=json.loads(row["people"]),
        tags=json.loads(row["tags"]),
        source_conversation=row["source_conversation"],
        created_at=row["created_at"],
        last_relevant_at=row["last_relevant_at"],
        status=row["status"],
        invalidated_by=row["invalidated_by"],
        superseded_by=row["superseded_by"],
        embedding=_blob_to_embedding(row["embedding"]),
        consolidated_by=consolidated_by,
        supporting_for=supporting_for,
        dream_created_by=dream_created_by,
    )


def _row_to_pending_dream(row: aiosqlite.Row) -> dict:
    """Convert a pending_dreams database row to a plain dict.

    Returned as a dict (not a dataclass) because the dream phase keeps
    its own richer in-memory representation; the DB row is just durable
    state for resume-on-boot and undo.
    """
    return {
        "batch_id": row["batch_id"],
        "submitted_at": row["submitted_at"],
        "candidate_ids": json.loads(row["candidate_ids"] or "[]"),
        "request_types": json.loads(row["request_types"] or "{}"),
        "status": row["status"],
        "completed_at": row["completed_at"],
        "summary": row["summary"],
    }


def _row_to_pending(row: aiosqlite.Row) -> PendingExtraction:
    """Convert a database row to a PendingExtraction dataclass."""
    try:
        block = row["injected_memories_block"] or ""
    except (IndexError, KeyError):
        # Legacy DBs that haven't run the migration yet.
        block = ""
    return PendingExtraction(
        conversation_id=row["conversation_id"],
        transcript=row["transcript"],
        accessed_memory_ids=json.loads(row["accessed_memory_ids"]),
        channel=row["channel"],
        participants=json.loads(row["participants"]),
        started_at=row["started_at"],
        status=row["status"],
        batch_id=row["batch_id"],
        submitted_at=row["submitted_at"],
        completed_at=row["completed_at"],
        error=row["error"],
        attempts=row["attempts"],
        transcript_purge_at=row["transcript_purge_at"],
        injected_memories_block=block,
    )


def _row_to_conversation(row: aiosqlite.Row) -> Conversation:
    """Convert a database row to a Conversation dataclass."""
    return Conversation(
        id=row["id"],
        channel=row["channel"],
        participants=json.loads(row["participants"]),
        started_at=row["started_at"],
        summary=row["summary"],
        topics=json.loads(row["topics"]),
        accessed_memories=json.loads(row["accessed_memories"]),
        embedding=_blob_to_embedding(row["embedding"]),
    )


def _apply_section_update(
    current: str, section: str, action: str, content: str
) -> str:
    """Apply a section-level update to system memory markdown.

    Args:
        current: Current file content.
        section: Target section heading.
        action: set, add_entry, or remove_entry.
        content: Content for the action.

    Returns:
        Updated file content.
    """
    # Parse sections: find ## headings and their content
    lines = current.split("\n")
    sections: dict[str, list[str]] = {}
    section_order: list[str] = []
    current_section: str | None = None

    for line in lines:
        if line.startswith("## "):
            current_section = line[3:].strip()
            if current_section not in sections:
                sections[current_section] = []
                section_order.append(current_section)
        elif current_section is not None:
            sections[current_section].append(line)

    # Ensure target section exists
    if section not in sections:
        sections[section] = []
        section_order.append(section)

    # Apply the action
    if action == "set":
        sections[section] = [content]
    elif action == "add_entry":
        # Remove placeholder entries
        sections[section] = [
            line for line in sections[section]
            if not (line.strip().startswith("- (no entries") and line.strip().endswith(")"))
        ]
        sections[section].append(f"- {content}")
    elif action == "remove_entry":
        sections[section] = [
            line for line in sections[section]
            if content.lower() not in line.lower()
        ]
        # If section is now empty (only whitespace), add placeholder
        if not any(line.strip() for line in sections[section]):
            sections[section] = [f"- (no entries yet)"]

    # Rebuild the file
    result_lines: list[str] = []
    for sec_name in section_order:
        result_lines.append(f"## {sec_name}")
        for line in sections[sec_name]:
            result_lines.append(line)
        # Ensure blank line between sections
        if result_lines and result_lines[-1].strip():
            result_lines.append("")

    return "\n".join(result_lines)
