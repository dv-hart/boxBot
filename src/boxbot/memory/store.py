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
from datetime import datetime
from pathlib import Path

import aiosqlite
import numpy as np

from boxbot.memory.embeddings import EMBEDDING_DIM, embed

logger = logging.getLogger(__name__)

# Paths relative to project root
DB_DIR = Path("data/memory")
DB_PATH = DB_DIR / "memory.db"
SYSTEM_MEMORY_PATH = DB_DIR / "system.md"

# Constraints
SYSTEM_MEMORY_MAX_BYTES = 4096
SYSTEM_MEMORY_MAX_VERSIONS = 20

# Valid values
MEMORY_TYPES = {"person", "household", "methodology", "operational"}
MEMORY_STATUSES = {"active", "archived", "invalidated"}

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
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),                # Anthropic / OpenAI keys
    re.compile(r"(?i)bearer\s+[a-zA-Z0-9._\-]{20,}"),  # Bearer tokens
    re.compile(r"AKIA[0-9A-Z]{16}"),                    # AWS access key IDs
    re.compile(r"gh[ps]_[a-zA-Z0-9]{36,}"),             # GitHub personal / server tokens
    re.compile(r"-----BEGIN .* KEY-----"),               # PEM private keys
    re.compile(r"xox[bpras]-[a-zA-Z0-9\-]{10,}"),      # Slack tokens
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

    FOREIGN KEY (source_conversation) REFERENCES conversations(id),
    FOREIGN KEY (superseded_by) REFERENCES memories(id)
);

CREATE TABLE IF NOT EXISTS conversations (
    id                TEXT PRIMARY KEY,
    channel           TEXT NOT NULL,
    participants      TEXT NOT NULL,
    started_at        TEXT NOT NULL,
    summary           TEXT NOT NULL,
    topics            TEXT NOT NULL,
    accessed_memories TEXT NOT NULL,
    embedding         BLOB
);

CREATE TABLE IF NOT EXISTS system_memory_versions (
    version           INTEGER PRIMARY KEY AUTOINCREMENT,
    content           TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    updated_by        TEXT NOT NULL,
    change_summary    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_person ON memories(person);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_last_relevant ON memories(last_relevant_at);
CREATE INDEX IF NOT EXISTS idx_conversations_started ON conversations(started_at);
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

        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        # Create tables and indexes
        await self._db.executescript(SCHEMA_SQL)
        await self._db.executescript(FTS_SCHEMA_SQL)
        await self._db.executescript(FTS_TRIGGERS_SQL)
        await self._db.commit()

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
            type: One of person, household, methodology, operational.
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

    async def invalidate_memory(
        self,
        memory_id: str,
        *,
        invalidated_by: str,
        superseded_by: str | None = None,
    ) -> None:
        """Mark a memory as invalidated (soft delete).

        Args:
            memory_id: The memory to invalidate.
            invalidated_by: Conversation ID that caused the invalidation.
            superseded_by: ID of the replacement memory, if any.
        """
        await self.db.execute(
            """UPDATE memories
               SET status = 'invalidated', invalidated_by = ?, superseded_by = ?
               WHERE id = ?""",
            (invalidated_by, superseded_by, memory_id),
        )
        await self.db.commit()

    async def delete_memory(self, memory_id: str) -> None:
        """Permanently delete a memory (used by storage cap eviction)."""
        await self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        await self.db.commit()

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
    ) -> str:
        """Create a conversation log entry.

        Returns:
            The generated conversation ID.
        """
        conv_id = _generate_id()
        now = started_at or _now_iso()
        topics_list = topics or []
        accessed_list = accessed_memories or []

        embedding_vec = embed(summary)

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
                _embedding_to_blob(embedding_vec),
            ),
        )
        await self.db.commit()

        logger.debug("Created conversation %s (channel=%s)", conv_id, channel)
        return conv_id

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
        """Permanently delete a conversation (used by storage cap eviction)."""
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
