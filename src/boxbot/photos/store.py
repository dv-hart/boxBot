"""Photo persistence via aiosqlite.

Manages the photos database at data/photos/photos.db, including:
- Photo metadata CRUD (create, get, update, list)
- Soft delete / restore / permanent delete
- Tag library management (merge, rename, delete)
- Slideshow membership
- Storage quota enforcement

Usage:
    from boxbot.photos.store import PhotoStore

    store = PhotoStore()
    await store.initialize()
    photo_id = await store.create_photo(...)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite
import numpy as np

from boxbot.core.config import get_config

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PhotoRecord:
    """A single photo record from the database."""

    id: str
    filename: str
    source: str
    sender: str | None
    description: str
    orientation: str
    width: int | None
    height: int | None
    file_size: int
    in_slideshow: bool
    created_at: str
    deleted_at: str | None
    updated_at: str
    tags: list[str] = field(default_factory=list)
    # People named in the photo. Each dict: label, person_id, bbox_*,
    # confidence ("confirmed" | "captioned" | "likely"), source, basis,
    # conversation_id, channel (where an agent-reported label came from).
    people: list[dict[str, Any]] = field(default_factory=list)
    # Tagger guesses too weak to count as a person (confidence "unsure").
    # Same dict shape as ``people``. Never matched by search.
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    # Set once the agent has reported that a person reviewed the people
    # on this photo (``set_people`` / ``confirm_people``). The tagger
    # (ingest and backfill) never touches a reviewed photo again.
    people_confirmed_at: str | None = None
    # Conversation and channel the agent's report came from.
    people_confirmed_conversation: str | None = None
    people_confirmed_channel: str | None = None


@dataclass
class TagRecord:
    """A tag from the tag library."""

    id: int
    name: str
    created_at: str
    count: int = 0


@dataclass
class StorageInfo:
    """Storage quota information."""

    used_bytes: int
    quota_bytes: int
    used_percent: float


# ---------------------------------------------------------------------------
# SQL Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS photos (
    id              TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    source          TEXT NOT NULL,
    sender          TEXT,
    description     TEXT NOT NULL DEFAULT '',
    orientation     TEXT NOT NULL DEFAULT 'landscape',
    width           INTEGER,
    height          INTEGER,
    file_size       INTEGER NOT NULL DEFAULT 0,
    in_slideshow    INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    deleted_at      TEXT,
    updated_at      TEXT NOT NULL,
    embedding       BLOB
);

CREATE TABLE IF NOT EXISTS tags (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT UNIQUE NOT NULL COLLATE NOCASE,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS photo_tags (
    photo_id        TEXT NOT NULL REFERENCES photos(id),
    tag_id          INTEGER NOT NULL REFERENCES tags(id),
    PRIMARY KEY (photo_id, tag_id)
);

CREATE TABLE IF NOT EXISTS photo_people (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id        TEXT NOT NULL REFERENCES photos(id),
    person_id       TEXT,
    label           TEXT NOT NULL,
    bbox_x          REAL,
    bbox_y          REAL,
    bbox_w          REAL,
    bbox_h          REAL
);

-- Household roster shown to the small-model tagger: one row per person,
-- with a one-line appearance/context note. ``person_id`` links to the
-- perception persons table when a record exists there (optional — a
-- person the box has never seen or heard can still be on the roster).
CREATE TABLE IF NOT EXISTS people_notes (
    name            TEXT PRIMARY KEY COLLATE NOCASE,
    note            TEXT NOT NULL DEFAULT '',
    person_id       TEXT,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_photos_deleted
    ON photos(deleted_at);
CREATE INDEX IF NOT EXISTS idx_photos_slideshow
    ON photos(in_slideshow) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_photos_created
    ON photos(created_at);
CREATE INDEX IF NOT EXISTS idx_photo_people_photo
    ON photo_people(photo_id);
CREATE INDEX IF NOT EXISTS idx_photo_people_person
    ON photo_people(person_id);
"""

# Columns added after first release — applied idempotently on open
# (same additive pattern as memory/store.py::_migrate_columns).
_COLUMN_MIGRATIONS: list[tuple[str, str, str]] = [
    # confidence:
    #   confirmed — a person confirmed it, as reported by the agent
    #   captioned — the caption passed at ingest names them (tagger)
    #   likely    — appearance/context vs the roster (tagger, detector)
    #   unsure    — weak guess; a suggestion, not a person
    ("photo_people", "confidence", "TEXT NOT NULL DEFAULT 'likely'"),
    # source: agent_reported | tagger | detector | legacy
    ("photo_people", "source", "TEXT NOT NULL DEFAULT 'detector'"),
    # basis: the tagger's one-line reason ("caption names her", …)
    ("photo_people", "basis", "TEXT NOT NULL DEFAULT ''"),
    # Where an agent_reported label came from.
    ("photo_people", "conversation_id", "TEXT"),
    ("photo_people", "channel", "TEXT"),
    ("photos", "people_confirmed_at", "TEXT"),
    ("photos", "people_confirmed_conversation", "TEXT"),
    ("photos", "people_confirmed_channel", "TEXT"),
]

# Confidence levels, strongest first. "unsure" rows are suggestions, not
# people: they are kept for review but never count as a person.
CONFIDENCE_LEVELS = ("confirmed", "captioned", "likely", "unsure")
PERSON_CONFIDENCES = ("confirmed", "captioned", "likely")

# Who wrote a people label. The agent makes the confirm/set calls, so a
# confirmation is "agent_reported" — never recorded as a human's own act.
SOURCE_AGENT_REPORTED = "agent_reported"
# Rows that predate the confidence/source columns: provenance unknown.
SOURCE_LEGACY = "legacy"

# Small key/value table for schema markers (see ``_migrate_fts``).
_META_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS photo_meta (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL
)"""
FTS_VERSION = "2"

# Standalone FTS5 table over description + tags + people labels, kept in
# sync from Python (``_reindex_photo``). Keyed by the photo id in an
# UNINDEXED column rather than by the photos rowid — ``photos`` has a
# TEXT primary key, so its implicit rowid is not stable across VACUUM.
_FTS_COLUMNS = ("photo_id", "description", "tags", "people")
_FTS_SCHEMA_SQL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS photos_fts USING fts5(
    photo_id UNINDEXED,
    description,
    tags,
    people
)"""


def _generate_photo_id() -> str:
    """Generate a prefixed UUID for a photo."""
    return f"photo_{uuid4().hex[:12]}"


def _now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# PhotoStore
# ---------------------------------------------------------------------------


class PhotoStore:
    """Async SQLite-backed photo storage and metadata management."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize the photo store.

        Args:
            db_path: Override path to the SQLite database. Defaults to
                     {config.photos.storage_path}/photos.db.
        """
        if db_path is not None:
            self._db_path = Path(db_path)
        else:
            config = get_config()
            self._db_path = Path(config.photos.storage_path) / "photos.db"

        self._db: aiosqlite.Connection | None = None
        # False when the FTS migration/rebuild failed on open: keyword
        # search is then off and index writes are skipped (the next
        # ``initialize`` rebuilds). Everything else keeps working.
        self._fts_ready = True

    @property
    def fts_ready(self) -> bool:
        return self._fts_ready

    @property
    def db_path(self) -> Path:
        return self._db_path

    async def initialize(self) -> None:
        """Open the database and create tables if needed."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        await self._db.executescript(_SCHEMA_SQL)
        await self._migrate_columns()
        try:
            await self._migrate_fts()
            self._fts_ready = True
        except Exception:
            # Leave the store usable: photos, tags and people still work,
            # search runs without the keyword signal, and the next open
            # retries (nothing of the failed attempt was committed).
            self._fts_ready = False
            await self._db.rollback()
            logger.exception(
                "Photo FTS migration/rebuild failed; keyword search is "
                "off until the next successful start"
            )
        await self._db.commit()
        logger.info("Photo store initialized at %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def _table_columns(self, table: str) -> list[str]:
        db = self._ensure_db()
        async with db.execute(f"PRAGMA table_info({table})") as cursor:
            return [row["name"] async for row in cursor]

    async def _migrate_columns(self) -> None:
        """Apply additive column migrations idempotently, in one transaction.

        Rows already in ``photo_people`` when the ``confidence`` column
        is first added predate provenance tracking. They keep the hedged
        ``likely`` and are marked ``source='legacy'``, which keeps them
        out of the review queue (``list_unconfirmed_people``).
        """
        db = self._ensure_db()
        await db.commit()
        existing = {
            table: set(await self._table_columns(table))
            for table in {t for t, _, _ in _COLUMN_MIGRATIONS}
        }
        missing = [
            m for m in _COLUMN_MIGRATIONS if m[1] not in existing[m[0]]
        ]
        await db.execute("BEGIN IMMEDIATE")
        try:
            for table, col, col_type in missing:
                await db.execute(
                    f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"
                )
            if ("photo_people", "confidence") in {
                (t, c) for t, c, _ in missing
            }:
                await db.execute(
                    "UPDATE photo_people SET source = ?", (SOURCE_LEGACY,)
                )
            # Vocabulary used by pre-release builds of this feature.
            await db.execute(
                "UPDATE photo_people SET source = ? WHERE source = 'human'",
                (SOURCE_AGENT_REPORTED,),
            )
            await db.execute(
                "UPDATE photo_people SET confidence = 'captioned' "
                "WHERE confidence = 'certain'"
            )
            await db.execute(_META_SCHEMA_SQL)
            await db.commit()
        except BaseException:
            await db.rollback()
            raise

    async def _migrate_fts(self) -> None:
        """Bring the FTS index to the current shape and make sure it is full.

        v1 was an external-content table over ``photos.description``
        maintained by triggers. v2 also indexes tags and people labels.

        The index is (re)built when any of these holds:
          - the table is missing or has the v1 shape
          - the ``fts_version`` marker is not the current version (the
            marker is written in the same transaction as the rebuild)
          - the index row count differs from the photo row count (an
            interrupted or partial build; every photo, soft-deleted
            included, has exactly one index row)

        Drop, create, rebuild and marker run in ONE transaction, so a
        crash at any point leaves the previous state and the next start
        does the whole thing again.
        """
        db = self._ensure_db()
        await db.commit()
        async with db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'photos_fts'"
        ) as cursor:
            exists = await cursor.fetchone() is not None
        cols = await self._table_columns("photos_fts") if exists else []
        shape_ok = tuple(cols) == _FTS_COLUMNS

        reason: str | None = None
        if not exists:
            reason = "no index"
        elif not shape_ok:
            reason = "old index shape"
        else:
            async with db.execute(
                "SELECT value FROM photo_meta WHERE key = 'fts_version'"
            ) as cursor:
                row = await cursor.fetchone()
            if row is None or row["value"] != FTS_VERSION:
                reason = "no completed-build marker"
            else:
                async with db.execute(
                    "SELECT (SELECT COUNT(*) FROM photos_fts) AS indexed, "
                    "(SELECT COUNT(*) FROM photos) AS photos"
                ) as cursor:
                    counts = await cursor.fetchone()
                if counts["indexed"] != counts["photos"]:
                    reason = (
                        f"index has {counts['indexed']} rows for "
                        f"{counts['photos']} photos"
                    )
        if reason is None:
            return

        await db.execute("BEGIN IMMEDIATE")
        try:
            if not shape_ok:
                for trigger in ("photos_ai", "photos_ad", "photos_au"):
                    await db.execute(f"DROP TRIGGER IF EXISTS {trigger}")
                await db.execute("DROP TABLE IF EXISTS photos_fts")
                await db.execute(_FTS_SCHEMA_SQL)
            count = await self._rebuild_fts_rows()
            await db.execute(
                "INSERT INTO photo_meta (key, value) VALUES "
                "('fts_version', ?) ON CONFLICT(key) DO UPDATE SET "
                "value = excluded.value",
                (FTS_VERSION,),
            )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        if exists or count:
            logger.info(
                "Rebuilt photos FTS index (%s): %d photos, covering "
                "description, tags and people", reason, count,
            )

    def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError(
                "PhotoStore not initialized. Call initialize() first."
            )
        return self._db

    # ------------------------------------------------------------------
    # Photo CRUD
    # ------------------------------------------------------------------

    async def create_photo(
        self,
        *,
        filename: str,
        source: str,
        sender: str | None = None,
        description: str = "",
        orientation: str = "landscape",
        width: int | None = None,
        height: int | None = None,
        file_size: int = 0,
        in_slideshow: bool = True,
        tags: list[str] | None = None,
        people: list[dict[str, Any]] | None = None,
        embedding: np.ndarray | None = None,
        photo_id: str | None = None,
    ) -> str:
        """Insert a new photo record.

        Args:
            filename: Relative path within the photos storage directory.
            source: Origin — "whatsapp", "camera", or "upload".
            sender: Person name if from messaging.
            description: Small-model generated description.
            orientation: "landscape", "portrait", or "square".
            width: Width in pixels after resize.
            height: Height in pixels after resize.
            file_size: Size of the image file in bytes.
            in_slideshow: Whether to include in slideshow rotation.
            tags: List of tag names to associate.
            people: List of person dicts with keys: label, person_id,
                    bbox_x, bbox_y, bbox_w, bbox_h.
            embedding: 384-dim float32 embedding of the description.
            photo_id: Optional explicit ID (for testing). Generated if None.

        Returns:
            The new photo ID.
        """
        db = self._ensure_db()
        pid = photo_id or _generate_photo_id()
        now = _now_iso()

        embedding_blob: bytes | None = None
        if embedding is not None:
            embedding_blob = embedding.astype(np.float32).tobytes()

        await db.execute(
            """INSERT INTO photos
               (id, filename, source, sender, description, orientation,
                width, height, file_size, in_slideshow,
                created_at, updated_at, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pid, filename, source, sender, description, orientation,
                width, height, file_size, 1 if in_slideshow else 0,
                now, now, embedding_blob,
            ),
        )

        # Associate tags
        if tags:
            for tag_name in tags:
                tag_id = await self._ensure_tag(tag_name)
                await db.execute(
                    "INSERT OR IGNORE INTO photo_tags (photo_id, tag_id) VALUES (?, ?)",
                    (pid, tag_id),
                )

        # Associate people
        for person in people or []:
            await self._insert_person(pid, person)

        await self._reindex_photo(pid)
        await db.commit()
        logger.debug("Created photo %s (%s)", pid, filename)
        return pid

    async def get_photo(self, photo_id: str) -> PhotoRecord | None:
        """Get a single photo by ID, including tags and people."""
        db = self._ensure_db()
        async with db.execute(
            "SELECT * FROM photos WHERE id = ?", (photo_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return await self._row_to_record(row)

    async def update_photo(
        self,
        photo_id: str,
        *,
        description: str | None = None,
        orientation: str | None = None,
        in_slideshow: bool | None = None,
        embedding: np.ndarray | None = None,
    ) -> bool:
        """Update mutable fields on a photo.

        Returns True if the photo was found and updated.
        """
        db = self._ensure_db()
        fields: list[str] = []
        values: list[Any] = []

        if description is not None:
            fields.append("description = ?")
            values.append(description)
        if orientation is not None:
            fields.append("orientation = ?")
            values.append(orientation)
        if in_slideshow is not None:
            fields.append("in_slideshow = ?")
            values.append(1 if in_slideshow else 0)
        if embedding is not None:
            fields.append("embedding = ?")
            values.append(embedding.astype(np.float32).tobytes())

        if not fields:
            return False

        fields.append("updated_at = ?")
        values.append(_now_iso())
        values.append(photo_id)

        result = await db.execute(
            f"UPDATE photos SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        if description is not None and result.rowcount > 0:
            await self._reindex_photo(photo_id)
        await db.commit()
        return result.rowcount > 0

    async def list_photos(
        self,
        *,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PhotoRecord]:
        """List photos ordered by creation date (newest first).

        Args:
            include_deleted: Whether to include soft-deleted photos.
            limit: Maximum number of results.
            offset: Number of results to skip.
        """
        db = self._ensure_db()
        if include_deleted:
            query = "SELECT * FROM photos ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params: tuple[Any, ...] = (limit, offset)
        else:
            query = (
                "SELECT * FROM photos WHERE deleted_at IS NULL "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?"
            )
            params = (limit, offset)

        records: list[PhotoRecord] = []
        async with db.execute(query, params) as cursor:
            async for row in cursor:
                records.append(await self._row_to_record(row))
        return records

    # ------------------------------------------------------------------
    # Soft delete / restore / permanent delete
    # ------------------------------------------------------------------

    async def soft_delete_photo(self, photo_id: str) -> bool:
        """Soft-delete a photo by setting deleted_at timestamp.

        Returns True if the photo was found and soft-deleted.
        """
        db = self._ensure_db()
        now = _now_iso()
        result = await db.execute(
            "UPDATE photos SET deleted_at = ?, in_slideshow = 0, updated_at = ? "
            "WHERE id = ? AND deleted_at IS NULL",
            (now, now, photo_id),
        )
        await db.commit()
        return result.rowcount > 0

    async def restore_photo(self, photo_id: str) -> bool:
        """Restore a soft-deleted photo.

        Returns True if the photo was found and restored.
        """
        db = self._ensure_db()
        now = _now_iso()
        result = await db.execute(
            "UPDATE photos SET deleted_at = NULL, updated_at = ? "
            "WHERE id = ? AND deleted_at IS NOT NULL",
            (now, photo_id),
        )
        await db.commit()
        return result.rowcount > 0

    async def permanent_delete(self, photo_id: str) -> bool:
        """Permanently delete a photo record and its file.

        Returns True if the photo was found and deleted.
        """
        db = self._ensure_db()

        # Get filename before deleting the record
        async with db.execute(
            "SELECT filename FROM photos WHERE id = ?", (photo_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return False
            filename = row["filename"]

        # Delete associations
        await db.execute("DELETE FROM photo_people WHERE photo_id = ?", (photo_id,))
        await db.execute("DELETE FROM photo_tags WHERE photo_id = ?", (photo_id,))
        await db.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
        if self._fts_ready:
            await db.execute(
                "DELETE FROM photos_fts WHERE photo_id = ?", (photo_id,)
            )
        await db.commit()

        # Delete the file from disk
        config = get_config()
        file_path = Path(config.photos.storage_path) / filename
        if file_path.exists():
            file_path.unlink()
            logger.debug("Deleted file %s", file_path)

            # Clean up empty parent directories
            parent = file_path.parent
            storage_root = Path(config.photos.storage_path)
            while parent != storage_root and parent.exists():
                try:
                    parent.rmdir()  # Only removes if empty
                    parent = parent.parent
                except OSError:
                    break

        logger.debug("Permanently deleted photo %s", photo_id)
        return True

    async def list_deleted(
        self, *, limit: int = 50, offset: int = 0
    ) -> list[PhotoRecord]:
        """List soft-deleted photos."""
        db = self._ensure_db()
        records: list[PhotoRecord] = []
        async with db.execute(
            "SELECT * FROM photos WHERE deleted_at IS NOT NULL "
            "ORDER BY deleted_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cursor:
            async for row in cursor:
                records.append(await self._row_to_record(row))
        return records

    # ------------------------------------------------------------------
    # Tag management
    # ------------------------------------------------------------------

    async def _ensure_tag(self, name: str) -> int:
        """Get or create a tag, returning its ID."""
        db = self._ensure_db()
        name = name.strip().lower()

        async with db.execute(
            "SELECT id FROM tags WHERE name = ?", (name,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return row["id"]

        await db.execute(
            "INSERT INTO tags (name, created_at) VALUES (?, ?)",
            (name, _now_iso()),
        )
        await db.commit()

        async with db.execute(
            "SELECT id FROM tags WHERE name = ?", (name,)
        ) as cursor:
            row = await cursor.fetchone()
            return row["id"]

    async def update_tags(
        self, photo_id: str, tags: list[str]
    ) -> bool:
        """Replace all tags on a photo with the given list.

        Returns True if the photo exists.
        """
        db = self._ensure_db()

        # Verify photo exists
        async with db.execute(
            "SELECT id FROM photos WHERE id = ?", (photo_id,)
        ) as cursor:
            if await cursor.fetchone() is None:
                return False

        # Remove existing tags
        await db.execute(
            "DELETE FROM photo_tags WHERE photo_id = ?", (photo_id,)
        )

        # Add new tags
        for tag_name in tags:
            tag_id = await self._ensure_tag(tag_name)
            await db.execute(
                "INSERT OR IGNORE INTO photo_tags (photo_id, tag_id) VALUES (?, ?)",
                (photo_id, tag_id),
            )

        await db.execute(
            "UPDATE photos SET updated_at = ? WHERE id = ?",
            (_now_iso(), photo_id),
        )
        await self._reindex_photo(photo_id)
        await db.commit()
        return True

    async def merge_tags(self, source_name: str, target_name: str) -> int:
        """Merge source tag into target tag.

        All photos tagged with source get tagged with target instead.
        The source tag is then deleted.

        Returns the number of photos re-tagged.
        """
        db = self._ensure_db()
        source_name = source_name.strip().lower()
        target_name = target_name.strip().lower()

        # Get source tag ID
        async with db.execute(
            "SELECT id FROM tags WHERE name = ?", (source_name,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return 0
            source_id = row["id"]

        # Ensure target tag exists
        target_id = await self._ensure_tag(target_name)

        # Get photos with the source tag
        async with db.execute(
            "SELECT photo_id FROM photo_tags WHERE tag_id = ?", (source_id,)
        ) as cursor:
            photo_ids = [r["photo_id"] async for r in cursor]

        count = 0
        for pid in photo_ids:
            # Add target tag (ignore if already present)
            await db.execute(
                "INSERT OR IGNORE INTO photo_tags (photo_id, tag_id) VALUES (?, ?)",
                (pid, target_id),
            )
            count += 1

        # Remove all source tag associations
        await db.execute("DELETE FROM photo_tags WHERE tag_id = ?", (source_id,))
        # Delete source tag
        await db.execute("DELETE FROM tags WHERE id = ?", (source_id,))
        for pid in photo_ids:
            await self._reindex_photo(pid)
        await db.commit()

        logger.info("Merged tag '%s' into '%s' (%d photos)", source_name, target_name, count)
        return count

    async def rename_tag(self, old_name: str, new_name: str) -> bool:
        """Rename a tag across all photos.

        Returns True if the tag was found and renamed.
        """
        db = self._ensure_db()
        old_name = old_name.strip().lower()
        new_name = new_name.strip().lower()

        result = await db.execute(
            "UPDATE tags SET name = ? WHERE name = ?",
            (new_name, old_name),
        )
        if result.rowcount > 0:
            async with db.execute(
                "SELECT pt.photo_id FROM photo_tags pt "
                "JOIN tags t ON t.id = pt.tag_id WHERE t.name = ?",
                (new_name,),
            ) as cursor:
                affected = [r["photo_id"] async for r in cursor]
            for pid in affected:
                await self._reindex_photo(pid)
        await db.commit()
        return result.rowcount > 0

    async def delete_tag(self, name: str) -> int:
        """Delete a tag and all its photo associations.

        Returns the number of photo associations removed.
        """
        db = self._ensure_db()
        name = name.strip().lower()

        async with db.execute(
            "SELECT id FROM tags WHERE name = ?", (name,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return 0
            tag_id = row["id"]

        # Count associations
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM photo_tags WHERE tag_id = ?", (tag_id,)
        ) as cursor:
            row = await cursor.fetchone()
            count = row["cnt"]

        async with db.execute(
            "SELECT photo_id FROM photo_tags WHERE tag_id = ?", (tag_id,)
        ) as cursor:
            affected = [r["photo_id"] async for r in cursor]

        await db.execute("DELETE FROM photo_tags WHERE tag_id = ?", (tag_id,))
        await db.execute("DELETE FROM tags WHERE id = ?", (tag_id,))
        for pid in affected:
            await self._reindex_photo(pid)
        await db.commit()

        logger.info("Deleted tag '%s' (%d associations)", name, count)
        return count

    async def list_tags(self) -> list[TagRecord]:
        """List all tags with their photo counts."""
        db = self._ensure_db()
        records: list[TagRecord] = []
        async with db.execute(
            """SELECT t.id, t.name, t.created_at,
                      COUNT(pt.photo_id) as count
               FROM tags t
               LEFT JOIN photo_tags pt ON t.id = pt.tag_id
               LEFT JOIN photos p ON pt.photo_id = p.id AND p.deleted_at IS NULL
               GROUP BY t.id
               ORDER BY count DESC, t.name ASC"""
        ) as cursor:
            async for row in cursor:
                records.append(
                    TagRecord(
                        id=row["id"],
                        name=row["name"],
                        created_at=row["created_at"],
                        count=row["count"],
                    )
                )
        return records

    # ------------------------------------------------------------------
    # People
    # ------------------------------------------------------------------

    async def _insert_person(
        self, photo_id: str, person: dict[str, Any]
    ) -> None:
        """Insert one photo_people row. Does not commit or reindex."""
        db = self._ensure_db()
        confidence = person.get("confidence") or "likely"
        if confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"unknown confidence: {confidence!r}")
        await db.execute(
            """INSERT INTO photo_people
               (photo_id, person_id, label, bbox_x, bbox_y, bbox_w, bbox_h,
                confidence, source, basis, conversation_id, channel)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                photo_id,
                person.get("person_id"),
                person["label"],
                person.get("bbox_x"),
                person.get("bbox_y"),
                person.get("bbox_w"),
                person.get("bbox_h"),
                confidence,
                person.get("source") or "detector",
                str(person.get("basis") or ""),
                person.get("conversation_id"),
                person.get("channel"),
            ),
        )

    async def update_people(
        self,
        photo_id: str,
        people: list[dict[str, Any]],
        *,
        agent_reported: bool = False,
        conversation_id: str | None = None,
        channel: str | None = None,
    ) -> bool:
        """Replace all people annotations on a photo.

        Each dict needs ``label``; optional ``person_id``, ``bbox_*``,
        ``confidence``, ``source``, ``basis``, ``conversation_id``,
        ``channel``.

        ``agent_reported`` = the agent says a person reviewed this photo
        (``set_people`` / ``confirm_people``). Every row is written as
        ``confidence="confirmed"`` / ``source="agent_reported"`` with the
        *conversation_id* and *channel* the call came from, and the
        photo is marked reviewed, which locks it against the tagger (see
        :meth:`apply_inferred_people`). The store cannot verify that a
        person was involved — the provenance says who reported it and
        where. An empty list with ``agent_reported`` records "nobody
        from the household is in this photo".

        Returns True if the photo exists.
        """
        db = self._ensure_db()

        async with db.execute(
            "SELECT id FROM photos WHERE id = ?", (photo_id,)
        ) as cursor:
            if await cursor.fetchone() is None:
                return False

        await db.execute(
            "DELETE FROM photo_people WHERE photo_id = ?", (photo_id,)
        )

        for person in people:
            row = dict(person)
            if agent_reported:
                row["confidence"] = "confirmed"
                row["source"] = SOURCE_AGENT_REPORTED
                row["conversation_id"] = conversation_id
                row["channel"] = channel
            await self._insert_person(photo_id, row)

        now = _now_iso()
        if agent_reported:
            await db.execute(
                "UPDATE photos SET updated_at = ?, people_confirmed_at = ?, "
                "people_confirmed_conversation = ?, "
                "people_confirmed_channel = ? WHERE id = ?",
                (now, now, conversation_id, channel, photo_id),
            )
        else:
            await db.execute(
                "UPDATE photos SET updated_at = ? WHERE id = ?",
                (now, photo_id),
            )
        await self._reindex_photo(photo_id)
        await db.commit()
        return True

    async def apply_inferred_people(
        self, photo_id: str, inferred: list[dict[str, Any]]
    ) -> bool:
        """Write the tagger's people inference for a photo.

        Replaces earlier tagger rows only, so re-running is idempotent.
        Rows from other sources (detector, agent_reported, legacy) are
        kept, and an inferred name already present from another source
        is dropped. A photo marked reviewed is left untouched.

        Returns True if rows were written, False if the photo is missing
        or marked reviewed.
        """
        db = self._ensure_db()
        async with db.execute(
            "SELECT people_confirmed_at FROM photos WHERE id = ?", (photo_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None or row["people_confirmed_at"]:
            return False

        await db.execute(
            "DELETE FROM photo_people WHERE photo_id = ? AND source = 'tagger'",
            (photo_id,),
        )
        async with db.execute(
            "SELECT label FROM photo_people WHERE photo_id = ?", (photo_id,)
        ) as cursor:
            taken = {r["label"].casefold() async for r in cursor}

        for person in inferred:
            key = str(person["label"]).casefold()
            if key in taken:
                continue
            taken.add(key)
            await self._insert_person(photo_id, {**person, "source": "tagger"})

        await db.execute(
            "UPDATE photos SET updated_at = ? WHERE id = ?",
            (_now_iso(), photo_id),
        )
        await self._reindex_photo(photo_id)
        await db.commit()
        return True

    async def list_unconfirmed_people(
        self, *, limit: int = 20
    ) -> list[PhotoRecord]:
        """Photos whose people still need a person's look, newest first.

        A photo qualifies when it is not marked reviewed and carries at
        least one ``likely`` or ``unsure`` label. Rows that predate the
        confidence columns (``source='legacy'``) do not count: they were
        never proposed for review and would flood the queue on upgrade.
        """
        db = self._ensure_db()
        records: list[PhotoRecord] = []
        async with db.execute(
            """SELECT p.* FROM photos p
               WHERE p.deleted_at IS NULL
                 AND p.people_confirmed_at IS NULL
                 AND EXISTS (
                     SELECT 1 FROM photo_people pp
                     WHERE pp.photo_id = p.id
                       AND pp.confidence IN ('likely', 'unsure')
                       AND pp.source != 'legacy')
               ORDER BY p.created_at DESC LIMIT ?""",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
        for row in rows:
            records.append(await self._row_to_record(row))
        return records

    async def known_person_names(self) -> dict[str, str]:
        """Names the library knows: roster + existing people labels.

        ``{casefolded name: name as stored}``. ``unsure`` suggestions
        are not people and are left out.
        """
        db = self._ensure_db()
        names: dict[str, str] = {}
        async with db.execute(
            "SELECT DISTINCT label FROM photo_people "
            "WHERE confidence != 'unsure'"
        ) as cursor:
            async for row in cursor:
                names[row["label"].casefold()] = row["label"]
        async with db.execute("SELECT name FROM people_notes") as cursor:
            async for row in cursor:
                names[row["name"].casefold()] = row["name"]
        return names

    # ------------------------------------------------------------------
    # Household roster (people_notes)
    # ------------------------------------------------------------------

    async def list_people_notes(self) -> list[dict[str, Any]]:
        """The household roster: ``[{name, note, person_id}]`` by name."""
        db = self._ensure_db()
        out: list[dict[str, Any]] = []
        async with db.execute(
            "SELECT name, note, person_id FROM people_notes "
            "ORDER BY name COLLATE NOCASE"
        ) as cursor:
            async for row in cursor:
                out.append({
                    "name": row["name"],
                    "note": row["note"],
                    "person_id": row["person_id"],
                })
        return out

    async def set_person_note(
        self, name: str, note: str, *, person_id: str | None = None
    ) -> None:
        """Add or replace a roster entry (name match is case-insensitive)."""
        db = self._ensure_db()
        await db.execute(
            """INSERT INTO people_notes (name, note, person_id, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                   name = excluded.name,
                   note = excluded.note,
                   person_id = excluded.person_id,
                   updated_at = excluded.updated_at""",
            (name, note, person_id, _now_iso()),
        )
        await db.commit()

    async def delete_person_note(self, name: str) -> bool:
        """Remove a roster entry. Returns True if one was removed."""
        db = self._ensure_db()
        result = await db.execute(
            "DELETE FROM people_notes WHERE name = ?", (name,)
        )
        await db.commit()
        return result.rowcount > 0

    async def repoint_person(
        self,
        old_person_id: str,
        new_person_id: str,
        new_label: str | None = None,
    ) -> int:
        """Re-point photo person tags after an identity rename or merge.

        For a rename, call with ``old_person_id == new_person_id`` and
        the new name as *new_label*. For a merge, the loser's id maps
        to the winner's id (and the winner's name as label).

        Returns the number of photo_people rows updated.
        """
        db = self._ensure_db()
        if new_label is not None:
            cursor = await db.execute(
                "UPDATE photo_people SET person_id = ?, label = ? "
                "WHERE person_id = ?",
                (new_person_id, new_label, old_person_id),
            )
        else:
            cursor = await db.execute(
                "UPDATE photo_people SET person_id = ? WHERE person_id = ?",
                (new_person_id, old_person_id),
            )
        if new_label is not None:
            # Keep the roster on the same name as the perception record.
            await db.execute(
                "UPDATE OR IGNORE people_notes SET name = ?, person_id = ? "
                "WHERE person_id = ?",
                (new_label, new_person_id, old_person_id),
            )
        async with db.execute(
            "SELECT DISTINCT photo_id FROM photo_people WHERE person_id = ?",
            (new_person_id,),
        ) as affected_cursor:
            affected = [r["photo_id"] async for r in affected_cursor]
        for pid in affected:
            await self._reindex_photo(pid)
        await db.commit()
        if cursor.rowcount:
            logger.info(
                "Repointed %d photo person tag(s): %s -> %s (%s)",
                cursor.rowcount, old_person_id, new_person_id, new_label,
            )
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Slideshow
    # ------------------------------------------------------------------

    async def add_to_slideshow(self, photo_id: str) -> bool:
        """Add a photo to slideshow rotation."""
        db = self._ensure_db()
        result = await db.execute(
            "UPDATE photos SET in_slideshow = 1, updated_at = ? "
            "WHERE id = ? AND deleted_at IS NULL",
            (_now_iso(), photo_id),
        )
        await db.commit()
        return result.rowcount > 0

    async def remove_from_slideshow(self, photo_id: str) -> bool:
        """Remove a photo from slideshow rotation."""
        db = self._ensure_db()
        result = await db.execute(
            "UPDATE photos SET in_slideshow = 0, updated_at = ? WHERE id = ?",
            (_now_iso(), photo_id),
        )
        await db.commit()
        return result.rowcount > 0

    async def get_slideshow_photos(
        self, *, limit: int = 100
    ) -> list[PhotoRecord]:
        """Get photos that are in the slideshow rotation."""
        db = self._ensure_db()
        records: list[PhotoRecord] = []
        async with db.execute(
            "SELECT * FROM photos "
            "WHERE in_slideshow = 1 AND deleted_at IS NULL "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ) as cursor:
            async for row in cursor:
                records.append(await self._row_to_record(row))
        return records

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    async def get_storage_info(self) -> StorageInfo:
        """Get current storage usage vs quota.

        Measures actual file sizes in the photos directory against the
        configurable percentage of disk capacity.
        """
        config = get_config()
        storage_path = Path(config.photos.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)

        # Calculate used bytes by summing all files in storage directory
        used_bytes = 0
        if storage_path.exists():
            for dirpath, _dirnames, filenames in os.walk(storage_path):
                for f in filenames:
                    fp = Path(dirpath) / f
                    try:
                        used_bytes += fp.stat().st_size
                    except OSError:
                        pass

        # Calculate quota from disk capacity
        disk_usage = shutil.disk_usage(str(storage_path))
        quota_bytes = int(
            disk_usage.total * config.photos.max_storage_percent / 100
        )

        used_percent = (used_bytes / quota_bytes * 100) if quota_bytes > 0 else 0.0

        return StorageInfo(
            used_bytes=used_bytes,
            quota_bytes=quota_bytes,
            used_percent=round(used_percent, 1),
        )

    async def count_photos(self, *, include_deleted: bool = False) -> int:
        """Count photos in the library (active by default)."""
        db = self._ensure_db()
        sql = "SELECT COUNT(*) AS cnt FROM photos"
        if not include_deleted:
            sql += " WHERE deleted_at IS NULL"
        async with db.execute(sql) as cursor:
            row = await cursor.fetchone()
            return int(row["cnt"]) if row else 0

    # ------------------------------------------------------------------
    # FTS rebuild
    # ------------------------------------------------------------------

    async def rebuild_fts_index(self) -> None:
        """Rebuild the FTS5 index from the photos, tags and people tables."""
        db = self._ensure_db()
        count = await self._rebuild_fts_rows()
        await db.commit()
        logger.info("Rebuilt photos FTS index (%d photos)", count)

    async def _rebuild_fts_rows(self) -> int:
        """Replace every FTS row from the live tables. Does not commit."""
        db = self._ensure_db()
        await db.execute("DELETE FROM photos_fts")
        tags: dict[str, list[str]] = {}
        async with db.execute(
            "SELECT pt.photo_id, t.name FROM photo_tags pt "
            "JOIN tags t ON t.id = pt.tag_id ORDER BY pt.photo_id, pt.tag_id"
        ) as cursor:
            async for row in cursor:
                tags.setdefault(row["photo_id"], []).append(row["name"])
        people: dict[str, list[str]] = {}
        async with db.execute(
            "SELECT photo_id, label FROM photo_people "
            "WHERE confidence != 'unsure' ORDER BY id"
        ) as cursor:
            async for row in cursor:
                people.setdefault(row["photo_id"], []).append(row["label"])
        async with db.execute("SELECT id, description FROM photos") as cursor:
            photos = await cursor.fetchall()
        await db.executemany(
            "INSERT INTO photos_fts (photo_id, description, tags, people) "
            "VALUES (?, ?, ?, ?)",
            [
                (
                    row["id"], row["description"],
                    " ".join(tags.get(row["id"], [])),
                    " ".join(people.get(row["id"], [])),
                )
                for row in photos
            ],
        )
        return len(photos)

    async def _reindex_photo(self, photo_id: str) -> None:
        """Refresh one photo's FTS row. Does not commit."""
        if not self._fts_ready:
            return
        db = self._ensure_db()
        await db.execute(
            "DELETE FROM photos_fts WHERE photo_id = ?", (photo_id,)
        )
        async with db.execute(
            "SELECT description FROM photos WHERE id = ?", (photo_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return
        async with db.execute(
            "SELECT t.name FROM tags t JOIN photo_tags pt ON t.id = pt.tag_id "
            "WHERE pt.photo_id = ?",
            (photo_id,),
        ) as cursor:
            tags = [r["name"] async for r in cursor]
        async with db.execute(
            "SELECT label FROM photo_people "
            "WHERE photo_id = ? AND confidence != 'unsure'",
            (photo_id,),
        ) as cursor:
            people = [r["label"] async for r in cursor]
        await db.execute(
            "INSERT INTO photos_fts (photo_id, description, tags, people) "
            "VALUES (?, ?, ?, ?)",
            (photo_id, row["description"], " ".join(tags), " ".join(people)),
        )

    # ------------------------------------------------------------------
    # Batch load (search)
    # ------------------------------------------------------------------

    async def load_records(
        self, *, include_deleted: bool = False
    ) -> tuple[dict[str, PhotoRecord], dict[str, bytes | None]]:
        """Every photo as a record, plus its raw embedding blob.

        Three queries in total (photos, tags, people) however many
        photos there are; the per-photo grouping is done in Python.
        """
        db = self._ensure_db()
        sql = "SELECT * FROM photos"
        if not include_deleted:
            sql += " WHERE deleted_at IS NULL"
        async with db.execute(sql) as cursor:
            rows = await cursor.fetchall()

        tags: dict[str, list[str]] = {}
        async with db.execute(
            "SELECT pt.photo_id, t.name FROM photo_tags pt "
            "JOIN tags t ON t.id = pt.tag_id ORDER BY pt.photo_id, pt.tag_id"
        ) as cursor:
            async for row in cursor:
                tags.setdefault(row["photo_id"], []).append(row["name"])
        people: dict[str, list[Any]] = {}
        async with db.execute(
            "SELECT * FROM photo_people ORDER BY id"
        ) as cursor:
            async for row in cursor:
                people.setdefault(row["photo_id"], []).append(row)

        records: dict[str, PhotoRecord] = {}
        embeddings: dict[str, bytes | None] = {}
        for row in rows:
            pid = row["id"]
            records[pid] = self._build_record(
                row, tags.get(pid, []), people.get(pid, [])
            )
            embeddings[pid] = row["embedding"]
        return records, embeddings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _row_to_record(self, row: aiosqlite.Row) -> PhotoRecord:
        """Convert a database row to a PhotoRecord with tags and people."""
        db = self._ensure_db()
        photo_id = row["id"]

        async with db.execute(
            "SELECT t.name FROM tags t "
            "JOIN photo_tags pt ON t.id = pt.tag_id "
            "WHERE pt.photo_id = ?",
            (photo_id,),
        ) as cursor:
            tags = [tag_row["name"] async for tag_row in cursor]

        async with db.execute(
            "SELECT * FROM photo_people WHERE photo_id = ? ORDER BY id",
            (photo_id,),
        ) as cursor:
            person_rows = await cursor.fetchall()

        return self._build_record(row, tags, list(person_rows))

    @staticmethod
    def _build_record(
        row: aiosqlite.Row, tags: list[str], person_rows: list[Any]
    ) -> PhotoRecord:
        """Assemble a PhotoRecord from already-fetched rows."""
        # "unsure" rows are suggestions, not people.
        people: list[dict[str, Any]] = []
        suggestions: list[dict[str, Any]] = []
        for person_row in person_rows:
            entry = {
                "label": person_row["label"],
                "person_id": person_row["person_id"],
                "bbox_x": person_row["bbox_x"],
                "bbox_y": person_row["bbox_y"],
                "bbox_w": person_row["bbox_w"],
                "bbox_h": person_row["bbox_h"],
                "confidence": person_row["confidence"],
                "source": person_row["source"],
                "basis": person_row["basis"],
                "conversation_id": person_row["conversation_id"],
                "channel": person_row["channel"],
            }
            if entry["confidence"] == "unsure":
                suggestions.append(entry)
            else:
                people.append(entry)

        return PhotoRecord(
            id=row["id"],
            filename=row["filename"],
            source=row["source"],
            sender=row["sender"],
            description=row["description"],
            orientation=row["orientation"],
            width=row["width"],
            height=row["height"],
            file_size=row["file_size"],
            in_slideshow=bool(row["in_slideshow"]),
            created_at=row["created_at"],
            deleted_at=row["deleted_at"],
            updated_at=row["updated_at"],
            tags=list(tags),
            people=people,
            suggestions=suggestions,
            people_confirmed_at=row["people_confirmed_at"],
            people_confirmed_conversation=row["people_confirmed_conversation"],
            people_confirmed_channel=row["people_confirmed_channel"],
        )

    async def _get_photo_raw(self, photo_id: str) -> aiosqlite.Row | None:
        """Get raw database row for a photo (no tags/people join)."""
        db = self._ensure_db()
        async with db.execute(
            "SELECT * FROM photos WHERE id = ?", (photo_id,)
        ) as cursor:
            return await cursor.fetchone()
