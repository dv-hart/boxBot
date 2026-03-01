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
    people: list[dict[str, Any]] = field(default_factory=list)


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

_FTS_SCHEMA_SQL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS photos_fts USING fts5(
    description,
    content='photos',
    content_rowid='rowid'
);
"""

# Triggers to keep the FTS index in sync with the photos table.
_FTS_TRIGGERS_SQL = """\
CREATE TRIGGER IF NOT EXISTS photos_ai AFTER INSERT ON photos BEGIN
    INSERT INTO photos_fts(rowid, description)
    VALUES (new.rowid, new.description);
END;

CREATE TRIGGER IF NOT EXISTS photos_ad AFTER DELETE ON photos BEGIN
    INSERT INTO photos_fts(photos_fts, rowid, description)
    VALUES ('delete', old.rowid, old.description);
END;

CREATE TRIGGER IF NOT EXISTS photos_au AFTER UPDATE OF description ON photos BEGIN
    INSERT INTO photos_fts(photos_fts, rowid, description)
    VALUES ('delete', old.rowid, old.description);
    INSERT INTO photos_fts(rowid, description)
    VALUES (new.rowid, new.description);
END;
"""


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
        await self._db.executescript(_FTS_SCHEMA_SQL)
        await self._db.executescript(_FTS_TRIGGERS_SQL)
        await self._db.commit()
        logger.info("Photo store initialized at %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

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
        if people:
            for person in people:
                await db.execute(
                    """INSERT INTO photo_people
                       (photo_id, person_id, label, bbox_x, bbox_y, bbox_w, bbox_h)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        pid,
                        person.get("person_id"),
                        person["label"],
                        person.get("bbox_x"),
                        person.get("bbox_y"),
                        person.get("bbox_w"),
                        person.get("bbox_h"),
                    ),
                )

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

        await db.execute("DELETE FROM photo_tags WHERE tag_id = ?", (tag_id,))
        await db.execute("DELETE FROM tags WHERE id = ?", (tag_id,))
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

    async def update_people(
        self, photo_id: str, people: list[dict[str, Any]]
    ) -> bool:
        """Replace all people annotations on a photo.

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
            await db.execute(
                """INSERT INTO photo_people
                   (photo_id, person_id, label, bbox_x, bbox_y, bbox_w, bbox_h)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    photo_id,
                    person.get("person_id"),
                    person["label"],
                    person.get("bbox_x"),
                    person.get("bbox_y"),
                    person.get("bbox_w"),
                    person.get("bbox_h"),
                ),
            )

        await db.execute(
            "UPDATE photos SET updated_at = ? WHERE id = ?",
            (_now_iso(), photo_id),
        )
        await db.commit()
        return True

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

    # ------------------------------------------------------------------
    # FTS rebuild
    # ------------------------------------------------------------------

    async def rebuild_fts_index(self) -> None:
        """Rebuild the FTS5 index from the photos table."""
        db = self._ensure_db()
        await db.execute("INSERT INTO photos_fts(photos_fts) VALUES('rebuild')")
        await db.commit()
        logger.info("Rebuilt photos FTS index")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _row_to_record(self, row: aiosqlite.Row) -> PhotoRecord:
        """Convert a database row to a PhotoRecord with tags and people."""
        db = self._ensure_db()
        photo_id = row["id"]

        # Fetch tags
        tags: list[str] = []
        async with db.execute(
            "SELECT t.name FROM tags t "
            "JOIN photo_tags pt ON t.id = pt.tag_id "
            "WHERE pt.photo_id = ?",
            (photo_id,),
        ) as cursor:
            async for tag_row in cursor:
                tags.append(tag_row["name"])

        # Fetch people
        people: list[dict[str, Any]] = []
        async with db.execute(
            "SELECT * FROM photo_people WHERE photo_id = ?", (photo_id,)
        ) as cursor:
            async for person_row in cursor:
                people.append({
                    "label": person_row["label"],
                    "person_id": person_row["person_id"],
                    "bbox_x": person_row["bbox_x"],
                    "bbox_y": person_row["bbox_y"],
                    "bbox_w": person_row["bbox_w"],
                    "bbox_h": person_row["bbox_h"],
                })

        return PhotoRecord(
            id=photo_id,
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
            tags=tags,
            people=people,
        )

    async def _get_photo_raw(self, photo_id: str) -> aiosqlite.Row | None:
        """Get raw database row for a photo (no tags/people join)."""
        db = self._ensure_db()
        async with db.execute(
            "SELECT * FROM photos WHERE id = ?", (photo_id,)
        ) as cursor:
            return await cursor.fetchone()
