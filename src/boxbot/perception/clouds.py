"""Embedding cloud storage for person identification.

Manages per-person visual and voice embeddings in SQLite at
data/perception/perception.db. Provides centroid computation,
embedding cap enforcement, and person record management.

Follows the same async patterns as memory/store.py and photos/store.py.

Usage:
    from boxbot.perception.clouds import CloudStore

    store = CloudStore()
    await store.initialize()
    person_id = await store.create_person("Jacob")
    await store.add_visual_embedding(person_id, embedding_vec)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import numpy as np

logger = logging.getLogger(__name__)

# Default database location
DB_DIR = Path("data/perception")
DB_PATH = DB_DIR / "perception.db"

# Embedding caps per person
MAX_VISUAL_EMBEDDINGS = 200
MAX_VOICE_EMBEDDINGS = 50

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS persons (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    is_user INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS visual_embeddings (
    id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    timestamp TEXT NOT NULL,
    voice_confirmed INTEGER NOT NULL DEFAULT 0,
    crop_path TEXT,
    FOREIGN KEY (person_id) REFERENCES persons(id)
);

CREATE TABLE IF NOT EXISTS voice_embeddings (
    id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES persons(id)
);

CREATE TABLE IF NOT EXISTS centroids (
    person_id TEXT PRIMARY KEY,
    visual_centroid BLOB,
    voice_centroid BLOB,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES persons(id)
);

CREATE INDEX IF NOT EXISTS idx_visual_person
    ON visual_embeddings(person_id);
CREATE INDEX IF NOT EXISTS idx_visual_timestamp
    ON visual_embeddings(timestamp);
CREATE INDEX IF NOT EXISTS idx_voice_person
    ON voice_embeddings(person_id);
"""


def _generate_id() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _embedding_to_blob(arr: np.ndarray) -> bytes:
    """Convert numpy array to bytes for SQLite BLOB storage."""
    return arr.astype(np.float32).tobytes()


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    """Convert SQLite BLOB back to numpy array."""
    return np.frombuffer(blob, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# CloudStore
# ---------------------------------------------------------------------------


class CloudStore:
    """Async SQLite-backed embedding cloud storage.

    Manages person records, visual/voice embeddings, and centroids.

    Args:
        db_path: Override path to SQLite database. Defaults to
                 data/perception/perception.db.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else DB_PATH
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create database and tables if needed."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()

        # Migrations for existing databases
        await self._migrate(self._db)

        logger.info("Cloud store initialized at %s", self._db_path)

    @staticmethod
    async def _migrate(db: aiosqlite.Connection) -> None:
        """Run schema migrations for existing databases."""
        # Add crop_path column to visual_embeddings if missing
        async with db.execute("PRAGMA table_info(visual_embeddings)") as cur:
            columns = {row[1] async for row in cur}
        if "crop_path" not in columns:
            await db.execute(
                "ALTER TABLE visual_embeddings ADD COLUMN crop_path TEXT"
            )
            await db.commit()
            logger.info("Migrated visual_embeddings: added crop_path column")

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError(
                "CloudStore not initialized. Call initialize() first."
            )
        return self._db

    # ------------------------------------------------------------------
    # Person management
    # ------------------------------------------------------------------

    async def create_person(self, name: str) -> str:
        """Create a person record.

        Args:
            name: Person's name (must be unique).

        Returns:
            The generated person_id (UUID).
        """
        db = self._ensure_db()
        person_id = _generate_id()
        now = _now_iso()

        await db.execute(
            "INSERT INTO persons (id, name, first_seen, last_seen) VALUES (?, ?, ?, ?)",
            (person_id, name, now, now),
        )
        await db.commit()

        logger.info("Created person %s (id=%s)", name, person_id)
        return person_id

    async def get_person_by_name(self, name: str) -> dict | None:
        """Look up person by name.

        Returns:
            Dict with id, name, first_seen, last_seen, is_user — or None.
        """
        db = self._ensure_db()
        async with db.execute(
            "SELECT * FROM persons WHERE name = ?", (name,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_person(row)

    async def get_person(self, person_id: str) -> dict | None:
        """Look up person by ID.

        Returns:
            Dict with id, name, first_seen, last_seen, is_user — or None.
        """
        db = self._ensure_db()
        async with db.execute(
            "SELECT * FROM persons WHERE id = ?", (person_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_person(row)

    async def list_persons(self) -> list[dict]:
        """List all known persons.

        Returns:
            List of person dicts sorted by last_seen (newest first).
        """
        db = self._ensure_db()
        persons: list[dict] = []
        async with db.execute(
            "SELECT * FROM persons ORDER BY last_seen DESC"
        ) as cursor:
            async for row in cursor:
                persons.append(_row_to_person(row))
        return persons

    async def update_last_seen(self, person_id: str) -> None:
        """Update a person's last_seen timestamp to now."""
        db = self._ensure_db()
        await db.execute(
            "UPDATE persons SET last_seen = ? WHERE id = ?",
            (_now_iso(), person_id),
        )
        await db.commit()

    # ------------------------------------------------------------------
    # Visual embeddings
    # ------------------------------------------------------------------

    async def add_visual_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        voice_confirmed: bool = False,
        crop_path: str | None = None,
    ) -> str:
        """Add a visual embedding to a person's cloud.

        Enforces the per-person embedding cap (oldest pruned first).

        Args:
            person_id: Person to add embedding for.
            embedding: Float32 embedding vector (typically 512-dim).
            voice_confirmed: Whether this embedding was voice-confirmed.
            crop_path: Optional path to the saved crop image.

        Returns:
            The generated embedding ID.
        """
        db = self._ensure_db()
        emb_id = _generate_id()
        now = _now_iso()

        await db.execute(
            """INSERT INTO visual_embeddings
               (id, person_id, embedding, timestamp, voice_confirmed, crop_path)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (emb_id, person_id, _embedding_to_blob(embedding), now,
             1 if voice_confirmed else 0, crop_path),
        )
        await db.commit()

        # Enforce cap
        await self._enforce_visual_cap(person_id)

        # Update last_seen
        await self.update_last_seen(person_id)

        return emb_id

    async def get_visual_embeddings(
        self, person_id: str
    ) -> list[tuple[str, np.ndarray]]:
        """Get all visual embeddings for a person.

        Returns:
            List of (embedding_id, embedding_vector) tuples, ordered by
            timestamp ascending (oldest first).
        """
        db = self._ensure_db()
        results: list[tuple[str, np.ndarray]] = []
        async with db.execute(
            "SELECT id, embedding FROM visual_embeddings "
            "WHERE person_id = ? ORDER BY timestamp ASC",
            (person_id,),
        ) as cursor:
            async for row in cursor:
                results.append((row["id"], _blob_to_embedding(row["embedding"])))
        return results

    async def count_visual_embeddings(self, person_id: str) -> int:
        """Count visual embeddings for a person."""
        db = self._ensure_db()
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM visual_embeddings WHERE person_id = ?",
            (person_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row["cnt"]

    # ------------------------------------------------------------------
    # Voice embeddings (schema ready, full implementation for future use)
    # ------------------------------------------------------------------

    async def add_voice_embedding(
        self, person_id: str, embedding: np.ndarray
    ) -> str:
        """Add a voice embedding to a person's cloud.

        Args:
            person_id: Person to add embedding for.
            embedding: Float32 embedding vector (typically 192-dim).

        Returns:
            The generated embedding ID.
        """
        db = self._ensure_db()
        emb_id = _generate_id()
        now = _now_iso()

        await db.execute(
            """INSERT INTO voice_embeddings
               (id, person_id, embedding, timestamp)
               VALUES (?, ?, ?, ?)""",
            (emb_id, person_id, _embedding_to_blob(embedding), now),
        )
        await db.commit()

        await self._enforce_voice_cap(person_id)
        await self.update_last_seen(person_id)

        return emb_id

    async def get_voice_embeddings(
        self, person_id: str
    ) -> list[tuple[str, np.ndarray]]:
        """Get all voice embeddings for a person.

        Returns:
            List of (embedding_id, embedding_vector) tuples, ordered by
            timestamp ascending.
        """
        db = self._ensure_db()
        results: list[tuple[str, np.ndarray]] = []
        async with db.execute(
            "SELECT id, embedding FROM voice_embeddings "
            "WHERE person_id = ? ORDER BY timestamp ASC",
            (person_id,),
        ) as cursor:
            async for row in cursor:
                results.append((row["id"], _blob_to_embedding(row["embedding"])))
        return results

    # ------------------------------------------------------------------
    # Centroids
    # ------------------------------------------------------------------

    async def get_centroids(self) -> dict[str, tuple[str, np.ndarray]]:
        """Get all visual centroids.

        Returns:
            {person_id: (person_name, centroid_vector)} dict.
            Only includes persons with a computed visual centroid.
        """
        db = self._ensure_db()
        centroids: dict[str, tuple[str, np.ndarray]] = {}
        async with db.execute(
            """SELECT c.person_id, p.name, c.visual_centroid
               FROM centroids c
               JOIN persons p ON c.person_id = p.id
               WHERE c.visual_centroid IS NOT NULL"""
        ) as cursor:
            async for row in cursor:
                centroids[row["person_id"]] = (
                    row["name"],
                    _blob_to_embedding(row["visual_centroid"]),
                )
        return centroids

    async def update_centroid(
        self, person_id: str, centroid: np.ndarray
    ) -> None:
        """Store or update a visual centroid for a person."""
        db = self._ensure_db()
        now = _now_iso()

        # Upsert — INSERT OR REPLACE on the primary key
        await db.execute(
            """INSERT INTO centroids (person_id, visual_centroid, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(person_id) DO UPDATE SET
                   visual_centroid = excluded.visual_centroid,
                   updated_at = excluded.updated_at""",
            (person_id, _embedding_to_blob(centroid), now),
        )
        await db.commit()

    async def recompute_centroid(self, person_id: str) -> np.ndarray | None:
        """Recompute and store centroid from current visual embeddings.

        Returns:
            The new centroid, or None if no embeddings exist.
        """
        embeddings = await self.get_visual_embeddings(person_id)
        if not embeddings:
            return None

        vectors = [emb for _, emb in embeddings]
        mean = np.mean(vectors, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm

        await self.update_centroid(person_id, mean)
        return mean

    async def get_voice_centroids(self) -> dict[str, tuple[str, np.ndarray]]:
        """Get all voice centroids.

        Returns:
            {person_id: (person_name, centroid_vector)} dict.
            Only includes persons with a computed voice centroid.
        """
        db = self._ensure_db()
        centroids: dict[str, tuple[str, np.ndarray]] = {}
        async with db.execute(
            """SELECT c.person_id, p.name, c.voice_centroid
               FROM centroids c
               JOIN persons p ON c.person_id = p.id
               WHERE c.voice_centroid IS NOT NULL"""
        ) as cursor:
            async for row in cursor:
                centroids[row["person_id"]] = (
                    row["name"],
                    _blob_to_embedding(row["voice_centroid"]),
                )
        return centroids

    async def update_voice_centroid(
        self, person_id: str, centroid: np.ndarray
    ) -> None:
        """Store or update a voice centroid for a person."""
        db = self._ensure_db()
        now = _now_iso()

        await db.execute(
            """INSERT INTO centroids (person_id, voice_centroid, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(person_id) DO UPDATE SET
                   voice_centroid = excluded.voice_centroid,
                   updated_at = excluded.updated_at""",
            (person_id, _embedding_to_blob(centroid), now),
        )
        await db.commit()

    async def recompute_voice_centroid(self, person_id: str) -> np.ndarray | None:
        """Recompute and store voice centroid from current voice embeddings.

        Returns:
            The new centroid, or None if no embeddings exist.
        """
        embeddings = await self.get_voice_embeddings(person_id)
        if not embeddings:
            return None

        vectors = [emb for _, emb in embeddings]
        mean = np.mean(vectors, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm

        await self.update_voice_centroid(person_id, mean)
        return mean

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    async def _enforce_visual_cap(
        self, person_id: str, max_count: int = MAX_VISUAL_EMBEDDINGS
    ) -> None:
        """Delete oldest visual embeddings if over cap."""
        db = self._ensure_db()
        count = await self.count_visual_embeddings(person_id)
        if count <= max_count:
            return

        excess = count - max_count
        async with db.execute(
            """SELECT id FROM visual_embeddings
               WHERE person_id = ?
               ORDER BY timestamp ASC
               LIMIT ?""",
            (person_id, excess),
        ) as cursor:
            ids_to_delete = [row["id"] async for row in cursor]

        for emb_id in ids_to_delete:
            await db.execute(
                "DELETE FROM visual_embeddings WHERE id = ?", (emb_id,)
            )
        await db.commit()

        logger.debug(
            "Pruned %d visual embeddings for person %s", excess, person_id
        )

    async def _enforce_voice_cap(
        self, person_id: str, max_count: int = MAX_VOICE_EMBEDDINGS
    ) -> None:
        """Delete oldest voice embeddings if over cap."""
        db = self._ensure_db()

        async with db.execute(
            "SELECT COUNT(*) as cnt FROM voice_embeddings WHERE person_id = ?",
            (person_id,),
        ) as cursor:
            row = await cursor.fetchone()
            count = row["cnt"]

        if count <= max_count:
            return

        excess = count - max_count
        async with db.execute(
            """SELECT id FROM voice_embeddings
               WHERE person_id = ?
               ORDER BY timestamp ASC
               LIMIT ?""",
            (person_id, excess),
        ) as cursor:
            ids_to_delete = [row["id"] async for row in cursor]

        for emb_id in ids_to_delete:
            await db.execute(
                "DELETE FROM voice_embeddings WHERE id = ?", (emb_id,)
            )
        await db.commit()

        logger.debug(
            "Pruned %d voice embeddings for person %s", excess, person_id
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_person(row: aiosqlite.Row) -> dict:
    """Convert a database row to a person dict."""
    return {
        "id": row["id"],
        "name": row["name"],
        "first_seen": row["first_seen"],
        "last_seen": row["last_seen"],
        "is_user": bool(row["is_user"]),
    }
