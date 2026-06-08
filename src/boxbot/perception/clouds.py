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

from boxbot.core.paths import PERCEPTION_DIR

logger = logging.getLogger(__name__)

# Default database location (anchored to project root via paths module)
DB_DIR = PERCEPTION_DIR
DB_PATH = DB_DIR / "perception.db"

# Embedding caps per person
MAX_VISUAL_EMBEDDINGS = 200
MAX_VOICE_EMBEDDINGS = 50

# ---------------------------------------------------------------------------
# Provenance — how we came to believe an embedding belongs to a person.
# ---------------------------------------------------------------------------
# Strongest → weakest. Drives weighted matching and eviction protection.
# See docs/plans/person-id-overhaul.md.
PROVENANCE_AGENT_IDENTIFY = "agent_identify"
PROVENANCE_VOICE_VISUAL_AGREE = "voice_visual_agree"
PROVENANCE_VOICE_DOA = "voice_doa"
PROVENANCE_VISUAL_REID = "visual_reid"
PROVENANCE_CONTEXT_INFERRED = "context_inferred"
PROVENANCE_SEED = "seed"
PROVENANCE_LEGACY = "legacy"

# Relative trust weight per provenance, used when scoring a cloud match so a
# single hand-confirmed anchor outranks many shaky auto-admitted points.
PROVENANCE_WEIGHT: dict[str, float] = {
    PROVENANCE_AGENT_IDENTIFY: 3.0,
    PROVENANCE_VOICE_VISUAL_AGREE: 2.0,
    PROVENANCE_VOICE_DOA: 1.5,
    PROVENANCE_VISUAL_REID: 1.0,
    PROVENANCE_CONTEXT_INFERRED: 0.7,
    PROVENANCE_SEED: 1.0,
    PROVENANCE_LEGACY: 1.0,
}
_DEFAULT_PROVENANCE_WEIGHT = 1.0

# Provenance tiers that are anchors — never evicted by the cap janitor, and the
# only ground truth the dream-cycle reconciliation self-labels against.
ANCHOR_PROVENANCES = frozenset({PROVENANCE_AGENT_IDENTIFY})


def provenance_weight(provenance: str | None) -> float:
    """Return the trust weight for a provenance label (default if unknown)."""
    if not provenance:
        return _DEFAULT_PROVENANCE_WEIGHT
    return PROVENANCE_WEIGHT.get(provenance, _DEFAULT_PROVENANCE_WEIGHT)

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
    provenance TEXT NOT NULL DEFAULT 'legacy',
    confidence REAL,
    source_ref TEXT,
    FOREIGN KEY (person_id) REFERENCES persons(id)
);

CREATE TABLE IF NOT EXISTS voice_embeddings (
    id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    timestamp TEXT NOT NULL,
    provenance TEXT NOT NULL DEFAULT 'legacy',
    confidence REAL,
    source_ref TEXT,
    FOREIGN KEY (person_id) REFERENCES persons(id)
);

CREATE TABLE IF NOT EXISTS centroids (
    person_id TEXT PRIMARY KEY,
    visual_centroid BLOB,
    voice_centroid BLOB,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES persons(id)
);

-- Audit log of identity corrections (agent_identify overrides, dream-cycle
-- merges/relabels). Enables undo and feeds reconciliation. See
-- docs/plans/person-id-overhaul.md.
CREATE TABLE IF NOT EXISTS id_corrections (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    source_ref TEXT,
    from_person_id TEXT,
    to_person_id TEXT,
    source TEXT NOT NULL,
    detail TEXT
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
            visual_cols = {row[1] async for row in cur}
        if "crop_path" not in visual_cols:
            await db.execute(
                "ALTER TABLE visual_embeddings ADD COLUMN crop_path TEXT"
            )
            await db.commit()
            logger.info("Migrated visual_embeddings: added crop_path column")

        # Provenance/confidence/source_ref columns (person-id-overhaul).
        # SQLite ALTER ... ADD COLUMN can't add a NOT NULL column without a
        # constant default, so we add with the 'legacy' default; pre-existing
        # rows keep that label (treated as low-trust, never an anchor).
        for table in ("visual_embeddings", "voice_embeddings"):
            async with db.execute(f"PRAGMA table_info({table})") as cur:
                cols = {row[1] async for row in cur}
            if "provenance" not in cols:
                await db.execute(
                    f"ALTER TABLE {table} ADD COLUMN "
                    "provenance TEXT NOT NULL DEFAULT 'legacy'"
                )
                logger.info("Migrated %s: added provenance column", table)
            if "confidence" not in cols:
                await db.execute(
                    f"ALTER TABLE {table} ADD COLUMN confidence REAL"
                )
                logger.info("Migrated %s: added confidence column", table)
            if "source_ref" not in cols:
                await db.execute(
                    f"ALTER TABLE {table} ADD COLUMN source_ref TEXT"
                )
                logger.info("Migrated %s: added source_ref column", table)
        await db.commit()

        # id_corrections table for older DBs created before it existed.
        await db.execute(
            """CREATE TABLE IF NOT EXISTS id_corrections (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                source_ref TEXT,
                from_person_id TEXT,
                to_person_id TEXT,
                source TEXT NOT NULL,
                detail TEXT
            )"""
        )
        await db.commit()

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
        provenance: str = PROVENANCE_LEGACY,
        confidence: float | None = None,
        source_ref: str | None = None,
    ) -> str:
        """Add a visual embedding to a person's cloud.

        Enforces the per-person embedding cap (isolation+age eviction).

        Args:
            person_id: Person to add embedding for.
            embedding: Float32 embedding vector (typically 512-dim).
            voice_confirmed: Whether this embedding was voice-confirmed.
            crop_path: Optional path to the saved crop image.
            provenance: How we came to believe this belongs to the person
                (see PROVENANCE_* constants). Drives matching weight + eviction.
            confidence: Match/admission confidence at enrollment time.
            source_ref: Session ref the embedding was captured under (audit).

        Returns:
            The generated embedding ID, or ``""`` if rejected as non-finite.
        """
        if embedding is None or not np.all(np.isfinite(embedding)):
            logger.warning(
                "Refusing to add non-finite visual embedding for person %s",
                person_id,
            )
            return ""

        db = self._ensure_db()
        emb_id = _generate_id()
        now = _now_iso()

        await db.execute(
            """INSERT INTO visual_embeddings
               (id, person_id, embedding, timestamp, voice_confirmed,
                crop_path, provenance, confidence, source_ref)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (emb_id, person_id, _embedding_to_blob(embedding), now,
             1 if voice_confirmed else 0, crop_path,
             provenance, confidence, source_ref),
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

    async def get_visual_records(self, person_id: str) -> list[dict]:
        """Get full visual embedding records for a person (for reconciliation).

        Returns dicts with id, embedding (L2-unit), provenance, confidence,
        crop_path, timestamp — everything the dream-cycle hygiene pass needs to
        score, relabel, or evict a point. Ordered oldest-first.
        """
        db = self._ensure_db()
        records: list[dict] = []
        async with db.execute(
            """SELECT id, embedding, provenance, confidence, crop_path, timestamp
               FROM visual_embeddings WHERE person_id = ?
               ORDER BY timestamp ASC""",
            (person_id,),
        ) as cursor:
            async for row in cursor:
                emb = _blob_to_embedding(row["embedding"])
                n = np.linalg.norm(emb)
                if n > 0:
                    emb = emb / n
                records.append({
                    "id": row["id"],
                    "embedding": emb.astype(np.float32),
                    "provenance": row["provenance"],
                    "confidence": row["confidence"],
                    "crop_path": row["crop_path"],
                    "timestamp": row["timestamp"],
                })
        return records

    async def get_visual_clouds(
        self,
    ) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
        """Get every person's full visual embedding cloud, with match weights.

        Returns ``{person_id: (name, embeddings, weights)}`` where
        ``embeddings`` is an ``(N, D)`` float32 array of that person's stored,
        L2-unit visual embeddings and ``weights`` is an ``(N,)`` float32 array
        of per-embedding provenance trust weights. This is the cloud-based
        counterpart to the single visual centroid — matching against the cloud
        (provenance-weighted top-k cosine) preserves per-appearance modes that
        a single averaged centroid collapses, and lets a hand-confirmed anchor
        outweigh shaky auto-admitted points. See
        docs/plans/person-id-overhaul.md.
        """
        db = self._ensure_db()
        acc: dict[str, tuple[str, list[np.ndarray], list[float]]] = {}
        async with db.execute(
            """SELECT v.person_id, p.name, v.embedding, v.provenance
               FROM visual_embeddings v
               JOIN persons p ON v.person_id = p.id"""
        ) as cursor:
            async for row in cursor:
                emb = _blob_to_embedding(row["embedding"])
                if not np.all(np.isfinite(emb)):
                    continue
                n = np.linalg.norm(emb)
                if n > 0:
                    emb = emb / n
                name, vecs, wts = acc.setdefault(
                    row["person_id"], (row["name"], [], [])
                )
                vecs.append(emb.astype(np.float32))
                wts.append(provenance_weight(row["provenance"]))
        clouds: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
        for pid, (name, vecs, wts) in acc.items():
            if vecs:
                clouds[pid] = (
                    name,
                    np.stack(vecs),
                    np.asarray(wts, dtype=np.float32),
                )
        return clouds

    # ------------------------------------------------------------------
    # Voice embeddings (schema ready, full implementation for future use)
    # ------------------------------------------------------------------

    async def add_voice_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        provenance: str = PROVENANCE_LEGACY,
        confidence: float | None = None,
        source_ref: str | None = None,
    ) -> str:
        """Add a voice embedding to a person's cloud.

        Embeddings containing NaN or Inf are rejected with an empty
        string return. A single NaN that lands in storage poisons the
        person's centroid (NaN propagates through ``np.mean``) and
        silently disables every cosine similarity check for that
        speaker — so the storage path is the right place to fail
        loudly rather than trust upstream callers.

        Args:
            person_id: Person to add embedding for.
            embedding: Float32 embedding vector (typically 192-dim).

        Returns:
            The generated embedding ID, or ``""`` if the embedding was
            rejected as non-finite.
        """
        if embedding is None or not np.all(np.isfinite(embedding)):
            logger.warning(
                "Refusing to add non-finite voice embedding for person %s "
                "(%d non-finite values)",
                person_id,
                int((~np.isfinite(embedding)).sum())
                if embedding is not None else -1,
            )
            return ""

        db = self._ensure_db()
        emb_id = _generate_id()
        now = _now_iso()

        await db.execute(
            """INSERT INTO voice_embeddings
               (id, person_id, embedding, timestamp,
                provenance, confidence, source_ref)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (emb_id, person_id, _embedding_to_blob(embedding), now,
             provenance, confidence, source_ref),
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

    async def get_voice_clouds(self) -> dict[str, tuple[str, np.ndarray]]:
        """Get every person's full voice embedding cloud.

        Returns ``{person_id: (name, embeddings)}`` where ``embeddings``
        is an ``(N, D)`` float32 array of that person's stored, L2-unit
        voice embeddings. Only includes persons with at least one usable
        (finite) embedding. This is the cloud-based counterpart to
        :meth:`get_voice_centroids` — matching against the cloud (top-k
        cosine) preserves per-condition modes that a single averaged
        centroid collapses. See docs/voice-id-redesign.md.
        """
        db = self._ensure_db()
        clouds: dict[str, tuple[str, np.ndarray]] = {}
        async with db.execute(
            """SELECT v.person_id, p.name, v.embedding
               FROM voice_embeddings v
               JOIN persons p ON v.person_id = p.id"""
        ) as cursor:
            acc: dict[str, tuple[str, list[np.ndarray]]] = {}
            async for row in cursor:
                emb = _blob_to_embedding(row["embedding"])
                if not np.all(np.isfinite(emb)):
                    continue
                n = np.linalg.norm(emb)
                if n > 0:
                    emb = emb / n
                name, vecs = acc.setdefault(row["person_id"], (row["name"], []))
                vecs.append(emb.astype(np.float32))
        for pid, (name, vecs) in acc.items():
            if vecs:
                clouds[pid] = (name, np.stack(vecs))
        return clouds

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

        Non-finite (NaN/Inf) embeddings are filtered out before
        averaging — a single NaN row would otherwise poison the entire
        centroid and silently disable downstream similarity checks
        (this happened in production on 2026-04-26).

        Returns:
            The new centroid, or None if no usable embeddings exist.
        """
        embeddings = await self.get_voice_embeddings(person_id)
        if not embeddings:
            return None

        usable = [emb for _, emb in embeddings if np.all(np.isfinite(emb))]
        skipped = len(embeddings) - len(usable)
        if skipped:
            logger.warning(
                "Skipped %d non-finite voice embedding(s) for person %s "
                "during centroid recompute",
                skipped, person_id,
            )
        if not usable:
            return None

        mean = np.mean(usable, axis=0).astype(np.float32)
        if not np.all(np.isfinite(mean)):
            logger.error(
                "Voice centroid for person %s is non-finite after "
                "filtering — this should be impossible; aborting update",
                person_id,
            )
            return None
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
        """Evict visual embeddings by isolation + age when over cap.

        Mirrors :meth:`_enforce_voice_cap` (docs/voice-id-redesign.md), with one
        addition: **anchor provenances are never evicted** — hand-confirmed
        ground truth is precious, so the janitor only sheds non-anchor points.
        When over cap, evict in one batch down to ~90% of cap, choosing the
        highest ``isolation + λ·age`` first:

        - ``isolation`` = 1 − mean(top-k cosine to nearest neighbours in the
          cloud). Mis-attributed faces have no neighbours → evicted first.
        - ``age`` (oldest = 1.0) breaks ties.

        Distance is to *neighbours*, never the global mean, so genuine
        per-appearance modes (different outfits/lighting) survive.
        """
        db = self._ensure_db()
        async with db.execute(
            """SELECT id, embedding, provenance FROM visual_embeddings
               WHERE person_id = ? ORDER BY timestamp ASC""",
            (person_id,),
        ) as cursor:
            rows = [
                (r["id"], _blob_to_embedding(r["embedding"]), r["provenance"])
                async for r in cursor
            ]
        count = len(rows)
        if count <= max_count:
            return

        low_water = max(1, int(max_count * 0.9))
        n_evict = count - low_water

        # Evictable = non-anchor rows only. Anchors are kept even over cap.
        evictable = [
            (i, rid) for i, (rid, _, prov) in enumerate(rows)
            if prov not in ANCHOR_PROVENANCES
        ]
        if not evictable:
            return
        n_evict = min(n_evict, len(evictable))

        vecs = np.stack([
            (e / np.linalg.norm(e)) if np.linalg.norm(e) > 0 else e
            for _, e, _ in rows
        ]).astype(np.float32)
        sims = vecs @ vecs.T
        np.fill_diagonal(sims, -1.0)
        k = min(3, count - 1)
        topk = np.sort(sims, axis=1)[:, -k:]
        neighbour_sim = topk.mean(axis=1)
        isolation = 1.0 - neighbour_sim
        age = 1.0 - np.arange(count) / max(1, count - 1)
        evict_score = isolation + 0.3 * age

        # Rank only the evictable (non-anchor) rows; take the worst n_evict.
        evictable_sorted = sorted(
            evictable, key=lambda t: evict_score[t[0]], reverse=True
        )
        ids_to_delete = [rid for _, rid in evictable_sorted[:n_evict]]
        for emb_id in ids_to_delete:
            await db.execute(
                "DELETE FROM visual_embeddings WHERE id = ?", (emb_id,)
            )
        await db.commit()

        logger.debug(
            "Evicted %d visual embeddings for person %s (isolation+age)",
            len(ids_to_delete), person_id,
        )

    async def _enforce_voice_cap(
        self, person_id: str, max_count: int = MAX_VOICE_EMBEDDINGS
    ) -> None:
        """Evict voice embeddings by isolation + age when over cap.

        Replaces oldest-first pruning with an outlier-aware policy
        (docs/voice-id-redesign.md). When over cap, evict in a single
        batch down to a low-water mark (~90% of cap), choosing the
        highest ``isolation + λ·age`` first:

        - ``isolation`` = 1 − mean(top-k cosine to nearest neighbours in
          the cloud). Isolated points (contaminated / mis-enrolled clips)
          score high; legit far/noisy clips have neighbours and survive.
        - ``age`` (oldest = 1.0) breaks ties, so order is: outliers, then
          old outliers, then just-old. λ is small so isolation dominates.

        Distance is to *neighbours*, never the global mean, so genuine
        per-condition modes are preserved.
        """
        db = self._ensure_db()
        # Ordered oldest-first so index → age rank.
        embeddings = await self.get_voice_embeddings(person_id)
        count = len(embeddings)
        if count <= max_count:
            return

        low_water = max(1, int(max_count * 0.9))
        n_evict = count - low_water

        ids = [eid for eid, _ in embeddings]
        vecs = np.stack([
            (e / np.linalg.norm(e)) if np.linalg.norm(e) > 0 else e
            for _, e in embeddings
        ]).astype(np.float32)

        sims = vecs @ vecs.T
        np.fill_diagonal(sims, -1.0)  # exclude self
        k = min(3, count - 1)
        # mean of each row's top-k neighbour similarities
        topk = np.sort(sims, axis=1)[:, -k:]
        neighbour_sim = topk.mean(axis=1)
        isolation = 1.0 - neighbour_sim

        # oldest (index 0) → age 1.0, newest → 0.0
        age = 1.0 - np.arange(count) / max(1, count - 1)
        evict_score = isolation + 0.3 * age

        evict_idx = np.argsort(evict_score)[-n_evict:]
        ids_to_delete = [ids[i] for i in evict_idx]
        for emb_id in ids_to_delete:
            await db.execute(
                "DELETE FROM voice_embeddings WHERE id = ?", (emb_id,)
            )
        await db.commit()

        logger.debug(
            "Evicted %d voice embeddings for person %s (isolation+age)",
            n_evict, person_id,
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
