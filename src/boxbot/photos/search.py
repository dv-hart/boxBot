"""Shared photo search backend.

Provides hybrid vector + BM25 retrieval for the photo library. Used by
both the search_photos tool and the boxbot_sdk.photos.search() function
(same DRY pattern as the memory search backend).

Search pipeline:
  1. Hybrid retrieval on descriptions
     - Vector cosine similarity (MiniLM embeddings) — weight 0.6
     - SQLite FTS5 BM25 keyword matching — weight 0.4
     - Combined score ranks candidate set
  2. Structured filters (tags, people, date range, source, slideshow)
  3. Return ranked results with metadata

Usage:
    from boxbot.photos.search import hybrid_search, get_photo_detail

    results = await hybrid_search(query="beach photos with Jacob")
    detail = await get_photo_detail("photo_abc123def456")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from boxbot.memory.embeddings import EMBEDDING_DIM, cosine_similarity, embed
from boxbot.photos.store import PhotoRecord, PhotoStore

logger = logging.getLogger(__name__)

# Hybrid scoring weights
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4


@dataclass
class SearchResult:
    """A single photo search result with relevance score."""

    photo: PhotoRecord
    score: float = 0.0


async def hybrid_search(
    store: PhotoStore,
    *,
    query: str | None = None,
    tags: list[str] | None = None,
    people: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    source: str | None = None,
    in_slideshow: bool | None = None,
    include_deleted: bool = False,
    limit: int = 20,
) -> list[SearchResult]:
    """Search photos using hybrid vector + BM25 retrieval with filters.

    Args:
        store: The PhotoStore instance to search against.
        query: Natural language search query. If None, only filters apply.
        tags: Filter by tags (AND logic — photo must have all listed tags).
        people: Filter by person labels (AND logic).
        date_from: ISO 8601 date string — only photos on or after this date.
        date_to: ISO 8601 date string — only photos on or before this date.
        source: Filter by source ("whatsapp", "camera", "upload").
        in_slideshow: Filter by slideshow membership.
        include_deleted: Whether to include soft-deleted photos.
        limit: Maximum number of results to return.

    Returns:
        List of SearchResult ordered by relevance score (highest first).
    """
    db = store._ensure_db()

    # --- Step 1: Candidate retrieval ---

    # Build base WHERE clause for filters
    where_clauses: list[str] = []
    params: list[Any] = []

    if not include_deleted:
        where_clauses.append("p.deleted_at IS NULL")

    if source:
        where_clauses.append("p.source = ?")
        params.append(source)

    if in_slideshow is not None:
        where_clauses.append("p.in_slideshow = ?")
        params.append(1 if in_slideshow else 0)

    if date_from:
        where_clauses.append("p.created_at >= ?")
        params.append(date_from)

    if date_to:
        where_clauses.append("p.created_at <= ?")
        params.append(date_to)

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    # Fetch candidate photos matching basic filters
    candidate_sql = f"SELECT * FROM photos p WHERE {where_sql}"
    candidates: dict[str, dict[str, Any]] = {}

    async with db.execute(candidate_sql, params) as cursor:
        async for row in cursor:
            candidates[row["id"]] = dict(row)

    if not candidates:
        return []

    # --- Step 2: Tag and people filtering (AND logic) ---

    if tags:
        tag_matched_ids = await _filter_by_tags(db, set(candidates.keys()), tags)
        candidates = {
            pid: c for pid, c in candidates.items() if pid in tag_matched_ids
        }

    if people:
        people_matched_ids = await _filter_by_people(db, set(candidates.keys()), people)
        candidates = {
            pid: c for pid, c in candidates.items() if pid in people_matched_ids
        }

    if not candidates:
        return []

    # --- Step 3: Hybrid scoring ---

    scored: dict[str, float] = {pid: 0.0 for pid in candidates}

    if query:
        # Vector similarity scoring
        query_embedding = embed(query)

        for pid, row in candidates.items():
            if row["embedding"]:
                photo_embedding = np.frombuffer(
                    row["embedding"], dtype=np.float32
                )
                if len(photo_embedding) == EMBEDDING_DIM:
                    sim = cosine_similarity(query_embedding, photo_embedding)
                    # Normalize similarity from [-1, 1] to [0, 1]
                    scored[pid] += VECTOR_WEIGHT * ((sim + 1.0) / 2.0)

        # BM25 scoring via FTS5
        bm25_scores = await _bm25_scores(db, query, set(candidates.keys()))
        if bm25_scores:
            # Normalize BM25 scores to [0, 1]
            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
            if max_bm25 > 0:
                for pid, bm25_score in bm25_scores.items():
                    if pid in scored:
                        scored[pid] += BM25_WEIGHT * (bm25_score / max_bm25)
    else:
        # No query: score by recency (newest first)
        all_dates = []
        for pid, row in candidates.items():
            try:
                dt = datetime.fromisoformat(row["created_at"])
                all_dates.append((pid, dt.timestamp()))
            except (ValueError, TypeError):
                all_dates.append((pid, 0.0))

        if all_dates:
            min_ts = min(ts for _, ts in all_dates)
            max_ts = max(ts for _, ts in all_dates)
            ts_range = max_ts - min_ts if max_ts > min_ts else 1.0
            for pid, ts in all_dates:
                scored[pid] = (ts - min_ts) / ts_range

    # --- Step 4: Build results sorted by score ---

    sorted_ids = sorted(scored.keys(), key=lambda pid: scored[pid], reverse=True)
    sorted_ids = sorted_ids[:limit]

    results: list[SearchResult] = []
    for pid in sorted_ids:
        record = await store.get_photo(pid)
        if record:
            results.append(SearchResult(photo=record, score=scored[pid]))

    return results


async def get_photo_detail(store: PhotoStore, photo_id: str) -> PhotoRecord | None:
    """Get full photo details by ID.

    Args:
        store: The PhotoStore instance.
        photo_id: The photo ID to retrieve.

    Returns:
        PhotoRecord with all metadata, tags, and people, or None.
    """
    return await store.get_photo(photo_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _filter_by_tags(
    db: Any, photo_ids: set[str], tags: list[str]
) -> set[str]:
    """Filter photo IDs to only those having ALL specified tags (AND logic)."""
    if not tags or not photo_ids:
        return photo_ids

    matched = set(photo_ids)
    for tag_name in tags:
        tag_name = tag_name.strip().lower()
        ids_with_tag: set[str] = set()
        placeholders = ",".join("?" * len(matched))
        async with db.execute(
            f"""SELECT pt.photo_id FROM photo_tags pt
                JOIN tags t ON pt.tag_id = t.id
                WHERE t.name = ? AND pt.photo_id IN ({placeholders})""",
            (tag_name, *matched),
        ) as cursor:
            async for row in cursor:
                ids_with_tag.add(row["photo_id"])
        matched &= ids_with_tag
        if not matched:
            break

    return matched


async def _filter_by_people(
    db: Any, photo_ids: set[str], people: list[str]
) -> set[str]:
    """Filter photo IDs to only those containing ALL specified people (AND logic)."""
    if not people or not photo_ids:
        return photo_ids

    matched = set(photo_ids)
    for person_label in people:
        person_label = person_label.strip()
        ids_with_person: set[str] = set()
        placeholders = ",".join("?" * len(matched))
        # Match by label (case-insensitive) or person_id
        async with db.execute(
            f"""SELECT photo_id FROM photo_people
                WHERE (label = ? COLLATE NOCASE OR person_id = ?)
                AND photo_id IN ({placeholders})""",
            (person_label, person_label, *matched),
        ) as cursor:
            async for row in cursor:
                ids_with_person.add(row["photo_id"])
        matched &= ids_with_person
        if not matched:
            break

    return matched


async def _bm25_scores(
    db: Any, query: str, candidate_ids: set[str]
) -> dict[str, float]:
    """Get BM25 relevance scores for a query against candidate photos.

    Uses the photos_fts FTS5 table. Maps FTS rowids back to photo IDs.
    """
    if not query or not candidate_ids:
        return {}

    scores: dict[str, float] = {}

    try:
        # FTS5 MATCH query — rank gives negative BM25 (lower = more relevant)
        async with db.execute(
            """SELECT p.id, -rank as bm25_score
               FROM photos_fts fts
               JOIN photos p ON p.rowid = fts.rowid
               WHERE photos_fts MATCH ?
               ORDER BY rank""",
            (query,),
        ) as cursor:
            async for row in cursor:
                pid = row["id"]
                if pid in candidate_ids:
                    scores[pid] = row["bm25_score"]
    except Exception as e:
        # FTS MATCH can fail on malformed queries; log and return empty
        logger.warning("FTS5 search failed for query '%s': %s", query, e)

    return scores
