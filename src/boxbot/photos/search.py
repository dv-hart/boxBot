"""Shared photo search backend.

One backend for both entry points — the ``search_photos`` tool and
``bb.photos.search`` — with the same defaults.

Scoring (query given), per photo:
  - vector: ``VECTOR_WEIGHT`` x cosine(query, description), negatives
    clipped to 0 (MiniLM embeddings)
  - keyword: ``BM25_WEIGHT`` x FTS5 BM25 over description + tags +
    people; any literal term hit earns at least half the weight
  - person: ``PERSON_WEIGHT`` x how well the photo matches each person
    name — from ``people=[...]`` or a household name found in the query.
    A name is a soft match across people labels, tags and description
    (``PERSON_MATCH``), not a hard filter on ``photo_people``.

Results under the relevance floor (``photos.search_min_score``) are
dropped, so an unrelated query returns nothing instead of everything.
No query and no people = newest first, no floor.

Hard filters: tags (AND), date range, source, slideshow. With
``people=[...]`` a photo must also mention at least one of the names
somewhere (label, tag or description).

An empty result is returned empty, with a ``hint`` saying which filter
to drop — never silently broadened, so a caller that shows results
straight on screen cannot show the wrong photos.

Usage:
    from boxbot.photos.search import search_photos

    out = await search_photos(query="Zara at the beach")
    out = await search_photos(mode="get", photo_id="photo_abc123def456")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from boxbot.memory.embeddings import EMBEDDING_DIM, cosine_similarity, embed
from boxbot.photos.store import PhotoRecord, PhotoStore

logger = logging.getLogger(__name__)

# Scoring weights
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
PERSON_WEIGHT = 1.0

# How strongly each kind of evidence says "this person is in the photo".
PERSON_MATCH = {
    "confirmed": 1.0,   # a human said so
    "certain": 1.0,     # the caption named them
    "likely": 0.8,      # tagger: appearance/context vs the roster
    "text": 0.6,        # name appears in tags or description only
}

# Fallback relevance floor when config is unavailable. The live value is
# ``photos.search_min_score``. A literal keyword hit scores >= 0.2 and a
# name match >= 0.6, so both clear it; a vector-only match needs cosine
# >= ~0.33; an unrelated photo (cosine ~0-0.2, no term hit) does not.
DEFAULT_MIN_SCORE = 0.2

# One default for both entry points.
DEFAULT_LIMIT = 10
MAX_LIMIT = 100

# Search rows carry a short description; ``get`` returns the full text.
ROW_DESCRIPTION_CHARS = 160

# Words that say "this is a photo search", not what to search for.
_STOPWORDS = frozenset({
    "a", "an", "and", "any", "all", "at", "for", "from", "in", "me", "my",
    "of", "on", "or", "our", "some", "the", "to", "us", "with",
    "find", "show", "see", "get", "have", "you",
    "photo", "photos", "pic", "pics", "picture", "pictures", "image",
    "images", "shot", "shots",
})
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class SearchResult:
    """A single photo search result with relevance score."""

    photo: PhotoRecord
    score: float = 0.0


def _min_score() -> float:
    try:
        from boxbot.core.config import get_config

        return float(get_config().photos.search_min_score)
    except Exception:
        return DEFAULT_MIN_SCORE


def query_terms(query: str | None) -> list[str]:
    """Lowercased word tokens of a query, minus search-request filler.

    Punctuation is discarded. If nothing but filler remains, the filler
    is kept so the query still means something.
    """
    if not query:
        return []
    tokens = [t.lower() for t in _TOKEN_RE.findall(query)]
    kept = [t for t in tokens if t not in _STOPWORDS]
    out: list[str] = []
    for token in kept or tokens:
        if token not in out:
            out.append(token)
    return out


def fts_match_expr(terms: list[str]) -> str:
    """FTS5 MATCH expression: quoted terms joined with OR.

    Quoting makes every term a literal string, so FTS operators and
    punctuation in user text can never produce a syntax error.
    """
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def _contains_word(text: str, word: str) -> bool:
    return re.search(
        rf"(?<!\w){re.escape(word)}(?!\w)", text, re.IGNORECASE
    ) is not None


def _person_match(record: PhotoRecord, name: str) -> float:
    """How strongly *record* matches person *name* (0 = no evidence)."""
    wanted = name.casefold()
    best = 0.0
    for person in record.people:
        label = str(person.get("label") or "")
        if label.casefold() == wanted or person.get("person_id") == name:
            best = max(
                best, PERSON_MATCH.get(person.get("confidence"), 0.8)
            )
    if best:
        return best
    if any(_contains_word(tag, name) for tag in record.tags):
        return PERSON_MATCH["text"]
    if _contains_word(record.description or "", name):
        return PERSON_MATCH["text"]
    return 0.0


async def _known_names(store: PhotoStore) -> set[str]:
    """Casefolded names the library knows: roster + existing labels."""
    db = store._ensure_db()
    names: set[str] = set()
    async with db.execute("SELECT name FROM people_notes") as cursor:
        async for row in cursor:
            names.add(row["name"].casefold())
    async with db.execute(
        "SELECT DISTINCT label FROM photo_people WHERE confidence != 'unsure'"
    ) as cursor:
        async for row in cursor:
            names.add(row["label"].casefold())
    return names


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
    limit: int = DEFAULT_LIMIT,
    min_score: float | None = None,
) -> list[SearchResult]:
    """Search photos; see the module docstring for the scoring model.

    Args:
        store: The PhotoStore instance to search against.
        query: Natural language query. None = newest first.
        tags: Hard filter — photo must carry all listed tags.
        people: Person names. Soft match across people labels, tags and
            description; a photo mentioning none of them is excluded.
        date_from: ISO 8601 date string — only photos on or after this date.
        date_to: ISO 8601 date string — only photos on or before this date.
        source: Filter by source ("whatsapp", "camera", "upload").
        in_slideshow: Filter by slideshow membership.
        include_deleted: Whether to include soft-deleted photos.
        limit: Maximum number of results to return.
        min_score: Relevance floor; None = ``photos.search_min_score``.

    Returns:
        List of SearchResult ordered by relevance score (highest first).
    """
    db = store._ensure_db()
    query = (query or "").strip() or None
    explicit_people = [p.strip() for p in people or [] if p and p.strip()]

    # --- Step 1: Candidates passing the hard filters ---

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

    rows: dict[str, Any] = {}
    async with db.execute(
        f"SELECT * FROM photos p WHERE {where_sql}", params
    ) as cursor:
        async for row in cursor:
            rows[row["id"]] = row

    if tags and rows:
        tag_matched_ids = await _filter_by_tags(db, set(rows.keys()), tags)
        rows = {pid: r for pid, r in rows.items() if pid in tag_matched_ids}

    if not rows:
        return []

    records: dict[str, PhotoRecord] = {}
    for pid, row in rows.items():
        records[pid] = await store._row_to_record(row)

    # --- Step 2: Person names (explicit + household names in the query) ---

    terms = query_terms(query)
    names = list(explicit_people)
    if terms:
        known = await _known_names(store)
        have = {n.casefold() for n in names}
        for term in terms:
            if term in known and term not in have:
                names.append(term)
                have.add(term)

    person_scores: dict[str, float] = {}
    if names:
        for pid, record in records.items():
            matches = [_person_match(record, n) for n in names]
            person_scores[pid] = sum(matches) / len(matches)

    if explicit_people:
        # Soft across fields, but the photo must mention someone asked for.
        records = {
            pid: r for pid, r in records.items()
            if any(_person_match(r, n) > 0 for n in explicit_people)
        }
        if not records:
            return []

    # --- Step 3: Scoring ---

    scored: dict[str, float] = {
        pid: PERSON_WEIGHT * person_scores.get(pid, 0.0) for pid in records
    }

    if query:
        # None query embedding = no embedding model installed; BM25 then
        # carries the full text weight.
        query_embedding = embed(query)
        bm25_weight = BM25_WEIGHT if query_embedding is not None else 1.0

        if query_embedding is not None:
            for pid in records:
                blob = rows[pid]["embedding"]
                if not blob:
                    continue
                photo_embedding = np.frombuffer(blob, dtype=np.float32)
                if len(photo_embedding) == EMBEDDING_DIM:
                    sim = float(
                        cosine_similarity(query_embedding, photo_embedding)
                    )
                    scored[pid] += VECTOR_WEIGHT * max(0.0, sim)

        bm25_scores = await _bm25_scores(db, terms, set(records.keys()))
        max_bm25 = max(bm25_scores.values(), default=0.0)
        for pid, bm25_score in bm25_scores.items():
            norm = bm25_score / max_bm25 if max_bm25 > 0 else 1.0
            # Any literal hit is worth half the weight; rank adds the rest.
            scored[pid] += bm25_weight * (0.5 + 0.5 * norm)

    relevance_ranked = bool(query or names)
    if relevance_ranked:
        floor = _min_score() if min_score is None else min_score
        scored = {pid: sc for pid, sc in scored.items() if sc >= floor}

    # --- Step 4: Order — score, then newest first ---

    def _ts(pid: str) -> float:
        try:
            return datetime.fromisoformat(records[pid].created_at).timestamp()
        except (ValueError, TypeError):
            return 0.0

    sorted_ids = sorted(
        scored.keys(), key=lambda pid: (scored[pid], _ts(pid)), reverse=True
    )[: max(1, limit)]

    return [
        SearchResult(photo=records[pid], score=scored[pid])
        for pid in sorted_ids
    ]


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
# Shared PhotoStore singleton
# ---------------------------------------------------------------------------

_store_singleton: PhotoStore | None = None


async def get_store() -> PhotoStore:
    """Return a lazily-initialized PhotoStore singleton.

    Both the ``search_photos`` tool and the sandbox ``photos.*`` action
    handlers share one store instance. ``initialize()`` is called on
    first access (creates the SQLite schema if needed).
    """
    global _store_singleton
    if _store_singleton is None:
        store = PhotoStore()
        await store.initialize()
        _store_singleton = store
    return _store_singleton


def _people_brief(people: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"name": p.get("label"), "confidence": p.get("confidence")}
        for p in people
    ]


def _record_to_dict(r: PhotoRecord) -> dict[str, Any]:
    """Full record — the ``get`` shape."""
    return {
        "id": r.id,
        "filename": r.filename,
        "source": r.source,
        "sender": r.sender,
        "description": r.description,
        "orientation": r.orientation,
        "width": r.width,
        "height": r.height,
        "file_size": r.file_size,
        "in_slideshow": r.in_slideshow,
        "created_at": r.created_at,
        "deleted_at": r.deleted_at,
        "updated_at": r.updated_at,
        "tags": list(r.tags),
        "people": [
            {
                "name": p.get("label"),
                "confidence": p.get("confidence"),
                "source": p.get("source"),
                "basis": p.get("basis"),
                "person_id": p.get("person_id"),
            }
            for p in r.people
        ],
        "unsure": [
            {"name": p.get("label"), "basis": p.get("basis")}
            for p in r.suggestions
        ],
        "people_confirmed": bool(r.people_confirmed_at),
    }


def _record_to_row(r: PhotoRecord) -> dict[str, Any]:
    """Slim record — the search-result shape. Full detail stays on ``get``."""
    description = r.description or ""
    if len(description) > ROW_DESCRIPTION_CHARS:
        description = description[: ROW_DESCRIPTION_CHARS - 1].rstrip() + "…"
    row: dict[str, Any] = {
        "id": r.id,
        "description": description,
        "tags": list(r.tags),
        "people": _people_brief(r.people),
        "created": (r.created_at or "")[:10],
        "source": r.source,
        "sender": r.sender,
    }
    if r.suggestions:
        row["unsure"] = [p.get("label") for p in r.suggestions]
    return row


async def show_photos(photo_ids: list[str]) -> dict[str, Any]:
    """Put photos on the 7" screen via the ``picture`` display."""
    from boxbot.displays.manager import get_display_manager

    mgr = get_display_manager()
    if mgr is None:
        logger.info("show_photos: no DisplayManager (ids=%s)", photo_ids)
        return {"dispatched": False, "reason": "display manager not running"}
    ok = await mgr.switch("picture", args={"image_ids": list(photo_ids)})
    return {"dispatched": bool(ok)}


async def _empty_hint(
    store: PhotoStore, filters: dict[str, Any]
) -> str:
    """Explain an empty search: which single filter, dropped, finds photos.

    Nothing is broadened on the caller's behalf — a relaxed result set
    presented as the answer is how the wrong photo ends up on screen.
    """
    suggestions: list[str] = []
    for key in ("tags", "people", "date", "source", "in_slideshow"):
        relaxed = dict(filters)
        if key == "date":
            if not (filters.get("date_from") or filters.get("date_to")):
                continue
            relaxed["date_from"] = relaxed["date_to"] = None
        else:
            if filters.get(key) in (None, [], ""):
                continue
            relaxed[key] = None
        count = len(await hybrid_search(store, **relaxed, limit=MAX_LIMIT))
        if count:
            label = "date range" if key == "date" else key
            suggestions.append(f"without {label}: {count}")
    if suggestions:
        return (
            "No match with these filters. " + "; ".join(suggestions)
            + ". Retry with that filter dropped and say what you dropped."
        )
    if filters.get("query") or filters.get("people"):
        return (
            "Nothing in the library matches. Try other words (what is in "
            "the photo, where, when) — do not show unrelated photos."
        )
    return "The library has no photos matching these filters."


async def search_photos(
    *,
    mode: str = "search",
    query: str | None = None,
    photo_id: str | None = None,
    tags: list[str] | None = None,
    people: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    source: str | None = None,
    in_slideshow: bool | None = None,
    include_deleted: bool = False,
    limit: int | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """High-level photo search used by both the tool and SDK handlers.

    Modes:
        - ``search``: ``{"results": [slim row + score, ...]}``; plus
          ``hint`` when empty, and ``shown`` when ``show`` is set (the
          results are put on the screen in the same call)
        - ``get``: the full photo dict, or ``{"error": ...}``
    """
    store = await get_store()

    if mode == "get":
        if not photo_id:
            return {"error": "photo_id is required for 'get' mode"}
        record = await get_photo_detail(store, photo_id)
        if record is None:
            return {"error": f"photo {photo_id} not found"}
        return _record_to_dict(record)

    if mode == "search":
        try:
            limit = DEFAULT_LIMIT if limit is None else int(limit)
        except (TypeError, ValueError):
            return {"error": "limit must be an integer"}
        limit = max(1, min(limit, MAX_LIMIT))
        filters: dict[str, Any] = {
            "query": query,
            "tags": tags,
            "people": people,
            "date_from": date_from,
            "date_to": date_to,
            "source": source,
            "in_slideshow": in_slideshow,
            "include_deleted": include_deleted,
        }
        results = await hybrid_search(store, **filters, limit=limit)
        out: dict[str, Any] = {
            "results": [
                {**_record_to_row(r.photo), "score": round(r.score, 2)}
                for r in results
            ]
        }
        if not results:
            out["hint"] = await _empty_hint(store, filters)
        if show:
            if results:
                shown = await show_photos([r.photo.id for r in results])
                out["shown"] = len(results) if shown["dispatched"] else 0
                if not shown["dispatched"]:
                    out["show_error"] = shown.get(
                        "reason", "picture display refused the switch"
                    )
            else:
                out["shown"] = 0
        return out

    return {"error": f"unknown mode: {mode!r}"}


async def resolve_photo_path(photo_id: str) -> Path | None:
    """Return the absolute filesystem path for a photo, or None if missing."""
    from boxbot.core.config import get_config

    store = await get_store()
    record = await store.get_photo(photo_id)
    if record is None or record.deleted_at:
        return None
    config = get_config()
    storage_path = Path(config.photos.storage_path)
    return (storage_path / record.filename).resolve()


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
        tag_name = str(tag_name).strip().lower()
        ids_with_tag: set[str] = set()
        async with db.execute(
            """SELECT pt.photo_id FROM photo_tags pt
               JOIN tags t ON pt.tag_id = t.id
               WHERE t.name = ?""",
            (tag_name,),
        ) as cursor:
            async for row in cursor:
                ids_with_tag.add(row["photo_id"])
        matched &= ids_with_tag
        if not matched:
            break

    return matched


async def _bm25_scores(
    db: Any, terms: list[str], candidate_ids: set[str]
) -> dict[str, float]:
    """BM25 scores (higher = better) for *terms* over candidate photos.

    Terms are OR-ed and quoted (see ``fts_match_expr``). Column weights:
    description 1.0, tags 1.5, people 2.0.
    """
    if not terms or not candidate_ids:
        return {}

    scores: dict[str, float] = {}
    expr = fts_match_expr(terms)
    try:
        async with db.execute(
            """SELECT photo_id,
                      -bm25(photos_fts, 0.0, 1.0, 1.5, 2.0) AS bm25_score
               FROM photos_fts
               WHERE photos_fts MATCH ?""",
            (expr,),
        ) as cursor:
            async for row in cursor:
                pid = row["photo_id"]
                if pid in candidate_ids:
                    # bm25() can round to 0 on tiny corpora; a hit is a hit.
                    scores[pid] = max(float(row["bm25_score"]), 1e-9)
    except Exception:
        # Quoted terms cannot be malformed, so this is a real fault
        # (missing/corrupt index) — surface it in the log, keep searching
        # on the other signals.
        logger.exception("Photo FTS query failed (expr=%r)", expr)

    return scores
