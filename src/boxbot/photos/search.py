"""Shared photo search backend.

One backend for both entry points — the ``search_photos`` tool and
``bb.photos.search`` — with the same defaults.

Scoring (query given), per photo:
  - vector: ``VECTOR_WEIGHT`` x cosine(query, description), negatives
    clipped to 0 (MiniLM embeddings)
  - keyword: ``BM25_WEIGHT`` x FTS5 BM25 over description + tags +
    people; any literal term hit earns at least half the weight
  - person: ``PERSON_WEIGHT`` x how well the photo matches each person
    name (``PERSON_MATCH``).

Person names are a HARD filter. A name comes from ``people=[...]`` or
from a known name (roster entry or existing label) found in the query —
the two are treated identically. A photo is returned only if it matches
every name: by people label, or by the name appearing as a whole name
in its tags or description. The other query words only rank within that
set. Names match whole: "Ann" is not "Ann-Marie" (``people.name_pattern``).
An ``unsure`` suggestion never matches, by label or by text.

Results under the relevance floor (``photos.search_min_score``) are
dropped, so an unrelated query returns nothing instead of everything.
No query (or nothing but filler: "show me photos") and no people =
newest first, no floor.

Other hard filters: tags (AND), date range, source, slideshow.

An empty result is returned empty, with a ``hint`` saying which filter
to drop — never silently broadened, so a caller that shows results
straight on screen cannot show the wrong photos.

Cost: one search = three row queries (photos, tags, people), one names
query and one FTS query, whatever the library size; the hint is
computed from the same loaded set.

Usage:
    from boxbot.photos.search import search_photos

    out = await search_photos(query="Zara at the beach")
    out = await search_photos(mode="get", photo_id="photo_abc123def456")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from boxbot.memory.embeddings import EMBEDDING_DIM, cosine_similarity, embed
from boxbot.photos.people import name_pattern, text_names
from boxbot.photos.store import PhotoRecord, PhotoStore

logger = logging.getLogger(__name__)

# Scoring weights
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
PERSON_WEIGHT = 1.0

# How strongly each kind of evidence says "this person is in the photo".
PERSON_MATCH = {
    "confirmed": 1.0,   # a person confirmed it, as reported by the agent
    "captioned": 1.0,   # the caption passed at ingest names them
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

    Punctuation is discarded. A query of nothing but filler ("show me
    photos") has no terms — the caller treats it as no query.
    """
    if not query:
        return []
    out: list[str] = []
    for token in _TOKEN_RE.findall(query):
        token = token.lower()
        if token not in _STOPWORDS and token not in out:
            out.append(token)
    return out


def fts_match_expr(terms: list[str]) -> str:
    """FTS5 MATCH expression: quoted terms joined with OR.

    Quoting makes every term a literal string, so FTS operators and
    punctuation in user text can never produce a syntax error.
    """
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def split_query_names(
    query: str | None, known: dict[str, str]
) -> tuple[list[str], str]:
    """Pull known person names out of a query.

    *known* is ``{casefolded: stored spelling}``. Returns the names
    found (stored spelling, in query order) and the query with those
    names removed. Longest names are taken first, so "Mary Jane" is one
    name even when "Mary" is also known.
    """
    rest = query or ""
    found: list[tuple[int, str]] = []
    for key in sorted(known, key=len, reverse=True):
        pattern = re.compile(
            name_pattern(known[key]).pattern + r"(?:['’]s(?!\w))?",
            re.IGNORECASE,
        )
        match = pattern.search(rest)
        if match is None:
            continue
        found.append((match.start(), known[key]))
        # Same-length blank keeps the positions of later matches stable.
        rest = pattern.sub(lambda m: " " * len(m.group(0)), rest)
    return [name for _, name in sorted(found)], rest


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
    # A weak guess must not become a match through the text fields.
    if any(
        str(s.get("label") or "").casefold() == wanted
        for s in record.suggestions
    ):
        return 0.0
    if any(text_names(tag, name) for tag in record.tags):
        return PERSON_MATCH["text"]
    if text_names(record.description, name):
        return PERSON_MATCH["text"]
    return 0.0


def _merge_names(*groups: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for name in group:
            key = name.casefold()
            if key not in seen:
                seen.add(key)
                out.append(name)
    return out


@dataclass
class _Prepared:
    """Everything one search needs, loaded once.

    Holds the library (minus soft-deleted unless asked for) and the
    query-dependent signals for every photo. Filters are then applied in
    Python by ``_rank``, so the empty-result hint can try relaxed
    filters without touching the database again.
    """

    records: dict[str, PhotoRecord]
    # Query with person names and filler accounted for; None = no query.
    query: str | None = None
    query_names: list[str] = field(default_factory=list)
    has_query_embedding: bool = False
    similarity: dict[str, float] = field(default_factory=dict)
    # Raw BM25 per photo (higher = better) for the non-name query terms.
    bm25: dict[str, float] = field(default_factory=dict)


async def _prepare(
    store: PhotoStore, query: str | None, *, include_deleted: bool = False
) -> _Prepared:
    records, embeddings = await store.load_records(
        include_deleted=include_deleted
    )
    query = (query or "").strip() or None
    prep = _Prepared(records=records)
    if not query or not records:
        return prep

    known = await store.known_person_names()
    prep.query_names, rest = split_query_names(query, known)
    terms = query_terms(rest)
    if not terms and not prep.query_names:
        # Nothing but filler ("show me photos"): same as no query.
        return prep
    prep.query = query

    # None query embedding = no embedding model installed; BM25 then
    # carries the full text weight.
    query_embedding = embed(query)
    if query_embedding is not None:
        prep.has_query_embedding = True
        for pid, blob in embeddings.items():
            if not blob:
                continue
            photo_embedding = np.frombuffer(blob, dtype=np.float32)
            if len(photo_embedding) == EMBEDDING_DIM:
                prep.similarity[pid] = float(
                    cosine_similarity(query_embedding, photo_embedding)
                )

    if getattr(store, "fts_ready", True):
        prep.bm25 = await _bm25_scores(
            store._ensure_db(), terms, set(records)
        )
    return prep


def _passes_filters(
    record: PhotoRecord,
    *,
    tags: list[str] | None,
    date_from: str | None,
    date_to: str | None,
    source: str | None,
    in_slideshow: bool | None,
) -> bool:
    if source and record.source != source:
        return False
    if in_slideshow is not None and record.in_slideshow != bool(in_slideshow):
        return False
    if date_from and not (record.created_at or "") >= str(date_from):
        return False
    if date_to and not (record.created_at or "") <= str(date_to):
        return False
    if tags:
        have = {t.casefold() for t in record.tags}
        if any(str(t).strip().casefold() not in have for t in tags):
            return False
    return True


def _rank(
    prep: _Prepared,
    *,
    names: list[str],
    tags: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    source: str | None = None,
    in_slideshow: bool | None = None,
    limit: int = DEFAULT_LIMIT,
    min_score: float | None = None,
) -> list[SearchResult]:
    """Filter, score and order the prepared library. No I/O."""
    records = {
        pid: r for pid, r in prep.records.items()
        if _passes_filters(
            r, tags=tags, date_from=date_from, date_to=date_to,
            source=source, in_slideshow=in_slideshow,
        )
    }

    # Person names: a photo must match every one of them.
    scored: dict[str, float] = {}
    if names:
        for pid, record in records.items():
            matches = [_person_match(record, n) for n in names]
            if all(m > 0 for m in matches):
                scored[pid] = PERSON_WEIGHT * sum(matches) / len(matches)
        records = {pid: records[pid] for pid in scored}
    else:
        scored = {pid: 0.0 for pid in records}

    if not records:
        return []

    if prep.query:
        for pid in records:
            sim = prep.similarity.get(pid)
            if sim is not None:
                scored[pid] += VECTOR_WEIGHT * max(0.0, sim)

        bm25_weight = BM25_WEIGHT if prep.has_query_embedding else 1.0
        hits = {pid: sc for pid, sc in prep.bm25.items() if pid in records}
        max_bm25 = max(hits.values(), default=0.0)
        for pid, bm25_score in hits.items():
            norm = bm25_score / max_bm25 if max_bm25 > 0 else 1.0
            # Any literal hit is worth half the weight; rank adds the rest.
            scored[pid] += bm25_weight * (0.5 + 0.5 * norm)

    if prep.query or names:
        floor = _min_score() if min_score is None else min_score
        scored = {pid: sc for pid, sc in scored.items() if sc >= floor}

    # Order — score, then newest first.
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


def _clean_people(people: list[str] | None) -> list[str]:
    return [p.strip() for p in people or [] if p and p.strip()]


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
        query: Natural language query. None / filler only = newest first.
            A known person name in it filters exactly like ``people``.
        tags: Hard filter — photo must carry all listed tags.
        people: Person names. Hard filter — the photo must match every
            name (people label, or whole name in tags/description).
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
    prep = await _prepare(store, query, include_deleted=include_deleted)
    return _rank(
        prep,
        names=_merge_names(_clean_people(people), prep.query_names),
        tags=tags, date_from=date_from, date_to=date_to, source=source,
        in_slideshow=in_slideshow, limit=limit, min_score=min_score,
    )


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


def _empty_hint(prep: _Prepared, filters: dict[str, Any]) -> str:
    """Explain an empty search: which single filter, dropped, finds photos.

    Nothing is broadened on the caller's behalf — a relaxed result set
    presented as the answer is how the wrong photo ends up on screen.
    Computed from the already-loaded library; no further queries.
    """
    explicit = _clean_people(filters.get("people"))
    names = _merge_names(explicit, prep.query_names)

    def count(relaxed: dict[str, Any], relaxed_names: list[str]) -> int:
        return len(_rank(
            prep, names=relaxed_names, tags=relaxed.get("tags"),
            date_from=relaxed.get("date_from"),
            date_to=relaxed.get("date_to"), source=relaxed.get("source"),
            in_slideshow=relaxed.get("in_slideshow"), limit=MAX_LIMIT,
        ))

    # Several names and no shared photo: say so before anything else.
    if len(names) > 1:
        each = [(name, count(filters, [name])) for name in names]
        if any(found for _, found in each):
            return (
                "No photo has all of " + ", ".join(names) + " together ("
                + "; ".join(f"{name} alone: {found}" for name, found in each)
                + "). Search one name at a time and say so — do not show "
                "one person's photos as the group."
            )
    suggestions: list[str] = []
    for key in ("tags", "people", "date", "source", "in_slideshow"):
        relaxed = dict(filters)
        relaxed_names = names
        if key == "date":
            if not (filters.get("date_from") or filters.get("date_to")):
                continue
            relaxed["date_from"] = relaxed["date_to"] = None
        elif key == "people":
            if not explicit:
                continue
            relaxed_names = list(prep.query_names)
        else:
            if filters.get(key) in (None, [], ""):
                continue
            relaxed[key] = None
        found = count(relaxed, relaxed_names)
        if found:
            label = "date range" if key == "date" else key
            suggestions.append(f"without {label}: {found}")
    if suggestions:
        return (
            "No match with these filters. " + "; ".join(suggestions)
            + ". Retry with that filter dropped and say what you dropped."
        )
    if names:
        return (
            "No photos of " + ", ".join(names) + " match. Do not show "
            "other people's or unrelated photos instead."
        )
    if prep.query:
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
            "tags": tags,
            "date_from": date_from,
            "date_to": date_to,
            "source": source,
            "in_slideshow": in_slideshow,
        }
        prep = await _prepare(
            store, query, include_deleted=include_deleted
        )
        results = _rank(
            prep,
            names=_merge_names(_clean_people(people), prep.query_names),
            **filters, limit=limit,
        )
        out: dict[str, Any] = {
            "results": [
                {**_record_to_row(r.photo), "score": round(r.score, 2)}
                for r in results
            ]
        }
        if not results:
            out["hint"] = _empty_hint(prep, {**filters, "people": people})
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
