"""Unified search backend for the memory system.

Provides hybrid vector + BM25 retrieval used by:
- search_memory tool (direct call during conversation)
- Memory injection at conversation start (retrieval.py)
- SDK boxbot_sdk.memory.search() (via execute_script)

Scoring: 0.6 * vector_cosine + 0.4 * BM25_normalized (configurable).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from boxbot.memory.embeddings import EMBEDDING_DIM, cosine_similarity, embed
from boxbot.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Hybrid search weights
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4

# Default result limits
DEFAULT_MEMORY_CANDIDATES = 30
DEFAULT_CONVERSATION_CANDIDATES = 10
DEFAULT_MEMORY_RESULTS = 10
DEFAULT_CONVERSATION_RESULTS = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SearchCandidate:
    """A candidate from hybrid search, before reranking."""

    id: str
    source: str  # "memory" or "conversation"
    type: str  # memory type or "conversation"
    person: str | None
    content: str
    summary: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0
    # Extra fields carried through for result formatting
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A final search result after reranking."""

    id: str
    source: str
    type: str
    person: str | None
    title: str
    summary: str
    relevance: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


async def hybrid_search(
    store: MemoryStore,
    query: str,
    *,
    types: list[str] | None = None,
    person: str | None = None,
    include_conversations: bool = True,
    include_archived: bool = False,
    memory_limit: int = DEFAULT_MEMORY_CANDIDATES,
    conversation_limit: int = DEFAULT_CONVERSATION_CANDIDATES,
) -> list[SearchCandidate]:
    """Perform hybrid vector + BM25 search across memories and conversations.

    Args:
        store: The MemoryStore instance.
        query: Search query text.
        types: Optional list of memory types to filter.
        person: Optional person name to filter/boost.
        include_conversations: Whether to search conversations too.
        include_archived: Whether to include archived memories.
        memory_limit: Max memory candidates to return.
        conversation_limit: Max conversation candidates to return.

    Returns:
        List of SearchCandidate sorted by combined_score descending.
    """
    query_embedding = embed(query)

    # --- Vector search on memories ---
    memory_candidates = await _vector_search_memories(
        store, query_embedding,
        types=types,
        person=person,
        include_archived=include_archived,
        limit=memory_limit * 2,  # Over-fetch for merging with BM25
    )

    # --- BM25 search on memories ---
    bm25_candidates = await _bm25_search_memories(
        store, query,
        types=types,
        person=person,
        include_archived=include_archived,
        limit=memory_limit * 2,
    )

    # --- Merge memory candidates ---
    merged_memories = _merge_candidates(
        memory_candidates, bm25_candidates, limit=memory_limit
    )

    all_candidates = merged_memories

    # --- Conversation search ---
    if include_conversations:
        conv_vector = await _vector_search_conversations(
            store, query_embedding, limit=conversation_limit * 2
        )
        conv_bm25 = await _bm25_search_conversations(
            store, query, limit=conversation_limit * 2
        )
        merged_convs = _merge_candidates(
            conv_vector, conv_bm25, limit=conversation_limit
        )
        all_candidates.extend(merged_convs)

    # Sort by combined score
    all_candidates.sort(key=lambda c: c.combined_score, reverse=True)
    return all_candidates


async def _vector_search_memories(
    store: MemoryStore,
    query_embedding: np.ndarray,
    *,
    types: list[str] | None,
    person: str | None,
    include_archived: bool,
    limit: int,
) -> list[SearchCandidate]:
    """Search memories by vector cosine similarity."""
    conditions = []
    params: list = []

    # Status filter
    if include_archived:
        conditions.append("status IN ('active', 'archived')")
    else:
        conditions.append("status = 'active'")

    if types:
        placeholders = ",".join("?" * len(types))
        conditions.append(f"type IN ({placeholders})")
        params.extend(types)

    if person:
        conditions.append("(person = ? OR people LIKE ?)")
        params.extend([person, f'%"{person}"%'])

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    cursor = await store.db.execute(
        f"SELECT * FROM memories {where}", params
    )
    rows = await cursor.fetchall()

    candidates = []
    for row in rows:
        embedding = row["embedding"]
        if embedding is None:
            continue
        mem_embedding = np.frombuffer(embedding, dtype=np.float32).copy()
        score = cosine_similarity(query_embedding, mem_embedding)
        # Clamp to [0, 1] for scoring
        score = max(0.0, score)

        candidates.append(SearchCandidate(
            id=row["id"],
            source="memory",
            type=row["type"],
            person=row["person"],
            content=row["content"],
            summary=row["summary"],
            vector_score=score,
            metadata={
                "people": json.loads(row["people"]),
                "tags": json.loads(row["tags"]),
                "created_at": row["created_at"],
                "last_relevant_at": row["last_relevant_at"],
                "status": row["status"],
            },
        ))

    # Sort by vector score and return top N
    candidates.sort(key=lambda c: c.vector_score, reverse=True)
    return candidates[:limit]


async def _bm25_search_memories(
    store: MemoryStore,
    query: str,
    *,
    types: list[str] | None,
    person: str | None,
    include_archived: bool,
    limit: int,
) -> list[SearchCandidate]:
    """Search memories by FTS5 BM25 scoring."""
    # Build FTS query — escape special characters
    fts_query = _escape_fts_query(query)
    if not fts_query.strip():
        return []

    # Join with memories table for status/type filtering
    conditions = ["memories_fts MATCH ?"]
    params: list = [fts_query]

    if include_archived:
        conditions.append("m.status IN ('active', 'archived')")
    else:
        conditions.append("m.status = 'active'")

    if types:
        placeholders = ",".join("?" * len(types))
        conditions.append(f"m.type IN ({placeholders})")
        params.extend(types)

    if person:
        conditions.append("(m.person = ? OR m.people LIKE ?)")
        params.extend([person, f'%"{person}"%'])

    where = " AND ".join(conditions)

    try:
        cursor = await store.db.execute(
            f"""SELECT m.*, bm25(memories_fts) AS rank
                FROM memories_fts
                JOIN memories m ON m.rowid = memories_fts.rowid
                WHERE {where}
                ORDER BY rank
                LIMIT ?""",
            params + [limit],
        )
        rows = await cursor.fetchall()
    except Exception as e:
        logger.warning("FTS5 search failed: %s", e)
        return []

    candidates = []
    for row in rows:
        # bm25() returns negative scores (lower is better)
        # Convert to positive score where higher is better
        raw_rank = row["rank"]
        score = -raw_rank if raw_rank < 0 else raw_rank

        candidates.append(SearchCandidate(
            id=row["id"],
            source="memory",
            type=row["type"],
            person=row["person"],
            content=row["content"],
            summary=row["summary"],
            bm25_score=score,
            metadata={
                "people": json.loads(row["people"]),
                "tags": json.loads(row["tags"]),
                "created_at": row["created_at"],
                "last_relevant_at": row["last_relevant_at"],
                "status": row["status"],
            },
        ))

    return candidates


async def _vector_search_conversations(
    store: MemoryStore,
    query_embedding: np.ndarray,
    *,
    limit: int,
) -> list[SearchCandidate]:
    """Search conversations by vector cosine similarity."""
    cursor = await store.db.execute("SELECT * FROM conversations")
    rows = await cursor.fetchall()

    candidates = []
    for row in rows:
        embedding = row["embedding"]
        if embedding is None:
            continue
        conv_embedding = np.frombuffer(embedding, dtype=np.float32).copy()
        score = cosine_similarity(query_embedding, conv_embedding)
        score = max(0.0, score)

        candidates.append(SearchCandidate(
            id=row["id"],
            source="conversation",
            type="conversation",
            person=None,
            content=row["summary"],
            summary=row["summary"],
            vector_score=score,
            metadata={
                "channel": row["channel"],
                "participants": json.loads(row["participants"]),
                "topics": json.loads(row["topics"]),
                "started_at": row["started_at"],
            },
        ))

    candidates.sort(key=lambda c: c.vector_score, reverse=True)
    return candidates[:limit]


async def _bm25_search_conversations(
    store: MemoryStore,
    query: str,
    *,
    limit: int,
) -> list[SearchCandidate]:
    """Search conversations by FTS5 BM25 scoring."""
    fts_query = _escape_fts_query(query)
    if not fts_query.strip():
        return []

    try:
        cursor = await store.db.execute(
            """SELECT c.*, bm25(conversations_fts) AS rank
               FROM conversations_fts
               JOIN conversations c ON c.rowid = conversations_fts.rowid
               WHERE conversations_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (fts_query, limit),
        )
        rows = await cursor.fetchall()
    except Exception as e:
        logger.warning("Conversation FTS5 search failed: %s", e)
        return []

    candidates = []
    for row in rows:
        raw_rank = row["rank"]
        score = -raw_rank if raw_rank < 0 else raw_rank

        candidates.append(SearchCandidate(
            id=row["id"],
            source="conversation",
            type="conversation",
            person=None,
            content=row["summary"],
            summary=row["summary"],
            bm25_score=score,
            metadata={
                "channel": row["channel"],
                "participants": json.loads(row["participants"]),
                "topics": json.loads(row["topics"]),
                "started_at": row["started_at"],
            },
        ))

    return candidates


def _merge_candidates(
    vector_candidates: list[SearchCandidate],
    bm25_candidates: list[SearchCandidate],
    *,
    limit: int,
) -> list[SearchCandidate]:
    """Merge vector and BM25 candidates with normalized score combination.

    Normalizes each score type to [0, 1] range before combining with weights.
    """
    # Build lookup by ID
    by_id: dict[str, SearchCandidate] = {}

    for c in vector_candidates:
        by_id[c.id] = c

    for c in bm25_candidates:
        if c.id in by_id:
            by_id[c.id].bm25_score = c.bm25_score
        else:
            by_id[c.id] = c

    candidates = list(by_id.values())

    # Normalize scores
    max_vector = max((c.vector_score for c in candidates), default=1.0) or 1.0
    max_bm25 = max((c.bm25_score for c in candidates), default=1.0) or 1.0

    for c in candidates:
        norm_vector = c.vector_score / max_vector
        norm_bm25 = c.bm25_score / max_bm25
        c.combined_score = VECTOR_WEIGHT * norm_vector + BM25_WEIGHT * norm_bm25

    candidates.sort(key=lambda c: c.combined_score, reverse=True)
    return candidates[:limit]


def _escape_fts_query(query: str) -> str:
    """Escape a query string for FTS5 MATCH syntax.

    Wraps each word in double quotes to avoid FTS5 syntax errors from
    special characters.
    """
    words = query.split()
    escaped = []
    for word in words:
        # Remove characters that break FTS5 syntax
        clean = "".join(c for c in word if c.isalnum() or c in "-_'")
        if clean:
            escaped.append(f'"{clean}"')
    return " ".join(escaped)


# ---------------------------------------------------------------------------
# Reranking stub
# ---------------------------------------------------------------------------


async def rerank_stub(
    candidates: list[SearchCandidate],
    query: str,
) -> list[SearchResult]:
    """Placeholder for small model reranking.

    In production, this sends batches of ~6 candidates to the small model
    (5-6 parallel calls) for relevance scoring, title generation, and
    summary generation. For now, it converts candidates directly to results.

    Args:
        candidates: Candidates from hybrid search.
        query: The original search query.

    Returns:
        Ranked SearchResult list.
    """
    results = []
    for c in candidates:
        results.append(SearchResult(
            id=c.id,
            source=c.source,
            type=c.type,
            person=c.person,
            title=c.summary[:80],
            summary=c.summary,
            relevance=f"Matched query (score: {c.combined_score:.2f})",
            metadata=c.metadata,
        ))
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def search_memories(
    store: MemoryStore,
    *,
    mode: str,
    query: str | None = None,
    memory_id: str | None = None,
    conversation_id: str | None = None,
    types: list[str] | None = None,
    person: str | None = None,
    include_conversations: bool = True,
    include_archived: bool = False,
) -> dict:
    """Main search entry point for the tool, SDK, and injection system.

    Args:
        store: The MemoryStore instance.
        mode: One of "lookup", "summary", "get", "transcript".
        query: Required for lookup, summary, and transcript-search modes.
        memory_id: Required for get mode.
        conversation_id: Optional for transcript mode (returns full
            transcript by id; otherwise the query is used to substring-
            search across all retained transcripts).
        types: Optional memory type filter.
        person: Optional person name filter.
        include_conversations: Include conversation log in results.
        include_archived: Include archived memories.

    Returns:
        Dict with mode-specific results:
        - lookup: {"facts": [...], "conversations": [...]}
        - summary: {"answer": str, "sources": [...]}
        - get: full memory record dict
        - transcript: {"conversation_id", "started_at", "transcript"} OR
                      {"matches": [{"conversation_id", "started_at", "snippet"}]}
    """
    if mode == "get":
        if not memory_id:
            raise ValueError("memory_id is required for get mode")
        return await _search_get(store, memory_id)

    if mode == "transcript":
        return await _search_transcript(
            store, query=query, conversation_id=conversation_id,
        )

    if not query:
        raise ValueError("query is required for lookup and summary modes")

    if mode == "lookup":
        return await _search_lookup(
            store, query,
            types=types,
            person=person,
            include_conversations=include_conversations,
            include_archived=include_archived,
        )
    elif mode == "summary":
        return await _search_summary(
            store, query,
            types=types,
            person=person,
            include_conversations=include_conversations,
            include_archived=include_archived,
        )
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Must be lookup, summary, get, or transcript."
        )


async def _search_transcript(
    store: MemoryStore,
    *,
    query: str | None,
    conversation_id: str | None,
) -> dict:
    """Transcript mode: retrieve raw conversation text within retention.

    Two sub-modes:
    - If ``conversation_id`` is given, return the full transcript for
      that conversation (or an error if purged / not found).
    - Otherwise, ``query`` is required and we substring-search all
      retained transcripts (default 14 days), returning up to 5 hits
      with ~300-char snippets around the first match.
    """
    if conversation_id:
        text = await store.get_transcript(conversation_id)
        if text is None:
            return {
                "error": (
                    f"Transcript for {conversation_id} not available "
                    f"(either never recorded or purged after retention window)"
                ),
            }
        # Pull metadata for context.
        row = await store.get_pending_extraction(conversation_id)
        return {
            "conversation_id": conversation_id,
            "started_at": row.started_at if row else None,
            "channel": row.channel if row else None,
            "participants": row.participants if row else [],
            "transcript": text,
        }

    if not query:
        raise ValueError(
            "transcript mode needs either conversation_id or query"
        )

    matches = await store.search_transcripts(query, limit=5)
    return {
        "matches": [
            {
                "conversation_id": cid,
                "started_at": started_at,
                "snippet": snippet,
            }
            for (cid, started_at, snippet) in matches
        ],
    }


async def _search_get(store: MemoryStore, memory_id: str) -> dict:
    """Get mode: retrieve full memory record by ID."""
    memory = await store.get_memory(memory_id)
    if memory is None:
        return {"error": f"Memory {memory_id} not found"}

    return {
        "id": memory.id,
        "type": memory.type,
        "person": memory.person,
        "content": memory.content,
        "summary": memory.summary,
        "tags": memory.tags,
        "people": memory.people,
        "source_conversation": memory.source_conversation,
        "created_at": memory.created_at,
        "last_relevant_at": memory.last_relevant_at,
        "status": memory.status,
    }


async def _search_lookup(
    store: MemoryStore,
    query: str,
    *,
    types: list[str] | None,
    person: str | None,
    include_conversations: bool,
    include_archived: bool,
) -> dict:
    """Lookup mode: return ranked facts and conversations."""
    candidates = await hybrid_search(
        store, query,
        types=types,
        person=person,
        include_conversations=include_conversations,
        include_archived=include_archived,
    )

    # Rerank (stub for now)
    results = await rerank_stub(candidates, query)

    # Update relevance timestamps for results that pass the filter
    for r in results:
        if r.source == "memory":
            await store.update_memory_relevance(r.id)
            # Auto-unarchive if archived memory surfaced
            if r.metadata.get("status") == "archived":
                await store.unarchive_memory(r.id)

    # Split into facts and conversations
    facts = [
        {
            "id": r.id,
            "type": r.type,
            "person": r.person,
            "title": r.title,
            "summary": r.summary,
            "relevance": r.relevance,
        }
        for r in results
        if r.source == "memory"
    ][:DEFAULT_MEMORY_RESULTS]

    conversations = [
        {
            "id": r.id,
            "date": r.metadata.get("started_at", "")[:10],
            "participants": r.metadata.get("participants", []),
            "title": r.title,
            "summary": r.summary,
            "relevance": r.relevance,
        }
        for r in results
        if r.source == "conversation"
    ][:DEFAULT_CONVERSATION_RESULTS]

    return {"facts": facts, "conversations": conversations}


async def _search_summary(
    store: MemoryStore,
    query: str,
    *,
    types: list[str] | None,
    person: str | None,
    include_conversations: bool,
    include_archived: bool,
) -> dict:
    """Summary mode: synthesize an answer from relevant memories.

    In production, this runs parallel small model calls for filtering,
    then a synthesis call. For now, it concatenates relevant summaries.
    """
    candidates = await hybrid_search(
        store, query,
        types=types,
        person=person,
        include_conversations=include_conversations,
        include_archived=include_archived,
    )

    # Update relevance timestamps
    source_ids = []
    for c in candidates[:15]:
        source_ids.append(c.id)
        if c.source == "memory":
            await store.update_memory_relevance(c.id)
            if c.metadata.get("status") == "archived":
                await store.unarchive_memory(c.id)

    # Stub synthesis: concatenate top summaries
    # In production, the small model synthesizes a natural language answer
    summaries = [c.summary for c in candidates[:10] if c.summary]
    if summaries:
        answer = " | ".join(summaries)
    else:
        answer = "No relevant memories found."

    return {"answer": answer, "sources": source_ids[:10]}
