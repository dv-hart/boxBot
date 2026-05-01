"""Unified search backend for the memory system.

Provides hybrid vector + BM25 retrieval used by:
- search_memory tool (direct call during conversation)
- Memory injection at conversation start (retrieval.py)
- SDK boxbot_sdk.memory.search() (via execute_script)

Scoring: 0.6 * vector_cosine + 0.4 * BM25_normalized (configurable).

## Anthropic client threading

Lookup-mode reranking and summary-mode synthesis call Haiku 4.5. The
search backend doesn't own a client — the agent does. Callers pass the
client explicitly via the ``client`` keyword argument (option (a) in
the roadmap). When ``client`` is ``None`` we lazily build one from
``config.api_keys.anthropic`` (option (c) fallback). If neither is
available, the backend falls back to ``rerank_stub`` for reranking and
a degenerate concat for summary mode — this keeps unit tests + offline
boots functional without forcing every search caller to thread the
client through.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from boxbot.memory.embeddings import EMBEDDING_DIM, cosine_similarity, embed
from boxbot.memory.store import MemoryStore

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)

# Hybrid search weights
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4

# Default result limits
DEFAULT_MEMORY_CANDIDATES = 30
DEFAULT_CONVERSATION_CANDIDATES = 10
DEFAULT_MEMORY_RESULTS = 10
DEFAULT_CONVERSATION_RESULTS = 3

# Reranking / summary tuning. These are deliberately conservative —
# Haiku is fast but it's still a per-call cost on the hot path.
RERANK_BATCH_SIZE = 6
RERANK_MAX_PARALLEL = 6
RERANK_MODEL = "claude-haiku-4-5-20251001"
RERANK_MAX_TOKENS = 2048


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
# Haiku-backed reranking + summary filtering
# ---------------------------------------------------------------------------


# Tool schema for reranking. Forces the model to emit one
# ``rerank_candidates`` call with a per-candidate judgment array. The
# schema is verbatim from docs/plans/memory-roadmap-post-phase-a.md §1.
RERANK_TOOL: dict[str, Any] = {
    "name": "rerank_candidates",
    "description": (
        "Emit relevance judgments for a batch of memory candidates. "
        "Call this exactly once with one judgment per input candidate."
    ),
    "input_schema": {
        "type": "object",
        "required": ["judgments"],
        "properties": {
            "judgments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "candidate_id", "relevant", "title",
                        "summary", "relevance_reason",
                    ],
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "relevant": {"type": "boolean"},
                        "title": {
                            "type": "string",
                            "description": (
                                "Contextual one-line title <= 80 chars."
                            ),
                        },
                        "summary": {
                            "type": "string",
                            "description": "One-sentence summary.",
                        },
                        "relevance_reason": {
                            "type": "string",
                            "description": (
                                "One sentence explaining why this "
                                "candidate is or isn't relevant."
                            ),
                        },
                    },
                },
            },
        },
    },
}


# Filter-mode tool. Used by summary mode to identify the candidates that
# materially contribute to answering a question and to extract the key
# snippet from each. Shares 90% of its prompt with reranking; only the
# per-item output fields differ.
FILTER_TOOL: dict[str, Any] = {
    "name": "filter_candidates",
    "description": (
        "Emit per-candidate relevance + the key snippet that would help "
        "answer the question. Call exactly once."
    ),
    "input_schema": {
        "type": "object",
        "required": ["judgments"],
        "properties": {
            "judgments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["candidate_id", "relevant", "snippet"],
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "relevant": {"type": "boolean"},
                        "snippet": {
                            "type": "string",
                            "description": (
                                "Short verbatim-or-paraphrased snippet "
                                "from the candidate that bears on the "
                                "question. Empty string if not relevant."
                            ),
                        },
                    },
                },
            },
        },
    },
}


# Synthesis tool for summary mode. One final Haiku call after filtering
# returns a natural-language answer with citations.
SYNTHESIZE_TOOL: dict[str, Any] = {
    "name": "synthesize_answer",
    "description": (
        "Emit a natural-language answer to the user's question grounded "
        "in the supplied snippets, plus the source IDs that materially "
        "contributed."
    ),
    "input_schema": {
        "type": "object",
        "required": ["answer", "source_ids"],
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "Natural-language answer. Cite specifics from the "
                    "snippets. If the snippets don't actually answer "
                    "the question, say so plainly."
                ),
            },
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Memory or conversation IDs whose snippets directly "
                    "supported the answer. Order doesn't matter. May be "
                    "empty if no snippets were useful."
                ),
            },
        },
    },
}


# System prompt shared by reranking and summary-filtering. Stable across
# calls so the prefix can be cached. The intent ("rank" vs "summarize")
# is communicated in the user message; the model knows which tool to
# call because we set ``tool_choice`` to that specific tool name.
FILTER_SYSTEM_PROMPT = """\
You are the relevance judge for boxBot's memory search.

You receive:
- A user query (or question).
- A small batch of candidate memories / conversation summaries, each \
with an id, type, optional person, summary, and content excerpt.

For each candidate, decide whether it is RELEVANT to the query:
- Relevant: the candidate's content directly bears on the query and \
would help answer it or surface useful related context.
- Not relevant: the candidate is off-topic, only superficially \
mentions a query keyword, or is too generic to add anything.

Be strict. It's better to drop a borderline candidate than to keep \
noise. The agent will see the kept results in a system prompt block, \
so quality matters more than recall.

When the per-call instruction is "rank":
- For relevant candidates: produce a contextual `title` (<=80 chars), \
a one-sentence `summary`, and a one-sentence `relevance_reason`.
- For irrelevant candidates: still emit the entry, set `relevant: \
false`, and you may keep the title/summary/reason short (they will \
be dropped).

When the per-call instruction is "summarize":
- For relevant candidates: extract the key `snippet` (verbatim or \
tightly paraphrased) that would help answer the question.
- For irrelevant candidates: set `relevant: false` and `snippet: ""`.

Always emit one judgment per input candidate, in the same order, with \
the matching `candidate_id`. Always call the requested tool exactly \
once. Never respond with prose.
"""


SYNTHESIZE_SYSTEM_PROMPT = """\
You are the answer synthesizer for boxBot's memory search.

You receive a question and a list of relevant snippets, each with an \
id. Compose a concise natural-language answer grounded ONLY in the \
snippets. Cite the snippet ids that materially contributed via the \
`source_ids` field.

Rules:
- Do NOT invent facts. If the snippets don't answer the question, say \
that plainly and return an empty `source_ids`.
- Prefer specifics from the snippets over generalities.
- 1-3 sentences typical. Don't pad.
- Always call the `synthesize_answer` tool exactly once.
"""


def _format_candidate_for_prompt(c: SearchCandidate) -> str:
    """Render a candidate as a short block for the user prompt.

    Truncates content aggressively — the model only needs enough
    signal to judge relevance, not the full memory.
    """
    person_label = f" person={c.person}" if c.person else ""
    body = c.content or c.summary or ""
    if len(body) > 600:
        body = body[:600] + "..."
    return (
        f"[id={c.id}] type={c.type}{person_label}\n"
        f"summary: {c.summary}\n"
        f"content: {body}"
    )


def _build_filter_user_message(
    query: str,
    batch: list[SearchCandidate],
    intent: str,
) -> str:
    """Build the per-batch user message for filter / rerank calls."""
    instruction = (
        "rank" if intent == "rank" else "summarize"
    )
    parts = [
        f"Per-call instruction: {instruction}",
        f"Query: {query}",
        "",
        f"Candidates ({len(batch)}):",
    ]
    for c in batch:
        parts.append("---")
        parts.append(_format_candidate_for_prompt(c))
    return "\n".join(parts)


def _maybe_get_anthropic_client(
    client: "anthropic.AsyncAnthropic | None",
) -> "anthropic.AsyncAnthropic | None":
    """Return the supplied client, or lazily build one from config.

    Falls back to ``None`` if no API key is configured. Callers must
    handle the ``None`` case (typically by using the stub path).
    """
    if client is not None:
        return client
    try:
        import anthropic

        from boxbot.core.config import get_config

        cfg = get_config()
        api_key = cfg.api_keys.anthropic
        if not api_key:
            return None
        return anthropic.AsyncAnthropic(api_key=api_key)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Could not lazily build Anthropic client: %s", e)
        return None


def _extract_tool_use(message: Any, tool_name: str) -> dict | None:
    """Pull the tool_use block named ``tool_name`` from a message."""
    content = getattr(message, "content", None) or []
    for block in content:
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type != "tool_use":
            continue
        block_name = getattr(block, "name", None) or (
            block.get("name") if isinstance(block, dict) else None
        )
        if block_name == tool_name:
            payload = getattr(block, "input", None)
            if payload is None and isinstance(block, dict):
                payload = block.get("input")
            if isinstance(payload, dict):
                return payload
    return None


async def _record_haiku_cost(
    store: MemoryStore | None,
    *,
    purpose: str,
    usage: Any,
    metadata: dict | None = None,
) -> None:
    """Compute USD cost from a Haiku call's usage and append cost_log."""
    if store is None or usage is None:
        return
    # Lazy import avoids circular load (extraction.py imports search-adjacent stuff).
    from boxbot.memory.extraction import compute_cost

    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_write = int(
        getattr(usage, "cache_creation_input_tokens", 0) or 0
    )
    cost = compute_cost(
        RERANK_MODEL,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_read_tokens=cache_read,
        cache_write_tokens_1h=cache_write,
        is_batch=False,
    )
    try:
        await store.record_cost(
            purpose=purpose,
            model=RERANK_MODEL,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            is_batch=False,
            cost_usd=cost,
            metadata=metadata,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to record %s cost: %s", purpose, e)


def _chunk(lst: list, n: int) -> list[list]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


async def filter_candidates(
    client: "anthropic.AsyncAnthropic",
    candidates: list[SearchCandidate],
    query: str,
    *,
    intent: str,
    store: MemoryStore | None = None,
    cost_purpose: str = "rerank",
) -> list[dict]:
    """Run parallel Haiku calls to judge candidate relevance.

    Shared helper used by both reranking (``intent="rank"``) and summary
    mode (``intent="summarize"``). The 90% shared prompt + tool schemas
    differ only in the per-candidate output fields.

    Returns a flat list of judgment dicts. Each dict carries (at
    minimum) ``candidate_id`` and ``relevant``. Rank judgments add
    ``title`` / ``summary`` / ``relevance_reason``; summarize judgments
    add ``snippet``.

    Cost is recorded against ``cost_purpose`` (one row per parallel
    call) so the total nightly spend can be attributed correctly.
    """
    if not candidates:
        return []

    if intent not in {"rank", "summarize"}:
        raise ValueError(f"intent must be rank or summarize, got {intent!r}")

    tool = RERANK_TOOL if intent == "rank" else FILTER_TOOL
    tool_name = tool["name"]

    batches = _chunk(candidates, RERANK_BATCH_SIZE)
    # The 5-6 parallel cap is a soft cap — RERANK_BATCH_SIZE * MAX_PARALLEL
    # = 36 candidates per fan-out, comfortably above hybrid_search's 40.
    if len(batches) > RERANK_MAX_PARALLEL:
        # Should be rare. If we ever exceed it, the extras run serially.
        logger.debug(
            "Rerank fan-out %d exceeds parallel cap %d; some batches will queue",
            len(batches), RERANK_MAX_PARALLEL,
        )

    async def _one_call(batch: list[SearchCandidate]) -> list[dict]:
        user_msg = _build_filter_user_message(query, batch, intent)
        try:
            response = await client.messages.create(
                model=RERANK_MODEL,
                max_tokens=RERANK_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": FILTER_SYSTEM_PROMPT,
                        # 5-min TTL: synchronous hot-path, not batch.
                        "cache_control": {
                            "type": "ephemeral", "ttl": "5m",
                        },
                    },
                ],
                tools=[tool],
                tool_choice={"type": "tool", "name": tool_name},
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as e:
            logger.warning(
                "Haiku %s call failed for %d candidates: %s",
                intent, len(batch), e,
            )
            return []

        await _record_haiku_cost(
            store,
            purpose=cost_purpose,
            usage=getattr(response, "usage", None),
            metadata={"intent": intent, "batch_size": len(batch)},
        )

        payload = _extract_tool_use(response, tool_name)
        if not payload:
            logger.warning(
                "Haiku %s response missing %s tool_use", intent, tool_name,
            )
            return []
        judgments = payload.get("judgments") or []
        if not isinstance(judgments, list):
            return []
        return judgments

    results = await asyncio.gather(
        *[_one_call(b) for b in batches],
        return_exceptions=False,
    )
    flat: list[dict] = []
    for batch_result in results:
        flat.extend(batch_result)
    return flat


async def rerank_with_haiku(
    candidates: list[SearchCandidate],
    query: str,
    *,
    client: "anthropic.AsyncAnthropic | None" = None,
    store: MemoryStore | None = None,
) -> list[SearchResult]:
    """Real Haiku-backed reranking. Replaces ``rerank_stub``.

    Pipeline (matches docs/plans/memory-roadmap-post-phase-a.md §1):
      1. Batch input candidates into groups of RERANK_BATCH_SIZE.
      2. Fan out parallel Haiku calls. Each returns per-candidate
         {relevant, title, summary, relevance_reason}.
      3. Drop candidates with relevant=False.
      4. Re-sort survivors by their original combined_score.
      5. Return top DEFAULT_MEMORY_RESULTS facts +
         DEFAULT_CONVERSATION_RESULTS conversations.
    """
    if not candidates:
        return []

    client = _maybe_get_anthropic_client(client)
    if client is None:
        logger.info(
            "No Anthropic client available; falling back to rerank_stub"
        )
        return await rerank_stub(candidates, query)

    judgments = await filter_candidates(
        client, candidates, query,
        intent="rank",
        store=store,
        cost_purpose="rerank",
    )

    # If the model utterly failed (no judgments returned), fall back
    # to the stub so the user still gets *something* back. Logged
    # above; we don't want the tool call to crash on a transient
    # API error.
    if not judgments:
        logger.warning("Reranking returned no judgments; using stub fallback")
        return await rerank_stub(candidates, query)

    # Index judgments by candidate id. Drop irrelevant entries.
    by_id: dict[str, dict] = {}
    for j in judgments:
        cid = j.get("candidate_id")
        if not cid:
            continue
        if not j.get("relevant"):
            continue
        by_id[cid] = j

    # Build SearchResults from kept candidates, preserving the
    # original combined_score ordering.
    survivors = [c for c in candidates if c.id in by_id]
    survivors.sort(key=lambda c: c.combined_score, reverse=True)

    results: list[SearchResult] = []
    for c in survivors:
        j = by_id[c.id]
        title = (j.get("title") or c.summary or "")[:80]
        summary = j.get("summary") or c.summary
        relevance = j.get("relevance_reason") or (
            f"Matched query (score: {c.combined_score:.2f})"
        )
        results.append(SearchResult(
            id=c.id,
            source=c.source,
            type=c.type,
            person=c.person,
            title=title,
            summary=summary,
            relevance=relevance,
            metadata=c.metadata,
        ))
    return results


async def synthesize_answer(
    client: "anthropic.AsyncAnthropic",
    question: str,
    snippets: list[tuple[str, str]],
    *,
    store: MemoryStore | None = None,
) -> dict:
    """One Haiku call to synthesize a natural-language answer.

    ``snippets`` is a list of ``(id, snippet)`` tuples — the relevant
    extracted material from filter_candidates. Returns
    ``{"answer": str, "source_ids": list[str]}``. Cost is recorded
    against ``"summary"``.
    """
    if not snippets:
        return {
            "answer": "No relevant memories found.",
            "source_ids": [],
        }

    snippet_block = "\n".join(
        f"[id={sid}] {snippet}" for sid, snippet in snippets if snippet
    )
    if not snippet_block.strip():
        return {
            "answer": "No relevant memories found.",
            "source_ids": [],
        }

    user_msg = (
        f"Question: {question}\n\nRelevant snippets:\n{snippet_block}\n\n"
        "Synthesize an answer grounded only in these snippets."
    )

    try:
        response = await client.messages.create(
            model=RERANK_MODEL,
            max_tokens=RERANK_MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": SYNTHESIZE_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                },
            ],
            tools=[SYNTHESIZE_TOOL],
            tool_choice={"type": "tool", "name": "synthesize_answer"},
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        logger.warning("Haiku synthesis call failed: %s", e)
        return {
            "answer": "Synthesis failed; falling back to raw snippets.",
            "source_ids": [sid for sid, _ in snippets],
        }

    await _record_haiku_cost(
        store,
        purpose="summary",
        usage=getattr(response, "usage", None),
        metadata={"step": "synthesize"},
    )

    payload = _extract_tool_use(response, "synthesize_answer")
    if not payload:
        logger.warning("Synthesis response missing synthesize_answer block")
        return {
            "answer": "Synthesis failed; falling back to raw snippets.",
            "source_ids": [sid for sid, _ in snippets],
        }

    answer = payload.get("answer") or ""
    source_ids = payload.get("source_ids") or []
    if not isinstance(source_ids, list):
        source_ids = []
    return {"answer": answer, "source_ids": source_ids}


# ---------------------------------------------------------------------------
# Reranking stub (legacy fallback)
# ---------------------------------------------------------------------------


async def rerank_stub(
    candidates: list[SearchCandidate],
    query: str,
) -> list[SearchResult]:
    """No-op fallback that turns candidates into results unchanged.

    Kept as a safety net for offline boots, tests that don't mock the
    Anthropic client, and transient API failures inside
    ``rerank_with_haiku``. Production code paths always go through
    ``rerank_with_haiku`` first.
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
    client: "anthropic.AsyncAnthropic | None" = None,
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
        client: Optional Anthropic client for reranking / summary
            synthesis. If ``None``, the backend lazily builds one from
            ``config.api_keys.anthropic``. If neither is available,
            lookup falls back to ``rerank_stub`` (no model calls) and
            summary falls back to a concatenated answer.

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
            client=client,
        )
    elif mode == "summary":
        return await _search_summary(
            store, query,
            types=types,
            person=person,
            include_conversations=include_conversations,
            include_archived=include_archived,
            client=client,
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
    client: "anthropic.AsyncAnthropic | None" = None,
) -> dict:
    """Lookup mode: return ranked facts and conversations.

    Pipeline: hybrid search → Haiku rerank (5-6 parallel calls) →
    drop irrelevant → re-sort by combined_score → top-K. Falls back
    to ``rerank_stub`` if no Anthropic client is available.
    """
    candidates = await hybrid_search(
        store, query,
        types=types,
        person=person,
        include_conversations=include_conversations,
        include_archived=include_archived,
    )

    results = await rerank_with_haiku(
        candidates, query, client=client, store=store,
    )

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
    client: "anthropic.AsyncAnthropic | None" = None,
) -> dict:
    """Summary mode: synthesize an answer from relevant memories.

    Pipeline: hybrid search → Haiku filter (parallel) → Haiku
    synthesis (one call) → ``{"answer": str, "sources": [...]}``.
    Falls back to a concatenated summary string if no client is
    available.
    """
    candidates = await hybrid_search(
        store, query,
        types=types,
        person=person,
        include_conversations=include_conversations,
        include_archived=include_archived,
    )

    if not candidates:
        return {"answer": "No relevant memories found.", "sources": []}

    client = _maybe_get_anthropic_client(client)

    if client is None:
        # Offline fallback. Better than crashing the tool call.
        logger.info("No Anthropic client available; summary fallback path")
        source_ids: list[str] = []
        for c in candidates[:15]:
            source_ids.append(c.id)
            if c.source == "memory":
                await store.update_memory_relevance(c.id)
                if c.metadata.get("status") == "archived":
                    await store.unarchive_memory(c.id)
        summaries = [c.summary for c in candidates[:10] if c.summary]
        answer = " | ".join(summaries) if summaries else "No relevant memories found."
        return {"answer": answer, "sources": source_ids[:10]}

    # Step 1: parallel filter + snippet extraction.
    judgments = await filter_candidates(
        client, candidates, query,
        intent="summarize",
        store=store,
        cost_purpose="summary",
    )

    relevant_ids = {
        j["candidate_id"] for j in judgments
        if j.get("candidate_id") and j.get("relevant")
    }
    snippet_by_id = {
        j["candidate_id"]: (j.get("snippet") or "")
        for j in judgments
        if j.get("candidate_id") and j.get("relevant")
    }

    # Update relevance for the candidates the filter kept.
    relevant_candidates = [c for c in candidates if c.id in relevant_ids]
    for c in relevant_candidates:
        if c.source == "memory":
            await store.update_memory_relevance(c.id)
            if c.metadata.get("status") == "archived":
                await store.unarchive_memory(c.id)

    if not relevant_candidates:
        return {"answer": "No relevant memories found.", "sources": []}

    # Build (id, snippet) pairs in combined-score order. Cap at the
    # same 10-source limit the lookup mode uses.
    ordered = sorted(
        relevant_candidates,
        key=lambda c: c.combined_score,
        reverse=True,
    )[:10]
    snippets = [
        (c.id, snippet_by_id.get(c.id) or c.summary or c.content)
        for c in ordered
    ]

    # Step 2: synthesis.
    synth = await synthesize_answer(
        client, query, snippets, store=store,
    )

    return {
        "answer": synth.get("answer", ""),
        "sources": synth.get("source_ids", []),
    }
