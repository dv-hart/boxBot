"""Nightly dream-phase consolidation (Phase B PR1).

Runs at 3 AM via a recurring scheduler trigger. The roadmap
(``docs/plans/memory-roadmap-post-phase-a.md`` §3) stages delivery
across multiple PRs; **this module covers PR1 only**:

1. Deterministic clustering of candidate memories at cosine ≥ 0.7
   (clusters of size ≥3 are *logged* — schema formation lands in PR2).
2. Pair near-duplicate memories at cosine ≥ 0.85 where the two were not
   co-injected during the day (cross-conversation only — daytime
   extraction handles co-injected dedup via invalidations).
3. Submit one Anthropic batch with one ``dedup_decision`` request per
   pair, structured-output via the ``DEDUP_TOOL`` schema.
4. Apply with confidence ≥ 0.8 — but **default audit_only=True**, so
   PR1 ships in a "log decisions, mutate nothing" safety mode.
5. Integrate ``boxbot.memory.maintenance.run_maintenance`` into the
   cycle (this is its first wiring; it ran nowhere before).

Soft-delete only: the dream phase NEVER ``DELETE``s memories. It marks
``status='invalidated'`` with ``superseded_by`` set, and stamps
``consolidated_by=<batch_id>`` for full auditability/undo via
``scripts/undo_last_dream.py``.

Public entrypoint: :func:`run_dream_cycle`.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable
from uuid import uuid4

import anthropic
import numpy as np

from boxbot.core.paths import WORKSPACE_DIR
from boxbot.memory.embeddings import cosine_similarity
from boxbot.memory.extraction import (
    DEFAULT_EXTRACTION_MODEL,
    STANDARD_PRICING,
    compute_cost,
)

if TYPE_CHECKING:
    from boxbot.memory.store import Memory, MemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------


# Cosine thresholds. These are deliberately conservative — dedup is the
# highest-leverage, lowest-risk dream operation, but we still want the
# model to be the final arbiter on close calls. Pairing at 0.85 catches
# near-duplicates; clustering at 0.7 catches the wider "related-fact"
# neighbourhood that PR2 will turn into schemas.
CLUSTER_THRESHOLD = 0.70
NEAR_DUP_THRESHOLD = 0.85

# Revisit pool sizes per the roadmap §3 "Cadence + scope per night".
REVISIT_USED_TODAY = 3
REVISIT_AGE_DECAYED = 2
REVISIT_UNIFORM = 1

# Age-decay half-life for pool B sampling (days). exp(-d/30) ≈ 0.5 at 21d.
AGE_DECAY_DAYS = 30

# Confidence gate for applying any dedup decision. The roadmap states
# 0.8 for dedup; lower than 0.9 used for revisit-driven invalidation
# (which lands in a later PR).
DEDUP_CONFIDENCE_THRESHOLD = 0.8

# Per-cycle hard cap on dedup pairs. Above this, lowest-confidence pairs
# are dropped before submission. Mirrors the roadmap's "30 memories per
# night" change budget at the pair level.
MAX_DEDUP_PAIRS_DEFAULT = 30


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CandidateSet:
    """All memories considered by a single dream cycle."""

    new_today: list[Memory] = field(default_factory=list)
    revisits: list[Memory] = field(default_factory=list)
    revisit_pool_origin: dict[str, str] = field(default_factory=dict)
    """Map of memory_id -> 'used_today' | 'age_decayed' | 'uniform'."""

    @property
    def all_memories(self) -> list[Memory]:
        seen: set[str] = set()
        out: list[Memory] = []
        for m in list(self.new_today) + list(self.revisits):
            if m.id not in seen:
                seen.add(m.id)
                out.append(m)
        return out

    @property
    def all_ids(self) -> list[str]:
        return [m.id for m in self.all_memories]


@dataclass
class Cluster:
    """A connected component of memories at cosine ≥ CLUSTER_THRESHOLD."""

    memory_ids: list[str]


@dataclass
class NearDupPair:
    """Two memories at cosine ≥ NEAR_DUP_THRESHOLD that were NOT co-injected."""

    memory_id_a: str
    memory_id_b: str
    cosine: float


@dataclass
class DedupDecision:
    """Parsed output of one dedup tool call."""

    custom_id: str
    pair: NearDupPair | None
    decision: str  # merge_into_a | merge_into_b | distinct | unsure
    merged_content: str | None
    merged_summary: str | None
    evidence: list[str]
    confidence: float
    notes: str


@dataclass
class DreamApplyResult:
    """Summary of what one apply pass did (or would have done)."""

    audit_only: bool
    decisions: list[DedupDecision] = field(default_factory=list)
    applied_merges: int = 0
    skipped_low_confidence: int = 0
    skipped_unsure_or_distinct: int = 0


# ---------------------------------------------------------------------------
# Tool definition + system prompt for dedup
# ---------------------------------------------------------------------------


# Output schema for one dedup decision. ``evidence`` is required so the
# apply step can reject creates without provenance (defence-in-depth
# against confabulation per the roadmap risks section).
DEDUP_TOOL: dict[str, Any] = {
    "name": "dedup_decision",
    "description": (
        "Decide whether two memories represent the same fact. Be "
        "conservative — only merge when they are clearly the same fact "
        "in different words. If there is any meaningful difference, "
        "return 'distinct'. If you cannot tell, return 'unsure'."
    ),
    "input_schema": {
        "type": "object",
        "required": ["decision", "evidence", "confidence"],
        "properties": {
            "decision": {
                "type": "string",
                "enum": [
                    "merge_into_a",
                    "merge_into_b",
                    "distinct",
                    "unsure",
                ],
            },
            "merged_content": {"type": "string"},
            "merged_summary": {"type": "string"},
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Memory IDs you cited. Required for any merge — "
                    "must include both IDs from the pair."
                ),
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "notes": {"type": "string"},
        },
    },
}


DEDUP_SYSTEM_PROMPT = """\
You are the dream-phase memory consolidator for boxBot.

Each request shows you exactly two memories from a household assistant's long-term store. Your single job: decide whether they record the same underlying fact in different words, or whether they are distinct.

Output rules (strictly via the `dedup_decision` tool):

- `merge_into_a`: keep memory A's id, fold in any extra information from B. Use this when both clearly state the same fact and A's wording is at least as good as B's.
- `merge_into_b`: keep memory B's id, fold in extra info from A.
- `distinct`: any meaningful difference — different person, different time period, different scope, different attribute. **When in doubt, prefer this.**
- `unsure`: you genuinely cannot tell. Returns are logged but no merge happens.

Hard rules:

1. Be conservative. Spurious merges destroy information; a missed merge is fixable on a future night. False merges are not.
2. If two memories are about different people, different times, or different aspects, they are `distinct` — even if the words overlap.
3. `confidence` must be ≥ 0.8 only if you are sure; below that, the apply step will skip the merge regardless of decision.
4. `evidence` MUST include both memory IDs from the pair when merging.
5. When merging, `merged_content` should be 1–3 sentences containing every fact present in either input. `merged_summary` should be ≤80 chars.
6. Operational memories (activity-log entries) are never merged — return `distinct`. They are append-only.

If the inputs are too thin to judge (e.g. single-word summaries), return `unsure` with confidence ≤ 0.4.
"""


# ---------------------------------------------------------------------------
# Phase 1 — deterministic
# ---------------------------------------------------------------------------


def _midnight_utc(now: datetime | None = None) -> datetime:
    """Return today's UTC midnight as an aware-naive datetime in UTC."""
    now = now or datetime.utcnow()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _weighted_sample(
    pool: list[Memory],
    *,
    k: int,
    weights: list[float],
    rng: random.Random,
) -> list[Memory]:
    """Sample ``k`` items without replacement, proportional to ``weights``."""
    if k <= 0 or not pool:
        return []
    # random.choices supports weights but only with replacement. Implement
    # weighted-without-replacement by repeatedly sampling and removing.
    available = list(zip(pool, weights))
    out: list[Memory] = []
    for _ in range(min(k, len(available))):
        total = sum(w for _, w in available)
        if total <= 0:
            break
        r = rng.uniform(0, total)
        upto = 0.0
        for i, (mem, w) in enumerate(available):
            upto += w
            if r <= upto:
                out.append(mem)
                available.pop(i)
                break
    return out


async def gather_candidates(
    store: MemoryStore,
    *,
    now: datetime | None = None,
    rng: random.Random | None = None,
) -> CandidateSet:
    """Collect today's new memories plus 6 revisits across three pools.

    Pools (per roadmap §3):
      A — "used today" (last_relevant_at >= midnight): pick 3
      B — age-decayed random across active memories: pick 2 weighted
          by exp(-age_days / AGE_DECAY_DAYS)
      C — uniform random across active memories: pick 1

    Memories already in the new-today set are excluded from the revisit
    pools so the same record isn't sampled twice.
    """
    rng = rng or random.Random()
    now = now or datetime.utcnow()
    midnight = _midnight_utc(now)
    midnight_iso = midnight.isoformat()

    # New memories created since midnight (active only — we don't dedup
    # archived/invalidated rows).
    new_today = await store.list_memories_created_since(midnight_iso)
    new_today_ids = {m.id for m in new_today}

    # Pool A: used today
    used_today_all = await store.list_memories_relevant_since(midnight_iso)
    pool_a = [m for m in used_today_all if m.id not in new_today_ids]

    # Pools B + C draw from "active memories of any age".
    active_all = await store.list_memories(status="active", limit=10_000)
    active_excluding = [
        m for m in active_all if m.id not in new_today_ids
    ]

    revisits: list[Memory] = []
    origin: dict[str, str] = {}

    # Pool A pick (uniform random within "used today" — heavy exploit).
    a_picked = rng.sample(pool_a, k=min(REVISIT_USED_TODAY, len(pool_a)))
    for m in a_picked:
        revisits.append(m)
        origin[m.id] = "used_today"

    picked_ids = {m.id for m in revisits}
    pool_b_candidates = [m for m in active_excluding if m.id not in picked_ids]

    # Pool B: weight ∝ exp(-age_days / AGE_DECAY_DAYS)
    weights: list[float] = []
    for m in pool_b_candidates:
        try:
            created = datetime.fromisoformat(m.created_at)
        except Exception:
            created = now
        age_days = max(0.0, (now - created).total_seconds() / 86400.0)
        weights.append(math.exp(-age_days / AGE_DECAY_DAYS))

    b_picked = _weighted_sample(
        pool_b_candidates, k=REVISIT_AGE_DECAYED,
        weights=weights, rng=rng,
    )
    for m in b_picked:
        revisits.append(m)
        origin[m.id] = "age_decayed"

    picked_ids = {m.id for m in revisits}
    pool_c_candidates = [m for m in active_excluding if m.id not in picked_ids]

    # Pool C: uniform random
    c_picked = rng.sample(
        pool_c_candidates, k=min(REVISIT_UNIFORM, len(pool_c_candidates)),
    )
    for m in c_picked:
        revisits.append(m)
        origin[m.id] = "uniform"

    return CandidateSet(
        new_today=new_today,
        revisits=revisits,
        revisit_pool_origin=origin,
    )


def _embedding_or_none(m: Memory) -> np.ndarray | None:
    return m.embedding if m.embedding is not None else None


async def cluster_candidates(
    candidates: CandidateSet,
) -> list[Cluster]:
    """Greedy single-link clustering at cosine ≥ CLUSTER_THRESHOLD.

    Clusters of size ≥3 are surfaced for the audit log (and become
    schema-formation requests in PR2). Memories without embeddings are
    skipped (they cannot cluster).
    """
    memories = [m for m in candidates.all_memories if m.embedding is not None]
    n = len(memories)
    if n < 2:
        return []

    # Union-Find for O(N²·α) connected-component clustering.
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(memories[i].embedding, memories[j].embedding)
            if sim >= CLUSTER_THRESHOLD:
                union(i, j)

    groups: dict[int, list[str]] = {}
    for i, m in enumerate(memories):
        root = find(i)
        groups.setdefault(root, []).append(m.id)

    return [Cluster(memory_ids=ids) for ids in groups.values() if len(ids) >= 2]


async def find_near_duplicates(
    store: MemoryStore,
    candidates: CandidateSet,
) -> list[NearDupPair]:
    """Pair memories at cosine ≥ NEAR_DUP_THRESHOLD that were NOT co-injected.

    "Co-injected" is determined by the conversation each memory cites as
    its source. If conversation X already recorded both memories as
    accessed (e.g. they were both in the [Active Memories] block), the
    daytime extraction had a chance to invalidate one in favour of the
    other — we don't want the dream phase to second-guess that.
    """
    memories = [m for m in candidates.all_memories if m.embedding is not None]
    pairs: list[NearDupPair] = []
    n = len(memories)
    if n < 2:
        return pairs

    # Build a co-injection map: memory_id -> set of conversation_ids that
    # listed it in accessed_memories.
    co_inject_index = await _build_co_injection_index(store, memories)

    # Operational memories are never deduped (append-only). Skip them.
    is_op = {m.id for m in memories if m.type == "operational"}

    for i in range(n):
        a = memories[i]
        if a.id in is_op:
            continue
        for j in range(i + 1, n):
            b = memories[j]
            if b.id in is_op:
                continue
            sim = cosine_similarity(a.embedding, b.embedding)
            if sim < NEAR_DUP_THRESHOLD:
                continue
            shared = co_inject_index.get(a.id, set()) & co_inject_index.get(b.id, set())
            if shared:
                # They were co-injected in some conversation — daytime
                # extraction owns this case. Skip.
                continue
            pairs.append(
                NearDupPair(
                    memory_id_a=a.id, memory_id_b=b.id, cosine=float(sim),
                )
            )

    # Stable order: highest cosine first so any cap-by-budget step keeps
    # the most-confident pairs.
    pairs.sort(key=lambda p: p.cosine, reverse=True)
    return pairs


async def _build_co_injection_index(
    store: MemoryStore,
    memories: list[Memory],
) -> dict[str, set[str]]:
    """For each memory id, list conversation ids where it was injected."""
    index: dict[str, set[str]] = {m.id: set() for m in memories}
    # We scan recent conversations rather than every row — co-injection
    # signals from years ago are noise. 90 days covers the
    # decay/retention horizons.
    cutoff = (datetime.utcnow() - timedelta(days=90)).isoformat()
    cursor = await store.db.execute(
        "SELECT id, accessed_memories FROM conversations "
        "WHERE started_at >= ?",
        (cutoff,),
    )
    rows = await cursor.fetchall()
    for row in rows:
        try:
            ids = json.loads(row["accessed_memories"] or "[]")
        except (json.JSONDecodeError, TypeError):
            continue
        if not ids:
            continue
        conv_id = row["id"]
        id_set = set(ids)
        for mid in id_set:
            if mid in index:
                index[mid].add(conv_id)
    return index


# ---------------------------------------------------------------------------
# Phase 2 — model batch
# ---------------------------------------------------------------------------


def _build_dedup_request(
    *,
    custom_id: str,
    memory_a: Memory,
    memory_b: Memory,
    cosine: float,
    model: str = DEFAULT_EXTRACTION_MODEL,
    max_tokens: int = 800,
) -> dict[str, Any]:
    """Build one Anthropic batch request for a single dedup pair."""
    user_msg = (
        f"You are comparing TWO memories. Return a single `dedup_decision` "
        f"tool call.\n\n"
        f"Embedding cosine: {cosine:.3f}\n\n"
        f"[Memory A]\n"
        f"id: {memory_a.id}\n"
        f"type: {memory_a.type}\n"
        f"person: {memory_a.person or '(none)'}\n"
        f"content: {memory_a.content}\n"
        f"summary: {memory_a.summary}\n"
        f"created_at: {memory_a.created_at}\n\n"
        f"[Memory B]\n"
        f"id: {memory_b.id}\n"
        f"type: {memory_b.type}\n"
        f"person: {memory_b.person or '(none)'}\n"
        f"content: {memory_b.content}\n"
        f"summary: {memory_b.summary}\n"
        f"created_at: {memory_b.created_at}\n"
    )
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": DEDUP_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
            ],
            "tools": [DEDUP_TOOL],
            "tool_choice": {"type": "tool", "name": "dedup_decision"},
            "messages": [{"role": "user", "content": user_msg}],
        },
    }


def _custom_id_for_pair(batch_uuid: str, index: int) -> str:
    return f"dream-{batch_uuid}-{index:04d}"


async def submit_dream_batch(
    client: anthropic.AsyncAnthropic,
    store: MemoryStore,
    pairs: list[NearDupPair],
    *,
    candidate_ids: list[str] | None = None,
    model: str = DEFAULT_EXTRACTION_MODEL,
    max_pairs: int = MAX_DEDUP_PAIRS_DEFAULT,
) -> str:
    """Submit one Anthropic batch with one dedup request per pair.

    Records a ``pending_dreams`` row immediately so the DreamPoller can
    resume polling after a crash. Returns the batch_id.

    Raises ValueError if ``pairs`` is empty (no batch to submit).
    """
    if not pairs:
        raise ValueError("submit_dream_batch called with no pairs")

    pairs = pairs[:max_pairs]

    # Pull memories in one pass so the request builder doesn't
    # round-trip the DB per pair.
    needed_ids: set[str] = set()
    for p in pairs:
        needed_ids.add(p.memory_id_a)
        needed_ids.add(p.memory_id_b)
    by_id: dict[str, Memory] = {}
    for mid in needed_ids:
        m = await store.get_memory_no_touch(mid)
        if m is not None:
            by_id[mid] = m

    # Filter out pairs where either memory vanished between candidate
    # gather and submission.
    valid: list[tuple[str, NearDupPair]] = []
    batch_uuid = uuid4().hex[:12]
    for i, p in enumerate(pairs):
        if p.memory_id_a not in by_id or p.memory_id_b not in by_id:
            continue
        custom_id = _custom_id_for_pair(batch_uuid, i)
        valid.append((custom_id, p))

    if not valid:
        raise ValueError("All dedup pairs referenced missing memories")

    requests: list[dict[str, Any]] = []
    for custom_id, p in valid:
        requests.append(
            _build_dedup_request(
                custom_id=custom_id,
                memory_a=by_id[p.memory_id_a],
                memory_b=by_id[p.memory_id_b],
                cosine=p.cosine,
                model=model,
            )
        )

    batch = await client.messages.batches.create(requests=requests)
    batch_id = batch.id

    request_types = {"dedup": len(valid)}
    summary = (
        f"Dream cycle: {len(valid)} dedup requests submitted "
        f"(uuid={batch_uuid})"
    )
    await store.create_pending_dream(
        batch_id=batch_id,
        candidate_ids=candidate_ids or [],
        request_types=request_types,
        summary=summary,
    )

    logger.info(
        "Submitted dream batch %s with %d dedup requests (model=%s)",
        batch_id, len(valid), model,
    )
    return batch_id


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def _parse_dedup_message(message: Any) -> dict[str, Any] | None:
    """Pull the ``dedup_decision`` tool input from one batch result message."""
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
        if block_name != "dedup_decision":
            continue
        payload = getattr(block, "input", None)
        if payload is None and isinstance(block, dict):
            payload = block.get("input")
        if isinstance(payload, dict):
            return payload
    return None


def _decision_from_payload(
    custom_id: str,
    pair: NearDupPair | None,
    payload: dict,
) -> DedupDecision:
    return DedupDecision(
        custom_id=custom_id,
        pair=pair,
        decision=str(payload.get("decision") or "unsure"),
        merged_content=payload.get("merged_content"),
        merged_summary=payload.get("merged_summary"),
        evidence=list(payload.get("evidence") or []),
        confidence=float(payload.get("confidence") or 0.0),
        notes=str(payload.get("notes") or ""),
    )


async def apply_dream_result(
    store: MemoryStore,
    batch_message_iter: Iterable[Any] | Any,
    *,
    batch_id: str,
    pairs_by_custom_id: dict[str, NearDupPair] | None = None,
    audit_only: bool = True,
) -> DreamApplyResult:
    """Parse JSONL batch results and apply dedup decisions.

    Args:
        store: MemoryStore.
        batch_message_iter: Either an async iterator of batch result
            entries (production) or a synchronous iterable (tests).
        batch_id: The Anthropic batch id; stamped onto every audited
            memory via ``consolidated_by``.
        pairs_by_custom_id: Map of custom_id -> NearDupPair so we can
            reattach pair metadata when applying. Pass ``{}`` if not
            available — the apply step will infer from ``evidence``.
        audit_only: If True (default), log decisions but DO NOT mutate
            the memory store. PR1 ships in this mode by default.
    """
    pairs_by_custom_id = pairs_by_custom_id or {}
    decisions: list[DedupDecision] = []

    # Normalise to a synchronous list of (custom_id, message_or_error).
    entries = await _collect_entries(batch_message_iter)

    for entry in entries:
        custom_id = (
            getattr(entry, "custom_id", None)
            or (entry.get("custom_id") if isinstance(entry, dict) else None)
        )
        result_obj = (
            getattr(entry, "result", None)
            or (entry.get("result") if isinstance(entry, dict) else None)
        )
        result_type = (
            getattr(result_obj, "type", None)
            or (result_obj.get("type") if isinstance(result_obj, dict) else None)
        )
        if result_type != "succeeded":
            logger.warning(
                "Dream batch %s entry %s ended with %s — skipping",
                batch_id, custom_id, result_type,
            )
            continue
        message = (
            getattr(result_obj, "message", None)
            or (result_obj.get("message") if isinstance(result_obj, dict) else None)
        )
        payload = _parse_dedup_message(message)
        if payload is None:
            logger.warning(
                "Dream entry %s missing dedup_decision tool call", custom_id,
            )
            continue
        pair = pairs_by_custom_id.get(custom_id)
        decisions.append(_decision_from_payload(custom_id, pair, payload))

    result = DreamApplyResult(audit_only=audit_only, decisions=decisions)

    for d in decisions:
        if d.decision in {"distinct", "unsure"}:
            result.skipped_unsure_or_distinct += 1
            continue
        if d.confidence < DEDUP_CONFIDENCE_THRESHOLD:
            result.skipped_low_confidence += 1
            continue
        # Resolve the pair. The model's evidence list should contain
        # both IDs even if pairs_by_custom_id wasn't supplied.
        pair = d.pair
        if pair is None and len(d.evidence) >= 2:
            pair = NearDupPair(
                memory_id_a=d.evidence[0],
                memory_id_b=d.evidence[1],
                cosine=0.0,
            )
        if pair is None:
            logger.warning(
                "Dream decision %s lacks pair + evidence; skipping",
                d.custom_id,
            )
            continue
        if d.decision not in {"merge_into_a", "merge_into_b"}:
            continue
        if not d.evidence or pair.memory_id_a not in d.evidence \
                or pair.memory_id_b not in d.evidence:
            logger.warning(
                "Dream decision %s evidence does not cite both memories; "
                "skipping (defence against confabulation)",
                d.custom_id,
            )
            continue

        if audit_only:
            # Audit-only: count the decision as if it would have
            # applied, but leave the store untouched.
            result.applied_merges += 1
            continue

        keeper, loser = (
            (pair.memory_id_a, pair.memory_id_b)
            if d.decision == "merge_into_a"
            else (pair.memory_id_b, pair.memory_id_a)
        )
        await _apply_merge(
            store,
            keeper_id=keeper,
            loser_id=loser,
            merged_content=d.merged_content,
            merged_summary=d.merged_summary,
            batch_id=batch_id,
        )
        result.applied_merges += 1

    logger.info(
        "Dream apply (audit_only=%s, batch=%s): merges=%d, "
        "skipped_low_conf=%d, skipped_distinct_or_unsure=%d",
        audit_only, batch_id,
        result.applied_merges,
        result.skipped_low_confidence,
        result.skipped_unsure_or_distinct,
    )
    return result


async def _collect_entries(batch_message_iter: Any) -> list[Any]:
    """Materialise an async or sync iterable of batch result entries."""
    if batch_message_iter is None:
        return []
    if hasattr(batch_message_iter, "__aiter__"):
        out: list[Any] = []
        async for entry in batch_message_iter:
            out.append(entry)
        return out
    return list(batch_message_iter)


async def _apply_merge(
    store: MemoryStore,
    *,
    keeper_id: str,
    loser_id: str,
    merged_content: str | None,
    merged_summary: str | None,
    batch_id: str,
) -> None:
    """Mutate the store to merge ``loser`` into ``keeper`` (soft delete).

    1. If merged_content/summary provided, update the keeper to that text.
    2. Invalidate the loser with superseded_by=keeper.
    3. Stamp consolidated_by=batch_id on both for auditability/undo.
    """
    keeper = await store.get_memory_no_touch(keeper_id)
    loser = await store.get_memory_no_touch(loser_id)
    if keeper is None or loser is None:
        logger.warning(
            "Merge skipped: keeper=%s loser=%s — at least one missing",
            keeper_id, loser_id,
        )
        return

    if merged_content and merged_summary:
        # Combine tags + people sets so no info is dropped on the merge.
        merged_tags = sorted(set(keeper.tags) | set(loser.tags))
        merged_people = sorted(set(keeper.people) | set(loser.people))
        await store.update_memory_content(
            keeper_id,
            content=merged_content,
            summary=merged_summary,
            tags=merged_tags,
            people=merged_people,
        )

    # Soft-delete the loser, citing the keeper as the replacement and the
    # dream batch as the cause. We use ``invalidated_by=batch_id`` so the
    # provenance chain reads "this was retired by dream batch X".
    await store.invalidate_memory(
        loser_id,
        invalidated_by=batch_id,
        superseded_by=keeper_id,
    )

    # Stamp consolidated_by on both rows so undo can find them.
    await store.set_dream_audit_fields(
        keeper_id, consolidated_by=batch_id,
    )
    await store.set_dream_audit_fields(
        loser_id, consolidated_by=batch_id,
    )


# ---------------------------------------------------------------------------
# Audit log (workspace markdown)
# ---------------------------------------------------------------------------


def _dream_log_path(now: datetime | None = None) -> Path:
    now = now or datetime.utcnow()
    day = now.strftime("%Y-%m-%d")
    return WORKSPACE_DIR / "notes" / "system" / "dream-log" / f"{day}.md"


def _format_revisits(candidates: CandidateSet) -> list[str]:
    lines: list[str] = []
    for m in candidates.revisits:
        pool = candidates.revisit_pool_origin.get(m.id, "?")
        lines.append(
            f"- ({pool}) `{m.id[:8]}` {m.summary}"
        )
    return lines


def write_dream_log(
    *,
    candidates: CandidateSet,
    clusters: list[Cluster],
    pairs: list[NearDupPair],
    batch_id: str | None,
    request_types: dict[str, int],
    decisions: list[DedupDecision] | None,
    audit_only: bool,
    cost_usd: float | None = None,
    maintenance_stats: dict[str, int] | None = None,
    now: datetime | None = None,
) -> Path:
    """Write a per-cycle markdown audit log to the workspace.

    Returns the path the log was written to. Lives outside the sandbox —
    this runs in the main process.
    """
    now = now or datetime.utcnow()
    path = _dream_log_path(now)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Dream cycle {now.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append(f"**Mode:** {'audit-only' if audit_only else 'apply'}")
    lines.append("")

    lines.append("## Candidates")
    lines.append(f"- {len(candidates.new_today)} new memories from today")
    for m in candidates.new_today:
        lines.append(f"  - `{m.id[:8]}` {m.summary}")
    lines.append(f"- {len(candidates.revisits)} revisits")
    lines.extend(f"  {l}" for l in _format_revisits(candidates))
    lines.append("")

    lines.append("## Clustering")
    big = [c for c in clusters if len(c.memory_ids) >= 3]
    lines.append(
        f"- {len(big)} clusters of size ≥3 "
        f"(would-be schemas; deferred to PR2)"
    )
    for c in big:
        lines.append(
            f"  - {len(c.memory_ids)} members: "
            + ", ".join(f"`{mid[:8]}`" for mid in c.memory_ids)
        )
    lines.append(f"- {len(pairs)} near-duplicate pairs flagged for dedup")
    for p in pairs[:50]:
        lines.append(
            f"  - cosine {p.cosine:.3f}: "
            f"`{p.memory_id_a[:8]}` ↔ `{p.memory_id_b[:8]}`"
        )
    lines.append("")

    lines.append("## Batch")
    if batch_id is None:
        lines.append("- (no batch — nothing to dedup)")
    else:
        lines.append(f"- batch_id: `{batch_id}`")
        lines.append(f"- request_types: {json.dumps(request_types)}")
    lines.append("")

    lines.append("## Decisions")
    if not decisions:
        lines.append(
            "- (none recorded yet — DreamPoller applies on batch end)"
        )
    else:
        for d in decisions:
            tag = d.decision
            pair_repr = (
                f"(`{d.pair.memory_id_a[:8]}`, `{d.pair.memory_id_b[:8]}`)"
                if d.pair else f"(evidence: {d.evidence})"
            )
            note = f' "{d.notes}"' if d.notes else ""
            lines.append(
                f"- pair {pair_repr}: {tag}, "
                f"confidence {d.confidence:.2f}{note}"
            )
    lines.append("")

    lines.append("## Cost")
    if cost_usd is None:
        lines.append("- (recorded by DreamPoller on batch end)")
    else:
        lines.append(f"- ${cost_usd:.4f} ({DEFAULT_EXTRACTION_MODEL} batch)")
    lines.append("")

    lines.append("## Applied")
    if audit_only:
        lines.append("- (in audit-only mode, nothing applied)")
    elif decisions is None:
        lines.append("- (pending poll)")
    else:
        merges = sum(
            1 for d in decisions
            if d.decision in {"merge_into_a", "merge_into_b"}
            and d.confidence >= DEDUP_CONFIDENCE_THRESHOLD
        )
        lines.append(f"- {merges} merges applied")
    lines.append("")

    lines.append("## Maintenance")
    if maintenance_stats is None:
        lines.append("- (skipped or failed — see boxbot.log)")
    else:
        lines.append(
            f"- archived: {maintenance_stats.get('archived_memories', 0)} "
            f"memories, {maintenance_stats.get('archived_conversations', 0)} "
            f"conversations"
        )
        lines.append(
            f"- evicted: {maintenance_stats.get('evicted_archived', 0)} "
            f"archived, {maintenance_stats.get('evicted_active', 0)} active"
        )
        lines.append(
            f"- fts_rebuilt: {maintenance_stats.get('fts_rebuilt', 0)}"
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Dream log written to %s", path)
    return path


# ---------------------------------------------------------------------------
# Top-level cycle
# ---------------------------------------------------------------------------


async def run_dream_cycle(
    store: MemoryStore,
    client: anthropic.AsyncAnthropic | None,
    *,
    audit_only: bool = True,
    max_dedup_pairs: int = MAX_DEDUP_PAIRS_DEFAULT,
    rng: random.Random | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """End-to-end nightly cycle: gather → cluster → dedup → submit → log.

    The DreamPoller picks up batch results and calls
    :func:`apply_dream_result` once Anthropic finishes the batch.

    Maintenance (archival + storage cap + FTS rebuild) runs *after* the
    batch is submitted — it's deterministic housekeeping independent of
    model results.

    Returns a dict summary suitable for logging / morning-brief use.
    """
    now = now or datetime.utcnow()
    rng = rng or random.Random()

    # Phase 1 — gather and cluster (no model calls).
    candidates = await gather_candidates(store, now=now, rng=rng)
    clusters = await cluster_candidates(candidates)
    pairs = await find_near_duplicates(store, candidates)
    pairs = pairs[:max_dedup_pairs]

    batch_id: str | None = None
    request_types: dict[str, int] = {}
    if pairs and client is not None:
        try:
            batch_id = await submit_dream_batch(
                client, store, pairs,
                candidate_ids=candidates.all_ids,
                max_pairs=max_dedup_pairs,
            )
            request_types = {"dedup": len(pairs)}
        except Exception:
            logger.exception("Dream batch submission failed")

    # Phase 3 — deterministic maintenance, runs after submission so the
    # batch is durably queued first.
    try:
        from boxbot.memory.maintenance import run_maintenance

        maintenance_stats = await run_maintenance(store)
    except Exception:
        logger.exception("Maintenance step failed in dream cycle")
        maintenance_stats = None

    # Write the audit log. Decisions are not yet known (DreamPoller
    # applies them later); the log captures the submission-time view.
    write_dream_log(
        candidates=candidates,
        clusters=clusters,
        pairs=pairs,
        batch_id=batch_id,
        request_types=request_types,
        decisions=None,
        audit_only=audit_only,
        maintenance_stats=maintenance_stats,
        now=now,
    )

    return {
        "now": now.isoformat(),
        "audit_only": audit_only,
        "candidate_count": len(candidates.all_ids),
        "new_today": len(candidates.new_today),
        "revisits": len(candidates.revisits),
        "clusters_big": sum(1 for c in clusters if len(c.memory_ids) >= 3),
        "near_dup_pairs": len(pairs),
        "batch_id": batch_id,
        "maintenance": maintenance_stats,
    }
