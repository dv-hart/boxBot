"""Nightly dream-phase consolidation (Phase B PR1).

Runs at 3 AM via a recurring scheduler trigger. The roadmap
(``docs/plans/memory-roadmap-post-phase-a.md`` §3) stages delivery
across multiple PRs; **this module covers PR1 only**:

1. Deterministic clustering of candidate memories at cosine ≥ 0.7
   (clusters of size ≥3 are *logged* — schema formation lands in PR2).
2. Pair near-duplicate memories at cosine ≥ NEAR_DUP_THRESHOLD and hand
   each pair to the model judge. Co-injected pairs are NOT excluded — a
   contradiction that keeps being injected together is exactly what
   needs adjudicating, and daytime extraction only resolves explicit
   in-conversation corrections.
3. Submit one Anthropic batch with one ``dedup_decision`` request per
   pair, structured-output via the ``DEDUP_TOOL`` schema. Each request
   carries provenance (who/what asserted each memory) and a
   "what else is known" block of related memories, and the model must
   reason before deciding.
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
# model to be the final arbiter on close calls. Clustering at 0.7 catches
# the wider "related-fact" neighbourhood that PR2 will turn into schemas.
#
# NEAR_DUP_THRESHOLD is the *default* recall filter for pairing; the live
# value is sourced from ``config.memory.dream_near_dup_threshold`` and
# threaded through ``run_dream_cycle``. The original 0.85 was calibrated
# by intuition and proved unreachable for all-MiniLM-L6-v2's compressed
# cosine range (real duplicates sit ~0.65-0.80; only near-verbatim text
# hits 0.85), so dedup never fired in production. 0.75 is the corrected
# default — the model + DEDUP_CONFIDENCE_THRESHOLD are the precision gate.
CLUSTER_THRESHOLD = 0.70
NEAR_DUP_THRESHOLD = 0.75

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

# dream_state key holding the ISO timestamp of the last completed cycle.
# Drives the "new since last run" candidate window (see gather_candidates).
DREAM_WATERMARK_KEY = "last_run_iso"


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
    """Two memories at cosine ≥ NEAR_DUP_THRESHOLD, surfaced for adjudication."""

    memory_id_a: str
    memory_id_b: str
    cosine: float


@dataclass
class DedupDecision:
    """Parsed output of one dedup tool call."""

    custom_id: str
    pair: NearDupPair | None
    decision: str  # merge_into_a|merge_into_b|supersede|flag|distinct|unsure
    merged_content: str | None
    merged_summary: str | None
    evidence: list[str]
    confidence: float
    notes: str
    stale_id: str | None = None  # supersede: the memory to invalidate
    reasoning: str = ""  # the model's think-it-through, captured for audit


@dataclass
class DreamApplyResult:
    """Summary of what one apply pass did (or would have done)."""

    audit_only: bool
    decisions: list[DedupDecision] = field(default_factory=list)
    applied_merges: int = 0
    applied_supersessions: int = 0
    flagged_contradictions: int = 0
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
        "Resolve two memories against each other, using the provenance "
        "and related context provided. Reason first, then decide: the "
        "same fact in different words (merge), genuinely different "
        "(distinct), or INCOMPATIBLE claims about the same subject "
        "(supersede / flag). When unsure whether they're the same, "
        "return 'distinct'; when they conflict but you can't tell which "
        "is current, return 'flag', never 'supersede'."
    ),
    "input_schema": {
        "type": "object",
        "required": ["reasoning", "decision", "evidence", "confidence"],
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Think it through BEFORE deciding. What does each "
                    "memory actually claim? Where did each come from — a "
                    "person stating or correcting it, or an automated "
                    "process reading an external source? What does the "
                    "related context say, and which memory does it "
                    "support? Which is more trustworthy, and why?"
                ),
            },
            "decision": {
                "type": "string",
                "enum": [
                    "merge_into_a",
                    "merge_into_b",
                    "supersede",
                    "flag",
                    "distinct",
                    "unsure",
                ],
            },
            "merged_content": {"type": "string"},
            "merged_summary": {"type": "string"},
            "stale_id": {
                "type": "string",
                "description": (
                    "For 'supersede' only: the id of the memory that is "
                    "now wrong or outdated and should be invalidated. "
                    "Must be one of the two memory ids in the pair; the "
                    "other is kept as the survivor."
                ),
            },
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Memory IDs you cited. Required for any merge or "
                    "supersede — must include both IDs from the pair."
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
You are the dream-phase memory consolidator for boxBot. Your real job is to keep what the household's assistant believes both true and tidy — not to mechanically match strings.

Each request gives you TWO memories, plus, for each, its **source** (who or what asserted it) and a **[What else is known]** block of related memories. Use all of it. Fill in `reasoning` first — actually think it through — then decide.

How to think it through:

1. **What does each memory claim?** State the underlying fact, not just the wording.
2. **Where did each come from?** A fact a person stated directly, or explicitly corrected, is strong evidence. A fact an automated process inferred from an ambiguous external source (a calendar event that names no one, an auto-generated review) is weak — it may be a re-derived guess. Weigh provenance accordingly: a human's correction outranks a machine's inference about the same thing, even if the machine's memory is newer.
3. **What else is known?** The related context often settles it outright (e.g. "Zara is 2 and not in preschool yet" decides who is graduating). A memory contradicted by well-established related facts is the stale one.
4. **Recency is the WEAKEST signal.** An automated process can re-derive the same error nightly, so the wrong fact often looks "newest." Only fall back to recency when provenance and context don't break the tie.

Decisions (strictly via the `dedup_decision` tool):

- `merge_into_a`: same fact; keep A's id and fold in any extra detail from B. Use when both clearly state the same fact and A's wording is at least as good.
- `merge_into_b`: same fact; keep B's id instead.
- `supersede`: the two make INCOMPATIBLE claims about the SAME subject — the same event attributed to different people, or a value that changed/was corrected ("lives in Portland" vs "moved to Seattle"). Set `stale_id` to the memory that is now WRONG or OUTDATED; the other survives. Choose the stale one by provenance and related context first, recency last. Use ONLY when the contradiction is clear AND you are confident which is current.
- `flag`: they seem to conflict but you are not sure they truly do, or not sure which is correct. Nothing changes; the conflict is recorded for review. **Prefer `flag` over `supersede` whenever you are genuinely uncertain** — invalidating a memory is destructive.
- `distinct`: any meaningful difference that is NOT a contradiction — different person, time, scope, or attribute that can both be true. **When in doubt between merge and distinct, prefer distinct.**
- `unsure`: you genuinely cannot tell anything. Logged; nothing happens.

Hard rules:

1. Be conservative about destruction. A spurious merge or false supersede loses information; a missed one is fixable on a future night. Torn between `supersede` and `flag` → choose `flag`.
2. Different people, times, or aspects that do NOT contradict are `distinct`, even if the words overlap. A contradiction means they cannot both be true of the same subject.
3. `confidence` ≥ 0.8 only when you are sure; below that the apply step skips any merge/supersede regardless of decision.
4. `evidence` MUST include both memory IDs from the pair for any merge or supersede. For `supersede`, `stale_id` MUST be one of the two.
5. When merging, `merged_content` is 1–3 sentences containing every fact from either input; `merged_summary` ≤80 chars. Do NOT fold content on a `supersede` — the stale memory is wrong, not additive.
6. Operational memories (activity-log entries) are append-only — never merge or supersede them; return `distinct`.

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
    since_iso: str | None = None,
) -> CandidateSet:
    """Collect newly-created memories plus 6 revisits across three pools.

    Pools (per roadmap §3):
      A — "used today" (last_relevant_at >= midnight): pick 3
      B — age-decayed random across active memories: pick 2 weighted
          by exp(-age_days / AGE_DECAY_DAYS)
      C — uniform random across active memories: pick 1

    ``since_iso`` bounds the "new" set: memories created at or after it.
    ``run_dream_cycle`` passes the watermark of the last run, so every
    new memory is scanned exactly once regardless of when in the day it
    was created. Falls back to today's UTC midnight when unset (the old
    behaviour) — but note that on its own midnight leaves a daily blind
    spot for memories created between the run time and the next midnight,
    which is why the watermark exists.

    Memories already in the new set are excluded from the revisit pools
    so the same record isn't sampled twice.
    """
    rng = rng or random.Random()
    now = now or datetime.utcnow()
    midnight = _midnight_utc(now)
    midnight_iso = midnight.isoformat()

    # New memories created since the watermark (active only — we don't
    # dedup archived/invalidated rows).
    new_window_iso = since_iso or midnight_iso
    new_today = await store.list_memories_created_since(new_window_iso)
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
    *,
    near_dup_threshold: float = NEAR_DUP_THRESHOLD,
) -> list[NearDupPair]:
    """Pair memories at cosine ≥ ``near_dup_threshold`` for adjudication.

    Two-pass:

    1. **Within-candidate pairs** — the historical behaviour. Cheap;
       catches near-dups between today's new memories and the 6
       refresh-sampled revisits.

    2. **Nearest-neighbour expansion of today's new memories** — for
       each new-today memory, find its top-K cosine neighbours in the
       full active pool. This is the load-bearing addition: without
       it, back-catalogue dupes only get caught if the matching pair
       lands in the same narrow revisit window (probability ~7% per
       night on a 90-memory corpus), so they accumulate forever.

    Co-injected pairs are deliberately NOT excluded. The old design
    skipped any pair that had appeared together in a conversation, on
    the theory that daytime extraction already resolved it — but
    extraction only invalidates on an *explicit* in-conversation
    correction, so two contradictory memories that keep getting injected
    together and merely produce a wrong answer fell through both nets
    forever. Repeated co-injection is a reason to adjudicate, not skip.
    The enriched judge (provenance + related context) is the precision
    gate; the ``MAX_DEDUP_PAIRS`` cap bounds batch size.
    """
    cand_memories = [
        m for m in candidates.all_memories if m.embedding is not None
    ]
    if len(cand_memories) == 0:
        return []

    # The active pool backs the nearest-neighbour expansion below.
    active_pool = await store.list_memories(status="active", limit=10_000)
    active_with_embed = [
        m for m in active_pool if m.embedding is not None
    ]

    pairs_by_key: dict[tuple[str, str], NearDupPair] = {}

    def _record(a: Memory, b: Memory, sim: float) -> None:
        """Record a near-dup pair if it's novel + above threshold. Order
        the IDs so each pair is unique regardless of which side surfaced
        it first."""
        if sim < near_dup_threshold:
            return
        if a.id == b.id:
            return
        key = (a.id, b.id) if a.id < b.id else (b.id, a.id)
        if key not in pairs_by_key:
            pairs_by_key[key] = NearDupPair(
                memory_id_a=key[0],
                memory_id_b=key[1],
                cosine=float(sim),
            )

    # Pass 1: within-candidate all-pairs (cheap — ~9 memories).
    n = len(cand_memories)
    for i in range(n):
        a = cand_memories[i]
        for j in range(i + 1, n):
            b = cand_memories[j]
            sim = cosine_similarity(a.embedding, b.embedding)
            _record(a, b, sim)

    # Pass 2: nearest-neighbour expansion. For each new-today memory,
    # score against every active memory and keep top-K above threshold.
    # On a corpus of <10K memories this is fine in-process; if we ever
    # scale past that, swap in a vector index.
    for new_mem in candidates.new_today:
        if new_mem.embedding is None:
            continue
        # Score against every active memory (excluding self).
        scored: list[tuple[float, Memory]] = []
        for other in active_with_embed:
            if other.id == new_mem.id:
                continue
            sim = cosine_similarity(new_mem.embedding, other.embedding)
            if sim >= near_dup_threshold:
                scored.append((sim, other))
        # Top-K above threshold. K=10 is generous; in practice
        # most memories have 0-2 above-threshold neighbours.
        scored.sort(key=lambda t: t[0], reverse=True)
        for sim, other in scored[:10]:
            _record(new_mem, other, sim)

    # Cap by global budget so we never submit a runaway batch even on
    # a particularly noisy day.
    pairs = sorted(
        pairs_by_key.values(), key=lambda p: p.cosine, reverse=True,
    )
    if len(pairs) > MAX_DEDUP_PAIRS_DEFAULT:
        pairs = pairs[:MAX_DEDUP_PAIRS_DEFAULT]
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
    source_a: str = "(source unknown)",
    source_b: str = "(source unknown)",
    related_context: str = "(no related memories on file)",
    model: str = DEFAULT_EXTRACTION_MODEL,
    max_tokens: int = 1200,
) -> dict[str, Any]:
    """Build one Anthropic batch request for a single dedup pair.

    ``source_a`` / ``source_b`` describe each memory's provenance (who or
    what asserted it). ``related_context`` is a rendered block of other
    active memories about the same people — "what else is known" — so the
    judge can resolve contradictions the two memories alone don't settle.
    """
    user_msg = (
        f"Resolve these TWO memories against each other. Reason it through "
        f"first, then return a single `dedup_decision` tool call.\n\n"
        f"Embedding cosine: {cosine:.3f}\n\n"
        f"[Memory A]\n"
        f"id: {memory_a.id}\n"
        f"type: {memory_a.type}\n"
        f"person: {memory_a.person or '(none)'}\n"
        f"content: {memory_a.content}\n"
        f"summary: {memory_a.summary}\n"
        f"created_at: {memory_a.created_at}\n"
        f"source: {source_a}\n\n"
        f"[Memory B]\n"
        f"id: {memory_b.id}\n"
        f"type: {memory_b.type}\n"
        f"person: {memory_b.person or '(none)'}\n"
        f"content: {memory_b.content}\n"
        f"summary: {memory_b.summary}\n"
        f"created_at: {memory_b.created_at}\n"
        f"source: {source_b}\n\n"
        f"[What else is known]\n"
        f"{related_context}\n"
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


# Channels where a human is the one talking. Anything else (a scheduled
# wake/review, a trigger firing, an empty channel on an automated job)
# is treated as machine-derived provenance.
_HUMAN_CHANNELS = {"voice", "whatsapp", "signal"}


def _classify_channel(channel: str | None) -> str:
    """Describe whether a memory's source conversation was human or automated."""
    base = (channel or "").split(":", 1)[0].strip().lower()
    if base in _HUMAN_CHANNELS:
        return "stated by a person"
    return (
        "recorded by an automated process (e.g. a scheduled review "
        "reading an external source) — treat as a possibly re-derived guess"
    )


async def _source_provenance(store: MemoryStore, memory: Memory) -> str:
    """Render a one-line provenance string for a memory's source conversation."""
    conv_id = memory.source_conversation
    if not conv_id:
        return "(source unknown)"
    conv = await store.get_conversation(conv_id)
    if conv is None:
        return "(source conversation no longer on file)"
    origin = _classify_channel(conv.channel)
    chan = conv.channel or "automated review"
    summ = (conv.summary or "").strip().replace("\n", " ")
    if len(summ) > 200:
        summ = summ[:200] + "…"
    tail = f' — "{summ}"' if summ else ""
    return f"{chan}; {origin}{tail}"


async def _related_context(
    store: MemoryStore,
    memory_a: Memory,
    memory_b: Memory,
    *,
    limit: int = 5,
) -> str:
    """Render other active memories about the same people — "what else is known".

    Pulls active memories whose primary ``person`` matches anyone named in
    either memory, excluding the pair itself. This is the block that lets
    the judge resolve a contradiction the two memories alone can't (e.g.
    a separate "Zara is 2, not in preschool yet" memory deciding who is
    actually graduating).
    """
    persons: set[str] = set()
    for m in (memory_a, memory_b):
        if m.person:
            persons.add(m.person)
        for p in m.people or []:
            if p:
                persons.add(p)

    seen = {memory_a.id, memory_b.id}
    related: list[Memory] = []
    for person in sorted(persons):
        for m in await store.list_memories(
            status="active", person=person, limit=20,
        ):
            if m.id in seen:
                continue
            seen.add(m.id)
            related.append(m)

    related.sort(key=lambda m: m.last_relevant_at, reverse=True)
    related = related[:limit]
    if not related:
        return "(no related memories on file)"

    lines: list[str] = []
    for m in related:
        who = m.person or (", ".join(m.people) if m.people else "—")
        lines.append(
            f"- [{who}] {m.summary} (id {m.id[:8]}, created {m.created_at[:10]})"
        )
    return "\n".join(lines)


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
        mem_a = by_id[p.memory_id_a]
        mem_b = by_id[p.memory_id_b]
        # Enrich each request: where each fact came from, and what else
        # the store knows about the same people. The judge weighs these.
        source_a = await _source_provenance(store, mem_a)
        source_b = await _source_provenance(store, mem_b)
        related_context = await _related_context(store, mem_a, mem_b)
        requests.append(
            _build_dedup_request(
                custom_id=custom_id,
                memory_a=mem_a,
                memory_b=mem_b,
                cosine=p.cosine,
                source_a=source_a,
                source_b=source_b,
                related_context=related_context,
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
        stale_id=(
            str(payload["stale_id"])
            if payload.get("stale_id") else None
        ),
        reasoning=str(payload.get("reasoning") or ""),
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
        # A flag is "these conflict but I'm not sure" — recorded for
        # review, never mutates. No confidence gate: it's already the
        # uncertain path.
        if d.decision == "flag":
            result.flagged_contradictions += 1
            continue
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
        if d.decision not in {"merge_into_a", "merge_into_b", "supersede"}:
            continue
        # Defence against confabulation: evidence must cite both members
        # of the pair for any destructive op (merge or supersede).
        if not d.evidence or pair.memory_id_a not in d.evidence \
                or pair.memory_id_b not in d.evidence:
            logger.warning(
                "Dream decision %s evidence does not cite both memories; "
                "skipping (defence against confabulation)",
                d.custom_id,
            )
            continue

        # --- Contradiction: invalidate the stale memory, keep the other.
        if d.decision == "supersede":
            stale = d.stale_id
            if stale not in {pair.memory_id_a, pair.memory_id_b}:
                logger.warning(
                    "Dream supersede %s: stale_id %r is not one of the "
                    "pair; skipping",
                    d.custom_id, stale,
                )
                continue
            survivor = (
                pair.memory_id_b if stale == pair.memory_id_a
                else pair.memory_id_a
            )
            if audit_only:
                result.applied_supersessions += 1
                continue
            await _apply_supersede(
                store,
                stale_id=stale,
                survivor_id=survivor,
                batch_id=batch_id,
            )
            result.applied_supersessions += 1
            continue

        # --- Merge: same fact, different words.
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
        "Dream apply (audit_only=%s, batch=%s): merges=%d, supersessions=%d, "
        "flagged=%d, skipped_low_conf=%d, skipped_distinct_or_unsure=%d",
        audit_only, batch_id,
        result.applied_merges,
        result.applied_supersessions,
        result.flagged_contradictions,
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


async def _apply_supersede(
    store: MemoryStore,
    *,
    stale_id: str,
    survivor_id: str,
    batch_id: str,
) -> None:
    """Resolve a contradiction: invalidate the stale memory, keep the other.

    Unlike a merge, no content is folded — the stale memory is *wrong*,
    not an additive restatement. Soft-delete only: the stale row is
    marked ``invalidated`` with ``superseded_by=survivor`` and both rows
    are stamped ``consolidated_by=batch_id`` for auditability/undo.
    """
    stale = await store.get_memory_no_touch(stale_id)
    survivor = await store.get_memory_no_touch(survivor_id)
    if stale is None or survivor is None:
        logger.warning(
            "Supersede skipped: stale=%s survivor=%s — at least one missing",
            stale_id, survivor_id,
        )
        return
    if stale.status != "active":
        # Already retired (e.g. invalidated earlier the same night) —
        # nothing to do; avoid clobbering an existing superseded_by chain.
        return
    await store.invalidate_memory(
        stale_id,
        invalidated_by=batch_id,
        superseded_by=survivor_id,
    )
    await store.set_dream_audit_fields(stale_id, consolidated_by=batch_id)
    await store.set_dream_audit_fields(survivor_id, consolidated_by=batch_id)
    logger.info(
        "Dream supersede (batch=%s): invalidated %s, kept %s",
        batch_id, stale_id, survivor_id,
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
            if d.reasoning:
                reasoning = d.reasoning.strip().replace("\n", " ")
                if len(reasoning) > 240:
                    reasoning = reasoning[:240] + "…"
                lines.append(f"  - reasoning: {reasoning}")
    lines.append("")

    lines.append("## Cost")
    if cost_usd is None:
        lines.append("- (recorded by DreamPoller on batch end)")
    else:
        lines.append(f"- ${cost_usd:.4f} ({DEFAULT_EXTRACTION_MODEL} batch)")
    lines.append("")

    lines.append("## Applied")
    flagged = (
        sum(1 for d in decisions if d.decision == "flag")
        if decisions else 0
    )
    if decisions is None:
        lines.append("- (pending poll)")
    else:
        merges = sum(
            1 for d in decisions
            if d.decision in {"merge_into_a", "merge_into_b"}
            and d.confidence >= DEDUP_CONFIDENCE_THRESHOLD
        )
        supersedes = sum(
            1 for d in decisions
            if d.decision == "supersede"
            and d.confidence >= DEDUP_CONFIDENCE_THRESHOLD
        )
        verb = "would apply" if audit_only else "applied"
        lines.append(f"- {merges} merges {verb}")
        lines.append(f"- {supersedes} contradiction supersessions {verb}")
        # Flags always surface — they are review items, not mutations.
        lines.append(f"- {flagged} contradictions flagged for review")
        if audit_only:
            lines.append("  (audit-only mode: store left untouched)")
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
    near_dup_threshold: float = NEAR_DUP_THRESHOLD,
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

    # Watermark: scan memories created since the last run, not since
    # midnight. The old "since midnight" window — combined with a run at
    # 10:00 UTC — never scanned memories created between the run and the
    # next midnight, so cross-conversation dupes/contradictions in that
    # ~14h band accumulated forever. On the first run (no watermark) look
    # back 24h to avoid scanning the entire history in one batch.
    watermark = await store.get_dream_state(DREAM_WATERMARK_KEY)
    since_iso = watermark or (now - timedelta(hours=24)).isoformat()

    # Phase 1 — gather and cluster (no model calls).
    candidates = await gather_candidates(
        store, now=now, rng=rng, since_iso=since_iso,
    )
    clusters = await cluster_candidates(candidates)
    pairs = await find_near_duplicates(
        store, candidates, near_dup_threshold=near_dup_threshold,
    )
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

    # Advance the watermark only after the cycle's work is durably
    # queued (batch submitted) and logged. If anything above threw, we
    # keep the old watermark so the next run re-covers this window.
    await store.set_dream_state(DREAM_WATERMARK_KEY, now.isoformat())

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
