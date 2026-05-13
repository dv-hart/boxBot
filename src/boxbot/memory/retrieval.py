"""Memory injection at conversation start.

Calls the shared search backend with person + utterance context and
formats results into a system prompt injection block.

Type-weighted: hybrid_search returns a single ranking blended across
all memory types, which means a cluster of near-duplicate operational
entries can crowd out single-instance methodology/person facts that
are actually load-bearing. We bucket results by type and apply a
per-type budget, so each category gets guaranteed slots regardless
of how many noisy operational siblings are in the pool.

Usage:
    from boxbot.memory.retrieval import inject_memories

    block = await inject_memories(store, person="Jacob", utterance="What's for dinner?")
    # Returns formatted string for system prompt injection
"""

from __future__ import annotations

import logging
from collections import defaultdict

from boxbot.memory.search import (
    DEFAULT_CONVERSATION_RESULTS,
    SearchCandidate,
    hybrid_search,
)
from boxbot.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Per-type injection budgets. The total fact count is the sum of all
# budgets (currently 13). Operational is 0 because lifecycle step 7
# is dropping the type — but until then we still want to suppress
# any stray operational rows from crowding the block.
TYPE_BUDGETS: dict[str, int] = {
    "person": 6,
    "methodology": 4,
    "household": 3,
    "operational": 0,
}
MAX_CONVERSATIONS = 3


async def inject_memories(
    store: MemoryStore,
    *,
    person: str | None = None,
    utterance: str | None = None,
    type_budgets: dict[str, int] | None = None,
    max_conversations: int = MAX_CONVERSATIONS,
) -> tuple[str, list[str]]:
    """Build the memory injection block for conversation start.

    Searches for relevant memories based on who is speaking and what
    they said. Returns a formatted text block (for system-prompt
    injection) AND the list of memory IDs that were surfaced — the
    caller stores those on the Conversation so post-conversation
    extraction knows which memories the model could see, and therefore
    which ones it's qualified to invalidate.

    Args:
        store: The MemoryStore instance.
        person: Name of the identified speaker(s).
        utterance: First utterance text (or task description for triggers).
        type_budgets: Per-type slot budgets (defaults to ``TYPE_BUDGETS``).
            Use to override for tests or future tuning.
        max_conversations: Maximum conversation summaries to inject.

    Returns:
        ``(block, memory_ids)``. ``block`` is the formatted injection
        string (empty if no results). ``memory_ids`` is the list of
        IDs (memories + conversations) actually surfaced.
    """
    if not person and not utterance:
        return "", []

    budgets = type_budgets if type_budgets is not None else TYPE_BUDGETS
    total_budget = sum(budgets.values())

    # Build the search query from available context
    query_parts = []
    if person:
        query_parts.append(person)
    if utterance:
        query_parts.append(utterance)
    query = " ".join(query_parts)

    # Over-fetch by 2x per category so per-type budgets have real
    # ranking room. hybrid_search returns mixed types; we bucket below.
    candidates = await hybrid_search(
        store,
        query,
        person=person,
        include_conversations=True,
        include_archived=False,
        memory_limit=total_budget * 3,
        conversation_limit=max_conversations * 2,
    )

    if not candidates:
        return "", []

    fact_candidates = [c for c in candidates if c.source == "memory"]
    conv_candidates = [c for c in candidates if c.source == "conversation"]

    # Bucket facts by type. Within a bucket, candidates retain their
    # hybrid-search ranking (hybrid_search returns sorted output).
    by_type: dict[str, list[SearchCandidate]] = defaultdict(list)
    for c in fact_candidates:
        by_type[c.type].append(c)

    # Apply per-type budget. Skip types with zero budget.
    chosen_facts: list[SearchCandidate] = []
    for mtype, budget in budgets.items():
        if budget <= 0:
            continue
        chosen_facts.extend(by_type.get(mtype, [])[:budget])

    # Update relevance timestamps for injected memories
    for c in chosen_facts:
        await store.update_memory_relevance(c.id)
    # (conversations don't have relevance tracking in the same way)

    # Format injection block
    lines: list[str] = []
    surfaced_ids: list[str] = []

    if chosen_facts:
        lines.append("[Active Memories]")
        for c in chosen_facts:
            person_label = f"/{c.person}" if c.person else ""
            lines.append(f"#{c.id[:8]} ({c.type}{person_label}): {c.summary}")
            surfaced_ids.append(c.id)
        lines.append("")

    if conv_candidates:
        lines.append("[Recent Conversations]")
        for c in conv_candidates[:max_conversations]:
            # Format: #id (date, participants, channel): summary
            started = c.metadata.get("started_at", "")[:10]
            participants = c.metadata.get("participants", [])
            channel = c.metadata.get("channel", "")
            parts_str = ", ".join(participants) if participants else "unknown"
            lines.append(
                f"#{c.id[:8]} ({started}, {parts_str}, {channel}): {c.summary}"
            )
            surfaced_ids.append(c.id)
        lines.append("")

    result = "\n".join(lines)
    logger.debug(
        "Injected %d facts (budgets=%s) + %d conversations for person=%s",
        len(chosen_facts), budgets,
        min(len(conv_candidates), max_conversations),
        person,
    )
    return result, surfaced_ids
