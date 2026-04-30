"""Memory injection at conversation start.

Calls the shared search backend with person + utterance context and
formats results into a system prompt injection block with memory IDs.

Usage:
    from boxbot.memory.retrieval import inject_memories

    block = await inject_memories(store, person="Jacob", utterance="What's for dinner?")
    # Returns formatted string for system prompt injection
"""

from __future__ import annotations

import logging
from datetime import datetime

from boxbot.memory.search import (
    DEFAULT_CONVERSATION_RESULTS,
    SearchCandidate,
    hybrid_search,
)
from boxbot.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Token budget limits (memory count, not actual tokens)
MAX_PERSON_FACTS = 10
MAX_TOPIC_FACTS = 5
MAX_CONVERSATIONS = 3


async def inject_memories(
    store: MemoryStore,
    *,
    person: str | None = None,
    utterance: str | None = None,
    max_facts: int = 15,
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
        max_facts: Maximum fact memories to inject.
        max_conversations: Maximum conversation summaries to inject.

    Returns:
        ``(block, memory_ids)``. ``block`` is the formatted injection
        string (empty if no results). ``memory_ids`` is the list of
        IDs (memories + conversations) actually surfaced.
    """
    if not person and not utterance:
        return "", []

    # Build the search query from available context
    query_parts = []
    if person:
        query_parts.append(person)
    if utterance:
        query_parts.append(utterance)
    query = " ".join(query_parts)

    candidates = await hybrid_search(
        store,
        query,
        person=person,
        include_conversations=True,
        include_archived=False,
        memory_limit=max_facts * 2,
        conversation_limit=max_conversations * 2,
    )

    if not candidates:
        return "", []

    # Split into facts and conversations
    fact_candidates = [c for c in candidates if c.source == "memory"]
    conv_candidates = [c for c in candidates if c.source == "conversation"]

    # Update relevance timestamps for injected memories
    for c in fact_candidates[:max_facts]:
        await store.update_memory_relevance(c.id)
    # (conversations don't have relevance tracking in the same way)

    # Format injection block
    lines: list[str] = []
    surfaced_ids: list[str] = []

    if fact_candidates:
        lines.append("[Active Memories]")
        for c in fact_candidates[:max_facts]:
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
        "Injected %d facts + %d conversations for person=%s",
        min(len(fact_candidates), max_facts),
        min(len(conv_candidates), max_conversations),
        person,
    )
    return result, surfaced_ids
