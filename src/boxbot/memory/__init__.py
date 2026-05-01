"""Memory system — persistent, contextual recall across conversations.

Three stores:
- System memory (data/memory/system.md) — always in system prompt
- Fact memories (memory.db → memories table) — contextual retrieval
- Conversation log (memory.db → conversations table) — rolling window

Four memory types: person, household, methodology, operational.

The search_memory tool, boxbot_sdk.memory.search(), and injection system
all share the same search backend (search.py).
"""

from boxbot.memory.dream import (
    DreamApplyResult,
    apply_dream_result,
    cluster_candidates,
    find_near_duplicates,
    gather_candidates,
    run_dream_cycle,
    submit_dream_batch,
)
from boxbot.memory.embeddings import cosine_similarity, embed, embed_batch
from boxbot.memory.extraction import (
    ExtractionResult,
    parse_extraction_result,
    process_extraction_result,
    record_extraction_cost,
    submit_extraction_batch,
)
from boxbot.memory.maintenance import run_maintenance
from boxbot.memory.retrieval import inject_memories
from boxbot.memory.search import search_memories
from boxbot.memory.store import Conversation, Memory, MemoryStore, PendingExtraction

__all__ = [
    # Store
    "MemoryStore",
    "Memory",
    "Conversation",
    "PendingExtraction",
    # Embeddings
    "embed",
    "embed_batch",
    "cosine_similarity",
    # Search
    "search_memories",
    # Injection
    "inject_memories",
    # Extraction
    "submit_extraction_batch",
    "parse_extraction_result",
    "process_extraction_result",
    "record_extraction_cost",
    "ExtractionResult",
    # Maintenance
    "run_maintenance",
    # Dream phase
    "run_dream_cycle",
    "gather_candidates",
    "cluster_candidates",
    "find_near_duplicates",
    "submit_dream_batch",
    "apply_dream_result",
    "DreamApplyResult",
]
