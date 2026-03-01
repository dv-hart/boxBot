"""Memory system — persistent, contextual recall across conversations.

Three stores:
- System memory (data/memory/system.md) — always in system prompt
- Fact memories (memory.db → memories table) — contextual retrieval
- Conversation log (memory.db → conversations table) — rolling window

Four memory types: person, household, methodology, operational.

The search_memory tool, boxbot_sdk.memory.search(), and injection system
all share the same search backend (search.py).
"""

from boxbot.memory.embeddings import cosine_similarity, embed, embed_batch
from boxbot.memory.extraction import (
    ExtractionResult,
    extract_memories,
    process_extraction_result,
)
from boxbot.memory.maintenance import run_maintenance
from boxbot.memory.retrieval import inject_memories
from boxbot.memory.search import search_memories
from boxbot.memory.store import Conversation, Memory, MemoryStore

__all__ = [
    # Store
    "MemoryStore",
    "Memory",
    "Conversation",
    # Embeddings
    "embed",
    "embed_batch",
    "cosine_similarity",
    # Search
    "search_memories",
    # Injection
    "inject_memories",
    # Extraction
    "extract_memories",
    "process_extraction_result",
    "ExtractionResult",
    # Maintenance
    "run_maintenance",
]
