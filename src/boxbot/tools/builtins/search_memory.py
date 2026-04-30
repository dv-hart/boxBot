"""search_memory tool — search, summarize, or retrieve stored memories.

Routes to the shared memory search backend (boxbot.memory.search). Three
modes: lookup (ranked results), summary (synthesized answer), get (full
record by ID). Shares backend with the injection system and SDK.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from boxbot.tools.base import Tool

logger = logging.getLogger(__name__)


class SearchMemoryTool(Tool):
    """Search, summarize, or retrieve stored memories."""

    name = "search_memory"
    description = (
        "Search, summarize, or retrieve stored memories. Modes: "
        "'lookup' returns ranked fact memories and conversation matches, "
        "'summary' synthesizes an answer from relevant memories, "
        "'get' retrieves a full memory record by ID, "
        "'transcript' recovers raw conversation text — pass conversation_id "
        "for a single conversation, or query to substring-search recent "
        "transcripts (last 14 days). Use 'transcript' when memory is "
        "thin and you need to reconstruct what was actually said."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["lookup", "summary", "get", "transcript"],
                "description": (
                    "Search mode: 'lookup' for ranked results, "
                    "'summary' for a synthesized answer, "
                    "'get' for full record by memory ID, "
                    "'transcript' for raw conversation text."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Search query text. Required for 'lookup' and 'summary' "
                    "modes; in 'transcript' mode used to substring-search "
                    "recent transcripts when conversation_id is not given."
                ),
            },
            "memory_id": {
                "type": "string",
                "description": "Memory ID. Required for 'get' mode.",
            },
            "conversation_id": {
                "type": "string",
                "description": (
                    "Conversation ID for 'transcript' mode. If provided, "
                    "returns the full transcript for that conversation."
                ),
            },
            "types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["person", "household", "methodology", "operational"],
                },
                "description": "Optional: filter by memory types.",
            },
            "person": {
                "type": "string",
                "description": "Optional: filter by person name.",
            },
            "include_conversations": {
                "type": "boolean",
                "description": (
                    "Include conversation log in results. Default: true."
                ),
            },
            "include_archived": {
                "type": "boolean",
                "description": (
                    "Include archived memories. Default: false."
                ),
            },
        },
        "required": ["mode"],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: Any) -> str:
        mode: str = kwargs["mode"]
        query: str | None = kwargs.get("query")
        memory_id: str | None = kwargs.get("memory_id")
        conversation_id: str | None = kwargs.get("conversation_id")
        types: list[str] | None = kwargs.get("types")
        person: str | None = kwargs.get("person")
        include_conversations: bool = kwargs.get("include_conversations", True)
        include_archived: bool = kwargs.get("include_archived", False)

        logger.info(
            "search_memory: mode=%s, query=%s, memory_id=%s, conv_id=%s",
            mode,
            query[:50] if query else None,
            memory_id,
            conversation_id,
        )

        try:
            from boxbot.memory.search import search_memories
            from boxbot.memory.store import MemoryStore

            # Get or create a memory store instance
            store = await _get_memory_store()

            result = await search_memories(
                store,
                mode=mode,
                query=query,
                memory_id=memory_id,
                conversation_id=conversation_id,
                types=types,
                person=person,
                include_conversations=include_conversations,
                include_archived=include_archived,
            )

            return json.dumps(result)

        except ImportError:
            logger.warning("Memory search backend not available")
            return json.dumps({
                "error": "Memory search backend not available.",
            })
        except ValueError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            logger.exception("search_memory error")
            return json.dumps({"error": f"Search failed: {e}"})


# ---------------------------------------------------------------------------
# Memory store singleton for the tool
# ---------------------------------------------------------------------------

_store: Any = None


async def _get_memory_store() -> Any:
    """Get or create a MemoryStore singleton for tool use."""
    global _store
    if _store is None:
        from boxbot.memory.store import MemoryStore

        _store = MemoryStore()
        await _store.initialize()
    return _store
