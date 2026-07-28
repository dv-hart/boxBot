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
        "Search, summarize, or retrieve memories. Modes:\n"
        "- lookup: ranked fact memories + conversation matches\n"
        "- summary: one synthesized answer from relevant memories\n"
        "- get: one full record by memory_id\n"
        "- transcript: raw conversation text. Pass conversation_id for "
        "one thread, or query to substring-search the last 14 days. Use "
        "when memory is thin and you need what was actually said."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["lookup", "summary", "get", "transcript"],
                "description": "See tool description.",
            },
            "query": {
                "type": "string",
                "description": (
                    "Required for 'lookup' and 'summary'. In 'transcript', "
                    "substring-searches recent transcripts when "
                    "conversation_id is absent."
                ),
            },
            "memory_id": {
                "type": "string",
                "description": "Required for 'get'.",
            },
            "conversation_id": {
                "type": "string",
                "description": "'transcript' only. Returns that full transcript.",
            },
            "types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["person", "household", "methodology"],
                },
                "description": "Filter by memory type.",
            },
            "person": {
                "type": "string",
                "description": "Filter by person name.",
            },
            "include_conversations": {
                "type": "boolean",
                "description": "Include the conversation log. Default true.",
            },
            "include_archived": {
                "type": "boolean",
                "description": "Include archived memories. Default false.",
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
