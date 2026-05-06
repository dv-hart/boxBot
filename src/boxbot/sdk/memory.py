"""Memory operations — save, search, and delete memories.

Shares the same backend as the search_memory core tool.

Usage:
    from boxbot_sdk import memory

    memory_id = memory.save(
        content="Jacob is allergic to peanuts",
        memory_type="person",
        people=["Jacob"],
        tags=["health", "allergy"],
    )

    results = memory.search("allergies", people=["Jacob"])
    for m in results:
        print(f"{m.content} (importance: {m.importance})")

    memory.delete(memory_id="abc-123")

All write calls (save, delete) wait for the main process to acknowledge
and raise ``MemoryError`` if the dispatcher rejects them. This is on
purpose — silent failures previously let writes look like they had
succeeded when no handler was wired up.
"""

from __future__ import annotations

from typing import Any

from . import _transport, _validators as v


class MemoryRecord:
    """A single memory record returned from search."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def id(self) -> str:
        return self._data.get("id", "")

    @property
    def content(self) -> str:
        return self._data.get("content", "")

    @property
    def memory_type(self) -> str:
        return self._data.get("type", "")

    @property
    def people(self) -> list[str]:
        return self._data.get("people", [])

    @property
    def tags(self) -> list[str]:
        return self._data.get("tags", [])

    @property
    def importance(self) -> float:
        return self._data.get("importance", 0.5)

    @property
    def created_at(self) -> str:
        return self._data.get("created_at", "")

    def __repr__(self) -> str:
        return f"MemoryRecord(id={self.id!r}, content={self.content!r})"


def _raise_on_error(response: dict, op: str) -> None:
    if response.get("status") != "ok":
        raise MemoryError(
            response.get("message")
            or response.get("error")
            or f"{op} failed"
        )


def save(content: str, *,
         memory_type: str = "household",
         summary: str | None = None,
         person: str | None = None,
         people: list[str] | None = None,
         tags: list[str] | None = None,
         importance: float | None = None) -> str:
    """Save a memory; return the new memory's ID.

    Args:
        content: The memory content text.
        memory_type: person, household, methodology, operational.
        summary: One-line summary used for retrieval injection. Auto-
            derived from ``content`` when omitted.
        person: Primary person this memory is about. Defaults to the
            first entry of ``people`` if not provided.
        people: People associated with this memory.
        tags: Tags for categorization.
        importance: Importance score (0.0 to 1.0). Currently informational
            — not yet persisted by the store.

    Raises:
        MemoryError: if the main process rejects the call (validation,
            store error, missing handler).
    """
    v.require_str(content, "content")
    v.validate_one_of(memory_type, "memory_type", v.VALID_MEMORY_TYPES)

    payload: dict[str, Any] = {
        "content": content,
        "type": memory_type,
    }
    if summary is not None:
        payload["summary"] = v.require_str(summary, "summary")
    if person is not None:
        payload["person"] = v.require_str(person, "person")
    if people is not None:
        payload["people"] = v.require_list(people, "people")
    if tags is not None:
        payload["tags"] = v.require_list(tags, "tags")
    if importance is not None:
        payload["importance"] = v.require_float(
            importance, "importance", min_val=0.0, max_val=1.0
        )

    response = _transport.request("memory.save", payload, timeout=30)
    _raise_on_error(response, "memory.save")
    return response.get("id", "")


def search(query: str, *,
           people: list[str] | None = None,
           memory_type: str | None = None,
           tags: list[str] | None = None,
           limit: int = 10) -> list[MemoryRecord]:
    """Search memories.

    Args:
        query: Search query text.
        people: Filter by associated people.
        memory_type: Filter by memory type.
        tags: Filter by tags.
        limit: Maximum results to return.

    Returns:
        List of MemoryRecord objects.
    """
    v.require_str(query, "query")
    v.require_int(limit, "limit", min_val=1, max_val=100)

    payload: dict[str, Any] = {
        "query": query,
        "limit": limit,
    }
    if people is not None:
        v.require_list(people, "people")
        payload["people"] = people
    if memory_type is not None:
        v.validate_one_of(memory_type, "memory_type", v.VALID_MEMORY_TYPES)
        payload["type"] = memory_type
    if tags is not None:
        v.require_list(tags, "tags")
        payload["tags"] = tags

    response = _transport.request("memory.search", payload, timeout=30)
    _raise_on_error(response, "memory.search")
    results = response.get("results", [])
    return [MemoryRecord(r) for r in results]


def delete(memory_id: str) -> None:
    """Delete (soft-delete) a memory.

    Args:
        memory_id: The ID of the memory to delete.

    Raises:
        MemoryError: if the main process rejects the call.
    """
    v.require_str(memory_id, "memory_id")
    response = _transport.request("memory.delete", {"id": memory_id}, timeout=30)
    _raise_on_error(response, "memory.delete")
