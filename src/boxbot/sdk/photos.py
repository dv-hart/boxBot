"""Photo management — search, tag, slideshow, and lifecycle operations.

Photo search shares the same backend as the search_photos core tool.

Usage:
    from boxbot_sdk import photos

    results = photos.search(query="beach sunset", tags=["vacation"])
    for p in results:
        print(f"{p.id}: {p.description}")

    photos.set_tags("photo_123", tags=["family", "beach"])
    photos.add_to_slideshow("photo_123")
    photos.delete("photo_456")
"""

from __future__ import annotations

from typing import Any

from . import _transport, _validators as v


class PhotoRecord:
    """A photo record returned from search or get."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def id(self) -> str:
        return self._data.get("id", "")

    @property
    def description(self) -> str:
        return self._data.get("description", "")

    @property
    def tags(self) -> list[str]:
        return self._data.get("tags", [])

    @property
    def people(self) -> list[str]:
        return self._data.get("people", [])

    @property
    def file_path(self) -> str:
        return self._data.get("file_path", "")

    @property
    def created_at(self) -> str:
        return self._data.get("created_at", "")

    @property
    def in_slideshow(self) -> bool:
        return self._data.get("in_slideshow", False)

    def __repr__(self) -> str:
        return f"PhotoRecord(id={self.id!r}, description={self.description!r})"


class StorageInfo:
    """Photo storage quota information."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def used_gb(self) -> float:
        return self._data.get("used_gb", 0.0)

    @property
    def quota_gb(self) -> float:
        return self._data.get("quota_gb", 0.0)

    @property
    def used_percent(self) -> float:
        return self._data.get("used_percent", 0.0)

    @property
    def photo_count(self) -> int:
        return self._data.get("photo_count", 0)

    def __repr__(self) -> str:
        return (f"StorageInfo({self.used_gb:.1f}GB / {self.quota_gb:.1f}GB, "
                f"{self.used_percent:.0f}%)")


# --- Search and retrieval ---

def search(query: str | None = None, *,
           tags: list[str] | None = None,
           people: list[str] | None = None,
           limit: int = 10) -> list[PhotoRecord]:
    """Search photos.

    Args:
        query: Text search query.
        tags: Filter by tags.
        people: Filter by people in photo.
        limit: Maximum results.

    Returns:
        List of PhotoRecord objects.
    """
    payload: dict[str, Any] = {"limit": limit}
    if query is not None:
        v.require_str(query, "query")
        payload["query"] = query
    if tags is not None:
        v.require_list(tags, "tags")
        payload["tags"] = tags
    if people is not None:
        v.require_list(people, "people")
        payload["people"] = people
    v.require_int(limit, "limit", min_val=1, max_val=100)

    _transport.emit_action("photos.search", payload)
    response = _transport.collect_response(timeout=30)
    results = response.get("results", [])
    return [PhotoRecord(r) for r in results]


def get(photo_id: str) -> PhotoRecord:
    """Get full details for a photo.

    Args:
        photo_id: Photo ID.

    Returns:
        PhotoRecord with full details.
    """
    v.require_str(photo_id, "photo_id")
    _transport.emit_action("photos.get", {"id": photo_id})
    response = _transport.collect_response(timeout=30)
    return PhotoRecord(response)


# --- Metadata updates ---

def update(photo_id: str, *, description: str) -> None:
    """Update a photo's description.

    Args:
        photo_id: Photo ID.
        description: New description text.
    """
    v.require_str(photo_id, "photo_id")
    v.require_str(description, "description")
    _transport.emit_action("photos.update", {
        "id": photo_id,
        "description": description,
    })


def set_tags(photo_id: str, *, tags: list[str]) -> None:
    """Set tags on a photo (replaces existing tags).

    Args:
        photo_id: Photo ID.
        tags: List of tag strings.
    """
    v.require_str(photo_id, "photo_id")
    v.require_list(tags, "tags")
    _transport.emit_action("photos.set_tags", {
        "id": photo_id,
        "tags": tags,
    })


def set_person(photo_id: str, *, person_index: int, name: str) -> None:
    """Tag a person in a photo.

    Args:
        photo_id: Photo ID.
        person_index: Index in the photo's people list (0-based).
        name: Person name to assign.
    """
    v.require_str(photo_id, "photo_id")
    v.require_int(person_index, "person_index", min_val=0)
    v.require_str(name, "name")
    _transport.emit_action("photos.set_person", {
        "id": photo_id,
        "person_index": person_index,
        "name": name,
    })


# --- Slideshow management ---

def add_to_slideshow(photo_id: str) -> None:
    """Add a photo to the slideshow rotation.

    Args:
        photo_id: Photo ID.
    """
    v.require_str(photo_id, "photo_id")
    _transport.emit_action("photos.add_to_slideshow", {"id": photo_id})


def remove_from_slideshow(photo_id: str) -> None:
    """Remove a photo from the slideshow rotation.

    Args:
        photo_id: Photo ID.
    """
    v.require_str(photo_id, "photo_id")
    _transport.emit_action("photos.remove_from_slideshow", {"id": photo_id})


# --- Tag library management ---

def merge_tags(source: str, *, into: str) -> None:
    """Merge one tag into another (consolidate synonyms).

    Args:
        source: Tag to merge from (will be removed).
        into: Tag to merge into (will be kept).
    """
    v.require_str(source, "source tag")
    v.require_str(into, "target tag")
    _transport.emit_action("photos.merge_tags", {
        "source": source,
        "into": into,
    })


def rename_tag(old: str, *, to: str) -> None:
    """Rename a tag (fix typos).

    Args:
        old: Current tag name.
        to: New tag name.
    """
    v.require_str(old, "old tag name")
    v.require_str(to, "new tag name")
    _transport.emit_action("photos.rename_tag", {
        "old": old,
        "new": to,
    })


def delete_tag(tag: str) -> None:
    """Delete a tag from the library.

    Args:
        tag: Tag name to delete.
    """
    v.require_str(tag, "tag")
    _transport.emit_action("photos.delete_tag", {"tag": tag})


# --- Soft delete / restore ---

def delete(photo_id: str) -> None:
    """Soft-delete a photo (30-day retention, restorable).

    Args:
        photo_id: Photo ID.
    """
    v.require_str(photo_id, "photo_id")
    _transport.emit_action("photos.delete", {"id": photo_id})


def restore(photo_id: str) -> None:
    """Restore a soft-deleted photo.

    Args:
        photo_id: Photo ID.
    """
    v.require_str(photo_id, "photo_id")
    _transport.emit_action("photos.restore", {"id": photo_id})


def list_deleted() -> list[PhotoRecord]:
    """List soft-deleted photos.

    Returns:
        List of PhotoRecord objects that are soft-deleted.
    """
    _transport.emit_action("photos.list_deleted", {})
    response = _transport.collect_response(timeout=30)
    results = response.get("results", [])
    return [PhotoRecord(r) for r in results]


# --- Storage info ---

def storage_info() -> StorageInfo:
    """Get photo storage quota information.

    Returns:
        StorageInfo with usage and quota details.
    """
    _transport.emit_action("photos.storage_info", {})
    response = _transport.collect_response(timeout=30)
    return StorageInfo(response)
