"""Photo management — search, tag, slideshow, and lifecycle operations.

Photo search shares the same backend as the search_photos core tool.

Usage:
    from boxbot_sdk import photos

    results = photos.search(query="Zara at the beach")
    for p in results:
        print(p)                      # id, description, people + confidence
    photos.search(query="Zara", show=True)   # find + put on screen

    photos.set_people("photo_123", names=["Zara"])   # human-confirmed

    photos.set_tags("photo_123", tags=["family", "beach"])
    photos.add_to_slideshow("photo_123")
    photos.delete("photo_456")
"""

from __future__ import annotations

from typing import Any

from . import _transport, _validators as v


class PhotosError(_transport.ActionError):
    """Raised when a photo operation fails on the main process side.

    Subclasses ``bb.ActionError`` so a generic ``except bb.ActionError``
    catches photo failures too.
    """


def _check(resp: dict[str, Any]) -> dict[str, Any]:
    if isinstance(resp, dict) and resp.get("status") == "error":
        raise PhotosError(
            resp.get("error", "photo operation failed"), response=resp
        )
    return resp


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
    def people(self) -> list[dict[str, Any]]:
        """``[{name, confidence}]`` — confirmed | certain | likely."""
        return self._data.get("people", [])

    @property
    def names(self) -> list[str]:
        """Names only. Check ``people`` for confidence before asserting."""
        return [p.get("name", "") for p in self.people]

    @property
    def unsure(self) -> list[Any]:
        """Weak tagger guesses — not people until a human confirms."""
        return self._data.get("unsure", [])

    @property
    def source(self) -> str:
        return self._data.get("source", "")

    @property
    def sender(self) -> str | None:
        return self._data.get("sender")

    @property
    def score(self) -> float | None:
        return self._data.get("score")

    @property
    def file_path(self) -> str:
        return self._data.get("file_path", "")

    @property
    def created_at(self) -> str:
        return self._data.get("created_at") or self._data.get("created", "")

    @property
    def in_slideshow(self) -> bool:
        return self._data.get("in_slideshow", False)

    def __repr__(self) -> str:
        who = [
            p.get("name", "") if p.get("confidence") in ("confirmed", "certain")
            else f"{p.get('name', '')} ({p.get('confidence')})"
            for p in self.people
        ]
        return (
            f"PhotoRecord(id={self.id!r}, description={self.description!r}, "
            f"people={who!r}, tags={self.tags!r}, "
            f"created={self.created_at[:10]!r})"
        )


class PhotoResults(list):
    """Search results: a list of ``PhotoRecord`` plus search metadata.

    ``hint`` — set when nothing matched: which filter to drop or how to
    rephrase. ``shown`` — photos put on screen (``show=True`` only).
    """

    hint: str | None = None
    shown: int | None = None


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
           date_from: str | None = None,
           date_to: str | None = None,
           source: str | None = None,
           in_slideshow: bool | None = None,
           limit: int | None = None,
           show: bool = False) -> PhotoResults:
    """Search photos. Same backend and defaults as the search_photos tool.

    Args:
        query: What to find — a person's name works ("Zara", "Zara at
            the beach"). None = newest first.
        tags: Hard filter; photo must carry all of them.
        people: Names; soft match across people labels, tags, description.
        date_from / date_to: ISO dates.
        source: "whatsapp", "signal", "camera", "upload".
        in_slideshow: Filter by slideshow membership.
        limit: Max results (backend default 10, max 100).
        show: Also put the results on the 7" screen (one call).

    Returns:
        ``PhotoResults`` (a list) sorted by relevance. Empty → ``.hint``.
    """
    payload: dict[str, Any] = {}
    if query is not None:
        v.require_str(query, "query")
        payload["query"] = query
    for key, value in (("tags", tags), ("people", people)):
        if value is not None:
            v.require_list(value, key)
            for item in value:
                v.require_str(item, key)
            payload[key] = value
    for key, value in (("date_from", date_from), ("date_to", date_to),
                       ("source", source)):
        if value is not None:
            v.require_str(value, key)
            payload[key] = value
    if in_slideshow is not None:
        payload["in_slideshow"] = v.require_bool(in_slideshow, "in_slideshow")
    if limit is not None:
        v.require_int(limit, "limit", min_val=1, max_val=100)
        payload["limit"] = limit
    if show:
        payload["show"] = v.require_bool(show, "show")

    resp = _check(_transport.request("photos.search", payload, timeout=30))
    results = PhotoResults(PhotoRecord(r) for r in resp.get("results", []))
    results.hint = resp.get("hint")
    results.shown = resp.get("shown")
    return results


def get(photo_id: str) -> PhotoRecord:
    """Get full details for a photo.

    Args:
        photo_id: Photo ID.

    Returns:
        PhotoRecord with full details.
    """
    v.require_str(photo_id, "photo_id")
    resp = _check(_transport.request("photos.get", {"id": photo_id}, timeout=30))
    return PhotoRecord(resp)


def view(photo_id: str) -> dict[str, Any]:
    """Attach a photo's pixels to the tool result so the agent sees it.

    Returns ``{id, filename, kind: "image", attached: True}``. The
    actual image bytes are delivered as an image content block on the
    tool result, not as a field in this dict.
    """
    v.require_str(photo_id, "photo_id")
    return _check(_transport.request("photos.view", {"id": photo_id}, timeout=30))


def view_path(path: str) -> dict[str, Any]:
    """Attach a local image file's pixels to the tool result.

    Use this for files that haven't been ingested into the photo
    library yet — most often an inbound Signal/WhatsApp image staged at
    ``[image attached at <path>]`` in the user's message. Path must
    live under one of the allowlisted roots (sandbox tmp, workspace,
    photos, perception crops); anything else is refused.

    Returns ``{path, filename, kind: "image", attached: True}``. The
    image bytes ride back as an image content block on the tool
    result, not as a field in this dict.
    """
    v.require_str(path, "path")
    return _check(_transport.request("photos.view_path", {"path": path}, timeout=30))


def ingest(
    path: str,
    *,
    source: str,
    sender: str | None = None,
    caption: str | None = None,
) -> str:
    """Hand a local image file to the photo intake pipeline.

    Use this once you've decided an inbound or scratch image belongs in
    the photo library. The pipeline copies the bytes into
    ``data/photos/``, runs detection + tagging, and indexes for search.
    The original ``path`` is deleted on success (it was only staging).

    Args:
        path: Local file path (must be under an allowlisted root).
        source: Where the image came from — ``"signal"``, ``"whatsapp"``,
            ``"camera"``, or ``"upload"``. Stored on the photo for
            filtering later.
        sender: Person who sent it, if known.
        caption: Optional human-supplied caption, verbatim. The tagger
            reads it to name household people (a caption naming someone
            on the roster = "certain"); also the fallback description.

    Returns the new photo's ID.
    """
    v.require_str(path, "path")
    v.require_str(source, "source")
    payload: dict[str, Any] = {"path": path, "source": source}
    # Treat empty / whitespace-only sender and caption as "not provided"
    # rather than rejecting them — callers (and the agent) naturally pass
    # caption="" when there is no caption, and that should be a no-op, not
    # a ValueError from require_str.
    if sender and sender.strip():
        v.require_str(sender, "sender")
        payload["sender"] = sender
    if caption and caption.strip():
        v.require_str(caption, "caption")
        payload["caption"] = caption
    resp = _check(_transport.request("photos.ingest", payload, timeout=120))
    return resp["photo_id"]


def show_on_screen(photo_ids: list[str]) -> dict[str, Any]:
    """Display one or more photos on the 7\" screen.

    Dispatches through the display manager to the ``picture`` display.
    The call returns immediately; the display manager handles rendering.
    """
    v.require_list(photo_ids, "photo_ids")
    for pid in photo_ids:
        v.require_str(pid, "photo_id")
    return _check(_transport.request(
        "photos.show_on_screen",
        {"photo_ids": photo_ids},
        timeout=10,
    ))


# --- Metadata updates ---

def update(photo_id: str, *, description: str) -> None:
    """Update a photo's description.

    Args:
        photo_id: Photo ID.
        description: New description text.
    """
    v.require_str(photo_id, "photo_id")
    v.require_str(description, "description")
    _check(_transport.request("photos.update", {
        "id": photo_id,
        "description": description,
    }, timeout=30))


def set_tags(photo_id: str, *, tags: list[str]) -> None:
    """Set tags on a photo (replaces existing tags).

    Args:
        photo_id: Photo ID.
        tags: List of tag strings.
    """
    v.require_str(photo_id, "photo_id")
    v.require_list(tags, "tags")
    _check(_transport.request("photos.set_tags", {
        "id": photo_id,
        "tags": tags,
    }, timeout=30))


def set_person(photo_id: str, *, person_index: int, name: str) -> None:
    """Rename one existing person slot (marks it human-confirmed).

    To add or remove people use ``set_people``.

    Args:
        photo_id: Photo ID.
        person_index: Index in the photo's people list (0-based).
        name: Person name to assign.
    """
    v.require_str(photo_id, "photo_id")
    v.require_int(person_index, "person_index", min_val=0)
    v.require_str(name, "name")
    _check(_transport.request("photos.set_person", {
        "id": photo_id,
        "person_index": person_index,
        "name": name,
    }, timeout=30))


# --- People: human confirmation ---

def set_people(photo_id: str, *, names: list[str]) -> list[str]:
    """Set the human-confirmed people on a photo (replaces all labels).

    Add, remove or correct in one call. Only after a human said who is
    in the photo: the result is stored as confirmed and the tagger never
    changes it again. ``names=[]`` = nobody from the household.

    Returns the stored names (household spelling).
    """
    v.require_str(photo_id, "photo_id")
    v.require_list(names, "names")
    for name in names:
        v.require_str(name, "name")
    resp = _check(_transport.request("photos.set_people", {
        "id": photo_id,
        "names": names,
    }, timeout=30))
    return resp.get("people", [])


def confirm_people(photo_id: str) -> list[str]:
    """Mark a photo's current certain/likely labels human-confirmed.

    Use when a human agreed with the labels as they stand. ``unsure``
    suggestions are dropped. Returns the confirmed names.
    """
    v.require_str(photo_id, "photo_id")
    resp = _check(_transport.request(
        "photos.confirm_people", {"id": photo_id}, timeout=30))
    return resp.get("people", [])


def unconfirmed(limit: int = 20) -> list[PhotoRecord]:
    """Photos with likely/unsure people no human has reviewed, newest first.

    Each record's ``people`` items carry ``basis`` (the tagger's reason);
    ``unsure`` is ``[{name, basis}]``.
    """
    v.require_int(limit, "limit", min_val=1, max_val=100)
    resp = _check(_transport.request(
        "photos.unconfirmed", {"limit": limit}, timeout=30))
    return [PhotoRecord(r) for r in resp.get("results", [])]


# --- Household roster (who the tagger may name) ---

def roster() -> list[dict[str, Any]]:
    """The household roster: ``[{name, note, person_id}]``."""
    resp = _check(_transport.request("photos.roster", {}, timeout=30))
    return resp.get("roster", [])


def set_roster(name: str, *, note: str) -> dict[str, Any]:
    """Add or update a roster entry.

    Args:
        name: Person's name. Matched to a known person case-insensitively
            (their spelling wins); a person the box has never met is fine.
        note: One line, max 200 chars: how to recognise them in a photo
            (age, hair, glasses, usual context).

    Returns ``{name, note, linked}`` — ``linked`` = tied to a known person.
    """
    v.require_str(name, "name")
    v.require_str(note, "note", allow_empty=True)
    return _check(_transport.request("photos.set_roster", {
        "name": name,
        "note": note,
    }, timeout=30))


def remove_from_roster(name: str) -> None:
    """Remove a roster entry (existing photo labels are kept)."""
    v.require_str(name, "name")
    _check(_transport.request(
        "photos.remove_from_roster", {"name": name}, timeout=30))


# --- Slideshow management ---

def add_to_slideshow(photo_id: str) -> None:
    """Add a photo to the slideshow rotation.

    Args:
        photo_id: Photo ID.
    """
    v.require_str(photo_id, "photo_id")
    _check(_transport.request(
        "photos.add_to_slideshow", {"id": photo_id}, timeout=30))


def remove_from_slideshow(photo_id: str) -> None:
    """Remove a photo from the slideshow rotation.

    Args:
        photo_id: Photo ID.
    """
    v.require_str(photo_id, "photo_id")
    _check(_transport.request(
        "photos.remove_from_slideshow", {"id": photo_id}, timeout=30))


# --- Tag library management ---

def merge_tags(source: str, *, into: str) -> None:
    """Merge one tag into another (consolidate synonyms).

    Args:
        source: Tag to merge from (will be removed).
        into: Tag to merge into (will be kept).
    """
    v.require_str(source, "source tag")
    v.require_str(into, "target tag")
    _check(_transport.request("photos.merge_tags", {
        "source": source,
        "into": into,
    }, timeout=30))


def rename_tag(old: str, *, to: str) -> None:
    """Rename a tag (fix typos).

    Args:
        old: Current tag name.
        to: New tag name.
    """
    v.require_str(old, "old tag name")
    v.require_str(to, "new tag name")
    _check(_transport.request("photos.rename_tag", {
        "old": old,
        "new": to,
    }, timeout=30))


def delete_tag(tag: str) -> None:
    """Delete a tag from the library.

    Args:
        tag: Tag name to delete.
    """
    v.require_str(tag, "tag")
    _check(_transport.request("photos.delete_tag", {"tag": tag}, timeout=30))


# --- Soft delete / restore ---

def delete(photo_id: str) -> None:
    """Soft-delete a photo (30-day retention, restorable).

    Args:
        photo_id: Photo ID.
    """
    v.require_str(photo_id, "photo_id")
    _check(_transport.request("photos.delete", {"id": photo_id}, timeout=30))


def restore(photo_id: str) -> None:
    """Restore a soft-deleted photo.

    Args:
        photo_id: Photo ID.
    """
    v.require_str(photo_id, "photo_id")
    _check(_transport.request("photos.restore", {"id": photo_id}, timeout=30))


def list_deleted() -> list[PhotoRecord]:
    """List soft-deleted photos.

    Returns:
        List of PhotoRecord objects that are soft-deleted.
    """
    response = _check(_transport.request("photos.list_deleted", {}, timeout=30))
    results = response.get("results", [])
    return [PhotoRecord(r) for r in results]


# --- Storage info ---

def storage_info() -> StorageInfo:
    """Get photo storage quota information.

    Returns:
        StorageInfo with usage and quota details.
    """
    response = _check(_transport.request("photos.storage_info", {}, timeout=30))
    return StorageInfo(response)
