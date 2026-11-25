"""People labels for the photo library — validation and the household roster.

The roster (``people_notes`` in the photo store) is the list of household
people the small-model tagger may name in a photo, each with a one-line
appearance/context note. It lives in the photo store rather than in
memory or the perception DB because the tagger is its only reader: a
small table with a validated shape beats scraping free-text memories,
and a person the box has never seen or heard (no perception record)
still needs an entry.

Names are tied to perception person records when one exists
(case-insensitive): the roster takes the perception spelling and stores
its ``person_id`` so identity renames/merges carry over.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

MAX_NAME_LEN = 40
MAX_NOTE_LEN = 200
MAX_ROSTER_SIZE = 24
MAX_NAMES_PER_PHOTO = 12

# Letters/digits (any script), then letters/digits/space/'/./- . No
# control characters, quotes, brackets or newlines — names are placed in
# a model prompt and in search queries.
_NAME_RE = re.compile(r"^[^\W_][\w .'\-]*$", re.UNICODE)
_PHOTO_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")


def validate_person_name(value: Any, field: str = "name") -> str:
    """Return a cleaned person name or raise ``ValueError``."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if _CONTROL_RE.search(value):
        raise ValueError(f"{field} must not contain control characters")
    name = " ".join(value.split())
    if not name:
        raise ValueError(f"{field} must not be empty")
    if len(name) > MAX_NAME_LEN:
        raise ValueError(f"{field} too long (max {MAX_NAME_LEN} chars)")
    if "_" in name or not _NAME_RE.match(name):
        raise ValueError(
            f"{field} may contain only letters, digits, spaces and . ' -"
        )
    return name


def validate_photo_id(value: Any) -> str:
    """Return a well-formed photo id or raise ``ValueError``."""
    if not isinstance(value, str) or not _PHOTO_ID_RE.match(value):
        raise ValueError("photo id is required (letters, digits, _ and -)")
    return value


def clean_note(value: Any) -> str:
    """Return a single-line roster note or raise ``ValueError``."""
    if not isinstance(value, str):
        raise ValueError("note must be a string")
    note = " ".join(_CONTROL_RE.sub(" ", value).split())
    if len(note) > MAX_NOTE_LEN:
        raise ValueError(f"note too long (max {MAX_NOTE_LEN} chars)")
    return note


def clean_untrusted_text(value: str | None, max_len: int) -> str:
    """Flatten message-supplied text (caption, sender) to one bounded line."""
    if not value:
        return ""
    return " ".join(_CONTROL_RE.sub(" ", str(value)).split())[:max_len]


async def resolve_perception_person(name: str) -> dict[str, Any] | None:
    """Find the perception person record for *name*, case-insensitively.

    Returns ``{"id", "name"}`` or None. Best-effort: None when the
    identity service is not running (dev, tests, the backfill CLI).
    """
    try:
        from boxbot.perception.identity import get_identity

        cloud_store = get_identity().cloud_store
        if cloud_store is None:
            return None
        persons = await cloud_store.list_persons()
    except RuntimeError:
        return None
    except Exception:
        logger.exception("Roster: perception person lookup failed")
        return None
    wanted = name.casefold()
    for person in persons:
        pname = person.get("name")
        if isinstance(pname, str) and pname.casefold() == wanted:
            return {"id": person.get("id"), "name": pname}
    return None


async def canonical_person(store: Any, name: str) -> dict[str, Any]:
    """Map a validated name onto the household's spelling of it.

    Order: perception record, then roster entry, else the name as given.
    Returns ``{"label", "person_id"}``.
    """
    person = await resolve_perception_person(name)
    if person is not None:
        return {"label": person["name"], "person_id": person["id"]}
    wanted = name.casefold()
    for entry in await store.list_people_notes():
        if entry["name"].casefold() == wanted:
            return {"label": entry["name"], "person_id": entry["person_id"]}
    return {"label": name, "person_id": None}
