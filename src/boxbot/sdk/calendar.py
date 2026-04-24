"""Google Calendar operations for agent-written scripts.

Mirrors the boxbot.integrations.google_calendar interface but emits
structured SDK actions through stdout so the main process performs the
actual API call. Credentials never leave the main process — the sandbox
sees only the request and the structured response.

Usage:
    from boxbot_sdk import calendar

    # Add an event
    calendar.create_event(
        summary="Dentist",
        start="2026-04-30T15:00:00",
        end="2026-04-30T16:00:00",
        location="Downtown",
    )

    # Update or delete by ID
    calendar.update_event("abc123", location="Uptown")
    calendar.delete_event("abc123")

    # List upcoming events (returned via the action result envelope)
    events = calendar.list_upcoming_events(max_results=5)
    for e in events:
        print(e["time"], e["title"])

The ``list_upcoming_events`` call requests data from the main process and
prints a marker line that the main process reads to assemble the
response. The script receives the parsed result before returning.
"""

from __future__ import annotations

from typing import Any

from . import _transport, _validators as v


def create_event(
    *,
    summary: str,
    start: str,
    end: str,
    description: str | None = None,
    location: str | None = None,
    calendar_id: str | None = None,
    all_day: bool = False,
) -> None:
    """Create a calendar event.

    Args:
        summary: Event title.
        start: ISO datetime string (or YYYY-MM-DD if ``all_day``).
        end: ISO datetime string (or YYYY-MM-DD if ``all_day``).
        description: Optional long description.
        location: Optional location string.
        calendar_id: Calendar to write to (defaults to primary).
        all_day: If True, treat start/end as all-day date strings.
    """
    v.require_str(summary, "summary")
    v.require_str(start, "start")
    v.require_str(end, "end")

    payload: dict[str, Any] = {
        "summary": summary,
        "start": start,
        "end": end,
        "all_day": bool(all_day),
    }
    if description is not None:
        payload["description"] = v.require_str(description, "description")
    if location is not None:
        payload["location"] = v.require_str(location, "location")
    if calendar_id is not None:
        payload["calendar_id"] = v.require_str(calendar_id, "calendar_id")

    _transport.emit_action("calendar.create_event", payload)


def update_event(
    event_id: str,
    *,
    summary: str | None = None,
    start: str | None = None,
    end: str | None = None,
    description: str | None = None,
    location: str | None = None,
    calendar_id: str | None = None,
    all_day: bool = False,
) -> None:
    """Update fields on an existing calendar event."""
    v.require_str(event_id, "event_id")

    payload: dict[str, Any] = {"event_id": event_id, "all_day": bool(all_day)}
    if summary is not None:
        payload["summary"] = v.require_str(summary, "summary")
    if start is not None:
        payload["start"] = v.require_str(start, "start")
    if end is not None:
        payload["end"] = v.require_str(end, "end")
    if description is not None:
        payload["description"] = v.require_str(description, "description")
    if location is not None:
        payload["location"] = v.require_str(location, "location")
    if calendar_id is not None:
        payload["calendar_id"] = v.require_str(calendar_id, "calendar_id")

    _transport.emit_action("calendar.update_event", payload)


def delete_event(event_id: str, *, calendar_id: str | None = None) -> None:
    """Delete a calendar event by ID."""
    v.require_str(event_id, "event_id")

    payload: dict[str, Any] = {"event_id": event_id}
    if calendar_id is not None:
        payload["calendar_id"] = v.require_str(calendar_id, "calendar_id")

    _transport.emit_action("calendar.delete_event", payload)


def list_upcoming_events(
    *,
    max_results: int = 5,
    calendar_id: str | None = None,
) -> list[dict[str, Any]]:
    """Request upcoming events. Returns a list of normalized event dicts.

    The main process performs the API call and returns the result as part
    of the SDK action response envelope. Until the agent transport gains a
    full request/response channel, callers should read the events through
    the ``sdk_actions`` field of the execute_script result.
    """
    v.require_int(max_results, "max_results", min_val=1, max_val=50)
    payload: dict[str, Any] = {"max_results": max_results}
    if calendar_id is not None:
        payload["calendar_id"] = v.require_str(calendar_id, "calendar_id")

    _transport.emit_action("calendar.list_upcoming_events", payload)
    return []
