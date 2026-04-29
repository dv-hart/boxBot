"""Google Calendar client used by display sources, tools, and the SDK.

Wraps google-api-python-client behind a small async interface that
returns display-ready dicts. All blocking API calls are run in a worker
thread so the event loop stays responsive.

Authentication uses OAuth 2.0 installed-app flow. The credential file
location (``client_secrets.json``) and the persisted user token live
under ``data/credentials/`` by default — both paths are configurable
through environment variables.

One-time auth is performed by ``scripts/calendar_auth.py``. After that,
the saved token auto-refreshes and never requires interaction again.

Usage:
    from boxbot.integrations import google_calendar as gc

    events = await gc.list_upcoming_events(max_results=5)
    new_id = await gc.create_event(
        summary="Dentist",
        start="2026-04-30T15:00:00",
        end="2026-04-30T16:00:00",
    )
    await gc.delete_event(new_id)
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from boxbot.core.paths import CREDENTIALS_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/calendar"]

_DEFAULT_CRED_DIR = CREDENTIALS_DIR
_CLIENT_SECRETS_ENV = "GOOGLE_CALENDAR_CLIENT_SECRETS"
_TOKEN_PATH_ENV = "GOOGLE_CALENDAR_TOKEN"
_CALENDAR_ID_ENV = "GOOGLE_CALENDAR_ID"


def _client_secrets_path() -> Path:
    return Path(
        os.environ.get(_CLIENT_SECRETS_ENV)
        or _DEFAULT_CRED_DIR / "google_client_secrets.json"
    )


def _token_path() -> Path:
    return Path(
        os.environ.get(_TOKEN_PATH_ENV)
        or _DEFAULT_CRED_DIR / "google_calendar_token.json"
    )


def _default_calendar_id() -> str:
    return os.environ.get(_CALENDAR_ID_ENV) or "primary"


# ---------------------------------------------------------------------------
# Auth and service
# ---------------------------------------------------------------------------


class CalendarNotAuthenticated(RuntimeError):
    """Raised when no valid OAuth token is present."""


def _load_credentials() -> Any:
    """Load saved OAuth credentials, refreshing if expired.

    Returns:
        google.oauth2.credentials.Credentials

    Raises:
        CalendarNotAuthenticated: if no token file exists.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token_file = _token_path()
    if not token_file.exists():
        raise CalendarNotAuthenticated(
            f"No Google Calendar token at {token_file}. "
            f"Run scripts/calendar_auth.py to authenticate."
        )

    creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_file.write_text(creds.to_json())
        else:
            raise CalendarNotAuthenticated(
                f"Saved token at {token_file} is invalid and cannot be "
                f"refreshed. Re-run scripts/calendar_auth.py."
            )
    return creds


def _build_service() -> Any:
    """Build a Google Calendar API service client."""
    from googleapiclient.discovery import build

    creds = _load_credentials()
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


async def _run_blocking(fn, *args, **kwargs) -> Any:
    """Run a blocking function in the default executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


async def list_upcoming_events(
    *,
    max_results: int = 5,
    calendar_id: str | None = None,
    time_min: datetime | None = None,
    time_max: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return upcoming calendar events as display-ready dicts.

    Args:
        max_results: Maximum number of events to return.
        calendar_id: Calendar to read from (defaults to primary).
        time_min: Lower time bound (defaults to now).
        time_max: Upper time bound (defaults to none).

    Returns:
        List of event dicts with keys: id, time, title, start, end,
        location, duration, all_day, description.
    """
    cal_id = calendar_id or _default_calendar_id()
    if time_min is None:
        time_min = datetime.now(timezone.utc)

    def _do_fetch() -> list[dict[str, Any]]:
        service = _build_service()
        params = {
            "calendarId": cal_id,
            "timeMin": time_min.isoformat(),
            "maxResults": max_results,
            "singleEvents": True,
            "orderBy": "startTime",
        }
        if time_max is not None:
            params["timeMax"] = time_max.isoformat()
        result = service.events().list(**params).execute()
        return [_normalize_event(e) for e in result.get("items", [])]

    try:
        return await _run_blocking(_do_fetch)
    except CalendarNotAuthenticated:
        raise
    except Exception as e:
        logger.warning("Calendar list_upcoming_events failed: %s", e)
        return []


async def get_event(event_id: str, *, calendar_id: str | None = None) -> dict[str, Any] | None:
    """Fetch a single event by ID."""
    cal_id = calendar_id or _default_calendar_id()

    def _do_get() -> dict[str, Any] | None:
        service = _build_service()
        try:
            result = service.events().get(
                calendarId=cal_id, eventId=event_id
            ).execute()
        except Exception as e:
            logger.warning("Calendar get_event(%s) failed: %s", event_id, e)
            return None
        return _normalize_event(result)

    return await _run_blocking(_do_get)


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


async def create_event(
    *,
    summary: str,
    start: str | datetime,
    end: str | datetime,
    description: str | None = None,
    location: str | None = None,
    calendar_id: str | None = None,
    all_day: bool = False,
) -> str:
    """Create a calendar event. Returns the new event ID.

    Args:
        summary: Event title.
        start: ISO datetime (or date string for all-day) or datetime.
        end: ISO datetime (or date string for all-day) or datetime.
        description: Optional long description.
        location: Optional location string.
        calendar_id: Calendar to write to (defaults to primary).
        all_day: If True, treat start/end as YYYY-MM-DD date strings.
    """
    cal_id = calendar_id or _default_calendar_id()
    body: dict[str, Any] = {"summary": summary}
    if description:
        body["description"] = description
    if location:
        body["location"] = location

    body["start"] = _time_field(start, all_day)
    body["end"] = _time_field(end, all_day)

    def _do_create() -> str:
        service = _build_service()
        result = service.events().insert(
            calendarId=cal_id, body=body
        ).execute()
        return result.get("id", "")

    return await _run_blocking(_do_create)


async def update_event(
    event_id: str,
    *,
    summary: str | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    description: str | None = None,
    location: str | None = None,
    calendar_id: str | None = None,
    all_day: bool = False,
) -> bool:
    """Update fields on an existing event. Returns True on success."""
    cal_id = calendar_id or _default_calendar_id()

    def _do_update() -> bool:
        service = _build_service()
        # Patch semantics: only include changed fields
        body: dict[str, Any] = {}
        if summary is not None:
            body["summary"] = summary
        if description is not None:
            body["description"] = description
        if location is not None:
            body["location"] = location
        if start is not None:
            body["start"] = _time_field(start, all_day)
        if end is not None:
            body["end"] = _time_field(end, all_day)
        if not body:
            return False
        service.events().patch(
            calendarId=cal_id, eventId=event_id, body=body
        ).execute()
        return True

    try:
        return await _run_blocking(_do_update)
    except Exception as e:
        logger.warning("Calendar update_event(%s) failed: %s", event_id, e)
        return False


async def delete_event(
    event_id: str, *, calendar_id: str | None = None
) -> bool:
    """Delete an event. Returns True on success."""
    cal_id = calendar_id or _default_calendar_id()

    def _do_delete() -> bool:
        service = _build_service()
        service.events().delete(
            calendarId=cal_id, eventId=event_id
        ).execute()
        return True

    try:
        return await _run_blocking(_do_delete)
    except Exception as e:
        logger.warning("Calendar delete_event(%s) failed: %s", event_id, e)
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _time_field(value: str | datetime, all_day: bool) -> dict[str, str]:
    """Build a Calendar API start/end field from a datetime or string."""
    if all_day:
        if isinstance(value, datetime):
            value = value.strftime("%Y-%m-%d")
        return {"date": value}
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.astimezone()
        value = value.isoformat()
    return {"dateTime": value}


def _normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    """Convert a Google Calendar event into a display-ready dict."""
    start = event.get("start", {})
    end = event.get("end", {})
    all_day = "date" in start

    if all_day:
        start_dt = _parse_date(start.get("date"))
        end_dt = _parse_date(end.get("date"))
        time_str = "all day"
        duration = ""
    else:
        start_dt = _parse_iso(start.get("dateTime"))
        end_dt = _parse_iso(end.get("dateTime"))
        time_str = _format_time(start_dt) if start_dt else ""
        duration = _format_duration(start_dt, end_dt)

    return {
        "id": event.get("id", ""),
        "title": event.get("summary", "(no title)"),
        "time": time_str,
        "start": start_dt.isoformat() if start_dt else None,
        "end": end_dt.isoformat() if end_dt else None,
        "duration": duration,
        "location": event.get("location", ""),
        "all_day": all_day,
        "description": event.get("description", ""),
    }


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _format_time(dt: datetime) -> str:
    """Format a datetime as a short local time string ('9:00 AM')."""
    local = dt.astimezone()
    return local.strftime("%-I:%M %p")


def _format_duration(start: datetime | None, end: datetime | None) -> str:
    if start is None or end is None:
        return ""
    delta: timedelta = end - start
    minutes = int(delta.total_seconds() / 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, mins = divmod(minutes, 60)
    if mins == 0:
        return f"{hours}h"
    return f"{hours}h {mins}m"


def is_authenticated() -> bool:
    """Return True if a token file is present (without validating it)."""
    return _token_path().exists()
