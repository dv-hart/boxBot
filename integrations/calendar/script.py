"""Google Calendar integration — sandbox-runnable.

Self-contained Calendar v3 REST client with OAuth refresh. Reads the
saved token JSON from ``BOXBOT_SECRET_GOOGLE_CALENDAR_TOKEN_JSON``,
makes the requested API call, and (on 401) refreshes the access token,
persists the rotated token back via ``bb.secrets.store(...)``, and
retries once.

Token format mirrors what ``google.oauth2.credentials.Credentials.to_json()``
writes — the same shape ``scripts/calendar_auth.py`` produces from the
OAuth installed-app flow:

    {
      "token": "<short-lived access token>",
      "refresh_token": "<long-lived>",
      "token_uri": "https://oauth2.googleapis.com/token",
      "client_id": "...",
      "client_secret": "...",
      "scopes": ["https://www.googleapis.com/auth/calendar"],
      "expiry": "2026-05-02T15:30:00.123456"  // naive UTC
    }

Migrated from ``src/boxbot/integrations/google_calendar.py``. The
helpers (_normalize_event, _format_time, etc.) move with the script
and are otherwise unchanged.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

# ``boxbot_sdk`` is only imported inside ``_refresh_token`` so the helpers
# below (``_normalize_event``, ``_time_field``) can be exercised in unit
# tests without the sandbox-only SDK installed.
from boxbot_sdk.integration import inputs as get_inputs, return_output


SECRET_NAME = "GOOGLE_CALENDAR_TOKEN_JSON"
TOKEN_URI = "https://oauth2.googleapis.com/token"
CAL_BASE = "https://www.googleapis.com/calendar/v3"
_TIMEOUT = httpx.Timeout(15.0)


# ---------------------------------------------------------------------------
# Token loading + refresh
# ---------------------------------------------------------------------------


def _load_token() -> dict[str, Any]:
    raw = os.environ.get(f"BOXBOT_SECRET_{SECRET_NAME}")
    if not raw:
        return_output(
            {
                "error": (
                    f"Calendar token not stored. Run scripts/calendar_auth.py "
                    f"on the Pi to authenticate, or migrate an existing "
                    f"data/credentials/google_calendar_token.json with "
                    f"scripts/migrate_calendar_secret.py."
                )
            }
        )
        sys.exit(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        return_output({"error": f"calendar token JSON is invalid: {exc}"})
        sys.exit(0)


def _refresh_token(token: dict[str, Any]) -> dict[str, Any]:
    """POST refresh_token to TOKEN_URI; return updated token dict.

    Persists the rotated token back to the secret store so the next
    call doesn't repeat the dance.
    """
    refresh_token = token.get("refresh_token")
    client_id = token.get("client_id")
    client_secret = token.get("client_secret")
    if not (refresh_token and client_id and client_secret):
        return_output(
            {
                "error": (
                    "calendar token missing refresh fields — "
                    "re-run scripts/calendar_auth.py to mint a fresh token."
                )
            }
        )
        sys.exit(0)

    resp = httpx.post(
        token.get("token_uri") or TOKEN_URI,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    new = resp.json()
    expires_in = int(new.get("expires_in", 3600))
    token["token"] = new["access_token"]
    # Match Credentials.to_json() shape: naive UTC isoformat, no tz suffix.
    token["expiry"] = (
        datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=expires_in)
    ).isoformat()
    if "refresh_token" in new:
        token["refresh_token"] = new["refresh_token"]
    import boxbot_sdk as bb  # local import — see top-of-file note

    bb.secrets.store(SECRET_NAME, json.dumps(token))
    return token


def _request(
    token: dict[str, Any],
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute a Calendar API call, refreshing the token on 401 once.

    Returns (token, response_json). The token may be mutated in-place
    if a refresh happened.
    """
    url = f"{CAL_BASE}{path}"
    for attempt in range(2):
        headers = {"Authorization": f"Bearer {token['token']}"}
        resp = httpx.request(
            method, url, params=params, json=json_body,
            headers=headers, timeout=_TIMEOUT,
        )
        if resp.status_code == 401 and attempt == 0:
            token = _refresh_token(token)
            continue
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content:
            return token, {}
        return token, resp.json()
    raise RuntimeError("calendar request failed after refresh+retry")


# ---------------------------------------------------------------------------
# Helpers (moved from src/boxbot/integrations/google_calendar.py)
# ---------------------------------------------------------------------------


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


def _normalize_event(event: dict[str, Any]) -> dict[str, Any]:
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


def _time_field(value: str, all_day: bool) -> dict[str, str]:
    """Build a Calendar API start/end field from an ISO string or YYYY-MM-DD."""
    if all_day:
        return {"date": value}
    return {"dateTime": value}


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------


def _list_upcoming_events(
    token: dict[str, Any], args: dict[str, Any]
) -> dict[str, Any]:
    cal_id = args.get("calendar_id") or "primary"
    max_results = max(1, min(int(args.get("max_results", 5)), 50))
    now = datetime.now(timezone.utc).isoformat()
    _, payload = _request(
        token, "GET", f"/calendars/{cal_id}/events",
        params={
            "timeMin": now,
            "maxResults": max_results,
            "singleEvents": "true",
            "orderBy": "startTime",
        },
    )
    events = [_normalize_event(e) for e in payload.get("items", [])]
    return {"events": events, "count": len(events)}


def _create_event(token: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    cal_id = args.get("calendar_id") or "primary"
    summary = args.get("summary")
    start = args.get("start")
    end = args.get("end")
    if not (summary and start and end):
        return {"error": "create_event requires summary, start, end"}
    all_day = bool(args.get("all_day", False))
    body: dict[str, Any] = {
        "summary": summary,
        "start": _time_field(start, all_day),
        "end": _time_field(end, all_day),
    }
    if args.get("description"):
        body["description"] = args["description"]
    if args.get("location"):
        body["location"] = args["location"]
    _, payload = _request(
        token, "POST", f"/calendars/{cal_id}/events", json_body=body,
    )
    return {"event_id": payload.get("id", ""), "ok": True}


def _update_event(token: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    cal_id = args.get("calendar_id") or "primary"
    event_id = args.get("event_id")
    if not event_id:
        return {"error": "update_event requires event_id"}
    all_day = bool(args.get("all_day", False))
    body: dict[str, Any] = {}
    if args.get("summary") is not None:
        body["summary"] = args["summary"]
    if args.get("description") is not None:
        body["description"] = args["description"]
    if args.get("location") is not None:
        body["location"] = args["location"]
    if args.get("start") is not None:
        body["start"] = _time_field(args["start"], all_day)
    if args.get("end") is not None:
        body["end"] = _time_field(args["end"], all_day)
    if not body:
        return {"ok": False, "error": "no fields to update"}
    _, _ = _request(
        token, "PATCH", f"/calendars/{cal_id}/events/{event_id}",
        json_body=body,
    )
    return {"ok": True}


def _delete_event(token: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    cal_id = args.get("calendar_id") or "primary"
    event_id = args.get("event_id")
    if not event_id:
        return {"error": "delete_event requires event_id"}
    _, _ = _request(
        token, "DELETE", f"/calendars/{cal_id}/events/{event_id}",
    )
    return {"ok": True}


_ACTIONS = {
    "list_upcoming_events": _list_upcoming_events,
    "create_event": _create_event,
    "update_event": _update_event,
    "delete_event": _delete_event,
}


def main() -> None:
    args = get_inputs()
    action = args.get("action")
    if action not in _ACTIONS:
        return_output(
            {
                "error": (
                    f"unknown action '{action}' — "
                    f"expected one of {sorted(_ACTIONS.keys())}"
                )
            }
        )
        return

    token = _load_token()
    try:
        result = _ACTIONS[action](token, args)
    except httpx.HTTPStatusError as exc:
        return_output(
            {
                "error": f"calendar API {exc.response.status_code}: {exc.response.text[:200]}",
            }
        )
        return
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"calendar action {action} failed: {exc}\n")
        return_output({"error": str(exc)})
        return

    return_output(result)


if __name__ == "__main__":
    main()
