"""Display SDK — read, write, edit display specs as JSON dicts.

A display is a JSON document. The agent reads, mutates, and writes it
like any other data file. There is no builder, no fluent API, no block
classes — just dicts and lists.

Spec shape::

    {
      "name": "morning_glance",
      "theme": "boxbot",
      "data_sources": [
        {"name": "weather", "type": "builtin", "refresh": 3600},
        {"name": "calendar", "type": "builtin"}
      ],
      "layout": {
        "type": "column",
        "padding": 24,
        "gap": 16,
        "children": [
          {"type": "row", "align": "spread", "children": [
            {"type": "clock", "size": "lg"},
            {"type": "text", "content": "{calendar.events[0].title}",
             "color": "muted"}
          ]},
          {"type": "card", "color": "muted", "padding": 18, "children": [
            {"type": "metric", "value": "{weather.temp}°",
             "icon": "{weather.icon}"}
          ]}
        ]
      }
    }

Lifecycle::

    spec = bb.display.load("morning_glance")   # → dict
    spec["theme"] = "midnight"
    spec["layout"]["children"][1]["color"] = "accent"
    bb.display.preview(spec)                   # render to PNG, attach
    bb.display.save(spec)                      # validate, write, register

Discovery::

    bb.display.list()                          # all displays + source
    bb.display.schema()                        # block reference, themes
    bb.display.describe_source("weather")      # data source field shape

The server validates on save and preview; errors come back as a list
the agent reads and acts on.
"""

from __future__ import annotations

from typing import Any

from . import _transport


def list() -> list[dict[str, Any]]:  # noqa: A001 - intentional shadow
    """Return every display known to the manager.

    Each entry is ``{"name": str, "source": "builtin"|"user"|"agent"}``.
    """
    return _check(_transport.request("display.list", {})).get("displays", [])


def load(name: str) -> dict[str, Any]:
    """Read an existing display spec.

    Returns the JSON dict the agent can mutate freely. Pass it back to
    :func:`save` (or :func:`preview`) when done.
    """
    return _check(_transport.request("display.load", {"name": name}))["spec"]


def save(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate, write, and register a display spec.

    The spec is written to ``data/displays/<name>.json`` and registered
    live with the running display manager — ``switch_display(<name>)``
    works immediately, no restart needed.

    Returns ``{"path": str, "registered": bool, "warnings": [...]}``.

    Raises:
        RuntimeError: If validation fails. The error message lists every
            problem found, one per line.
    """
    return _check(_transport.request("display.save", {"spec": spec}))


def preview(
    spec: dict[str, Any],
    *,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a spec to a 1024x600 PNG and attach it to the tool result.

    The image is reachable via ``result["path"]`` and (when there's
    capacity in this script run) attached as a multimodal content block
    so the agent literally sees the layout.

    Args:
        spec: The display spec dict.
        data: Optional override mapping source names to fetched data.
            Useful for testing ``http_json`` ``fields`` mappings without
            a real fetch — pass the JSON the API would return and the
            transform runs against it.

    Returns ``{"path": str, "attached": bool, "warnings": [...]}``.

    Raises:
        RuntimeError: If validation fails.
    """
    payload: dict[str, Any] = {"spec": spec}
    if data is not None:
        payload["data"] = data
    return _check(_transport.request("display.preview", payload))


def delete(name: str) -> None:
    """Remove an agent-saved display.

    Built-in displays cannot be deleted. If the display is currently
    active, the manager falls back to the idle rotation.
    """
    _check(_transport.request("display.delete", {"name": name}))


def describe_source(name: str) -> dict[str, Any]:
    """Return the field schema for a data source.

    For built-ins (``weather``, ``calendar``, ``tasks``, ``people``,
    ``agent_status``, ``clock``) returns ``{"fields": {...}, "example":
    {...}}`` so the agent can discover what bindings are available.
    """
    return _check(_transport.request("display.describe_source", {"name": name}))


def schema() -> dict[str, Any]:
    """Return the full block + spec reference as a dict.

    Shape::

        {
          "blocks": {
            "row": {"fields": {...}, "kind": "container"},
            "text": {"fields": {...}, "kind": "content"},
            ...
          },
          "themes": ["boxbot", "midnight", ...],
          "data_source_types": ["builtin", "http_json", ...],
          "transitions": ["crossfade", ...],
        }

    Use this to introspect what's available without re-reading the doc.
    """
    return _check(_transport.request("display.schema", {}))


def update_data(display_name: str, source_name: str, *,
                value: Any) -> dict[str, Any]:
    """Push a new value into a ``static`` source on the active display.

    Returns the response dict on success.

    Raises:
        RuntimeError: If the display is not active or the source isn't
            of type ``static``.
    """
    return _check(_transport.request("display.update_data", {
        "display": display_name,
        "source": source_name,
        "value": value,
    }))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check(resp: dict[str, Any]) -> dict[str, Any]:
    """Raise ``RuntimeError`` with a multi-line message on a non-ok response.

    The dispatcher returns either ``{"status": "ok", ...}`` or
    ``{"status": "error", "errors": [...]}`` (or ``{"error": "..."}``
    for single-message failures). This helper normalizes the error
    surface so callers always get the same exception shape.
    """
    if resp.get("status") == "ok":
        return resp
    errors = resp.get("errors") or [resp.get("error", "unknown error")]
    raise RuntimeError("display error:\n  " + "\n  ".join(str(e) for e in errors))
