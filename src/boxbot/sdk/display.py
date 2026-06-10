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
    bb.display.get_active()                    # what's on screen + pin/rotation state
    bb.display.screenshot()                    # live screen → PNG, attached
    bb.display.schema()                        # block reference, themes
    bb.display.describe_source("weather")      # data source field shape

Control::

    bb.display.unpin()                         # release pin, resume rotation
    bb.display.set_rotation(displays=[...],    # configure idle rotation
                            interval=30)       #   (also clears pin)

The server validates on save and preview; errors come back as a list
the agent reads and acts on.

Pinning model:
    ``switch_display(name, args)`` is pinned by default — the display
    stays put until you call ``switch_display`` again or
    ``bb.display.unpin()``. While pinned, idle rotation is paused.
    ``unpin()`` resumes the configured rotation; ``set_rotation()`` is
    how you reshape that list at runtime (e.g. swap in a slideshow
    list at night, then back in the morning).
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


def get_active() -> dict[str, Any]:
    """Return what is currently on the 7" screen, plus pin/rotation state.

    Shape::

        {
          "name": "morning_brief" | None,    # None when nothing is active
          "args": {...},                     # the args switch_display received
          "theme": "boxbot" | None,
          "pinned": True | False,            # was this set explicitly?
          "rotation": {
              "active": False,               # is the rotation loop running?
              "displays": [...],             # rotation list
              "interval": 30,                # seconds between switches
              "next_in_sec": null,           # ~time until next tick (None if inactive)
          },
        }

    Use this to inspect state before deciding what to do: e.g. ``if
    state["pinned"] and state["name"] == "picture": ...``. Also handy
    to confirm a ``switch_display`` actually took effect.
    """
    return _check(_transport.request("display.get_active", {}))


def unpin() -> dict[str, Any]:
    """Release the pin so idle rotation resumes.

    After ``switch_display(...)`` (which pins by default), the screen
    holds the chosen display until ``unpin()`` is called or another
    ``switch_display`` replaces it. ``unpin()`` clears the pin flag and
    restarts the configured idle rotation. If no rotation list is
    configured, the current display simply stays on screen unpinned.
    """
    return _check(_transport.request("display.unpin", {}))


def set_rotation(
    displays: list[str] | None = None,
    interval: int | None = None,
) -> dict[str, Any]:
    """Configure and start idle rotation. Clears any pin.

    Args:
        displays: List of display names to cycle through. ``None`` uses
            the config defaults. An empty list stops rotation
            altogether (display sits unpinned on whatever was last
            shown).
        interval: Seconds between switches. ``None`` keeps the current
            interval (or config default).
    """
    payload: dict[str, Any] = {}
    if displays is not None:
        payload["displays"] = displays
    if interval is not None:
        payload["interval"] = interval
    return _check(_transport.request("display.set_rotation", payload))


def screenshot() -> dict[str, Any]:
    """Capture the live 1024x600 display surface and attach it as an image.

    Renders the *current* state of the screen (live data and all) to a
    PNG and attaches it to the tool result so you literally see what
    the household sees. Use this to verify your spec looks right with
    real data flowing — preview() shows placeholders, screenshot() is
    the truth.

    Returns ``{"path": str, "attached": bool, "name": str | None}``.
    Subject to the per-call image-attachment cap; ``attached`` is
    ``False`` if the cap was already hit (the PNG path is still
    viewable via ``bb.workspace.view``).

    Raises:
        RuntimeError: If no display is currently active or the display
            manager is not running.
    """
    return _check(_transport.request("display.screenshot", {}))


def update_data(display_name: str, source_name: str, *,
                value: Any) -> dict[str, Any]:
    """Push a new value into a ``static`` source on any registered display.

    Works whether or not the display is currently active. If active, the
    update is applied to the live source and the screen re-renders. If
    inactive, the spec is mutated so the next activation picks up the
    new value; for agent-saved displays the spec is also persisted to
    ``data/displays/<name>.json`` so the value survives a restart.

    Returns ``{"status": "ok", "live": bool, "persisted": bool}`` where
    ``live`` is True if the on-screen source was updated and ``persisted``
    is True if the spec on disk was rewritten.

    Raises:
        RuntimeError: If the display isn't registered, the source name
            doesn't exist on the spec, or the source isn't of type
            ``static``.
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
    """Raise ``bb.ActionError`` with a multi-line message on a non-ok response.

    The dispatcher returns either ``{"status": "ok", ...}`` or
    ``{"status": "error", "errors": [...]}`` (or ``{"error": "..."}``
    for single-message failures). This helper normalizes the error
    surface so callers always get the same exception shape.
    (``ActionError`` subclasses ``RuntimeError``, so older ``except
    RuntimeError`` handlers keep working.)
    """
    if resp.get("status") == "ok":
        return resp
    errors = resp.get("errors") or [resp.get("error", "unknown error")]
    raise _transport.ActionError(
        "display error:\n  " + "\n  ".join(str(e) for e in errors),
        response=resp,
    )
