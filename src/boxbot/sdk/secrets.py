"""Secret store — write-only credential storage from the sandbox.

The agent stores credentials by name (``store``), confirms they exist
(``has`` / ``use``), removes them (``delete``), and lists what's stored
(``list``). It cannot read values back: integrations and sandbox
scripts receive their values as ``BOXBOT_SECRET_<NAME>`` env vars,
injected by the runner at launch.

Usage:
    import boxbot_sdk as bb

    bb.secrets.store("POLYGON_API_KEY", "pk_live_...")
    # → {"status": "ok", "name": "POLYGON_API_KEY", "previous": "created"}

    bb.secrets.list()
    # → {"status": "ok",
    #    "secrets": [{"name": "POLYGON_API_KEY", "stored_at": "2026-05-02T…Z"}]}

    if bb.secrets.has("POLYGON_API_KEY"):
        # The integration that needs it declares it in its manifest;
        # the runner injects BOXBOT_SECRET_POLYGON_API_KEY at launch.
        ...

    bb.secrets.delete("OLD_API_KEY")
    # → {"status": "ok", "name": "OLD_API_KEY"}

Naming: SCREAMING_SNAKE_CASE only (``[A-Z][A-Z0-9_]*``), ≤64 chars.
This matches the shape integrations declare in their manifests.
"""

from __future__ import annotations

from typing import Any

from . import _transport, _validators as v


def store(name: str, value: str) -> dict[str, Any]:
    """Store a secret value under ``name``.

    Returns the dispatcher response — typically
    ``{"status": "ok", "name": ..., "previous": "created"|"replaced"}``,
    or ``{"status": "error", "message": ...}`` on validation failure
    (bad name shape, oversized value, store full).

    The value is never echoed back through any SDK call.
    """
    v.require_str(name, "name")
    v.require_str(value, "value")
    return _transport.request("secrets.store", {
        "name": name,
        "value": value,
    })


def delete(name: str) -> dict[str, Any]:
    """Remove a stored secret by name.

    Returns ``{"status": "ok", "name": ...}`` on success,
    ``{"status": "missing", "name": ...}`` if the name isn't stored.
    """
    v.require_str(name, "name")
    return _transport.request("secrets.delete", {"name": name})


def list() -> dict[str, Any]:  # noqa: A001 — shadows builtin intentionally; SDK module convention
    """Return the names (and stored-at timestamps) of every stored secret.

    Values are never returned. The agent uses this to decide whether
    a credential is already on file (so it doesn't ask the user
    again). Shape:
    ``{"status": "ok", "secrets": [{"name": ..., "stored_at": ...}, ...]}``.
    """
    return _transport.request("secrets.list", {})


def has(name: str) -> bool:
    """True iff a secret with this name is stored.

    Convenience wrapper around :func:`use` — returns a plain bool so
    a script can branch without parsing the response dict. Validates
    the name shape locally; an invalid name returns ``False`` rather
    than raising.
    """
    if not isinstance(name, str) or not name:
        return False
    response = _transport.request("secrets.use", {"name": name})
    return response.get("status") == "ok"


def use(name: str) -> str | None:
    """Return the env-var name that will carry this secret, or ``None``.

    Returns ``"BOXBOT_SECRET_<NAME>"`` iff the secret is stored.
    Returns ``None`` if the name isn't on file — preferring fail-fast
    over a silently-empty env var.

    For integration scripts: reading
    ``os.environ["BOXBOT_SECRET_<NAME>"]`` directly is the typical
    path; the runner injects declared secrets at launch. ``use`` is
    the diagnostic primitive for ad-hoc ``execute_script`` calls.
    """
    v.require_str(name, "name")
    response = _transport.request("secrets.use", {"name": name})
    if response.get("status") != "ok":
        return None
    env_var = response.get("env_var")
    return env_var if isinstance(env_var, str) else None
