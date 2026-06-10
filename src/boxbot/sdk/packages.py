"""Package installation — request packages, admins approve out-of-band.

The sandbox cannot install packages itself (seccomp, owner-only pip,
read-only site-packages). ``request()`` queues a durable request and
notifies every registered admin on their messaging channel; an admin
replies ``approve pkg <id>`` or ``deny pkg <id>``. The request does
**not** block waiting for the human — it returns immediately with
status ``"pending"`` and the request id; check back later with
``status()`` / ``list()``.

Lifecycle: ``pending → approved → installed | failed`` or
``pending → denied``. Once status is ``"installed"``, the package is
importable in the *next* ``execute_script`` run.

Usage:
    import boxbot_sdk as bb

    req = bb.packages.request(
        "google-api-python-client",
        reason="Needed for the Gmail integration",
    )
    print(req["id"], req["status"])      # e.g. "ab12cd34 pending"

    # …later (next wake cycle, next conversation, or after setting a
    # fire_after trigger to remind yourself):
    req = bb.packages.status("ab12cd34")
    if req["status"] == "installed":
        import googleapiclient
    elif req["status"] == "denied":
        print(f"Denied by admin: {req.get('note') or 'no reason given'}")

Error semantics: ``request()`` raises ``bb.ActionError`` if the package
spec is invalid or the request couldn't be queued — a *system* failure.
A human saying no is **not** an error: it arrives as a normal
``status() == "denied"`` read. The two are never conflated.
"""

from __future__ import annotations

from typing import Any

from . import _transport, _validators as v


def request(package: str, *, reason: str) -> dict[str, Any]:
    """Request installation of a Python package into the sandbox venv.

    Queues a durable request and notifies all registered admins with
    reply instructions. Returns immediately — approval is asynchronous
    and out-of-band; nothing the sandbox does can resolve the request.

    Args:
        package: A bare PyPI project name (``"requests"``) or an exact
            pin (``"requests==2.32.3"``). URLs, local paths, extras,
            and range specifiers are rejected.
        reason: Why this package is needed — shown to the admins,
            so make it convincing and specific.

    Returns:
        The request record::

            {"id": "ab12cd34", "package": ..., "reason": ...,
             "status": "pending", "requested_by": ...,
             "requested_at": ..., "duplicate": bool,
             "admins_notified": int}

        ``duplicate: True`` means an identical pending request already
        existed and was returned instead (no admin re-notified). If
        ``admins_notified`` is 0, no admin received the message (none
        registered, or their channel is down) — the request still
        waits in the queue.

    Raises:
        ActionError: invalid package spec, missing reason, or any
            other dispatch failure. "Denied by a human" is never an
            exception — see :func:`status`.
    """
    v.require_str(package, "package")
    v.require_str(reason, "reason")

    response = _transport.dispatch_or_raise("packages.request", {
        "package": package,
        "reason": reason,
    }, timeout=60)
    record: dict[str, Any] = dict(response.get("request") or {})
    record["duplicate"] = bool(response.get("duplicate", False))
    record["admins_notified"] = int(response.get("admins_notified", 0))
    return record


def status(request_id: str) -> dict[str, Any]:
    """Return the current state of a package request.

    Returns the request record (see :func:`request`); its ``status``
    field is one of ``pending``, ``approved`` (install in flight),
    ``installed``, ``failed`` (``note`` carries the pip error tail), or
    ``denied`` (``note`` may carry the admin's reason).

    Raises:
        ActionError: if no request with that id exists.
    """
    v.require_str(request_id, "request_id")
    response = _transport.dispatch_or_raise(
        "packages.status", {"id": request_id}
    )
    return dict(response.get("request") or {})


def list(status: str | None = None) -> Any:  # noqa: A001 — SDK module convention; returns list[dict]
    """List package requests, newest first.

    Args:
        status: Optional filter — ``pending``, ``approved``,
            ``installed``, ``failed``, or ``denied``.

    Returns:
        A list of request records (see :func:`request`).
    """
    payload: dict[str, Any] = {}
    if status is not None:
        v.require_str(status, "status")
        payload["request_status"] = status
    response = _transport.dispatch_or_raise("packages.list", payload)
    return [dict(r) for r in response.get("requests") or []]
