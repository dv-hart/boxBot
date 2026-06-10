"""Internal transport layer for SDK ↔ main process communication.

SDK actions are emitted as structured JSON lines on stdout, prefixed with
a marker so execute_script can separate them from regular print() output.

Two flavours:

- :func:`emit_action` — fire-and-forget. The main process processes the
  action and its result is attached to the tool result, but the sandbox
  script does not read a reply. Use for effect-only operations where the
  agent can see the outcome in the returned ``sdk_actions`` array.

- :func:`request` — emit an action tagged with ``_expects_response: true``
  and block on a single JSON line on stdin for the reply. Use when the
  script needs the result to continue (e.g. ``read``, ``search``, ``get``).
  The main process writes exactly one reply line per such request.

Error semantics (the SDK-wide rule):

- **Mutating calls raise.** Every write-path SDK function dispatches via
  :func:`dispatch_or_raise` (or an equivalent module-level checker) and
  raises :class:`ActionError` when the main process answers with
  ``status != "ok"``. A write that didn't happen must never look like
  it did.
- **Read calls return raw response dicts** with shapes documented in
  each function's docstring.

Module-specific exceptions (``WorkspaceError``, ``PhotosError``)
subclass :class:`ActionError`, so a script can always
``except bb.ActionError`` to catch any rejected SDK action.
"""

from __future__ import annotations

import json
import sys
import threading


class ActionError(RuntimeError):
    """A mutating SDK action was rejected by the main process.

    Carries the dispatcher's error message. Exposed as ``bb.ActionError``
    so sandbox scripts can catch any failed SDK write uniformly::

        try:
            bb.secrets.store("MY_KEY", value)
        except bb.ActionError as e:
            print(f"store failed: {e}")

    Attributes:
        action: The action type that failed (e.g. ``"secrets.store"``).
        response: The full response dict from the main process.
    """

    def __init__(
        self,
        message: str,
        *,
        action: str | None = None,
        response: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.action = action
        self.response = response or {}

# Marker prefix that execute_script looks for to identify SDK actions
_MARKER = "__BOXBOT_SDK_ACTION__:"

_write_lock = threading.Lock()


def emit_action(
    action_type: str,
    payload: dict,
    *,
    expects_response: bool = False,
) -> None:
    """Emit a structured SDK action to stdout.

    Args:
        action_type: The action type, e.g. "display.save", "memory.save".
        payload: The action payload dict.
        expects_response: Tag the action so the main process writes a
            reply line on stdin. Pair with :func:`collect_response`.
            Most callers should use :func:`request` instead.
    """
    message = {"_sdk": action_type, **payload}
    if expects_response:
        message["_expects_response"] = True
    line = _MARKER + json.dumps(message, separators=(",", ":"))
    with _write_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def collect_response(timeout: int = 30) -> dict:
    """Read a single response line from stdin (blocking).

    Pairs with :func:`emit_action` called with ``expects_response=True``
    (or with :func:`request`, which bundles the two).

    Args:
        timeout: Maximum seconds to wait for a response.

    Returns:
        Parsed JSON response dict from the main process.

    Raises:
        TimeoutError: If no response within timeout.
        RuntimeError: If stdin is closed or response is malformed.
    """
    import select

    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        raise TimeoutError(
            f"No response from main process within {timeout}s"
        )

    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("stdin closed — no response from main process")

    try:
        return json.loads(line.strip())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Malformed response from main process: {e}")


def request(action_type: str, payload: dict, *, timeout: int = 30) -> dict:
    """Emit an action and block on its reply."""
    emit_action(action_type, payload, expects_response=True)
    return collect_response(timeout=timeout)


def dispatch_or_raise(
    action_type: str,
    payload: dict,
    *,
    timeout: int = 30,
) -> dict:
    """Emit an action, block on its reply, and raise on a non-ok status.

    The standard transport for **mutating** SDK calls: the caller gets
    the response dict back iff ``status == "ok"``; anything else raises
    :class:`ActionError` carrying the dispatcher's message. Transport
    failures (timeout, closed stdin, malformed reply) propagate from
    :func:`collect_response` as ``TimeoutError`` / ``RuntimeError`` —
    "the system broke" stays distinguishable from "the main process
    said no".
    """
    response = request(action_type, payload, timeout=timeout)
    status = response.get("status")
    if status != "ok":
        errors = response.get("errors")
        if isinstance(errors, list) and errors:
            message = "\n  ".join(str(e) for e in errors)
        else:
            message = (
                response.get("message")
                or response.get("error")
                or f"{action_type} failed (status={status!r})"
            )
        raise ActionError(message, action=action_type, response=response)
    return response
