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
"""

from __future__ import annotations

import json
import sys
import threading

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
