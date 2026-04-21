"""Internal transport layer for SDK ↔ main process communication.

SDK actions are emitted as structured JSON lines on stdout, prefixed with
a marker so execute_script can separate them from regular print() output.
"""

from __future__ import annotations

import json
import sys
import threading

# Marker prefix that execute_script looks for to identify SDK actions
_MARKER = "__BOXBOT_SDK_ACTION__:"

_write_lock = threading.Lock()


def emit_action(action_type: str, payload: dict) -> None:
    """Emit a structured SDK action to stdout.

    Args:
        action_type: The action type, e.g. "display.save", "memory.save"
        payload: The action payload dict
    """
    message = {"_sdk": action_type, **payload}
    line = _MARKER + json.dumps(message, separators=(",", ":"))
    with _write_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def collect_response(timeout: int = 30) -> dict:
    """Read a response from stdin (blocking call).

    Used for operations that need a result from the main process,
    such as packages.request() which blocks until user approval.

    .. note::
        This function uses ``select.select()`` on stdin, which requires a
        Unix-like OS (Linux, macOS). It does not work on Windows because
        ``select`` cannot monitor non-socket file descriptors there. This
        is fine for the Raspberry Pi target but will raise an error in
        Windows development environments. Use WSL or a Linux VM for local
        testing of SDK scripts that call ``collect_response()``.

    Args:
        timeout: Maximum seconds to wait for a response.

    Returns:
        Parsed JSON response dict from the main process.

    Raises:
        TimeoutError: If no response within timeout.
        RuntimeError: If stdin is closed or response is malformed.
        OSError: On Windows where select() does not support stdin.
    """
    import select

    # Use select for timeout support on the stdin file descriptor
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
