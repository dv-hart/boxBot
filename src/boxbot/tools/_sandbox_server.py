"""Server script that runs **inside the sandbox** for the lifetime of a
conversation.

Reads JSON-framed requests from stdin, execs the agent's scripts in a
shared globals dict (so state persists across turns), and emits a
``__BOXBOT_SANDBOX_DONE__:`` marker on stdout when each run completes.

Protocol (one JSON object per line, newline-terminated):

    Host → server:
        {"op": "run",  "script_id": "<id>", "code": "<src>",
                       "env_vars": {"K": "V", ...}}
        {"op": "exit"}

    Server → host:
        - regular stdout (anything the script prints)
        - ``__BOXBOT_SDK_ACTION__:{...}`` lines emitted by the bb SDK
          while the script is exec'ing (handled by the host's existing
          pump_stdout loop)
        - ``__BOXBOT_SANDBOX_DONE__:{"script_id":..,"status":..,"error":..}``
          when the script finishes (success or error)

The shared globals dict carries Python state (e.g. ``last_image``) across
runs within the conversation. It does **not** carry SDK action context —
each ``run`` gets a fresh tool result on the host side.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

DONE_MARKER = "__BOXBOT_SANDBOX_DONE__:"


def _emit_done(payload: dict[str, Any]) -> None:
    sys.stdout.write(DONE_MARKER + json.dumps(payload) + "\n")
    sys.stdout.flush()


def _emit_fatal(message: str) -> None:
    sys.stderr.write(f"sandbox server fatal: {message}\n")
    sys.stderr.flush()


def main() -> int:
    # Pre-import the SDK once so the first run isn't paying the import
    # cost. If the SDK can't load, every run is going to fail anyway —
    # better to die now with a clear error than to fail mysteriously
    # later.
    try:
        import boxbot_sdk  # noqa: F401
        import boxbot_sdk as bb
    except Exception:
        _emit_fatal(f"failed to import boxbot_sdk:\n{traceback.format_exc()}")
        return 2

    shared_globals: dict[str, Any] = {
        "__name__": "__sandbox__",
        "bb": bb,
    }

    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            return 0
        if not line:
            return 0
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _emit_fatal(f"bad request line: {line[:200]}")
            continue

        op = req.get("op")
        if op == "exit":
            return 0
        if op != "run":
            _emit_fatal(f"unknown op: {op!r}")
            continue

        script_id = str(req.get("script_id", "?"))
        code = req.get("code", "")
        env_vars = req.get("env_vars") or {}

        # Per-run env: apply, then restore so values can't leak across
        # runs in a long-lived sandbox.
        prev_env: dict[str, str | None] = {
            k: os.environ.get(k) for k in env_vars
        }
        for k, v in env_vars.items():
            os.environ[k] = str(v)

        status = "success"
        error_text: str | None = None
        try:
            try:
                compiled = compile(code, f"<sandbox:{script_id}>", "exec")
            except SyntaxError:
                status = "error"
                error_text = traceback.format_exc()
            else:
                try:
                    exec(compiled, shared_globals)  # noqa: S102
                except SystemExit as e:
                    status = "exit"
                    error_text = f"SystemExit({e.code!r})"
                except KeyboardInterrupt:
                    status = "interrupted"
                    error_text = "KeyboardInterrupt"
                except BaseException:
                    status = "error"
                    error_text = traceback.format_exc()
        finally:
            for k, prev in prev_env.items():
                if prev is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = prev

        _emit_done({
            "script_id": script_id,
            "status": status,
            "error": error_text,
        })


if __name__ == "__main__":
    sys.exit(main())
