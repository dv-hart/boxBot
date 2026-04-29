"""Long-lived sandbox process tied to one conversation.

Boots when the conversation activates (wake word, trigger fire, inbound
text), serves all of that conversation's ``execute_script`` calls, and
shuts down when the conversation ends. This trades the per-call startup
cost (sudo + python + import bb, ~200 ms) for a process that lives a
few minutes during a conversation, and keeps Python state — captured
images, parsed CSVs, partial computations — across turns.

Wire model:

    BoxBotAgent (host)              SandboxRunner.proc (boxbot-sandbox)
    ──────────────────              ──────────────────────────────────
    start()  ─ stdin ──────────────► _sandbox_server.main()
                                          loop: read JSON request line
                                                exec(code, shared_globals)
                                                emit DONE marker
    run_script(code) ─stdin {op:run}──►   exec runs, emits SDK actions
                          ◄──stdout── output lines / SDK action lines
                          ──stdin──►   action responses
                          ◄──stdout── __BOXBOT_SANDBOX_DONE__:{...}
    return body, attachments

    stop()   ─ stdin {op:exit} ────►  return; subprocess exits

Concurrency: one script at a time per conversation (asyncio.Lock). A
single sandbox process must not interleave two scripts because they'd
share globals and stdin. Two conversations each get their own runner
and run truly in parallel.

Failure mode: if a script times out or the subprocess dies mid-run, the
runner is considered poisoned and is torn down. The next ``run_script``
call on a poisoned runner will fail until ``start()`` is invoked again.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from boxbot.tools._sandbox_actions import ActionContext, process_action

logger = logging.getLogger(__name__)

SDK_ACTION_MARKER = "__BOXBOT_SDK_ACTION__:"
DONE_MARKER = "__BOXBOT_SANDBOX_DONE__:"
SERVER_SCRIPT = Path(__file__).resolve().parent / "_sandbox_server.py"

# The server lives under the project source tree, which is owned by the
# operator and (on stock Linux) sits behind a 0700 home directory the
# sandbox user cannot traverse. We read the file at host-side import
# time and pass its contents inline via ``python3 -c`` so the sandbox
# subprocess never needs to touch the project tree. This also means the
# sandbox always runs the *currently checked-out* server code, even if
# the operator forgot to redeploy after a server change.
try:
    _SERVER_CODE = SERVER_SCRIPT.read_text(encoding="utf-8")
except OSError:  # pragma: no cover — only hit if the package is broken
    _SERVER_CODE = ""

_SAFE_ENV_KEYS = frozenset({
    "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    "PYTHONPATH", "VIRTUAL_ENV", "PYTHONDONTWRITEBYTECODE",
    "PYTHONUNBUFFERED",
})


class SandboxRunner:
    """One long-lived sandbox subprocess shared by all execute_script
    calls within a single Conversation."""

    def __init__(
        self,
        *,
        venv_python: Path,
        sandbox_user: str | None,
        enforce_sandbox: bool,
        timeout: int,
        label: str = "sandbox",
    ) -> None:
        self._venv_python = venv_python
        self._sandbox_user = sandbox_user
        self._enforce_sandbox = enforce_sandbox
        self._timeout = timeout
        self._label = label

        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        # One script at a time per process — exec into shared globals
        # cannot be interleaved.
        self._lock = asyncio.Lock()
        # Marks the runner as having had a fatal failure (timeout, proc
        # died mid-script). Once poisoned, run_script raises and the
        # caller should rebuild a new runner.
        self._poisoned = False
        self._start_lock = asyncio.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return (
            self._proc is not None
            and self._proc.returncode is None
            and not self._poisoned
        )

    async def start(self) -> None:
        """Spawn the sandbox subprocess. Idempotent."""
        async with self._start_lock:
            if self._proc is not None and self._proc.returncode is None:
                return  # already running
            if not _SERVER_CODE:
                logger.error(
                    "Sandbox server code unavailable (%s could not be "
                    "loaded at import) — runner cannot start.",
                    SERVER_SCRIPT,
                )
                self._poisoned = True
                return
            if not self._venv_python.exists():
                logger.warning(
                    "Sandbox venv missing (%s) — runner cannot start. "
                    "Run scripts/setup-sandbox.sh on this machine.",
                    self._venv_python,
                )
                self._poisoned = True
                return

            # Pass the server source via ``python3 -c`` so the sandbox
            # user doesn't need read access to the project source tree.
            if self._enforce_sandbox and self._sandbox_user:
                cmd: list[str] = [
                    "sudo", "-n", "-u", self._sandbox_user,
                    "--", str(self._venv_python), "-c", _SERVER_CODE,
                ]
            else:
                if self._sandbox_user:
                    logger.warning(
                        "Sandbox enforcement disabled "
                        "(BOXBOT_SANDBOX_ENFORCE=0) — runner uses current user"
                    )
                cmd = [str(self._venv_python), "-c", _SERVER_CODE]

            env = {
                k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS
            }
            env["PYTHONUNBUFFERED"] = "1"

            try:
                self._proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=str(Path.cwd()),
                )
            except FileNotFoundError as e:
                logger.warning(
                    "Sandbox runner spawn failed (%s): %s. "
                    "Run scripts/setup-sandbox.sh on this machine.",
                    cmd[0], e,
                )
                self._poisoned = True
                return
            except Exception:
                logger.exception("Sandbox runner spawn failed")
                self._poisoned = True
                return

            self._poisoned = False
            self._stderr_task = asyncio.create_task(
                self._pump_stderr(), name=f"{self._label}-stderr"
            )
            logger.info(
                "SandboxRunner[%s] started: pid=%d", self._label, self._proc.pid,
            )

    async def stop(self) -> None:
        """Send exit, wait briefly, escalate to terminate/kill if needed."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None and not proc.stdin.is_closing():
                try:
                    proc.stdin.write(
                        (json.dumps({"op": "exit"}) + "\n").encode("utf-8")
                    )
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        logger.warning(
                            "SandboxRunner[%s]: subprocess unresponsive to "
                            "SIGKILL (pid=%d)",
                            self._label, proc.pid,
                        )
        finally:
            if self._stderr_task is not None:
                self._stderr_task.cancel()
                try:
                    await self._stderr_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._stderr_task = None
            logger.info("SandboxRunner[%s] stopped", self._label)

    # ── Run a script ───────────────────────────────────────────────

    async def run_script(
        self,
        code: str,
        *,
        env_vars: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], list[Path]]:
        """Run ``code`` in the sandbox process; return (body, attachments).

        Body fields: ``status`` (``success``/``error``/``exit``),
        ``script_id``, optional ``output`` (concatenated stdout lines),
        ``stderr`` (traceback if the script raised), ``sdk_actions``
        (list of action records). Caller assembles the final tool result.

        Raises RuntimeError if the runner is not running or has been
        poisoned by a prior fatal failure.
        """
        if not self.is_running:
            raise RuntimeError(
                f"Sandbox runner[{self._label}] not running"
            )

        async with self._lock:
            if not self.is_running:  # re-check after lock; stop() may have run
                raise RuntimeError(
                    f"Sandbox runner[{self._label}] stopped while waiting"
                )

            ctx = ActionContext()
            output_lines: list[str] = []
            script_id = uuid4().hex[:12]
            request = {
                "op": "run",
                "script_id": script_id,
                "code": code,
                "env_vars": env_vars or {},
            }

            assert self._proc is not None
            assert self._proc.stdin is not None
            try:
                self._proc.stdin.write(
                    (json.dumps(request) + "\n").encode("utf-8")
                )
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as e:
                self._poisoned = True
                raise RuntimeError(
                    f"Sandbox stdin write failed: {e}"
                ) from e

            try:
                done_payload = await asyncio.wait_for(
                    self._pump_stdout_until_done(ctx, output_lines),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Sandbox script %s timed out after %ds; tearing down "
                    "runner[%s] (state may be inconsistent)",
                    script_id, self._timeout, self._label,
                )
                self._poisoned = True
                # Kill the subprocess; caller will rebuild if needed.
                await self.stop()
                return (
                    {
                        "status": "error",
                        "script_id": script_id,
                        "error": (
                            f"Script timed out after {self._timeout} seconds"
                        ),
                    },
                    [],
                )
            except RuntimeError:
                self._poisoned = True
                raise

            body: dict[str, Any] = {
                "status": done_payload.get("status", "success"),
                "script_id": script_id,
            }
            if output_lines:
                joined = "\n".join(output_lines).strip()
                if joined:
                    body["output"] = joined
            error_text = done_payload.get("error")
            if error_text:
                body["stderr"] = error_text
            if ctx.action_log:
                body["sdk_actions"] = ctx.action_log
            return body, list(ctx.image_attachments)

    # ── Internals ──────────────────────────────────────────────────

    async def _pump_stdout_until_done(
        self,
        ctx: ActionContext,
        output_lines: list[str],
    ) -> dict[str, Any]:
        """Read lines from sandbox stdout until the DONE marker arrives.

        Routes SDK action lines to ``process_action`` and writes the
        host's response back to the sandbox's stdin (the existing
        bidirectional protocol). Plain stdout lines accumulate into
        ``output_lines``. Returns the parsed DONE payload.
        """
        assert self._proc is not None
        assert self._proc.stdout is not None
        assert self._proc.stdin is not None

        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                raise RuntimeError(
                    "Sandbox process exited mid-script "
                    "(no DONE marker received)"
                )
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")

            if line.startswith(DONE_MARKER):
                payload_text = line[len(DONE_MARKER):]
                try:
                    return json.loads(payload_text)
                except json.JSONDecodeError:
                    return {
                        "status": "error",
                        "error": f"bad DONE payload: {payload_text[:200]}",
                    }

            idx = line.find(SDK_ACTION_MARKER)
            if idx == -1:
                output_lines.append(line)
                continue

            json_str = line[idx + len(SDK_ACTION_MARKER):].strip()
            try:
                action = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning(
                    "bad SDK action payload: %s", json_str[:200]
                )
                output_lines.append(line)
                continue

            response = await process_action(action, ctx)

            if action.get("_expects_response"):
                try:
                    self._proc.stdin.write(
                        (json.dumps(response) + "\n").encode("utf-8")
                    )
                    await self._proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    raise RuntimeError(
                        "Sandbox stdin closed before response could be sent"
                    )

    async def _pump_stderr(self) -> None:
        """Drain stderr to the host log; never blocks the main read loop."""
        assert self._proc is not None
        assert self._proc.stderr is not None
        while True:
            raw = await self._proc.stderr.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line:
                logger.debug("sandbox[%s] stderr: %s", self._label, line)
