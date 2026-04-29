"""execute_script tool — run Python in the sandbox.

The agent's universal gateway to the SDK and general-purpose computation.

Two execution paths:

1. **Conversation-scoped runner** (preferred). When invoked from inside
   a Conversation, the script runs in that conversation's long-lived
   ``SandboxRunner`` subprocess — eager-started at conversation creation,
   shut down when the conversation ends. Python state (last captured
   image, parsed CSV, partial computation) persists across turns.
2. **Per-call subprocess** (fallback). When no conversation context is
   available — tests, ad-hoc invocations — the script runs in a fresh
   subprocess per call, the way it always did.

Both paths use the same ``__BOXBOT_SDK_ACTION__:`` action protocol and
the same ``ActionContext`` accumulator, so tool result shape is
identical from the agent's perspective.

Tool result shape:
- Text-only runs return a JSON-encoded ``str`` containing status,
  exit_code, output, stderr, sdk_actions.
- Runs that attach images via ``bb.workspace.view(<image>)`` /
  ``bb.camera.capture`` / ``bb.photos.view`` return a ``list`` of
  content blocks: one ``text`` block with the JSON body followed by one
  ``image`` block per attachment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from boxbot.tools._sandbox_actions import (
    ActionContext,
    build_image_block,
    process_action,
)
from boxbot.tools._tool_context import get_current_conversation
from boxbot.tools.base import Tool

logger = logging.getLogger(__name__)

SDK_ACTION_MARKER = "__BOXBOT_SDK_ACTION__:"

# Fallback paths used only when the config can't be loaded (e.g. tests).
# Real runs read the paths from ``cfg.sandbox`` so a deployment can put
# the sandbox under ``/var/lib/boxbot-sandbox`` (default) or anywhere
# else without touching code.
_FALLBACK_RUNTIME_DIR = Path("/var/lib/boxbot-sandbox")


class ExecuteScriptTool(Tool):
    """Run a Python script in the sandbox environment."""

    name = "execute_script"
    description = (
        "Run a Python script in the sandboxed environment. The script can "
        "import from boxbot_sdk (aliased as bb) to manage photos, displays, "
        "memories, skills, the agent workspace (notes, CSVs), secrets, "
        "tasks, the calendar, and packages. Use this for anything not "
        "covered by a dedicated tool — composing multiple operations in a "
        "single turn, taking notes, capturing/viewing images, or general "
        "computation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": "Python source code to execute in the sandbox.",
            },
            "description": {
                "type": "string",
                "description": "Brief description of what the script does (for logging).",
            },
            "env_vars": {
                "type": "object",
                "description": (
                    "Optional environment variables to inject into the script's "
                    "environment (e.g., API credentials for a specific service)."
                ),
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["script", "description"],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: Any) -> str | list[dict[str, Any]]:
        script: str = kwargs["script"]
        description: str = kwargs["description"]
        env_vars: dict[str, str] = kwargs.get("env_vars") or {}

        logger.info("execute_script: %s", description)

        # Prefer the conversation's long-lived runner if one is up.
        # Falls through to the per-call subprocess path on any failure.
        conv = get_current_conversation()
        runner = (
            conv.sandbox_runner if conv is not None else None
        )
        if runner is not None and runner.is_running:
            try:
                body, attachments = await runner.run_script(
                    script, env_vars=env_vars
                )
                return _assemble_result(body, attachments)
            except RuntimeError as e:
                # Runner died or is poisoned — fall through to per-call.
                logger.warning(
                    "Conversation sandbox runner unusable (%s); "
                    "falling back to per-call subprocess",
                    e,
                )

        # Lazy import to avoid pulling boxbot.core at module load (which
        # triggers the core package __init__ and a registry cycle).
        from boxbot.core.config import get_config

        try:
            config = get_config()
            venv_python = Path(config.sandbox.venv_path) / "bin" / "python3"
            timeout = config.sandbox.timeout
            sandbox_user = config.sandbox.user
            scripts_dir = Path(config.sandbox.scripts_dir)
            output_dir = Path(config.sandbox.output_dir)
        except RuntimeError:
            venv_python = _FALLBACK_RUNTIME_DIR / "venv" / "bin" / "python3"
            timeout = 30
            sandbox_user = "boxbot-sandbox"
            scripts_dir = _FALLBACK_RUNTIME_DIR / "scripts"
            output_dir = _FALLBACK_RUNTIME_DIR / "output"

        # Best-effort dir create. Real runs go to /var/lib/boxbot-sandbox/
        # which the setup script chowns to boxbot-sandbox; if we're
        # called outside a configured environment (tests), the fallback
        # dirs may need creating.
        try:
            scripts_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            pass

        script_id = uuid4().hex[:12]
        script_path = scripts_dir / f"{script_id}.py"
        script_path.write_text(script, encoding="utf-8")

        SAFE_ENV_KEYS = {
            "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
            "PYTHONPATH", "VIRTUAL_ENV", "PYTHONDONTWRITEBYTECODE",
            "PYTHONUNBUFFERED",
        }
        env = {k: v for k, v in os.environ.items() if k in SAFE_ENV_KEYS}
        env.update(env_vars)
        # Force unbuffered stdout so we can stream action lines in real time.
        env["PYTHONUNBUFFERED"] = "1"

        enforce_sandbox = os.environ.get("BOXBOT_SANDBOX_ENFORCE", "1") != "0"
        if enforce_sandbox and sandbox_user:
            cmd = [
                "sudo", "-n", "-u", sandbox_user,
                "--", str(venv_python), str(script_path),
            ]
        else:
            if sandbox_user:
                logger.warning(
                    "Sandbox enforcement disabled (BOXBOT_SANDBOX_ENFORCE=0) "
                    "— script runs as current user"
                )
            cmd = [str(venv_python), str(script_path)]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(Path.cwd()),
            )
        except FileNotFoundError:
            return json.dumps({
                "status": "error",
                "error": (
                    f"Sandbox python not found at {venv_python}. "
                    "The sandbox venv may not be set up yet."
                ),
                "script_id": script_id,
            })

        ctx = ActionContext()
        output_lines: list[str] = []

        async def pump_stdout() -> None:
            assert proc.stdout is not None
            assert proc.stdin is not None
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    return
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")

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
                        proc.stdin.write(
                            (json.dumps(response) + "\n").encode("utf-8")
                        )
                        await proc.stdin.drain()
                    except (BrokenPipeError, ConnectionResetError):
                        # Sandbox exited before reading its reply.
                        return

        try:
            await asyncio.wait_for(pump_stdout(), timeout=timeout)
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            logger.warning("Script %s timed out after %ds", script_id, timeout)
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            return json.dumps({
                "status": "error",
                "error": f"Script timed out after {timeout} seconds.",
                "script_id": script_id,
            })

        assert proc.stderr is not None
        stderr_bytes = await proc.stderr.read()
        stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()
        regular_output = "\n".join(output_lines).strip()

        body: dict[str, Any] = {
            "status": "success" if proc.returncode == 0 else "error",
            "exit_code": proc.returncode,
            "script_id": script_id,
        }
        if regular_output:
            body["output"] = regular_output
        if stderr_str:
            body["stderr"] = stderr_str
        if ctx.action_log:
            body["sdk_actions"] = ctx.action_log

        return _assemble_result(body, list(ctx.image_attachments))


def _assemble_result(
    body: dict[str, Any],
    image_attachments: list[Path],
) -> str | list[dict[str, Any]]:
    """Wrap a tool body + attachments into the right Anthropic content shape.

    No attachments → JSON-encoded string. Any attachments → a list of
    content blocks (one ``text`` with the JSON body, then one ``image``
    per attachable path that passes the allowlist + size checks).
    """
    if not image_attachments:
        return json.dumps(body)
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": json.dumps(body)}
    ]
    attached = 0
    for path in image_attachments:
        block = build_image_block(path)
        if block is not None:
            blocks.append(block)
            attached += 1
    logger.info(
        "execute_script attached %d/%d image(s)",
        attached,
        len(image_attachments),
    )
    return blocks
