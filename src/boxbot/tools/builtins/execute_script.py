"""execute_script tool — run Python in the sandbox.

The agent's universal gateway to the SDK and general-purpose computation.
Writes script to data/sandbox/scripts/{uuid}.py, executes via the sandbox
venv python3, captures stdout/stderr, and parses SDK action lines.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from boxbot.core.config import get_config
from boxbot.tools.base import Tool

logger = logging.getLogger(__name__)

# SDK action marker emitted by boxbot_sdk.transport
SDK_ACTION_MARKER = "__BOXBOT_SDK_ACTION__"

# Directories relative to project root
SCRIPTS_DIR = Path("data/sandbox/scripts")
OUTPUT_DIR = Path("data/sandbox/output")

# Environment variable allowlist — only these are passed to sandbox scripts.
# All secrets (API keys, tokens, credentials) are excluded by default.
# The agent can pass additional vars via the env_vars parameter, but those
# are explicitly chosen per-invocation (e.g., a skill's own API key).
SAFE_ENV_VARS = frozenset({
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUNBUFFERED",
    "TERM",
})


class ExecuteScriptTool(Tool):
    """Run a Python script in the sandbox environment."""

    name = "execute_script"
    description = (
        "Run a Python script in the sandboxed environment. The script can "
        "import from boxbot_sdk to create displays, manage photos, install "
        "packages, query memories, build skills, or do general computation. "
        "Use this for any operation not covered by the other 8 dedicated tools."
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

    async def execute(self, **kwargs: Any) -> str:
        script: str = kwargs["script"]
        description: str = kwargs["description"]
        env_vars: dict[str, str] = kwargs.get("env_vars") or {}

        logger.info("execute_script: %s", description)

        # Ensure directories exist
        SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Write script to file
        script_id = uuid4().hex[:12]
        script_path = SCRIPTS_DIR / f"{script_id}.py"
        script_path.write_text(script, encoding="utf-8")

        # Get sandbox config
        try:
            config = get_config()
            venv_python = Path(config.sandbox.venv_path) / "bin" / "python3"
            timeout = config.sandbox.timeout
            sandbox_user = config.sandbox.user
        except RuntimeError:
            # Config not loaded — use defaults (development mode)
            config = None
            venv_python = Path("data/sandbox/venv/bin/python3")
            timeout = 30
            sandbox_user = "boxbot-sandbox"

        # Build environment — allowlist only, no secrets leak through.
        # Only explicitly safe variables are passed to the sandbox.
        env = {k: v for k, v in os.environ.items() if k in SAFE_ENV_VARS}
        # Inject caller-provided env vars (e.g., a skill's own API key)
        env.update(env_vars)

        # Build command — in production, run as sandbox user with seccomp
        enforce = config.sandbox.enforce if config is not None else False
        if enforce:
            # Production: privilege-drop via sudo + seccomp via bwrap
            seccomp_profile = Path(config.sandbox.seccomp_profile)
            if seccomp_profile.exists():
                cmd = [
                    "bwrap",
                    "--ro-bind", "/", "/",
                    "--bind", str(SCRIPTS_DIR.resolve()), str(SCRIPTS_DIR.resolve()),
                    "--bind", str(OUTPUT_DIR.resolve()), str(OUTPUT_DIR.resolve()),
                    "--seccomp", "3",  # fd 3 for seccomp filter
                    "--", "sudo", "-u", sandbox_user,
                    str(venv_python), str(script_path),
                ]
                logger.debug(
                    "Sandbox enforced: sudo -u %s + seccomp", sandbox_user
                )
            else:
                cmd = [
                    "sudo", "-u", sandbox_user,
                    str(venv_python), str(script_path),
                ]
                logger.warning(
                    "Seccomp profile not found at %s — running with "
                    "sudo -u %s only (no seccomp)",
                    seccomp_profile,
                    sandbox_user,
                )
        else:
            # Development: run directly without privilege drop
            logger.warning(
                "Sandbox enforcement DISABLED (sandbox.enforce=false) — "
                "script runs as current user without isolation"
            )
            cmd = [str(venv_python), str(script_path)]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(Path.cwd()),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Script %s timed out after %ds", script_id, timeout
            )
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
        except FileNotFoundError:
            return json.dumps({
                "status": "error",
                "error": (
                    f"Sandbox python not found at {venv_python}. "
                    "The sandbox venv may not be set up yet."
                ),
                "script_id": script_id,
            })

        stdout_str = stdout_bytes.decode("utf-8", errors="replace")
        stderr_str = stderr_bytes.decode("utf-8", errors="replace")

        # Parse SDK actions from stdout
        sdk_actions: list[dict[str, Any]] = []
        output_lines: list[str] = []

        for line in stdout_str.splitlines():
            if SDK_ACTION_MARKER in line:
                # Extract JSON after the marker
                marker_idx = line.index(SDK_ACTION_MARKER) + len(SDK_ACTION_MARKER)
                json_str = line[marker_idx:].strip()
                try:
                    action = json.loads(json_str)
                    sdk_actions.append(action)
                    logger.debug("SDK action: %s", action.get("action", "unknown"))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse SDK action: %s", json_str[:200])
                    output_lines.append(line)
            else:
                output_lines.append(line)

        regular_output = "\n".join(output_lines).strip()

        # Process SDK actions (stub — in production, these are applied by
        # the main process: file creation, DB writes, approval queuing)
        action_results: list[dict[str, Any]] = []
        for action in sdk_actions:
            result = await _process_sdk_action(action)
            action_results.append(result)

        # Build response
        response: dict[str, Any] = {
            "status": "success" if proc.returncode == 0 else "error",
            "exit_code": proc.returncode,
            "script_id": script_id,
        }

        if regular_output:
            response["output"] = regular_output
        if stderr_str.strip():
            response["stderr"] = stderr_str.strip()
        if action_results:
            response["sdk_actions"] = action_results

        return json.dumps(response)


async def _process_sdk_action(action: dict[str, Any]) -> dict[str, Any]:
    """Process a single SDK action emitted by a sandbox script.

    In production, this routes actions to the appropriate subsystems:
    - display.create -> display manager (with approval queue)
    - skill.create -> skill manager (auto-activate)
    - packages.request -> package approval queue
    - memory.save -> memory store
    - etc.

    For now, returns a stub acknowledgment.
    """
    action_type = action.get("action", "unknown")
    logger.info("Processing SDK action: %s", action_type)

    return {
        "action": action_type,
        "status": "processed",
        "message": f"SDK action '{action_type}' acknowledged.",
    }
