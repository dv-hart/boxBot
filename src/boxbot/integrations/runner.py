"""Integration runner — execute a registered integration in the sandbox.

The runner is the single entrypoint for invoking an integration,
called from anywhere in the main process:

- The dispatcher's ``integrations.get`` handler (when an agent script
  in the sandbox calls ``bb.integrations.get(...)``).
- Display data sources (``WeatherSource`` and friends) on their own
  refresh cadence.
- Future scheduler triggers.

Pipe model: every call spawns a fresh sandbox subprocess, runs the
integration's ``script.py``, captures its returned output via
``BOXBOT_INTEGRATION_OUTPUT_PATH``, logs the run, and returns. There
is no caching here — consumers cache at their own cadence.

Stdout from the integration script can interleave ``bb.*`` SDK
action lines (e.g. ``bb.secrets.get(...)``); the runner demuxes the
same way ``execute_script`` does, dispatching actions through
:func:`boxbot.tools._sandbox_actions.process_action`.

Hard timeout: the manifest's ``timeout`` field. Scripts that exceed
it are killed and logged as ``status: timeout``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from boxbot.integrations import logs as run_logs
from boxbot.integrations.loader import get_integration
from boxbot.integrations.manifest import IntegrationMeta

logger = logging.getLogger(__name__)

SDK_ACTION_MARKER = "__BOXBOT_SDK_ACTION__:"


class IntegrationRunError(RuntimeError):
    """Raised for caller-correctable errors (unknown integration, bad inputs).

    The dispatcher catches this and returns ``status: "error"``; direct
    callers (display data sources) should handle it explicitly.
    """


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_inputs(meta: IntegrationMeta, supplied: dict[str, Any]) -> dict[str, Any]:
    """Apply manifest defaults and check required fields.

    Type coercion is intentionally absent — the manifest is descriptive
    in v1, and the script can do its own coercion if it cares.
    """
    out: dict[str, Any] = dict(supplied)
    for name, spec in meta.inputs.items():
        if name in out:
            continue
        if "default" in spec:
            out[name] = spec["default"]
        elif spec.get("required"):
            raise IntegrationRunError(
                f"integration '{meta.name}' requires input '{name}'"
            )
    # Reject unknown inputs only if the manifest declares any inputs at all.
    # An empty inputs section means "I accept whatever; pass it through."
    if meta.inputs:
        unknown = set(out.keys()) - set(meta.inputs.keys())
        if unknown:
            raise IntegrationRunError(
                f"integration '{meta.name}' got unexpected inputs: "
                f"{sorted(unknown)} (declared: {sorted(meta.inputs.keys())})"
            )
    return out


# ---------------------------------------------------------------------------
# Subprocess spawn (mirrors execute_script's per-call path)
# ---------------------------------------------------------------------------


def _build_command(
    meta: IntegrationMeta,
    *,
    venv_python: Path,
    bootstrap_path: Path,
    sandbox_user: str | None,
    enforce_sandbox: bool,
    secret_env_names: list[str] | None = None,
) -> list[str]:
    if enforce_sandbox and sandbox_user:
        preserve = [
            "BOXBOT_SECCOMP_MODE", "BOXBOT_SECCOMP_DISABLE",
            "BOXBOT_SKILLS_ROOT",
            "BOXBOT_INTEGRATION_INPUTS_PATH", "BOXBOT_INTEGRATION_OUTPUT_PATH",
        ]
        if secret_env_names:
            # sudo strips env vars not in --preserve-env=, so any
            # BOXBOT_SECRET_* we set in the env dict has to be named here
            # too, or it never reaches the script.
            preserve.extend(secret_env_names)
        return [
            "sudo", "-n",
            "--preserve-env=" + ",".join(preserve),
            "-u", sandbox_user,
            "--", str(venv_python), str(bootstrap_path),
            str(meta.script_path),
        ]
    return [str(venv_python), str(bootstrap_path), str(meta.script_path)]


def _build_env(
    *,
    inputs_path: Path,
    output_path: Path,
    meta: IntegrationMeta | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Subset of os.environ + integration-specific vars + declared secrets.

    Returns ``(env, secret_env_names)``. ``secret_env_names`` is the
    list of ``BOXBOT_SECRET_*`` keys that were resolved successfully —
    the caller passes them to :func:`_build_command` so sudo's
    ``--preserve-env`` lets them through privilege drop. Missing
    secrets are logged but don't block the run; the script can detect
    ``os.environ.get(...)`` returning ``None`` and surface a helpful
    error in its output.
    """
    SAFE_ENV_KEYS = {
        "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
        "PYTHONPATH", "VIRTUAL_ENV", "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
    }
    env = {k: v for k, v in os.environ.items() if k in SAFE_ENV_KEYS}
    env["PYTHONUNBUFFERED"] = "1"
    env["BOXBOT_INTEGRATION_INPUTS_PATH"] = str(inputs_path)
    env["BOXBOT_INTEGRATION_OUTPUT_PATH"] = str(output_path)

    secret_env_names: list[str] = []
    if meta is not None and meta.secrets:
        from boxbot.secrets import get_secret_store

        store = get_secret_store()
        for name in meta.secrets:
            value = store.load(name)
            if value is None:
                logger.warning(
                    "integration '%s' declared secret '%s' but it is not "
                    "stored — script will see an empty env var",
                    meta.name, name,
                )
                continue
            env_key = f"BOXBOT_SECRET_{name}"
            env[env_key] = value
            secret_env_names.append(env_key)

    # Reuse the same seccomp policy as agent scripts.
    try:
        from boxbot.core.config import get_config
        cfg = get_config()
        env["BOXBOT_SECCOMP_MODE"] = cfg.sandbox.seccomp_mode
    except Exception:  # noqa: BLE001
        env["BOXBOT_SECCOMP_MODE"] = "disabled"
    if os.environ.get("BOXBOT_SECCOMP_DISABLE") == "1":
        env["BOXBOT_SECCOMP_DISABLE"] = "1"

    # Skills/ on sys.path for parity with execute_script (an integration
    # script may import a bundled skill helper if it really wants to).
    try:
        from boxbot.skills.loader import _DEFAULT_SKILLS_ROOT
        env["BOXBOT_SKILLS_ROOT"] = str(_DEFAULT_SKILLS_ROOT)
    except Exception:  # noqa: BLE001
        pass

    return env, secret_env_names


# ---------------------------------------------------------------------------
# Action stream pump (drains stdout, dispatches bb.* actions)
# ---------------------------------------------------------------------------


async def _pump_actions(proc: asyncio.subprocess.Process) -> tuple[list[str], list[dict[str, Any]]]:
    """Read stdout line-by-line, dispatching bb.* action markers.

    Returns ``(non_action_output_lines, action_log)``. Same protocol as
    ``execute_script.pump_stdout`` so integration scripts can use the
    full bb.* surface — ``bb.secrets.get`` is the load-bearing case.
    """
    # Lazy import to dodge the boxbot.core init chain at module import time.
    from boxbot.tools._sandbox_actions import ActionContext, process_action

    ctx = ActionContext()
    output_lines: list[str] = []

    assert proc.stdout is not None
    assert proc.stdin is not None
    while True:
        raw = await proc.stdout.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        idx = line.find(SDK_ACTION_MARKER)
        if idx == -1:
            output_lines.append(line)
            continue
        json_str = line[idx + len(SDK_ACTION_MARKER):].strip()
        try:
            action = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("bad SDK action payload from integration: %s", json_str[:200])
            output_lines.append(line)
            continue
        response = await process_action(action, ctx)
        if action.get("_expects_response"):
            try:
                proc.stdin.write((json.dumps(response) + "\n").encode("utf-8"))
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                break

    return output_lines, ctx.action_log


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def run(
    name: str,
    inputs: dict[str, Any] | None = None,
    *,
    timeout_override: int | None = None,
) -> dict[str, Any]:
    """Execute integration ``name`` with ``inputs`` and return its output.

    Always logs the run to ``data/integrations/runs.db`` regardless of
    outcome. Raises :class:`IntegrationRunError` for caller errors
    (unknown integration, missing required input). Failures inside the
    script (non-zero exit, no output, timeout) come back as a structured
    response so the caller can decide what to do.
    """
    inputs = inputs or {}
    meta = get_integration(name)
    if meta is None:
        raise IntegrationRunError(f"unknown integration '{name}'")

    validated_inputs = _validate_inputs(meta, inputs)
    timeout = timeout_override if timeout_override is not None else meta.timeout

    started_at = run_logs.now()

    inputs_path: Path | None = None
    output_path: Path | None = None
    status = "error"
    output: Any = None
    error: str | None = None

    try:
        # Lazy imports to keep this module light at module-load time.
        from boxbot.core.config import get_config
        from boxbot.tools.builtins.execute_script import _resolve_bootstrap_path

        try:
            cfg = get_config()
            venv_python = Path(cfg.sandbox.venv_path) / "bin" / "python3"
            sandbox_user = cfg.sandbox.user
            runtime_dir = Path(cfg.sandbox.runtime_dir)
        except RuntimeError:
            # Fallback for tests / dev environments without a real config.
            venv_python = Path("/var/lib/boxbot-sandbox/venv/bin/python3")
            sandbox_user = "boxbot-sandbox"
            runtime_dir = Path("/var/lib/boxbot-sandbox")
        bootstrap_path = _resolve_bootstrap_path(runtime_dir)
        enforce_sandbox = os.environ.get("BOXBOT_SANDBOX_ENFORCE", "1") != "0"

        # Use temp files so the sandbox can read/write without touching
        # the project tree. tmp paths are inside `_sandbox_tmp_dir()`
        # below, which the setup script chowns to the sandbox group.
        tmp_root = _resolve_tmp_dir()
        run_id = uuid4().hex[:12]
        inputs_path = tmp_root / f"integration-{run_id}-in.json"
        output_path = tmp_root / f"integration-{run_id}-out.json"
        inputs_path.write_text(json.dumps(validated_inputs), encoding="utf-8")
        output_path.write_text("", encoding="utf-8")

        env, secret_env_names = _build_env(
            inputs_path=inputs_path, output_path=output_path, meta=meta,
        )
        cmd = _build_command(
            meta,
            venv_python=venv_python,
            bootstrap_path=bootstrap_path,
            sandbox_user=sandbox_user,
            enforce_sandbox=enforce_sandbox,
            secret_env_names=secret_env_names,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(Path.cwd()),
            )
        except FileNotFoundError as exc:
            error = f"sandbox python not found at {venv_python}: {exc}"
            return {"status": "error", "error": error}

        try:
            output_lines, _action_log = await asyncio.wait_for(
                _pump_actions(proc), timeout=timeout
            )
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            status = "timeout"
            error = f"integration '{name}' exceeded its {timeout}s timeout"
            return {"status": "timeout", "error": error}

        stderr_bytes = await proc.stderr.read() if proc.stderr is not None else b""
        stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            error = (
                f"integration '{name}' exited with code {proc.returncode}"
                + (f": {stderr_str}" if stderr_str else "")
            )
            return {"status": "error", "error": error, "exit_code": proc.returncode}

        # Read the captured output file. Empty file means the script
        # forgot to call return_output.
        try:
            raw = output_path.read_text(encoding="utf-8")
        except OSError as exc:
            error = f"failed to read integration output file: {exc}"
            return {"status": "error", "error": error}
        if not raw.strip():
            error = (
                f"integration '{name}' did not call return_output(); "
                "no output recorded"
            )
            return {"status": "error", "error": error}
        try:
            output = json.loads(raw)
        except json.JSONDecodeError as exc:
            error = f"integration '{name}' returned invalid JSON: {exc}"
            return {"status": "error", "error": error}

        status = "ok"
        return {"status": "ok", "output": output}
    finally:
        finished_at = run_logs.now()
        run_logs.record_run(
            name=name,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            inputs=validated_inputs,
            output=output if status == "ok" else None,
            error=error,
        )
        # Best-effort cleanup of temp files.
        for p in (inputs_path, output_path):
            if p is not None:
                try:
                    p.unlink()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_tmp_dir() -> Path:
    """Resolve the sandbox tmp dir (writeable by both main and sandbox).

    Mirrors :func:`boxbot.tools._sandbox_actions._sandbox_tmp_dir` but
    we re-resolve here to avoid an import cycle.
    """
    try:
        from boxbot.core.config import get_config
        cfg = get_config()
        path = Path(cfg.sandbox.tmp_dir)
    except Exception:  # noqa: BLE001
        path = Path("/var/lib/boxbot-sandbox/tmp")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        # Tests / dev: use the OS tmp.
        path = Path(tempfile.gettempdir())
    return path
