"""Regression tests for the long-lived sandbox runner.

The runner boots its server by composing ``_SECCOMP_PROLOGUE +
_SERVER_CODE`` and passing the result to ``python3 -c``
(``sandbox_runner._server_code_with_seccomp``). Historically
``_sandbox_server.py`` carried ``from __future__ import annotations``
below its docstring; with the prologue prepended, the composed source
raised "from __future__ imports must occur at the beginning of the
file" — the server died at startup and every ``execute_script`` call
silently fell back to per-call subprocess spawning. The warm runner
had therefore *never* run in production.

These tests pin three things so that failure mode can never silently
return:

1. The exact composed string the runner executes must ``compile()``.
2. ``_sandbox_server.py`` must also compile standalone.
3. The runner's startup path actually works end-to-end on a dev box —
   no boxbot-sandbox user, sudo, or sandbox venv required: the runner
   is exercised with ``enforce_sandbox=False`` against the current
   interpreter, with a stub ``boxbot_sdk`` on PYTHONPATH so the
   server's pre-import succeeds — and an unexpected server death is
   loud (WARNING + recorded ``failure_reason``), not silent.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import pytest

from boxbot.tools import sandbox_runner as sandbox_runner_module
from boxbot.tools.sandbox_runner import (
    _SECCOMP_PROLOGUE,
    _SERVER_CODE,
    SERVER_SCRIPT,
    SandboxRunner,
    _server_code_with_seccomp,
)


# ── Composition / compilation regressions ───────────────────────────


def test_server_code_was_loaded_at_import():
    assert _SERVER_CODE, "_sandbox_server.py could not be read at import"


def test_composed_server_code_compiles():
    """The exact string the runner passes to ``python3 -c`` must compile.

    This is the regression for the ``from __future__ import annotations``
    placement bug: any statement in ``_sandbox_server.py`` that is only
    legal at the very top of a module breaks the composition.
    """
    compile(_SECCOMP_PROLOGUE + _SERVER_CODE, "<sandbox-server-composed>", "exec")


def test_server_code_with_seccomp_is_the_composition():
    """Pin that the helper really is prologue + server source, so the
    compile test above covers what the runner actually executes."""
    assert _server_code_with_seccomp() == _SECCOMP_PROLOGUE + _SERVER_CODE


def test_sandbox_server_compiles_standalone():
    source = SERVER_SCRIPT.read_text(encoding="utf-8")
    compile(source, str(SERVER_SCRIPT), "exec")


def test_seccomp_prologue_compiles_standalone():
    compile(_SECCOMP_PROLOGUE, "<seccomp-prologue>", "exec")


# ── Runner startup path (no sandbox user / venv needed) ─────────────


def _make_runner(venv_python: Path, *, timeout: int = 20) -> SandboxRunner:
    return SandboxRunner(
        venv_python=venv_python,
        sandbox_user=None,
        enforce_sandbox=False,
        timeout=timeout,
        label="test",
    )


@pytest.fixture
def sdk_stub_env(tmp_path, monkeypatch):
    """Environment for spawning the real server with the dev interpreter.

    - Stub ``boxbot_sdk`` package on PYTHONPATH so the server's
      pre-import succeeds (the real SDK only exists in the sandbox venv).
    - ``BOXBOT_SECCOMP_DISABLE=1`` so the prologue skips filter install
      regardless of what any loaded config says (no libseccomp on dev,
      and we must not filter the test interpreter anyway).
    """
    pkg = tmp_path / "boxbot_sdk"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "stub = True\n", encoding="utf-8"
    )
    existing = os.environ.get("PYTHONPATH")
    pythonpath = str(tmp_path) + (os.pathsep + existing if existing else "")
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setenv("BOXBOT_SECCOMP_DISABLE", "1")
    return tmp_path


async def test_runner_starts_and_runs_scripts(sdk_stub_env):
    """End-to-end startup: spawn the composed server, run scripts, and
    confirm Python state persists across runs (the whole point of the
    warm runner). Before the __future__ fix this failed immediately:
    the server died with a SyntaxError before reading stdin."""
    runner = _make_runner(Path(sys.executable))
    await runner.start()
    try:
        assert runner.is_running, (
            f"runner failed to start: {runner.failure_reason}"
        )

        body, attachments = await runner.run_script(
            "print('hello from server')"
        )
        assert body["status"] == "success"
        assert "hello from server" in body.get("output", "")
        assert attachments == []

        # Cross-turn state: a variable bound in one run is visible in
        # the next (shared globals in the long-lived server process).
        body2, _ = await runner.run_script("x = 41")
        assert body2["status"] == "success"
        body3, _ = await runner.run_script("print(x + 1)")
        assert body3["status"] == "success"
        assert "42" in body3.get("output", "")

        # Script errors are reported, not fatal to the runner.
        body4, _ = await runner.run_script("raise ValueError('boom')")
        assert body4["status"] == "error"
        assert "boom" in body4.get("stderr", "")
        assert runner.is_running
    finally:
        await runner.stop()
    assert not runner.is_running


async def test_runner_start_is_idempotent(sdk_stub_env):
    runner = _make_runner(Path(sys.executable))
    await runner.start()
    try:
        assert runner.is_running
        pid_attr = runner._proc.pid  # noqa: SLF001 — pin same process
        await runner.start()
        assert runner.is_running
        assert runner._proc.pid == pid_attr  # noqa: SLF001
    finally:
        await runner.stop()


async def test_runner_poisoned_when_venv_missing(tmp_path):
    """Startup failure is recorded, not just swallowed: the runner
    poisons itself, exposes a failure_reason, and run_script raises."""
    missing = tmp_path / "venv" / "bin" / "python3"
    runner = _make_runner(missing)
    await runner.start()
    assert not runner.is_running
    assert runner.failure_reason is not None
    assert str(missing) in runner.failure_reason
    with pytest.raises(RuntimeError):
        await runner.run_script("print(1)")


async def test_unexpected_server_death_is_loud(
    sdk_stub_env, monkeypatch, caplog
):
    """If the server process dies on its own (the silent-fallback bug's
    signature), the runner must log a WARNING with the exit code and
    stderr tail, and record a failure_reason for execute_script to
    surface — never a silent DEBUG-only death."""
    monkeypatch.setattr(
        sandbox_runner_module,
        "_SERVER_CODE",
        "import sys\nsys.stderr.write('synthetic startup failure\\n')\n"
        "sys.exit(7)\n",
    )
    runner = _make_runner(Path(sys.executable))
    with caplog.at_level(
        logging.WARNING, logger="boxbot.tools.sandbox_runner"
    ):
        await runner.start()
        for _ in range(200):  # up to ~10s for the death to be noticed
            if runner.failure_reason is not None:
                break
            await asyncio.sleep(0.05)

    assert not runner.is_running
    assert runner.failure_reason is not None
    assert "rc=7" in runner.failure_reason
    assert "synthetic startup failure" in runner.failure_reason

    warnings = [
        r for r in caplog.records if "exited unexpectedly" in r.getMessage()
    ]
    assert warnings, "server death did not produce a WARNING log"
    assert any("synthetic startup failure" in r.getMessage() for r in warnings)

    await runner.stop()


async def test_deliberate_stop_is_not_loud(sdk_stub_env, caplog):
    """A clean stop() must not trip the unexpected-death warning."""
    runner = _make_runner(Path(sys.executable))
    await runner.start()
    assert runner.is_running, runner.failure_reason
    with caplog.at_level(
        logging.WARNING, logger="boxbot.tools.sandbox_runner"
    ):
        await runner.stop()
        await asyncio.sleep(0.1)  # let any stray pump callbacks land
    assert not any(
        "exited unexpectedly" in r.getMessage() for r in caplog.records
    )
