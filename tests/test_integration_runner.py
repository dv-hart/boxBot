"""Tests for the integration runner.

These run the runner end-to-end against fake integrations under a
temp dir. The sandbox is bypassed (``BOXBOT_SANDBOX_ENFORCE=0``) so
the integration script runs as the test user — fast, no sudo.
"""

from __future__ import annotations

import asyncio
import os
import textwrap
from pathlib import Path

import pytest

from boxbot.integrations import logs as run_logs
from boxbot.integrations import runner as run_mod
from boxbot.integrations.loader import _DEFAULT_INTEGRATIONS_ROOT


def _make_integration(
    root: Path,
    name: str,
    *,
    description: str = "Test integration.",
    inputs: str = "",
    timeout: int = 10,
    script: str,
    secrets: str = "",
) -> None:
    """Write a minimal integration directory under ``root``."""
    d = root / name
    d.mkdir(parents=True)
    parts = [f"name: {name}", f"description: {description}", f"timeout: {timeout}"]
    if inputs:
        parts.append("inputs:\n" + textwrap.indent(inputs, "  "))
    if secrets:
        parts.append("secrets:\n" + textwrap.indent(secrets, "  "))
    (d / "manifest.yaml").write_text("\n".join(parts) + "\n")
    (d / "script.py").write_text(script)


def _patch_integrations_root(monkeypatch, root: Path) -> None:
    """Point the loader and runner at our temp root."""
    from boxbot.integrations import loader as loader_mod

    monkeypatch.setattr(loader_mod, "_DEFAULT_INTEGRATIONS_ROOT", root)


def _patch_logs_root(monkeypatch, root: Path) -> None:
    """Point the logs DB at our temp root so tests don't share state."""
    monkeypatch.setattr(run_logs, "_db_path", lambda root=None: root_path(root) if root else (root_dir / "runs.db"))


# We point the logs DB by passing root= explicitly to record_run via
# a monkeypatch on the runner.
@pytest.fixture
def isolated_logs(monkeypatch, tmp_path):
    log_root = tmp_path / "logs"
    log_root.mkdir()
    real_record = run_logs.record_run
    real_list = run_logs.list_runs

    def record_run(**kwargs):
        kwargs.setdefault("root", log_root)
        return real_record(**kwargs)

    def list_runs(name, **kwargs):
        kwargs.setdefault("root", log_root)
        return real_list(name, **kwargs)

    monkeypatch.setattr(run_logs, "record_run", record_run)
    monkeypatch.setattr(run_logs, "list_runs", list_runs)
    return log_root


@pytest.fixture(autouse=True)
def disable_sandbox_enforcement(monkeypatch):
    """Run integration scripts as the test user, no sudo."""
    monkeypatch.setenv("BOXBOT_SANDBOX_ENFORCE", "0")


@pytest.fixture(autouse=True)
def venv_python_points_at_test_python(monkeypatch):
    """Point the runner at whatever python is running the tests.

    Real deployments use the sandbox venv. Tests bypass that — we
    don't need seccomp or the sandbox user, just a working python.
    """
    import sys
    test_python = Path(sys.executable)

    # The runner reads venv_path from config; substitute by patching
    # the lazy import sites. Simplest: monkeypatch get_config.
    class _FakeSandbox:
        venv_path = str(test_python.parent.parent)
        user = None
        runtime_dir = "/tmp"
        seccomp_mode = "disabled"
        tmp_dir = "/tmp"
        timeout = 30
        scripts_dir = "/tmp"
        output_dir = "/tmp"

    class _FakeConfig:
        sandbox = _FakeSandbox()

    def fake_get_config():
        return _FakeConfig()

    from boxbot.core import config as config_mod
    monkeypatch.setattr(config_mod, "get_config", fake_get_config)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_output(tmp_path, monkeypatch, isolated_logs):
    _patch_integrations_root(monkeypatch, tmp_path)
    _make_integration(
        tmp_path,
        "echo",
        script=(
            "from boxbot_sdk.integration import inputs, return_output\n"
            "args = inputs()\n"
            "return_output({'echoed': args.get('msg', 'nothing')})\n"
        ),
    )
    result = await run_mod.run("echo", {"msg": "hello"})
    assert result["status"] == "ok"
    assert result["output"] == {"echoed": "hello"}


@pytest.mark.asyncio
async def test_run_logs_success(tmp_path, monkeypatch, isolated_logs):
    _patch_integrations_root(monkeypatch, tmp_path)
    _make_integration(
        tmp_path,
        "echo",
        script=(
            "from boxbot_sdk.integration import return_output\n"
            "return_output({'value': 42})\n"
        ),
    )
    await run_mod.run("echo", {})
    runs = run_logs.list_runs("echo")
    assert len(runs) == 1
    assert runs[0]["status"] == "ok"
    assert runs[0]["output"] == {"value": 42}


@pytest.mark.asyncio
async def test_unknown_integration_raises(tmp_path, monkeypatch, isolated_logs):
    _patch_integrations_root(monkeypatch, tmp_path)
    with pytest.raises(run_mod.IntegrationRunError, match="unknown"):
        await run_mod.run("ghost", {})


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_required_input_raises(tmp_path, monkeypatch, isolated_logs):
    _patch_integrations_root(monkeypatch, tmp_path)
    _make_integration(
        tmp_path,
        "weather",
        inputs=(
            "lat:\n  type: float\n  required: true\n"
            "lon:\n  type: float\n  required: true\n"
        ),
        script=(
            "from boxbot_sdk.integration import return_output\n"
            "return_output({'ok': True})\n"
        ),
    )
    with pytest.raises(run_mod.IntegrationRunError, match="lat|required"):
        await run_mod.run("weather", {"lon": -122.7})


@pytest.mark.asyncio
async def test_default_input_applied(tmp_path, monkeypatch, isolated_logs):
    _patch_integrations_root(monkeypatch, tmp_path)
    _make_integration(
        tmp_path,
        "weather",
        inputs="days:\n  type: int\n  default: 5\n",
        script=(
            "from boxbot_sdk.integration import inputs, return_output\n"
            "return_output({'days_seen': inputs()['days']})\n"
        ),
    )
    result = await run_mod.run("weather", {})
    assert result["output"] == {"days_seen": 5}


@pytest.mark.asyncio
async def test_unknown_input_rejected(tmp_path, monkeypatch, isolated_logs):
    _patch_integrations_root(monkeypatch, tmp_path)
    _make_integration(
        tmp_path,
        "weather",
        inputs="lat:\n  type: float\n  required: true\n",
        script=(
            "from boxbot_sdk.integration import return_output\n"
            "return_output({'ok': True})\n"
        ),
    )
    with pytest.raises(run_mod.IntegrationRunError, match="unexpected"):
        await run_mod.run("weather", {"lat": 45.5, "junk": "bad"})


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_return_output_is_error(tmp_path, monkeypatch, isolated_logs):
    _patch_integrations_root(monkeypatch, tmp_path)
    _make_integration(
        tmp_path,
        "silent",
        script="x = 1 + 1\n",
    )
    result = await run_mod.run("silent", {})
    assert result["status"] == "error"
    assert "did not call return_output" in result["error"]
    runs = run_logs.list_runs("silent")
    assert runs[0]["status"] == "error"


@pytest.mark.asyncio
async def test_script_exception_is_error(tmp_path, monkeypatch, isolated_logs):
    _patch_integrations_root(monkeypatch, tmp_path)
    _make_integration(
        tmp_path,
        "boom",
        script="raise RuntimeError('kaboom')\n",
    )
    result = await run_mod.run("boom", {})
    assert result["status"] == "error"
    assert result.get("exit_code") not in (0, None)
    runs = run_logs.list_runs("boom")
    assert runs[0]["status"] == "error"


@pytest.mark.asyncio
async def test_timeout_kills_script(tmp_path, monkeypatch, isolated_logs):
    _patch_integrations_root(monkeypatch, tmp_path)
    _make_integration(
        tmp_path,
        "slow",
        timeout=1,
        script=(
            "import time\n"
            "time.sleep(5)\n"
            "from boxbot_sdk.integration import return_output\n"
            "return_output({'ok': True})\n"
        ),
    )
    result = await run_mod.run("slow", {})
    assert result["status"] == "timeout"
    assert "1s" in result["error"]
    runs = run_logs.list_runs("slow")
    assert runs[0]["status"] == "timeout"


@pytest.mark.asyncio
async def test_invalid_json_output_is_error(tmp_path, monkeypatch, isolated_logs):
    _patch_integrations_root(monkeypatch, tmp_path)
    _make_integration(
        tmp_path,
        "garbled",
        script=(
            "import os\n"
            "with open(os.environ['BOXBOT_INTEGRATION_OUTPUT_PATH'], 'w') as f:\n"
            "    f.write('not json {')\n"
        ),
    )
    result = await run_mod.run("garbled", {})
    assert result["status"] == "error"
    assert "invalid JSON" in result["error"]
