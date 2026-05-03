"""Tests for the secret store backend, sandbox handler, and runner injection.

Covers:
- :class:`boxbot.secrets.SecretStore` round-trip and validation.
- The ``secrets.*`` sandbox action handler (store / delete / list / use).
- Integration runner injects ``BOXBOT_SECRET_<NAME>`` env vars and adds
  them to sudo's ``--preserve-env=`` list.
- ``execute_script`` resolves named secrets server-side.

Sandbox enforcement is bypassed throughout — these tests run as the
test user and inspect what the runner *would* hand to subprocess.
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from pathlib import Path

import pytest

from boxbot.secrets import SecretStore, SecretStoreError


# ---------------------------------------------------------------------------
# SecretStore unit tests
# ---------------------------------------------------------------------------


def test_store_creates_file_at_mode_0600(tmp_path):
    store = SecretStore(path=tmp_path / "secrets.json")
    store.store("POLYGON_API_KEY", "pk_live_abc")
    f = tmp_path / "secrets.json"
    assert f.exists()
    # Mode bits beyond the user-rw should all be zero.
    assert (f.stat().st_mode & 0o077) == 0


def test_store_round_trip(tmp_path):
    store = SecretStore(path=tmp_path / "secrets.json")
    result = store.store("POLYGON_API_KEY", "pk_live_abc")
    assert result["status"] == "ok"
    assert result["previous"] == "created"
    assert store.has("POLYGON_API_KEY")
    assert store.load("POLYGON_API_KEY") == "pk_live_abc"


def test_store_replace_reports_previous(tmp_path):
    store = SecretStore(path=tmp_path / "secrets.json")
    store.store("KEY", "v1")
    result = store.store("KEY", "v2")
    assert result["previous"] == "replaced"
    assert store.load("KEY") == "v2"


def test_store_rejects_bad_name(tmp_path):
    store = SecretStore(path=tmp_path / "secrets.json")
    with pytest.raises(SecretStoreError):
        store.store("lowercase", "v")
    with pytest.raises(SecretStoreError):
        store.store("123STARTS_WITH_DIGIT", "v")
    with pytest.raises(SecretStoreError):
        store.store("HAS-DASH", "v")
    with pytest.raises(SecretStoreError):
        store.store("HAS SPACE", "v")
    with pytest.raises(SecretStoreError):
        store.store("", "v")


def test_store_rejects_empty_value(tmp_path):
    store = SecretStore(path=tmp_path / "secrets.json")
    with pytest.raises(SecretStoreError):
        store.store("KEY", "")


def test_store_rejects_oversized_value(tmp_path):
    store = SecretStore(path=tmp_path / "secrets.json")
    with pytest.raises(SecretStoreError):
        store.store("KEY", "x" * (8 * 1024 + 1))


def test_store_full_rejects_new(tmp_path, monkeypatch):
    monkeypatch.setattr("boxbot.secrets.store._MAX_SECRETS", 2)
    store = SecretStore(path=tmp_path / "secrets.json")
    store.store("KEY_A", "v")
    store.store("KEY_B", "v")
    with pytest.raises(SecretStoreError, match="full"):
        store.store("KEY_C", "v")
    # Replacing an existing entry stays allowed.
    store.store("KEY_A", "v2")


def test_delete_round_trip(tmp_path):
    store = SecretStore(path=tmp_path / "secrets.json")
    store.store("KEY", "v")
    assert store.delete("KEY")["status"] == "ok"
    assert not store.has("KEY")


def test_delete_missing_returns_missing(tmp_path):
    store = SecretStore(path=tmp_path / "secrets.json")
    assert store.delete("NEVER_STORED")["status"] == "missing"


def test_list_names_omits_values(tmp_path):
    store = SecretStore(path=tmp_path / "secrets.json")
    store.store("ALPHA", "av")
    store.store("BETA", "bv")
    listing = store.list_names()
    assert {row["name"] for row in listing} == {"ALPHA", "BETA"}
    assert all("value" not in row for row in listing)
    assert all(row["stored_at"] for row in listing)


def test_list_names_sorted(tmp_path):
    store = SecretStore(path=tmp_path / "secrets.json")
    store.store("ZULU", "v")
    store.store("ALPHA", "v")
    names = [row["name"] for row in store.list_names()]
    assert names == ["ALPHA", "ZULU"]


def test_load_returns_none_for_missing(tmp_path):
    store = SecretStore(path=tmp_path / "secrets.json")
    assert store.load("NEVER_STORED") is None


def test_load_returns_none_for_invalid_name(tmp_path):
    store = SecretStore(path=tmp_path / "secrets.json")
    # invalid name — load swallows the validation error and returns None
    # (so a hostile lookup can't probe whether validation differs from
    # absence).
    assert store.load("lowercase") is None


def test_atomic_write_no_partial_file(tmp_path, monkeypatch):
    """Forcing os.replace to fail should leave the original file intact."""
    store = SecretStore(path=tmp_path / "secrets.json")
    store.store("KEY", "original")

    real_replace = os.replace

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.store("KEY", "new")
    monkeypatch.setattr(os, "replace", real_replace)
    # Original survives.
    assert store.load("KEY") == "original"
    # No tempfile detritus left in the secrets dir.
    leftover = list((tmp_path).glob(".secrets-*.json.tmp"))
    assert leftover == []


# ---------------------------------------------------------------------------
# Sandbox action handler
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_secret_store(tmp_path, monkeypatch):
    """Point the module-level singleton at a tmp file."""
    from boxbot.secrets import store as store_mod

    fake = SecretStore(path=tmp_path / "secrets.json")
    monkeypatch.setattr(store_mod, "_INSTANCE", fake)
    return fake


@pytest.mark.asyncio
async def test_handler_store_persists(isolated_secret_store):
    from boxbot.tools._sandbox_actions import ActionContext, process_action

    ctx = ActionContext()
    response = await process_action(
        {"_sdk": "secrets.store", "name": "POLYGON_API_KEY", "value": "pk_v1"},
        ctx,
    )
    assert response["status"] == "ok"
    assert response["previous"] == "created"
    assert isolated_secret_store.load("POLYGON_API_KEY") == "pk_v1"


@pytest.mark.asyncio
async def test_handler_action_log_redacts_value(isolated_secret_store):
    from boxbot.tools._sandbox_actions import ActionContext, process_action

    ctx = ActionContext()
    await process_action(
        {"_sdk": "secrets.store", "name": "POLYGON_API_KEY", "value": "pk_v1"},
        ctx,
    )
    # The action log should never contain the value.
    serialized = json.dumps(ctx.action_log)
    assert "pk_v1" not in serialized
    assert "POLYGON_API_KEY" in serialized  # name is fine, value is not


@pytest.mark.asyncio
async def test_handler_use_returns_env_var_iff_present(isolated_secret_store):
    from boxbot.tools._sandbox_actions import ActionContext, process_action

    ctx = ActionContext()

    missing = await process_action(
        {"_sdk": "secrets.use", "name": "GHOST"}, ctx,
    )
    assert missing["status"] == "missing"

    isolated_secret_store.store("REAL", "v")
    found = await process_action(
        {"_sdk": "secrets.use", "name": "REAL"}, ctx,
    )
    assert found["status"] == "ok"
    assert found["env_var"] == "BOXBOT_SECRET_REAL"


@pytest.mark.asyncio
async def test_handler_list(isolated_secret_store):
    from boxbot.tools._sandbox_actions import ActionContext, process_action

    isolated_secret_store.store("ALPHA", "av")
    isolated_secret_store.store("BETA", "bv")

    ctx = ActionContext()
    response = await process_action(
        {"_sdk": "secrets.list"}, ctx,
    )
    assert response["status"] == "ok"
    names = {row["name"] for row in response["secrets"]}
    assert names == {"ALPHA", "BETA"}
    # No values surface from list either.
    serialized = json.dumps(response)
    assert "av" not in serialized and "bv" not in serialized


@pytest.mark.asyncio
async def test_handler_delete(isolated_secret_store):
    from boxbot.tools._sandbox_actions import ActionContext, process_action

    isolated_secret_store.store("KEY", "v")
    ctx = ActionContext()
    response = await process_action(
        {"_sdk": "secrets.delete", "name": "KEY"}, ctx,
    )
    assert response["status"] == "ok"
    assert not isolated_secret_store.has("KEY")


@pytest.mark.asyncio
async def test_handler_unknown_subaction(isolated_secret_store):
    from boxbot.tools._sandbox_actions import ActionContext, process_action

    ctx = ActionContext()
    response = await process_action(
        {"_sdk": "secrets.bogus", "name": "K"}, ctx,
    )
    assert response["status"] == "error"


# ---------------------------------------------------------------------------
# Runner injection
# ---------------------------------------------------------------------------


def _make_integration_dir(
    root: Path, name: str, *, secrets: list[str], script: str = ""
) -> None:
    d = root / name
    d.mkdir(parents=True)
    secret_lines = "\n".join(f"  - {n}" for n in secrets) if secrets else ""
    manifest_parts = [
        f"name: {name}",
        "description: test",
        "timeout: 5",
    ]
    if secrets:
        manifest_parts.append("secrets:\n" + secret_lines)
    (d / "manifest.yaml").write_text("\n".join(manifest_parts) + "\n")
    (d / "script.py").write_text(
        script
        or "from boxbot_sdk.integration import return_output\nreturn_output({'ok': True})\n"
    )


def test_runner_build_env_injects_declared_secrets(tmp_path, isolated_secret_store):
    from boxbot.integrations.loader import discover_integrations
    from boxbot.integrations.runner import _build_env

    _make_integration_dir(tmp_path, "echo", secrets=["FOO_KEY", "BAR_KEY"])
    isolated_secret_store.store("FOO_KEY", "foo-value")
    # BAR_KEY intentionally absent to test the missing-secret path.

    metas = discover_integrations(root=tmp_path)
    meta = next(m for m in metas if m.name == "echo")

    env, names = _build_env(
        inputs_path=tmp_path / "in.json",
        output_path=tmp_path / "out.json",
        meta=meta,
    )
    assert env.get("BOXBOT_SECRET_FOO_KEY") == "foo-value"
    assert "BOXBOT_SECRET_BAR_KEY" not in env
    assert names == ["BOXBOT_SECRET_FOO_KEY"]


def test_runner_build_command_extends_preserve_env(tmp_path):
    from boxbot.integrations.manifest import IntegrationMeta
    from boxbot.integrations.runner import _build_command

    meta = IntegrationMeta(
        name="echo",
        description="t",
        inputs={},
        outputs={},
        secrets=("FOO_KEY",),
        timeout=5,
        root_path=tmp_path,
        manifest_path=tmp_path / "manifest.yaml",
        script_path=tmp_path / "script.py",
    )
    cmd = _build_command(
        meta,
        venv_python=Path("/usr/bin/python3"),
        bootstrap_path=Path("/tmp/bootstrap.py"),
        sandbox_user="boxbot-sandbox",
        enforce_sandbox=True,
        secret_env_names=["BOXBOT_SECRET_FOO_KEY"],
    )
    preserve_arg = next(a for a in cmd if a.startswith("--preserve-env="))
    assert "BOXBOT_SECRET_FOO_KEY" in preserve_arg
    # Doesn't drop the default keys.
    assert "BOXBOT_SECCOMP_MODE" in preserve_arg


# ---------------------------------------------------------------------------
# execute_script secret resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_script_resolves_named_secrets(
    tmp_path, monkeypatch, isolated_secret_store,
):
    """The agent supplies ``secrets=[...]``; the tool resolves and injects.

    We don't actually spawn a subprocess — we patch the long-lived
    runner so the tool's pre-launch resolution path is the only thing
    exercised.
    """
    from boxbot.tools.builtins.execute_script import ExecuteScriptTool

    isolated_secret_store.store("POLYGON_API_KEY", "pk_v1")

    captured: dict[str, dict[str, str]] = {}

    class _FakeRunner:
        is_running = True

        async def run_script(self, code, *, env_vars=None):
            captured["env_vars"] = dict(env_vars or {})
            return ({"status": "success", "output": ""}, [])

    class _FakeConv:
        sandbox_runner = _FakeRunner()

    monkeypatch.setattr(
        "boxbot.tools.builtins.execute_script.get_current_conversation",
        lambda: _FakeConv(),
    )

    tool = ExecuteScriptTool()
    await tool.execute(
        script="pass",
        description="probe",
        secrets=["POLYGON_API_KEY", "GHOST"],  # GHOST is not stored
    )

    assert captured["env_vars"]["BOXBOT_SECRET_POLYGON_API_KEY"] == "pk_v1"
    assert "BOXBOT_SECRET_GHOST" not in captured["env_vars"]


# ---------------------------------------------------------------------------
# Status line includes secret count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_line_reports_secret_count(
    tmp_path, monkeypatch, isolated_secret_store,
):
    from boxbot.core import scheduler

    # Stub the DB-backed counts so the test doesn't need a real scheduler DB.
    class _FakeDB:
        async def execute(self, sql):
            class _Cursor:
                async def fetchone(self_inner):
                    return (0,)
            return _Cursor()

        async def close(self):
            pass

    async def fake_get_db():
        return _FakeDB()

    monkeypatch.setattr(scheduler, "_get_db", fake_get_db)

    isolated_secret_store.store("KEY_A", "v")
    isolated_secret_store.store("KEY_B", "v")

    line = await scheduler.get_status_line()
    assert "Secrets: 2 stored" in line
