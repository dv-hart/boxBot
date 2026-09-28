"""Tests for the shared sandbox privilege-drop helper.

Security-critical: both launch paths (persistent ``SandboxRunner`` and
the per-call ``execute_script`` fallback) build their subprocess argv
through :func:`build_sandbox_launch`, so this pins:

- ``privilege_drop`` resolution (auto/sudo/setuid/none, ENFORCE=0).
- The exact argv + popen_kwargs each mode produces.
- The Pi/sudo path argv is byte-for-byte what it was before the refactor
  (regression guard — this is the path in production today).
- The setuid ``preexec_fn`` calls ``setgid → setgroups → setuid`` in that
  order, with ``extra_groups`` folded into the supplementary set.

The module under test is stdlib-only, but ``boxbot.tools.__init__`` pulls
in the full tool registry (numpy, httpx, …). To keep these
security-critical tests runnable without the whole dependency tree, we
load ``_sandbox_launch.py`` directly from its file path rather than
importing it through the package.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_MODPATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "boxbot" / "tools" / "_sandbox_launch.py"
)
_spec = importlib.util.spec_from_file_location("_bb_sandbox_launch", _MODPATH)
sl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sl)

build_sandbox_launch = sl.build_sandbox_launch
resolve_privilege_drop = sl.resolve_privilege_drop


# ── privilege_drop resolution ───────────────────────────────────────


def test_resolve_auto_root_picks_setuid(monkeypatch):
    monkeypatch.delenv("BOXBOT_SANDBOX_ENFORCE", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    assert resolve_privilege_drop("auto", "boxbot-sandbox") == "setuid"


def test_resolve_auto_nonroot_picks_sudo(monkeypatch):
    monkeypatch.delenv("BOXBOT_SANDBOX_ENFORCE", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert resolve_privilege_drop("auto", "boxbot-sandbox") == "sudo"


def test_resolve_auto_no_user_is_none(monkeypatch):
    monkeypatch.delenv("BOXBOT_SANDBOX_ENFORCE", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    assert resolve_privilege_drop("auto", None) == "none"


def test_resolve_explicit_modes_honored(monkeypatch):
    monkeypatch.delenv("BOXBOT_SANDBOX_ENFORCE", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert resolve_privilege_drop("setuid", "u") == "setuid"
    assert resolve_privilege_drop("sudo", "u") == "sudo"
    assert resolve_privilege_drop("none", "u") == "none"


def test_resolve_enforce_env_forces_none(monkeypatch):
    monkeypatch.setenv("BOXBOT_SANDBOX_ENFORCE", "0")
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    # Even an explicit setuid request is overridden by the kill-switch.
    assert resolve_privilege_drop("setuid", "boxbot-sandbox") == "none"


def test_resolve_enforce_param_forces_none(monkeypatch):
    monkeypatch.delenv("BOXBOT_SANDBOX_ENFORCE", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert resolve_privilege_drop("sudo", "u", enforce=False) == "none"


def test_resolve_unknown_mode_fails_safe(monkeypatch):
    monkeypatch.delenv("BOXBOT_SANDBOX_ENFORCE", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert resolve_privilege_drop("bogus", "u") == "none"


# ── argv construction per mode ──────────────────────────────────────


def test_build_sudo_argv_is_byte_for_byte(monkeypatch):
    """The Pi/sudo argv must match the pre-refactor hardcoded form
    exactly — same flags, same --preserve-env spelling, same order."""
    monkeypatch.delenv("BOXBOT_SANDBOX_ENFORCE", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    argv = ["/venv/bin/python3", "/rt/sandbox_bootstrap.py", "/rt/scripts/x.py"]
    preserve = [
        "BOXBOT_SECCOMP_MODE",
        "BOXBOT_SECCOMP_DISABLE",
        "BOXBOT_SKILLS_ROOT",
        "BOXBOT_SECRETS_PATH",
    ]
    cmd, kwargs = build_sandbox_launch(
        argv,
        user="boxbot-sandbox",
        privilege_drop="auto",
        preserve_env_keys=preserve,
    )
    assert cmd == [
        "sudo", "-n",
        "--preserve-env=BOXBOT_SECCOMP_MODE,BOXBOT_SECCOMP_DISABLE,"
        "BOXBOT_SKILLS_ROOT,BOXBOT_SECRETS_PATH",
        "-u", "boxbot-sandbox",
        "--",
        "/venv/bin/python3", "/rt/sandbox_bootstrap.py", "/rt/scripts/x.py",
    ]
    assert kwargs == {}


def test_build_runner_sudo_argv_is_byte_for_byte(monkeypatch):
    """The persistent runner's sudo argv (python3 -c <server>) form."""
    monkeypatch.delenv("BOXBOT_SANDBOX_ENFORCE", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    argv = ["/venv/bin/python3", "-c", "SERVER_CODE"]
    cmd, kwargs = build_sandbox_launch(
        argv,
        user="boxbot-sandbox",
        privilege_drop="auto",
        preserve_env_keys=["BOXBOT_SECCOMP_MODE", "BOXBOT_SECCOMP_DISABLE"],
    )
    assert cmd == [
        "sudo", "-n",
        "--preserve-env=BOXBOT_SECCOMP_MODE,BOXBOT_SECCOMP_DISABLE",
        "-u", "boxbot-sandbox",
        "--",
        "/venv/bin/python3", "-c", "SERVER_CODE",
    ]
    assert kwargs == {}


def test_build_none_argv_unchanged_no_preexec(monkeypatch):
    monkeypatch.setenv("BOXBOT_SANDBOX_ENFORCE", "0")
    monkeypatch.setattr(os, "geteuid", lambda: 1000)  # non-root: no root guard
    argv = ["/venv/bin/python3", "boot.py", "x.py"]
    cmd, kwargs = build_sandbox_launch(
        argv, user="boxbot-sandbox", privilege_drop="auto",
    )
    assert cmd == argv
    assert cmd is not argv  # returns a copy, not the caller's list
    assert kwargs == {}


def _stub_user(monkeypatch, *, uid=4242, gid=4343, grouplist=(4343, 27)):
    """Make the sandbox user resolvable without a real system account."""
    import pwd

    monkeypatch.setattr(
        pwd, "getpwnam",
        lambda name: type("PW", (), {"pw_uid": uid, "pw_gid": gid})(),
    )
    monkeypatch.setattr(os, "getgrouplist", lambda name, g: list(grouplist))


def test_build_setuid_returns_preexec(monkeypatch):
    monkeypatch.delenv("BOXBOT_SANDBOX_ENFORCE", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    _stub_user(monkeypatch)

    argv = ["/venv/bin/python3", "-c", "SERVER"]
    cmd, kwargs = build_sandbox_launch(
        argv, user="boxbot-sandbox", privilege_drop="setuid",
        extra_groups=[3003],
    )
    assert cmd == argv
    assert "preexec_fn" in kwargs
    assert callable(kwargs["preexec_fn"])


def test_build_setuid_unknown_user_raises_at_build(monkeypatch):
    monkeypatch.delenv("BOXBOT_SANDBOX_ENFORCE", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    import pwd

    def _boom(name):
        raise KeyError(name)

    monkeypatch.setattr(pwd, "getpwnam", _boom)
    with pytest.raises(RuntimeError, match="not found"):
        build_sandbox_launch(
            ["python3"], user="ghost", privilege_drop="setuid",
        )


# ── setuid preexec ordering ─────────────────────────────────────────


def test_setuid_preexec_call_order_and_args(monkeypatch):
    """setgid → setgroups → setuid, in that order; the group set
    REPLICATES initgroups (full getgrouplist) UNIONed with extra_groups —
    it must NOT collapse to just [primary, extra] (that would drop the
    boxbot group and every group-gated read)."""
    # getgrouplist returns the user's real supplementary groups incl. a
    # 'boxbot'-like gid (99) that must survive into the child's set.
    _stub_user(monkeypatch, uid=4242, gid=4343, grouplist=(4343, 99))

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(os, "setgid", lambda g: calls.append(("setgid", g)))
    monkeypatch.setattr(os, "setgroups", lambda gs: calls.append(("setgroups", list(gs))))
    monkeypatch.setattr(os, "setuid", lambda u: calls.append(("setuid", u)))
    monkeypatch.setattr(os, "umask", lambda m: calls.append(("umask", m)))

    preexec = sl._make_setuid_preexec("boxbot-sandbox", [3003])
    preexec()

    names = [c[0] for c in calls]
    assert names.index("setgid") < names.index("setgroups") < names.index("setuid")
    assert ("setgid", 4343) in calls
    assert ("setuid", 4242) in calls
    assert ("umask", 0o077) in calls
    # The supplementary set carries the primary gid, the getgrouplist
    # member (99 ≈ boxbot), AND the configured extra_groups (3003).
    groups = next(g for n, g in calls if n == "setgroups")
    assert set(groups) == {4343, 99, 3003}


# ── root fail-closed guard (#4) ─────────────────────────────────────


def test_none_as_root_raises_without_override(monkeypatch):
    monkeypatch.setenv("BOXBOT_SANDBOX_ENFORCE", "0")  # → resolved none
    monkeypatch.delenv("BOXBOT_SANDBOX_ALLOW_ROOT", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    with pytest.raises(RuntimeError, match="root"):
        build_sandbox_launch(["python3"], user="boxbot-sandbox")


def test_none_as_root_allowed_with_override(monkeypatch):
    monkeypatch.setenv("BOXBOT_SANDBOX_ENFORCE", "0")
    monkeypatch.setenv("BOXBOT_SANDBOX_ALLOW_ROOT", "1")
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    cmd, kwargs = build_sandbox_launch(["python3"], user="boxbot-sandbox")
    assert cmd == ["python3"]
    assert kwargs == {}


def test_none_as_nonroot_unchanged(monkeypatch):
    monkeypatch.setenv("BOXBOT_SANDBOX_ENFORCE", "0")
    monkeypatch.delenv("BOXBOT_SANDBOX_ALLOW_ROOT", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    cmd, kwargs = build_sandbox_launch(["python3"], user="boxbot-sandbox")
    assert cmd == ["python3"]
    assert kwargs == {}
