"""Tests for the sandbox bootstrap + seccomp filter.

Two layers:

1. **Bootstrap mechanics** (no libseccomp required): the bootstrap
   script reads BOXBOT_SECCOMP_MODE, applies the requested mode (or
   skips when disabled), then hands off to the user script with
   sys.argv set up for direct invocation.

2. **Should-fail enforcement** (libseccomp required, skipped otherwise):
   when mode=enforce is on and a sandboxed script tries to spawn a
   subprocess, the kernel kills the process with SIGSYS. This is the
   ground truth that the filter is actually doing what we think.

Tests run the real bootstrap as a subprocess so we exercise the same
code path the sandbox runner does in production.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = PROJECT_ROOT / "scripts" / "sandbox_bootstrap.py"


def _run_bootstrap(
    user_script: str,
    *,
    mode: str | None = None,
    disable: bool = False,
    extra_args: list[str] | None = None,
    timeout: float = 10.0,
) -> subprocess.CompletedProcess:
    """Invoke ``sandbox_bootstrap.py`` with a temp user script."""
    env = os.environ.copy()
    if mode is not None:
        env["BOXBOT_SECCOMP_MODE"] = mode
    else:
        env.pop("BOXBOT_SECCOMP_MODE", None)
    if disable:
        env["BOXBOT_SECCOMP_DISABLE"] = "1"
    else:
        env.pop("BOXBOT_SECCOMP_DISABLE", None)

    script_dir = PROJECT_ROOT / ".pytest-tmp"
    script_dir.mkdir(exist_ok=True)
    script_path = script_dir / "user_script.py"
    script_path.write_text(textwrap.dedent(user_script), encoding="utf-8")

    cmd = [sys.executable, str(BOOTSTRAP), str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env,
    )


# ---------------------------------------------------------------------------
# Bootstrap mechanics — runnable without libseccomp
# ---------------------------------------------------------------------------


class TestBootstrapMechanics:
    def test_disabled_mode_runs_user_script(self):
        """No filter, script executes normally, exits 0."""
        result = _run_bootstrap(
            "print('hello'); print('world')",
            mode="disabled",
        )
        assert result.returncode == 0, result.stderr
        assert "hello" in result.stdout
        assert "world" in result.stdout

    def test_default_is_disabled(self):
        """Unset BOXBOT_SECCOMP_MODE → bootstrap runs script without filter."""
        result = _run_bootstrap("print('ok')", mode=None)
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_kill_switch_bypasses_filter(self):
        """BOXBOT_SECCOMP_DISABLE=1 wins over mode=enforce — script runs."""
        result = _run_bootstrap(
            "print('bypassed')",
            mode="enforce",
            disable=True,
        )
        assert result.returncode == 0, result.stderr
        assert "bypassed" in result.stdout
        assert "BOXBOT_SECCOMP_DISABLE=1" in result.stderr

    def test_unknown_mode_warns_runs_unfiltered(self):
        """Mode like 'verbose' isn't valid — bootstrap warns + runs."""
        result = _run_bootstrap(
            "print('ran')",
            mode="verbose",
        )
        assert result.returncode == 0
        assert "ran" in result.stdout
        assert "unknown BOXBOT_SECCOMP_MODE" in result.stderr

    def test_argv_passed_through(self):
        """The user script sees its own path as argv[0] + extra args."""
        result = _run_bootstrap(
            "import sys; print(sys.argv)",
            mode="disabled",
            extra_args=["foo", "bar"],
        )
        assert result.returncode == 0, result.stderr
        # argv[0] is the user script path, then our extras
        assert "foo" in result.stdout
        assert "bar" in result.stdout
        assert "user_script.py" in result.stdout

    def test_user_script_sees_main_name(self):
        """runpy with run_name='__main__' so the script's
        ``if __name__ == '__main__':`` block fires."""
        result = _run_bootstrap(
            "if __name__ == '__main__': print('is main')\n"
            "else: print('NOT main')\n",
            mode="disabled",
        )
        assert result.returncode == 0, result.stderr
        assert "is main" in result.stdout
        assert "NOT main" not in result.stdout

    def test_missing_script_path_returns_2(self):
        """No argv → usage error, exit 2."""
        env = os.environ.copy()
        env.pop("BOXBOT_SECCOMP_MODE", None)
        result = subprocess.run(
            [sys.executable, str(BOOTSTRAP)],
            capture_output=True, text=True, timeout=5, env=env,
        )
        assert result.returncode == 2
        assert "usage" in result.stderr.lower()

    def test_enforce_without_libseccomp_refuses(self):
        """If pyseccomp isn't available and mode=enforce, refuse to run."""
        # We can't easily fake "no libseccomp" when it IS installed, so
        # this only checks the failure path when it's actually missing.
        # On systems where libseccomp is present, this test is meaningful
        # only as documentation of intent — the assertion below has to
        # accept BOTH possible outcomes.
        result = _run_bootstrap(
            "print('should not appear in enforce-without-lib')",
            mode="enforce",
        )

        if _has_libseccomp():
            # With the lib present, enforce mode succeeds and the user
            # script runs to completion (it doesn't trigger any blocked
            # syscall).
            assert result.returncode == 0, result.stderr
        else:
            # No libseccomp + enforce mode → bootstrap exits 70 BEFORE
            # running user script.
            assert result.returncode == 70
            assert "refusing to run" in result.stderr
            assert "should not appear" not in result.stdout


# ---------------------------------------------------------------------------
# Should-fail enforcement — requires libseccomp
# ---------------------------------------------------------------------------


# Linux SIGSYS is signal 31. The kernel kills with this signal when a
# seccomp KILL_PROCESS rule fires. subprocess returns -signum on signal
# death.
SIGSYS_EXIT = -31


def _has_libseccomp() -> bool:
    """Either the apt 'seccomp' module or the PyPI 'pyseccomp' is fine —
    bootstrap.py tries both."""
    try:
        import seccomp  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import pyseccomp  # noqa: F401
        return True
    except ImportError:
        return False


# The enforcement tests verify that the kernel actually kills the
# process when a blocked syscall fires. Without the libseccomp Python
# binding we can't install the filter and the assertions are
# meaningless, so we skip the class.
@pytest.mark.skipif(
    not _has_libseccomp(),
    reason="python3-seccomp not installed",
)
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="seccomp is Linux-only",
)
class TestEnforcement:
    def test_enforce_blocks_subprocess_run(self):
        """The flagship case: subprocess.run goes through execve →
        kernel kills the bootstrap with SIGSYS."""
        result = _run_bootstrap(
            """
            import subprocess
            subprocess.run(["true"])
            print("if you see this, the filter is OFF")
            """,
            mode="enforce",
        )
        assert result.returncode != 0
        # Either SIGSYS death or non-zero exit. Different kernels signal
        # this slightly differently but the print should NEVER appear.
        assert "if you see this, the filter is OFF" not in result.stdout

    def test_enforce_blocks_os_fork(self):
        """os.fork() goes through clone (or fork on older arches)."""
        result = _run_bootstrap(
            """
            import os
            try:
                pid = os.fork()
            except OSError as e:
                print(f"errno: {e.errno}")
                raise SystemExit(0)
            if pid == 0:
                # child
                os._exit(0)
            print("parent continued — filter is OFF")
            """,
            mode="enforce",
        )
        assert result.returncode != 0
        assert "parent continued" not in result.stdout

    def test_enforce_blocks_os_system(self):
        """os.system shells out via execve."""
        result = _run_bootstrap(
            """
            import os
            os.system("echo escaped")
            print("os.system returned — filter is OFF")
            """,
            mode="enforce",
        )
        assert result.returncode != 0
        assert "os.system returned" not in result.stdout

    def test_log_mode_does_not_kill(self):
        """Same offending call in log mode → script still completes."""
        result = _run_bootstrap(
            """
            import os
            try:
                os.system("true")
            except Exception as e:
                print("exception:", e)
            print("survived")
            """,
            mode="log",
            timeout=15.0,
        )
        # In log mode the kernel logs but does not kill. The script
        # should reach the final print. ``os.system`` may itself return
        # an error code (since the spawned shell can't actually exec),
        # but the parent process keeps running.
        assert result.returncode == 0, result.stderr
        assert "survived" in result.stdout

    def test_enforce_allows_normal_python_work(self):
        """Real workloads (file IO, math, threading) should NOT trigger
        the filter — the rule set is about exec/fork/etc., not normal
        work."""
        result = _run_bootstrap(
            """
            import threading
            import json

            data = {"a": 1, "b": [2, 3]}
            blob = json.dumps(data)
            assert json.loads(blob) == data

            results = []
            def worker():
                results.append("ok")
            t = threading.Thread(target=worker)
            t.start()
            t.join()
            assert results == ["ok"]

            with open("/tmp/seccomp_test_marker", "w") as f:
                f.write("hi")
            with open("/tmp/seccomp_test_marker") as f:
                assert f.read() == "hi"

            print("normal work succeeded")
            """,
            mode="enforce",
        )
        assert result.returncode == 0, (
            f"normal Python work should not trigger filter; "
            f"stderr={result.stderr}"
        )
        assert "normal work succeeded" in result.stdout
