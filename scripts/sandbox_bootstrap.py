#!/usr/bin/env python3
"""Sandbox bootstrap — applies a seccomp BPF filter, then runs a user script.

Invoked by ``boxbot.tools.builtins.execute_script`` as the entry point of
every sandboxed Python execution. The cmd line shape is:

    python3 sandbox_bootstrap.py <user_script.py>

This bootstrap:

1. Reads ``BOXBOT_SECCOMP_MODE`` from the environment. Three modes:
   - ``disabled`` (default): no filter applied. Equivalent to running the
     user script directly. Use only for debugging the bootstrap itself.
   - ``log``: filter is installed with ``SCMP_ACT_LOG`` for forbidden
     syscalls. The kernel logs each forbidden call to dmesg / audit.log
     but does NOT kill the process. Use this for a soak period to see
     which syscalls real workloads need before flipping to enforce.
   - ``enforce``: filter is installed with ``SCMP_ACT_KILL_PROCESS`` for
     forbidden syscalls. The first forbidden call kills the process with
     SIGSYS. This is the production-secure setting.

2. Applies the filter via ``python3-seccomp`` (the libseccomp Python
   bindings). If the binding isn't installed, the bootstrap warns and
   continues without a filter — graceful degradation. In ``enforce``
   mode this is treated as a fatal startup error; we'd rather refuse to
   run than run unguarded.

3. Hands off to the user script via ``runpy.run_path`` so the script
   sees ``__name__ == "__main__"`` and a clean ``sys.argv``.

This file is intentionally **self-contained** — no imports from
``boxbot``. The sandbox venv installs only ``boxbot_sdk`` (the
constrained agent surface), not the full main-process package.

The set of blocked syscalls and the per-syscall arg-matching rules
mirror the OCI-style profile that ``scripts/setup-sandbox.sh`` writes
to ``config/seccomp-sandbox.json``. The Python source here is the
source of truth for runtime; the JSON file is documentation.

Override at runtime:

    BOXBOT_SECCOMP_DISABLE=1   # bypass entirely, even if mode says enforce

Used as a kill-switch when the filter breaks something unexpected; the
operator can set this and restart the agent without redeploying code.
"""

from __future__ import annotations

import os
import runpy
import sys


# ---------------------------------------------------------------------------
# Filter definition
# ---------------------------------------------------------------------------

# Syscalls that should always fail. The most important are:
# - exec family: blocks spawning ANY new program from the sandbox
# - fork/vfork: blocks process duplication
# - ptrace: blocks debugger attach (would let a script read another
#   process's memory if it shared a UID)
# - kexec_*, init_module, delete_module, create_module: block kernel
#   manipulation
# - mount, umount2, pivot_root, chroot: block FS namespace games
# - reboot, swap*: block system-level controls
#
# ``clone`` is special — Python's threading.Thread uses ``clone`` with
# CLONE_THREAD set. We allow CLONE_THREAD-style clone and block other
# uses (full process clone) via argument matching below.
_BLOCKED_SYSCALLS: tuple[str, ...] = (
    "execve",
    "execveat",
    "fork",
    "vfork",
    "ptrace",
    "kexec_load",
    "kexec_file_load",
    "init_module",
    "finit_module",
    "delete_module",
    "create_module",
    "swapon",
    "swapoff",
    "mount",
    "umount2",
    "pivot_root",
    "chroot",
    "reboot",
    "perf_event_open",
    "bpf",
)


def _apply_filter(mode: str) -> None:
    """Install the seccomp BPF filter on the calling process.

    Raises on failure. The caller decides whether failure is fatal
    based on mode.
    """
    # Two compatible bindings exist:
    # - ``seccomp`` (Debian apt package ``python3-seccomp``, default
    #   on the Pi via setup-sandbox.sh)
    # - ``pyseccomp`` (PyPI, useful for dev machines without apt)
    # Both wrap libseccomp with the same API.
    try:
        import seccomp  # type: ignore[import-not-found]
    except ImportError:
        import pyseccomp as seccomp  # type: ignore[import-not-found]

    if mode == "log":
        forbidden_action = seccomp.LOG
    elif mode == "enforce":
        # KILL_PROCESS kills the whole process (vs KILL which only
        # kills the offending thread). Either is acceptable for our
        # use; KILL_PROCESS is more decisive.
        forbidden_action = (
            getattr(seccomp, "KILL_PROCESS", None) or seccomp.KILL
        )
    else:
        raise ValueError(f"unknown BOXBOT_SECCOMP_MODE: {mode!r}")

    f = seccomp.SyscallFilter(defaction=seccomp.ALLOW)

    skipped: list[tuple[str, str]] = []
    added = 0
    for name in _BLOCKED_SYSCALLS:
        try:
            f.add_rule(forbidden_action, name)
            added += 1
        except Exception as e:  # NOQA: BLE001  surface to log
            # libseccomp will refuse rules for syscalls unknown on the
            # current arch. We log and continue — partial coverage is
            # better than no filter.
            skipped.append((name, str(e)))

    # Non-thread clone — block clone() that's NOT setting CLONE_THREAD.
    # If we can't add the masked rule (older libseccomp on some
    # architectures), we still added kill/log for fork+vfork above so
    # most subprocess paths are still covered.
    try:
        clone_thread = 0x10000  # CLONE_THREAD
        f.add_rule(
            forbidden_action,
            "clone",
            seccomp.Arg(0, seccomp.MASKED_EQ, clone_thread, 0),
        )
        added += 1
    except Exception as e:  # NOQA: BLE001
        skipped.append(("clone (non-thread)", str(e)))

    f.load()

    sys.stderr.write(
        f"[sandbox-bootstrap] seccomp filter installed: "
        f"mode={mode}, added={added}, skipped={len(skipped)}\n"
    )
    if skipped:
        for name, err in skipped:
            sys.stderr.write(f"[sandbox-bootstrap]   skip {name}: {err}\n")


def _maybe_apply_filter() -> None:
    """Read env, apply filter if mode requests it, handle missing
    libseccomp gracefully."""
    if os.environ.get("BOXBOT_SECCOMP_DISABLE") == "1":
        sys.stderr.write(
            "[sandbox-bootstrap] BOXBOT_SECCOMP_DISABLE=1 — "
            "kill-switch set, no filter applied\n"
        )
        return

    mode = os.environ.get("BOXBOT_SECCOMP_MODE", "disabled").lower()
    if mode == "disabled":
        return
    if mode not in ("log", "enforce"):
        sys.stderr.write(
            f"[sandbox-bootstrap] unknown BOXBOT_SECCOMP_MODE={mode!r}, "
            "running without filter\n"
        )
        return

    try:
        _apply_filter(mode)
    except ImportError as e:
        # libseccomp Python binding not installed. Warn loudly. In log
        # mode keep going (operator wants to soak); in enforce mode
        # refuse to run unguarded.
        msg = (
            f"[sandbox-bootstrap] python3-seccomp not available ({e}). "
            "Install via 'apt install python3-seccomp' on Debian/Ubuntu, "
            "or set BOXBOT_SECCOMP_MODE=disabled to silence this warning.\n"
        )
        sys.stderr.write(msg)
        if mode == "enforce":
            sys.stderr.write(
                "[sandbox-bootstrap] mode=enforce but no filter could "
                "be applied — refusing to run user script.\n"
            )
            sys.exit(70)
    except Exception as e:  # NOQA: BLE001
        sys.stderr.write(
            f"[sandbox-bootstrap] seccomp apply failed: {e}\n"
        )
        if mode == "enforce":
            sys.stderr.write(
                "[sandbox-bootstrap] mode=enforce but filter setup "
                "errored — refusing to run user script.\n"
            )
            sys.exit(70)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        sys.stderr.write(
            "usage: sandbox_bootstrap.py <script.py> [script_args...]\n"
        )
        return 2

    _maybe_apply_filter()

    script_path = argv[1]
    # Hand off to the user script. runpy.run_path mimics ``python script.py``:
    # the module sees __name__ == '__main__' and sys.argv as if invoked
    # directly. Keeps backwards compat with anything written assuming
    # direct invocation.
    sys.argv = [script_path, *argv[2:]]
    runpy.run_path(script_path, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
