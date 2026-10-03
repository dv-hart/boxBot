"""Shared privilege-drop for the sandbox subprocess.

Both sandbox launch paths — the persistent ``SandboxRunner`` and the
per-call fallback in ``execute_script`` — go through
:func:`build_sandbox_launch` so they can never diverge on how the child
process sheds root. Two drop mechanisms, chosen by config:

- **sudo** (Pi): boxBot runs non-root; ``sudo -n -u <user>`` re-execs
  the child as the sandbox user. sudo strips the environment, so the
  caller names the vars it needs preserved.
- **setuid** (root hosts): boxBot already runs as root (e.g. a container)
  where there is no ``sudo`` and root bypasses the file-mode fences.
  The parent forks and, in the child (``preexec_fn``), drops directly:
  ``setgid → setgroups → setuid`` — supplementary groups and the real
  gid MUST be set while still root, before ``setuid`` drops the ability
  to do so. ``extra_groups`` are injected here, no ``/etc/group`` edit
  required.
- **none**: run as the current user. This is the ``BOXBOT_SANDBOX_ENFORCE=0``
  escape hatch and the no-``user``-configured fallback.

uid/gid are resolved at build time (in the parent) and closed over as
ints, so ``preexec_fn`` — which runs post-fork in a child that must
touch only async-signal-safe operations — never calls ``getpwnam``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

_warned_no_enforce = False


def resolve_privilege_drop(
    mode: str,
    user: str | None,
    *,
    enforce: bool = True,
) -> str:
    """Resolve a configured ``privilege_drop`` to a concrete mechanism.

    Returns one of ``sudo`` / ``setuid`` / ``none``.

    - ``BOXBOT_SANDBOX_ENFORCE=0`` (or ``enforce=False``) → ``none``,
      regardless of ``mode`` — the operator kill-switch wins.
    - No ``user`` configured → ``none`` (nothing to drop to).
    - ``auto`` → ``setuid`` when running as root (euid 0), else ``sudo``.
    - Explicit ``sudo`` / ``setuid`` / ``none`` are honoured as-is.
    """
    if not enforce or os.environ.get("BOXBOT_SANDBOX_ENFORCE", "1") == "0":
        return "none"
    if not user:
        return "none"
    if mode == "auto":
        return "setuid" if os.geteuid() == 0 else "sudo"
    if mode in ("sudo", "setuid", "none"):
        return mode
    # Unknown value slipped past config validation — fail safe.
    logger.warning("unknown privilege_drop mode %r; running as current user", mode)
    return "none"


def _make_setuid_preexec(
    user: str,
    extra_groups: Sequence[int],
) -> Callable[[], None]:
    """Build the post-fork ``preexec_fn`` that drops root to ``user``.

    uid/gid and the full group set are resolved here, in the parent, and
    closed over as ints so the child never calls into libc name services.
    Raises ``RuntimeError`` at build time (not silently in the child) if
    ``user`` does not exist.

    The group set REPLICATES ``initgroups`` (what ``sudo``/``runuser`` do):
    the user's *full* supplementary group list from the group database —
    critically ``boxbot``, which gates displays/, config/, data/, skills/,
    and the 0640 secrets file — UNIONed with the configured ``extra_groups``
    (e.g. a network-access gid). A naive ``[gid, *extra]`` would
    silently DROP ``boxbot`` and break every group-gated read.
    """
    import pwd

    try:
        pw = pwd.getpwnam(user)
    except KeyError as exc:
        raise RuntimeError(
            f"sandbox privilege_drop=setuid: user {user!r} not found "
            "(run scripts/setup-sandbox.sh)"
        ) from exc

    uid = pw.pw_uid
    gid = pw.pw_gid
    # os.getgrouplist includes the primary gid, so this matches the
    # coverage runuser/sudo give the child. Sorted+deduped for a stable set.
    groups = sorted(
        set(os.getgrouplist(user, gid)) | {int(g) for g in extra_groups}
    )

    def _preexec() -> None:
        # Runs post-fork in the child. MUST stay async-signal-safe: only
        # os.* syscalls here — NO logging, imports, or allocation — or we
        # reintroduce fork-deadlock risk. All name-service lookups already
        # happened in the parent above.
        #
        # Order is mandatory: set the gid and supplementary groups while
        # still root, THEN drop the uid. After setuid the process can no
        # longer change its groups.
        os.setgid(gid)
        os.setgroups(groups)
        os.setuid(uid)
        os.umask(0o077)

    return _preexec


def build_sandbox_launch(
    argv: list[str],
    *,
    user: str | None,
    privilege_drop: str = "auto",
    extra_groups: Sequence[int] = (),
    preserve_env_keys: Sequence[str] = (),
    enforce: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Return ``(final_argv, popen_kwargs)`` applying the privilege drop.

    ``popen_kwargs`` carries a ``preexec_fn`` for the setuid path and is
    empty otherwise. ``preserve_env_keys`` matters only for the sudo path
    (sudo strips the environment); the setuid/none paths rely on the
    caller's ``env=`` dict passing through untouched.
    """
    global _warned_no_enforce

    resolved = resolve_privilege_drop(privilege_drop, user, enforce=enforce)

    if resolved == "sudo":
        assert user is not None  # resolve() returns "none" when user is unset
        final_argv = [
            "sudo",
            "-n",
            "--preserve-env=" + ",".join(preserve_env_keys),
            "-u",
            user,
            "--",
            *argv,
        ]
        return final_argv, {}

    if resolved == "setuid":
        assert user is not None
        return list(argv), {"preexec_fn": _make_setuid_preexec(user, extra_groups)}

    # resolved == "none"
    if os.geteuid() == 0:
        # Fail-closed: "none" as root would run agent-authored code AS ROOT
        # (e.g. if the sandbox user/venv isn't set up). Refuse
        # unless the operator explicitly overrides. Non-root (Pi/dev) is
        # unaffected. Warn on EVERY launch here — this is a standing risk,
        # not a one-time dev convenience.
        if os.environ.get("BOXBOT_SANDBOX_ALLOW_ROOT") != "1":
            raise RuntimeError(
                "refusing to run the sandbox as root with no privilege drop. "
                "Run scripts/setup-sandbox.sh and set sandbox.privilege_drop="
                "setuid, or set BOXBOT_SANDBOX_ALLOW_ROOT=1 to override "
                "(NOT recommended — agent code would run as root)."
            )
        logger.warning(
            "Sandbox running as ROOT with no privilege drop "
            "(BOXBOT_SANDBOX_ALLOW_ROOT=1) — agent code runs unconfined."
        )
        return list(argv), {}
    if user and not _warned_no_enforce:
        logger.warning(
            "Sandbox privilege drop disabled — script runs as current user "
            "(BOXBOT_SANDBOX_ENFORCE=0 or privilege_drop=none)"
        )
        _warned_no_enforce = True
    return list(argv), {}
