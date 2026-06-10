"""Install approved packages into the sandbox venv.

Runs as the **main process user** — who owns the sandbox venv (see
``scripts/setup-sandbox.sh``: the venv is ``chown $REAL_USER:boxbot``
with pip binaries mode 700, owner-only). The sandbox user can never do
this; the main user always can, without sudo.

After a successful install the venv's permission policy is re-applied
to ``lib/`` and ``bin/`` (dirs 750, files 640, ``python3*`` 750,
``pip*`` 700, group ``boxbot``) because pip creates files with the
default umask and the invoking user's primary group — without the
fixup the freshly installed package may be unreadable (or too readable)
for the ``boxbot-sandbox`` user. Bytecode is then pre-compiled so the
sandbox user (who cannot write ``__pycache__``) doesn't pay the
compile-per-import cost.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Generous ceiling — wheels for big packages on a Pi over slow links.
INSTALL_TIMEOUT_SECONDS = 600

SANDBOX_GROUP = "boxbot"


def _sandbox_venv() -> Path:
    from boxbot.core.config import get_config

    return Path(get_config().sandbox.venv_path)


async def install_package(
    spec: str,
    *,
    venv: Path | None = None,
    timeout: int = INSTALL_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """Install ``spec`` (a validated name or ``name==version``) via pip.

    Returns ``(ok, output)`` where ``output`` is the combined
    stdout+stderr of pip (or an explanatory message when pip never ran).
    Never raises — callers record the outcome on the request row.
    """
    venv = venv or _sandbox_venv()
    pip = venv / "bin" / "pip"
    if not pip.exists():
        return False, f"sandbox venv pip not found at {pip}"

    try:
        proc = await asyncio.create_subprocess_exec(
            str(pip),
            "install",
            "--no-input",
            "--disable-pip-version-check",
            spec,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as e:
        return False, f"could not launch pip: {e}"

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return False, f"pip install timed out after {timeout}s"

    output = (stdout or b"").decode("utf-8", errors="replace")
    if proc.returncode != 0:
        logger.warning("pip install %s failed (rc=%s)", spec, proc.returncode)
        return False, output

    # Permission fixup + bytecode precompile are best-effort: the
    # install itself succeeded, and on dev machines (no boxbot group)
    # parts of the fixup are expected to no-op.
    try:
        await asyncio.to_thread(fix_sandbox_permissions, venv)
    except Exception:  # noqa: BLE001
        logger.exception("sandbox permission fixup failed after install")
    await _precompile_bytecode(venv)

    logger.info("Installed %s into sandbox venv %s", spec, venv)
    return True, output


def fix_sandbox_permissions(venv: Path, *, group: str = SANDBOX_GROUP) -> None:
    """Re-apply the setup-sandbox.sh permission policy to the venv.

    Mirrors the setup script: group ``boxbot``; dirs 750; files 640;
    ``bin/python3*`` real files 750; ``bin/pip*`` 700; symlinks left
    alone. Per-entry errors are logged and skipped — a partially fixed
    venv is still strictly better than an unfixed one.
    """
    gid: int | None = None
    try:
        import grp

        gid = grp.getgrnam(group).gr_gid
    except (KeyError, ImportError):
        logger.warning(
            "group %r not found — skipping group ownership fixup", group
        )

    def _apply(path: Path, mode: int) -> None:
        try:
            if path.is_symlink():
                return
            if gid is not None:
                os.chown(path, -1, gid)
            os.chmod(path, mode)
        except OSError as e:
            logger.debug("perm fixup skipped %s: %s", path, e)

    lib = venv / "lib"
    if lib.exists():
        for root, dirs, files in os.walk(lib):
            root_path = Path(root)
            _apply(root_path, 0o750)
            for name in files:
                _apply(root_path / name, 0o640)
        # os.walk visits the top dir via root, but make it explicit:
        _apply(lib, 0o750)

    bin_dir = venv / "bin"
    if bin_dir.exists():
        _apply(bin_dir, 0o750)
        for entry in bin_dir.iterdir():
            if entry.is_symlink():
                continue
            name = entry.name
            if name.startswith("pip"):
                _apply(entry, 0o700)
            elif name.startswith("python3"):
                _apply(entry, 0o750)
            else:
                # Console scripts etc. The sandbox can't exec anything
                # (seccomp), so readable-not-executable is correct.
                _apply(entry, 0o640)


async def _precompile_bytecode(venv: Path) -> None:
    """Best-effort ``compileall`` so the sandbox doesn't need __pycache__ writes."""
    python = venv / "bin" / "python3"
    lib = venv / "lib"
    if not python.exists() or not lib.exists():
        return
    try:
        proc = await asyncio.create_subprocess_exec(
            str(python),
            "-m",
            "compileall",
            "-q",
            str(lib),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), 300)
    except (OSError, asyncio.TimeoutError):
        logger.warning("bytecode precompile after install did not finish")
    else:
        # compileall writes __pycache__ dirs owned by us with umask
        # perms — bring them in line so the sandbox user can read them.
        try:
            await asyncio.to_thread(fix_sandbox_permissions, venv)
        except Exception:  # noqa: BLE001
            logger.exception("perm fixup after precompile failed")
