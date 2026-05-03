"""Integration discovery — scan ``src/boxbot/integrations/`` for manifests.

Each subdirectory containing both ``manifest.yaml`` and ``script.py``
is one integration. Malformed entries are logged and skipped; the
loader never raises.

Loaders are read-only. Writes (create / update / delete) live in
:mod:`boxbot.integrations.persist` (Phase 3).
"""

from __future__ import annotations

import logging
from pathlib import Path

from boxbot.integrations.manifest import IntegrationMeta, load_manifest_file

logger = logging.getLogger(__name__)

# Repo-root ``integrations/`` directory, sibling to ``skills/``. Both
# built-in and agent-authored integrations live here — built-ins ship
# in git, agent-authored land at runtime via ``bb.integrations.create``
# and are gitignored unless explicitly allowlisted in ``.gitignore``.
# Lives at the repo root (not under ``src/boxbot/integrations/``) so
# the main process — running as ``boxbot`` — can write to it without
# needing privileged access to the source tree.
#
# loader.py is at ``<repo>/src/boxbot/integrations/loader.py``;
# four parents up is the repo root.
_DEFAULT_INTEGRATIONS_ROOT: Path = (
    Path(__file__).resolve().parent.parent.parent.parent / "integrations"
)

_MANIFEST_FILE = "manifest.yaml"
_SCRIPT_FILE = "script.py"


def _resolve_root(root: Path | None) -> Path:
    return (root if root is not None else _DEFAULT_INTEGRATIONS_ROOT).resolve()


def discover_integrations(root: Path | None = None) -> list[IntegrationMeta]:
    """Scan an integrations root and return :class:`IntegrationMeta` records.

    - Returns an empty list if ``root`` does not exist.
    - Skips symlinks (security: integrations are local-only).
    - Skips directories without both ``manifest.yaml`` and ``script.py``.
    - Skips manifests that fail validation (logged at WARNING).
    - Stable order: alphabetical by name.
    """
    resolved = _resolve_root(root)
    if not resolved.exists() or not resolved.is_dir():
        return []

    out: list[IntegrationMeta] = []
    try:
        entries = sorted(resolved.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        logger.warning("Cannot list integrations root %s: %s", resolved, exc)
        return []

    for entry in entries:
        if entry.is_symlink():
            logger.warning("Skipping symlinked integrations entry: %s", entry.name)
            continue
        if not entry.is_dir():
            continue
        manifest_path = entry / _MANIFEST_FILE
        script_path = entry / _SCRIPT_FILE
        if not manifest_path.is_file() or not script_path.is_file():
            continue
        manifest = load_manifest_file(manifest_path)
        if manifest is None:
            continue
        # Cross-check: directory name must match manifest name. Renaming
        # the directory after the fact is a footgun; refuse rather than
        # paper over it.
        if manifest["name"] != entry.name:
            logger.warning(
                "Integration directory '%s' has manifest name '%s'; "
                "rename one or the other to match",
                entry.name,
                manifest["name"],
            )
            continue
        out.append(
            IntegrationMeta(
                name=manifest["name"],
                description=manifest["description"],
                inputs=manifest["inputs"],
                outputs=manifest["outputs"],
                secrets=tuple(manifest["secrets"]),
                timeout=manifest["timeout"],
                root_path=entry,
                manifest_path=manifest_path,
                script_path=script_path,
            )
        )

    return out


def get_integration(name: str, *, root: Path | None = None) -> IntegrationMeta | None:
    """Return the :class:`IntegrationMeta` for ``name`` if registered.

    Returns ``None`` if the integration doesn't exist or is malformed.
    """
    for meta in discover_integrations(root=root):
        if meta.name == name:
            return meta
    return None
