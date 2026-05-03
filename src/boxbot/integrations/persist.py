"""Integration persistence — write/update/delete bundles on disk.

Receives validated payloads from the sandbox via ``integrations.create``
and ``integrations.update`` actions and renders them onto disk. Each
integration is a directory containing ``manifest.yaml`` and
``script.py``. The directory layout mirrors the loader's expectations
in :mod:`boxbot.integrations.loader`.

Validation runs again here on the trusted side — the SDK validators
are convenience, not security.

Conflict policy:
- ``create`` returns ``status: "exists"`` and writes nothing if the
  target dir is already there.
- ``update`` returns ``status: "missing"`` if the target doesn't exist
  yet — it never auto-creates.
- ``delete`` returns ``status: "missing"`` if the target doesn't exist.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from boxbot.integrations.loader import _DEFAULT_INTEGRATIONS_ROOT
from boxbot.integrations.manifest import (
    ManifestError,
    render_manifest_yaml,
    validate_manifest,
)

logger = logging.getLogger(__name__)

_MANIFEST_FILE = "manifest.yaml"
_SCRIPT_FILE = "script.py"


def _resolve_target(name: str, root: Path) -> Path:
    """Resolve ``integrations/<name>`` and refuse paths that escape root."""
    target = (root / name).resolve()
    root_resolved = root.resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"resolved path escapes integrations root: {target}"
        ) from exc
    return target


def _validate_script(script: Any) -> str:
    if not isinstance(script, str) or not script.strip():
        raise ManifestError("script must be a non-empty string")
    return script


def create_integration(
    payload: dict[str, Any],
    *,
    integrations_root: Path | None = None,
) -> dict[str, Any]:
    """Write a new integration. Refuses to overwrite an existing one.

    Returns:
        ``{"status": "ok", "name", "path", "files": [...]}`` on success
        ``{"status": "exists", "name", "path", "message"}`` if the dir is taken
        ``{"status": "error", "message"}`` on validation failure
    """
    try:
        manifest_payload = dict(payload)
        script = _validate_script(manifest_payload.pop("script", None))
        manifest = validate_manifest(manifest_payload)
    except (ManifestError, ValueError) as exc:
        return {"status": "error", "message": str(exc)}

    root = integrations_root or _DEFAULT_INTEGRATIONS_ROOT
    try:
        target = _resolve_target(manifest["name"], root)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    if target.exists():
        return {
            "status": "exists",
            "name": manifest["name"],
            "path": str(target),
            "message": (
                f"integration '{manifest['name']}' already exists at {target}; "
                "delete it or call update() instead"
            ),
        }

    target.mkdir(parents=True, exist_ok=False)
    (target / _MANIFEST_FILE).write_text(
        render_manifest_yaml(manifest), encoding="utf-8"
    )
    (target / _SCRIPT_FILE).write_text(script, encoding="utf-8")

    logger.info("Created integration '%s' at %s", manifest["name"], target)
    return {
        "status": "ok",
        "name": manifest["name"],
        "path": str(target),
        "files": [_MANIFEST_FILE, _SCRIPT_FILE],
    }


def update_integration(
    payload: dict[str, Any],
    *,
    integrations_root: Path | None = None,
) -> dict[str, Any]:
    """Update an existing integration's manifest and/or script.

    The payload's ``name`` identifies the integration. Provide either
    ``manifest`` (a dict matching the manifest schema) or ``script``
    (a str) or both. Errors if the integration doesn't exist — never
    auto-promotes.
    """
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        return {"status": "error", "message": "update requires 'name'"}

    root = integrations_root or _DEFAULT_INTEGRATIONS_ROOT
    try:
        target = _resolve_target(name, root)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    if not target.is_dir():
        return {
            "status": "missing",
            "name": name,
            "message": f"integration '{name}' does not exist; create it first",
        }

    manifest_payload = payload.get("manifest")
    script = payload.get("script")
    if manifest_payload is None and script is None:
        return {
            "status": "error",
            "message": "update requires 'manifest' and/or 'script'",
        }

    written: list[str] = []
    if manifest_payload is not None:
        try:
            manifest = validate_manifest({**manifest_payload, "name": name})
        except (ManifestError, ValueError) as exc:
            return {"status": "error", "message": str(exc)}
        (target / _MANIFEST_FILE).write_text(
            render_manifest_yaml(manifest), encoding="utf-8"
        )
        written.append(_MANIFEST_FILE)
    if script is not None:
        try:
            text = _validate_script(script)
        except ManifestError as exc:
            return {"status": "error", "message": str(exc)}
        (target / _SCRIPT_FILE).write_text(text, encoding="utf-8")
        written.append(_SCRIPT_FILE)

    logger.info("Updated integration '%s' (%s)", name, ", ".join(written))
    return {
        "status": "ok",
        "name": name,
        "path": str(target),
        "files": written,
    }


def delete_integration(
    name: str,
    *,
    integrations_root: Path | None = None,
) -> dict[str, Any]:
    """Remove an integration directory. Errors if it doesn't exist."""
    if not isinstance(name, str) or not name:
        return {"status": "error", "message": "delete requires 'name'"}

    root = integrations_root or _DEFAULT_INTEGRATIONS_ROOT
    try:
        target = _resolve_target(name, root)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    if not target.is_dir():
        return {
            "status": "missing",
            "name": name,
            "message": f"integration '{name}' does not exist",
        }

    shutil.rmtree(target)
    logger.info("Deleted integration '%s' at %s", name, target)
    return {"status": "ok", "name": name, "path": str(target)}
