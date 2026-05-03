"""Integration manifest schema + parser.

An integration manifest declares the contract a sandbox-runnable
script presents to the rest of the system: what inputs the script
expects, what shape it returns, what secrets it needs, and how long
it's allowed to take. The manifest is **informational** in v1 —
inputs and outputs are documented, not strict-typed at runtime. The
runner uses ``timeout`` and ``secrets``; everything else is for
agent and human readers.

There is **no schedule** in the manifest. Integrations are pipes:
consumers (display data sources, the agent, the scheduler) decide
when to call them.

On-disk layout::

    src/boxbot/integrations/<name>/
      manifest.yaml
      script.py

Example manifest::

    name: weather
    description: Get NOAA weather forecasts. Returns today + N-day outlook.
    inputs:
      lat: {type: float, required: true}
      lon: {type: float, required: true}
      days: {type: int, default: 5}
    outputs:
      today: {high: int, low: int, condition: str}
      forecast: [{day: str, high: int, low: int}]
    secrets:
      - NWS_USER_AGENT
    timeout: 30

This module never raises on a malformed manifest from disk: bad
manifests log a warning and the loader skips them. Manifests built
by the SDK builder (Phase 3) raise on construction so the agent
sees the error at the source.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Anthropic Agent Skills naming rules apply to integrations too.
NAME_MAX_LEN = 64
DESCRIPTION_MAX_LEN = 1024
DEFAULT_TIMEOUT = 30
TIMEOUT_MAX = 300  # 5 min hard ceiling — anything longer should be background work
RESERVED_NAMES = {"anthropic", "claude"}
NAME_PATTERN = re.compile(r"^[a-z0-9_-]+$")

VALID_TYPE_NAMES = {"string", "str", "int", "integer", "float", "number", "bool", "boolean", "list", "array", "dict", "object"}


class ManifestError(ValueError):
    """Raised when a manifest payload is invalid (SDK-side construction)."""


@dataclass(frozen=True)
class IntegrationMeta:
    """Loader's view of one integration on disk."""

    name: str
    description: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    secrets: tuple[str, ...]
    timeout: int
    root_path: Path
    manifest_path: Path
    script_path: Path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError("integration name must be a non-empty string")
    if len(value) > NAME_MAX_LEN:
        raise ManifestError(f"integration name must be ≤{NAME_MAX_LEN} chars, got {len(value)}")
    if value != value.lower():
        raise ManifestError(f"integration name must be lowercase, got '{value}'")
    if value in RESERVED_NAMES:
        raise ManifestError(f"'{value}' is a reserved integration name")
    if not NAME_PATTERN.match(value):
        raise ManifestError(
            f"integration name must match [a-z0-9_-]+, got '{value}'"
        )
    return value


def validate_description(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("description must be a non-empty string")
    if len(value) > DESCRIPTION_MAX_LEN:
        raise ManifestError(
            f"description must be ≤{DESCRIPTION_MAX_LEN} chars, got {len(value)}"
        )
    if "<" in value or ">" in value:
        raise ManifestError("description must not contain XML brackets")
    return value


def validate_timeout(value: Any) -> int:
    if value is None:
        return DEFAULT_TIMEOUT
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"timeout must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ManifestError(f"timeout must be ≥1 second, got {value}")
    if value > TIMEOUT_MAX:
        raise ManifestError(
            f"timeout must be ≤{TIMEOUT_MAX} seconds, got {value} "
            "(longer-running work belongs in a scheduled trigger, not an integration call)"
        )
    return value


def validate_secrets(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ManifestError("secrets must be a list of strings")
    out: list[str] = []
    for i, item in enumerate(value):
        # Accept both "FOO" and {"name": "FOO"} for friendliness.
        if isinstance(item, dict):
            name = item.get("name")
        else:
            name = item
        if not isinstance(name, str) or not name:
            raise ManifestError(f"secrets[{i}] must be a non-empty string")
        if not re.match(r"^[A-Z][A-Z0-9_]*$", name):
            raise ManifestError(
                f"secrets[{i}] '{name}' must be SCREAMING_SNAKE_CASE"
            )
        out.append(name)
    # Preserve order, dedupe.
    seen: set[str] = set()
    deduped = [s for s in out if not (s in seen or seen.add(s))]
    return tuple(deduped)


def _validate_input_spec(name: str, spec: Any) -> dict[str, Any]:
    """One input declaration — informational shape, lightly validated."""
    if not isinstance(spec, dict):
        raise ManifestError(f"input '{name}' must be a dict")
    out = dict(spec)
    if "type" in out:
        if not isinstance(out["type"], str):
            raise ManifestError(f"input '{name}' type must be a string")
        if out["type"] not in VALID_TYPE_NAMES:
            raise ManifestError(
                f"input '{name}' type must be one of {sorted(VALID_TYPE_NAMES)}, "
                f"got '{out['type']}'"
            )
    if "required" in out and not isinstance(out["required"], bool):
        raise ManifestError(f"input '{name}' required must be a bool")
    return out


def validate_inputs(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ManifestError("inputs must be a dict")
    return {str(k): _validate_input_spec(str(k), v) for k, v in value.items()}


def validate_outputs(value: Any) -> dict[str, Any]:
    """Outputs are descriptive only in v1 — accept any dict shape."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ManifestError("outputs must be a dict")
    return dict(value)


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a manifest payload (e.g. from the SDK builder).

    Returns the normalized manifest dict. Raises :class:`ManifestError`
    on the first failure. Use :func:`load_manifest_file` for forgiving
    disk reads (it warns and returns ``None`` on failure).
    """
    if not isinstance(payload, dict):
        raise ManifestError("manifest must be a dict")
    name = validate_name(payload.get("name"))
    description = validate_description(payload.get("description"))
    inputs = validate_inputs(payload.get("inputs"))
    outputs = validate_outputs(payload.get("outputs"))
    secrets = validate_secrets(payload.get("secrets"))
    timeout = validate_timeout(payload.get("timeout"))
    return {
        "name": name,
        "description": description,
        "inputs": inputs,
        "outputs": outputs,
        "secrets": list(secrets),
        "timeout": timeout,
    }


# ---------------------------------------------------------------------------
# Disk read (forgiving)
# ---------------------------------------------------------------------------


def load_manifest_file(path: Path) -> dict[str, Any] | None:
    """Read and validate a manifest from disk. Returns None on failure.

    Logs a warning when the file is missing, malformed YAML, or fails
    validation — the loader uses this to skip broken integrations
    without taking the whole registry down.
    """
    try:
        import yaml
    except ImportError:  # pragma: no cover — yaml is a hard dep, but be defensive
        logger.error("PyYAML missing; cannot load integration manifests")
        return None

    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot read integration manifest %s: %s", path, exc)
        return None
    try:
        parsed = yaml.safe_load(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Malformed YAML in integration manifest %s: %s", path, exc)
        return None
    if not isinstance(parsed, dict):
        logger.warning("Integration manifest %s did not parse to a dict", path)
        return None
    try:
        return validate_manifest(parsed)
    except ManifestError as exc:
        logger.warning("Invalid integration manifest %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Render (SDK → on-disk YAML)
# ---------------------------------------------------------------------------


def render_manifest_yaml(manifest: dict[str, Any]) -> str:
    """Serialize a validated manifest dict to YAML.

    Used by the persist writer (Phase 3). Produces stable ordering so
    diffs are predictable and reviewable.
    """
    try:
        import yaml
    except ImportError:  # pragma: no cover
        raise RuntimeError("PyYAML is required to render integration manifests")

    ordered_keys = ("name", "description", "inputs", "outputs", "secrets", "timeout")
    ordered = {k: manifest[k] for k in ordered_keys if k in manifest}
    return yaml.safe_dump(ordered, sort_keys=False, default_flow_style=False, allow_unicode=True)
