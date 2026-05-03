"""Secret store — file-backed credential vault.

Values live in a JSON file at ``data/credentials/secrets.json``. The
file is written atomically (tempfile + rename) at mode ``0600`` owned by
the main-process user (``boxbot``) — the ``boxbot-sandbox`` user has no
read permission, same protection class as ``.env``.

This module is the main-process backend. Sandbox scripts reach it
through the ``secrets.*`` action handler in
``boxbot.tools._sandbox_actions``; integration scripts reach values
through ``BOXBOT_SECRET_<NAME>`` env vars set by the runner.

Constraints (chosen to keep the file small and the surface boring):

- Names must match ``^[A-Z][A-Z0-9_]*$`` — same shape integrations
  declare in their manifests.
- Up to 64 secrets total; values up to 8 KB each.
- The file is JSON, not a database. There is no concurrent-write
  protection beyond atomic-rename — the main process is the only
  writer, and it serializes through the asyncio loop.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from boxbot.core.paths import CREDENTIALS_DIR

logger = logging.getLogger(__name__)


_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_NAME_MAX_LEN = 64
_VALUE_MAX_BYTES = 8 * 1024
_MAX_SECRETS = 64

DEFAULT_PATH = CREDENTIALS_DIR / "secrets.json"


class SecretStoreError(Exception):
    """Raised for invalid operations (bad name, oversized value, full store)."""


def _validate_name(name: Any) -> str:
    if not isinstance(name, str):
        raise SecretStoreError(
            f"secret name must be a string, got {type(name).__name__}"
        )
    if len(name) > _NAME_MAX_LEN:
        raise SecretStoreError(
            f"secret name must be ≤{_NAME_MAX_LEN} chars, got {len(name)}"
        )
    if not _NAME_RE.match(name):
        raise SecretStoreError(
            f"secret name must be SCREAMING_SNAKE_CASE "
            f"(letters, digits, underscores; first char a letter), got '{name}'"
        )
    return name


def _validate_value(value: Any) -> str:
    if not isinstance(value, str):
        raise SecretStoreError(
            f"secret value must be a string, got {type(value).__name__}"
        )
    if not value:
        raise SecretStoreError("secret value must not be empty")
    if len(value.encode("utf-8")) > _VALUE_MAX_BYTES:
        raise SecretStoreError(
            f"secret value must be ≤{_VALUE_MAX_BYTES} bytes UTF-8"
        )
    return value


class SecretStore:
    """File-backed K/V vault for credentials.

    Methods return plain dicts so handlers can pass them through to
    the sandbox unchanged. Values are returned only by :meth:`load`,
    which is the private path the env-injectors call — it is never
    surfaced through any SDK action.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _read(self) -> dict[str, dict[str, Any]]:
        """Return the raw on-disk dict, or empty if the file is absent.

        Schema: ``{name: {"value": str, "stored_at": iso8601}}``.
        """
        if not self.path.exists():
            return {}
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SecretStoreError(f"failed to read {self.path}: {exc}") from exc
        if not text.strip():
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SecretStoreError(
                f"secrets file is not valid JSON: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise SecretStoreError("secrets file root must be a JSON object")
        return data

    def _write(self, data: dict[str, dict[str, Any]]) -> None:
        """Atomically replace the on-disk file with mode 0600.

        On any error after tempfile creation, the tempfile is cleaned
        up so a failed write doesn't leave detritus next to the live
        secrets file.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=".secrets-", suffix=".json.tmp", dir=str(self.path.parent)
        )
        try:
            os.write(fd, json.dumps(data, indent=2, sort_keys=True).encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, name: str, value: str) -> dict[str, Any]:
        """Insert or replace a secret.

        Returns ``{"status": "ok", "name": name, "previous": "created"|"replaced"}``.
        Never echoes the value.
        """
        name = _validate_name(name)
        value = _validate_value(value)
        data = self._read()
        previous = "replaced" if name in data else "created"
        if previous == "created" and len(data) >= _MAX_SECRETS:
            raise SecretStoreError(
                f"secret store is full ({_MAX_SECRETS} max). "
                "Delete an unused secret before adding more."
            )
        data[name] = {
            "value": value,
            "stored_at": _now_iso(),
        }
        self._write(data)
        return {"status": "ok", "name": name, "previous": previous}

    def delete(self, name: str) -> dict[str, Any]:
        """Remove a secret. Returns ``{"status": "ok"|"missing", "name": name}``."""
        name = _validate_name(name)
        data = self._read()
        if name not in data:
            return {"status": "missing", "name": name}
        del data[name]
        self._write(data)
        return {"status": "ok", "name": name}

    def has(self, name: str) -> bool:
        """True iff a secret with this name is stored."""
        try:
            name = _validate_name(name)
        except SecretStoreError:
            return False
        return name in self._read()

    def list_names(self) -> list[dict[str, str]]:
        """Return ``[{"name": ..., "stored_at": ...}, ...]``, sorted by name.

        Values are never returned. Names are credential *labels*, not
        secrets — exposing them lets the agent decide whether it has
        what it needs without prompting the user again.
        """
        data = self._read()
        return [
            {"name": name, "stored_at": entry.get("stored_at", "")}
            for name, entry in sorted(data.items())
        ]

    def count(self) -> int:
        """Number of stored secrets. Cheap; reads the file."""
        return len(self._read())

    def load(self, name: str) -> str | None:
        """Return the secret value, or ``None`` if absent.

        This is the **only** method that returns values. It is called by
        the integration runner and ``execute_script`` tool when
        building the launch env for a sandboxed subprocess. It is not
        reachable from any SDK action.
        """
        try:
            name = _validate_name(name)
        except SecretStoreError:
            return None
        entry = self._read().get(name)
        if entry is None:
            return None
        value = entry.get("value")
        return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_INSTANCE: SecretStore | None = None


def get_secret_store() -> SecretStore:
    """Return the process-wide :class:`SecretStore` (lazy singleton).

    Tests that need a tmp-path store should construct ``SecretStore(path=...)``
    directly rather than going through this accessor.
    """
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = SecretStore()
    return _INSTANCE


def _now_iso() -> str:
    """UTC timestamp in ISO 8601 (seconds resolution, ``Z`` suffix)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
