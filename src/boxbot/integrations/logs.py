"""Per-integration run log — every invocation, retained for debugging.

Stored at ``data/integrations/runs.db`` (SQLite). Every call to
:func:`boxbot.integrations.runner.run` records a row with status,
timing, inputs, outputs (or error), so the agent can debug failures
without operator help — e.g. spot 5 consecutive auth errors and ask
the user for a fresh API key.

Retention: last 100 runs per integration. Pruning runs after every
insert. Cap is small enough that disk impact is negligible (~5MB
total across all integrations) and large enough to span a typical
debugging window.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from boxbot.core.paths import INTEGRATIONS_DIR

logger = logging.getLogger(__name__)

# Retention cap per integration. Old rows beyond this are deleted on
# every insert.
MAX_RUNS_PER_INTEGRATION = 100

# Truncate large payloads in the log to keep one row from blowing the
# budget. Logs are debugging info, not the source of truth — the
# caller already has the live output.
MAX_LOGGED_BYTES = 32 * 1024


def _db_path(root: Path | None = None) -> Path:
    """Resolve the runs.db path, defaulting to ``data/integrations/runs.db``."""
    if root is None:
        root = INTEGRATIONS_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root / "runs.db"


def _truncate_for_log(value: str) -> str:
    if len(value) <= MAX_LOGGED_BYTES:
        return value
    return value[:MAX_LOGGED_BYTES] + f"\n…(truncated, original {len(value)} bytes)"


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                started_at REAL NOT NULL,
                finished_at REAL NOT NULL,
                duration_ms INTEGER NOT NULL,
                status TEXT NOT NULL,
                inputs_json TEXT,
                output_json TEXT,
                error TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_name_started "
            "ON runs (name, started_at DESC)"
        )
        conn.commit()
        yield conn
    finally:
        conn.close()


def record_run(
    *,
    name: str,
    started_at: float,
    finished_at: float,
    status: str,
    inputs: dict[str, Any] | None,
    output: Any | None,
    error: str | None,
    root: Path | None = None,
) -> int:
    """Append a run row and prune old runs for the same integration.

    Returns the rowid of the inserted row. Errors are logged and
    swallowed — the runner shouldn't fail because logging failed.
    """
    if status not in {"ok", "error", "timeout"}:
        logger.warning("Unknown run status '%s' for '%s'; storing anyway", status, name)

    duration_ms = max(0, int((finished_at - started_at) * 1000))
    inputs_json = (
        _truncate_for_log(json.dumps(inputs, default=str)) if inputs is not None else None
    )
    output_json = (
        _truncate_for_log(json.dumps(output, default=str)) if output is not None else None
    )
    error_text = _truncate_for_log(error) if error else None

    try:
        path = _db_path(root)
        with _connect(path) as conn:
            cur = conn.execute(
                "INSERT INTO runs "
                "(name, started_at, finished_at, duration_ms, status, "
                " inputs_json, output_json, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (name, started_at, finished_at, duration_ms, status,
                 inputs_json, output_json, error_text),
            )
            row_id = int(cur.lastrowid or 0)
            # Prune to the most recent MAX_RUNS_PER_INTEGRATION rows for this name.
            conn.execute(
                "DELETE FROM runs WHERE name = ? AND id NOT IN "
                "(SELECT id FROM runs WHERE name = ? "
                " ORDER BY started_at DESC, id DESC LIMIT ?)",
                (name, name, MAX_RUNS_PER_INTEGRATION),
            )
            conn.commit()
            return row_id
    except sqlite3.Error as exc:
        logger.warning("Failed to record integration run for '%s': %s", name, exc)
        return 0


def list_runs(
    name: str,
    *,
    limit: int = 20,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent runs for ``name``, newest first.

    Each row is a plain dict with parsed inputs/outputs (None if the
    column was empty or not valid JSON).
    """
    if limit <= 0:
        return []
    try:
        path = _db_path(root)
        with _connect(path) as conn:
            rows = conn.execute(
                "SELECT id, name, started_at, finished_at, duration_ms, "
                " status, inputs_json, output_json, error "
                "FROM runs WHERE name = ? "
                "ORDER BY started_at DESC, id DESC LIMIT ?",
                (name, limit),
            ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("Failed to list runs for '%s': %s", name, exc)
        return []

    return [_row_to_dict(row) for row in rows]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    def _maybe_json(text: Any) -> Any:
        if text is None:
            return None
        try:
            return json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return text

    return {
        "id": row["id"],
        "name": row["name"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "duration_ms": row["duration_ms"],
        "status": row["status"],
        "inputs": _maybe_json(row["inputs_json"]),
        "output": _maybe_json(row["output_json"]),
        "error": row["error"],
    }


def now() -> float:
    """Wall-clock time helper, isolated for tests that want to freeze it."""
    return time.time()
