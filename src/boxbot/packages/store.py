"""Durable store for package-install requests.

SQLite at ``data/packages/packages.db`` — same conventions as the
scheduler and auth stores (aiosqlite, WAL, ``CREATE TABLE IF NOT
EXISTS`` on open, ISO-8601 timestamps).

Statuses:

- ``pending``    — queued, waiting for an admin reply
- ``approved``   — admin said yes; install in flight
- ``denied``     — admin said no (``note`` may carry the reason)
- ``installed``  — pip succeeded; package available in the sandbox venv
- ``failed``     — pip (or the permission fixup) failed; ``note`` has
  the tail of the error output

Request ids are short hex strings (8 chars) so the admin reply
``approve pkg ab12cd34`` stays typeable on a phone.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import aiosqlite

from boxbot.core.paths import PACKAGES_DIR

DB_PATH = PACKAGES_DIR / "packages.db"

VALID_STATUSES = ("pending", "approved", "denied", "installed", "failed")

_DDL = """\
CREATE TABLE IF NOT EXISTS package_requests (
    id TEXT PRIMARY KEY,
    package TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    requested_by TEXT,
    requested_at TEXT NOT NULL,
    resolved_by TEXT,
    resolved_at TEXT,
    note TEXT
)"""

_COLUMNS = (
    "id, package, reason, status, requested_by, requested_at, "
    "resolved_by, resolved_at, note"
)


class PackageStore:
    """Async SQLite store for package-install requests.

    Args:
        db_path: Path to the SQLite database. Defaults to
            ``data/packages/packages.db``.
    """

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self._db_path = Path(db_path)

    async def _get_db(self) -> aiosqlite.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        db = await aiosqlite.connect(str(self._db_path))
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute(_DDL)
        await db.commit()
        return db

    async def create_request(
        self,
        package: str,
        reason: str,
        *,
        requested_by: str | None = None,
    ) -> dict:
        """Insert a new pending request and return its row dict."""
        request_id = uuid.uuid4().hex[:8]
        now = datetime.now().isoformat(timespec="seconds")
        db = await self._get_db()
        try:
            await db.execute(
                "INSERT INTO package_requests "
                "(id, package, reason, status, requested_by, requested_at) "
                "VALUES (?, ?, ?, 'pending', ?, ?)",
                (request_id, package, reason, requested_by, now),
            )
            await db.commit()
        finally:
            await db.close()
        return {
            "id": request_id,
            "package": package,
            "reason": reason,
            "status": "pending",
            "requested_by": requested_by,
            "requested_at": now,
            "resolved_by": None,
            "resolved_at": None,
            "note": None,
        }

    async def get_request(self, request_id: str) -> dict | None:
        """Return the request row dict, or None. Id match is exact."""
        db = await self._get_db()
        try:
            cursor = await db.execute(
                f"SELECT {_COLUMNS} FROM package_requests WHERE id = ?",
                (request_id,),
            )
            row = await cursor.fetchone()
        finally:
            await db.close()
        return dict(row) if row is not None else None

    async def find_pending(self, package: str) -> dict | None:
        """Return the oldest pending request for ``package``, or None."""
        db = await self._get_db()
        try:
            cursor = await db.execute(
                f"SELECT {_COLUMNS} FROM package_requests "
                "WHERE package = ? AND status = 'pending' "
                "ORDER BY requested_at LIMIT 1",
                (package,),
            )
            row = await cursor.fetchone()
        finally:
            await db.close()
        return dict(row) if row is not None else None

    async def list_requests(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return requests, newest first, optionally filtered by status."""
        db = await self._get_db()
        try:
            if status is not None:
                cursor = await db.execute(
                    f"SELECT {_COLUMNS} FROM package_requests "
                    "WHERE status = ? "
                    "ORDER BY requested_at DESC LIMIT ?",
                    (status, limit),
                )
            else:
                cursor = await db.execute(
                    f"SELECT {_COLUMNS} FROM package_requests "
                    "ORDER BY requested_at DESC LIMIT ?",
                    (limit,),
                )
            rows = await cursor.fetchall()
        finally:
            await db.close()
        return [dict(r) for r in rows]

    async def set_status(
        self,
        request_id: str,
        status: str,
        *,
        resolved_by: str | None = None,
        note: str | None = None,
        expect: str | None = None,
    ) -> dict | None:
        """Transition a request to ``status``; return the updated row.

        Args:
            request_id: The request to update.
            status: New status (one of :data:`VALID_STATUSES`).
            resolved_by: Admin phone (or other principal) responsible.
            note: Deny reason / install error tail.
            expect: When given, the update only applies if the current
                status matches — the compare-and-set that prevents two
                admins double-approving (or an install result clobbering
                a manual fix).

        Returns:
            The updated row dict, or None if the request doesn't exist
            or ``expect`` didn't match.
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status {status!r}")
        now = datetime.now().isoformat(timespec="seconds")
        db = await self._get_db()
        try:
            if expect is not None:
                cursor = await db.execute(
                    "UPDATE package_requests "
                    "SET status = ?, resolved_by = COALESCE(?, resolved_by), "
                    "    resolved_at = ?, note = COALESCE(?, note) "
                    "WHERE id = ? AND status = ?",
                    (status, resolved_by, now, note, request_id, expect),
                )
            else:
                cursor = await db.execute(
                    "UPDATE package_requests "
                    "SET status = ?, resolved_by = COALESCE(?, resolved_by), "
                    "    resolved_at = ?, note = COALESCE(?, note) "
                    "WHERE id = ?",
                    (status, resolved_by, now, note, request_id),
                )
            await db.commit()
            updated = cursor.rowcount > 0
        finally:
            await db.close()
        if not updated:
            return None
        return await self.get_request(request_id)


# Process-wide singleton — set by tests / main, lazily defaulted.
_store: PackageStore | None = None


def get_package_store() -> PackageStore:
    """Return the process-wide PackageStore, creating the default lazily."""
    global _store
    if _store is None:
        _store = PackageStore()
    return _store


def set_package_store(store: PackageStore | None) -> None:
    """Replace the process-wide PackageStore (tests, alternate paths)."""
    global _store
    _store = store
