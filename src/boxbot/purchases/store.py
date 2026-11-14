"""Durable store for purchases and the decisions made about them.

SQLite at ``data/purchases/purchases.db`` (file mode 0600 — it holds
checkout review links). Same conventions as the package store:
aiosqlite, WAL, ``CREATE TABLE IF NOT EXISTS`` on open. Timestamps are
ISO-8601 **UTC** with a fixed width (see :func:`utc_iso`), so they sort
as strings and convert to an instant without daylight-saving ambiguity.

Purchase statuses:

| Status | Meaning |
|---|---|
| ``review`` | checkout built; a human finishes it from the review link |
| ``scheduled`` | approved for automatic purchase; inside the veto window |
| ``verifying`` | veto window over; checkout being re-read. No payment sent yet |
| ``paying`` | set immediately before the one payment request is sent |
| ``ordered`` | Shop confirmed the order |
| ``cancelled`` | stopped by an admin or the agent before payment |
| ``failed`` | Shop refused payment; nothing was bought |
| ``unknown`` | payment outcome could not be determined; never retried |

``verifying`` and ``paying`` are separate so a crash can be read
correctly afterwards: a row left in ``verifying`` never had payment
sent; a row left in ``paying`` may have.

``shop_approvals`` holds stores an admin has allowed for a limited
time by text command (``allow shop <domain>``), on top of the
configured allow-list.

``decisions`` holds one row per policy evaluation — the deterministic
rule results plus the checker model's full input and raw output. It is
the training set for a future local policy model, so nothing in it is
truncated.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from boxbot.core.paths import PURCHASES_DIR

DB_PATH = PURCHASES_DIR / "purchases.db"

VALID_STATUSES = (
    "review", "scheduled", "verifying", "paying", "ordered",
    "cancelled", "failed", "unknown",
)
# Money that is spent or may still be spent — counted against caps and
# against the duplicate-purchase rule.
COMMITTED_STATUSES = ("scheduled", "verifying", "paying", "ordered", "unknown")

_DDL = (
    """\
CREATE TABLE IF NOT EXISTS purchases (
    id TEXT PRIMARY KEY,
    shop TEXT NOT NULL,
    items TEXT NOT NULL,
    total_cents INTEGER,
    currency TEXT,
    reason TEXT NOT NULL,
    requested_by TEXT,
    channel_key TEXT,
    status TEXT NOT NULL,
    decision TEXT,
    checkout_id TEXT NOT NULL,
    continue_url TEXT,
    complete_after TEXT,
    link_sent_at TEXT,
    order_ref TEXT,
    note TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
    """\
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    purchase_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    rules TEXT NOT NULL,
    rules_pass INTEGER NOT NULL,
    checker_model TEXT,
    checker_input TEXT,
    checker_output TEXT,
    checker_decision TEXT,
    checker_latency_ms INTEGER,
    final TEXT NOT NULL
)""",
    """\
CREATE TABLE IF NOT EXISTS shop_approvals (
    shop TEXT PRIMARY KEY,
    approved_by TEXT NOT NULL,
    approved_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
)""",
    "CREATE INDEX IF NOT EXISTS idx_purchases_created ON purchases(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_purchase ON decisions(purchase_id)",
)

_COLUMNS = (
    "id, shop, items, total_cents, currency, reason, requested_by, "
    "channel_key, status, "
    "decision, checkout_id, continue_url, complete_after, link_sent_at, "
    "order_ref, note, created_at, updated_at"
)
_UPDATABLE = {
    "decision", "complete_after", "link_sent_at", "order_ref", "note",
    "total_cents", "currency", "continue_url",
}


def utc_iso(moment: datetime | None = None) -> str:
    """``moment`` (default: now) as a fixed-width UTC ISO-8601 string.

    A naive ``moment`` is taken to be local time. Fixed width so the
    strings sort chronologically; microseconds so "did the person speak
    after that purchase" has a real answer.
    """
    moment = moment or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _now() -> str:
    return utc_iso()


def _row(row: aiosqlite.Row) -> dict[str, Any]:
    record = dict(row)
    try:
        record["items"] = json.loads(record["items"])
    except (TypeError, ValueError):
        record["items"] = []
    return record


class PurchaseStore:
    """Async SQLite store for purchases and policy decisions."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self._db_path = Path(db_path)

    async def _get_db(self) -> aiosqlite.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self._db_path.exists():
            # Owner-only from the first byte, and the directory too even
            # if it already existed with looser permissions.
            os.chmod(self._db_path.parent, 0o700)
            os.close(os.open(
                self._db_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            ))
        db = await aiosqlite.connect(str(self._db_path))
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        for statement in _DDL:
            await db.execute(statement)
        await db.commit()
        return db

    def exists(self) -> bool:
        """True once a purchase database has been created."""
        return self._db_path.exists() and self._db_path.stat().st_size > 0

    # -- purchases ---------------------------------------------------------

    async def create(
        self,
        *,
        shop: str,
        items: list[dict[str, Any]],
        total_cents: int | None,
        currency: str | None,
        reason: str,
        requested_by: str | None,
        channel_key: str | None,
        checkout_id: str,
        continue_url: str | None,
    ) -> dict[str, Any]:
        """Insert a purchase in ``review`` and return its row."""
        purchase_id = uuid.uuid4().hex[:8]
        now = _now()
        db = await self._get_db()
        try:
            await db.execute(
                "INSERT INTO purchases (id, shop, items, total_cents, "
                "currency, reason, requested_by, channel_key, status, "
                "checkout_id, continue_url, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'review', ?, ?, ?, ?)",
                (purchase_id, shop, json.dumps(items), total_cents, currency,
                 reason, requested_by, channel_key, checkout_id, continue_url,
                 now, now),
            )
            await db.commit()
        finally:
            await db.close()
        created = await self.get(purchase_id)
        assert created is not None
        return created

    async def get(self, purchase_id: str) -> dict[str, Any] | None:
        db = await self._get_db()
        try:
            cursor = await db.execute(
                f"SELECT {_COLUMNS} FROM purchases WHERE id = ?", (purchase_id,)
            )
            row = await cursor.fetchone()
        finally:
            await db.close()
        return _row(row) if row is not None else None

    async def list(
        self, *, status: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        db = await self._get_db()
        try:
            if status is not None:
                cursor = await db.execute(
                    f"SELECT {_COLUMNS} FROM purchases WHERE status = ? "
                    "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (status, limit),
                )
            else:
                cursor = await db.execute(
                    f"SELECT {_COLUMNS} FROM purchases "
                    "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (limit,),
                )
            rows = await cursor.fetchall()
        finally:
            await db.close()
        return [_row(r) for r in rows]

    async def transition(
        self,
        purchase_id: str,
        status: str,
        *,
        expect: str | tuple[str, ...],
        **fields: Any,
    ) -> dict[str, Any] | None:
        """Compare-and-set status; return the updated row or None.

        The update applies only while the current status is in
        ``expect``. That single atomic statement is what guarantees one
        purchase is paid for at most once (``scheduled → verifying``
        and ``verifying → paying`` each succeed for only one caller)
        and that a cancel and a payment cannot both win.
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status {status!r}")
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"cannot update {sorted(unknown)}")
        expected = (expect,) if isinstance(expect, str) else tuple(expect)
        assignments = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status, _now()]
        for name, value in fields.items():
            assignments.append(f"{name} = ?")
            params.append(value)
        placeholders = ", ".join("?" for _ in expected)
        params.extend([purchase_id, *expected])
        db = await self._get_db()
        try:
            cursor = await db.execute(
                f"UPDATE purchases SET {', '.join(assignments)} "
                f"WHERE id = ? AND status IN ({placeholders})",
                params,
            )
            await db.commit()
            changed = cursor.rowcount > 0
        finally:
            await db.close()
        return await self.get(purchase_id) if changed else None

    async def update_fields(self, purchase_id: str, **fields: Any) -> None:
        """Set non-status columns without touching the status."""
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"cannot update {sorted(unknown)}")
        if not fields:
            return
        assignments = [f"{name} = ?" for name in fields] + ["updated_at = ?"]
        db = await self._get_db()
        try:
            await db.execute(
                f"UPDATE purchases SET {', '.join(assignments)} WHERE id = ?",
                [*fields.values(), _now(), purchase_id],
            )
            await db.commit()
        finally:
            await db.close()

    async def committed_cents_since(self, since_iso: str) -> int:
        """Total of purchases that spent, or may still spend, money."""
        placeholders = ", ".join("?" for _ in COMMITTED_STATUSES)
        db = await self._get_db()
        try:
            cursor = await db.execute(
                "SELECT COALESCE(SUM(total_cents), 0) FROM purchases "
                f"WHERE created_at >= ? AND status IN ({placeholders})",
                (since_iso, *COMMITTED_STATUSES),
            )
            row = await cursor.fetchone()
        finally:
            await db.close()
        return int(row[0] or 0)

    async def since(self, since_iso: str) -> list[dict[str, Any]]:
        """Every purchase proposed since ``since_iso``, newest first."""
        db = await self._get_db()
        try:
            cursor = await db.execute(
                f"SELECT {_COLUMNS} FROM purchases WHERE created_at >= ? "
                "ORDER BY created_at DESC, rowid DESC",
                (since_iso,),
            )
            rows = await cursor.fetchall()
        finally:
            await db.close()
        return [_row(r) for r in rows]

    async def last_proposal_at(self, channel_key: str) -> str | None:
        """``created_at`` of the newest purchase proposed from a conversation."""
        db = await self._get_db()
        try:
            cursor = await db.execute(
                "SELECT MAX(created_at) FROM purchases WHERE channel_key = ?",
                (channel_key,),
            )
            row = await cursor.fetchone()
        finally:
            await db.close()
        return row[0] if row and row[0] else None

    async def count_since(self, since_iso: str) -> int:
        """Number of purchases proposed since ``since_iso``, any status."""
        db = await self._get_db()
        try:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM purchases WHERE created_at >= ?",
                (since_iso,),
            )
            row = await cursor.fetchone()
        finally:
            await db.close()
        return int(row[0] or 0)

    # -- shop approvals ----------------------------------------------------

    async def approve_shop(self, shop: str, *, approved_by: str, expires_at: str) -> None:
        """Record (or renew) an admin's time-limited approval of a store."""
        db = await self._get_db()
        try:
            await db.execute(
                "INSERT INTO shop_approvals (shop, approved_by, approved_at, "
                "expires_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(shop) DO UPDATE SET approved_by = excluded.approved_by, "
                "approved_at = excluded.approved_at, expires_at = excluded.expires_at",
                (shop, approved_by, _now(), expires_at),
            )
            await db.commit()
        finally:
            await db.close()

    async def shop_approved(self, shop: str) -> bool:
        """True while an admin's approval of ``shop`` has not expired."""
        db = await self._get_db()
        try:
            cursor = await db.execute(
                "SELECT 1 FROM shop_approvals WHERE shop = ? AND expires_at > ?",
                (shop, _now()),
            )
            row = await cursor.fetchone()
        finally:
            await db.close()
        return row is not None

    # -- decisions ---------------------------------------------------------

    async def record_decision(
        self,
        *,
        purchase_id: str,
        mode: str,
        rules: list[dict[str, Any]],
        rules_pass: bool,
        checker_model: str | None,
        checker_input: str | None,
        checker_output: str | None,
        checker_decision: str | None,
        checker_latency_ms: int | None,
        final: str,
    ) -> None:
        db = await self._get_db()
        try:
            await db.execute(
                "INSERT INTO decisions (purchase_id, created_at, mode, rules, "
                "rules_pass, checker_model, checker_input, checker_output, "
                "checker_decision, checker_latency_ms, final) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (purchase_id, _now(), mode, json.dumps(rules),
                 1 if rules_pass else 0, checker_model, checker_input,
                 checker_output, checker_decision, checker_latency_ms, final),
            )
            await db.commit()
        finally:
            await db.close()

    async def decisions_for(self, purchase_id: str) -> list[dict[str, Any]]:
        db = await self._get_db()
        try:
            cursor = await db.execute(
                "SELECT * FROM decisions WHERE purchase_id = ? ORDER BY id",
                (purchase_id,),
            )
            rows = await cursor.fetchall()
        finally:
            await db.close()
        return [dict(r) for r in rows]


_store: PurchaseStore | None = None


def get_purchase_store() -> PurchaseStore:
    """Return the process-wide PurchaseStore, creating the default lazily."""
    global _store
    if _store is None:
        _store = PurchaseStore()
    return _store


def set_purchase_store(store: PurchaseStore | None) -> None:
    """Replace the process-wide PurchaseStore (tests, alternate paths)."""
    global _store
    _store = store
