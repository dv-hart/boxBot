"""Agent task management and wake/sleep lifecycle.

Manages two subsystems — **triggers** (event-driven wake conditions) and
**to-do items** (persistent action items) — plus a background process that
evaluates trigger conditions and emits events when they fire.

This is the agent's *internal planning system*, distinct from the family
calendar. Calendar events are managed by calendar skills. The scheduler
manages when the agent wakes, what it needs to do, and what it's tracking.

Usage:
    from boxbot.core.scheduler import Scheduler

    scheduler = Scheduler()
    await scheduler.start()          # background loop + event subscriptions
    ...
    await scheduler.stop()           # graceful shutdown

    # CRUD (usable before start() — only needs init_db)
    await scheduler.init_db()
    tid = await create_trigger(description="...", instructions="...", fire_after="30m")
    did = await create_todo(description="Return library books")
    line = await get_status_line()   # "[To-do: 3 items | Triggers: 2 active]"
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite

from boxbot.core.config import BoxBotConfig, get_config
from boxbot.core.events import PersonIdentified, TriggerFired, get_event_bus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = Path("data/scheduler/scheduler.db")

_TRIGGERS_DDL = """\
CREATE TABLE IF NOT EXISTS triggers (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    instructions TEXT NOT NULL,
    fire_at TEXT,
    cron TEXT,
    person TEXT,
    for_person TEXT,
    todo_id TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    source TEXT NOT NULL DEFAULT 'agent',
    created_at TEXT NOT NULL,
    expires TEXT,
    last_fired TEXT,
    fire_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (todo_id) REFERENCES todos(id)
)"""

_TODOS_DDL = """\
CREATE TABLE IF NOT EXISTS todos (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    notes TEXT,
    for_person TEXT,
    due_date TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    completed_at TEXT,
    source TEXT NOT NULL DEFAULT 'agent'
)"""

# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"^(\d+)\s*([mhd])$", re.IGNORECASE)
_MAX_DURATION = timedelta(hours=24)


def parse_duration(duration_str: str) -> timedelta:
    """Parse a human-friendly duration string into a timedelta.

    Supported formats: "30m" (minutes), "2h" (hours), "1d" (days).
    Raises ValueError for durations exceeding 24 hours.
    """
    m = _DURATION_RE.match(duration_str.strip())
    if not m:
        raise ValueError(
            f"Invalid duration format: {duration_str!r}. "
            "Use '<number><m|h|d>' (e.g. '30m', '2h', '1d')."
        )
    value = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "m":
        td = timedelta(minutes=value)
    elif unit == "h":
        td = timedelta(hours=value)
    else:  # "d"
        td = timedelta(days=value)

    if td > _MAX_DURATION:
        raise ValueError(
            f"Duration {duration_str!r} exceeds maximum of 24 hours."
        )
    return td


# ---------------------------------------------------------------------------
# Cron expression support (basic, no croniter dependency)
# ---------------------------------------------------------------------------


class CronExpr:
    """Minimal cron expression parser supporting 5-field standard cron.

    Fields: minute hour day-of-month month day-of-week

    Supports:
    - Wildcards (``*``)
    - Single values (``0``, ``7``, ``15``)
    - Ranges (``1-5``)
    - Lists (``1,3,5``)
    - Step values (``*/15``, ``1-30/5``)
    """

    __slots__ = ("minutes", "hours", "days", "months", "weekdays")

    def __init__(self, expr: str) -> None:
        parts = expr.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"Cron expression must have 5 fields, got {len(parts)}: {expr!r}"
            )
        self.minutes = self._parse_field(parts[0], 0, 59)
        self.hours = self._parse_field(parts[1], 0, 23)
        self.days = self._parse_field(parts[2], 1, 31)
        self.months = self._parse_field(parts[3], 1, 12)
        self.weekdays = self._parse_field(parts[4], 0, 6)  # 0=Mon ... 6=Sun

    @staticmethod
    def _parse_field(field: str, lo: int, hi: int) -> set[int]:
        """Parse one cron field into a set of valid integer values."""
        values: set[int] = set()
        for part in field.split(","):
            step = 1
            if "/" in part:
                range_part, step_str = part.split("/", 1)
                step = int(step_str)
            else:
                range_part = part

            if range_part == "*":
                start, end = lo, hi
            elif "-" in range_part:
                s, e = range_part.split("-", 1)
                start, end = int(s), int(e)
            else:
                val = int(range_part)
                values.add(val)
                continue

            values.update(range(start, end + 1, step))

        return values

    def matches(self, dt: datetime) -> bool:
        """Return True if ``dt`` matches this cron expression."""
        # Python weekday: 0=Monday. Cron standard here: 0=Monday.
        return (
            dt.minute in self.minutes
            and dt.hour in self.hours
            and dt.day in self.days
            and dt.month in self.months
            and dt.weekday() in self.weekdays
        )

    def next_occurrence(self, after: datetime) -> datetime:
        """Return the next datetime matching this expression after ``after``.

        Steps forward minute-by-minute (capped at ~2 years to avoid infinite
        loops on impossible expressions).
        """
        # Start from the next minute boundary
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        limit = after + timedelta(days=366 * 2)
        while candidate < limit:
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        raise ValueError(
            f"No occurrence found within 2 years for cron expression"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s: str | None) -> datetime | None:
    """Parse an ISO 8601 string to a timezone-aware datetime, or None."""
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    """Convert an aiosqlite Row to a plain dict."""
    return dict(row)


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------


async def _get_db() -> aiosqlite.Connection:
    """Open (or create) the scheduler database, ensuring tables exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await db.execute(_TRIGGERS_DDL)
    await db.execute(_TODOS_DDL)
    await db.commit()
    return db


async def init_db() -> None:
    """Ensure the database and tables exist.

    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    """
    db = await _get_db()
    await db.close()


# ── Triggers ──────────────────────────────────────────────────────────────


async def create_trigger(
    description: str,
    instructions: str,
    *,
    fire_at: str | None = None,
    fire_after: str | None = None,
    cron: str | None = None,
    person: str | None = None,
    for_person: str | None = None,
    expires: str | None = None,
    todo_id: str | None = None,
    source: str = "agent",
) -> str:
    """Create a new trigger and return its ID.

    Args:
        description: Human-readable summary.
        instructions: What the agent should do when fired.
        fire_at: Absolute ISO datetime for the time condition.
        fire_after: Relative duration (e.g. "30m"). Converted to ``fire_at``.
        cron: Cron expression (recurring). Mutually exclusive with fire_at/fire_after.
        person: Person-presence condition.
        for_person: Who this task relates to (context).
        expires: Explicit expiry datetime (ISO).
        todo_id: Optional link to a to-do item.
        source: One of "config", "agent", "conversation".

    Returns:
        The new trigger's ID (prefixed with "t_").
    """
    trigger_id = f"t_{uuid4().hex[:12]}"
    now = _now_iso()

    # Resolve fire_after → fire_at
    resolved_fire_at = fire_at
    if fire_after is not None:
        if fire_at is not None or cron is not None:
            raise ValueError(
                "fire_after is mutually exclusive with fire_at and cron"
            )
        td = parse_duration(fire_after)
        resolved_fire_at = (_now_utc() + td).isoformat()

    # Validate cron exclusivity
    if cron is not None and (fire_at is not None or fire_after is not None):
        raise ValueError("cron is mutually exclusive with fire_at/fire_after")

    # Validate cron expression
    if cron is not None:
        CronExpr(cron)  # raises on bad syntax

    # For recurring cron triggers, compute the first fire_at from the expression
    if cron is not None and resolved_fire_at is None:
        next_time = CronExpr(cron).next_occurrence(_now_utc())
        resolved_fire_at = next_time.isoformat()

    # Apply default expiry rules
    if expires is None:
        config = _try_get_config()
        person_expiry_days = (
            config.schedule.person_trigger_expiry_days if config else 7
        )
        if cron is not None:
            # Recurring triggers do not auto-expire
            expires = None
        elif person is not None and resolved_fire_at is not None:
            # Time + person: fire_at + person expiry window
            ft = _parse_iso(resolved_fire_at)
            if ft is not None:
                expires = (ft + timedelta(days=person_expiry_days)).isoformat()
        elif person is not None:
            # Person-only: default window from now
            expires = (
                _now_utc() + timedelta(days=person_expiry_days)
            ).isoformat()
        # Time-only with no person: no auto-expiry needed (fires once, done)

    db = await _get_db()
    try:
        await db.execute(
            """INSERT INTO triggers
               (id, description, instructions, fire_at, cron, person,
                for_person, todo_id, status, source, created_at, expires,
                last_fired, fire_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, NULL, 0)""",
            (
                trigger_id,
                description,
                instructions,
                resolved_fire_at,
                cron,
                person,
                for_person,
                todo_id,
                source,
                now,
                expires,
            ),
        )
        await db.commit()
    finally:
        await db.close()

    logger.info("Created trigger %s: %s", trigger_id, description)
    return trigger_id


async def get_trigger(trigger_id: str) -> dict[str, Any] | None:
    """Fetch a single trigger by ID, or None if not found."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM triggers WHERE id = ?", (trigger_id,)
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None
    finally:
        await db.close()


async def list_triggers(
    *,
    status: str | None = None,
    for_person: str | None = None,
) -> list[dict[str, Any]]:
    """List triggers with optional filters.

    Args:
        status: Filter by status (e.g. "active", "fired").
        for_person: Filter by for_person field.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if for_person is not None:
        clauses.append("for_person = ?")
        params.append(for_person)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    db = await _get_db()
    try:
        cursor = await db.execute(
            f"SELECT * FROM triggers{where} ORDER BY created_at DESC", params
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        await db.close()


async def update_trigger(trigger_id: str, **fields: Any) -> bool:
    """Update specific fields on a trigger. Returns True if a row was updated."""
    allowed = {
        "description",
        "instructions",
        "fire_at",
        "cron",
        "person",
        "for_person",
        "todo_id",
        "status",
        "expires",
        "last_fired",
        "fire_count",
    }
    to_set = {k: v for k, v in fields.items() if k in allowed}
    if not to_set:
        return False

    set_clause = ", ".join(f"{k} = ?" for k in to_set)
    params = list(to_set.values()) + [trigger_id]

    db = await _get_db()
    try:
        cursor = await db.execute(
            f"UPDATE triggers SET {set_clause} WHERE id = ?", params
        )
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


async def cancel_trigger(trigger_id: str) -> bool:
    """Cancel an active trigger."""
    return await update_trigger(trigger_id, status="cancelled")


# ── Todos ─────────────────────────────────────────────────────────────────


async def create_todo(
    description: str,
    *,
    notes: str | None = None,
    for_person: str | None = None,
    due_date: str | None = None,
    source: str = "agent",
) -> str:
    """Create a new to-do item and return its ID.

    Args:
        description: Brief summary shown in list view.
        notes: Detailed context loaded on demand.
        for_person: Who this relates to.
        due_date: Soft deadline (ISO date string).
        source: One of "agent", "conversation".

    Returns:
        The new to-do's ID (prefixed with "d_").
    """
    todo_id = f"d_{uuid4().hex[:12]}"
    now = _now_iso()

    db = await _get_db()
    try:
        await db.execute(
            """INSERT INTO todos
               (id, description, notes, for_person, due_date, status,
                created_at, completed_at, source)
               VALUES (?, ?, ?, ?, ?, 'pending', ?, NULL, ?)""",
            (todo_id, description, notes, for_person, due_date, now, source),
        )
        await db.commit()
    finally:
        await db.close()

    logger.info("Created todo %s: %s", todo_id, description)
    return todo_id


async def get_todo(todo_id: str) -> dict[str, Any] | None:
    """Fetch a single to-do item by ID, or None if not found."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM todos WHERE id = ?", (todo_id,)
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None
    finally:
        await db.close()


async def list_todos(
    *,
    status: str | None = None,
    for_person: str | None = None,
) -> list[dict[str, Any]]:
    """List to-do items with optional filters.

    Args:
        status: Filter by status (e.g. "pending", "completed").
        for_person: Filter by for_person field.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if for_person is not None:
        clauses.append("for_person = ?")
        params.append(for_person)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    db = await _get_db()
    try:
        cursor = await db.execute(
            f"SELECT * FROM todos{where} ORDER BY created_at DESC", params
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        await db.close()


async def update_todo(todo_id: str, **fields: Any) -> bool:
    """Update specific fields on a to-do item. Returns True if a row was updated."""
    allowed = {
        "description",
        "notes",
        "for_person",
        "due_date",
        "status",
        "completed_at",
    }
    to_set = {k: v for k, v in fields.items() if k in allowed}
    if not to_set:
        return False

    set_clause = ", ".join(f"{k} = ?" for k in to_set)
    params = list(to_set.values()) + [todo_id]

    db = await _get_db()
    try:
        cursor = await db.execute(
            f"UPDATE todos SET {set_clause} WHERE id = ?", params
        )
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


async def complete_todo(todo_id: str) -> bool:
    """Mark a to-do item as completed."""
    return await update_todo(
        todo_id, status="completed", completed_at=_now_iso()
    )


async def cancel_todo(todo_id: str) -> bool:
    """Cancel a to-do item."""
    return await update_todo(todo_id, status="cancelled")


# ── Status line ───────────────────────────────────────────────────────────


async def get_status_line() -> str:
    """Return a compact status line for conversation start injection.

    Format: ``[To-do: N items | Triggers: N active]``
    """
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM todos WHERE status = 'pending'"
        )
        todo_count = (await cursor.fetchone())[0]

        cursor = await db.execute(
            "SELECT COUNT(*) FROM triggers WHERE status = 'active'"
        )
        trigger_count = (await cursor.fetchone())[0]

        return f"[To-do: {todo_count} items | Triggers: {trigger_count} active]"
    finally:
        await db.close()


# ── Config seeding ────────────────────────────────────────────────────────


async def seed_from_config(config: BoxBotConfig) -> None:
    """Seed wake-cycle triggers from config on first boot.

    Only seeds if the triggers table is empty (no pre-existing data).
    After seeding, the runtime DB is authoritative.
    """
    db = await _get_db()
    try:
        cursor = await db.execute("SELECT COUNT(*) FROM triggers")
        count = (await cursor.fetchone())[0]
        if count > 0:
            logger.info(
                "Triggers table has %d rows, skipping config seed", count
            )
            return
    finally:
        await db.close()

    for entry in config.schedule.wake_cycle:
        await create_trigger(
            description=entry.description,
            instructions=entry.instructions,
            cron=entry.cron,
            source="config",
        )
    logger.info(
        "Seeded %d wake-cycle triggers from config",
        len(config.schedule.wake_cycle),
    )


# ---------------------------------------------------------------------------
# Trigger condition evaluation
# ---------------------------------------------------------------------------


def evaluate_time_condition(trigger: dict[str, Any]) -> bool:
    """Check whether a trigger's time condition is met.

    Time conditions are met when current time >= fire_at. Once met, they
    stay met (the trigger then waits only on remaining conditions).

    Returns True if there is no time condition (vacuously true) or if the
    time has passed.
    """
    fire_at = _parse_iso(trigger.get("fire_at"))
    if fire_at is None:
        return True
    return _now_utc() >= fire_at


def evaluate_person_condition(
    trigger: dict[str, Any],
    present_people: set[str],
) -> bool:
    """Check whether a trigger's person condition is met.

    Person conditions are transient — the person must be present at
    evaluation time.

    Returns True if there is no person condition (vacuously true) or if
    the person is in ``present_people``.
    """
    person = trigger.get("person")
    if not person:
        return True
    return person in present_people


def evaluate_trigger(
    trigger: dict[str, Any],
    present_people: set[str] | None = None,
) -> bool:
    """Evaluate all conditions on a trigger (AND logic).

    Returns True only if every specified condition is met. Only considers
    active triggers.
    """
    if trigger.get("status") != "active":
        return False
    if present_people is None:
        present_people = set()
    return evaluate_time_condition(trigger) and evaluate_person_condition(
        trigger, present_people
    )


# ---------------------------------------------------------------------------
# Helpers (internal)
# ---------------------------------------------------------------------------


def _try_get_config() -> BoxBotConfig | None:
    """Try to get the config singleton, returning None if not loaded."""
    try:
        return get_config()
    except RuntimeError:
        return None


def _is_expired(trigger: dict[str, Any]) -> bool:
    """Check whether a trigger has passed its expiry time."""
    expires = _parse_iso(trigger.get("expires"))
    if expires is None:
        return False
    return _now_utc() >= expires


# ---------------------------------------------------------------------------
# Scheduler class (background process)
# ---------------------------------------------------------------------------


class Scheduler:
    """Background process that monitors trigger conditions and fires events.

    Responsibilities:
    - Check time-based triggers every 60 seconds
    - Subscribe to ``PersonIdentified`` events for person conditions
    - Emit ``TriggerFired`` events when all conditions are met
    - Handle recurring trigger reset and expiry sweeps
    """

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._present_people: set[str] = set()
        self._person_last_seen: dict[str, datetime] = {}
        # How long a person remains "present" after last detection
        self._person_presence_window = timedelta(minutes=5)

    async def start(self) -> None:
        """Start the background scheduler loop and event subscriptions."""
        if self._running:
            return

        await init_db()

        # Seed config on first boot
        config = _try_get_config()
        if config:
            await seed_from_config(config)

        bus = get_event_bus()
        bus.subscribe(PersonIdentified, self._on_person_identified)

        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="scheduler")
        logger.info("Scheduler started")

    async def stop(self) -> None:
        """Stop the background scheduler loop."""
        if not self._running:
            return

        self._running = False
        bus = get_event_bus()
        bus.unsubscribe(PersonIdentified, self._on_person_identified)

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Scheduler stopped")

    async def _run_loop(self) -> None:
        """Main background loop: check time triggers every 60 seconds."""
        last_expiry_sweep = _now_utc()
        while self._running:
            try:
                self._refresh_present_people()
                await self._check_time_triggers()

                # Run expiry sweep once per hour
                if (_now_utc() - last_expiry_sweep).total_seconds() >= 3600:
                    await self._sweep_expired()
                    last_expiry_sweep = _now_utc()

            except Exception:
                logger.exception("Error in scheduler loop")

            await asyncio.sleep(60)

    async def _on_person_identified(self, event: PersonIdentified) -> None:
        """Handle a person identification event from the perception pipeline."""
        name = event.person_name
        if not name:
            return

        self._person_last_seen[name] = _now_utc()
        self._present_people.add(name)

        # Check if any active triggers are waiting on this person
        await self._check_person_triggers(name)

    def _refresh_present_people(self) -> None:
        """Remove people who haven't been seen within the presence window."""
        now = _now_utc()
        stale = [
            name
            for name, last_seen in self._person_last_seen.items()
            if (now - last_seen) > self._person_presence_window
        ]
        for name in stale:
            self._present_people.discard(name)
            del self._person_last_seen[name]

    async def _check_time_triggers(self) -> None:
        """Check all active triggers for met time conditions."""
        triggers = await list_triggers(status="active")
        for trigger in triggers:
            if _is_expired(trigger):
                await update_trigger(trigger["id"], status="expired")
                logger.info(
                    "Trigger %s expired: %s",
                    trigger["id"],
                    trigger["description"],
                )
                continue

            if evaluate_trigger(trigger, self._present_people):
                await self._fire_trigger(trigger)

    async def _check_person_triggers(self, person_name: str) -> None:
        """Check active triggers waiting on a specific person."""
        triggers = await list_triggers(status="active")
        for trigger in triggers:
            if trigger.get("person") != person_name:
                continue
            if _is_expired(trigger):
                await update_trigger(trigger["id"], status="expired")
                continue
            if evaluate_trigger(trigger, self._present_people):
                await self._fire_trigger(trigger)

    async def _fire_trigger(self, trigger: dict[str, Any]) -> None:
        """Fire a trigger: update state, emit event, handle recurrence."""
        trigger_id = trigger["id"]
        is_recurring = trigger.get("cron") is not None
        now = _now_iso()

        logger.info(
            "Firing trigger %s: %s", trigger_id, trigger["description"]
        )

        if is_recurring:
            # Recurring: update last_fired, increment count, compute next fire_at
            cron_expr = CronExpr(trigger["cron"])
            next_fire = cron_expr.next_occurrence(_now_utc())
            await update_trigger(
                trigger_id,
                last_fired=now,
                fire_count=trigger["fire_count"] + 1,
                fire_at=next_fire.isoformat(),
            )
        else:
            # One-shot: mark as fired
            await update_trigger(
                trigger_id,
                status="fired",
                last_fired=now,
                fire_count=trigger["fire_count"] + 1,
            )

        # Emit event
        event = TriggerFired(
            trigger_id=trigger_id,
            description=trigger["description"],
            instructions=trigger["instructions"],
            person=trigger.get("person"),
            for_person=trigger.get("for_person"),
            todo_id=trigger.get("todo_id"),
            is_recurring=is_recurring,
        )
        bus = get_event_bus()
        await bus.publish(event)

    async def _sweep_expired(self) -> None:
        """Mark expired active triggers."""
        triggers = await list_triggers(status="active")
        count = 0
        for trigger in triggers:
            if _is_expired(trigger):
                await update_trigger(trigger["id"], status="expired")
                count += 1
        if count:
            logger.info("Expiry sweep: marked %d triggers as expired", count)
