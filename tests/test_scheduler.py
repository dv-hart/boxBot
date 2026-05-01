"""Tests for boxbot.core.scheduler — triggers, todos, cron, duration parsing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from boxbot.core.scheduler import (
    CronExpr,
    Scheduler,
    cancel_todo,
    cancel_trigger,
    complete_todo,
    create_todo,
    create_trigger,
    evaluate_person_condition,
    evaluate_time_condition,
    evaluate_trigger,
    get_status_line,
    get_todo,
    get_trigger,
    list_todos,
    list_triggers,
    parse_duration,
    seed_from_config,
    update_todo,
    update_trigger,
)


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------


class TestParseDuration:
    """Test parse_duration() with valid and invalid inputs."""

    def test_parse_minutes(self):
        td = parse_duration("30m")
        assert td == timedelta(minutes=30)

    def test_parse_hours(self):
        td = parse_duration("2h")
        assert td == timedelta(hours=2)

    def test_parse_days(self):
        td = parse_duration("1d")
        assert td == timedelta(days=1)

    def test_case_insensitive(self):
        assert parse_duration("30M") == timedelta(minutes=30)
        assert parse_duration("2H") == timedelta(hours=2)

    def test_rejects_exceeding_24h(self):
        with pytest.raises(ValueError, match="exceeds maximum"):
            parse_duration("25h")

    def test_rejects_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("abc")

    def test_rejects_missing_unit(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("30")

    def test_whitespace_is_stripped(self):
        td = parse_duration("  15m  ")
        assert td == timedelta(minutes=15)


# ---------------------------------------------------------------------------
# Cron expression
# ---------------------------------------------------------------------------


class TestCronExpr:
    """Test the minimal CronExpr parser and matcher."""

    def test_matches_exact_time(self):
        cron = CronExpr("0 7 * * *")
        dt = datetime(2025, 6, 15, 7, 0, tzinfo=timezone.utc)
        assert cron.matches(dt) is True

    def test_does_not_match_wrong_minute(self):
        cron = CronExpr("0 7 * * *")
        dt = datetime(2025, 6, 15, 7, 30, tzinfo=timezone.utc)
        assert cron.matches(dt) is False

    def test_wildcard_matches_any(self):
        cron = CronExpr("* * * * *")
        dt = datetime(2025, 1, 1, 12, 30, tzinfo=timezone.utc)
        assert cron.matches(dt) is True

    def test_range_field(self):
        cron = CronExpr("0 9-17 * * *")
        assert cron.matches(datetime(2025, 6, 15, 9, 0, tzinfo=timezone.utc))
        assert cron.matches(datetime(2025, 6, 15, 17, 0, tzinfo=timezone.utc))
        assert not cron.matches(datetime(2025, 6, 15, 18, 0, tzinfo=timezone.utc))

    def test_step_field(self):
        cron = CronExpr("*/15 * * * *")
        assert cron.matches(datetime(2025, 6, 15, 10, 0, tzinfo=timezone.utc))
        assert cron.matches(datetime(2025, 6, 15, 10, 15, tzinfo=timezone.utc))
        assert not cron.matches(datetime(2025, 6, 15, 10, 7, tzinfo=timezone.utc))

    def test_list_field(self):
        cron = CronExpr("0 7,12,20 * * *")
        assert cron.matches(datetime(2025, 6, 15, 7, 0, tzinfo=timezone.utc))
        assert cron.matches(datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc))
        assert not cron.matches(datetime(2025, 6, 15, 8, 0, tzinfo=timezone.utc))

    def test_invalid_field_count_raises(self):
        with pytest.raises(ValueError, match="5 fields"):
            CronExpr("0 7 *")

    def test_next_occurrence_finds_future_match(self):
        cron = CronExpr("0 7 * * *")
        after = datetime(2025, 6, 15, 8, 0, tzinfo=timezone.utc)
        nxt = cron.next_occurrence(after)
        assert nxt.hour == 7
        assert nxt.minute == 0
        assert nxt > after


# ---------------------------------------------------------------------------
# Trigger CRUD (requires monkeypatched DB_PATH)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_scheduler_db(tmp_path):
    """Point the scheduler module's DB_PATH to a temp directory."""
    test_db = tmp_path / "scheduler" / "scheduler.db"
    with patch("boxbot.core.scheduler.DB_PATH", test_db):
        yield test_db


class TestTriggerCRUD:
    """Test trigger creation, retrieval, listing, and status updates."""

    @pytest.mark.asyncio
    async def test_create_trigger_returns_prefixed_id(self):
        tid = await create_trigger(
            description="Test trigger",
            instructions="Do something",
            fire_after="30m",
        )
        assert tid.startswith("t_")

    @pytest.mark.asyncio
    async def test_get_trigger_returns_data(self):
        tid = await create_trigger(
            description="Fetch trigger",
            instructions="Fetch instructions",
        )
        trigger = await get_trigger(tid)
        assert trigger is not None
        assert trigger["description"] == "Fetch trigger"
        assert trigger["status"] == "active"

    @pytest.mark.asyncio
    async def test_get_trigger_nonexistent_returns_none(self):
        result = await get_trigger("t_nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_triggers_filters_by_status(self):
        tid = await create_trigger(
            description="Active trigger", instructions="Do it"
        )
        await cancel_trigger(tid)

        active = await list_triggers(status="active")
        cancelled = await list_triggers(status="cancelled")

        active_ids = {t["id"] for t in active}
        cancelled_ids = {t["id"] for t in cancelled}

        assert tid not in active_ids
        assert tid in cancelled_ids

    @pytest.mark.asyncio
    async def test_fire_after_and_cron_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            await create_trigger(
                description="Bad",
                instructions="Conflict",
                fire_after="30m",
                cron="0 7 * * *",
            )

    @pytest.mark.asyncio
    async def test_cancel_trigger_changes_status(self):
        tid = await create_trigger(
            description="Cancel me", instructions="..."
        )
        result = await cancel_trigger(tid)
        assert result is True
        trigger = await get_trigger(tid)
        assert trigger["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_update_trigger_fields(self):
        tid = await create_trigger(
            description="Original", instructions="..."
        )
        updated = await update_trigger(tid, description="Updated")
        assert updated is True
        trigger = await get_trigger(tid)
        assert trigger["description"] == "Updated"


# ---------------------------------------------------------------------------
# Todo CRUD
# ---------------------------------------------------------------------------


class TestTodoCRUD:
    """Test to-do item creation, retrieval, completion, and cancellation."""

    @pytest.mark.asyncio
    async def test_create_todo_returns_prefixed_id(self):
        did = await create_todo(description="Buy groceries")
        assert did.startswith("d_")

    @pytest.mark.asyncio
    async def test_get_todo_returns_data(self):
        did = await create_todo(
            description="Test todo", notes="Detailed notes here"
        )
        todo = await get_todo(did)
        assert todo is not None
        assert todo["description"] == "Test todo"
        assert todo["notes"] == "Detailed notes here"
        assert todo["status"] == "pending"

    @pytest.mark.asyncio
    async def test_complete_todo_sets_status_and_timestamp(self):
        did = await create_todo(description="Complete me")
        result = await complete_todo(did)
        assert result is True
        todo = await get_todo(did)
        assert todo["status"] == "completed"
        assert todo["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_cancel_todo(self):
        did = await create_todo(description="Cancel me")
        await cancel_todo(did)
        todo = await get_todo(did)
        assert todo["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_list_todos_filters_by_status(self):
        d1 = await create_todo(description="Pending")
        d2 = await create_todo(description="Done")
        await complete_todo(d2)

        pending = await list_todos(status="pending")
        completed = await list_todos(status="completed")

        pending_ids = {t["id"] for t in pending}
        completed_ids = {t["id"] for t in completed}

        assert d1 in pending_ids
        assert d2 in completed_ids

    @pytest.mark.asyncio
    async def test_list_todos_filters_by_for_person(self):
        d1 = await create_todo(description="For Jacob", for_person="Jacob")
        d2 = await create_todo(description="For Alice", for_person="Alice")

        jacob_todos = await list_todos(for_person="Jacob")
        assert any(t["id"] == d1 for t in jacob_todos)
        assert not any(t["id"] == d2 for t in jacob_todos)


# ---------------------------------------------------------------------------
# Status line
# ---------------------------------------------------------------------------


class TestStatusLine:
    """Test the compact status line generation."""

    @pytest.mark.asyncio
    async def test_status_line_format(self):
        await create_trigger(description="T1", instructions="...")
        await create_todo(description="D1")
        line = await get_status_line()
        assert "[To-do:" in line
        assert "Triggers:" in line

    @pytest.mark.asyncio
    async def test_status_line_counts_correctly(self):
        await create_trigger(description="T1", instructions="...")
        await create_trigger(description="T2", instructions="...")
        await create_todo(description="D1")
        line = await get_status_line()
        assert "1 items" in line
        assert "2 active" in line


# ---------------------------------------------------------------------------
# Trigger condition evaluation
# ---------------------------------------------------------------------------


class TestConditionEvaluation:
    """Test trigger condition evaluation functions."""

    def test_evaluate_time_condition_no_fire_at_is_true(self):
        trigger = {"fire_at": None}
        assert evaluate_time_condition(trigger) is True

    def test_evaluate_time_condition_past_fire_at_is_true(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        trigger = {"fire_at": past}
        assert evaluate_time_condition(trigger) is True

    def test_evaluate_time_condition_future_fire_at_is_false(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        trigger = {"fire_at": future}
        assert evaluate_time_condition(trigger) is False

    def test_evaluate_person_condition_no_person_is_true(self):
        trigger = {"person": None}
        assert evaluate_person_condition(trigger, set()) is True

    def test_evaluate_person_condition_present_is_true(self):
        trigger = {"person": "Jacob"}
        assert evaluate_person_condition(trigger, {"Jacob"}) is True

    def test_evaluate_person_condition_absent_is_false(self):
        trigger = {"person": "Jacob"}
        assert evaluate_person_condition(trigger, {"Alice"}) is False

    def test_evaluate_trigger_inactive_is_false(self):
        trigger = {"status": "cancelled", "fire_at": None, "person": None}
        assert evaluate_trigger(trigger) is False

    def test_evaluate_trigger_active_all_met(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        trigger = {"status": "active", "fire_at": past, "person": "Jacob"}
        assert evaluate_trigger(trigger, {"Jacob"}) is True

    def test_evaluate_trigger_active_person_not_present(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        trigger = {"status": "active", "fire_at": past, "person": "Jacob"}
        assert evaluate_trigger(trigger, {"Alice"}) is False


# ---------------------------------------------------------------------------
# Seed from config
# ---------------------------------------------------------------------------


class TestSeedFromConfig:
    """Test seed_from_config() — seeding triggers from config on first boot."""

    @pytest.mark.asyncio
    async def test_seeds_three_default_triggers(self, mock_config):
        await seed_from_config(mock_config)
        triggers = await list_triggers()
        # Default config has 3 wake cycles + 1 dream-cycle trigger
        assert len(triggers) >= 3

    @pytest.mark.asyncio
    async def test_skips_seeding_when_triggers_exist(self, mock_config):
        await create_trigger(description="Pre-existing", instructions="...")
        await seed_from_config(mock_config)
        triggers = await list_triggers()
        # Wake-cycle seed is skipped (DB not empty), but the dream-cycle
        # trigger is still added if missing — so we expect 1 pre-existing
        # + 1 dream-cycle = 2.
        assert len(triggers) == 2
        descriptions = [t["description"] for t in triggers]
        assert "Pre-existing" in descriptions
        assert any(d.startswith("[dream-cycle]") for d in descriptions)
