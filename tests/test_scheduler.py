"""Tests for boxbot.core.scheduler — triggers, todos, cron, duration parsing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from boxbot.core.events import PersonDetected, PersonIdentified
from boxbot.core.scheduler import (
    ANY_PERSON,
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

    def test_evaluate_person_condition_wildcard_with_anyone_present_is_true(self):
        # "*" means any person — met as long as someone is present, regardless
        # of who, including the unidentified-presence sentinel.
        trigger = {"person": "*"}
        assert evaluate_person_condition(trigger, {"Alice"}) is True
        assert evaluate_person_condition(trigger, {"*"}) is True

    def test_evaluate_person_condition_wildcard_with_nobody_present_is_false(self):
        trigger = {"person": "*"}
        assert evaluate_person_condition(trigger, set()) is False

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

    @pytest.mark.asyncio
    async def test_resyncs_dream_cron_when_config_changes(self, mock_config):
        # First seed with the default config cron.
        await seed_from_config(mock_config)
        triggers = await list_triggers()
        dream = next(
            t for t in triggers if t["description"].startswith("[dream-cycle]")
        )
        original_cron = dream["cron"]
        original_fire_at = dream["fire_at"]

        # Change config and re-seed; the existing config-sourced row
        # should be updated, not duplicated.
        mock_config.memory.dream_cron = "30 11 * * *"
        await seed_from_config(mock_config)

        triggers = await list_triggers()
        dream_rows = [
            t for t in triggers if t["description"].startswith("[dream-cycle]")
        ]
        assert len(dream_rows) == 1
        assert dream_rows[0]["cron"] == "30 11 * * *"
        assert dream_rows[0]["cron"] != original_cron
        assert dream_rows[0]["fire_at"] != original_fire_at

    @pytest.mark.asyncio
    async def test_resync_leaves_user_modified_dream_trigger_alone(
        self, mock_config,
    ):
        # Simulate a user/agent override: same description prefix, but
        # source != 'config'.
        await create_trigger(
            description="[dream-cycle] custom",
            instructions="user override",
            cron="0 4 * * *",
            source="agent",
        )
        mock_config.memory.dream_cron = "0 10 * * *"
        await seed_from_config(mock_config)

        triggers = await list_triggers()
        dream_rows = [
            t for t in triggers if t["description"].startswith("[dream-cycle]")
        ]
        # The user-owned row stays at 0 4 * * *; nothing else gets seeded
        # because the dream_count check sees ≥1 [dream-cycle] row.
        assert len(dream_rows) == 1
        assert dream_rows[0]["cron"] == "0 4 * * *"
        assert dream_rows[0]["source"] == "agent"


# ---------------------------------------------------------------------------
# Person-trigger firing (named + wildcard)
# ---------------------------------------------------------------------------


class TestPersonTriggerFiring:
    """Test the scheduler's person-presence trigger firing paths."""

    @pytest.mark.asyncio
    async def test_wildcard_trigger_fires_on_person_detected(self):
        # "Greet the next person, no ID required" — fires on a raw detection,
        # without waiting for visual identification.
        tid = await create_trigger(
            description="Greet next person",
            instructions="Say hi",
            person=ANY_PERSON,
        )
        sched = Scheduler()
        await sched._on_person_detected(
            PersonDetected(person_ref="A", confidence=0.9)
        )
        trigger = await get_trigger(tid)
        assert trigger["status"] == "fired"
        assert trigger["fire_count"] == 1

    @pytest.mark.asyncio
    async def test_named_trigger_not_fired_by_unidentified_detection(self):
        # A named trigger must wait for identification — a raw detection
        # (no name) must not fire it.
        tid = await create_trigger(
            description="Greet Jacob",
            instructions="Say hi Jacob",
            person="Jacob",
        )
        sched = Scheduler()
        await sched._on_person_detected(
            PersonDetected(person_ref="A", confidence=0.9)
        )
        trigger = await get_trigger(tid)
        assert trigger["status"] == "active"
        assert trigger["fire_count"] == 0

    @pytest.mark.asyncio
    async def test_named_trigger_fires_on_person_identified(self):
        tid = await create_trigger(
            description="Greet Jacob",
            instructions="Say hi Jacob",
            person="Jacob",
        )
        sched = Scheduler()
        await sched._on_person_identified(
            PersonIdentified(person_id="p1", person_name="Jacob", confidence=0.9)
        )
        trigger = await get_trigger(tid)
        assert trigger["status"] == "fired"

    @pytest.mark.asyncio
    async def test_wildcard_trigger_also_fires_for_identified_person(self):
        # An identified person is also "any person" — a wildcard trigger
        # should fire on identification too.
        tid = await create_trigger(
            description="Greet next person",
            instructions="Say hi",
            person=ANY_PERSON,
        )
        sched = Scheduler()
        await sched._on_person_identified(
            PersonIdentified(person_id="p1", person_name="Jacob", confidence=0.9)
        )
        trigger = await get_trigger(tid)
        assert trigger["status"] == "fired"

    @pytest.mark.asyncio
    async def test_person_detected_is_throttled_after_rising_edge(self):
        # First detection (rising edge) scans; a second within the throttle
        # window is suppressed so per-frame detections don't hammer the DB.
        sched = Scheduler()
        sched._check_person_triggers = AsyncMock()
        await sched._on_person_detected(PersonDetected(person_ref="A"))
        await sched._on_person_detected(PersonDetected(person_ref="A"))
        assert sched._check_person_triggers.await_count == 1


# ---------------------------------------------------------------------------
# Re-arming condition triggers (rearm_after_s)
# ---------------------------------------------------------------------------


class TestRearmingTriggers:
    """rearm_after_s: "whenever X", not "next time X"."""

    def _entity_on(self, entity="binary_sensor.front_door_person"):
        from boxbot.core.events import EntityStateChanged

        return EntityStateChanged(
            entity_id=entity, new_state="on", old_state="off",
            friendly_name="Front Door person", snapshot=False,
        )

    def _entity_off(self, entity="binary_sensor.front_door_person"):
        from boxbot.core.events import EntityStateChanged

        return EntityStateChanged(
            entity_id=entity, new_state="off", old_state="on",
            friendly_name="Front Door person", snapshot=False,
        )

    async def _rearm_trigger(self, **kwargs):
        return await create_trigger(
            description="Front door person alert",
            instructions="Text Jacob",
            entity="binary_sensor.front_door_person",
            rearm_after_s=0,
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_stays_active_and_refires_after_cooldown(self):
        tid = await self._rearm_trigger()
        sched = Scheduler()
        await sched._on_entity_state(self._entity_on())
        trigger = await get_trigger(tid)
        assert trigger["status"] == "active"
        assert trigger["fire_count"] == 1

        # Age the first firing past the refractory floor, then a fresh
        # off->on edge must fire again.
        old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        await update_trigger(tid, last_fired=old)
        await sched._on_entity_state(self._entity_off())
        await sched._on_entity_state(self._entity_on())
        trigger = await get_trigger(tid)
        assert trigger["status"] == "active"
        assert trigger["fire_count"] == 2

    @pytest.mark.asyncio
    async def test_refractory_floor_blocks_immediate_refire(self):
        tid = await self._rearm_trigger()
        sched = Scheduler()
        await sched._on_entity_state(self._entity_on())
        await sched._on_entity_state(self._entity_off())
        await sched._on_entity_state(self._entity_on())
        trigger = await get_trigger(tid)
        assert trigger["fire_count"] == 1

    @pytest.mark.asyncio
    async def test_time_scan_does_not_double_fire_while_condition_holds(self):
        # The 60s scan evaluates entity conditions against the live
        # mirror; while the momentary "on" is still held it must not
        # re-fire an edge that already fired.
        tid = await self._rearm_trigger()
        sched = Scheduler()
        await sched._on_entity_state(self._entity_on())
        await sched._check_time_triggers()
        trigger = await get_trigger(tid)
        assert trigger["fire_count"] == 1

    @pytest.mark.asyncio
    async def test_cooldown_longer_than_floor_is_honoured(self):
        tid = await create_trigger(
            description="Front door person alert",
            instructions="Text Jacob",
            entity="binary_sensor.front_door_person",
            rearm_after_s=3600,
        )
        sched = Scheduler()
        await sched._on_entity_state(self._entity_on())
        # Aged past the floor but inside the requested cooldown: no refire.
        old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        await update_trigger(tid, last_fired=old)
        await sched._on_entity_state(self._entity_off())
        await sched._on_entity_state(self._entity_on())
        trigger = await get_trigger(tid)
        assert trigger["fire_count"] == 1

    @pytest.mark.asyncio
    async def test_fired_event_reports_recurring(self):
        from boxbot.core.events import TriggerFired, get_event_bus

        received = []

        async def handler(event):
            received.append(event)

        bus = get_event_bus()
        bus.subscribe(TriggerFired, handler)
        try:
            await self._rearm_trigger()
            sched = Scheduler()
            await sched._on_entity_state(self._entity_on())
        finally:
            bus.unsubscribe(TriggerFired, handler)
        assert len(received) == 1
        assert received[0].is_recurring is True

    @pytest.mark.asyncio
    async def test_person_rearm_is_edge_triggered(self):
        """A person who stays in the room fires a "whenever Jacob" trigger
        once, not every presence scan; leaving and coming back fires it
        again."""
        tid = await create_trigger(
            description="Greet Jacob whenever he comes in",
            instructions="Say hi",
            person="Jacob",
            rearm_after_s=0,
        )
        sched = Scheduler()
        await sched._on_person_identified(PersonIdentified(person_name="Jacob"))
        assert (await get_trigger(tid))["fire_count"] == 1

        # Still present: repeated identifications and time scans must not
        # refire even once the refractory floor has passed.
        old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        await update_trigger(tid, last_fired=old)
        await sched._on_person_identified(PersonIdentified(person_name="Jacob"))
        await sched._check_time_triggers()
        assert (await get_trigger(tid))["fire_count"] == 1

        # Leaves (presence window lapses) → scan reads false → latch
        # released; the next arrival is a fresh edge.
        sched._present_people.clear()
        sched._person_last_seen.clear()
        await sched._check_time_triggers()
        await sched._on_person_identified(PersonIdentified(person_name="Jacob"))
        assert (await get_trigger(tid))["fire_count"] == 2

    @pytest.mark.asyncio
    async def test_entity_rearm_held_on_does_not_refire_on_scan(self):
        """A sensor left "on" (door propped open) fires once per on-period."""
        tid = await self._rearm_trigger()
        sched = Scheduler()
        await sched._on_entity_state(self._entity_on())
        old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        await update_trigger(tid, last_fired=old)
        await sched._check_time_triggers()
        await sched._check_time_triggers()
        assert (await get_trigger(tid))["fire_count"] == 1

    @pytest.mark.asyncio
    async def test_requires_person_or_entity_condition(self):
        with pytest.raises(ValueError, match="person or entity"):
            await create_trigger(
                description="x", instructions="y",
                fire_at="2027-01-01T00:00:00", rearm_after_s=60,
            )

    @pytest.mark.asyncio
    async def test_rejected_with_cron(self):
        with pytest.raises(ValueError, match="mutually exclusive with cron"):
            await create_trigger(
                description="x", instructions="y",
                cron="0 8 * * *", person="Jacob", rearm_after_s=60,
            )

    @pytest.mark.asyncio
    async def test_rejects_negative_or_non_int(self):
        with pytest.raises(ValueError, match="non-negative integer"):
            await create_trigger(
                description="x", instructions="y",
                person="Jacob", rearm_after_s=-1,
            )

    @pytest.mark.asyncio
    async def test_default_expiry_extends_to_30_days(self):
        tid = await self._rearm_trigger()
        trigger = await get_trigger(tid)
        expires = datetime.fromisoformat(trigger["expires"])
        delta = expires - datetime.now(timezone.utc)
        assert timedelta(days=29) < delta <= timedelta(days=30)


# ---------------------------------------------------------------------------
# Stale cron re-anchor (schedule.catch_up_grace_seconds)
# ---------------------------------------------------------------------------


class TestStaleCronReanchor:
    """A cron slot missed by more than the grace window is skipped, not
    replayed; fresh slots and one-shot fire_at triggers are untouched."""

    async def _cron_trigger(self, fire_at: datetime) -> str:
        tid = await create_trigger(
            description="Morning brief",
            instructions="Brief the house",
            cron="0 8 * * *",
        )
        await update_trigger(tid, fire_at=fire_at.isoformat())
        return tid

    @pytest.mark.asyncio
    async def test_overdue_cron_is_reanchored_without_firing(self, caplog):
        stale = datetime.now(timezone.utc) - timedelta(hours=3)
        tid = await self._cron_trigger(stale)
        sched = Scheduler()
        with caplog.at_level("WARNING"):
            await sched._check_time_triggers()
        trigger = await get_trigger(tid)
        assert trigger["fire_count"] == 0
        assert trigger["status"] == "active"
        assert datetime.fromisoformat(trigger["fire_at"]) > datetime.now(timezone.utc)
        assert "re-anchored" in caplog.text

    @pytest.mark.asyncio
    async def test_cron_inside_grace_fires_normally(self):
        recent = datetime.now(timezone.utc) - timedelta(seconds=30)
        tid = await self._cron_trigger(recent)
        sched = Scheduler()
        await sched._check_time_triggers()
        trigger = await get_trigger(tid)
        assert trigger["fire_count"] == 1
        assert trigger["status"] == "active"

    @pytest.mark.asyncio
    async def test_overdue_one_shot_still_fires(self):
        stale = datetime.now(timezone.utc) - timedelta(hours=3)
        tid = await create_trigger(
            description="Remind Jacob",
            instructions="Remind",
            fire_at=stale.isoformat(),
        )
        sched = Scheduler()
        await sched._check_time_triggers()
        trigger = await get_trigger(tid)
        assert trigger["fire_count"] == 1
        assert trigger["status"] == "fired"

    @pytest.mark.asyncio
    async def test_overdue_compound_cron_is_not_reanchored(self):
        """A cron gated on presence is event-driven; leave it alone."""
        stale = datetime.now(timezone.utc) - timedelta(hours=3)
        tid = await create_trigger(
            description="Evening check-in when Jacob is home",
            instructions="Check in",
            cron="0 18 * * *",
            person="Jacob",
        )
        await update_trigger(tid, fire_at=stale.isoformat())
        sched = Scheduler()
        await sched._check_time_triggers()
        trigger = await get_trigger(tid)
        assert trigger["fire_at"] == stale.isoformat()
        assert trigger["fire_count"] == 0
