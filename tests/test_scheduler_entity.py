"""Tests for entity trigger conditions (Home Assistant events bridge)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import aiosqlite
import pytest

from boxbot.core.events import EntityStateChanged, TriggerFired, get_event_bus
from boxbot.core.scheduler import (
    Scheduler,
    create_trigger,
    evaluate_entity_condition,
    evaluate_trigger,
    get_trigger,
    watched_entities,
)


@pytest.fixture(autouse=True)
def patch_scheduler_db(tmp_path):
    """Point the scheduler module's DB_PATH to a temp directory."""
    test_db = tmp_path / "scheduler" / "scheduler.db"
    with patch("boxbot.core.scheduler.DB_PATH", test_db):
        yield test_db


# ---------------------------------------------------------------------------
# Creation + validation
# ---------------------------------------------------------------------------


class TestEntityTriggerCreation:
    @pytest.mark.asyncio
    async def test_entity_state_defaults_to_on(self):
        tid = await create_trigger(
            description="Person at front door",
            instructions="Tell Jacob someone is at the door",
            entity="binary_sensor.front_door_person",
        )
        trigger = await get_trigger(tid)
        assert trigger["entity"] == "binary_sensor.front_door_person"
        assert trigger["entity_state"] == "on"

    @pytest.mark.asyncio
    async def test_explicit_entity_state(self):
        tid = await create_trigger(
            description="Garage door opened",
            instructions="Note it",
            entity="cover.garage_door",
            entity_state="open",
        )
        trigger = await get_trigger(tid)
        assert trigger["entity_state"] == "open"

    @pytest.mark.asyncio
    async def test_invalid_entity_id_rejected(self):
        with pytest.raises(ValueError, match="entity_id"):
            await create_trigger(
                description="x", instructions="y",
                entity="not a valid entity",
            )

    @pytest.mark.asyncio
    async def test_entity_state_without_entity_rejected(self):
        with pytest.raises(ValueError, match="requires entity"):
            await create_trigger(
                description="x", instructions="y",
                fire_after="30m", entity_state="on",
            )

    @pytest.mark.asyncio
    async def test_entity_only_trigger_gets_default_expiry(self):
        tid = await create_trigger(
            description="x", instructions="y",
            entity="binary_sensor.front_door_person",
        )
        trigger = await get_trigger(tid)
        assert trigger["expires"] is not None

    @pytest.mark.asyncio
    async def test_watched_entities_reflects_active_triggers(self):
        await create_trigger(
            description="a", instructions="b",
            entity="binary_sensor.front_door_person",
        )
        tid2 = await create_trigger(
            description="c", instructions="d",
            entity="binary_sensor.garage_person",
        )
        from boxbot.core.scheduler import cancel_trigger

        await cancel_trigger(tid2)
        watched = await watched_entities()
        assert watched == {"binary_sensor.front_door_person"}


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------


class TestEvaluateEntityCondition:
    def test_no_condition_is_vacuously_true(self):
        assert evaluate_entity_condition({"entity": None}, {}) is True

    def test_matching_state(self):
        trigger = {"entity": "binary_sensor.x", "entity_state": "on"}
        assert evaluate_entity_condition(
            trigger, {"binary_sensor.x": "on"}
        ) is True

    def test_mismatching_state(self):
        trigger = {"entity": "binary_sensor.x", "entity_state": "on"}
        assert evaluate_entity_condition(
            trigger, {"binary_sensor.x": "off"}
        ) is False

    def test_unseen_entity_is_false(self):
        trigger = {"entity": "binary_sensor.x", "entity_state": "on"}
        assert evaluate_entity_condition(trigger, {}) is False

    def test_missing_entity_state_defaults_to_on(self):
        trigger = {"entity": "binary_sensor.x", "entity_state": None}
        assert evaluate_entity_condition(
            trigger, {"binary_sensor.x": "on"}
        ) is True


# ---------------------------------------------------------------------------
# Scheduler event-driven firing
# ---------------------------------------------------------------------------


class TestSchedulerEntityTriggers:
    @pytest.mark.asyncio
    async def test_live_event_fires_trigger(self):
        tid = await create_trigger(
            description="Person at front door",
            instructions="Announce it",
            entity="binary_sensor.front_door_person",
        )
        sched = Scheduler()
        await sched._on_entity_state(EntityStateChanged(
            entity_id="binary_sensor.front_door_person",
            new_state="on", old_state="off",
        ))
        trigger = await get_trigger(tid)
        assert trigger["status"] == "fired"

    @pytest.mark.asyncio
    async def test_fired_event_carries_entity(self):
        await create_trigger(
            description="Person at front door",
            instructions="Announce it",
            entity="binary_sensor.front_door_person",
        )
        fired: list[TriggerFired] = []

        async def capture(event: TriggerFired) -> None:
            fired.append(event)

        get_event_bus().subscribe(TriggerFired, capture)
        sched = Scheduler()
        await sched._on_entity_state(EntityStateChanged(
            entity_id="binary_sensor.front_door_person",
            new_state="on", old_state="off",
        ))
        assert len(fired) == 1
        assert fired[0].entity == "binary_sensor.front_door_person"

    @pytest.mark.asyncio
    async def test_non_matching_state_does_not_fire(self):
        tid = await create_trigger(
            description="x", instructions="y",
            entity="binary_sensor.front_door_person",
        )
        sched = Scheduler()
        await sched._on_entity_state(EntityStateChanged(
            entity_id="binary_sensor.front_door_person",
            new_state="off", old_state="on",
        ))
        trigger = await get_trigger(tid)
        assert trigger["status"] == "active"

    @pytest.mark.asyncio
    async def test_snapshot_updates_map_but_never_fires(self):
        tid = await create_trigger(
            description="x", instructions="y",
            entity="binary_sensor.front_door_person",
        )
        sched = Scheduler()
        await sched._on_entity_state(EntityStateChanged(
            entity_id="binary_sensor.front_door_person",
            new_state="on", snapshot=True,
        ))
        trigger = await get_trigger(tid)
        assert trigger["status"] == "active"
        assert sched._entity_states["binary_sensor.front_door_person"] == "on"

    @pytest.mark.asyncio
    async def test_unknown_state_clears_map_entry(self):
        sched = Scheduler()
        sched._entity_states["binary_sensor.x"] = "on"
        await sched._on_entity_state(EntityStateChanged(
            entity_id="binary_sensor.x", new_state="unknown",
        ))
        assert "binary_sensor.x" not in sched._entity_states

    @pytest.mark.asyncio
    async def test_compound_time_and_entity(self):
        # Time already passed + entity event arriving → fires.
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        tid = await create_trigger(
            description="After 9, person at door",
            instructions="Announce",
            fire_at=past,
            entity="binary_sensor.front_door_person",
        )
        sched = Scheduler()
        await sched._on_entity_state(EntityStateChanged(
            entity_id="binary_sensor.front_door_person",
            new_state="on",
        ))
        assert (await get_trigger(tid))["status"] == "fired"

    @pytest.mark.asyncio
    async def test_compound_future_time_blocks_entity_fire(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        tid = await create_trigger(
            description="Later, person at door",
            instructions="Announce",
            fire_at=future,
            entity="binary_sensor.front_door_person",
        )
        sched = Scheduler()
        await sched._on_entity_state(EntityStateChanged(
            entity_id="binary_sensor.front_door_person",
            new_state="on",
        ))
        assert (await get_trigger(tid))["status"] == "active"

    @pytest.mark.asyncio
    async def test_entity_condition_blocks_time_only_fire(self):
        # Time passed but entity not in wanted state → loop scan must not fire.
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        trigger = {
            "status": "active",
            "fire_at": past,
            "person": None,
            "entity": "binary_sensor.x",
            "entity_state": "on",
        }
        assert evaluate_trigger(trigger, set(), {}) is False
        assert evaluate_trigger(trigger, set(), {"binary_sensor.x": "on"}) is True


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


_LEGACY_TRIGGERS_DDL = """\
CREATE TABLE triggers (
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
    fire_count INTEGER NOT NULL DEFAULT 0
)"""


class TestSchemaMigration:
    @pytest.mark.asyncio
    async def test_entity_columns_added_to_legacy_db(self, patch_scheduler_db):
        # Build a pre-entity-column database like the one live on the Pi.
        patch_scheduler_db.parent.mkdir(parents=True, exist_ok=True)
        db = await aiosqlite.connect(str(patch_scheduler_db))
        await db.execute(_LEGACY_TRIGGERS_DDL)
        await db.execute(
            "INSERT INTO triggers (id, description, instructions, created_at) "
            "VALUES ('t_old', 'legacy', 'legacy', '2026-01-01T00:00:00')"
        )
        await db.commit()
        await db.close()

        # Any scheduler call opens the DB and applies the migration.
        tid = await create_trigger(
            description="new", instructions="new",
            entity="binary_sensor.front_door_person",
        )
        assert (await get_trigger(tid))["entity_state"] == "on"
        legacy = await get_trigger("t_old")
        assert legacy["entity"] is None
