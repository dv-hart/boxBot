"""Tests for the scheduler prefetch lookahead (_check_upcoming_triggers)."""

from __future__ import annotations

from types import SimpleNamespace

import boxbot.core.scheduler as sched_mod
from boxbot.core.events import TriggerUpcoming
from boxbot.core.scheduler import Scheduler, _now_utc


class _FakeBus:
    def __init__(self):
        self.published = []

    async def publish(self, event):
        self.published.append(event)


def _prefetch_cfg(monkeypatch, *, enabled=True, lookahead=5,
                  channels=("trigger",)):
    cfg = SimpleNamespace(
        prefetch=SimpleNamespace(
            enabled=enabled, lookahead_minutes=lookahead, channels=list(channels),
        )
    )
    monkeypatch.setattr(sched_mod, "_try_get_config", lambda: cfg)


def _install_triggers(monkeypatch, triggers):
    async def _list(status=None):
        return triggers

    monkeypatch.setattr(sched_mod, "list_triggers", _list)


def _iso(dt):
    return dt.isoformat()


class TestUpcoming:
    async def test_emits_for_time_trigger_within_window(self, monkeypatch):
        from datetime import timedelta

        now = _now_utc()
        _prefetch_cfg(monkeypatch)
        _install_triggers(monkeypatch, [
            {"id": "soon", "description": "remind", "instructions": "do it",
             "person": None, "for_person": "Jacob", "todo_id": "d1",
             "cron": None, "fire_at": _iso(now + timedelta(minutes=2))},
        ])
        bus = _FakeBus()
        monkeypatch.setattr(sched_mod, "get_event_bus", lambda: bus)

        s = Scheduler()
        await s._check_upcoming_triggers()

        assert len(bus.published) == 1
        ev = bus.published[0]
        assert isinstance(ev, TriggerUpcoming)
        assert ev.trigger_id == "soon"
        assert ev.for_person == "Jacob"
        assert ev.todo_id == "d1"
        # Signaled set prevents a second emit on the next tick.
        await s._check_upcoming_triggers()
        assert len(bus.published) == 1

    async def test_skips_far_future_and_person_only(self, monkeypatch):
        from datetime import timedelta

        now = _now_utc()
        _prefetch_cfg(monkeypatch, lookahead=5)
        _install_triggers(monkeypatch, [
            {"id": "far", "description": "d", "instructions": "i",
             "person": None, "for_person": None, "todo_id": None,
             "cron": None, "fire_at": _iso(now + timedelta(minutes=30))},
            {"id": "person-only", "description": "d", "instructions": "i",
             "person": "Jacob", "for_person": None, "todo_id": None,
             "cron": None, "fire_at": None},
        ])
        bus = _FakeBus()
        monkeypatch.setattr(sched_mod, "get_event_bus", lambda: bus)

        s = Scheduler()
        await s._check_upcoming_triggers()
        assert bus.published == []

    async def test_noop_when_prefetch_disabled(self, monkeypatch):
        from datetime import timedelta

        now = _now_utc()
        _prefetch_cfg(monkeypatch, enabled=False)
        _install_triggers(monkeypatch, [
            {"id": "soon", "description": "d", "instructions": "i",
             "person": None, "for_person": None, "todo_id": None,
             "cron": None, "fire_at": _iso(now + timedelta(minutes=2))},
        ])
        bus = _FakeBus()
        monkeypatch.setattr(sched_mod, "get_event_bus", lambda: bus)

        s = Scheduler()
        await s._check_upcoming_triggers()
        assert bus.published == []

    async def test_noop_when_trigger_channel_excluded(self, monkeypatch):
        from datetime import timedelta

        now = _now_utc()
        _prefetch_cfg(monkeypatch, channels=("whatsapp", "signal"))
        _install_triggers(monkeypatch, [
            {"id": "soon", "description": "d", "instructions": "i",
             "person": None, "for_person": None, "todo_id": None,
             "cron": None, "fire_at": _iso(now + timedelta(minutes=2))},
        ])
        bus = _FakeBus()
        monkeypatch.setattr(sched_mod, "get_event_bus", lambda: bus)

        s = Scheduler()
        await s._check_upcoming_triggers()
        assert bus.published == []
