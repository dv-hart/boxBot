"""Tests for the live wiring of display data sources to backends."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from boxbot.core.events import (
    AgentSpeaking,
    AgentSpeakingDone,
    ConversationEnded,
    ConversationStarted,
    TranscriptReady,
    WakeWordHeard,
    get_event_bus,
)
from boxbot.displays.data_sources import (
    AgentStatusSource,
    IntegrationSource,
    MemoryQuerySource,
    TasksSource,
    create_source,
)


@pytest.fixture(autouse=True)
def _reset_event_bus():
    """Ensure each test starts with a clean event bus."""
    get_event_bus().clear()
    yield
    get_event_bus().clear()


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class TestTasksSourceWiring:
    async def test_returns_pending_todos_from_scheduler(self):
        async def fake_list(*, status=None):
            assert status == "pending"
            return [
                {"id": "d_1", "description": "Buy milk", "due_date": "2026-04-25",
                 "for_person": None, "status": "pending"},
                {"id": "d_2", "description": "Call mom", "due_date": None,
                 "for_person": "Jacob", "status": "pending"},
            ]

        with patch("boxbot.core.scheduler.list_todos", new=fake_list):
            data = await TasksSource().fetch()

        assert data["count"] == 2
        assert data["items"][0]["description"] == "Buy milk"
        assert data["items"][1]["for_person"] == "Jacob"

    async def test_returns_empty_on_failure(self):
        async def boom(*, status=None):
            raise RuntimeError("DB unavailable")

        with patch("boxbot.core.scheduler.list_todos", new=boom):
            data = await TasksSource().fetch()

        assert data == {"items": [], "count": 0}


# ---------------------------------------------------------------------------
# Memory query
# ---------------------------------------------------------------------------


def _candidate(i: int, *, created_at: str | None = None,
               summary: str | None = None):
    """Build a SearchCandidate like hybrid_search returns."""
    from boxbot.memory.search import SearchCandidate

    return SearchCandidate(
        id=f"mem_{i}",
        source="memory",
        type="household",
        person=None,
        content=f"Full content of memory {i}",
        summary=summary if summary is not None else f"Memory {i}",
        combined_score=1.0 - i * 0.1,
        metadata={"created_at": created_at or "2026-06-07T00:00:00"},
    )


class TestMemoryQuerySourceWiring:
    """memory_query goes straight to hybrid_search — no reranking, no
    model calls — and returns the display-ready
    ``{results: [{text, type, age}], count, query}`` shape."""

    async def test_returns_shaped_results(self):
        calls = {}

        async def fake_hybrid(store, query, **kwargs):
            calls["query"] = query
            calls["kwargs"] = kwargs
            return [_candidate(0), _candidate(1)]

        async def fake_store():
            return object()

        with patch("boxbot.memory.search.hybrid_search", new=fake_hybrid), \
             patch("boxbot.displays.data_sources._get_memory_store",
                   new=fake_store):
            src = MemoryQuerySource(
                "recent", {"query": "kitchen renovation"},
            )
            data = await src.fetch()

        assert calls["query"] == "kitchen renovation"
        # Cost constraint: facts only, hybrid scoring only.
        assert calls["kwargs"]["include_conversations"] is False
        assert data["query"] == "kitchen renovation"
        assert data["count"] == 2
        assert data["results"][0] == {
            "text": "Memory 0", "type": "household",
            "age": data["results"][0]["age"],
        }
        assert all(set(r) == {"text", "type", "age"}
                   for r in data["results"])
        # created_at 2026-06-07 vs frozen "now" — age is a short string.
        assert data["results"][0]["age"]

    async def test_text_falls_back_to_content(self):
        async def fake_hybrid(store, query, **kwargs):
            return [_candidate(0, summary="")]

        async def fake_store():
            return object()

        with patch("boxbot.memory.search.hybrid_search", new=fake_hybrid), \
             patch("boxbot.displays.data_sources._get_memory_store",
                   new=fake_store):
            data = await MemoryQuerySource("m", {"query": "x"}).fetch()

        assert data["results"][0]["text"] == "Full content of memory 0"

    async def test_empty_query_returns_empty_without_backend_call(self):
        async def explode():
            raise AssertionError("backend must not be touched")

        with patch("boxbot.displays.data_sources._get_memory_store",
                   new=explode):
            src = MemoryQuerySource("recent", {"query": "   "})
            data = await src.fetch()

        assert data == {"results": [], "count": 0, "query": ""}

    async def test_limit_respected_and_passed_to_backend(self):
        captured = {}

        async def fake_hybrid(store, query, **kwargs):
            captured.update(kwargs)
            return [_candidate(i) for i in range(10)]

        async def fake_store():
            return object()

        with patch("boxbot.memory.search.hybrid_search", new=fake_hybrid), \
             patch("boxbot.displays.data_sources._get_memory_store",
                   new=fake_store):
            data = await MemoryQuerySource(
                "recent", {"query": "x", "limit": 3},
            ).fetch()

        assert data["count"] == 3
        assert len(data["results"]) == 3
        assert captured["memory_limit"] == 3

    async def test_default_limit_is_five(self):
        async def fake_hybrid(store, query, **kwargs):
            return [_candidate(i) for i in range(10)]

        async def fake_store():
            return object()

        with patch("boxbot.memory.search.hybrid_search", new=fake_hybrid), \
             patch("boxbot.displays.data_sources._get_memory_store",
                   new=fake_store):
            data = await MemoryQuerySource("recent", {"query": "x"}).fetch()

        assert data["count"] == 5

    async def test_backend_error_keeps_cache_and_sets_fetch_error(self):
        """do_fetch semantics: a failing fetch logs a warning, keeps the
        previously cached data, and sets _fetch_error."""
        async def fake_store():
            return object()

        async def good_hybrid(store, query, **kwargs):
            return [_candidate(0)]

        async def bad_hybrid(store, query, **kwargs):
            raise RuntimeError("memory DB unavailable")

        src = MemoryQuerySource("recent", {"query": "x"})

        with patch("boxbot.displays.data_sources._get_memory_store",
                   new=fake_store):
            with patch("boxbot.memory.search.hybrid_search", new=good_hybrid):
                first = await src.do_fetch()
            assert first["count"] == 1
            assert src._fetch_error is None

            with patch("boxbot.memory.search.hybrid_search", new=bad_hybrid):
                second = await src.do_fetch()

        assert second == first          # cached data survives
        assert src.cached_data == first
        assert src._fetch_error is not None
        assert "memory DB unavailable" in src._fetch_error

    async def test_backend_error_with_no_cache_returns_empty(self):
        async def fake_store():
            raise RuntimeError("no store")

        src = MemoryQuerySource("recent", {"query": "x"})
        with patch("boxbot.displays.data_sources._get_memory_store",
                   new=fake_store):
            data = await src.do_fetch()

        assert data == {}               # never crashes the refresh loop
        assert src._fetch_error is not None

    def test_refresh_default_300_even_when_manager_passes_none(self):
        assert MemoryQuerySource("m", {"query": "x"}).refresh_interval == 300
        # The display manager passes refresh=None when the spec omits it.
        assert MemoryQuerySource(
            "m", {"query": "x", "refresh": None},
        ).refresh_interval == 300
        assert MemoryQuerySource(
            "m", {"query": "x", "refresh": 600},
        ).refresh_interval == 600

    def test_create_source_routes_memory_query_type(self):
        src = create_source("recent", "memory_query", {"query": "x"})
        assert isinstance(src, MemoryQuerySource)


class TestFormatAge:
    """_format_age renders short age strings for display bindings."""

    def _age(self, **delta_kwargs):
        from datetime import datetime, timedelta, timezone

        from boxbot.displays.data_sources import _format_age

        ts = datetime.now(timezone.utc) - timedelta(**delta_kwargs)
        # The memory store stamps naive-UTC ISO strings.
        return _format_age(ts.replace(tzinfo=None).isoformat())

    def test_buckets(self):
        assert self._age(seconds=30) == "now"
        assert self._age(minutes=5) == "5m"
        assert self._age(hours=3) == "3h"
        assert self._age(days=3) == "3d"
        assert self._age(days=15) == "2w"
        assert self._age(days=90) == "3mo"
        assert self._age(days=800) == "2y"

    def test_aware_timestamp_accepted(self):
        from datetime import datetime, timedelta, timezone

        from boxbot.displays.data_sources import _format_age

        ts = datetime.now(timezone.utc) - timedelta(days=3)
        assert _format_age(ts.isoformat()) == "3d"

    def test_garbage_and_missing_are_empty(self):
        from boxbot.displays.data_sources import _format_age

        assert _format_age(None) == ""
        assert _format_age("") == ""
        assert _format_age("not-a-date") == ""

    def test_future_timestamp_clamps_to_now(self):
        from datetime import datetime, timedelta, timezone

        from boxbot.displays.data_sources import _format_age

        ts = datetime.now(timezone.utc) + timedelta(days=2)
        assert _format_age(ts.isoformat()) == "now"


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def _calendar_source(inputs=None):
    """Build an IntegrationSource pointed at the calendar integration.

    Replaces the old standalone ``CalendarSource`` class. Inputs default
    to a ``list_upcoming_events`` call so the existing tests don't need
    to repeat the action everywhere.
    """
    return IntegrationSource(
        "calendar",
        {
            "inputs": inputs or {"action": "list_upcoming_events",
                                 "max_results": 5},
        },
    )


class TestCalendarSourceWiring:
    async def test_returns_events_from_integration(self):
        events = [
            {"id": "a", "title": "Team Standup", "time": "9:00 AM",
             "duration": "30m", "location": "Zoom", "all_day": False,
             "start": "2026-04-23T09:00:00", "end": "2026-04-23T09:30:00",
             "description": ""},
        ]

        async def fake_run(name, inputs, **_kwargs):
            assert name == "calendar"
            assert inputs["action"] == "list_upcoming_events"
            assert inputs["max_results"] == 5
            return {"status": "ok", "output": {"events": events, "count": 1}}

        with patch("boxbot.integrations.runner.run", new=fake_run):
            data = await _calendar_source().fetch()

        assert data["count"] == 1
        assert data["events"][0]["title"] == "Team Standup"

    async def test_returns_empty_when_script_reports_error(self):
        """Unified behavior: script-reported errors come back empty so the
        renderer falls through to its preview placeholder. The old
        CalendarSource synthesised placeholder events here; that
        special-case was the bifurcation we just removed."""
        async def fake_run(name, inputs, **_kwargs):
            return {
                "status": "ok",
                "output": {"error": "Calendar token not stored."},
            }

        with patch("boxbot.integrations.runner.run", new=fake_run):
            data = await _calendar_source().fetch()

        assert data == {}

    async def test_returns_empty_when_integration_missing(self):
        from boxbot.integrations.runner import IntegrationRunError

        async def fake_run(name, inputs, **_kwargs):
            raise IntegrationRunError("unknown integration 'calendar'")

        with patch("boxbot.integrations.runner.run", new=fake_run):
            data = await _calendar_source().fetch()

        assert data == {}


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class TestAgentStateTracker:
    async def test_starts_in_sleeping(self):
        from boxbot.core.agent_state import AgentStateTracker

        tracker = AgentStateTracker()
        snap = await tracker.snapshot()
        assert snap["state"] == "sleeping"
        assert snap["last_active"] is None

    async def test_wake_word_moves_to_listening(self):
        from boxbot.core.agent_state import AgentStateTracker

        tracker = AgentStateTracker()
        await tracker.start()
        try:
            await get_event_bus().publish(WakeWordHeard(confidence=0.9))
            assert tracker.state == "listening"
            assert tracker.last_active is not None
        finally:
            await tracker.stop()

    async def test_speaking_then_done_returns_to_sleeping(self):
        from boxbot.core.agent_state import AgentStateTracker

        tracker = AgentStateTracker()
        await tracker.start()
        try:
            bus = get_event_bus()
            await bus.publish(AgentSpeaking(conversation_id="c1", text="hi"))
            assert tracker.state == "speaking"
            await bus.publish(
                AgentSpeakingDone(conversation_id="c1", interrupted=False)
            )
            # No active conversation, so back to sleeping
            assert tracker.state == "sleeping"
        finally:
            await tracker.stop()

    async def test_conversation_keeps_listening_after_speak_done(self):
        from boxbot.core.agent_state import AgentStateTracker

        tracker = AgentStateTracker()
        await tracker.start()
        try:
            bus = get_event_bus()
            await bus.publish(
                ConversationStarted(conversation_id="c1", channel="voice")
            )
            await bus.publish(AgentSpeaking(conversation_id="c1", text="hi"))
            await bus.publish(
                AgentSpeakingDone(conversation_id="c1", interrupted=False)
            )
            assert tracker.state == "listening"  # conversation still active
            await bus.publish(
                ConversationEnded(conversation_id="c1", channel="voice")
            )
            assert tracker.state == "sleeping"
        finally:
            await tracker.stop()

    async def test_transcript_moves_to_thinking(self):
        from boxbot.core.agent_state import AgentStateTracker

        tracker = AgentStateTracker()
        await tracker.start()
        try:
            await get_event_bus().publish(
                TranscriptReady(conversation_id="c1", transcript="hi")
            )
            assert tracker.state == "thinking"
        finally:
            await tracker.stop()

    async def test_agent_status_source_uses_tracker(self):
        from boxbot.core.agent_state import get_agent_state_tracker

        tracker = get_agent_state_tracker()
        await tracker.start()
        try:
            await get_event_bus().publish(WakeWordHeard(confidence=0.95))
            data = await AgentStatusSource().fetch()
            assert data["state"] == "listening"
            assert data["last_active"] is not None
        finally:
            await tracker.stop()


# ---------------------------------------------------------------------------
# Calendar integration unit tests
# ---------------------------------------------------------------------------


def _weather_source(inputs=None):
    """Build an IntegrationSource pointed at the weather integration."""
    return IntegrationSource("weather", {"inputs": inputs or {}})


class TestWeatherSourceWiring:
    """Weather flows through ``IntegrationSource`` like every other
    integration. Lat/lon defaults live in the manifest's ``default_env``
    (BOXBOT_WEATHER_LAT/LON); see test_integration_runner for that
    coverage.
    """

    async def test_returns_data_from_runner(self):
        from boxbot.integrations import runner as runner_mod

        async def fake_run(name, inputs, *, timeout_override=None):
            assert name == "weather"
            assert inputs["lat"] == 47.6062
            assert inputs["lon"] == -122.3321
            return {
                "status": "ok",
                "output": {
                    "temp": "62", "condition": "Cloudy", "icon": "cloud",
                    "humidity": "85", "wind": "10 mph SW",
                    "forecast": [
                        {"day": "Mon", "icon": "sun", "high": "68", "low": "52"}
                    ],
                },
            }

        with patch.object(runner_mod, "run", new=fake_run):
            data = await _weather_source(
                {"lat": 47.6062, "lon": -122.3321}
            ).fetch()

        assert data["temp"] == "62"
        assert data["icon"] == "cloud"
        assert len(data["forecast"]) == 1

    async def test_returns_empty_on_runner_error(self):
        from boxbot.integrations import runner as runner_mod

        async def boom(name, inputs, *, timeout_override=None):
            return {"status": "error", "error": "API down"}

        with patch.object(runner_mod, "run", new=boom):
            data = await _weather_source(
                {"lat": 47.6062, "lon": -122.3321}
            ).fetch()
        assert data == {}

    async def test_returns_empty_on_unregistered_integration(self):
        from boxbot.integrations import runner as runner_mod

        async def absent(name, inputs, *, timeout_override=None):
            raise runner_mod.IntegrationRunError(f"unknown integration '{name}'")

        with patch.object(runner_mod, "run", new=absent):
            data = await _weather_source(
                {"lat": 47.6062, "lon": -122.3321}
            ).fetch()
        assert data == {}


class TestIntegrationSourceGeneric:
    """Generic IntegrationSource behavior — applies to any integration,
    pre-seeded or agent-authored. Pulled out to assert the unified
    path explicitly (no two-tracks regression)."""

    async def test_routes_integration_name_override(self):
        """``integration`` field overrides which integration is called
        while keeping the source name (and binding) distinct."""
        async def fake_run(name, inputs, **_kwargs):
            assert name == "solar"   # the integration, not the binding
            return {"status": "ok", "output": {"kwh": 42.0}}

        with patch("boxbot.integrations.runner.run", new=fake_run):
            src = IntegrationSource(
                "solar_today",
                {"integration": "solar", "inputs": {"date": "2026-05-15"}},
            )
            data = await src.fetch()
        assert data == {"kwh": 42.0}

    async def test_default_integration_name_is_source_name(self):
        async def fake_run(name, inputs, **_kwargs):
            assert name == "stocks"
            return {"status": "ok", "output": {"price": 184.20}}

        with patch("boxbot.integrations.runner.run", new=fake_run):
            data = await IntegrationSource("stocks", {}).fetch()
        assert data["price"] == 184.20

    async def test_refresh_interval_defaults_to_300(self):
        assert IntegrationSource("foo").refresh_interval == 300
        assert IntegrationSource("foo", {"refresh": 60}).refresh_interval == 60

    def test_create_source_routes_integration_type(self):
        """The spec form ``{"type": "integration"}`` produces an
        IntegrationSource. This is the agent-facing path."""
        src = create_source(
            "solar", "integration",
            {"inputs": {"date": "2026-05-15"}, "refresh": 600},
        )
        assert isinstance(src, IntegrationSource)
        assert src.refresh_interval == 600

    def test_unknown_builtin_name_promotes_to_integration(self):
        """Backward compat: a bare ``{"name": "weather"}`` (no type)
        from old specs becomes an IntegrationSource since weather is
        no longer in the builtin allowlist."""
        src = create_source("weather", "builtin", {})
        assert isinstance(src, IntegrationSource)


class TestCalendarIntegrationNormalization:
    """Helpers moved into integrations/calendar/script.py with the migration.

    The script isn't on sys.path (it's sandbox-runnable, loaded by the
    runner), so we load it via importlib.util to keep coverage of
    ``_normalize_event``.
    """

    @staticmethod
    def _load_script():
        """Load integrations/calendar/script.py as a module.

        The script imports from ``boxbot_sdk`` (a package only installed
        into the sandbox venv). We stub the two specific names the
        script needs at import time — ``boxbot_sdk.integration.inputs``
        and ``return_output`` — so the module body executes and its
        helpers become introspectable.
        """
        import importlib.util
        import sys
        import types
        from pathlib import Path

        if "boxbot_sdk" not in sys.modules:
            pkg = types.ModuleType("boxbot_sdk")
            integ = types.ModuleType("boxbot_sdk.integration")
            integ.inputs = lambda: {}
            integ.return_output = lambda payload: None
            pkg.integration = integ
            sys.modules["boxbot_sdk"] = pkg
            sys.modules["boxbot_sdk.integration"] = integ

        script_path = (
            Path(__file__).resolve().parents[1]
            / "integrations" / "calendar" / "script.py"
        )
        spec = importlib.util.spec_from_file_location(
            "calendar_integration_script", script_path,
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_normalizes_timed_event(self):
        normalize = self._load_script()._normalize_event
        raw = {
            "id": "abc",
            "summary": "Standup",
            "location": "Zoom",
            "description": "Daily sync",
            "start": {"dateTime": "2026-04-23T09:00:00+00:00"},
            "end": {"dateTime": "2026-04-23T09:30:00+00:00"},
        }
        result = normalize(raw)
        assert result["id"] == "abc"
        assert result["title"] == "Standup"
        assert result["location"] == "Zoom"
        assert result["all_day"] is False
        assert result["duration"] == "30m"

    def test_normalizes_all_day_event(self):
        normalize = self._load_script()._normalize_event
        raw = {
            "id": "xyz",
            "summary": "Holiday",
            "start": {"date": "2026-04-25"},
            "end": {"date": "2026-04-26"},
        }
        result = normalize(raw)
        assert result["title"] == "Holiday"
        assert result["all_day"] is True
        assert result["time"] == "all day"

    def test_normalizes_missing_summary(self):
        normalize = self._load_script()._normalize_event
        raw = {
            "id": "1",
            "start": {"dateTime": "2026-04-23T10:00:00+00:00"},
            "end": {"dateTime": "2026-04-23T11:00:00+00:00"},
        }
        assert normalize(raw)["title"] == "(no title)"

    def test_time_field_dispatches_on_all_day(self):
        time_field = self._load_script()._time_field
        assert time_field("2026-04-25", True) == {"date": "2026-04-25"}
        assert time_field(
            "2026-04-25T15:00:00+00:00", False
        ) == {"dateTime": "2026-04-25T15:00:00+00:00"}

    def test_when_field_today(self):
        """Timed events on today's date prefix with "Today"."""
        mod = self._load_script()
        # Build a start datetime that's "today" in local zone, 4pm.
        now_local = datetime.now().astimezone()
        start_local = now_local.replace(hour=16, minute=0, second=0, microsecond=0)
        raw = {
            "id": "t",
            "summary": "Standup",
            "start": {"dateTime": start_local.isoformat()},
            "end": {"dateTime": start_local.isoformat()},
        }
        result = mod._normalize_event(raw)
        assert result["when"].startswith("Today · ")
        assert "4:00 PM" in result["when"] or "16:00" in result["when"]

    def test_when_field_tomorrow_all_day(self):
        mod = self._load_script()
        # Use _format_day directly so we don't depend on tomorrow being a
        # specific date when the test runs.
        from datetime import timedelta
        now_local = datetime.now().astimezone()
        tomorrow_local = (now_local + timedelta(days=1)).replace(
            hour=12, minute=0, second=0, microsecond=0,
        )
        assert mod._format_day(tomorrow_local) == "Tomorrow"
        assert mod._format_when(tomorrow_local, all_day=True) == "Tomorrow · all day"

    def test_when_field_far_future(self):
        """Events more than a week out use absolute Mon Mmm D format."""
        mod = self._load_script()
        from datetime import timedelta
        now_local = datetime.now().astimezone()
        far = (now_local + timedelta(days=30)).replace(
            hour=10, minute=0, second=0, microsecond=0,
        )
        day = mod._format_day(far)
        # Should be "Mon May 15" style — three-letter day, three-letter month, day-of-month.
        parts = day.split()
        assert len(parts) == 3, f"expected 3 tokens, got {day!r}"
        assert len(parts[0]) == 3
        assert len(parts[1]) == 3
        assert parts[2].isdigit()

    def test_when_field_empty_when_start_unknown(self):
        mod = self._load_script()
        assert mod._format_when(None, all_day=False) == ""
        assert mod._format_when(None, all_day=True) == ""

    def test_normalize_event_includes_when(self):
        """_normalize_event surfaces ``when`` alongside ``time``."""
        mod = self._load_script()
        raw = {
            "id": "a",
            "summary": "Meeting",
            "start": {"dateTime": "2026-04-23T09:00:00+00:00"},
            "end": {"dateTime": "2026-04-23T09:30:00+00:00"},
        }
        out = mod._normalize_event(raw)
        assert "when" in out
        assert isinstance(out["when"], str)
        assert out["when"]  # non-empty
        # Both fields coexist — old consumers reading `time` still work.
        assert out["time"]

    def test_all_day_event_when_includes_all_day_marker(self):
        mod = self._load_script()
        raw = {
            "id": "h",
            "summary": "Holiday",
            "start": {"date": "2026-04-25"},
            "end": {"date": "2026-04-26"},
        }
        out = mod._normalize_event(raw)
        assert "all day" in out["when"]

    def test_all_day_date_is_naive(self):
        """All-day events parse without a UTC tzinfo so subsequent
        ``astimezone()`` calls don't shift them across midnight.
        """
        mod = self._load_script()
        parsed = mod._parse_date("2026-05-22")
        assert parsed is not None
        assert parsed.tzinfo is None
        assert parsed.date().isoformat() == "2026-05-22"

    def test_format_day_no_shift_for_all_day_in_negative_offset_zone(
        self, monkeypatch
    ):
        """The original bug: an all-day event on day D was rendered as
        D-1 because UTC midnight converted to the previous day in
        zones west of UTC. With naive datetimes, the day stays put.
        """
        mod = self._load_script()
        from datetime import datetime as _dt

        # All-day "Camp Sherman" on Fri May 22 — naive.
        all_day = mod._parse_date("2026-05-22")
        # Reference "today" in PDT (UTC-7) on Wed May 20.
        today = _dt(2026, 5, 20, 9, 0, 0)  # naive local
        day = mod._format_day(all_day, today=today)
        # 2 days out → weekday name. Must be Fri, not Thu.
        assert day == "Fri"

    def test_format_day_aware_event_normalizes_to_local(self):
        """Timed events keep tzinfo and get normalized via ``astimezone``.

        Sanity check that we didn't break the aware-datetime branch
        while adding the naive branch.
        """
        mod = self._load_script()
        from datetime import datetime as _dt, timezone, timedelta

        # 7am Mountain (UTC-6 in DST) on a future Tuesday.
        tz = timezone(timedelta(hours=-6))
        evt = _dt(2026, 6, 9, 7, 0, 0, tzinfo=tz)
        today = _dt(2026, 6, 8, 12, 0, 0)  # naive local
        day = mod._format_day(evt, today=today)
        # 1 day out from today → "Tomorrow".
        assert day == "Tomorrow"
