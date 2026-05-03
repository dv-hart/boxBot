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
    CalendarSource,
    TasksSource,
    WeatherSource,
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
# Calendar
# ---------------------------------------------------------------------------


class TestCalendarSourceWiring:
    async def test_returns_events_from_integration(self):
        from boxbot.integrations import google_calendar as gc

        events = [
            {"id": "a", "title": "Team Standup", "time": "9:00 AM",
             "duration": "30m", "location": "Zoom", "all_day": False,
             "start": "2026-04-23T09:00:00", "end": "2026-04-23T09:30:00",
             "description": ""},
        ]

        async def fake_list(*, max_results=5, calendar_id=None,
                            time_min=None, time_max=None):
            assert max_results == 5
            return events

        with patch.object(gc, "list_upcoming_events", new=fake_list):
            data = await CalendarSource().fetch()

        assert data["count"] == 1
        assert data["events"][0]["title"] == "Team Standup"

    async def test_falls_back_to_placeholder_when_unauth(self):
        from boxbot.integrations import google_calendar as gc

        async def boom(*, max_results=5, calendar_id=None,
                       time_min=None, time_max=None):
            raise gc.CalendarNotAuthenticated("no token")

        with patch.object(gc, "list_upcoming_events", new=boom):
            data = await CalendarSource().fetch()

        assert "events" in data
        assert len(data["events"]) >= 1  # placeholder data
        assert data["events"][0]["title"] == "Team Standup"


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


class TestWeatherSourceWiring:
    """WeatherSource now goes through the integration runner.

    The pure-data logic (icon mapping, day/night forecast pairing)
    moved into ``integrations/weather/script.py``. End-to-end coverage
    of that logic lives in ``tests/test_integration_runner.py`` once
    the weather integration is registered against a test root.
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
            data = await WeatherSource(
                {"lat": 47.6062, "lon": -122.3321}
            ).fetch()

        assert data["temp"] == "62"
        assert data["icon"] == "cloud"
        assert len(data["forecast"]) == 1

    async def test_falls_back_to_placeholder_on_no_coords(self):
        # No env vars, no config — should return placeholder
        with patch.dict("os.environ", {}, clear=False):
            for var in ("BOXBOT_WEATHER_LAT", "BOXBOT_WEATHER_LON"):
                if var in __import__("os").environ:
                    del __import__("os").environ[var]
            data = await WeatherSource().fetch()
        assert "icon" in data  # placeholder has icon

    async def test_falls_back_to_placeholder_on_runner_error(self):
        from boxbot.integrations import runner as runner_mod

        async def boom(name, inputs, *, timeout_override=None):
            return {"status": "error", "error": "API down"}

        with patch.object(runner_mod, "run", new=boom):
            data = await WeatherSource(
                {"lat": 47.6062, "lon": -122.3321}
            ).fetch()
        assert "icon" in data  # placeholder fallback

    async def test_falls_back_to_placeholder_on_unregistered_integration(self):
        from boxbot.integrations import runner as runner_mod

        async def absent(name, inputs, *, timeout_override=None):
            raise runner_mod.IntegrationRunError(f"unknown integration '{name}'")

        with patch.object(runner_mod, "run", new=absent):
            data = await WeatherSource(
                {"lat": 47.6062, "lon": -122.3321}
            ).fetch()
        assert "icon" in data


class TestCalendarIntegrationNormalization:
    def test_normalizes_timed_event(self):
        from boxbot.integrations.google_calendar import _normalize_event

        raw = {
            "id": "abc",
            "summary": "Standup",
            "location": "Zoom",
            "description": "Daily sync",
            "start": {"dateTime": "2026-04-23T09:00:00+00:00"},
            "end": {"dateTime": "2026-04-23T09:30:00+00:00"},
        }
        result = _normalize_event(raw)
        assert result["id"] == "abc"
        assert result["title"] == "Standup"
        assert result["location"] == "Zoom"
        assert result["all_day"] is False
        assert result["duration"] == "30m"

    def test_normalizes_all_day_event(self):
        from boxbot.integrations.google_calendar import _normalize_event

        raw = {
            "id": "xyz",
            "summary": "Holiday",
            "start": {"date": "2026-04-25"},
            "end": {"date": "2026-04-26"},
        }
        result = _normalize_event(raw)
        assert result["title"] == "Holiday"
        assert result["all_day"] is True
        assert result["time"] == "all day"

    def test_normalizes_missing_summary(self):
        from boxbot.integrations.google_calendar import _normalize_event

        raw = {
            "id": "1",
            "start": {"dateTime": "2026-04-23T10:00:00+00:00"},
            "end": {"dateTime": "2026-04-23T11:00:00+00:00"},
        }
        assert _normalize_event(raw)["title"] == "(no title)"
