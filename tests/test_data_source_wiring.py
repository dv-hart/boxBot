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
    async def test_returns_data_from_noaa(self):
        from boxbot.integrations import noaa_weather as wx

        async def fake_fetch(*, lat, lon, forecast_days=5):
            assert lat == 47.6062
            assert lon == -122.3321
            return {
                "temp": "62", "condition": "Cloudy", "icon": "cloud",
                "humidity": "85", "wind": "10 mph SW",
                "forecast": [{"day": "Mon", "icon": "sun", "high": "68", "low": "52"}],
            }

        with patch.object(wx, "fetch_weather", new=fake_fetch):
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

    async def test_falls_back_to_placeholder_on_fetch_failure(self):
        from boxbot.integrations import noaa_weather as wx

        async def boom(**kwargs):
            raise RuntimeError("API down")

        with patch.object(wx, "fetch_weather", new=boom):
            data = await WeatherSource(
                {"lat": 47.6062, "lon": -122.3321}
            ).fetch()
        assert "icon" in data  # placeholder fallback


class TestNOAAIconMapping:
    def test_maps_common_conditions(self):
        from boxbot.integrations.noaa_weather import _map_icon

        assert _map_icon("Sunny", {"isDaytime": True}) == "sun"
        assert _map_icon("Clear", {"isDaytime": False}) == "moon"
        assert _map_icon("Partly Cloudy", {"isDaytime": True}) == "cloud-sun"
        assert _map_icon("Partly Cloudy", {"isDaytime": False}) == "cloud-moon"
        assert _map_icon("Cloudy", {"isDaytime": True}) == "cloud"
        assert _map_icon("Light Rain", {"isDaytime": True}) == "cloud-drizzle"
        assert _map_icon("Heavy Rain", {"isDaytime": True}) == "cloud-rain"
        assert _map_icon("Snow Showers", {"isDaytime": True}) == "cloud-snow"
        assert _map_icon("Thunderstorms", {"isDaytime": True}) == "cloud-lightning"
        assert _map_icon("Foggy", {"isDaytime": True}) == "cloud-fog"

    def test_falls_back_for_unknown(self):
        from boxbot.integrations.noaa_weather import _map_icon

        # Unknown condition with "cloud" word
        assert _map_icon("Unknown cloud thing", {"isDaytime": True}) == "cloud"
        # Pure unknown
        assert _map_icon("Bizarre weather", {"isDaytime": True}) == "sun"


class TestNOAAForecastBuilding:
    def test_pairs_day_and_night(self):
        from boxbot.integrations.noaa_weather import _build_forecast

        periods = [
            {"name": "Friday", "isDaytime": True, "temperature": 75,
             "shortForecast": "Sunny",
             "startTime": "2026-04-24T06:00:00-07:00"},
            {"name": "Friday Night", "isDaytime": False, "temperature": 55,
             "shortForecast": "Clear",
             "startTime": "2026-04-24T18:00:00-07:00"},
            {"name": "Saturday", "isDaytime": True, "temperature": 78,
             "shortForecast": "Mostly Sunny",
             "startTime": "2026-04-25T06:00:00-07:00"},
            {"name": "Saturday Night", "isDaytime": False, "temperature": 58,
             "shortForecast": "Partly Cloudy",
             "startTime": "2026-04-25T18:00:00-07:00"},
        ]

        result = _build_forecast(periods, days=5)
        assert len(result) == 2
        assert result[0]["day"] == "Fri"
        assert result[0]["high"] == "75"
        assert result[0]["low"] == "55"
        assert result[1]["high"] == "78"
        assert result[1]["low"] == "58"


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
