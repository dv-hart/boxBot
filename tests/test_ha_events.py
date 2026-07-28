"""Tests for the Home Assistant events bridge (boxbot.integrations.ha_events)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from boxbot.core.events import EntityStateChanged, get_event_bus
from boxbot.integrations.ha_events import HAEventsBridge, _ws_url


class _Recorder:
    """Collects EntityStateChanged events off the bus."""

    def __init__(self) -> None:
        self.events: list[EntityStateChanged] = []
        get_event_bus().subscribe(EntityStateChanged, self._on)

    async def _on(self, event: EntityStateChanged) -> None:
        self.events.append(event)


def _bridge(watched: set[str]) -> HAEventsBridge:
    async def provider() -> set[str]:
        return watched

    return HAEventsBridge(watch_provider=provider, watch_ttl_s=0.0)


class TestWsUrl:
    def test_http(self):
        assert _ws_url("http://192.168.0.5:8123") == (
            "ws://192.168.0.5:8123/api/websocket"
        )

    def test_https(self):
        assert _ws_url("https://ha.example.com") == (
            "wss://ha.example.com/api/websocket"
        )

    def test_bare_host_defaults_to_ws(self):
        assert _ws_url("ha.local:8123") == "ws://ha.local:8123/api/websocket"


class TestEventHandling:
    @pytest.mark.asyncio
    async def test_watched_state_change_is_published(self):
        rec = _Recorder()
        bridge = _bridge({"binary_sensor.front_door_person"})
        await bridge._handle_event({
            "type": "event",
            "event": {
                "event_type": "state_changed",
                "data": {
                    "entity_id": "binary_sensor.front_door_person",
                    "new_state": {
                        "state": "on",
                        "attributes": {"friendly_name": "Front Door Person"},
                    },
                    "old_state": {"state": "off"},
                },
            },
        })
        assert len(rec.events) == 1
        ev = rec.events[0]
        assert ev.entity_id == "binary_sensor.front_door_person"
        assert ev.new_state == "on"
        assert ev.old_state == "off"
        assert ev.friendly_name == "Front Door Person"
        assert ev.snapshot is False

    @pytest.mark.asyncio
    async def test_unwatched_entity_is_ignored(self):
        rec = _Recorder()
        bridge = _bridge({"binary_sensor.front_door_person"})
        await bridge._handle_event({
            "type": "event",
            "event": {
                "event_type": "state_changed",
                "data": {
                    "entity_id": "sun.sun",
                    "new_state": {"state": "below_horizon"},
                    "old_state": {"state": "above_horizon"},
                },
            },
        })
        assert rec.events == []

    @pytest.mark.asyncio
    async def test_non_state_changed_event_is_ignored(self):
        rec = _Recorder()
        bridge = _bridge({"binary_sensor.x"})
        await bridge._handle_event({
            "type": "event",
            "event": {"event_type": "call_service", "data": {}},
        })
        assert rec.events == []

    @pytest.mark.asyncio
    async def test_removed_entity_publishes_unknown(self):
        # new_state is null when an entity is removed from HA.
        rec = _Recorder()
        bridge = _bridge({"binary_sensor.x"})
        await bridge._handle_event({
            "type": "event",
            "event": {
                "event_type": "state_changed",
                "data": {
                    "entity_id": "binary_sensor.x",
                    "new_state": None,
                    "old_state": {"state": "on"},
                },
            },
        })
        assert len(rec.events) == 1
        assert rec.events[0].new_state == "unknown"

    @pytest.mark.asyncio
    async def test_snapshot_marks_events(self):
        rec = _Recorder()
        bridge = _bridge({"binary_sensor.x"})
        await bridge._handle_snapshot([
            {"entity_id": "binary_sensor.x", "state": "off",
             "attributes": {"friendly_name": "X"}},
            {"entity_id": "light.kitchen", "state": "on", "attributes": {}},
        ])
        assert len(rec.events) == 1
        assert rec.events[0].snapshot is True
        assert rec.events[0].new_state == "off"

    @pytest.mark.asyncio
    async def test_publish_unknown_resets_reported_entities(self):
        rec = _Recorder()
        bridge = _bridge({"binary_sensor.x"})
        await bridge._handle_snapshot([
            {"entity_id": "binary_sensor.x", "state": "on", "attributes": {}},
        ])
        await bridge._publish_unknown()
        assert [e.new_state for e in rec.events] == ["on", "unknown"]
        # Second call is a no-op — the reported set was consumed.
        await bridge._publish_unknown()
        assert len(rec.events) == 2


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_is_noop_without_secrets(self):
        bridge = _bridge(set())
        with patch(
            "boxbot.integrations.ha_events._load_connection", return_value=None
        ):
            await bridge.start()
        assert bridge._task is None
        assert bridge._running is False
