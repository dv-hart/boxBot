"""Tests for boxbot.core.events — async event bus and event dataclasses."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from boxbot.core.events import (
    ButtonPressed,
    ConversationEnded,
    ConversationStarted,
    DisplaySwitch,
    Event,
    EventBus,
    MotionDetected,
    PersonDetected,
    PersonIdentified,
    TriggerFired,
    WakeWordHeard,
    WhatsAppMessage,
    get_event_bus,
)


class TestEventDataclasses:
    """Test that event dataclasses have proper defaults and are frozen."""

    def test_event_has_timestamp(self):
        event = Event()
        assert isinstance(event.timestamp, datetime)

    def test_person_detected_default_values(self):
        event = PersonDetected()
        assert event.person_ref == ""
        assert event.confidence == 0.0
        assert event.source == "visual"
        assert event.bbox is None

    def test_person_identified_fields(self):
        event = PersonIdentified(
            person_id="p1", person_name="Jacob", confidence=0.95, source="fused"
        )
        assert event.person_name == "Jacob"
        assert event.confidence == 0.95
        assert event.source == "fused"

    def test_whatsapp_message_fields(self):
        event = WhatsAppMessage(
            sender_name="Alice",
            sender_phone="+15551234567",
            text="Hello BB",
            media_url=None,
        )
        assert event.sender_name == "Alice"
        assert event.text == "Hello BB"
        assert event.media_url is None

    def test_trigger_fired_fields(self):
        event = TriggerFired(
            trigger_id="t_abc",
            description="Morning check",
            instructions="Do stuff",
            is_recurring=True,
        )
        assert event.trigger_id == "t_abc"
        assert event.is_recurring is True

    def test_conversation_started_has_participants(self):
        event = ConversationStarted(
            conversation_id="c1",
            channel="voice",
            participants=["Jacob", "BB"],
        )
        assert "Jacob" in event.participants

    def test_display_switch_args(self):
        event = DisplaySwitch(
            display_name="weather",
            args={"mode": "detailed"},
        )
        assert event.args["mode"] == "detailed"

    def test_events_are_frozen(self):
        event = PersonDetected(person_ref="ref1")
        with pytest.raises(AttributeError):
            event.person_ref = "ref2"  # type: ignore[misc]


class TestEventBusSubscribePublish:
    """Test EventBus subscribe/publish/unsubscribe behavior."""

    @pytest.mark.asyncio
    async def test_handler_receives_published_event(self, event_bus):
        received = []

        async def handler(event: PersonDetected):
            received.append(event)

        event_bus.subscribe(PersonDetected, handler)
        await event_bus.publish(PersonDetected(person_ref="test"))

        assert len(received) == 1
        assert received[0].person_ref == "test"

    @pytest.mark.asyncio
    async def test_multiple_handlers_all_called(self, event_bus):
        results = {"a": False, "b": False}

        async def handler_a(event: WakeWordHeard):
            results["a"] = True

        async def handler_b(event: WakeWordHeard):
            results["b"] = True

        event_bus.subscribe(WakeWordHeard, handler_a)
        event_bus.subscribe(WakeWordHeard, handler_b)
        await event_bus.publish(WakeWordHeard(confidence=0.9))

        assert results["a"] is True
        assert results["b"] is True

    @pytest.mark.asyncio
    async def test_handler_only_receives_subscribed_event_type(self, event_bus):
        received_person = []
        received_button = []

        async def on_person(event: PersonDetected):
            received_person.append(event)

        async def on_button(event: ButtonPressed):
            received_button.append(event)

        event_bus.subscribe(PersonDetected, on_person)
        event_bus.subscribe(ButtonPressed, on_button)

        await event_bus.publish(PersonDetected(person_ref="ref"))
        await event_bus.publish(ButtonPressed(button_id="btn1"))

        assert len(received_person) == 1
        assert len(received_button) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_handler(self, event_bus):
        received = []

        async def handler(event: MotionDetected):
            received.append(event)

        event_bus.subscribe(MotionDetected, handler)
        await event_bus.publish(MotionDetected(region="left"))
        assert len(received) == 1

        event_bus.unsubscribe(MotionDetected, handler)
        await event_bus.publish(MotionDetected(region="right"))
        assert len(received) == 1  # still 1, handler was removed

    @pytest.mark.asyncio
    async def test_publish_with_no_handlers_does_not_raise(self, event_bus):
        # Should complete without error
        await event_bus.publish(ConversationEnded(conversation_id="c1"))

    @pytest.mark.asyncio
    async def test_handler_error_does_not_affect_other_handlers(self, event_bus):
        received = []

        async def bad_handler(event: PersonDetected):
            raise ValueError("Intentional test error")

        async def good_handler(event: PersonDetected):
            received.append(event)

        event_bus.subscribe(PersonDetected, bad_handler)
        event_bus.subscribe(PersonDetected, good_handler)
        await event_bus.publish(PersonDetected(person_ref="test"))

        # good_handler should still have been called
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_duplicate_subscribe_ignored(self, event_bus):
        call_count = 0

        async def handler(event: WakeWordHeard):
            nonlocal call_count
            call_count += 1

        event_bus.subscribe(WakeWordHeard, handler)
        event_bus.subscribe(WakeWordHeard, handler)  # duplicate
        await event_bus.publish(WakeWordHeard(confidence=0.8))

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_clear_removes_all_handlers(self, event_bus):
        received = []

        async def handler(event: PersonDetected):
            received.append(event)

        event_bus.subscribe(PersonDetected, handler)
        event_bus.clear()
        await event_bus.publish(PersonDetected(person_ref="test"))

        assert len(received) == 0


class TestGetEventBusSingleton:
    """Test get_event_bus() singleton behavior."""

    def test_returns_same_instance(self):
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2
