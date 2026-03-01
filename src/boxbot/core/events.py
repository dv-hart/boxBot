"""Internal event bus for decoupled communication between subsystems.

Provides an async publish/subscribe event bus with typed event dataclasses.
Subsystems communicate without direct imports — the perception pipeline
doesn't know about skills, the display manager doesn't know about WhatsApp.

Usage:
    from boxbot.core.events import get_event_bus, PersonIdentified

    bus = get_event_bus()

    # Subscribe
    async def on_person(event: PersonIdentified):
        print(f"Saw {event.person_name}")

    bus.subscribe(PersonIdentified, on_person)

    # Publish
    await bus.publish(PersonIdentified(person_name="Jacob", confidence=0.92))
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Type alias for event handler: async callable that takes an event and returns None
EventHandler = Callable[..., Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """Base class for all events."""

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class MotionDetected(Event):
    """Perception pipeline detected motion in the camera frame.

    Source: Perception (CPU motion detection)
    Consumers: Perception (triggers YOLO check)
    """

    region: str = ""


@dataclass(frozen=True)
class PersonDetected(Event):
    """Perception pipeline detected a person in frame (YOLO).

    Source: Perception
    Consumers: Agent, Display, Scheduler
    """

    person_ref: str = ""  # temporary tracking reference for this detection
    bbox: tuple[int, int, int, int] | None = None
    confidence: float = 0.0
    source: str = "visual"  # "visual"


@dataclass(frozen=True)
class PersonIdentified(Event):
    """Perception pipeline identified a known person.

    Source: Perception (ReID + speaker ID fusion)
    Consumers: Agent, Memory, Scheduler
    """

    person_id: str = ""  # stable person identifier
    person_name: str = ""
    confidence: float = 0.0
    source: str = ""  # "visual", "voice", "fused"


@dataclass(frozen=True)
class WakeWordHeard(Event):
    """Wake word was detected in the audio stream.

    Source: Communication (OpenWakeWord)
    Consumers: Agent, Display, Perception
    """

    confidence: float = 0.0


@dataclass(frozen=True)
class ButtonPressed(Event):
    """Physical button press on the box (KB2040 or touchscreen).

    Source: Hardware (buttons / screen touch)
    Consumers: Agent, Display
    """

    button_id: str = ""
    action: str = "press"  # "press", "long_press", "release"


@dataclass(frozen=True)
class WhatsAppMessage(Event):
    """Incoming WhatsApp message from an authorized user.

    Source: Communication (webhook)
    Consumers: Agent
    """

    sender_name: str = ""
    sender_phone: str = ""
    text: str = ""
    media_url: str | None = None
    media_type: str | None = None  # "image", "audio", "document"


@dataclass(frozen=True)
class TriggerFired(Event):
    """A scheduler trigger's conditions have all been met.

    Source: Scheduler
    Consumers: Agent
    """

    trigger_id: str = ""
    description: str = ""
    instructions: str = ""
    person: str | None = None
    for_person: str | None = None
    todo_id: str | None = None
    is_recurring: bool = False


@dataclass(frozen=True)
class ConversationStarted(Event):
    """Agent began a conversation (voice or WhatsApp).

    Source: Agent
    Consumers: Display, Memory, Perception
    """

    conversation_id: str = ""
    channel: str = ""  # "voice", "whatsapp", "trigger"
    person_name: str | None = None
    participants: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConversationEnded(Event):
    """Agent finished a conversation.

    Source: Agent
    Consumers: Memory, Scheduler, Perception
    """

    conversation_id: str = ""
    channel: str = ""
    person_name: str | None = None
    turn_count: int = 0
    summary: str = ""


@dataclass(frozen=True)
class DisplaySwitch(Event):
    """Request to switch the active display.

    Source: Agent / Display Manager
    Consumers: Display
    """

    display_name: str = ""
    args: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------


class EventBus:
    """Async event bus with publish/subscribe pattern.

    Handlers are called concurrently for each published event. Errors in
    one handler do not affect other handlers — each is isolated and logged.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[EventHandler]] = defaultdict(list)

    def subscribe(
        self,
        event_type: type[Event],
        handler: EventHandler,
    ) -> None:
        """Register an async handler for an event type.

        Args:
            event_type: The event class to subscribe to.
            handler: Async callable that receives the event instance.
        """
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)
            logger.debug(
                "Subscribed %s to %s", handler.__qualname__, event_type.__name__
            )

    def unsubscribe(
        self,
        event_type: type[Event],
        handler: EventHandler,
    ) -> None:
        """Remove a handler from an event type.

        Args:
            event_type: The event class to unsubscribe from.
            handler: The handler to remove.
        """
        handlers = self._handlers.get(event_type)
        if handlers and handler in handlers:
            handlers.remove(handler)
            logger.debug(
                "Unsubscribed %s from %s",
                handler.__qualname__,
                event_type.__name__,
            )

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribed handlers.

        Handlers run concurrently via asyncio.gather. Errors in individual
        handlers are caught and logged — they never propagate to the
        publisher or affect other handlers.

        Args:
            event: The event instance to publish.
        """
        event_name = type(event).__name__
        handlers = self._handlers.get(type(event), [])

        if not handlers:
            logger.debug("Published %s (no handlers)", event_name)
            return

        logger.debug(
            "Publishing %s to %d handler(s)", event_name, len(handlers)
        )

        async def _safe_call(handler: EventHandler) -> None:
            try:
                await handler(event)
            except Exception:
                logger.exception(
                    "Error in handler %s for %s",
                    handler.__qualname__,
                    event_name,
                )

        await asyncio.gather(*(_safe_call(h) for h in handlers))

    def clear(self) -> None:
        """Remove all subscriptions. Primarily for testing."""
        self._handlers.clear()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the global event bus singleton, creating it on first access."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
