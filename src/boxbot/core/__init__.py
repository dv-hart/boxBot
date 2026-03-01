"""boxBot core — configuration, events, agent lifecycle, and scheduling."""

from boxbot.core.agent import BoxBotAgent
from boxbot.core.config import BoxBotConfig, get_config, load_config
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
from boxbot.core.scheduler import Scheduler

__all__ = [
    # Agent
    "BoxBotAgent",
    # Config
    "BoxBotConfig",
    "get_config",
    "load_config",
    # Events
    "Event",
    "EventBus",
    "get_event_bus",
    "ButtonPressed",
    "ConversationEnded",
    "ConversationStarted",
    "DisplaySwitch",
    "MotionDetected",
    "PersonDetected",
    "PersonIdentified",
    "TriggerFired",
    "WakeWordHeard",
    "WhatsAppMessage",
    # Scheduler
    "Scheduler",
]
