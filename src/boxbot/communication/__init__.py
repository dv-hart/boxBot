"""Communication layer — all external channels between boxBot and the world.

Provides user authentication, WhatsApp integration, and message routing.
Every path through this module is authenticated and intentional.
"""

from boxbot.communication.auth import AuthManager, RegisterResult, User
from boxbot.communication.router import Channel, MessageRouter
from boxbot.communication.whatsapp import (
    IncomingMessage,
    WhatsAppClient,
    WhatsAppWebhook,
)

__all__ = [
    "AuthManager",
    "Channel",
    "IncomingMessage",
    "MessageRouter",
    "RegisterResult",
    "User",
    "WhatsAppClient",
    "WhatsAppWebhook",
]
