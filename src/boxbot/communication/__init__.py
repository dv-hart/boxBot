"""Communication layer — all external channels between boxBot and the world.

Provides user authentication, WhatsApp integration, and message routing.
Every path through this module is authenticated and intentional.
"""

from boxbot.communication.auth import AuthManager, RegisterResult, User
from boxbot.communication.channels import (
    Channel,
    OutboundChannel,
    get_outbound_channel,
    register_outbound_channel,
)
from boxbot.communication.router import MessageRouter
from boxbot.communication.signal_client import SignalClient
from boxbot.communication.signal_inbound import SignalInbound
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
    "OutboundChannel",
    "RegisterResult",
    "SignalClient",
    "SignalInbound",
    "User",
    "WhatsAppClient",
    "WhatsAppWebhook",
    "get_outbound_channel",
    "register_outbound_channel",
]
