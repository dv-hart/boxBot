"""Channel identity and the outbound-channel Protocol.

This module is the seam between channel-agnostic code (output dispatcher,
router, agent) and channel-specific clients (WhatsApp, Signal, …).

Two pieces:

* ``Channel`` — the routing identifier enum. Lives here (not in
  ``router``) so non-router callers can import it without pulling in
  the router's auth/whatsapp dependencies.
* ``OutboundChannel`` — the Protocol every outbound transport client
  satisfies. ``WhatsAppClient`` and (forthcoming) ``SignalClient`` both
  implement it. The dispatcher reaches the right client via the small
  registry below, keyed on the ``Channel`` value.

Voice intentionally is NOT a registered ``OutboundChannel`` — speech
goes through ``VoiceSession``, not the output dispatcher. The enum
keeps ``VOICE`` only because the inbound router uses it as a routing
identifier.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from boxbot.communication.auth import User

logger = logging.getLogger(__name__)


class Channel(Enum):
    """Communication channel types — both inbound routing and outbound dispatch."""

    VOICE = "voice"
    WHATSAPP = "whatsapp"
    SIGNAL = "signal"


@runtime_checkable
class OutboundChannel(Protocol):
    """Per-transport outbound client.

    Implementations: ``WhatsAppClient``, ``SignalClient`` (forthcoming).
    """

    name: str

    async def send_text(self, phone: str, message: str) -> bool: ...

    async def send_attachment(
        self,
        phone: str,
        file_path: str,
        caption: str | None = None,
    ) -> bool: ...

    async def download_media(
        self, media_id: str
    ) -> tuple[bytes, str] | None: ...


# Process-wide registry. ``set_whatsapp_client`` / (forthcoming)
# ``set_signal_client`` register here so the dispatcher can resolve the
# right client without importing the transport module directly.
_registry: dict[Channel, OutboundChannel] = {}


def register_outbound_channel(
    channel: Channel, client: OutboundChannel | None
) -> None:
    """Register (or, if client is None, unregister) an outbound client."""
    if client is None:
        _registry.pop(channel, None)
        return
    _registry[channel] = client


def get_outbound_channel(channel: Channel) -> OutboundChannel | None:
    """Return the registered outbound client for a channel, or None."""
    return _registry.get(channel)


def registered_channels() -> dict[Channel, OutboundChannel]:
    """Return a copy of the live registry. Primarily for diagnostics/tests."""
    return dict(_registry)


async def send_to_admins(
    text: str,
    *,
    skip: Callable[["User"], bool] | None = None,
    on_sent: Callable[["User"], None] | None = None,
) -> tuple[int, int]:
    """Send ``text`` to every registered admin on their channel.

    For each admin, unless ``skip(admin)`` is true: resolve its channel, send,
    and on successful delivery call ``on_sent(admin)``. Returns
    ``(delivered, total_admins)``. Never raises — a per-admin failure is logged
    and skipped. Shared by every admin-approval prompt (package install, etc.).
    """
    from boxbot.communication.auth import get_auth_manager

    auth = get_auth_manager()
    if auth is None:
        logger.warning("Auth manager not initialised — cannot reach admins")
        return 0, 0

    admins = [u for u in await auth.list_users() if u.role == "admin"]
    delivered = 0
    for admin in admins:
        if skip is not None and skip(admin):
            continue
        try:
            channel = Channel(admin.channel)
        except ValueError:
            logger.warning("Admin %s has unknown channel %r; skipping", admin.phone, admin.channel)
            continue
        out = get_outbound_channel(channel)
        if out is None:
            continue
        try:
            sent = await out.send_text(admin.phone, text)
        except Exception:  # noqa: BLE001
            logger.exception("send to admin %s failed", admin.phone)
            continue
        if sent:
            delivered += 1
            if on_sent is not None:
                on_sent(admin)
    return delivered, len(admins)
