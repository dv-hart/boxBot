"""Inbound Signal message dispatcher.

Subscribes to the signal-cli daemon's JSON-RPC notification stream
(``subscribeReceive``), parses each ``receive`` event, and routes the
resulting ``SignalMessage`` through ``MessageRouter`` — same path the
WhatsApp inbound poller uses, just a different transport.

Unlike WhatsApp's SQS path, there is no separate poller process — the
signal-cli daemon pushes notifications on the same socket connection
``SignalClient`` already maintains. We just wire a notification handler
into that connection.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from boxbot.communication.channels import Channel
from boxbot.communication.router import MessageRouter
from boxbot.communication.signal_client import SignalClient

logger = logging.getLogger(__name__)


# signal-cli reports the attachment's MIME ("image/jpeg") rather than a
# coarse category. The agent's inbound staging keys on category names
# ("image", "audio", …), so we normalise here.
_MEDIA_CATEGORY_PREFIX = {
    "image/": "image",
    "audio/": "audio",
    "video/": "video",
    "application/pdf": "document",
}


def _media_category(mime: str | None) -> str | None:
    if not mime:
        return None
    mime = mime.lower()
    for prefix, category in _MEDIA_CATEGORY_PREFIX.items():
        if mime.startswith(prefix):
            return category
    return "document"


class SignalInbound:
    """Bridges signal-cli notifications into the MessageRouter.

    Args:
        router: The MessageRouter that handles channel-agnostic auth and
            event publication.
        client: The connected ``SignalClient`` whose notification stream
            we subscribe to.
        dedup_window: Number of (sender, timestamp) tuples to remember
            for duplicate suppression. signal-cli can re-deliver a
            message if a JSON-RPC reconnect happens mid-ack. 1024 covers
            several hours at household-scale volumes.
    """

    def __init__(
        self,
        *,
        router: MessageRouter,
        client: SignalClient,
        dedup_window: int = 1024,
    ) -> None:
        self._router = router
        self._client = client
        self._dedup_window = dedup_window
        self._seen_keys: deque[tuple[str, int]] = deque(maxlen=dedup_window)
        self._seen_set: set[tuple[str, int]] = set()
        self._subscribed = False

    async def start(self) -> None:
        """Wire up the notification handler and subscribe on the daemon."""
        if self._subscribed:
            return
        self._client.set_notification_handler(self._on_notification)
        # If the client isn't connected yet, connect now. Idempotent.
        await self._client.connect()
        await self._client.subscribe_receive()
        self._subscribed = True
        logger.info("Signal inbound subscribed (account=%s)", self._client.account)

    async def stop(self) -> None:
        """Detach the notification handler. Does not unsubscribe on the
        daemon — disconnecting the socket implicitly unsubscribes, and
        callers that want a clean shutdown should call
        ``SignalClient.disconnect`` after this.
        """
        self._client.set_notification_handler(None)
        self._subscribed = False

    # -----------------------------------------------------------------
    # Notification handling
    # -----------------------------------------------------------------

    async def _on_notification(self, msg: dict[str, Any]) -> None:
        """Dispatch a single JSON-RPC notification."""
        method = msg.get("method")
        if method != "receive":
            # subscribeReceive may also yield "syncMessage" / control
            # frames — ignore quietly.
            return
        params = msg.get("params") or {}
        envelope = params.get("envelope") or {}
        if not envelope:
            return

        data = envelope.get("dataMessage")
        if not isinstance(data, dict):
            # Receipt / typing / sync — not a user-authored message.
            return

        sender_phone = envelope.get("sourceNumber") or envelope.get("source") or ""
        if not sender_phone or not sender_phone.startswith("+"):
            # No usable phone — could be a UUID-only sender during
            # contact discovery transitions. Skip rather than spoof.
            logger.debug("Signal: notification without phone source; skipping")
            return

        timestamp = int(envelope.get("timestamp") or data.get("timestamp") or 0)
        dedup_key = (sender_phone, timestamp)
        if dedup_key in self._seen_set:
            logger.debug("Signal: duplicate envelope %s, dropping", dedup_key)
            return
        self._seen_keys.append(dedup_key)
        self._seen_set.add(dedup_key)
        # Trim the set when the deque rotates out the oldest entry.
        while len(self._seen_set) > self._dedup_window:
            stale = self._seen_keys.popleft()
            self._seen_set.discard(stale)

        text = data.get("message") or ""
        sender_name = envelope.get("sourceName") or None

        media_url: str | None = None
        media_type: str | None = None
        attachments = data.get("attachments") or []
        if attachments:
            first = attachments[0]
            media_url = first.get("id") or None
            media_type = _media_category(first.get("contentType"))

        await self._router.route_incoming(
            Channel.SIGNAL,
            sender_phone,
            text,
            media_url=media_url,
            media_type=media_type,
            sender_name=sender_name,
            message_id=str(timestamp),
        )
