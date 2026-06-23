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

import asyncio
import logging
from collections import deque
from typing import Any, Callable

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

    Owns a **dedicated** ``SignalClient`` connection used only for the
    inbound notification stream — separate from the outbound/send client.
    That separation lets the liveness watchdog recycle the inbound
    connection on a timer without ever disturbing in-flight sends.

    Why the watchdog exists: in ``--receive-mode on-start`` the daemon
    auto-pushes inbound messages to connected JSON-RPC clients, but a
    long-lived client connection can silently stop receiving those pushes
    after the daemon's receive-websocket to Signal cycles (observed
    2026-06-10 — inbound went dead for ~12 days while outbound kept
    working, no error logged). A fresh connection re-arms delivery, so we
    periodically refresh using make-before-break: a new connection
    subscribes before the old one is torn down, so there is no window in
    which a message could slip through unreceived. Dedup covers the brief
    overlap when both connections are live.

    Args:
        router: The MessageRouter that handles channel-agnostic auth and
            event publication.
        client_factory: Builds a fresh, unconnected ``SignalClient`` for
            the inbound stream. Called once at ``start`` and again on each
            watchdog refresh.
        refresh_interval: Seconds between liveness refreshes. 0 disables
            the watchdog (the connection is then only re-armed on its own
            socket reconnects).
        dedup_window: Number of (sender, timestamp) tuples to remember
            for duplicate suppression. signal-cli can re-deliver a
            message if a JSON-RPC reconnect happens mid-ack, and the
            make-before-break overlap can briefly double-deliver. 1024
            covers several hours at household-scale volumes.
    """

    def __init__(
        self,
        *,
        router: MessageRouter,
        client_factory: Callable[[], SignalClient],
        refresh_interval: float = 900.0,
        dedup_window: int = 1024,
    ) -> None:
        self._router = router
        self._client_factory = client_factory
        self._refresh_interval = refresh_interval
        self._dedup_window = dedup_window
        self._seen_keys: deque[tuple[str, int]] = deque(maxlen=dedup_window)
        self._seen_set: set[tuple[str, int]] = set()
        self._client: SignalClient | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        """Connect the dedicated inbound client, subscribe, start watchdog."""
        if self._client is not None:
            return
        self._stopped.clear()
        self._client = await self._connect_subscribed()
        if self._refresh_interval > 0:
            self._watchdog_task = asyncio.create_task(
                self._watchdog_loop(), name="signal-inbound-watchdog"
            )
        logger.info(
            "Signal inbound subscribed (account=%s); refresh every %.0fs",
            self._client.account,
            self._refresh_interval,
        )

    async def stop(self) -> None:
        """Stop the watchdog and disconnect the dedicated inbound client."""
        self._stopped.set()
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("Signal inbound watchdog stop error", exc_info=True)
            self._watchdog_task = None
        if self._client is not None:
            self._client.set_notification_handler(None)
            self._client.set_reconnect_handler(None)
            try:
                await self._client.disconnect()
            except Exception:
                logger.debug("Signal inbound client disconnect error", exc_info=True)
            self._client = None

    # -----------------------------------------------------------------
    # Connection lifecycle / liveness
    # -----------------------------------------------------------------

    async def _connect_subscribed(self) -> SignalClient:
        """Build, connect, and subscribe a fresh inbound client."""
        client = self._client_factory()
        client.set_notification_handler(self._on_notification)
        # Re-arm the subscription if this client's own socket reconnects
        # (e.g. the daemon restarts) — the daemon forgets it otherwise.
        client.set_reconnect_handler(client.subscribe_receive)
        await client.connect()
        await client.subscribe_receive()
        return client

    async def _watchdog_loop(self) -> None:
        """Periodically refresh the inbound connection (make-before-break)."""
        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(
                    self._stopped.wait(), timeout=self._refresh_interval
                )
                return  # stopped
            except asyncio.TimeoutError:
                pass
            try:
                await self._refresh()
            except Exception:
                # Keep the existing connection; try again next cycle.
                logger.exception(
                    "Signal inbound refresh failed; keeping current connection"
                )

    async def _refresh(self) -> None:
        """Stand up a new subscribed connection, then retire the old one.

        Make-before-break: the new client is connected and subscribed
        *before* the old one is detached, so there is no instant during
        which no client is receiving. Any message delivered by both during
        the overlap is suppressed by dedup.
        """
        new_client = await self._connect_subscribed()
        old_client = self._client
        self._client = new_client
        logger.debug("Signal inbound connection refreshed")
        if old_client is not None:
            old_client.set_notification_handler(None)
            old_client.set_reconnect_handler(None)
            try:
                await old_client.disconnect()
            except Exception:
                logger.debug(
                    "Signal inbound: old client disconnect error", exc_info=True
                )

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

        raw_sender = envelope.get("sourceNumber") or envelope.get("source") or ""
        if not raw_sender or not raw_sender.startswith("+"):
            # No usable phone — could be a UUID-only sender during
            # contact discovery transitions. Skip rather than spoof.
            logger.debug("Signal: notification without phone source; skipping")
            return
        # boxBot's user database stores phones WITHOUT the leading +
        # (the WhatsApp Cloud API webhook delivers raw digits, and the
        # users table was first populated from there). Strip the +
        # before handing to the router so auth lookups match.
        sender_phone = raw_sender.lstrip("+")

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
