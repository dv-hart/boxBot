"""Tests for the dev/eval "local chat" channel (``communication/local_chat.py``).

The channel spoofs Signal over TCP: inbound lines route as
``Channel.SIGNAL`` from the dev user, outbound sends broadcast as
``BB> …`` lines to connected clients. Covered:

* inbound — a TCP client's line reaches ``router.route_incoming`` with
  ``Channel.SIGNAL`` + the text + the dev phone;
* outbound — ``send_text`` broadcasts the ``BB> ``-framed line to a
  connected client;
* auth — ``start`` registers the dev user end to end through a real
  ``AuthManager`` on a temp sqlite so ``route_incoming`` accepts it.
"""

from __future__ import annotations

import asyncio

import pytest

from boxbot.communication.auth import AuthManager
from boxbot.communication.channels import Channel, get_outbound_channel
from boxbot.communication.local_chat import LocalChatChannel

DEV_PHONE = "15550000000"
DEV_NAME = "Dev"


class FakeRouter:
    """Captures route_incoming calls; matches the real kwargs signature."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.event = asyncio.Event()

    async def route_incoming(
        self,
        channel: Channel,
        sender_phone: str,
        message: str | None,
        *,
        media_url: str | None = None,
        media_type: str | None = None,
        sender_name: str | None = None,
        message_id: str = "",
    ) -> bool:
        self.calls.append((channel, sender_phone, message, sender_name))
        self.event.set()
        return True


async def _make_auth(tmp_path) -> AuthManager:
    auth = AuthManager(db_path=tmp_path / "users.db")
    await auth.init_db()
    return auth


async def _open_client(channel: LocalChatChannel):
    return await asyncio.open_connection("127.0.0.1", channel.port)


async def test_inbound_line_routes_as_signal(tmp_path) -> None:
    auth = await _make_auth(tmp_path)
    router = FakeRouter()
    channel = LocalChatChannel(
        host="127.0.0.1", port=0, dev_phone=DEV_PHONE, dev_name=DEV_NAME,
        router=router, auth=auth,
    )
    await channel.start()
    try:
        reader, writer = await _open_client(channel)
        writer.write(b"hello box\n")
        await writer.drain()

        await asyncio.wait_for(router.event.wait(), timeout=2.0)
        assert router.calls, "route_incoming was not awaited"
        chan, phone, message, name = router.calls[0]
        assert chan is Channel.SIGNAL
        assert message == "hello box"
        assert phone == DEV_PHONE
        assert name == DEV_NAME

        writer.close()
        await writer.wait_closed()
    finally:
        await channel.stop()


async def test_send_text_broadcasts_framed_line(tmp_path) -> None:
    auth = await _make_auth(tmp_path)
    channel = LocalChatChannel(
        host="127.0.0.1", port=0, dev_phone=DEV_PHONE, dev_name=DEV_NAME,
        router=FakeRouter(), auth=auth,
    )
    await channel.start()
    try:
        reader, writer = await _open_client(channel)
        # Give the server a moment to register the client writer.
        await asyncio.sleep(0.05)

        assert await channel.send_text(DEV_PHONE, "hi there") is True

        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        assert line == b"BB> hi there\n"

        writer.close()
        await writer.wait_closed()
    finally:
        await channel.stop()


async def test_start_registers_dev_user_and_outbound(tmp_path) -> None:
    auth = await _make_auth(tmp_path)
    assert await auth.get_user(DEV_PHONE) is None

    channel = LocalChatChannel(
        host="127.0.0.1", port=0, dev_phone=DEV_PHONE, dev_name=DEV_NAME,
        router=FakeRouter(), auth=auth,
    )
    await channel.start()
    try:
        user = await auth.get_user(DEV_PHONE)
        assert user is not None
        assert user.name == DEV_NAME
        # Registered on a fresh box → first admin; channel stored as signal.
        assert user.role == "admin"
        assert user.channel == "signal"

        # start() also registered itself as the Signal outbound channel.
        assert get_outbound_channel(Channel.SIGNAL) is channel

        # Idempotent: a second start must be a no-op. port=0 means a
        # non-idempotent start would bind a NEW ephemeral port instead
        # of raising, so assert the server object and port are both
        # unchanged — a bare "did not raise" proves nothing here.
        server, port = channel._server, channel.port
        await channel.start()
        assert channel._server is server
        assert channel.port == port
        assert await auth.get_user(DEV_PHONE) is not None
    finally:
        await channel.stop()
        assert get_outbound_channel(Channel.SIGNAL) is None


async def test_inbound_routes_end_to_end_through_real_auth(tmp_path) -> None:
    """A registered dev user's line survives the real router auth gate."""
    from boxbot.communication.router import MessageRouter

    auth = await _make_auth(tmp_path)
    router = MessageRouter(auth=auth)

    # Capture what the router publishes to the event bus.
    from boxbot.core.events import SignalMessage, get_event_bus

    received: list[SignalMessage] = []
    seen = asyncio.Event()

    async def _handler(event: SignalMessage) -> None:
        received.append(event)
        seen.set()

    get_event_bus().subscribe(SignalMessage, _handler)

    channel = LocalChatChannel(
        host="127.0.0.1", port=0, dev_phone=DEV_PHONE, dev_name=DEV_NAME,
        router=router, auth=auth,
    )
    await channel.start()
    try:
        reader, writer = await _open_client(channel)
        writer.write(b"end to end\n")
        await writer.drain()

        await asyncio.wait_for(seen.wait(), timeout=2.0)
        assert received, "message was dropped before reaching the agent"
        assert received[0].text == "end to end"
        assert received[0].sender_phone == DEV_PHONE

        writer.close()
        await writer.wait_closed()
    finally:
        await channel.stop()
