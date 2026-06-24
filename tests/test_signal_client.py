"""Unit tests for SignalClient — focused on framing and dispatch.

The unix socket I/O is exercised end-to-end at deploy time against the
real signal-cli daemon. These tests cover the in-process state machine:
request/response correlation by id, notification fan-out, and the error
shape.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from boxbot.communication.signal_client import (
    SignalClient,
    SignalRpcError,
)


def _make_client() -> SignalClient:
    return SignalClient(
        socket_path="/dev/null",
        account="+15039858519",
        attachments_dir="/tmp",
    )


@pytest.mark.asyncio
async def test_dispatch_frame_resolves_pending_future_by_id():
    client = _make_client()
    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    client._pending[42] = fut

    await client._dispatch_frame(
        {"jsonrpc": "2.0", "id": 42, "result": {"version": "0.14.4.1"}}
    )

    assert fut.done()
    assert fut.result() == {"version": "0.14.4.1"}
    assert 42 not in client._pending


@pytest.mark.asyncio
async def test_dispatch_frame_propagates_error_as_signal_rpc_error():
    client = _make_client()
    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    client._pending[7] = fut

    await client._dispatch_frame(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "error": {
                "code": -32601,
                "message": "Method not implemented",
                "data": None,
            },
        }
    )

    assert fut.done()
    with pytest.raises(SignalRpcError) as exc:
        fut.result()
    assert exc.value.code == -32601
    assert "Method not implemented" in str(exc.value)


@pytest.mark.asyncio
async def test_notification_invokes_registered_handler():
    client = _make_client()
    seen: list[dict] = []

    async def handler(msg: dict) -> None:
        seen.append(msg)

    client.set_notification_handler(handler)
    notification = {
        "jsonrpc": "2.0",
        "method": "receive",
        "params": {"envelope": {"sourceNumber": "+15035086292"}},
    }
    await client._dispatch_frame(notification)

    assert seen == [notification]


@pytest.mark.asyncio
async def test_notification_without_handler_is_silently_ignored():
    client = _make_client()
    # No handler set
    await client._dispatch_frame(
        {"jsonrpc": "2.0", "method": "receive", "params": {}}
    )
    # No exception is the assertion.


@pytest.mark.asyncio
async def test_dispatch_frame_with_unknown_id_does_not_raise():
    """Late response to a cancelled / forgotten request must not crash."""
    client = _make_client()
    await client._dispatch_frame(
        {"jsonrpc": "2.0", "id": 999, "result": "stale"}
    )
    # No state should change.
    assert client._pending == {}


@pytest.mark.asyncio
async def test_handler_exception_does_not_propagate():
    """A handler that raises must not poison the read loop."""
    client = _make_client()

    async def boom(msg: dict) -> None:
        raise RuntimeError("intentional test failure")

    client.set_notification_handler(boom)
    # Should swallow the exception.
    await client._dispatch_frame(
        {"jsonrpc": "2.0", "method": "receive", "params": {}}
    )


@pytest.mark.asyncio
async def test_connect_failure_raises_connection_error():
    """Bogus socket path → ConnectionError, not a bare OSError."""
    client = SignalClient(
        socket_path="/tmp/nonexistent-signal-cli-socket-xyz",
        account="+15039858519",
    )
    with pytest.raises(ConnectionError):
        await client.connect()


@pytest.mark.asyncio
async def test_send_text_without_connection_raises_connection_error():
    client = _make_client()
    # No connect() was called.
    ok = await client.send_text("+15551234567", "hi")
    # send_text catches ConnectionError and returns False so the
    # dispatcher's error path stays uniform.
    assert ok is False


@pytest.mark.asyncio
async def test_attachment_download_returns_none_for_missing_file():
    client = SignalClient(
        socket_path="/dev/null",
        account="+15039858519",
        attachments_dir="/tmp/nonexistent-signal-attachments-dir-xyz",
    )
    assert await client.download_media("any-id") is None


def test_phone_normalisation_adds_leading_plus_for_e164():
    """Stored phones (digits only) get the leading + for signal-cli."""
    from boxbot.communication.signal_client import _to_e164
    assert _to_e164("15035086292") == "+15035086292"
    assert _to_e164("+15035086292") == "+15035086292"
    assert _to_e164("  15035086292  ") == "+15035086292"


class _FakeReader:
    """StreamReader stand-in driven by an in-process queue of frames."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[bytes] = asyncio.Queue()

    def feed(self, line: bytes) -> None:
        self._q.put_nowait(line)

    async def readline(self) -> bytes:
        return await self._q.get()


class _FakeWriter:
    def __init__(self, on_write) -> None:
        self._on_write = on_write

    def write(self, data: bytes) -> None:
        self._on_write(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


@pytest.mark.asyncio
async def test_reconnect_handler_runs_concurrently_without_deadlock():
    """Regression: re-subscribe on reconnect must not deadlock the reader.

    The reconnect handler issues a JSON-RPC request (``subscribeReceive``)
    and awaits the daemon's reply — but only ``_read_loop`` can read and
    dispatch that reply. If the loop awaits the handler inline, the
    response is never read and the request times out, leaving the
    reconnected client subscribed to nothing (the bug that left Signal
    deaf after a daemon restart). The handler must run concurrently.
    """
    client = _make_client()

    reader = _FakeReader()

    def on_write(data: bytes) -> None:
        # Daemon side: answer subscribeReceive so the in-flight request
        # can resolve once the read loop reads this frame.
        msg = json.loads(data.decode())
        if msg.get("method") == "subscribeReceive":
            reader.feed(
                (json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": 99}) + "\n").encode()
            )

    writer = _FakeWriter(on_write)

    async def fake_open() -> None:
        client._reader = reader
        client._writer = writer

    client._open = fake_open  # type: ignore[assignment]

    subscribed = asyncio.Event()
    sub_ids: list[int] = []

    async def reconnect_handler() -> None:
        sub_ids.append(await client.subscribe_receive())
        subscribed.set()

    client.set_reconnect_handler(reconnect_handler)

    # Force the open branch so the loop reconnects and arms the handler.
    client._reader = None
    task = asyncio.create_task(client._read_loop())

    # Inline-await would deadlock here and never set the event.
    await asyncio.wait_for(subscribed.wait(), timeout=2.0)
    assert sub_ids == [99]

    client._stopped.set()
    reader.feed(b"")  # EOF unblocks readline so the loop can exit
    await asyncio.wait_for(task, timeout=2.0)
