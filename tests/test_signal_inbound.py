"""Unit tests for signal-cli notification parsing + routing.

These don't talk to a real daemon — they feed canned JSON-RPC frames
into SignalInbound's handler and assert that MessageRouter sees the
expected ``route_incoming`` calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from boxbot.communication.channels import Channel
from boxbot.communication.signal_inbound import SignalInbound


def _make_inbound() -> tuple[SignalInbound, MagicMock]:
    router = MagicMock()
    router.route_incoming = AsyncMock(return_value=True)
    client = MagicMock()
    client.account = "+15039858519"
    client.set_notification_handler = MagicMock()
    inbound = SignalInbound(router=router, client=client)
    return inbound, router


def _text_notification(
    *,
    sender: str = "+15035086292",
    name: str = "Jacob",
    text: str = "hello",
    timestamp: int = 1780780876665,
) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "receive",
        "params": {
            "envelope": {
                "source": sender,
                "sourceNumber": sender,
                "sourceName": name,
                "sourceDevice": 1,
                "timestamp": timestamp,
                "dataMessage": {
                    "timestamp": timestamp,
                    "message": text,
                },
            },
            "account": "+15039858519",
            "subscription": 1,
        },
    }


@pytest.mark.asyncio
async def test_text_message_routes_through_router():
    inbound, router = _make_inbound()
    await inbound._on_notification(_text_notification())
    router.route_incoming.assert_awaited_once()
    args, kwargs = router.route_incoming.call_args
    assert args[0] == Channel.SIGNAL
    assert args[1] == "+15035086292"
    assert args[2] == "hello"
    assert kwargs["sender_name"] == "Jacob"
    assert kwargs["media_url"] is None
    assert kwargs["media_type"] is None
    assert kwargs["message_id"] == "1780780876665"


@pytest.mark.asyncio
async def test_image_attachment_normalises_to_image_category():
    inbound, router = _make_inbound()
    msg = _text_notification(text="caption")
    msg["params"]["envelope"]["dataMessage"]["attachments"] = [
        {
            "contentType": "image/jpeg",
            "id": "attach-abc-123",
            "filename": "IMG.jpg",
            "size": 12345,
        }
    ]
    await inbound._on_notification(msg)
    _, kwargs = router.route_incoming.call_args
    assert kwargs["media_url"] == "attach-abc-123"
    assert kwargs["media_type"] == "image"


@pytest.mark.asyncio
async def test_audio_attachment_normalises_to_audio_category():
    inbound, router = _make_inbound()
    msg = _text_notification(text="")
    msg["params"]["envelope"]["dataMessage"]["attachments"] = [
        {"contentType": "audio/m4a", "id": "voice-1", "size": 5000}
    ]
    await inbound._on_notification(msg)
    _, kwargs = router.route_incoming.call_args
    assert kwargs["media_type"] == "audio"


@pytest.mark.asyncio
async def test_pdf_attachment_categorised_as_document():
    inbound, router = _make_inbound()
    msg = _text_notification(text="")
    msg["params"]["envelope"]["dataMessage"]["attachments"] = [
        {"contentType": "application/pdf", "id": "pdf-1", "size": 100000}
    ]
    await inbound._on_notification(msg)
    _, kwargs = router.route_incoming.call_args
    assert kwargs["media_type"] == "document"


@pytest.mark.asyncio
async def test_unknown_mime_falls_back_to_document():
    inbound, router = _make_inbound()
    msg = _text_notification(text="")
    msg["params"]["envelope"]["dataMessage"]["attachments"] = [
        {"contentType": "application/x-weird", "id": "x-1", "size": 1}
    ]
    await inbound._on_notification(msg)
    _, kwargs = router.route_incoming.call_args
    assert kwargs["media_type"] == "document"


@pytest.mark.asyncio
async def test_typing_indicator_is_ignored():
    """Envelopes with no dataMessage (typing, receipts, sync) drop silently."""
    inbound, router = _make_inbound()
    msg = {
        "jsonrpc": "2.0",
        "method": "receive",
        "params": {
            "envelope": {
                "source": "+15035086292",
                "sourceNumber": "+15035086292",
                "timestamp": 1234,
                "typingMessage": {"action": "STARTED"},
            },
            "account": "+15039858519",
        },
    }
    await inbound._on_notification(msg)
    router.route_incoming.assert_not_awaited()


@pytest.mark.asyncio
async def test_receipt_message_is_ignored():
    inbound, router = _make_inbound()
    msg = {
        "jsonrpc": "2.0",
        "method": "receive",
        "params": {
            "envelope": {
                "source": "+15035086292",
                "sourceNumber": "+15035086292",
                "timestamp": 1234,
                "receiptMessage": {"when": 1234, "type": "DELIVERY"},
            },
            "account": "+15039858519",
        },
    }
    await inbound._on_notification(msg)
    router.route_incoming.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_envelope_deduped_by_sender_timestamp():
    """Same (sender, timestamp) tuple delivered twice → routed once."""
    inbound, router = _make_inbound()
    msg = _text_notification(timestamp=42)
    await inbound._on_notification(msg)
    await inbound._on_notification(msg)
    assert router.route_incoming.await_count == 1


@pytest.mark.asyncio
async def test_distinct_timestamps_not_deduped():
    inbound, router = _make_inbound()
    await inbound._on_notification(_text_notification(timestamp=1, text="a"))
    await inbound._on_notification(_text_notification(timestamp=2, text="b"))
    assert router.route_incoming.await_count == 2


@pytest.mark.asyncio
async def test_uuid_only_sender_dropped_silently():
    """Notifications without a phone-shaped source skip routing."""
    inbound, router = _make_inbound()
    msg = _text_notification()
    msg["params"]["envelope"].pop("sourceNumber")
    msg["params"]["envelope"]["source"] = "abc-uuid-without-plus"
    await inbound._on_notification(msg)
    router.route_incoming.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_receive_method_ignored():
    """Notifications with a different method (e.g. control frames) drop silently."""
    inbound, router = _make_inbound()
    msg = {"jsonrpc": "2.0", "method": "control", "params": {}}
    await inbound._on_notification(msg)
    router.route_incoming.assert_not_awaited()
