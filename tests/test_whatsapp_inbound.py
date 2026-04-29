"""Tests for the SQS-backed WhatsApp inbound poller."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from boxbot.communication.router import Channel
from boxbot.communication.whatsapp_inbound import WhatsAppInboundPoller


def _make_payload(phone: str = "+15551234567", text: str = "hi BB") -> str:
    return json.dumps(
        {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "ENTRY",
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "messaging_product": "whatsapp",
                                "contacts": [
                                    {"wa_id": phone, "profile": {"name": "Tester"}}
                                ],
                                "messages": [
                                    {
                                        "from": phone,
                                        "id": "wamid.X",
                                        "timestamp": "1700000000",
                                        "type": "text",
                                        "text": {"body": text},
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
        }
    )


def _build_poller(router) -> WhatsAppInboundPoller:
    return WhatsAppInboundPoller(
        router=router,
        queue_url="https://sqs.example/q",
        region="us-west-2",
        access_key_id="AK",
        secret_access_key="SK",
    )


@pytest.mark.asyncio
async def test_handle_message_routes_and_deletes():
    router = MagicMock()
    router.route_incoming = AsyncMock(return_value=True)
    poller = _build_poller(router)
    sqs = MagicMock()
    sqs.delete_message = AsyncMock()

    raw = {"Body": _make_payload(), "ReceiptHandle": "RH-1"}
    await poller._handle_message(sqs, raw)

    router.route_incoming.assert_awaited_once()
    args, kwargs = router.route_incoming.call_args
    assert args[0] is Channel.WHATSAPP
    assert args[1] == "+15551234567"
    assert args[2] == "hi BB"
    assert kwargs["sender_name"] == "Tester"

    sqs.delete_message.assert_awaited_once_with(
        QueueUrl="https://sqs.example/q", ReceiptHandle="RH-1"
    )


@pytest.mark.asyncio
async def test_handle_message_deletes_unparseable_payload():
    router = MagicMock()
    router.route_incoming = AsyncMock()
    poller = _build_poller(router)
    sqs = MagicMock()
    sqs.delete_message = AsyncMock()

    raw = {"Body": "not json", "ReceiptHandle": "RH-2"}
    await poller._handle_message(sqs, raw)

    router.route_incoming.assert_not_called()
    sqs.delete_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_message_keeps_on_route_failure():
    """Routing errors must NOT delete — SQS visibility timeout retries."""
    router = MagicMock()
    router.route_incoming = AsyncMock(side_effect=RuntimeError("boom"))
    poller = _build_poller(router)
    sqs = MagicMock()
    sqs.delete_message = AsyncMock()

    raw = {"Body": _make_payload(), "ReceiptHandle": "RH-3"}
    await poller._handle_message(sqs, raw)

    router.route_incoming.assert_awaited_once()
    sqs.delete_message.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_skips_payload_with_no_messages():
    """Status updates / read receipts have empty messages — no-op delete."""
    router = MagicMock()
    router.route_incoming = AsyncMock()
    poller = _build_poller(router)
    sqs = MagicMock()
    sqs.delete_message = AsyncMock()

    payload = json.dumps(
        {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messaging_product": "whatsapp",
                                "statuses": [{"id": "wamid.x", "status": "delivered"}],
                            }
                        }
                    ]
                }
            ]
        }
    )
    raw = {"Body": payload, "ReceiptHandle": "RH-4"}
    await poller._handle_message(sqs, raw)

    router.route_incoming.assert_not_called()
    sqs.delete_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_when_never_started():
    poller = _build_poller(MagicMock())
    # Should not raise even though .start() was never called.
    await poller.stop()
