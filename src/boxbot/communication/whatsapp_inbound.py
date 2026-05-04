"""SQS-backed inbound WhatsApp poller.

The cloud relay (API Gateway + Lambda) verifies Meta webhook signatures
and forwards raw payloads onto an SQS queue. This module long-polls that
queue from the Pi, parses each payload via the existing
``WhatsAppWebhook.parse_webhook``, and routes through ``MessageRouter`` —
the same path the (formerly hypothetical) on-Pi webhook server would
have used. Reachability without an open inbound port comes from the Pi
making outbound long-poll requests to SQS rather than Meta reaching it.
"""

from __future__ import annotations

import asyncio
import json
import logging

import aioboto3

from boxbot.communication.router import Channel, MessageRouter
from boxbot.communication.whatsapp import IncomingMessage, WhatsAppWebhook

logger = logging.getLogger(__name__)


class WhatsAppInboundPoller:
    def __init__(
        self,
        *,
        router: MessageRouter,
        queue_url: str,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        wait_seconds: int = 20,
    ) -> None:
        self._router = router
        self._queue_url = queue_url
        self._region = region
        self._key = access_key_id
        self._secret = secret_access_key
        self._wait = wait_seconds
        self._task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="whatsapp-sqs-poller")
        logger.info("WhatsApp SQS poller started: %s", self._queue_url)

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=self._wait + 5)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        session = aioboto3.Session(
            aws_access_key_id=self._key,
            aws_secret_access_key=self._secret,
            region_name=self._region,
        )
        async with session.client("sqs") as sqs:
            while not self._stopped.is_set():
                try:
                    resp = await sqs.receive_message(
                        QueueUrl=self._queue_url,
                        MaxNumberOfMessages=10,
                        WaitTimeSeconds=self._wait,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("SQS receive failed; backing off")
                    try:
                        await asyncio.wait_for(self._stopped.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        pass
                    continue

                for raw in resp.get("Messages") or []:
                    await self._handle_message(sqs, raw)

        logger.info("WhatsApp SQS poller stopped")

    async def _handle_message(self, sqs, raw: dict) -> None:
        body = raw.get("Body", "") or ""
        receipt = raw.get("ReceiptHandle", "")
        try:
            messages = self._parse(body)
        except Exception:
            logger.exception("Discarding unparseable SQS message")
            await self._delete(sqs, receipt)
            return

        try:
            for m in messages:
                await self._route(m)
        except Exception:
            # Leave on the queue; SQS visibility timeout will redeliver.
            logger.exception("Failed to route message; will retry")
            return

        await self._delete(sqs, receipt)

    @staticmethod
    def _parse(body: str) -> list[IncomingMessage]:
        payload = json.loads(body)
        return WhatsAppWebhook.parse_webhook(payload)

    async def _route(self, m: IncomingMessage) -> None:
        await self._router.route_incoming(
            Channel.WHATSAPP,
            m.sender_phone,
            m.message_text,
            media_url=m.media_url,
            media_type=m.media_type,
            sender_name=m.sender_name or None,
            message_id=m.message_id,
        )

    async def _delete(self, sqs, receipt: str) -> None:
        if not receipt:
            return
        try:
            await sqs.delete_message(QueueUrl=self._queue_url, ReceiptHandle=receipt)
        except Exception:
            logger.exception("SQS delete_message failed")
