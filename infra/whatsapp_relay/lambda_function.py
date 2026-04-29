"""WhatsApp webhook relay Lambda.

Sits at a public Lambda Function URL. Meta calls it with:
  GET  — webhook verification handshake (hub.mode/hub.verify_token/hub.challenge)
  POST — incoming message payloads, signed with X-Hub-Signature-256

For valid POSTs, the raw JSON body is forwarded to SQS so the Pi can
poll over the public internet without exposing an inbound port.

Env vars (set in Lambda config):
  VERIFY_TOKEN  — must match WHATSAPP_VERIFY_TOKEN on the Pi
  APP_SECRET    — must match WHATSAPP_APP_SECRET on the Pi
  QUEUE_URL     — SQS queue URL to enqueue raw payloads
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "")
APP_SECRET = os.environ.get("APP_SECRET", "")
QUEUE_URL = os.environ["QUEUE_URL"]

_sqs = boto3.client("sqs")


def _resp(status: int, body: str = "", content_type: str = "text/plain") -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": content_type},
        "body": body,
    }


def _raw_body(event: dict) -> bytes:
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        return base64.b64decode(body)
    return body.encode("utf-8")


def _verify_signature(raw: bytes, header: str | None) -> bool:
    if not header or not header.startswith("sha256="):
        return False
    expected = header[len("sha256=") :]
    computed = hmac.new(APP_SECRET.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, computed)


def handler(event: dict, _context) -> dict:
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or "GET"
    ).upper()

    if method == "GET":
        params = event.get("queryStringParameters") or {}
        mode = params.get("hub.mode")
        token = params.get("hub.verify_token")
        challenge = params.get("hub.challenge", "")
        if not VERIFY_TOKEN:
            logger.warning("VERIFY_TOKEN env var not configured")
            return _resp(503, "verify token not configured")
        if mode == "subscribe" and hmac.compare_digest(token or "", VERIFY_TOKEN):
            logger.info("Webhook verification OK")
            return _resp(200, challenge)
        logger.warning("Webhook verification failed: mode=%s", mode)
        return _resp(403, "forbidden")

    if method == "POST":
        raw = _raw_body(event)
        headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
        sig = headers.get("x-hub-signature-256")
        if not APP_SECRET:
            logger.error("APP_SECRET env var not configured")
            return _resp(503, "app secret not configured")
        if not _verify_signature(raw, sig):
            logger.warning("Signature validation failed")
            return _resp(403, "forbidden")

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.exception("Invalid JSON body")
            return _resp(400, "invalid json")

        # Only enqueue if it actually contains messages — drop status/read receipts
        # cheaply at the edge to avoid waking the Pi for nothing.
        has_messages = False
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                if change.get("value", {}).get("messages"):
                    has_messages = True
                    break
            if has_messages:
                break

        if has_messages:
            _sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=raw.decode("utf-8"))
            logger.info("Forwarded payload to SQS")
        else:
            logger.info("Webhook had no messages (status update); skipped SQS")

        # Always 200 — Meta retries aggressively on non-2xx
        return _resp(200, "ok")

    return _resp(405, "method not allowed")
