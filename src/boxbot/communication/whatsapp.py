"""WhatsApp Business API client and webhook handler.

Handles outbound messages (text, image) via the Meta Graph API and
inbound webhooks. All HTTP calls use httpx for async operation.

Security:
- Webhook payloads should be signature-verified before calling parse_webhook
- Unknown senders are handled upstream by auth.py (silent drop)
- Access token and phone number ID come from environment variables

Usage:
    from boxbot.communication.whatsapp import WhatsAppClient, WhatsAppWebhook

    client = WhatsAppClient(access_token="...", phone_number_id="...")
    await client.send_text("+15551234567", "Hello from boxBot!")

    webhook = WhatsAppWebhook(verify_token="my-verify-token", app_secret="my-app-secret")
    messages = webhook.parse_webhook(payload)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = "v21.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


# Singleton accessor — set by main during startup, read by the output dispatcher
# so the agent loop can reach the WhatsApp client without being passed a
# reference. Mirrors the voice session singleton pattern.
_whatsapp_client: "WhatsAppClient | None" = None


def get_whatsapp_client() -> "WhatsAppClient | None":
    """Return the process-wide WhatsAppClient instance, or None if unset."""
    return _whatsapp_client


def set_whatsapp_client(client: "WhatsAppClient | None") -> None:
    """Register the process-wide WhatsAppClient instance."""
    global _whatsapp_client
    _whatsapp_client = client


@dataclass(frozen=True)
class IncomingMessage:
    """A parsed incoming WhatsApp message."""

    sender_phone: str
    sender_name: str
    message_text: str | None = None
    media_url: str | None = None
    media_type: str | None = None  # "image", "audio", "document", "video"
    timestamp: str = ""
    message_id: str = ""


class WhatsAppClient:
    """Async client for the WhatsApp Business Cloud API.

    Sends text messages and images to registered users via the Meta
    Graph API. Uses httpx for non-blocking HTTP requests.

    Args:
        access_token: Meta Graph API access token.
        phone_number_id: The WhatsApp Business phone number ID.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        *,
        timeout: float = 30.0,
    ) -> None:
        self._access_token = access_token
        self._phone_number_id = phone_number_id
        self._timeout = timeout
        self._base_url = f"{GRAPH_API_BASE}/{phone_number_id}/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

    async def send_text(self, phone: str, message: str) -> bool:
        """Send a text message to a phone number.

        Args:
            phone: Recipient phone in E.164 format (e.g. "+15551234567").
            message: The text message body.

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": phone,
            "type": "text",
            "text": {"preview_url": False, "body": message},
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._base_url,
                    headers=self._headers(),
                    json=payload,
                )
                if response.status_code == 200:
                    logger.debug("Message sent to %s", phone)
                    return True
                logger.error(
                    "Failed to send message to %s: %d %s",
                    phone,
                    response.status_code,
                    response.text,
                )
                return False
        except httpx.HTTPError as e:
            logger.error("HTTP error sending message to %s: %s", phone, e)
            return False

    async def send_image(
        self,
        phone: str,
        image_url: str,
        caption: str | None = None,
    ) -> bool:
        """Send an image message to a phone number.

        Args:
            phone: Recipient phone in E.164 format.
            image_url: Public URL of the image to send.
            caption: Optional caption text.

        Returns:
            True if sent successfully, False otherwise.
        """
        image_obj: dict[str, str] = {"link": image_url}
        if caption:
            image_obj["caption"] = caption

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": phone,
            "type": "image",
            "image": image_obj,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._base_url,
                    headers=self._headers(),
                    json=payload,
                )
                if response.status_code == 200:
                    logger.debug("Image sent to %s", phone)
                    return True
                logger.error(
                    "Failed to send image to %s: %d %s",
                    phone,
                    response.status_code,
                    response.text,
                )
                return False
        except httpx.HTTPError as e:
            logger.error("HTTP error sending image to %s: %s", phone, e)
            return False

    async def upload_and_send_image(
        self,
        phone: str,
        image_path: str,
        caption: str | None = None,
    ) -> bool:
        """Upload a local image and send it to a phone number.

        Uploads the image to the WhatsApp media endpoint first, then
        sends a message referencing the uploaded media ID.

        Args:
            phone: Recipient phone in E.164 format.
            image_path: Local filesystem path to the image.
            caption: Optional caption text.

        Returns:
            True if sent successfully, False otherwise.
        """
        upload_url = f"{GRAPH_API_BASE}/{self._phone_number_id}/media"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                # Upload the image
                with open(image_path, "rb") as f:
                    upload_response = await client.post(
                        upload_url,
                        headers={"Authorization": f"Bearer {self._access_token}"},
                        data={"messaging_product": "whatsapp"},
                        files={"file": (image_path, f, "image/jpeg")},
                    )

                if upload_response.status_code != 200:
                    logger.error(
                        "Failed to upload image: %d %s",
                        upload_response.status_code,
                        upload_response.text,
                    )
                    return False

                media_id = upload_response.json().get("id")
                if not media_id:
                    logger.error("No media ID in upload response")
                    return False

                # Send using media ID
                image_obj: dict[str, str] = {"id": media_id}
                if caption:
                    image_obj["caption"] = caption

                payload = {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": phone,
                    "type": "image",
                    "image": image_obj,
                }

                response = await client.post(
                    self._base_url,
                    headers=self._headers(),
                    json=payload,
                )
                if response.status_code == 200:
                    logger.debug("Image uploaded and sent to %s", phone)
                    return True
                logger.error(
                    "Failed to send uploaded image to %s: %d %s",
                    phone,
                    response.status_code,
                    response.text,
                )
                return False
        except (httpx.HTTPError, OSError) as e:
            logger.error("Error uploading/sending image to %s: %s", phone, e)
            return False


class WhatsAppWebhook:
    """Handles incoming WhatsApp webhook verification and message parsing.

    Args:
        verify_token: The token configured in Meta Developer Portal for
                      webhook GET verification.
        app_secret: The app secret for webhook signature validation.
                    Required in production — validates X-Hub-Signature-256 headers.
    """

    def __init__(
        self,
        verify_token: str,
        app_secret: str,
    ) -> None:
        if not app_secret:
            raise ValueError(
                "WhatsApp app_secret is required for webhook signature "
                "validation. Set WHATSAPP_APP_SECRET in .env."
            )
        self._verify_token = verify_token
        self._app_secret = app_secret

    def verify_webhook(
        self,
        mode: str,
        token: str,
        challenge: str,
    ) -> str | None:
        """Verify a webhook subscription request (GET from Meta).

        Meta sends a GET request with hub.mode, hub.verify_token, and
        hub.challenge. We must return the challenge if the token matches.

        Args:
            mode: The hub.mode parameter (should be "subscribe").
            token: The hub.verify_token parameter.
            challenge: The hub.challenge parameter.

        Returns:
            The challenge string if verification succeeds, None otherwise.
        """
        if mode == "subscribe" and token == self._verify_token:
            logger.info("Webhook verified successfully")
            return challenge
        logger.warning("Webhook verification failed: mode=%s", mode)
        return None

    def validate_signature(
        self,
        payload_body: bytes,
        signature_header: str,
    ) -> bool:
        """Validate the X-Hub-Signature-256 header from Meta.

        Args:
            payload_body: The raw request body bytes.
            signature_header: The X-Hub-Signature-256 header value.

        Returns:
            True if the signature is valid, False otherwise.
        """
        if not signature_header:
            return False

        # Header format: "sha256=<hex_digest>"
        if not signature_header.startswith("sha256="):
            return False

        expected_sig = signature_header[7:]
        computed_sig = hmac.new(
            self._app_secret.encode("utf-8"),
            payload_body,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected_sig, computed_sig)

    def parse_webhook(self, payload: dict[str, Any]) -> list[IncomingMessage]:
        """Parse an incoming webhook payload into messages.

        The Meta webhook payload has a deeply nested structure:
        payload.entry[].changes[].value.messages[]

        Args:
            payload: The parsed JSON webhook body.

        Returns:
            List of IncomingMessage objects. Empty if the payload contains
            no messages (e.g., status updates).
        """
        messages: list[IncomingMessage] = []

        entries = payload.get("entry", [])
        for entry in entries:
            changes = entry.get("changes", [])
            for change in changes:
                value = change.get("value", {})
                if value.get("messaging_product") != "whatsapp":
                    continue

                # Build contact name lookup
                contacts = value.get("contacts", [])
                name_map: dict[str, str] = {}
                for contact in contacts:
                    wa_id = contact.get("wa_id", "")
                    profile = contact.get("profile", {})
                    name = profile.get("name", "")
                    if wa_id:
                        name_map[wa_id] = name

                raw_messages = value.get("messages", [])
                for msg in raw_messages:
                    sender_phone = msg.get("from", "")
                    sender_name = name_map.get(sender_phone, "")
                    msg_type = msg.get("type", "")
                    message_id = msg.get("id", "")
                    timestamp = msg.get("timestamp", "")

                    message_text = None
                    media_url = None
                    media_type = None

                    if msg_type == "text":
                        text_body = msg.get("text", {})
                        message_text = text_body.get("body", "")
                    elif msg_type in ("image", "audio", "video", "document"):
                        media_obj = msg.get(msg_type, {})
                        media_url = media_obj.get("id")  # Media ID, needs download
                        media_type = msg_type
                        # Images/videos may have captions
                        caption = media_obj.get("caption")
                        if caption:
                            message_text = caption

                    messages.append(
                        IncomingMessage(
                            sender_phone=sender_phone,
                            sender_name=sender_name,
                            message_text=message_text,
                            media_url=media_url,
                            media_type=media_type,
                            timestamp=timestamp,
                            message_id=message_id,
                        )
                    )

        return messages

    async def download_media(
        self,
        media_id: str,
        access_token: str,
    ) -> bytes | None:
        """Download media content by its WhatsApp media ID.

        Two-step process: first get the media URL, then download the content.

        Args:
            media_id: The WhatsApp media ID from the webhook payload.
            access_token: The Graph API access token.

        Returns:
            The raw media bytes, or None on failure.
        """
        headers = {"Authorization": f"Bearer {access_token}"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: Get the media URL
                url_response = await client.get(
                    f"{GRAPH_API_BASE}/{media_id}",
                    headers=headers,
                )
                if url_response.status_code != 200:
                    logger.error(
                        "Failed to get media URL for %s: %d",
                        media_id,
                        url_response.status_code,
                    )
                    return None

                media_url = url_response.json().get("url")
                if not media_url:
                    logger.error("No URL in media response for %s", media_id)
                    return None

                # Step 2: Download the media
                media_response = await client.get(media_url, headers=headers)
                if media_response.status_code != 200:
                    logger.error(
                        "Failed to download media from %s: %d",
                        media_url,
                        media_response.status_code,
                    )
                    return None

                return media_response.content
        except httpx.HTTPError as e:
            logger.error("HTTP error downloading media %s: %s", media_id, e)
            return None
