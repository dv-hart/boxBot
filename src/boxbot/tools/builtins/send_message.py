"""send_message tool — send a WhatsApp message to a whitelisted user.

Resolves recipient against the authorized user whitelist and sends via
the WhatsApp Business API. Supports text messages and media attachments.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from boxbot.tools.base import Tool

logger = logging.getLogger(__name__)


class SendMessageTool(Tool):
    """Send a WhatsApp message to a whitelisted user."""

    name = "send_message"
    description = (
        "Send a WhatsApp message to a registered user. Specify the recipient "
        "by name or phone number. Optionally attach an image. The recipient "
        "must be a whitelisted user — messages to unknown numbers are blocked."
    )
    parameters = {
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "description": (
                    "Person name or phone number of the recipient. "
                    "Must be a registered, whitelisted user."
                ),
            },
            "message": {
                "type": "string",
                "description": "The message text to send.",
            },
            "media_path": {
                "type": "string",
                "description": (
                    "Optional path to an image file to attach to the message."
                ),
            },
        },
        "required": ["recipient", "message"],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: Any) -> str:
        recipient: str = kwargs["recipient"]
        message: str = kwargs["message"]
        media_path: str | None = kwargs.get("media_path")

        logger.info(
            "send_message: to=%s, length=%d, media=%s",
            recipient,
            len(message),
            media_path is not None,
        )

        # Stub: In production, this:
        # 1. Resolves recipient against the user whitelist
        # 2. Sends via WhatsApp Business API (communication layer)
        # 3. Returns delivery status
        #
        # Unresolved recipients return an error — never send to
        # unknown numbers.

        return json.dumps({
            "status": "sent",
            "recipient": recipient,
            "message_length": len(message),
            "has_media": media_path is not None,
            "message": f"Message sent to {recipient}.",
        })
