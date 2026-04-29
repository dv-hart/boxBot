"""message tool — the ONLY way to reach a person.

The agent's text output is constrained to a private internal-notes JSON
shape (see ``output_dispatcher.INTERNAL_NOTES_SCHEMA``); nothing the
agent says in text reaches a human. To actually speak through the box
speaker or send a WhatsApp message, the agent calls this tool.

Multiple calls per turn are allowed and expected (e.g. an interim
"One moment, looking that up" alongside an ``execute_script`` call, or
a spoken acknowledgement to the room plus a text to an absent spouse).

Routing is delegated to ``output_dispatcher.dispatch_outputs`` so this
tool and trigger-fired turns share one delivery implementation.

Channel naming: the dispatcher's internal channel id is ``"voice"``,
but the tool exposes it to the agent as ``"speak"`` (a verb, parallel
with ``"text"``). The mapping is done here.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from boxbot.tools.base import Tool

logger = logging.getLogger(__name__)


# Tool-facing channel name → dispatcher channel id.
_CHANNEL_MAP = {
    "speak": "voice",
    "text": "text",
}


class MessageTool(Tool):
    """Send one message to one recipient through one channel."""

    name = "message"
    description = (
        "The ONLY way to reach a person. Call this when you want a human "
        "to actually hear or read what you have to say. Without a "
        "message call, the user gets silence — your text output is "
        "private notes, not speech.\n\n"
        "Call it as many times per turn as you need:\n"
        "- Interim acknowledgement before a tool call: "
        "  message(\"current_speaker\", \"speak\", \"Sure thing, let me find that.\") "
        "  alongside execute_script(...) in the same response.\n"
        "- Final answer after a tool returns.\n"
        "- Multi-recipient: speak to the room AND text to an absent user "
        "  in the same response (two calls).\n\n"
        "Recipient (`to`):\n"
        "- \"current_speaker\" — whoever just addressed you in this "
        "  conversation. Resolves to their name at dispatch time.\n"
        "- \"room\" — broadcast voice to anyone physically present. "
        "  Speak channel only.\n"
        "- A registered user's name exactly as it appears in the "
        "  Registered users list (e.g. \"Sarah\") — reach an absent person "
        "  via text, or address someone present by name via speak.\n\n"
        "Channel:\n"
        "- \"speak\" — speak through the box speaker (everyone in the "
        "  room hears). Good for replies in person and announcements.\n"
        "- \"text\" — send a WhatsApp message to the named user's "
        "  registered phone. Requires `to` be a registered user by name. "
        "  Cannot text \"room\" or unknown people.\n\n"
        "Be concise when speaking — no one wants a lecture from a box. "
        "Stay silent (do not call this tool) when people are talking to "
        "each other, not to you."
    )
    parameters = {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": (
                    "Recipient identifier. Use \"current_speaker\" for "
                    "whoever addressed you most recently, \"room\" to "
                    "broadcast spoken audio to anyone present, or a "
                    "registered user's name exactly as it appears in the "
                    "Registered users list."
                ),
            },
            "channel": {
                "type": "string",
                "enum": ["speak", "text"],
                "description": (
                    "Delivery medium. \"speak\" plays TTS through the box "
                    "speaker (everyone in the room hears). \"text\" sends "
                    "a WhatsApp message to the named user's registered "
                    "phone."
                ),
            },
            "content": {
                "type": "string",
                "description": "The exact words to deliver.",
            },
        },
        "required": ["to", "channel", "content"],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: Any) -> str:
        from boxbot.core.output_dispatcher import dispatch_outputs
        from boxbot.tools._tool_context import get_current_conversation

        to = str(kwargs.get("to", "")).strip()
        channel = str(kwargs.get("channel", "")).strip()
        content = str(kwargs.get("content", "")).strip()

        if not to or not channel or not content:
            return json.dumps({
                "status": "error",
                "message": (
                    "message requires non-empty to, channel, and content"
                ),
            })

        dispatcher_channel = _CHANNEL_MAP.get(channel)
        if dispatcher_channel is None:
            return json.dumps({
                "status": "error",
                "message": (
                    f"unknown channel '{channel}'; use 'speak' or 'text'"
                ),
            })

        conv = get_current_conversation()
        if conv is None:
            # Ad-hoc / out-of-conversation invocation (tests, triggers
            # before a Conversation wraps them). Fall back to a minimal
            # context so the dispatcher can still route.
            logger.warning(
                "message called outside a Conversation context; "
                "dispatching with no segment recorder. to=%s channel=%s",
                to, channel,
            )
            await dispatch_outputs(
                [{"to": to, "channel": dispatcher_channel, "content": content}],
                conversation_id="ad-hoc",
                channel_context="unknown",
                current_speaker=None,
                segment_recorder=None,
            )
            return json.dumps({
                "status": "delivered",
                "to": to,
                "channel": channel,
            })

        # Resolve current_speaker from the conversation's participants.
        # Mirrors how _publish_started picks primary_person.
        current_speaker: str | None = None
        for p in conv.participants:
            current_speaker = p
            break

        await dispatch_outputs(
            [{"to": to, "channel": dispatcher_channel, "content": content}],
            conversation_id=conv.conversation_id,
            channel_context=conv.channel,
            current_speaker=current_speaker,
            segment_recorder=conv.record_segment,
        )

        return json.dumps({
            "status": "delivered",
            "to": to,
            "channel": channel,
        })
