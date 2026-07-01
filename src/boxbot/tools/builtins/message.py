"""message tool — the ONLY way to reach a person.

The agent's text output is constrained to a private internal-notes JSON
shape (see ``output_dispatcher.INTERNAL_NOTES_SCHEMA``); nothing the
agent says in text reaches a human. To actually speak through the box
speaker or send a text message, the agent calls this tool.

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
        "The ONLY way to reach a human; your text output is private "
        "notes, not speech. Call as many times per turn as needed (e.g. "
        "an interim acknowledgement before a tool call, then the final "
        "answer; multiple recipients = multiple calls).\n"
        "`to`: \"current_speaker\" (whoever just addressed you), \"room\" "
        "(broadcast spoken audio, speak channel only), or a registered "
        "user's exact name.\n"
        "`channel`: \"speak\" (box speaker, everyone in the room hears) or "
        "\"text\" (text a registered user by name — cannot text \"room\" "
        "or unknown people).\n"
        "Default to the channel you were contacted through. Be concise, "
        "and stay silent (don't call this) when people are talking to "
        "each other, not to you."
    )
    parameters = {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": (
                    "\"current_speaker\" (whoever just addressed you), "
                    "\"room\" (spoken broadcast), or a registered user's "
                    "exact name."
                ),
            },
            "channel": {
                "type": "string",
                "enum": ["speak", "text"],
                "description": (
                    "\"speak\" (box speaker) or \"text\" (text a "
                    "registered user)."
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

    @staticmethod
    def _result_json(results: list[Any], to: str, channel: str) -> str:
        """Turn the dispatcher's per-entry result into a tool result.

        ``message`` always dispatches exactly one entry, so we inspect
        ``results[0]``. A dropped delivery comes back as a ``status:
        error`` the agent can act on (e.g. retry with a valid recipient
        name) — never a false ``delivered``.
        """
        result = results[0] if results else None
        if result is not None and result.status == "delivered":
            return json.dumps({
                "status": "delivered", "to": to, "channel": channel,
            })
        payload: dict[str, Any] = {
            "status": "error",
            "message": (
                result.reason if result and result.reason
                else "message could not be delivered"
            ),
        }
        if result is not None and result.valid_recipients is not None:
            payload["valid_recipients"] = result.valid_recipients
        return json.dumps(payload)

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
            results = await dispatch_outputs(
                [{"to": to, "channel": dispatcher_channel, "content": content}],
                conversation_id="ad-hoc",
                channel_context="unknown",
                current_speaker=None,
                segment_recorder=None,
            )
            return self._result_json(results, to, channel)

        # Resolve current_speaker from the conversation's participants.
        # Mirrors how _publish_started picks primary_person.
        current_speaker: str | None = None
        for p in conv.participants:
            current_speaker = p
            break

        results = await dispatch_outputs(
            [{"to": to, "channel": dispatcher_channel, "content": content}],
            conversation_id=conv.conversation_id,
            channel_context=conv.channel,
            current_speaker=current_speaker,
            segment_recorder=conv.record_segment,
        )

        return self._result_json(results, to, channel)
