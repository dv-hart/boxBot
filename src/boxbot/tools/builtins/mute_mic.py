"""mute_mic tool — drop further mic input on the current voice conversation.

For sustained talk that is clearly not meant for BB (TV, a long
side-conversation). The wake word stays armed throughout — anyone in
the room can re-engage by saying it.

Refused (``status: "not_muted"``, mic left open) when BB has already
spoken in the current turn: BB just addressed the room, so a person is
about to answer, and a muted mic would force them to say the wake word
to do it. The signal is ``Conversation.spoke_this_turn``.

The mute persists until one of:
- The agent emits ``message(channel="speak")`` — the speech handler
  unmutes at TTS-end before re-attaching audio_capture.
- The conversation ends naturally (silence_timeout, etc.). Teardown
  resets audio_capture state.
- The wake word fires (re-engages the session or starts a fresh one).

No-op on non-voice conversations (WhatsApp, trigger-fired) — there is
no live mic to mute there.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from boxbot.tools.base import Tool

logger = logging.getLogger(__name__)


class MuteMicTool(Tool):
    """Stop receiving room audio until BB next speaks or the conversation ends."""

    name = "mute_mic"
    description = (
        "Drop mic input on the current voice conversation. No effect on "
        "text or trigger conversations.\n"
        "Only for sustained talk clearly not meant for you: TV, people "
        "talking among themselves at length.\n"
        "Never after you asked something or a reply is plausible — they "
        "could not answer without the wake word. Never on one garbled "
        "fragment. Unsure: stay silent, do not mute; the mic sleeps on "
        "its own.\n"
        "Refused (status \"not_muted\") if you already spoke this turn.\n"
        "Muted: wake word stays armed. Clears when you next speak, on "
        "the wake word, or at conversation end. Also drops utterances "
        "queued while you thought."
    )
    parameters = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": (
                    "Why. Recorded for extraction and debugging. e.g. "
                    "\"TV audio\", \"long side-conversation\"."
                ),
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: Any) -> str:
        from boxbot.communication.voice import get_voice_session
        from boxbot.tools._tool_context import get_current_conversation

        reason = str(kwargs.get("reason") or "").strip()

        conv = get_current_conversation()
        if conv is None:
            logger.warning("mute_mic called outside a conversation context")
            return json.dumps({
                "status": "noop",
                "message": "no active conversation",
            })

        if conv.channel != "voice":
            return json.dumps({
                "status": "noop",
                "message": (
                    f"mute_mic is a no-op on '{conv.channel}' "
                    "conversations (no live mic)"
                ),
            })

        # BB spoke in this turn → someone is about to answer. Muting
        # now would make them say the wake word to reply (observed:
        # question asked, mic muted, room left talking to a dead mic).
        if getattr(conv, "spoke_this_turn", False):
            logger.info(
                "mute_mic: refused — BB already spoke this turn "
                "(conv=%s reason=%s)",
                conv.conversation_id, reason or "<none>",
            )
            return json.dumps({
                "status": "not_muted",
                "message": (
                    "Not muted: you spoke this turn, so a reply may be "
                    "coming. Mic stays open. Nothing else to do: end "
                    "the turn."
                ),
            })

        session = get_voice_session()
        if session is None:
            return json.dumps({
                "status": "noop",
                "message": "no active voice session",
            })

        muted = session.mute_mic()
        if not muted:
            return json.dumps({
                "status": "noop",
                "message": "voice session has no audio_capture attached",
            })

        # Drop any utterances that were queued while BB was speaking —
        # they're the same ambient input the agent is trying to ignore.
        try:
            conv._pending_inputs.clear()  # type: ignore[attr-defined]
        except AttributeError:
            pass

        logger.info(
            "mute_mic: muted voice conversation %s (reason=%s)",
            conv.conversation_id,
            reason or "<none>",
        )
        return json.dumps({
            "status": "muted",
            "reason": reason,
            "message": (
                "Mic muted; wake word still armed. Clears when you "
                "next speak, on the wake word, or at conversation end."
            ),
        })
