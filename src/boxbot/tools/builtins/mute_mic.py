"""mute_mic tool — drop further mic input on the current voice conversation.

Use when the agent wants to ignore the room without ending the
conversation: garbled / empty / unrelated transcripts arriving while a
task is in flight, ambient side-conversation, etc. The wake word stays
armed throughout — anyone in the room can re-engage by saying it.

The mute persists until one of:
- The agent emits ``message(channel="speak")`` — the speech handler
  unmutes at TTS-end before re-attaching audio_capture. The agent
  speaking implies it is engaging the room and wants follow-up.
- The conversation ends naturally (silence_timeout, etc.). Teardown
  resets audio_capture state.
- The wake word fires and starts a fresh session.

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
        "Stop mic input on the current voice conversation (no effect on "
        "text or trigger conversations). The wake word stays armed, so the "
        "room can re-engage you by saying it; the mute auto-clears on your "
        "next message(channel=\"speak\") or when the conversation ends. It "
        "also drops utterances queued while you were thinking.\n"
        "If you are interrupted by background noise or unrelated speech "
        "from the mic, call mute_mic immediately.\n"
        "Do not pair mute_mic with message(channel=\"speak\") in the same "
        "turn — speaking re-opens the mic and cancels the mute."
    )
    parameters = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": (
                    "Short free-form note on why you're muting. Recorded "
                    "for memory extraction and debugging. Examples: "
                    "\"ambient chatter\", \"focusing on calendar lookup\", "
                    "\"unrelated interruption\"."
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
                "Microphone muted. Will auto-unmute on your next "
                "message(speak), or when the conversation ends."
            ),
        })
