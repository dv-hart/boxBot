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
        "Stop receiving microphone input on the current voice conversation. "
        "The wake word stays armed, so the room can still re-engage you by "
        "saying it. The mute clears automatically the next time you call "
        "message(channel=\"speak\"), or when the conversation ends.\n\n"
        "## When to mute — be DECISIVE\n"
        "\n"
        "The single most important rule: **if you have an active task and "
        "the next utterance you see is unrelated to that task, mute on the "
        "SAME turn you saw the unrelated utterance**. Do not wait for a "
        "second confirming turn. Do not 'see what comes next.' An unrelated "
        "utterance almost always leads to more unrelated utterances — "
        "people in a room follow threads. The user gave you a task and "
        "started talking to their spouse / kids / a guest; that side "
        "conversation will go on for a while and you must not waste a "
        "turn on each fragment.\n"
        "\n"
        "Concrete triggers — call mute_mic IMMEDIATELY when:\n"
        "- You are mid-task (looking up calendar, reading workspace, "
        "  composing a reply, anything that took a tool call) AND a "
        "  transcript arrives that is not the user continuing or correcting "
        "  their request to you.\n"
        "- A speaker is clearly addressing someone else in the room (a "
        "  child, a spouse, a guest, the TV).\n"
        "- The transcript is garbled, empty, or fragmentary noise.\n"
        "- A different language appears mid-conversation that the original "
        "  speaker only uses for side talk.\n"
        "\n"
        "Muting does NOT interrupt your current work. Tools keep running, "
        "the API call you're inside finishes, and you can still call "
        "message later with the result. Mute is purely defensive — it "
        "stops future ambient turns from yanking you off course.\n"
        "\n"
        "## Default to muting when uncertain mid-task\n"
        "\n"
        "If you are mid-task and unsure whether an utterance is for you, "
        "mute. The wake word brings the user right back if they actually "
        "needed you. The cost of a wrong mute is one extra wake word; the "
        "cost of NOT muting is a derailed task, wasted turns, and a "
        "response to chatter the user didn't address to you.\n"
        "\n"
        "## Hard rule: never mute in the same turn as message(speak)\n"
        "\n"
        "Speaking automatically re-opens the mic at the end of TTS, so a "
        "mute paired with a speak cancels out. If you want to say one "
        "last thing and let the room go quiet, just call message(speak) "
        "and stop — the silence timer ends the conversation on its own. "
        "If you want to deliver an answer AND stay muted afterward, "
        "speak first; on your NEXT turn (after the user follows up or "
        "doesn't), mute then.\n"
        "\n"
        "Calling mute_mic also drops any utterances that were queued "
        "while you were thinking — that's the point, those queued lines "
        "are the same side-talk you are choosing to ignore.\n"
        "\n"
        "No effect on WhatsApp or trigger conversations."
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
