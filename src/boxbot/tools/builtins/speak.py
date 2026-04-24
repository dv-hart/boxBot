"""speak tool — say something through the speaker.

Real-time TTS output for the agent. Used for proactive speech when the
agent isn't in an active voice conversation (scheduled reminders, greetings,
etc.). Urgent priority interrupts current audio.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from boxbot.tools.base import Tool

logger = logging.getLogger(__name__)


class SpeakTool(Tool):
    """Say something through the speaker via TTS — for mid-turn interims."""

    name = "speak"
    description = (
        "Say something aloud IMMEDIATELY. Use this during a turn for brief "
        "interim acknowledgments while you work — 'One moment', 'Let me check', "
        "'Looking that up now'. Also use for proactive speech when not in an "
        "active voice conversation (scheduled reminders, greetings). "
        "\n\n"
        "DO NOT use speak() for the final answer to a question — put that in "
        "the `response_text` field of your end-of-turn decision so the system "
        "can route it correctly across channels and track it. Calling speak() "
        "with your final answer causes double-speaking and breaks the silent/"
        "respond gate. "
        "\n\n"
        "Urgent priority interrupts current audio; use sparingly."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to speak aloud.",
            },
            "priority": {
                "type": "string",
                "enum": ["normal", "urgent"],
                "description": (
                    "Priority level. 'urgent' interrupts current audio. "
                    "Default: normal."
                ),
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: Any) -> str:
        text: str = kwargs["text"]
        priority: str = kwargs.get("priority", "normal")

        logger.info("speak [%s]: %s", priority, text[:100])

        # Get the voice session singleton
        from boxbot.communication.voice import get_voice_session

        session = get_voice_session()

        if session is None:
            logger.warning("Voice pipeline not available for speak tool")
            return json.dumps({
                "status": "error",
                "message": "Voice pipeline not available.",
            })

        try:
            await session.speak(text, priority=priority)
            return json.dumps({
                "status": "spoken",
                "text": text,
                "priority": priority,
            })
        except Exception as e:
            logger.exception("speak tool failed")
            return json.dumps({
                "status": "error",
                "message": f"Speech failed: {e}",
            })
