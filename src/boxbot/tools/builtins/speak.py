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
    """Say something through the speaker via TTS."""

    name = "speak"
    description = (
        "Say something out loud through the speaker. Use this for proactive "
        "output when you aren't in an active voice conversation — scheduled "
        "reminders, greetings when someone is detected, etc. Urgent priority "
        "interrupts current audio."
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

        # Stub: In production, this sends text to the TTS engine
        # (ElevenLabs streaming) and plays through the speaker via HAL.
        # Urgent priority interrupts any current audio playback.

        return json.dumps({
            "status": "spoken",
            "text": text,
            "priority": priority,
            "message": "Speech output queued successfully.",
        })
