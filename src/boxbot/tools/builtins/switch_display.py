"""switch_display tool — change the active display on the 7" screen.

Thin dispatcher that tells the display manager to switch to a named display,
passing through any display-specific args. The tool does not interpret args —
the display module handles its own arguments.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from boxbot.tools.base import Tool

logger = logging.getLogger(__name__)


class SwitchDisplayTool(Tool):
    """Change the active display on the screen."""

    name = "switch_display"
    description = (
        "Change what's shown on the 7-inch screen. Pass the display name and "
        "optional display-specific arguments. For example: "
        "switch_display('picture', args={}) for slideshow mode, "
        "switch_display('picture', args={'image_ids': ['abc', 'def']}) for "
        "specific photos, switch_display('weather') for weather display."
    )
    parameters = {
        "type": "object",
        "properties": {
            "display_name": {
                "type": "string",
                "description": "Name of the display to activate.",
            },
            "args": {
                "type": "object",
                "description": (
                    "Optional display-specific arguments passed through to "
                    "the display's render context. Each display defines what "
                    "args it accepts."
                ),
            },
        },
        "required": ["display_name"],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: Any) -> str:
        display_name: str = kwargs["display_name"]
        args: dict[str, Any] = kwargs.get("args") or {}

        logger.info(
            "switch_display: %s (args=%s)", display_name, list(args.keys())
        )

        # Stub: In production, this tells the display manager to switch
        # to the named display. The display manager validates the name
        # against registered displays and passes args through.

        # Stub list of available displays — in production, populated from
        # the display manager's registry.
        available_displays = [
            "clock",
            "weather",
            "calendar",
            "picture",
            "tasks",
            "status",
        ]

        return json.dumps({
            "status": "switched",
            "display_name": display_name,
            "args": args,
            "available_displays": available_displays,
            "message": f"Display switched to '{display_name}'.",
        })
