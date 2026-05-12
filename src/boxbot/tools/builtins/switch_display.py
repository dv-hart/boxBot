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
        "Change what's shown on the 7-inch screen. By default the display "
        "is pinned — it stays until you call switch_display again or "
        "release the pin via bb.display.unpin(). Idle rotation is paused "
        "while pinned. Examples: "
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
            "pin": {
                "type": "boolean",
                "description": (
                    "Pin this display so idle rotation does not switch "
                    "away. Default true. Use bb.display.unpin() to "
                    "release."
                ),
            },
        },
        "required": ["display_name"],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: Any) -> str:
        display_name: str = kwargs["display_name"]
        args: dict[str, Any] = kwargs.get("args") or {}
        pin: bool = kwargs.get("pin", True)

        logger.info(
            "switch_display: %s (args=%s, pin=%s)",
            display_name, list(args.keys()), pin,
        )

        from boxbot.displays.manager import get_display_manager

        mgr = get_display_manager()
        if mgr is None:
            # Display subsystem isn't running (dev, tests, startup race).
            # Report rather than pretend success.
            return json.dumps({
                "status": "error",
                "error": "display manager not running",
                "display_name": display_name,
                "args": args,
            })

        available = mgr.list_available() if hasattr(mgr, "list_available") else []
        if available and display_name not in available:
            return json.dumps({
                "status": "error",
                "error": f"unknown display '{display_name}'",
                "available_displays": available,
            })

        ok = await mgr.switch(display_name, args=args, pin=pin)
        return json.dumps({
            "status": "ok" if ok else "error",
            "display_name": display_name,
            "args": args,
            "pinned": pin and ok,
            "available_displays": available,
        })
