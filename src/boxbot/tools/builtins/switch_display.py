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
        "Change the 7-inch screen. Pinned by default — holds until the "
        "next switch_display or bb.display.unpin(). Idle rotation pauses "
        "while pinned.\n"
        "switch_display('picture', args={}) — slideshow\n"
        "switch_display('picture', args={'image_ids': ['abc', 'def']}) — "
        "specific photos\n"
        "switch_display('weather')"
    )
    parameters = {
        "type": "object",
        "properties": {
            "display_name": {
                "type": "string",
                "description": "Display to activate.",
            },
            "args": {
                "type": "object",
                "description": (
                    "Passed to the display's render context. Each display "
                    "defines its own args."
                ),
            },
            "pin": {
                "type": "boolean",
                "description": (
                    "Block idle rotation. Default true. Release with "
                    "bb.display.unpin()."
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
