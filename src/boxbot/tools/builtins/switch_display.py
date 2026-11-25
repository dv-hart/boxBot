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
        "switch_display('weather')\n"
        "switch_display('web', args={'url': 'https://…'}) — YouTube video "
        "(autoplays) or any public https page, in a browser over the "
        "screen. Has sound: pauses while you speak or listen, so on "
        "voice it mutes the mic (wake word re-engages). Closes itself "
        "10 s after a video ends or after 5 idle min (args.idle_minutes "
        "to change); close now: switch to another display. No touch "
        "input exists — never tell anyone to tap the screen; control "
        "playback with bb.display.web_media."
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
        if not ok:
            return json.dumps({
                "status": "error",
                "error": getattr(mgr, "last_switch_error", None)
                or f"could not switch to '{display_name}'",
                "display_name": display_name,
                "args": args,
            })

        result: dict[str, Any] = {
            "status": "ok",
            "display_name": display_name,
            "args": args,
            "pinned": mgr.is_pinned(),
            "available_displays": available,
        }
        web = mgr.get_web_state()
        if web is not None:
            result["web"] = web
            result["mic_muted"] = _mute_mic_for_playback()
        return json.dumps(result)


def _mute_mic_for_playback() -> bool:
    """Close the mic on the voice conversation that opened the web view.

    The view plays only while the mic is closed (BB does not listen
    over a video), so leaving the mic open would hold the page paused
    until the conversation timed out. Same mechanism as the
    ``mute_mic`` tool: the wake word stays armed and re-engages, which
    pauses playback. The display manager is told, so BB's own speech
    does not reopen the mic while the view is up.
    """
    from boxbot.communication.voice import get_voice_session
    from boxbot.tools._tool_context import get_current_conversation

    conv = get_current_conversation()
    if conv is None or conv.channel != "voice":
        return False
    session = get_voice_session()
    if session is None or not session.mute_mic():
        return False
    from boxbot.displays.manager import get_display_manager

    mgr = get_display_manager()
    if mgr is not None:
        mgr.note_web_mic_muted()
    return True
