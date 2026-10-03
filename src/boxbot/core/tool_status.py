"""Map in-flight tool calls to short user-facing status lines.

While the agent works a turn, the only truthful progress signal is the
stream of tool calls it makes. ``status_text_for`` turns a tool call
into a short present-tense line ("Searching the web…"), and
``publish_tool_status`` wraps it as a best-effort ``AgentToolCalled``
publish. Both generation paths call the publisher immediately before
``tool.execute``; the display manager renders the line as a transient
pill over the active display.

The mapping is deliberately deterministic — no model involvement — so
the pill can never claim work that isn't happening.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Instant or self-evident on screen — no status pill.
_SILENT_TOOLS = frozenset({"message", "switch_display", "mute_mic"})

_TOOL_STATUS: dict[str, str] = {
    "web_search": "Searching the web…",
    "search_memory": "Recalling…",
    "search_photos": "Searching photos…",
    "manage_tasks": "Checking tasks…",
    "identify_person": "Recognizing speaker…",
    "load_skill": "Reading up…",
}

# execute_script: sniff the script source for bb-module usage. First
# match wins — ordered by how specific the module is, so a script that
# grabs a camera still *and* saves a note reads as a camera check.
# Matches both spellings the sandbox accepts (``bb.`` / ``boxbot_sdk.``).
_SCRIPT_MODULE_STATUS: tuple[tuple[str, str], ...] = (
    ("camera", "Checking the camera…"),
    ("integrations", "Fetching data…"),
    ("photos", "Searching photos…"),
    ("display", "Updating the screen…"),
    ("workspace", "Checking notes…"),
    ("memory", "Recalling…"),
    ("tasks", "Checking tasks…"),
)

_DEFAULT_SCRIPT_STATUS = "Working on it…"


def status_text_for(
    tool_name: str, tool_input: dict[str, Any] | None
) -> str | None:
    """Status line for a tool call, or None when no pill should show.

    Unknown tool names return None — new tools opt in by appearing in
    the tables above, so a missing entry degrades to no pill, never to
    a wrong one.
    """
    if tool_name in _SILENT_TOOLS:
        return None
    if tool_name == "execute_script":
        script = ""
        if isinstance(tool_input, dict):
            raw = tool_input.get("script")
            if isinstance(raw, str):
                script = raw
        for module, text in _SCRIPT_MODULE_STATUS:
            if f"bb.{module}" in script or f"boxbot_sdk.{module}" in script:
                return text
        return _DEFAULT_SCRIPT_STATUS
    return _TOOL_STATUS.get(tool_name)


async def publish_tool_status(
    conv: Any, tool_name: str, tool_input: dict[str, Any] | None
) -> None:
    """Publish ``AgentToolCalled`` for a tool dispatch. Never raises.

    ``conv`` is the owning ``Conversation`` (or None in tests /
    conversation-less contexts, in which case nothing is published).
    """
    if conv is None:
        return
    try:
        text = status_text_for(tool_name, tool_input)
        if text is None:
            return
        from boxbot.core.events import AgentToolCalled, get_event_bus

        await get_event_bus().publish(
            AgentToolCalled(
                conversation_id=conv.conversation_id,
                channel=conv.channel,
                tool_name=tool_name,
                status_text=text,
            )
        )
    except Exception:
        logger.debug(
            "AgentToolCalled publish failed for %s", tool_name, exc_info=True
        )
