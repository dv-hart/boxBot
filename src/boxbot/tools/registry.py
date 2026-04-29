"""Tool registration and discovery.

Discovers and loads all tools from the builtins/ package. Provides
get_tools() for agent initialization and get_tool() for name-based lookup.

Usage:
    from boxbot.tools.registry import get_tools, get_tool

    tools = get_tools()                       # list[Tool] for agent init
    tool = get_tool("message")                # Tool | None
"""

from __future__ import annotations

import logging

from boxbot.tools.base import Tool
from boxbot.tools.builtins.message import MessageTool
from boxbot.tools.builtins.execute_script import ExecuteScriptTool
from boxbot.tools.builtins.identify_person import IdentifyPersonTool
from boxbot.tools.builtins.load_skill import LoadSkillTool
from boxbot.tools.builtins.manage_tasks import ManageTasksTool
from boxbot.tools.builtins.search_memory import SearchMemoryTool
from boxbot.tools.builtins.search_photos import SearchPhotosTool
from boxbot.tools.builtins.switch_display import SwitchDisplayTool
from boxbot.tools.builtins.web_search import WebSearchTool

# All outgoing communication to humans flows through the ``message``
# tool. The agent's text output is constrained to a private
# internal-notes JSON shape (see
# ``output_dispatcher.INTERNAL_NOTES_SCHEMA``) — by construction nothing
# in the model's text reaches a person. To speak or text, the model must
# call ``message(to, channel, content)`` where channel is "speak" or
# "text". Multiple calls per turn are allowed and expected (interim
# acknowledgements alongside ``execute_script``, multi-recipient
# deliveries, etc.).
#
# ``speak`` and ``send_message`` files remain on disk for reference but
# are not imported or registered.

logger = logging.getLogger(__name__)

# Singleton registry populated on first access
_tools: list[Tool] | None = None
_tools_by_name: dict[str, Tool] | None = None


def _load_tools() -> list[Tool]:
    """Instantiate all built-in tools."""
    return [
        MessageTool(),
        ExecuteScriptTool(),
        SwitchDisplayTool(),
        IdentifyPersonTool(),
        ManageTasksTool(),
        SearchMemoryTool(),
        SearchPhotosTool(),
        WebSearchTool(),
        LoadSkillTool(),
    ]


def _ensure_loaded() -> None:
    """Ensure tools are loaded into the singleton registry."""
    global _tools, _tools_by_name
    if _tools is None:
        _tools = _load_tools()
        _tools_by_name = {t.name: t for t in _tools}
        logger.info(
            "Loaded %d tools: %s",
            len(_tools),
            ", ".join(t.name for t in _tools),
        )


def get_tools() -> list[Tool]:
    """Return the full list of tools for agent initialization."""
    _ensure_loaded()
    assert _tools is not None
    return list(_tools)


def get_tool(name: str) -> Tool | None:
    """Look up a tool by name. Returns None if not found."""
    _ensure_loaded()
    assert _tools_by_name is not None
    return _tools_by_name.get(name)
