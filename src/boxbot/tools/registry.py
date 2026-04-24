"""Tool registration and discovery.

Discovers and loads all tools from the builtins/ package. Provides
get_tools() for agent initialization and get_tool() for name-based lookup.

Usage:
    from boxbot.tools.registry import get_tools, get_tool

    tools = get_tools()           # list[Tool] for agent init
    tool = get_tool("speak")      # Tool | None
"""

from __future__ import annotations

import logging

from boxbot.tools.base import Tool
from boxbot.tools.builtins.execute_script import ExecuteScriptTool
from boxbot.tools.builtins.identify_person import IdentifyPersonTool
from boxbot.tools.builtins.load_skill import LoadSkillTool
from boxbot.tools.builtins.manage_tasks import ManageTasksTool
from boxbot.tools.builtins.search_memory import SearchMemoryTool
from boxbot.tools.builtins.search_photos import SearchPhotosTool
from boxbot.tools.builtins.switch_display import SwitchDisplayTool
from boxbot.tools.builtins.web_search import WebSearchTool

# All outgoing communication to humans flows through the structured
# ``outputs`` array of the agent's response (see ``output_dispatcher`` and
# ``agent._prompt_etiquette``). Each output entry names its recipient and
# channel explicitly (voice or text), and the dispatcher routes accordingly.
#
# ``speak`` and ``send_message`` were previously registered here but are
# now subsumed by that outputs path:
# - ``speak(text)``          → ``outputs[{to, channel="voice", content}]``
# - ``send_message(to,text)``→ ``outputs[{to, channel="text", content}]``
#
# The source files under ``builtins/`` remain on disk (for reference and
# potential future re-wiring), but they are not imported or registered.
# Tools register only actions that DO things, not speech.

logger = logging.getLogger(__name__)

# Singleton registry populated on first access
_tools: list[Tool] | None = None
_tools_by_name: dict[str, Tool] | None = None


def _load_tools() -> list[Tool]:
    """Instantiate all built-in tools.

    ``speak`` is intentionally absent — speech is produced via structured
    output (`response_text`) rather than a tool call. See the module
    docstring note at the top of this file.
    """
    return [
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
