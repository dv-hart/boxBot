"""boxBot tools — always-loaded capabilities for the agent.

Public API:
    Tool        — base class for all tools
    get_tools   — list of all tools for agent init
    get_tool    — look up a tool by name
"""

from boxbot.tools.base import Tool
from boxbot.tools.registry import get_tool, get_tools

__all__ = ["Tool", "get_tool", "get_tools"]
