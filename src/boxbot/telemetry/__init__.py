"""Lightweight, best-effort observability writes.

This package holds append-only telemetry that grounds the prefetch
layer: a per-invocation ``tool_invocations`` log (what the agent
actually searched/loaded, per turn) and, later, ``prefetch_events``.
All writes are best-effort — callers wrap them in try/except so a
telemetry failure can never break a live turn.

The tables live in the memory database (``memory.db``) alongside
``cost_log`` so they share one connection lifecycle and join cleanly on
``conversation_id``. See ``boxbot.cost.record`` for the sibling pattern.
"""

from boxbot.telemetry.tool_log import ToolInvocation, record_tool_invocation

__all__ = ["ToolInvocation", "record_tool_invocation"]
