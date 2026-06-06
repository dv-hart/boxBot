"""Adapter between boxBot's tool registry and the ``claude_agent_sdk``.

This module is the single seam between the in-house ``Tool`` base class
and the Claude Agent SDK's in-process MCP server. It exists so the SDK
backend can be turned on without rewriting every tool's interface.

What this module provides:

* :func:`wrap_tool` — turns a boxBot ``Tool`` (which exposes
  ``execute(**kwargs) -> str | list[content_block]``) into an
  ``SdkMcpTool`` that the SDK can register with its MCP server. The
  wrapper optionally sets the ``current_conversation`` ContextVar so
  tools that need conversation-scoped state (``execute_script``,
  ``message``, ``identify_person``) find it — necessary because the
  SDK dispatches MCP tools from its own task tree where ContextVar
  values set in our agent loop don't propagate.
* :func:`build_mcp_server` — bundles a list of wrapped tools into a
  single in-process MCP server registered under the ``boxbot_tools``
  namespace. Pass ``conv`` to bind the server to a specific
  Conversation; pass ``None`` for a conversation-agnostic server
  (used by tests).
* :func:`build_options` — assembles a :class:`ClaudeAgentOptions`
  with all the levers the conversation loop relies on: pinned
  ``output_format`` (the ``INTERNAL_NOTES_SCHEMA`` private scratchpad),
  the MCP server registered under the right name, ``allowed_tools``
  scoped to ours, an optional ``can_use_tool`` gate for per-turn tool
  filtering (used for the final-turn ``message``-only restriction),
  and the chosen model + max_turns from config.
* :func:`flatten_system_prompt` — concatenates the existing two-block
  system prompt into the single string the SDK accepts. Cache markers
  are dropped (the SDK manages caching internally).
* :func:`mcp_tool_name` — canonical ``mcp__boxbot_tools__<name>`` form
  used for ``allowed_tools`` and for filtering in the ``can_use_tool``
  callback.

The module is import-safe even when ``claude_agent_sdk`` is not
installed in the venv — imports are deferred to runtime so the raw
backend stays operational on machines that haven't yet added the
SDK dependency.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        McpSdkServerConfig,
        SdkMcpTool,
    )

    from boxbot.tools.base import Tool

logger = logging.getLogger(__name__)


MCP_SERVER_NAME = "boxbot_tools"


def mcp_tool_name(tool_name: str) -> str:
    """Return the SDK-side fully-qualified name for ``tool_name``.

    The SDK exposes MCP tools to the model as
    ``mcp__<server>__<tool>``. We surface this so callers
    (``allowed_tools``, ``can_use_tool`` gate, telemetry filters) can
    work in the same namespace without duplicating the prefix string.
    """
    return f"mcp__{MCP_SERVER_NAME}__{tool_name}"


def base_tool_name(name: str) -> str:
    """Inverse of :func:`mcp_tool_name`: strip the ``mcp__<server>__``
    prefix so post-hoc thread scanners match a tool by its bare name
    regardless of which backend recorded it.

    The SDK backend records tool calls as ``mcp__boxbot_tools__message``;
    the legacy direct-API loop recorded the bare ``message``. Helpers that
    scan a persisted thread (delivery bridge, trigger receipt, transcript
    rendering) must accept both. Bare names pass through unchanged.
    """
    prefix = f"mcp__{MCP_SERVER_NAME}__"
    return name[len(prefix):] if name.startswith(prefix) else name


def _content_blocks(result: Any) -> list[dict[str, Any]]:
    """Coerce a ``Tool.execute`` return value into MCP content blocks.

    boxBot tools return one of:
      * ``str`` — wrap as a single text block.
      * ``list[dict]`` — already in Anthropic content-block shape
        (text + image blocks from ``execute_script`` /
        ``identify_person``). Pass through.
      * anything else — JSON-encode as a text block (defensive
        fallback; should not occur in practice).
    """
    if isinstance(result, str):
        return [{"type": "text", "text": result}]
    if isinstance(result, list):
        return result
    return [{"type": "text", "text": json.dumps(result)}]


def wrap_tool(tool: "Tool", conv: Any = None) -> "SdkMcpTool[Any]":
    """Wrap a boxBot ``Tool`` as an ``SdkMcpTool`` the SDK can serve.

    The wrapper preserves the tool's ``name``, ``description``, and
    JSON-schema ``parameters`` exactly so the model sees the same tool
    surface as on the raw Anthropic path. At execution time the wrapper
    sets the ``current_conversation`` ContextVar to ``conv`` (the
    Conversation this MCP server is bound to), then calls
    ``tool.execute(**args)`` and shapes the return as MCP-format
    ``{"content": [...]}``.

    Pass ``conv=None`` for a conversation-agnostic wrapper (used by
    tests). Production callers build a fresh MCP server per
    Conversation so tools like ``execute_script`` reach their
    long-lived sandbox runner without ambiguity.
    """
    from claude_agent_sdk import tool as sdk_tool

    @sdk_tool(tool.name, tool.description, tool.parameters)
    async def _wrapped(args: dict[str, Any]) -> dict[str, Any]:
        from boxbot.tools._tool_context import current_conversation

        token = current_conversation.set(conv)
        try:
            result = await tool.execute(**args)
        except Exception as exc:
            logger.exception(
                "Tool %r raised during SDK-dispatched execution", tool.name,
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"error": str(exc)}),
                    }
                ],
                "isError": True,
            }
        finally:
            current_conversation.reset(token)
        return {"content": _content_blocks(result)}

    return _wrapped


def build_mcp_server(
    tools: list["Tool"],
    conv: Any = None,
) -> "McpSdkServerConfig":
    """Build the in-process MCP server that fronts every boxBot tool.

    The SDK reaches every tool through this single server. Pass ``conv``
    so the wrapped tool handlers can set the ``current_conversation``
    ContextVar to that Conversation — necessary for ``execute_script``
    and any other tool that resolves conversation-scoped state.
    """
    from claude_agent_sdk import create_sdk_mcp_server

    wrapped = [wrap_tool(t, conv=conv) for t in tools]
    return create_sdk_mcp_server(
        name=MCP_SERVER_NAME,
        version="1.0.0",
        tools=wrapped,
    )


def flatten_system_prompt(blocks: list[dict[str, Any]]) -> str:
    """Flatten the existing two-block system prompt into a single string.

    The raw Anthropic path passes a list of text blocks with mixed
    ``cache_control`` markers; the SDK's ``system_prompt`` field is a
    plain string. We concatenate with a blank-line separator. The SDK
    caches the prefix automatically, so the static block — which is
    always identical for a given config — still hits the cache on the
    second and subsequent turns of a conversation.
    """
    parts: list[str] = []
    for block in blocks:
        text = block.get("text") if isinstance(block, dict) else None
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def build_options(
    *,
    model: str,
    max_turns: int,
    system_prompt: str,
    tools: list["Tool"],
    output_format: dict[str, Any] | None,
    conv: Any = None,
    can_use_tool: Callable[..., Awaitable[Any]] | None = None,
) -> "ClaudeAgentOptions":
    """Assemble the ``ClaudeAgentOptions`` for one conversation.

    Pins the structured-output ``output_format`` (the
    ``INTERNAL_NOTES_SCHEMA`` private-scratchpad contract), registers
    the in-process MCP server under :data:`MCP_SERVER_NAME`, restricts
    ``allowed_tools`` to that server's namespace so the model can't
    invoke built-in Claude Code tools, and threads through ``model``,
    ``max_turns``, and the optional per-call ``can_use_tool`` gate.

    Pass ``conv`` so wrapped tools receive the right Conversation in
    their ContextVar. Caller is responsible for owning the resulting
    ``ClaudeSDKClient`` lifecycle; one Conversation, one client.
    """
    from claude_agent_sdk import ClaudeAgentOptions

    mcp_server = build_mcp_server(tools, conv=conv)
    allowed = [mcp_tool_name(t.name) for t in tools]

    return ClaudeAgentOptions(
        model=model,
        max_turns=max_turns,
        system_prompt=system_prompt,
        mcp_servers={MCP_SERVER_NAME: mcp_server},
        allowed_tools=allowed,
        output_format=output_format,
        can_use_tool=can_use_tool,
    )


__all__ = [
    "MCP_SERVER_NAME",
    "base_tool_name",
    "build_mcp_server",
    "build_options",
    "flatten_system_prompt",
    "mcp_tool_name",
    "wrap_tool",
]
