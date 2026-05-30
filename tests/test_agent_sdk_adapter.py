"""Tests for the Claude Agent SDK adapter.

Covers the shim between boxBot's tool registry and the SDK's in-process
MCP server: tool wrapping, result-shape coercion, ContextVar binding,
and options-builder field wiring.

Skipped wholesale when ``claude_agent_sdk`` is not importable so the
suite still passes on environments that haven't yet picked up the new
dependency.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("claude_agent_sdk")

from boxbot.core import agent_sdk_adapter as A
from boxbot.tools._tool_context import current_conversation
from boxbot.tools.base import Tool


class _StubTool(Tool):
    name = "stub"
    description = "Echoes its kwargs."
    parameters = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }

    def __init__(self, *, returns: Any = None, raises: Exception | None = None):
        self._returns = returns
        self._raises = raises
        self.calls: list[dict[str, Any]] = []
        self.observed_conv = None

    async def execute(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        # Capture the ContextVar value as observed inside the tool —
        # this is what production tools rely on.
        self.observed_conv = current_conversation.get()
        if self._raises is not None:
            raise self._raises
        return self._returns


def test_mcp_tool_name_namespaces_correctly():
    assert A.mcp_tool_name("message") == "mcp__boxbot_tools__message"
    assert A.mcp_tool_name("search_memory") == "mcp__boxbot_tools__search_memory"


def test_flatten_system_prompt_joins_blocks_with_blank_line():
    blocks = [
        {"type": "text", "text": "STATIC persona"},
        {"type": "text", "text": "DYNAMIC context"},
    ]
    assert A.flatten_system_prompt(blocks) == "STATIC persona\n\nDYNAMIC context"


def test_flatten_system_prompt_skips_non_text_blocks():
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": "first"},
        {"type": "image", "source": {}},
        {"type": "text", "text": "second"},
    ]
    # Image block has no "text" key — should be silently dropped.
    assert A.flatten_system_prompt(blocks) == "first\n\nsecond"


def test_wrap_tool_preserves_name_description_and_schema():
    tool = _StubTool()
    wrapped = A.wrap_tool(tool)

    assert wrapped.name == "stub"
    assert wrapped.description == "Echoes its kwargs."
    assert wrapped.input_schema is tool.parameters


@pytest.mark.asyncio
async def test_wrap_tool_returns_string_wraps_as_text_block():
    tool = _StubTool(returns="hello")
    wrapped = A.wrap_tool(tool)

    result = await wrapped.handler({"q": "hi"})

    assert tool.calls == [{"q": "hi"}]
    assert result == {
        "content": [{"type": "text", "text": "hello"}],
    }


@pytest.mark.asyncio
async def test_wrap_tool_returns_list_passes_through_as_content_blocks():
    """execute_script and identify_person return multimodal content."""
    multimodal = [
        {"type": "text", "text": "{\"ok\": true}"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "Zm9v",
            },
        },
    ]
    tool = _StubTool(returns=multimodal)
    wrapped = A.wrap_tool(tool)

    result = await wrapped.handler({"q": "snapshot"})

    assert result == {"content": multimodal}


@pytest.mark.asyncio
async def test_wrap_tool_returns_error_envelope_when_tool_raises():
    tool = _StubTool(raises=ValueError("tool exploded"))
    wrapped = A.wrap_tool(tool)

    result = await wrapped.handler({"q": "boom"})

    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "tool exploded" in text


@pytest.mark.asyncio
async def test_wrap_tool_sets_current_conversation_context_var():
    sentinel_conv = object()
    tool = _StubTool(returns="ok")
    wrapped = A.wrap_tool(tool, conv=sentinel_conv)

    await wrapped.handler({"q": "x"})

    assert tool.observed_conv is sentinel_conv


@pytest.mark.asyncio
async def test_wrap_tool_resets_context_var_after_exception():
    tool = _StubTool(raises=RuntimeError("boom"))
    wrapped = A.wrap_tool(tool, conv="my_conv")

    # The wrapper itself should not raise — it returns the error envelope.
    result = await wrapped.handler({"q": "x"})
    assert result["isError"] is True

    # After the wrapped call returns, the ContextVar should be restored
    # to its prior value (None on a clean test). If reset() didn't fire
    # in the except path, the sentinel would leak.
    assert current_conversation.get() is None


def test_build_mcp_server_registers_under_namespace():
    server = A.build_mcp_server([_StubTool()])
    # McpSdkServerConfig is a TypedDict — exposed as a dict at runtime.
    assert isinstance(server, dict)
    assert server.get("name") == A.MCP_SERVER_NAME or server.get("type") == "sdk"


def test_build_options_pins_output_format_and_allowed_tools():
    schema = {"type": "json_schema", "schema": {"type": "object"}}
    tools = [_StubTool()]

    opts = A.build_options(
        model="claude-opus-4-7-20260415",
        max_turns=17,
        system_prompt="you are bb",
        tools=tools,
        output_format=schema,
    )

    assert opts.model == "claude-opus-4-7-20260415"
    assert opts.max_turns == 17
    assert opts.system_prompt == "you are bb"
    assert opts.output_format == schema
    assert opts.allowed_tools == ["mcp__boxbot_tools__stub"]
    assert A.MCP_SERVER_NAME in opts.mcp_servers


def test_build_options_threads_can_use_tool_callback():
    async def gate(*args, **kwargs):
        from claude_agent_sdk import PermissionResultAllow
        return PermissionResultAllow(behavior="allow")

    opts = A.build_options(
        model="m",
        max_turns=3,
        system_prompt="sp",
        tools=[_StubTool()],
        output_format=None,
        can_use_tool=gate,
    )

    assert opts.can_use_tool is gate
