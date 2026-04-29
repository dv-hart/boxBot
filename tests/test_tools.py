"""Tests for the tool system — base class, registry, and builtin tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from boxbot.tools.base import Tool
from boxbot.tools._sandbox_actions import ActionContext, process_action
from boxbot.tools.builtins.execute_script import (
    SDK_ACTION_MARKER,
    ExecuteScriptTool,
)
from boxbot.tools.builtins.manage_tasks import ManageTasksTool
from boxbot.tools.builtins.search_memory import SearchMemoryTool
from boxbot.tools.builtins.search_photos import SearchPhotosTool
from boxbot.tools.builtins.web_search import (
    WebSearchTool,
    _html_to_text,
    _parse_small_agent_response,
)


# ---------------------------------------------------------------------------
# Tool base class
# ---------------------------------------------------------------------------


class TestToolBaseClass:
    """Test the Tool ABC contract."""

    def test_tool_is_abstract(self):
        """Cannot instantiate Tool directly."""
        with pytest.raises(TypeError):
            Tool()  # type: ignore[abstract]

    def test_concrete_tool_has_required_attributes(self):
        tool = ExecuteScriptTool()
        assert isinstance(tool.name, str)
        assert len(tool.name) > 0
        assert isinstance(tool.description, str)
        assert isinstance(tool.parameters, dict)

    def test_to_schema_returns_valid_structure(self):
        tool = ExecuteScriptTool()
        schema = tool.to_schema()
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
        assert schema["name"] == "execute_script"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    """Test tool discovery and registry functions."""

    def test_get_tools_count_and_composition(self):
        """9 tools. All outbound speech/text to humans flows through the
        ``message`` tool (channel='speak' or 'text'). The other tools DO
        things — they don't speak. The legacy ``speak`` and ``send_message``
        files remain on disk for reference but are not registered."""
        # Reset the singleton to force fresh load
        import boxbot.tools.registry as reg
        reg._tools = None
        reg._tools_by_name = None

        from boxbot.tools.registry import get_tools
        tools = get_tools()
        assert len(tools) == 9
        names = {t.name for t in tools}
        assert "message" in names  # the only path to a human
        assert "speak" not in names  # subsumed by message(channel='speak')
        assert "send_message" not in names  # subsumed by message(channel='text')
        assert "load_skill" in names

    def test_get_tools_returns_tool_instances(self):
        from boxbot.tools.registry import get_tools
        tools = get_tools()
        for tool in tools:
            assert isinstance(tool, Tool)

    def test_get_tool_by_name(self):
        import boxbot.tools.registry as reg
        reg._tools = None
        reg._tools_by_name = None

        from boxbot.tools.registry import get_tool
        tool = get_tool("execute_script")
        assert tool is not None
        assert tool.name == "execute_script"

    def test_get_tool_nonexistent_returns_none(self):
        from boxbot.tools.registry import get_tool
        assert get_tool("nonexistent_tool") is None

    def test_all_expected_tool_names_present(self):
        import boxbot.tools.registry as reg
        reg._tools = None
        reg._tools_by_name = None

        from boxbot.tools.registry import get_tools
        names = {t.name for t in get_tools()}
        expected = {
            "message",
            "execute_script",
            "switch_display",
            "identify_person",
            "manage_tasks",
            "search_memory",
            "search_photos",
            "web_search",
            "load_skill",
        }
        assert names == expected

    def test_each_tool_has_unique_name(self):
        from boxbot.tools.registry import get_tools
        tools = get_tools()
        names = [t.name for t in tools]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# ExecuteScriptTool
# ---------------------------------------------------------------------------


class TestExecuteScriptTool:
    """Test the execute_script tool behaviour.

    End-to-end subprocess flow (streaming IO, sandbox action dispatch,
    image attachment) is covered by tests/test_workspace.py which runs
    the real subprocess pipeline. These tests cover schema + the
    action dispatcher in isolation.
    """

    def test_sdk_action_marker_constant(self):
        # Marker matches the one emitted by boxbot_sdk._transport so the
        # stream reader can locate JSON payloads unambiguously.
        assert SDK_ACTION_MARKER == "__BOXBOT_SDK_ACTION__:"

    def test_schema(self):
        tool = ExecuteScriptTool()
        schema = tool.to_schema()
        assert schema["name"] == "execute_script"
        assert "script" in schema["parameters"]["properties"]
        assert "description" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["script", "description"]

    @pytest.mark.asyncio
    async def test_process_action_unknown_returns_stub(self):
        ctx = ActionContext()
        result = await process_action(
            {"_sdk": "memory.save", "content": "test"}, ctx
        )
        # memory.* not yet wired to a real handler, so the dispatcher
        # acknowledges without failing — keeps the sandbox un-blocked.
        assert result["status"] == "stub"
        assert ctx.action_log[-1]["action"] == "memory.save"

    @pytest.mark.asyncio
    async def test_process_action_workspace_routes_correctly(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "agent" / "workspace").mkdir(parents=True)

        ctx = ActionContext()
        result = await process_action(
            {"_sdk": "workspace.write", "path": "a.md", "content": "hi"},
            ctx,
        )
        assert result["status"] == "ok"
        assert result["kind"] == "text"


# ---------------------------------------------------------------------------
# ManageTasksTool
# ---------------------------------------------------------------------------


class TestManageTasksTool:
    """Test the manage_tasks tool routing."""

    def test_tool_name_and_schema(self):
        tool = ManageTasksTool()
        assert tool.name == "manage_tasks"
        schema = tool.to_schema()
        assert "parameters" in schema
        props = schema["parameters"]["properties"]
        assert "action" in props

    @pytest.mark.asyncio
    async def test_create_trigger_action(self, tmp_path):
        """Test that create_trigger action routes to the scheduler."""
        tool = ManageTasksTool()
        with patch("boxbot.core.scheduler.DB_PATH", tmp_path / "sched.db"):
            result_json = await tool.execute(
                action="create_trigger",
                description="Morning check",
                instructions="Check weather",
            )
            result = json.loads(result_json)
            assert "trigger_id" in result or "id" in result or "t_" in result_json

    @pytest.mark.asyncio
    async def test_create_todo_action(self, tmp_path):
        tool = ManageTasksTool()
        with patch("boxbot.core.scheduler.DB_PATH", tmp_path / "sched.db"):
            result_json = await tool.execute(
                action="create_todo",
                description="Buy milk",
            )
            result = json.loads(result_json)
            assert "todo_id" in result or "id" in result or "d_" in result_json


# ---------------------------------------------------------------------------
# SearchMemoryTool
# ---------------------------------------------------------------------------


class TestSearchMemoryTool:
    """Test the search_memory tool."""

    def test_tool_name_and_modes(self):
        tool = SearchMemoryTool()
        assert tool.name == "search_memory"
        props = tool.parameters["properties"]
        assert "mode" in props
        assert set(props["mode"]["enum"]) == {"lookup", "summary", "get"}


# ---------------------------------------------------------------------------
# SearchPhotosTool
# ---------------------------------------------------------------------------


class TestSearchPhotosTool:
    """Test the search_photos tool."""

    def test_tool_name_and_modes(self):
        tool = SearchPhotosTool()
        assert tool.name == "search_photos"
        props = tool.parameters["properties"]
        assert "mode" in props
        assert set(props["mode"]["enum"]) == {"search", "get"}

    @pytest.mark.asyncio
    async def test_handles_missing_backend_gracefully(self):
        tool = SearchPhotosTool()
        with patch(
            "boxbot.tools.builtins.search_photos.SearchPhotosTool._search_via_backend",
            side_effect=ImportError("Not available"),
        ):
            result_json = await tool.execute(mode="search", query="sunset")
            result = json.loads(result_json)
            assert "error" in result


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    """Test the web_search tool."""

    def test_tool_name_and_params(self):
        tool = WebSearchTool()
        assert tool.name == "web_search"
        props = tool.parameters["properties"]
        assert "query" in props

    def test_parse_small_agent_response_extracts_sources(self):
        text = (
            "Here is the answer.\n\n"
            "SOURCES:\n"
            "- Example Site: https://example.com\n"
            "- Another: https://other.com\n"
        )
        result = _parse_small_agent_response(text)
        assert "summary" in result
        assert "sources" in result
        assert len(result["sources"]) >= 1

    def test_parse_small_agent_response_no_sources(self):
        text = "Just a plain answer without any sources section."
        result = _parse_small_agent_response(text)
        assert result["summary"] == text.strip()
        assert result["sources"] == []

    def test_html_to_text_strips_tags(self):
        html = "<html><body><h1>Title</h1><p>Content</p></body></html>"
        text = _html_to_text(html)
        assert "Title" in text
        assert "Content" in text
        assert "<h1>" not in text
