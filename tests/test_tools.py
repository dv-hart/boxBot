"""Tests for the tool system — base class, registry, and builtin tools."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from boxbot.core.output_dispatcher import BUDGET_SPENT
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
        """10 tools. All outbound speech/text to humans flows through the
        ``message`` tool (channel='speak' or 'text'). The other tools DO
        things — they don't speak. The legacy ``speak`` and ``send_message``
        files remain on disk for reference but are not registered."""
        # Reset the singleton to force fresh load
        import boxbot.tools.registry as reg
        reg._tools = None
        reg._tools_by_name = None

        from boxbot.tools.registry import get_tools
        tools = get_tools()
        assert len(tools) == 10
        names = {t.name for t in tools}
        assert "message" in names  # the only path to a human
        assert "speak" not in names  # subsumed by message(channel='speak')
        assert "send_message" not in names  # subsumed by message(channel='text')
        assert "load_skill" in names
        assert "mute_mic" in names

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
            "mute_mic",
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
    async def test_process_action_unknown_returns_error(self):
        ctx = ActionContext()
        result = await process_action(
            {"_sdk": "bogus.verb", "x": 1}, ctx
        )
        # No handler registered for prefix 'bogus' — the dispatcher
        # returns an error so the sandbox sees the failure rather than
        # a silent acknowledgement that masks unimplemented features.
        assert result["status"] == "error"
        assert "bogus" in result["message"]
        assert ctx.action_log[-1]["action"] == "bogus.verb"

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
# MessageTool
# ---------------------------------------------------------------------------


class TestMessageTool:
    """The message tool must report the dispatcher's *real* outcome.

    Regression: it used to return ``status: delivered`` unconditionally,
    so when the agent addressed a message to an unresolvable recipient
    (its own name, "Jarvis") the drop was invisible — it "delivered"
    three messages that nobody received.
    """

    def test_result_json_delivered(self):
        from boxbot.core.output_dispatcher import DispatchResult
        from boxbot.tools.builtins.message import MessageTool

        out = json.loads(MessageTool._result_json(
            [DispatchResult(to="Jacob", channel="text", status="delivered")],
            "Jacob", "text",
        ))
        assert out == {"status": "delivered", "to": "Jacob", "channel": "text"}

    def test_result_json_unknown_recipient_surfaces_valid_names(self):
        from boxbot.core.output_dispatcher import DispatchResult
        from boxbot.tools.builtins.message import MessageTool

        dropped = DispatchResult(
            to="Jarvis", channel="text", status="dropped",
            reason=(
                "unknown recipient 'Jarvis'. Valid recipients: Jacob, "
                "Carina. Use one of those names, or 'current_speaker'."
            ),
            valid_recipients=["Jacob", "Carina"],
        )
        out = json.loads(MessageTool._result_json(dropped and [dropped],
                                                  "Jarvis", "text"))
        assert out["status"] == "error"
        assert "Jarvis" in out["message"]
        assert out["valid_recipients"] == ["Jacob", "Carina"]

    def test_result_json_empty_results(self):
        from boxbot.tools.builtins.message import MessageTool

        out = json.loads(MessageTool._result_json([], "Jacob", "text"))
        assert out["status"] == "error"

    @pytest.mark.asyncio
    async def test_execute_returns_dispatcher_drop(self):
        """End to end: a dropped delivery becomes a tool-level error."""
        from boxbot.core.output_dispatcher import DispatchResult
        from boxbot.tools.builtins.message import MessageTool

        drop = DispatchResult(
            to="Jarvis", channel="text", status="dropped",
            reason="unknown recipient 'Jarvis'. Valid recipients: Jacob.",
            valid_recipients=["Jacob"],
        )
        with patch(
            "boxbot.core.output_dispatcher.dispatch_outputs",
            new=AsyncMock(return_value=[drop]),
        ), patch(
            "boxbot.tools._tool_context.get_current_conversation",
            return_value=None,
        ):
            result = json.loads(await MessageTool().execute(
                to="Jarvis", channel="text", content="the display is ready",
            ))
        assert result["status"] == "error"
        assert result["valid_recipients"] == ["Jacob"]


class TestTriggerMessageBudget:
    """A wake cycle nobody is waiting on gets a fixed number of
    deliveries. Past it the tool refuses and points at ``final_turn``.
    """

    @staticmethod
    def _conv(channel: str):
        return SimpleNamespace(
            conversation_id="conv-1", channel=channel, participants=set(),
            record_segment=lambda _seg: None, delivered_messages=0,
        )

    async def _send(self, conv, results):
        from boxbot.tools.builtins.message import MessageTool

        with patch(
            "boxbot.core.output_dispatcher.dispatch_outputs",
            new=AsyncMock(return_value=results),
        ), patch(
            "boxbot.tools._tool_context.get_current_conversation",
            return_value=conv,
        ):
            return json.loads(await MessageTool().execute(
                to="Jacob", channel="text", content="Bins go out tonight.",
            ))

    @pytest.mark.asyncio
    async def test_third_trigger_message_is_refused(self, mock_config):
        from boxbot.core.output_dispatcher import DispatchResult

        mock_config.agent.max_messages_trigger = 2
        conv = self._conv("trigger")
        ok = [DispatchResult(to="Jacob", channel="text", status="delivered")]

        assert (await self._send(conv, ok))["status"] == "delivered"
        assert (await self._send(conv, ok))["status"] == "delivered"
        third = await self._send(conv, ok)

        assert third["status"] == "error"
        assert "final_turn=true" in third["message"]
        assert conv.delivered_messages == 2
        # Tagged unretryable: every further call is refused identically, so
        # the trigger backstop must be able to end the run on it.
        assert third["reason_code"] == BUDGET_SPENT

    @pytest.mark.asyncio
    async def test_dropped_messages_do_not_spend_the_budget(self, mock_config):
        from boxbot.core.output_dispatcher import DispatchResult

        mock_config.agent.max_messages_trigger = 2
        conv = self._conv("trigger")
        dropped = [DispatchResult(
            to="Jacob", channel="text", status="dropped",
            reason="content contains tool-call syntax; not delivered",
        )]

        for _ in range(3):
            assert (await self._send(conv, dropped))["status"] == "error"
        assert conv.delivered_messages == 0

    @pytest.mark.asyncio
    async def test_interactive_channels_are_not_budgeted(self, mock_config):
        from boxbot.core.output_dispatcher import DispatchResult

        conv = self._conv("signal")
        ok = [DispatchResult(to="Jacob", channel="text", status="delivered")]

        for _ in range(5):
            assert (await self._send(conv, ok))["status"] == "delivered"
        assert conv.delivered_messages == 5


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
        assert set(props["mode"]["enum"]) == {
            "lookup", "summary", "get", "transcript",
        }
        assert "conversation_id" in props


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


class TestMessageAttachments:
    """Attachment paths are resolved against the sandbox/workspace roots
    and must pass the image-attach allowlist before they reach a phone."""

    @staticmethod
    def _png(path):
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), (0, 128, 128)).save(path, format="PNG")
        return path

    @pytest.fixture
    def roots(self, tmp_path, monkeypatch):
        import boxbot.tools._sandbox_actions as actions

        tmp_dir = tmp_path / "sandbox" / "tmp"
        tmp_dir.mkdir(parents=True)
        monkeypatch.setattr(actions, "_sandbox_tmp_dir", lambda: tmp_dir)
        monkeypatch.setattr(actions, "_attach_roots", lambda: (tmp_dir.resolve(),))
        monkeypatch.setattr(
            "boxbot.tools.builtins.message._attachment_bases",
            lambda: [tmp_path / "sandbox", tmp_dir],
        )
        return tmp_path, tmp_dir

    def test_relative_sandbox_path_resolves(self, roots):
        from boxbot.tools.builtins.message import _resolve_attachments

        tmp_path, tmp_dir = roots
        self._png(tmp_dir / "camera_abc.jpg")
        paths, err = _resolve_attachments(["tmp/camera_abc.jpg"])
        assert err is None
        assert paths == [str((tmp_dir / "camera_abc.jpg").resolve())]

    def test_outside_allowlist_is_refused(self, roots, tmp_path):
        from boxbot.tools.builtins.message import _resolve_attachments

        stray = self._png(tmp_path / "elsewhere" / "x.png")
        paths, err = _resolve_attachments([str(stray)])
        assert paths == [] and "not in an allowed location" in err

    def test_non_image_is_refused(self, roots):
        from boxbot.tools.builtins.message import _resolve_attachments

        _tmp_path, tmp_dir = roots
        (tmp_dir / "notes.txt").write_text("hello")
        paths, err = _resolve_attachments(["notes.txt"])
        assert paths == [] and "not a recognised image" in err

    @pytest.mark.asyncio
    async def test_speak_channel_rejects_attachments(self, mock_config):
        from boxbot.tools.builtins.message import MessageTool

        result = json.loads(await MessageTool().execute(
            to="current_speaker", channel="speak", content="Look",
            attachments=["tmp/x.jpg"],
        ))
        assert result["status"] == "error"
        assert "channel \"text\"" in result["message"]
