"""Tests for the tool system — base class, registry, and builtin tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from boxbot.tools.base import Tool
from boxbot.tools.builtins.execute_script import (
    SAFE_ENV_VARS,
    SDK_ACTION_MARKER,
    ExecuteScriptTool,
    _process_sdk_action,
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

    def test_get_tools_returns_nine_tools(self):
        # Reset the singleton to force fresh load
        import boxbot.tools.registry as reg
        reg._tools = None
        reg._tools_by_name = None

        from boxbot.tools.registry import get_tools
        tools = get_tools()
        assert len(tools) == 9

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
            "execute_script",
            "speak",
            "switch_display",
            "send_message",
            "identify_person",
            "manage_tasks",
            "search_memory",
            "search_photos",
            "web_search",
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
    """Test the execute_script tool behavior."""

    def test_sdk_action_marker_constant(self):
        assert SDK_ACTION_MARKER == "__BOXBOT_SDK_ACTION__"

    @pytest.mark.asyncio
    async def test_execute_captures_stdout(self, tmp_path):
        tool = ExecuteScriptTool()
        with patch("boxbot.tools.builtins.execute_script.SCRIPTS_DIR", tmp_path / "scripts"):
            with patch("boxbot.tools.builtins.execute_script.OUTPUT_DIR", tmp_path / "output"):
                with patch("boxbot.tools.builtins.execute_script.get_config", side_effect=RuntimeError):
                    # Use the system python since sandbox venv doesn't exist
                    import sys
                    venv_python = sys.executable
                    with patch(
                        "asyncio.create_subprocess_exec",
                        new_callable=AsyncMock,
                    ) as mock_proc:
                        proc_mock = AsyncMock()
                        proc_mock.communicate.return_value = (b"Hello World\n", b"")
                        proc_mock.returncode = 0
                        mock_proc.return_value = proc_mock

                        result_json = await tool.execute(
                            script='print("Hello World")',
                            description="Test print",
                        )
                        result = json.loads(result_json)
                        assert result["status"] == "success"
                        assert result["output"] == "Hello World"

    @pytest.mark.asyncio
    async def test_execute_parses_sdk_actions(self, tmp_path):
        tool = ExecuteScriptTool()
        sdk_line = f'{SDK_ACTION_MARKER}{{"action":"memory.save","content":"test"}}'
        stdout = f"regular output\n{sdk_line}\nmore output\n"

        with patch("boxbot.tools.builtins.execute_script.SCRIPTS_DIR", tmp_path / "scripts"):
            with patch("boxbot.tools.builtins.execute_script.OUTPUT_DIR", tmp_path / "output"):
                with patch("boxbot.tools.builtins.execute_script.get_config", side_effect=RuntimeError):
                    with patch(
                        "asyncio.create_subprocess_exec",
                        new_callable=AsyncMock,
                    ) as mock_proc:
                        proc_mock = AsyncMock()
                        proc_mock.communicate.return_value = (stdout.encode(), b"")
                        proc_mock.returncode = 0
                        mock_proc.return_value = proc_mock

                        result_json = await tool.execute(
                            script="...", description="SDK test"
                        )
                        result = json.loads(result_json)
                        assert "sdk_actions" in result
                        assert len(result["sdk_actions"]) == 1

    @pytest.mark.asyncio
    async def test_process_sdk_action_returns_ack(self):
        result = await _process_sdk_action({"action": "memory.save"})
        assert result["status"] == "processed"
        assert result["action"] == "memory.save"

    def test_safe_env_vars_excludes_secrets(self):
        """SAFE_ENV_VARS must not include any known secret env vars."""
        secret_vars = {
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "DEEPGRAM_API_KEY",
            "ELEVENLABS_API_KEY",
            "WHATSAPP_ACCESS_TOKEN",
            "WHATSAPP_VERIFY_TOKEN",
            "WHATSAPP_APP_SECRET",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        }
        assert SAFE_ENV_VARS.isdisjoint(secret_vars), (
            f"SAFE_ENV_VARS must not contain secrets: "
            f"{SAFE_ENV_VARS & secret_vars}"
        )

    def test_safe_env_vars_includes_essentials(self):
        """SAFE_ENV_VARS must include PATH and LANG for basic operation."""
        assert "PATH" in SAFE_ENV_VARS
        assert "LANG" in SAFE_ENV_VARS

    @pytest.mark.asyncio
    async def test_env_allowlist_applied(self, tmp_path):
        """Verify that only allowlisted env vars reach the subprocess."""
        tool = ExecuteScriptTool()

        captured_env = {}

        async def fake_subprocess(*args, **kwargs):
            nonlocal captured_env
            captured_env = kwargs.get("env", {})
            proc_mock = AsyncMock()
            proc_mock.communicate.return_value = (b"ok\n", b"")
            proc_mock.returncode = 0
            return proc_mock

        with patch("boxbot.tools.builtins.execute_script.SCRIPTS_DIR", tmp_path / "scripts"):
            with patch("boxbot.tools.builtins.execute_script.OUTPUT_DIR", tmp_path / "output"):
                with patch("boxbot.tools.builtins.execute_script.get_config", side_effect=RuntimeError):
                    with patch.dict("os.environ", {
                        "PATH": "/usr/bin",
                        "ANTHROPIC_API_KEY": "sk-secret-key",
                        "AWS_SECRET_ACCESS_KEY": "aws-secret",
                        "HOME": "/home/test",
                        "RANDOM_VAR": "should-not-pass",
                    }, clear=True):
                        with patch(
                            "asyncio.create_subprocess_exec",
                            side_effect=fake_subprocess,
                        ):
                            await tool.execute(
                                script="print('test')",
                                description="Env test",
                            )

        # Allowlisted vars should be present
        assert captured_env.get("PATH") == "/usr/bin"
        assert captured_env.get("HOME") == "/home/test"
        # Secrets must NOT be present
        assert "ANTHROPIC_API_KEY" not in captured_env
        assert "AWS_SECRET_ACCESS_KEY" not in captured_env
        # Non-allowlisted non-secrets must NOT be present
        assert "RANDOM_VAR" not in captured_env


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
