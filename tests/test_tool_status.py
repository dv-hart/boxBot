"""Tests for tool_status — tool-call → status-line mapping + publishing."""

from __future__ import annotations

import pytest

from boxbot.core.tool_status import publish_tool_status, status_text_for


class TestStatusTextFor:
    def test_simple_tools_map_to_lines(self):
        assert status_text_for("web_search", {}) == "Searching the web…"
        assert status_text_for("search_memory", {}) == "Recalling…"
        assert status_text_for("load_skill", {"name": "weather"}) == "Reading up…"

    def test_silent_tools_return_none(self):
        assert status_text_for("message", {"content": "hi"}) is None
        assert status_text_for("switch_display", {"name": "clock"}) is None
        assert status_text_for("mute_mic", {}) is None

    def test_unknown_tool_returns_none(self):
        assert status_text_for("some_future_tool", {}) is None

    def test_execute_script_sniffs_bb_modules(self):
        script = "img = bb.camera.capture()\nbb.workspace.write('a.md', 'x')"
        assert status_text_for("execute_script", {"script": script}) == (
            "Checking the camera…"
        )

    def test_execute_script_priority_order(self):
        # workspace + memory both present — workspace outranks memory.
        script = "bb.memory.search('x')\nbb.workspace.read('notes.md')"
        assert status_text_for("execute_script", {"script": script}) == (
            "Checking notes…"
        )

    def test_execute_script_boxbot_sdk_spelling(self):
        script = "import boxbot_sdk\nboxbot_sdk.photos.search('dog')"
        assert status_text_for("execute_script", {"script": script}) == (
            "Searching photos…"
        )

    def test_execute_script_default_when_no_module(self):
        assert status_text_for("execute_script", {"script": "print(1 + 1)"}) == (
            "Working on it…"
        )

    def test_execute_script_tolerates_missing_input(self):
        assert status_text_for("execute_script", None) == "Working on it…"
        assert status_text_for("execute_script", {"script": 42}) == (
            "Working on it…"
        )


class _FakeConv:
    conversation_id = "voice_room"
    channel = "voice"


class TestPublishToolStatus:
    @pytest.mark.asyncio
    async def test_publishes_agent_tool_called(self, event_bus):
        from boxbot.core.events import AgentToolCalled

        received = []

        async def handler(event):
            received.append(event)

        event_bus.subscribe(AgentToolCalled, handler)
        await publish_tool_status(_FakeConv(), "web_search", {"query": "x"})

        assert len(received) == 1
        ev = received[0]
        assert ev.conversation_id == "voice_room"
        assert ev.channel == "voice"
        assert ev.tool_name == "web_search"
        assert ev.status_text == "Searching the web…"

    @pytest.mark.asyncio
    async def test_no_publish_for_silent_tool(self, event_bus):
        from boxbot.core.events import AgentToolCalled

        received = []

        async def handler(event):
            received.append(event)

        event_bus.subscribe(AgentToolCalled, handler)
        await publish_tool_status(_FakeConv(), "message", {"content": "hi"})
        assert received == []

    @pytest.mark.asyncio
    async def test_no_publish_without_conversation(self, event_bus):
        from boxbot.core.events import AgentToolCalled

        received = []

        async def handler(event):
            received.append(event)

        event_bus.subscribe(AgentToolCalled, handler)
        await publish_tool_status(None, "web_search", {})
        assert received == []

    @pytest.mark.asyncio
    async def test_never_raises_on_malformed_conv(self):
        # Test-style sentinel convs (plain object()) lack the attrs —
        # the publish must swallow, not raise (mirrors wrap_tool tests).
        await publish_tool_status(object(), "web_search", {})
