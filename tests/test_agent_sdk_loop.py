"""Tests for ``_agent_loop_sdk`` and the backend-dispatch wiring.

Mocks ``ClaudeSDKClient`` so we can control the stream of messages
the loop sees, then asserts on the side effects (cost rows, message
dispatch tracking, fallback firing, ContextVar binding, interrupt
plumbing).

Skipped wholesale when ``claude_agent_sdk`` is not importable.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from boxbot.core import agent_sdk_adapter as A
from boxbot.core.agent import BoxBotAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_message(
    *,
    num_turns: int = 1,
    total_cost_usd: float = 0.05,
    is_error: bool = False,
    stop_reason: str | None = "end_turn",
) -> ResultMessage:
    """Build a realistic ResultMessage. All fields populated."""
    return ResultMessage(
        subtype="success",
        duration_ms=1000,
        duration_api_ms=900,
        is_error=is_error,
        num_turns=num_turns,
        session_id="sess_test",
        stop_reason=stop_reason,
        total_cost_usd=total_cost_usd,
        usage={
            "input_tokens": 100,
            "output_tokens": 30,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        result=None,
        structured_output=None,
        model_usage={
            "claude-opus-4-7-20260415": {
                "inputTokens": 100,
                "outputTokens": 30,
                "cacheReadInputTokens": 0,
                "cacheCreationInputTokens": 0,
                "costUSD": total_cost_usd,
                "contextWindow": 200000,
                "maxOutputTokens": 8192,
            }
        },
        permission_denials=None,
        deferred_tool_use=None,
        errors=None,
        api_error_status=None,
        uuid="msg_test",
    )


def _internal_notes_text(thought: str = "thinking", observations=None) -> str:
    """Return a JSON string matching INTERNAL_NOTES_SCHEMA."""
    import json
    return json.dumps({
        "thought": thought,
        "observations": observations or [],
    })


class _FakeSdkClient:
    """A minimal stand-in for ClaudeSDKClient.

    Yields the canned message stream we pass to ``__init__``; records
    every ``query``, ``interrupt``, and ``disconnect`` call so tests can
    assert on them.
    """

    def __init__(self, messages: list[Any]):
        self._messages = messages
        self.queries: list[str] = []
        self.interrupt_calls = 0
        self.disconnect_calls = 0
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def query(self, prompt: str, session_id: str = "default") -> None:
        self.queries.append(prompt)

    async def receive_response(self):
        for msg in self._messages:
            yield msg

    async def interrupt(self) -> None:
        self.interrupt_calls += 1

    async def disconnect(self) -> None:
        self.disconnect_calls += 1


class _FakeConv:
    """Minimal Conversation stand-in for the SDK loop's bookkeeping."""

    def __init__(self, conv_id: str = "conv_test", channel: str = "voice"):
        self.conversation_id = conv_id
        self.channel = channel


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_conversation_routes_to_sdk_loop_when_backend_is_sdk(
    mock_config,
):
    """When agent.backend = claude_agent_sdk, ``_agent_loop_sdk`` runs."""
    from boxbot.core import config as config_module

    mock_config.agent.backend = "claude_agent_sdk"
    config_module._config = mock_config

    mem = MagicMock()
    mem.read_system_memory = MagicMock(return_value="")
    agent = BoxBotAgent(memory_store=mem)
    agent._client = MagicMock()  # bypass "Agent not started" assert

    # Stub both loops to assert which one fires.
    agent._agent_loop = AsyncMock(
        return_value=([{"role": "assistant", "content": "raw"}], 1),
    )
    agent._agent_loop_sdk = AsyncMock(
        return_value=([{"role": "assistant", "content": "sdk"}], 1),
    )
    agent._build_system_prompt_blocks = AsyncMock(
        return_value=[{"type": "text", "text": "sys"}],
    )
    agent._get_most_recent_person = MagicMock(return_value=None)
    agent._extract_summary = MagicMock(return_value=None)

    conv = MagicMock()
    conv.thread = [{"role": "user", "content": "hello"}]
    conv.conversation_id = "c1"
    conv.channel = "voice"
    conv.current_context = None
    conv.set_state = MagicMock()

    await agent._generate_for_conversation(conv)

    agent._agent_loop_sdk.assert_awaited_once()
    agent._agent_loop.assert_not_called()


@pytest.mark.asyncio
async def test_run_conversation_routes_to_raw_loop_when_backend_is_raw(
    mock_config,
):
    """Default backend keeps the raw_anthropic path live."""
    from boxbot.core import config as config_module

    mock_config.agent.backend = "raw_anthropic"
    config_module._config = mock_config

    mem = MagicMock()
    mem.read_system_memory = MagicMock(return_value="")
    agent = BoxBotAgent(memory_store=mem)
    agent._client = MagicMock()  # bypass "Agent not started" assert

    agent._agent_loop = AsyncMock(
        return_value=([{"role": "assistant", "content": "raw"}], 1),
    )
    agent._agent_loop_sdk = AsyncMock(
        return_value=([{"role": "assistant", "content": "sdk"}], 1),
    )
    agent._build_system_prompt_blocks = AsyncMock(
        return_value=[{"type": "text", "text": "sys"}],
    )
    agent._get_most_recent_person = MagicMock(return_value=None)
    agent._extract_summary = MagicMock(return_value=None)

    conv = MagicMock()
    conv.thread = [{"role": "user", "content": "hello"}]
    conv.conversation_id = "c1"
    conv.channel = "voice"
    conv.current_context = None
    conv.set_state = MagicMock()

    await agent._generate_for_conversation(conv)

    agent._agent_loop.assert_awaited_once()
    agent._agent_loop_sdk.assert_not_called()


# ---------------------------------------------------------------------------
# _agent_loop_sdk behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sdk_loop_records_cost_event_on_result_message(mock_config):
    """ResultMessage → from_agent_sdk_result → record_cost."""
    from boxbot.core import config as config_module

    mock_config.agent.backend = "claude_agent_sdk"
    config_module._config = mock_config

    mem = MagicMock()
    agent = BoxBotAgent(memory_store=mem)

    fake = _FakeSdkClient(messages=[
        _result_message(num_turns=2, total_cost_usd=0.123),
    ])
    conv = _FakeConv()
    conv._sdk_client = fake

    with patch("boxbot.core.agent.record_cost", new=AsyncMock()) as rec:
        messages, turn_count = await agent._agent_loop_sdk(
            conv=conv,
            channel="voice",
            system_prompt_blocks=[{"type": "text", "text": "sys"}],
            initial_message="hi",
            person_name=None,
            max_turns=20,
            prior_history=None,
        )

    assert turn_count == 2
    assert fake.queries == ["hi"]
    rec.assert_awaited()  # at least one cost record
    # First positional arg to record_cost is the memory store.
    call_args = rec.await_args_list[0]
    assert call_args.args[0] is mem
    ev = call_args.args[1]
    assert ev.purpose == "conversation"
    assert ev.cost_usd == pytest.approx(0.123)
    assert ev.metadata.get("backend") == "claude_agent_sdk"


@pytest.mark.asyncio
async def test_sdk_loop_tracks_message_tool_dispatch(mock_config):
    """A message tool_use → no fallback. No message tool_use → fallback fires."""
    from boxbot.core import config as config_module

    mock_config.agent.backend = "claude_agent_sdk"
    config_module._config = mock_config

    mem = MagicMock()
    agent = BoxBotAgent(memory_store=mem)
    agent._dispatch_max_turns_fallback = AsyncMock()

    # Case 1: message tool dispatched + below cap → no fallback.
    fake1 = _FakeSdkClient(messages=[
        AssistantMessage(
            content=[
                TextBlock(text=_internal_notes_text()),
                ToolUseBlock(
                    id="t1",
                    name=A.mcp_tool_name("message"),
                    input={"to": "Jacob", "channel": "speak", "content": "hi"},
                ),
            ],
            model="opus", parent_tool_use_id=None, error=None, usage=None,
            message_id="m1", stop_reason=None, session_id="s", uuid="u1",
        ),
        _result_message(num_turns=2),
    ])
    conv1 = _FakeConv("c1")
    conv1._sdk_client = fake1
    with patch("boxbot.core.agent.record_cost", new=AsyncMock()):
        await agent._agent_loop_sdk(
            conv=conv1,
            channel="voice",
            system_prompt_blocks=[{"type": "text", "text": "sys"}],
            initial_message="x",
            person_name=None,
            max_turns=20,
            prior_history=None,
        )
    agent._dispatch_max_turns_fallback.assert_not_called()

    # Case 2: no message tool + cap reached → fallback fires.
    fake2 = _FakeSdkClient(messages=[
        _result_message(num_turns=20),
    ])
    conv2 = _FakeConv("c2")
    conv2._sdk_client = fake2
    with patch("boxbot.core.agent.record_cost", new=AsyncMock()):
        await agent._agent_loop_sdk(
            conv=conv2,
            channel="voice",
            system_prompt_blocks=[{"type": "text", "text": "sys"}],
            initial_message="x",
            person_name=None,
            max_turns=20,
            prior_history=None,
        )
    agent._dispatch_max_turns_fallback.assert_awaited_once()


@pytest.mark.asyncio
async def test_sdk_loop_calls_interrupt_on_cancellation(mock_config):
    """Cancellation mid-stream → ``sdk_client.interrupt()`` then re-raises."""
    from boxbot.core import config as config_module

    mock_config.agent.backend = "claude_agent_sdk"
    config_module._config = mock_config

    mem = MagicMock()
    agent = BoxBotAgent(memory_store=mem)

    class _SlowFakeClient(_FakeSdkClient):
        async def receive_response(self):
            # Yield one assistant message, then block forever waiting
            # for the next — the task that drives this gets cancelled
            # by the test.
            yield AssistantMessage(
                content=[TextBlock(text=_internal_notes_text())],
                model="opus", parent_tool_use_id=None, error=None,
                usage=None, message_id="m1", stop_reason=None,
                session_id="s", uuid="u1",
            )
            await asyncio.Future()  # hang

    fake = _SlowFakeClient(messages=[])
    conv = _FakeConv()
    conv._sdk_client = fake

    async def _run():
        with patch("boxbot.core.agent.record_cost", new=AsyncMock()):
            await agent._agent_loop_sdk(
                conv=conv,
                channel="voice",
                system_prompt_blocks=[{"type": "text", "text": "sys"}],
                initial_message="hi",
                person_name=None,
                max_turns=20,
                prior_history=None,
            )

    task = asyncio.create_task(_run())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake.interrupt_calls == 1


@pytest.mark.asyncio
async def test_sdk_loop_reuses_attached_client_does_not_reconnect(mock_config):
    """A second turn in the same Conversation reuses the existing client."""
    from boxbot.core import config as config_module

    mock_config.agent.backend = "claude_agent_sdk"
    config_module._config = mock_config

    mem = MagicMock()
    agent = BoxBotAgent(memory_store=mem)

    fake = _FakeSdkClient(messages=[_result_message()])
    conv = _FakeConv()
    conv._sdk_client = fake  # pre-attached

    with patch("boxbot.core.agent.record_cost", new=AsyncMock()):
        await agent._agent_loop_sdk(
            conv=conv,
            channel="voice",
            system_prompt_blocks=[{"type": "text", "text": "sys"}],
            initial_message="second utterance",
            person_name=None,
            max_turns=20,
            prior_history=None,
        )

    # The pre-attached fake had connected=False; if we reused it, the
    # loop should NOT have called connect().
    assert fake.connected is False
    assert fake.queries == ["second utterance"]


# ---------------------------------------------------------------------------
# Lifecycle: ConversationEnded reaps the SDK client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_conversation_ended_disconnects_sdk_client():
    """The SDK client subprocess gets reaped when the Conversation ends."""
    from boxbot.core.events import ConversationEnded

    mem = MagicMock()
    agent = BoxBotAgent(memory_store=mem)

    # Build a fake Conversation registered in the agent's index, with
    # an attached SDK client. Persistent conversations have extraction
    # routed elsewhere, so set lifecycle_mode="persistent" to skip the
    # extraction path and isolate the disconnect call.
    fake_client = _FakeSdkClient(messages=[])
    conv = MagicMock()
    conv.conversation_id = "c_end"
    conv.sandbox_runner = None
    conv._sdk_client = fake_client
    conv.thread = []
    conv.accessed_memory_ids = []
    conv.injected_memories_block = ""
    conv.lifecycle_mode = "persistent"
    conv.started_at_iso = MagicMock(return_value="2026-05-29T00:00:00Z")

    async with agent._index_lock:
        agent._conversations["c_end"] = conv

    await agent._on_conversation_ended(
        ConversationEnded(
            conversation_id="c_end",
            channel="voice",
            person_name=None,
            turn_count=1,
            summary="",
        )
    )

    # disconnect runs as a fire-and-forget task; give it a tick.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert fake_client.disconnect_calls == 1
    assert conv._sdk_client is None


# ---------------------------------------------------------------------------
# Start() auth gates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_requires_oauth_token_when_backend_is_sdk(mock_config):
    """Hard fail at start() if backend = claude_agent_sdk but no OAuth token."""
    from boxbot.core import config as config_module

    mock_config.agent.backend = "claude_agent_sdk"
    mock_config.api_keys.anthropic = "sk-test"
    mock_config.api_keys.claude_code_oauth_token = None
    config_module._config = mock_config

    mem = MagicMock()
    agent = BoxBotAgent(memory_store=mem)

    with pytest.raises(RuntimeError, match="CLAUDE_CODE_OAUTH_TOKEN"):
        await agent.start()


@pytest.mark.asyncio
async def test_start_still_requires_api_key_when_backend_is_sdk(mock_config):
    """Peripheral Haiku calls always need ANTHROPIC_API_KEY."""
    from boxbot.core import config as config_module

    mock_config.agent.backend = "claude_agent_sdk"
    mock_config.api_keys.anthropic = None
    mock_config.api_keys.claude_code_oauth_token = "oauth-test"
    config_module._config = mock_config

    mem = MagicMock()
    agent = BoxBotAgent(memory_store=mem)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await agent.start()
