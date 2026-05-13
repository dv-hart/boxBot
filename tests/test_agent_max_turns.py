"""Tests for the max-turns graceful close-out in ``_agent_loop``.

When the agent loop would otherwise loop past the configured turn cap,
two things must happen:

1. The penultimate iteration appends a ``[system]`` heads-up to the
   tool-result user block so the model knows its next response is
   final and only the ``message`` tool will be available.
2. The final iteration calls ``messages.create`` with ``tools``
   filtered to just the ``message`` definition.
3. After the final turn, the loop exits. If a ``message`` was
   dispatched, the user has been told what happened. If not, a
   hardcoded fallback dispatches via ``output_dispatcher``.

These tests mock the Anthropic client to drive the loop deterministically.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from boxbot.core.agent import BoxBotAgent


# ---------------------------------------------------------------------------
# Response builders — match the Anthropic SDK shape the loop walks
# ---------------------------------------------------------------------------


def _text_block(text: str = '{"thought":"working","observations":[]}') -> Any:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name: str, tool_use_id: str, **kwargs: Any) -> Any:
    return SimpleNamespace(
        type="tool_use", name=name, id=tool_use_id, input=kwargs,
    )


def _response(*content: Any, stop_reason: str = "tool_use") -> Any:
    return SimpleNamespace(
        content=list(content),
        stop_reason=stop_reason,
        model="claude-opus-4-7",
        usage=None,
    )


# ---------------------------------------------------------------------------
# Fixture: an agent wired so _agent_loop can run end-to-end without a
# real Anthropic client, real tools, or a real conversation index.
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_with_mock_client(monkeypatch, mock_config):
    mem = MagicMock()
    mem.read_system_memory = MagicMock(return_value="")
    agent = BoxBotAgent(memory_store=mem)
    agent._client = MagicMock()
    agent._client.messages = MagicMock()
    agent._client.messages.create = AsyncMock()
    agent._running = True

    # Stub _process_tool_calls: the message tool's real implementation
    # would call dispatch_outputs; we don't want to exercise that here.
    # The dispatcher itself is patched separately in tests that care.
    async def _fake_process(response, tools, *, conversation_id=None):
        results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": '{"status":"delivered"}',
            })
        return results
    monkeypatch.setattr(agent, "_process_tool_calls", _fake_process)

    # Stub cost recording — irrelevant to the loop's control flow.
    async def _noop_cost(*a, **kw):
        return None
    monkeypatch.setattr("boxbot.core.agent.record_cost", _noop_cost)

    # Stub _build_tool_definitions to return a small fixed list including
    # one named "message". The loop filters by name on the final turn.
    def _fake_tool_definitions(_self_tools):
        return [
            {"name": "message", "description": "deliver", "input_schema": {}},
            {"name": "execute_script", "description": "x", "input_schema": {}},
            {"name": "search_memory", "description": "m", "input_schema": {}},
        ]
    monkeypatch.setattr(
        agent, "_build_tool_definitions", _fake_tool_definitions,
    )
    # get_tools() is imported inside _agent_loop; patch the import target
    # so the call returns an empty list (we've replaced _process_tool_calls).
    monkeypatch.setattr(
        "boxbot.tools.registry.get_tools", lambda: [],
    )

    return agent


# ---------------------------------------------------------------------------
# Happy path: model emits ``message`` on the final turn → no fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_turn_message_call_closes_out_cleanly(
    agent_with_mock_client, monkeypatch,
):
    agent = agent_with_mock_client
    max_turns = 4

    # The mock returns tool_use(execute_script) for turns 1..3 and
    # tool_use(message) for turn 4 (the final allowed turn).
    responses = [
        _response(_text_block(), _tool_use_block("execute_script", "t1")),
        _response(_text_block(), _tool_use_block("execute_script", "t2")),
        _response(_text_block(), _tool_use_block("execute_script", "t3")),
        _response(_text_block(), _tool_use_block(
            "message", "t4",
            to="current_speaker", channel="text",
            content="I hit the limit. Tried X but couldn't verify.",
        )),
    ]
    agent._client.messages.create.side_effect = responses

    # The fallback should NOT be called — the model handled it.
    fallback = AsyncMock()
    monkeypatch.setattr(
        agent, "_dispatch_max_turns_fallback", fallback,
    )

    messages, turns = await agent._agent_loop(
        conversation_id="conv-test",
        channel="whatsapp",
        system_prompt_blocks=[{"type": "text", "text": "sys"}],
        initial_message="hi",
        person_name="Jacob",
        max_turns=max_turns,
    )

    # Loop went the full distance.
    assert turns == max_turns

    # Penultimate iteration's user block contains the heads-up text.
    user_blocks_with_headsup = [
        m for m in messages
        if m["role"] == "user"
        and isinstance(m["content"], list)
        and any(
            isinstance(b, dict) and b.get("type") == "text"
            and "turn cap" in b.get("text", "")
            for b in m["content"]
        )
    ]
    assert len(user_blocks_with_headsup) == 1, (
        "expected exactly one heads-up user message"
    )

    # Final turn API call: tools filtered to just `message`.
    final_call = agent._client.messages.create.call_args_list[-1]
    tool_names = [t["name"] for t in final_call.kwargs["tools"]]
    assert tool_names == ["message"], (
        f"final turn must only offer message; got {tool_names}"
    )

    # Earlier turns still had the full tool set.
    earlier_call = agent._client.messages.create.call_args_list[0]
    earlier_tool_names = [t["name"] for t in earlier_call.kwargs["tools"]]
    assert "execute_script" in earlier_tool_names
    assert "search_memory" in earlier_tool_names

    # Fallback wasn't called because the model emitted a message call.
    fallback.assert_not_awaited()


# ---------------------------------------------------------------------------
# Fallback path: model ignores the cap on the final turn → fallback fires
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_turn_without_message_triggers_fallback(
    agent_with_mock_client, monkeypatch,
):
    agent = agent_with_mock_client
    max_turns = 3

    # Turns 1..2 emit tool_use; turn 3 (the final) emits text only
    # (no message tool call) and stop_reason=end_turn.
    responses = [
        _response(_text_block(), _tool_use_block("execute_script", "t1")),
        _response(_text_block(), _tool_use_block("execute_script", "t2")),
        _response(
            _text_block('{"thought":"giving up","observations":[]}'),
            stop_reason="end_turn",
        ),
    ]
    agent._client.messages.create.side_effect = responses

    fallback = AsyncMock()
    monkeypatch.setattr(
        agent, "_dispatch_max_turns_fallback", fallback,
    )

    _messages, turns = await agent._agent_loop(
        conversation_id="conv-test",
        channel="voice",
        system_prompt_blocks=[{"type": "text", "text": "sys"}],
        initial_message="hi",
        person_name="Jacob",
        max_turns=max_turns,
    )

    assert turns == max_turns
    fallback.assert_awaited_once()
    kwargs = fallback.await_args.kwargs
    assert kwargs["conversation_id"] == "conv-test"
    assert kwargs["channel"] == "voice"
    assert kwargs["person_name"] == "Jacob"
    assert kwargs["max_turns"] == max_turns


# ---------------------------------------------------------------------------
# Negative control: model finishes naturally before the cap → neither
# heads-up nor fallback should fire.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_natural_end_turn_skips_headsup_and_fallback(
    agent_with_mock_client, monkeypatch,
):
    agent = agent_with_mock_client

    responses = [
        _response(_text_block(), _tool_use_block("execute_script", "t1")),
        _response(_text_block(), stop_reason="end_turn"),
    ]
    agent._client.messages.create.side_effect = responses

    fallback = AsyncMock()
    monkeypatch.setattr(
        agent, "_dispatch_max_turns_fallback", fallback,
    )

    messages, turns = await agent._agent_loop(
        conversation_id="conv-test",
        channel="whatsapp",
        system_prompt_blocks=[{"type": "text", "text": "sys"}],
        initial_message="hi",
        person_name="Jacob",
        max_turns=10,
    )

    assert turns == 2
    fallback.assert_not_awaited()

    # No user block should contain "turn cap" since the heads-up never fired.
    assert not any(
        isinstance(m.get("content"), list)
        and any(
            isinstance(b, dict)
            and b.get("type") == "text"
            and "turn cap" in b.get("text", "")
            for b in m["content"]
        )
        for m in messages
        if m["role"] == "user"
    )

    # All API calls had the full tool set.
    for call in agent._client.messages.create.call_args_list:
        names = [t["name"] for t in call.kwargs["tools"]]
        assert "execute_script" in names
