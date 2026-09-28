"""Tests for the ``final_turn`` end-of-turn flag.

Without a terminal action the only way an agent turn ends is a response
carrying no tool calls — but the model is told its text output never
reaches a person, so it calls ``message`` to "conclude" and buys another
round-trip. ``final_turn`` in the internal notes is that terminal
action: the flag rides the same response as the last ``message`` call,
so ending a turn costs no extra API call.

The rules under test:

1. ``final_turn: true`` + every tool result successful → the loop breaks
   after appending the results to history.
2. A failed tool result overrides the flag — the model gets one more
   round so it sees the error.
3. Input that landed mid-turn overrides the flag — somebody is still
   talking.
4. No flag (or no text block at all) → status quo, the loop continues.

Both the raw-Anthropic and the OpenAI loop are covered; the Claude Agent
SDK backend owns its own loop and cannot enforce the flag.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from boxbot.core.agent import BoxBotAgent, _tool_results_ok
from boxbot.core.output_dispatcher import DEGENERATE_CONTENT, UNRETRYABLE_DROPS


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _notes(final_turn: bool | None = None, thought: str = "working") -> str:
    body: dict[str, Any] = {"thought": thought, "observations": []}
    if final_turn is not None:
        body["final_turn"] = final_turn
    return json.dumps(body)


def _text_block(text: str) -> Any:
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


def _message_response(final_turn: bool | None, block_id: str = "t1") -> Any:
    return _response(
        _text_block(_notes(final_turn)),
        _tool_use_block(
            "message", block_id,
            to="current_speaker", channel="text", content="All set.",
        ),
    )


# ---------------------------------------------------------------------------
# Fixture — an agent whose Anthropic client and tool dispatch are mocked.
# ``tool_result_content`` is mutable so a test can make a tool fail.
# ---------------------------------------------------------------------------


@pytest.fixture
def agent(monkeypatch, mock_config):
    mem = MagicMock()
    mem.read_system_memory = MagicMock(return_value="")
    a = BoxBotAgent(memory_store=mem)
    a._client = MagicMock()
    a._client.messages = MagicMock()
    a._client.messages.create = AsyncMock()
    a._running = True
    a._tool_result_content = '{"status":"delivered"}'

    async def _fake_process(
        response, tools, *, conversation_id=None, turn_number=None,
        channel=None, backend=None,
    ):
        return [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": a._tool_result_content,
            }
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
        ]

    monkeypatch.setattr(a, "_process_tool_calls", _fake_process)

    async def _noop_cost(*_a, **_kw):
        return None
    monkeypatch.setattr("boxbot.core.agent.record_cost", _noop_cost)

    monkeypatch.setattr(
        a, "_build_tool_definitions",
        lambda _tools: [
            {"name": "message", "description": "deliver", "input_schema": {}},
            {"name": "execute_script", "description": "x", "input_schema": {}},
        ],
    )
    monkeypatch.setattr("boxbot.tools.registry.get_tools", lambda: [])
    return a


async def _run(
    agent: BoxBotAgent, max_turns: int = 6, channel: str = "trigger",
):
    return await agent._agent_loop(
        conversation_id="conv-test",
        channel=channel,
        system_prompt_blocks=[{"type": "text", "text": "sys"}],
        initial_message="wake",
        person_name="Jacob",
        max_turns=max_turns,
    )


# ---------------------------------------------------------------------------
# Schema + parsing
# ---------------------------------------------------------------------------


def test_flag_is_required_in_the_schema():
    from boxbot.core.output_dispatcher import INTERNAL_NOTES_SCHEMA

    assert "final_turn" in INTERNAL_NOTES_SCHEMA["required"]
    assert (
        INTERNAL_NOTES_SCHEMA["properties"]["final_turn"]["type"] == "boolean"
    )


def test_parsers_surface_the_flag():
    from boxbot.core.output_dispatcher import (
        parse_internal_notes,
        parse_structured_notes,
    )

    assert parse_internal_notes(_notes(True)).final_turn is True
    assert parse_internal_notes(_notes(False)).final_turn is False
    # Absent (a backend that drops the field) reads as "keep going".
    assert parse_internal_notes(_notes(None)).final_turn is False
    assert parse_structured_notes({
        "thought": "done", "final_turn": True,
    }).final_turn is True


def test_flag_is_or_ed_across_text_blocks():
    """A response may split its notes; the flag can ride any block."""
    from boxbot.core.agent import _log_internal_notes

    notes = _log_internal_notes(
        _response(
            _text_block(_notes(False, thought="still working")),
            _text_block(_notes(True, thought="done")),
        ),
        "conv-1", 1,
    )
    assert notes is not None
    assert notes.thought == "still working"
    assert notes.final_turn is True


# ---------------------------------------------------------------------------
# _tool_results_ok
# ---------------------------------------------------------------------------


class TestToolResultsOk:

    def _block(self, content: Any) -> dict[str, Any]:
        return {"type": "tool_result", "tool_use_id": "t1", "content": content}

    def test_delivered_is_ok(self):
        assert _tool_results_ok([self._block('{"status":"delivered"}')])

    def test_status_error_is_not_ok(self):
        assert not _tool_results_ok(
            [self._block('{"status":"error","message":"unknown recipient"}')]
        )

    def test_top_level_error_key_is_not_ok(self):
        assert not _tool_results_ok(
            [self._block('{"error":"Unknown tool: nope"}')]
        )

    def test_non_json_content_is_ok(self):
        assert _tool_results_ok([self._block("plain prose result")])

    def test_multimodal_content_reads_the_text_block(self):
        assert not _tool_results_ok([self._block([
            {"type": "text", "text": '{"status":"error","error":"boom"}'},
            {"type": "image", "source": {}},
        ])])

    def test_one_failure_fails_the_batch(self):
        assert not _tool_results_ok([
            self._block('{"status":"delivered"}'),
            self._block('{"status":"error"}'),
        ])


# ---------------------------------------------------------------------------
# Raw-Anthropic loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_ends_the_loop_after_a_successful_message(agent):
    agent._client.messages.create.side_effect = [_message_response(True)]

    messages, turns = await _run(agent)

    assert turns == 1, "the flag must end the turn without another API call"
    assert agent._client.messages.create.await_count == 1
    # The tool result is still in history — the bridge and the
    # conversation store read delivery status off it.
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"][0]["type"] == "tool_result"


@pytest.mark.asyncio
async def test_flag_on_the_message_call_ends_the_loop(agent):
    """``final_turn: true`` in the message call's own input ends the
    turn even when the response carries no notes text block at all."""
    agent._client.messages.create.side_effect = [
        _response(
            _tool_use_block(
                "message", "t1",
                to="current_speaker", channel="text",
                content="All set.", final_turn=True,
            ),
        ),
    ]

    _messages, turns = await _run(agent)

    assert turns == 1
    assert agent._client.messages.create.await_count == 1


@pytest.mark.asyncio
async def test_flag_beside_a_successful_command_ends_the_loop(agent):
    """The fire-and-forget actuation SOP: command + spoken ack +
    final_turn in ONE response ends the turn — no close-out call. The
    sibling result still lands in history (no orphaned tool_use)."""
    agent._client.messages.create.side_effect = [
        _response(
            _tool_use_block(
                "message", "t1",
                to="current_speaker", channel="text",
                content="Lock command sent.", final_turn=True,
            ),
            _tool_use_block("execute_script", "t2", code="x"),
        ),
    ]

    messages, turns = await _run(agent)

    assert turns == 1
    assert agent._client.messages.create.await_count == 1
    result_ids = {
        b["tool_use_id"]
        for m in messages if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    }
    assert {"t1", "t2"} <= result_ids, "sibling results must be in history"


@pytest.mark.asyncio
async def test_flag_beside_a_failed_command_keeps_the_loop_running(agent):
    """An errored sibling vetoes the early end — the model must see the
    failure before the turn can close."""
    agent._tool_result_content = json.dumps({
        "status": "error", "message": "panel bridge is not responding",
    })
    agent._client.messages.create.side_effect = [
        _response(
            _tool_use_block(
                "message", "t1",
                to="current_speaker", channel="text",
                content="Lock command sent.", final_turn=True,
            ),
            _tool_use_block("execute_script", "t2", code="x"),
        ),
        _response(_text_block(_notes(True)), stop_reason="end_turn"),
    ]

    _messages, turns = await _run(agent)

    assert turns == 2


@pytest.mark.asyncio
async def test_flag_false_keeps_the_loop_running(agent):
    agent._client.messages.create.side_effect = [
        _response(
            _text_block(_notes(False)),
            _tool_use_block("execute_script", "t1"),
        ),
        _message_response(True, block_id="t2"),
    ]

    _messages, turns = await _run(agent)

    assert turns == 2


@pytest.mark.asyncio
async def test_missing_flag_keeps_the_loop_running(agent):
    """A response with no text block carries no flag — status quo."""
    agent._client.messages.create.side_effect = [
        _response(_tool_use_block("execute_script", "t1")),
        _response(_text_block(_notes(True)), stop_reason="end_turn"),
    ]

    _messages, turns = await _run(agent)

    assert turns == 2


@pytest.mark.asyncio
async def test_failed_tool_result_overrides_the_flag(agent):
    """The model must see the error, so it gets one more round."""
    agent._tool_result_content = json.dumps({
        "status": "error", "message": "unknown recipient 'Dev'",
    })
    agent._client.messages.create.side_effect = [
        _message_response(True),
        _response(_text_block(_notes(True)), stop_reason="end_turn"),
    ]

    _messages, turns = await _run(agent)

    assert turns == 2


@pytest.mark.asyncio
async def test_mid_turn_input_overrides_the_flag(agent, monkeypatch):
    """Someone spoke while we were thinking — answer them, don't stop."""
    monkeypatch.setattr(
        agent, "_drain_pending_into",
        lambda _cid, blocks: (
            blocks.append({"type": "text", "text": "wait, one more thing"})
            or 1
        ),
    )
    agent._client.messages.create.side_effect = [
        _message_response(True),
        _response(_text_block(_notes(True)), stop_reason="end_turn"),
    ]

    _messages, turns = await _run(agent)

    assert turns == 2


@pytest.mark.asyncio
async def test_flag_suppresses_the_turn_cap_notice(agent):
    """Ending on the penultimate turn must not inject the cap heads-up."""
    agent._client.messages.create.side_effect = [_message_response(True)]

    messages, turns = await _run(agent, max_turns=2)

    assert turns == 1
    assert not any(
        isinstance(b, dict) and "turn cap" in str(b.get("text", ""))
        for m in messages if isinstance(m.get("content"), list)
        for b in m["content"]
    )


# ---------------------------------------------------------------------------
# Backstop — a trigger run whose whole batch is ``message`` calls has
# nothing left to do, flag or no flag.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backstop_ends_a_message_only_trigger_run(agent):
    agent._client.messages.create.side_effect = [_message_response(None)]

    _messages, turns = await _run(agent)

    assert turns == 1


@pytest.mark.asyncio
async def test_backstop_ends_when_the_message_was_dropped_as_filler(agent):
    """A dropped junk message answered with an apology is another junk
    message — the point of the backstop is to not buy that round-trip."""
    agent._tool_result_content = json.dumps({
        "status": "error", "message": "filler content is not deliverable",
        "reason_code": DEGENERATE_CONTENT,
    })
    agent._client.messages.create.side_effect = [_message_response(None)]

    _messages, turns = await _run(agent)

    assert turns == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reason_code", sorted(UNRETRYABLE_DROPS))
async def test_backstop_ends_on_every_unretryable_drop(agent, reason_code):
    """Leaked tool syntax and a spent wake-cycle budget are as unretryable
    as filler: the same call comes back refused the same way. Untagged,
    they burned the whole turn cap in silence."""
    agent._tool_result_content = json.dumps({
        "status": "error", "message": "nope", "reason_code": reason_code,
    })
    agent._client.messages.create.side_effect = [_message_response(None)]

    _messages, turns = await _run(agent)

    assert turns == 1


@pytest.mark.asyncio
async def test_backstop_grants_a_retry_on_a_recoverable_drop(agent):
    """An unregistered recipient came back with the valid names — the
    model can fix that. Ending here delivers nothing, silently."""
    agent._tool_result_content = json.dumps({
        "status": "error", "message": "unknown recipient 'Jake'",
        "valid_recipients": ["Jacob"],
    })
    agent._client.messages.create.side_effect = [
        _message_response(None),
        _response(_text_block(_notes(True)), stop_reason="end_turn"),
    ]

    _messages, turns = await _run(agent)

    assert turns == 2


@pytest.mark.asyncio
async def test_backstop_is_trigger_channel_only(agent):
    """Interactive channels keep the status quo until this is validated."""
    agent._client.messages.create.side_effect = [
        _message_response(None),
        _response(_text_block(_notes(True)), stop_reason="end_turn"),
    ]

    _messages, turns = await _run(agent, channel="voice")

    assert turns == 2


@pytest.mark.asyncio
async def test_backstop_leaves_filler_then_tool_alone(agent):
    """The documented pattern: an interim ack alongside the real work.
    The batch has a non-message tool, so the run continues."""
    agent._client.messages.create.side_effect = [
        _response(
            _text_block(_notes(False)),
            _tool_use_block(
                "message", "t1", to="current_speaker", channel="text",
                content="One moment, looking that up.",
            ),
            _tool_use_block("execute_script", "t2"),
        ),
        _message_response(True, block_id="t3"),
    ]

    _messages, turns = await _run(agent)

    assert turns == 2


# ---------------------------------------------------------------------------
# OpenAI loop — same rules, driven through the adapter
# ---------------------------------------------------------------------------


def _completion(
    *, content: str | None, tool_calls: list[Any] | None,
    finish_reason: str,
) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=content, tool_calls=tool_calls, refusal=None,
            ),
            finish_reason=finish_reason,
        )],
        model="gpt-5.6-luna-2026-07-09",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
    )


def _openai_message_turn(
    final_turn: bool, call_id: str = "c1", arguments: str = "{}",
) -> Any:
    return _completion(
        content=_notes(final_turn),
        tool_calls=[SimpleNamespace(
            id=call_id, type="function",
            function=SimpleNamespace(name="message", arguments=arguments),
        )],
        finish_reason="tool_calls",
    )


@pytest.fixture
def openai_agent(agent):
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    agent._openai_client = client
    return agent


async def _run_openai(
    agent: BoxBotAgent, max_turns: int = 6, channel: str = "trigger",
):
    return await agent._agent_loop_openai(
        conversation_id="conv-test",
        channel=channel,
        system_prompt_blocks=[{"type": "text", "text": "sys"}],
        initial_message="wake",
        person_name="Jacob",
        model="gpt-5.6-luna",
        max_turns=max_turns,
    )


@pytest.mark.asyncio
async def test_openai_flag_ends_the_loop(openai_agent):
    openai_agent._openai_client.chat.completions.create.side_effect = [
        _openai_message_turn(True),
    ]

    _messages, turns = await _run_openai(openai_agent)

    assert turns == 1


@pytest.mark.asyncio
async def test_openai_backstop_ends_a_message_only_trigger_run(openai_agent):
    """OpenAI returns ``content: null`` beside tool calls, so a tool
    turn carries no notes at all — exactly what the backstop is for."""
    openai_agent._openai_client.chat.completions.create.side_effect = [
        _completion(
            content=None,
            tool_calls=[SimpleNamespace(
                id="c1", type="function",
                function=SimpleNamespace(name="message", arguments="{}"),
            )],
            finish_reason="tool_calls",
        ),
    ]

    _messages, turns = await _run_openai(openai_agent)

    assert turns == 1


@pytest.mark.asyncio
async def test_openai_backstop_grants_a_retry_on_a_recoverable_drop(openai_agent):
    """Same rule as the raw loop: a fixable delivery failure is not an
    end of turn."""
    openai_agent._tool_result_content = json.dumps({
        "status": "error", "message": "unknown recipient 'Jake'",
        "valid_recipients": ["Jacob"],
    })
    openai_agent._openai_client.chat.completions.create.side_effect = [
        _openai_message_turn(False),
        _completion(
            content=_notes(True), tool_calls=None, finish_reason="stop",
        ),
    ]

    _messages, turns = await _run_openai(openai_agent)

    assert turns == 2


@pytest.mark.asyncio
async def test_openai_malformed_arguments_override_the_flag(openai_agent):
    """An unparseable ``arguments`` string is an error the model
    has not seen yet, so it does not get to stop on it."""
    openai_agent._openai_client.chat.completions.create.side_effect = [
        _openai_message_turn(True, arguments="{not json"),
        _completion(
            content=_notes(True), tool_calls=None, finish_reason="stop",
        ),
    ]

    _messages, turns = await _run_openai(openai_agent)

    assert turns == 2
