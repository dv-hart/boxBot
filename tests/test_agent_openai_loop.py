"""Tests for ``_agent_loop_openai`` and fast-tier model routing.

The loop must behave identically to ``_agent_loop`` where it matters:
same signature and return shape, same turn cap + final-turn
``message``-only filter, same max-turns fallback, one cost row per turn.
What differs is tested explicitly: structured output pinned via
``response_format``, reasoning off, and malformed tool arguments fed
back to the model instead of reaching a tool.

The OpenAI client is mocked throughout — no live calls.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from boxbot.core.agent import BoxBotAgent
from boxbot.core.output_dispatcher import INTERNAL_NOTES_SCHEMA

NOTES = '{"thought":"working","observations":[]}'

# The id we request vs the snapshot Chat Completions echoes back. They
# differ on purpose: pricing.yaml keys on the alias, so a fixture that
# echoes the alias would mask a $0.00 billing bug.
ALIAS = "gpt-5.6-luna"
SNAPSHOT = "gpt-5.6-luna-2026-07-09"


# ---------------------------------------------------------------------------
# Completion builders — match the OpenAI Chat Completions shape
# ---------------------------------------------------------------------------


def _tool_call(call_id: str, name: str, arguments: str = "{}") -> Any:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _completion(
    *,
    content: str | None = NOTES,
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    refusal: str | None = None,
) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=content, tool_calls=tool_calls, refusal=refusal,
            ),
            finish_reason=finish_reason,
        )],
        model=SNAPSHOT,
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


def _tool_turn(name: str, call_id: str, arguments: str = "{}") -> Any:
    return _completion(
        content=None,
        tool_calls=[_tool_call(call_id, name, arguments)],
        finish_reason="tool_calls",
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_with_openai(monkeypatch, mock_config):
    mem = MagicMock()
    mem.read_system_memory = MagicMock(return_value="")
    agent = BoxBotAgent(memory_store=mem)
    agent._running = True

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    agent._openai_client = client

    dispatched: list[str] = []

    backends: list[str] = []

    async def _fake_process(
        response, tools, *, conversation_id=None, turn_number=None,
        channel=None, backend=None,
    ):
        backends.append(backend)
        results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            dispatched.append(block.name)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": '{"status":"delivered"}',
            })
        return results

    monkeypatch.setattr(agent, "_process_tool_calls", _fake_process)
    agent._dispatched_tools = dispatched
    agent._tool_backends = backends

    async def _noop_cost(*a, **kw):
        return None
    monkeypatch.setattr("boxbot.core.agent.record_cost", _noop_cost)

    def _fake_tool_definitions(_tools):
        return [
            {"name": "message", "description": "deliver", "input_schema": {}},
            {"name": "execute_script", "description": "x",
             "input_schema": {"type": "object", "properties": {}}},
        ]
    monkeypatch.setattr(
        agent, "_build_tool_definitions", _fake_tool_definitions,
    )
    monkeypatch.setattr("boxbot.tools.registry.get_tools", lambda: [])
    return agent


async def _run(
    agent, *, max_turns=25, channel="voice", prior_history=None,
    model=ALIAS,
):
    return await agent._agent_loop_openai(
        conversation_id="conv-1",
        channel=channel,
        system_prompt_blocks=[
            {"type": "text", "text": "static"},
            {"type": "text", "text": "dynamic"},
        ],
        initial_message="hello",
        person_name="Jacob",
        model=model,
        max_turns=max_turns,
        prior_history=prior_history,
    )


def _create_kwargs(agent, index: int = 0) -> dict[str, Any]:
    return agent._openai_client.chat.completions.create.call_args_list[
        index
    ].kwargs


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------


def test_voice_uses_fast_tier_when_configured(mock_config, monkeypatch):
    mock_config.models.fast = "gpt-5.6-luna"
    mock_config.models.large = "claude-opus-4-7"
    assert BoxBotAgent._resolve_model("voice") == "gpt-5.6-luna"


def test_non_voice_channels_stay_on_large(mock_config):
    mock_config.models.fast = "gpt-5.6-luna"
    mock_config.models.large = "claude-opus-4-7"
    for channel in ("whatsapp", "signal", "trigger"):
        assert BoxBotAgent._resolve_model(channel) == "claude-opus-4-7"


def test_fast_tier_off_keeps_voice_on_large(mock_config):
    mock_config.models.fast = None
    mock_config.models.large = "claude-opus-4-7"
    assert BoxBotAgent._resolve_model("voice") == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_generate_dispatches_to_openai_loop_for_gpt_model(
    mock_config, monkeypatch,
):
    """Provider is derived from the resolved id, not from agent.backend."""
    mock_config.models.fast = "gpt-5.6-luna"
    mock_config.agent.backend = "claude_agent_sdk"

    agent = BoxBotAgent(memory_store=MagicMock())
    agent._client = MagicMock()

    conv = MagicMock()
    conv.thread = [{"role": "user", "content": "hi"}]
    conv.channel = "voice"
    conv.conversation_id = "conv-1"
    conv.current_context = None
    conv.injected_memories_block = ""

    async def _blocks(**kw):
        return []
    monkeypatch.setattr(agent, "_build_system_prompt_blocks", _blocks)
    monkeypatch.setattr(agent, "_get_most_recent_person", lambda: "Jacob")

    openai_loop = AsyncMock(return_value=([], 1))
    sdk_loop = AsyncMock(return_value=([], 1))
    raw_loop = AsyncMock(return_value=([], 1))
    monkeypatch.setattr(agent, "_agent_loop_openai", openai_loop)
    monkeypatch.setattr(agent, "_agent_loop_sdk", sdk_loop)
    monkeypatch.setattr(agent, "_agent_loop", raw_loop)

    await agent._generate_for_conversation(conv)

    openai_loop.assert_awaited_once()
    assert openai_loop.await_args.kwargs["model"] == "gpt-5.6-luna"
    sdk_loop.assert_not_awaited()
    raw_loop.assert_not_awaited()


@pytest.mark.asyncio
async def test_anthropic_fast_tier_id_reaches_the_api_call(
    mock_config, monkeypatch,
):
    """Finding #4: an Anthropic ``models.fast`` must not be a no-op.

    ``_agent_loop`` used to re-read ``models.large`` and ignore the
    resolved id entirely.
    """
    mock_config.models.large = "claude-opus-4-7"
    agent = BoxBotAgent(memory_store=MagicMock())
    agent._client = MagicMock()
    agent._client.messages = MagicMock()
    agent._client.messages.create = AsyncMock(return_value=SimpleNamespace(
        content=[SimpleNamespace(type="text", text=NOTES)],
        stop_reason="end_turn",
        model="claude-haiku-4-5-20251001",
        usage=SimpleNamespace(input_tokens=10, output_tokens=2),
    ))
    monkeypatch.setattr("boxbot.tools.registry.get_tools", lambda: [])
    monkeypatch.setattr(agent, "_build_tool_definitions", lambda _t: [])

    async def _noop_cost(*a, **kw):
        return None
    monkeypatch.setattr("boxbot.core.agent.record_cost", _noop_cost)

    await agent._agent_loop(
        conversation_id="conv-1",
        channel="voice",
        system_prompt_blocks=[{"type": "text", "text": "sys"}],
        initial_message="hello",
        person_name="Jacob",
        model="claude-haiku-4-5-20251001",
        max_turns=3,
    )

    kwargs = agent._client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_anthropic_fast_tier_id_reaches_the_sdk_loop(
    mock_config, monkeypatch,
):
    """Same for the Agent SDK backend: the resolved id wins."""
    mock_config.models.fast = "claude-haiku-4-5-20251001"
    mock_config.models.large = "claude-opus-4-7"
    mock_config.agent.backend = "claude_agent_sdk"

    agent = BoxBotAgent(memory_store=MagicMock())
    agent._client = MagicMock()

    conv = MagicMock()
    conv.thread = [{"role": "user", "content": "hi"}]
    conv.channel = "voice"
    conv.conversation_id = "conv-1"
    conv.current_context = None

    async def _blocks(**kw):
        return []
    monkeypatch.setattr(agent, "_build_system_prompt_blocks", _blocks)
    monkeypatch.setattr(agent, "_get_most_recent_person", lambda: None)

    sdk_loop = AsyncMock(return_value=([], 1))
    monkeypatch.setattr(agent, "_agent_loop_sdk", sdk_loop)

    await agent._generate_for_conversation(conv)

    assert sdk_loop.await_args.kwargs["model"] == "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_generate_keeps_text_channels_on_the_anthropic_loop(
    mock_config, monkeypatch,
):
    mock_config.models.fast = "gpt-5.6-luna"
    mock_config.models.large = "claude-opus-4-7"
    mock_config.agent.backend = "raw_anthropic"

    agent = BoxBotAgent(memory_store=MagicMock())
    agent._client = MagicMock()

    conv = MagicMock()
    conv.thread = [{"role": "user", "content": "hi"}]
    conv.channel = "whatsapp"
    conv.conversation_id = "conv-2"
    conv.current_context = None

    async def _blocks(**kw):
        return []
    monkeypatch.setattr(agent, "_build_system_prompt_blocks", _blocks)
    monkeypatch.setattr(agent, "_get_most_recent_person", lambda: None)

    openai_loop = AsyncMock(return_value=([], 1))
    raw_loop = AsyncMock(return_value=([], 1))
    monkeypatch.setattr(agent, "_agent_loop_openai", openai_loop)
    monkeypatch.setattr(agent, "_agent_loop", raw_loop)

    await agent._generate_for_conversation(conv)

    raw_loop.assert_awaited_once()
    assert raw_loop.await_args.kwargs["model"] == "claude-opus-4-7"
    openai_loop.assert_not_awaited()


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def test_missing_openai_key_raises_an_actionable_error(mock_config):
    mock_config.api_keys.openai = None
    agent = BoxBotAgent(memory_store=MagicMock())
    with pytest.raises(RuntimeError) as exc:
        agent._ensure_openai_client()
    msg = str(exc.value)
    assert "OPENAI_API_KEY" in msg
    assert "BOXBOT_MODEL_FAST" in msg


@pytest.mark.asyncio
async def test_boot_rejects_an_openai_models_large_without_a_key(mock_config):
    """The key check covers every model that can reach a loop."""
    mock_config.api_keys.anthropic = "sk-ant-test"
    mock_config.api_keys.openai = None
    mock_config.models.large = "gpt-5.6-luna"
    mock_config.models.fast = None

    agent = BoxBotAgent(memory_store=MagicMock())
    with pytest.raises(RuntimeError) as exc:
        await agent.start()
    assert "models.large = 'gpt-5.6-luna'" in str(exc.value)
    assert "OPENAI_API_KEY" in str(exc.value)


def test_azure_config_builds_an_azure_client(mock_config):
    """OPENAI_API_TYPE=azure must not fall through to public OpenAI."""
    import openai

    mock_config.api_keys.openai = "azure-key"
    mock_config.openai.api_type = "azure"
    mock_config.openai.api_base = "https://r.openai.azure.com/"
    mock_config.openai.api_version = "2025-01-01-preview"

    client = BoxBotAgent(memory_store=MagicMock())._ensure_openai_client()

    assert isinstance(client, openai.AsyncAzureOpenAI)


@pytest.mark.parametrize("missing", ["api_base", "api_version"])
def test_azure_without_endpoint_or_version_raises(mock_config, missing):
    """Silently using an Azure key against api.openai.com would 401."""
    mock_config.api_keys.openai = "azure-key"
    mock_config.openai.api_type = "azure"
    mock_config.openai.api_base = "https://r.openai.azure.com/"
    mock_config.openai.api_version = "2025-01-01-preview"
    setattr(mock_config.openai, missing, None)

    agent = BoxBotAgent(memory_store=MagicMock())
    with pytest.raises(RuntimeError) as exc:
        agent._ensure_openai_client()
    assert missing.upper().replace("API_", "OPENAI_API_") in str(exc.value)


def test_public_openai_config_builds_a_plain_client(mock_config):
    import openai

    mock_config.api_keys.openai = "sk-test"
    mock_config.openai.api_type = None
    mock_config.openai.api_base = None

    client = BoxBotAgent(memory_store=MagicMock())._ensure_openai_client()

    assert isinstance(client, openai.AsyncOpenAI)
    assert not isinstance(client, openai.AsyncAzureOpenAI)


def test_client_is_built_once(mock_config):
    mock_config.api_keys.openai = "sk-test"
    agent = BoxBotAgent(memory_store=MagicMock())
    first = agent._ensure_openai_client()
    assert agent._ensure_openai_client() is first


def test_client_gets_the_configured_timeout_and_retries(mock_config):
    """SDK defaults (600s, no early cutoff) let one server-side stall
    silence voice for minutes — the config knobs must reach the client."""
    mock_config.api_keys.openai = "sk-test"
    mock_config.agent.openai_timeout_seconds = 12.5
    mock_config.agent.openai_max_retries = 4

    client = BoxBotAgent(memory_store=MagicMock())._ensure_openai_client()

    assert client.timeout == 12.5
    assert client.max_retries == 4


# ---------------------------------------------------------------------------
# Call shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_turn_returns_history_and_turn_count(agent_with_openai):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _completion()
    ]

    messages, turns = await _run(agent)

    assert turns == 1
    assert messages[0] == {"role": "user", "content": "hello"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == [{"type": "text", "text": NOTES}]


@pytest.mark.asyncio
async def test_structured_output_and_reasoning_are_pinned(agent_with_openai):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _completion()
    ]

    await _run(agent)
    kwargs = _create_kwargs(agent)

    fmt = kwargs["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert set(schema["properties"]) == set(
        INTERNAL_NOTES_SCHEMA["properties"]
    )
    # Fast tier is a latency tier — no thinking.
    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["model"] == ALIAS
    assert kwargs["max_completion_tokens"] == 8192


@pytest.mark.asyncio
async def test_reasoning_effort_follows_the_model_id(agent_with_openai):
    """Older reasoning ids bottom out at "minimal" — "none" 400s."""
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _completion()
    ]

    await _run(agent, model="gpt-5.1")

    assert _create_kwargs(agent)["reasoning_effort"] == "minimal"


@pytest.mark.asyncio
async def test_reasoning_effort_is_omitted_for_non_reasoning_ids(
    agent_with_openai,
):
    """gpt-4o rejects the parameter outright."""
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _completion()
    ]

    await _run(agent, model="gpt-4o")

    assert "reasoning_effort" not in _create_kwargs(agent)


@pytest.mark.asyncio
async def test_system_prompt_blocks_are_flattened_into_one_system_message(
    agent_with_openai,
):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _completion()
    ]

    await _run(agent)
    msgs = _create_kwargs(agent)["messages"]

    assert msgs[0] == {"role": "system", "content": "static\n\ndynamic"}
    assert msgs[1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_tools_are_translated_to_openai_function_shape(
    agent_with_openai,
):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _completion()
    ]

    await _run(agent)
    tools = _create_kwargs(agent)["tools"]

    assert [t["function"]["name"] for t in tools] == [
        "message", "execute_script",
    ]
    assert all(t["type"] == "function" for t in tools)


@pytest.mark.asyncio
async def test_prior_history_seeds_the_thread(agent_with_openai):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _completion()
    ]

    messages, _ = await _run(agent, prior_history=[
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": [{"type": "text", "text": NOTES}]},
    ])

    assert messages[0]["content"] == "earlier"
    assert messages[2] == {"role": "user", "content": "hello"}


# ---------------------------------------------------------------------------
# Tool loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_turn_feeds_results_back_and_continues(agent_with_openai):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _tool_turn("execute_script", "c1"),
        _completion(),
    ]

    messages, turns = await _run(agent)

    assert turns == 2
    assert agent._dispatched_tools == ["execute_script"]
    tool_turn = messages[2]
    assert tool_turn["role"] == "user"
    assert tool_turn["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "c1",
        "content": '{"status":"delivered"}',
    }
    # Second call carries the tool result as a role:"tool" message.
    second = _create_kwargs(agent, 1)["messages"]
    assert second[-1]["role"] == "tool"
    assert second[-1]["tool_call_id"] == "c1"


@pytest.mark.asyncio
async def test_malformed_arguments_never_reach_a_tool(agent_with_openai):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _tool_turn("execute_script", "c1", '{"code": '),
        _completion(),
    ]

    messages, _ = await _run(agent)

    assert agent._dispatched_tools == []
    result = messages[2]["content"][0]
    assert result["tool_use_id"] == "c1"
    assert "Re-issue the call" in json.loads(result["content"])["error"]
    # The call still exists in history so the id resolves.
    assert messages[1]["content"][0]["id"] == "c1"


# ---------------------------------------------------------------------------
# Turn cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_turn_offers_only_the_message_tool(agent_with_openai):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _tool_turn("execute_script", "c1"),
        _tool_turn("message", "c2"),
    ]

    _, turns = await _run(agent, max_turns=2)

    assert turns == 2
    assert [t["function"]["name"] for t in _create_kwargs(agent, 0)["tools"]] \
        == ["message", "execute_script"]
    assert [t["function"]["name"] for t in _create_kwargs(agent, 1)["tools"]] \
        == ["message"]


@pytest.mark.asyncio
async def test_penultimate_turn_gets_the_cap_heads_up(agent_with_openai):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _tool_turn("execute_script", "c1"),
        _tool_turn("message", "c2"),
    ]

    await _run(agent, max_turns=2)

    final_msgs = _create_kwargs(agent, 1)["messages"]
    trailing = final_msgs[-1]
    assert trailing["role"] == "user"
    text = trailing["content"][0]["text"]
    assert "turn cap (2 turns)" in text
    assert "``message``" in text


@pytest.mark.asyncio
async def test_cap_without_a_message_call_fires_the_fallback(
    agent_with_openai, monkeypatch,
):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _tool_turn("execute_script", "c1"),
        _tool_turn("execute_script", "c2"),
    ]
    fallback = AsyncMock()
    monkeypatch.setattr(agent, "_dispatch_max_turns_fallback", fallback)

    await _run(agent, max_turns=2)

    fallback.assert_awaited_once()
    assert fallback.await_args.kwargs["max_turns"] == 2


@pytest.mark.asyncio
async def test_malformed_message_on_the_final_turn_fires_the_fallback(
    agent_with_openai, monkeypatch,
):
    """Finding #1: the call is in history but never ran — dead air.

    The flag must follow what dispatched, not what the model asked for.
    """
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _tool_turn("message", "c1", '{"content": '),
    ]
    fallback = AsyncMock()
    monkeypatch.setattr(agent, "_dispatch_max_turns_fallback", fallback)

    await _run(agent, max_turns=1)

    assert agent._dispatched_tools == []
    fallback.assert_awaited_once()


@pytest.mark.asyncio
async def test_final_turn_arg_on_message_call_ends_loop_in_one_call(
    agent_with_openai,
):
    """``final_turn: true`` inside the message call's arguments ends the
    turn on that same response — the loop must not spend another
    full-context call just to collect the notes flag (Chat Completions
    returns ``content: null`` on tool-call responses)."""
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _tool_turn(
            "message", "c1",
            '{"to":"room","channel":"speak","content":"hi",'
            '"final_turn":true}',
        ),
    ]

    _messages, turns = await _run(agent)

    assert turns == 1
    assert agent._openai_client.chat.completions.create.call_count == 1
    assert agent._dispatched_tools == ["message"]


@pytest.mark.asyncio
async def test_final_turn_arg_beside_a_command_ends_the_loop(
    agent_with_openai,
):
    """Fire-and-forget actuation SOP: command + spoken ack + final_turn
    in ONE response ends the turn with no close-out call. Errored
    siblings still veto via _tool_results_ok (covered in
    test_agent_final_turn.py)."""
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _completion(
            content=None,
            tool_calls=[
                _tool_call(
                    "c1", "message",
                    '{"to":"room","channel":"speak",'
                    '"content":"Lock command sent.","final_turn":true}',
                ),
                _tool_call("c2", "execute_script", "{}"),
            ],
            finish_reason="tool_calls",
        ),
    ]

    _messages, turns = await _run(agent)

    assert turns == 1
    assert agent._dispatched_tools == ["message", "execute_script"]


@pytest.mark.asyncio
async def test_message_without_final_turn_arg_keeps_looping(
    agent_with_openai,
):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _tool_turn(
            "message", "c1",
            '{"to":"room","channel":"speak","content":"hi"}',
        ),
        _completion(
            content='{"thought":"done","observations":[],"final_turn":true}',
        ),
    ]

    _messages, turns = await _run(agent)

    assert turns == 2
    assert agent._openai_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_message_on_the_final_turn_skips_the_fallback(
    agent_with_openai, monkeypatch,
):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _tool_turn("execute_script", "c1"),
        _tool_turn("message", "c2"),
    ]
    fallback = AsyncMock()
    monkeypatch.setattr(agent, "_dispatch_max_turns_fallback", fallback)

    await _run(agent, max_turns=2)

    fallback.assert_not_awaited()


# ---------------------------------------------------------------------------
# Terminal stop reasons
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_length_finish_reason_stops_the_loop(agent_with_openai):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _completion(finish_reason="length"),
    ]
    _, turns = await _run(agent)
    assert turns == 1


@pytest.mark.asyncio
async def test_content_filter_stops_the_loop(agent_with_openai):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _completion(finish_reason="content_filter"),
    ]
    _, turns = await _run(agent)
    assert turns == 1


@pytest.mark.asyncio
async def test_refusal_is_logged_and_closed_out(
    agent_with_openai, monkeypatch, caplog,
):
    """Finding #2: a refusal arrives as finish_reason "stop".

    Unhandled it reads as a clean end_turn: empty assistant turn, no
    log, nothing said. On voice that is indistinguishable from a crash.
    """
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _completion(content=None, refusal="I won't do that."),
    ]
    close_out = AsyncMock()
    monkeypatch.setattr(agent, "_dispatch_close_out", close_out)

    with caplog.at_level("WARNING"):
        _, turns = await _run(agent)

    assert turns == 1
    assert "I won't do that." in caplog.text
    close_out.assert_awaited_once()
    assert close_out.await_args.kwargs["channel"] == "voice"
    assert close_out.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_tool_rows_are_labelled_with_the_openai_backend(
    agent_with_openai,
):
    """Latency/cost comparison across tiers reads this label."""
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _tool_turn("execute_script", "c1"),
        _completion(),
    ]

    await _run(agent)

    assert agent._tool_backends == ["openai"]


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_cost_row_per_turn_via_from_openai_usage(
    agent_with_openai, monkeypatch,
):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _tool_turn("execute_script", "c1"),
        _completion(),
    ]

    calls: list[dict[str, Any]] = []

    def _fake_usage(**kwargs):
        calls.append(kwargs)
        return "event"
    monkeypatch.setattr("boxbot.core.agent.from_openai_usage", _fake_usage)

    recorded: list[Any] = []

    async def _record(store, event):
        recorded.append(event)
    monkeypatch.setattr("boxbot.core.agent.record_cost", _record)

    await _run(agent, channel="voice")

    assert len(calls) == 2
    assert recorded == ["event", "event"]
    assert calls[0]["purpose"] == "conversation"
    assert calls[0]["model"] == ALIAS
    assert calls[0]["correlation_id"] == "conv-1"
    assert calls[0]["metadata"] == {
        "channel": "voice", "turn": 1, "response_model": SNAPSHOT,
    }
    assert calls[0]["usage"].prompt_tokens == 100


@pytest.mark.asyncio
async def test_cost_failure_does_not_break_the_turn(
    agent_with_openai, monkeypatch,
):
    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        _completion()
    ]

    def _boom(**kwargs):
        raise RuntimeError("pricing unavailable")
    monkeypatch.setattr("boxbot.core.agent.from_openai_usage", _boom)

    _, turns = await _run(agent)
    assert turns == 1


# ---------------------------------------------------------------------------
# API errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_error_retries_once_then_gives_up(
    agent_with_openai, monkeypatch,
):
    import openai

    async def _no_sleep(_seconds):
        return None
    monkeypatch.setattr("boxbot.core.agent.asyncio.sleep", _no_sleep)

    agent = agent_with_openai
    err = openai.APIError("boom", request=MagicMock(), body=None)
    agent._openai_client.chat.completions.create.side_effect = [err, err]

    messages, turns = await _run(agent)

    assert turns == 1
    assert agent._openai_client.chat.completions.create.call_count == 2
    assert messages[-1]["role"] == "assistant"
    assert "API error" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_api_error_then_success_continues(
    agent_with_openai, monkeypatch,
):
    import openai

    async def _no_sleep(_seconds):
        return None
    monkeypatch.setattr("boxbot.core.agent.asyncio.sleep", _no_sleep)

    agent = agent_with_openai
    agent._openai_client.chat.completions.create.side_effect = [
        openai.APIError("boom", request=MagicMock(), body=None),
        _completion(),
    ]

    messages, turns = await _run(agent)

    assert turns == 1
    assert messages[-1]["content"] == [{"type": "text", "text": NOTES}]
