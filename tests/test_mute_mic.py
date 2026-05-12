"""Tests for the mute_mic primitive: AudioCapture mute gate, the
``mute_mic`` tool, and the speech-handler auto-unmute contract.

These tests deliberately avoid spinning up the full voice session.
The contract is small: a flag at the audio_capture gate, a tool that
flips it on the active voice session, and an unmute call at TTS-end
inside the speak() context manager.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from boxbot.communication.audio_capture import AudioCapture
from boxbot.hardware.base import AudioChunk


def _make_capture():
    from boxbot.core.config import TurnDetectionConfig, VADConfig
    from boxbot.communication.vad import VoiceActivityDetector

    vad = VoiceActivityDetector(VADConfig())
    # Pretend VAD always reports speech so a non-muted chunk would
    # definitely accumulate.
    vad.process_chunk = AsyncMock(return_value=0.99)
    return AudioCapture(vad, TurnDetectionConfig())


def _make_speech_chunk(frames: int = 1024) -> AudioChunk:
    data = (np.ones(frames, dtype=np.int16) * 10_000).tobytes()
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=16000,
        channels=1,
        frames=frames,
    )


# ---------------------------------------------------------------------------
# AudioCapture mute gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_muted_capture_drops_chunks_before_vad():
    capture = _make_capture()
    callback = AsyncMock()
    capture.set_utterance_callback(callback)

    capture.mute()
    assert capture.is_muted is True

    # Even with simulated speech, nothing should be buffered or fired.
    for _ in range(20):
        await capture._on_audio_chunk(_make_speech_chunk())

    # VAD should never have been consulted while muted.
    assert capture._vad.process_chunk.await_count == 0
    assert len(capture._buffer) == 0
    callback.assert_not_called()


@pytest.mark.asyncio
async def test_mute_drops_in_flight_buffer():
    capture = _make_capture()
    capture.set_utterance_callback(AsyncMock())

    # Accumulate some speech first (not muted).
    for _ in range(3):
        await capture._on_audio_chunk(_make_speech_chunk())
    assert capture._is_speaking is True
    assert len(capture._buffer) > 0

    capture.mute()

    # Mute should have reset() the buffer so nothing leaks out post-unmute.
    assert capture._is_speaking is False
    assert len(capture._buffer) == 0


@pytest.mark.asyncio
async def test_unmute_resumes_capture():
    capture = _make_capture()
    callback = AsyncMock()
    capture.set_utterance_callback(callback)

    capture.mute()
    for _ in range(5):
        await capture._on_audio_chunk(_make_speech_chunk())
    assert len(capture._buffer) == 0

    capture.unmute()
    assert capture.is_muted is False

    for _ in range(3):
        await capture._on_audio_chunk(_make_speech_chunk())
    # VAD ran, speech accumulated.
    assert capture._vad.process_chunk.await_count == 3
    assert len(capture._buffer) > 0


def test_mute_unmute_idempotent():
    capture = _make_capture()
    capture.mute()
    capture.mute()  # no error, still muted
    assert capture.is_muted is True
    capture.unmute()
    capture.unmute()  # no error, still unmuted
    assert capture.is_muted is False


# ---------------------------------------------------------------------------
# mute_mic tool
# ---------------------------------------------------------------------------


def _patch_voice_session(monkeypatch, session):
    from boxbot.communication import voice as voice_mod
    monkeypatch.setattr(voice_mod, "_voice_session", session)


def _patch_conversation(monkeypatch, conv):
    from boxbot.tools import _tool_context

    token = _tool_context.current_conversation.set(conv)
    # Return the token so the test can clean up if it likes; pytest
    # tears the contextvar down with the test anyway.
    return token


class _FakeConversation:
    def __init__(self, channel: str = "voice"):
        self.channel = channel
        self.conversation_id = "test-conv"
        self._pending_inputs: list = ["queued garble"]


@pytest.mark.asyncio
async def test_mute_mic_tool_mutes_voice_session(monkeypatch):
    from boxbot.tools.builtins.mute_mic import MuteMicTool

    capture = _make_capture()
    session = MagicMock()
    session.mute_mic = MagicMock(return_value=True)
    # mute_mic tool only touches session.mute_mic(), but route it to
    # the capture so we can assert the audio-side flag too.
    session.mute_mic.side_effect = lambda: (capture.mute(), True)[1]

    conv = _FakeConversation(channel="voice")
    _patch_conversation(monkeypatch, conv)
    _patch_voice_session(monkeypatch, session)

    result = json.loads(await MuteMicTool().execute(reason="ambient chatter"))

    assert result["status"] == "muted"
    assert result["reason"] == "ambient chatter"
    session.mute_mic.assert_called_once()
    assert capture.is_muted is True
    # The tool should clear queued inputs so they don't sneak through later.
    assert conv._pending_inputs == []


@pytest.mark.asyncio
async def test_mute_mic_tool_no_op_on_whatsapp(monkeypatch):
    from boxbot.tools.builtins.mute_mic import MuteMicTool

    session = MagicMock()
    session.mute_mic = MagicMock(return_value=True)
    conv = _FakeConversation(channel="whatsapp")
    _patch_conversation(monkeypatch, conv)
    _patch_voice_session(monkeypatch, session)

    result = json.loads(await MuteMicTool().execute())

    assert result["status"] == "noop"
    session.mute_mic.assert_not_called()
    # Pending inputs left alone on non-voice channels.
    assert conv._pending_inputs == ["queued garble"]


@pytest.mark.asyncio
async def test_mute_mic_tool_no_op_when_no_session(monkeypatch):
    from boxbot.tools.builtins.mute_mic import MuteMicTool

    conv = _FakeConversation(channel="voice")
    _patch_conversation(monkeypatch, conv)
    _patch_voice_session(monkeypatch, None)

    result = json.loads(await MuteMicTool().execute())

    assert result["status"] == "noop"


@pytest.mark.asyncio
async def test_mute_mic_tool_no_op_outside_conversation(monkeypatch):
    from boxbot.tools.builtins.mute_mic import MuteMicTool
    from boxbot.tools import _tool_context

    _tool_context.current_conversation.set(None)
    result = json.loads(await MuteMicTool().execute())
    assert result["status"] == "noop"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


def test_mute_mic_in_registry():
    from boxbot.tools.registry import get_tool, get_tools

    tool = get_tool("mute_mic")
    assert tool is not None
    assert tool.name == "mute_mic"

    names = {t.name for t in get_tools()}
    assert "mute_mic" in names


# ---------------------------------------------------------------------------
# Inject-don't-interrupt: agent loop folds drained pending inputs
# ---------------------------------------------------------------------------


def _make_text_block(text: str):
    """Anthropic SDK-shaped text content block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(name: str, args: dict, tool_use_id: str = "tu-1"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = args
    block.id = tool_use_id
    return block


def _make_response(content_blocks, stop_reason: str):
    resp = MagicMock()
    resp.content = content_blocks
    resp.stop_reason = stop_reason
    # Usage fields the cost tracking reads.
    resp.usage = MagicMock(
        input_tokens=0, output_tokens=0,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    )
    return resp


@pytest.mark.asyncio
async def test_agent_loop_folds_pending_into_tool_result_user_turn(mock_config):
    """Inject-don't-interrupt: the agent loop's middle path appends
    drained Conversation._pending_inputs to the role:"user" turn that
    carries tool_result blocks. This is the Claude Code pattern — the
    model sees tool_results AND new user input in one round-trip."""
    from boxbot.core.agent import BoxBotAgent
    from boxbot.core.conversation import Conversation
    from boxbot.tools.registry import _ensure_loaded

    _ensure_loaded()

    # Two-call sequence: tool_use → end_turn.
    call_log: list[list[dict]] = []

    async def fake_create(**kwargs):
        call_log.append(list(kwargs["messages"]))
        if len(call_log) == 1:
            return _make_response(
                [
                    _make_text_block('{"thought":"looking up","observations":[]}'),
                    _make_tool_use_block(
                        "search_memory",
                        {"mode": "lookup", "query": "Zara dental"},
                        tool_use_id="tu-1",
                    ),
                ],
                stop_reason="tool_use",
            )
        return _make_response(
            [
                _make_text_block(
                    '{"thought":"answered","observations":[]}'
                ),
            ],
            stop_reason="end_turn",
        )

    # Stub search_memory so we don't hit the real memory store.
    from boxbot.tools.builtins.search_memory import SearchMemoryTool
    original_execute = SearchMemoryTool.execute

    async def stub_execute(self, **kwargs):
        return '{"results": [{"id": "m1", "text": "Wed May 13 8:30"}]}'

    SearchMemoryTool.execute = stub_execute  # type: ignore[assignment]
    try:
        # Build a minimal agent with the fake client.
        mem = MagicMock()
        mem.read_system_memory = MagicMock(return_value="")
        agent = BoxBotAgent(memory_store=mem)
        agent._client = MagicMock()
        agent._client.messages = MagicMock()
        agent._client.messages.create = fake_create
        agent._running = True
        agent._cost_tracker = MagicMock()
        agent._cost_tracker.record_turn = MagicMock()

        # Build a conversation by hand so we can pre-load pending inputs.
        conv = Conversation(
            conversation_id="conv_test",
            channel="voice",
            channel_key="voice:room",
            generate_fn=AsyncMock(),
            silence_timeout=0,
        )
        agent._conversations["conv_test"] = conv

        # Pre-load some "queued during THINKING" inputs.
        conv._pending_inputs = [
            {"role": "user", "content": "[Kid]: Coming"},
            {"role": "user", "content": "[Kid]: Thinking?"},
        ]

        # Run one agent-loop pass directly.
        messages, _turns = await agent._agent_loop(
            conversation_id="conv_test",
            channel="voice",
            system_prompt_blocks=[
                {"type": "text", "text": "test system prompt"},
            ],
            initial_message="When's Zara's dental?",
            person_name="Jacob",
            max_turns=4,
            prior_history=None,
        )
    finally:
        SearchMemoryTool.execute = original_execute  # type: ignore[assignment]

    # Two API calls happened.
    assert len(call_log) == 2

    # Second call's messages list contains the user turn that carries
    # tool_result AND the two drained pending inputs as text blocks.
    second_call_messages = call_log[1]
    # Find the user turn whose content is a list (multi-block).
    user_multi_blocks = [
        m for m in second_call_messages
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert len(user_multi_blocks) >= 1
    final_user_turn = user_multi_blocks[-1]
    block_types = [b.get("type") for b in final_user_turn["content"]]
    assert "tool_result" in block_types
    text_blocks = [
        b for b in final_user_turn["content"] if b.get("type") == "text"
    ]
    text_payloads = [b.get("text", "") for b in text_blocks]
    assert any("Coming" in t for t in text_payloads), text_payloads
    assert any("Thinking" in t for t in text_payloads), text_payloads

    # Drain emptied the queue.
    assert conv._pending_inputs == []


# ---------------------------------------------------------------------------
# Wake-word interrupt carve-out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_requested_event_calls_conv_interrupt():
    """When the wake word fires during an active conversation, voice.py
    publishes ConversationInterruptRequested. The agent's handler
    must call Conversation.interrupt() on the targeted conversation —
    the explicit carve-out against inject-don't-interrupt."""
    from boxbot.core.agent import BoxBotAgent
    from boxbot.core.events import ConversationInterruptRequested

    mem = MagicMock()
    mem.read_system_memory = MagicMock(return_value="")
    agent = BoxBotAgent(memory_store=mem)
    agent._client = MagicMock()
    agent._running = True

    conv = MagicMock()
    conv.conversation_id = "conv_test"
    conv.interrupt = AsyncMock()
    agent._conversations["conv_test"] = conv

    event = ConversationInterruptRequested(conversation_id="conv_test")
    await agent._on_conversation_interrupt_requested(event)

    conv.interrupt.assert_awaited_once()


@pytest.mark.asyncio
async def test_interrupt_requested_unknown_conversation_is_noop():
    from boxbot.core.agent import BoxBotAgent
    from boxbot.core.events import ConversationInterruptRequested

    mem = MagicMock()
    mem.read_system_memory = MagicMock(return_value="")
    agent = BoxBotAgent(memory_store=mem)
    agent._client = MagicMock()
    agent._running = True

    event = ConversationInterruptRequested(conversation_id="conv_does_not_exist")
    # Should not raise.
    await agent._on_conversation_interrupt_requested(event)
