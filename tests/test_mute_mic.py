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
