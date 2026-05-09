"""Tests for the audio playback path: AudioPlayer + audio.play action.

Decoding via miniaudio is mocked so the test box doesn't need the
package or any sample files. The point of these tests is to exercise
the *contract* between the SDK and the main process: validation,
caps, volume snapshot/restore, interrupted vs. natural-completion
detection, and the dispatcher's path/format/quota gates.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# AudioPlayer
# ---------------------------------------------------------------------------


class _FakeSpeaker:
    """Stand-in for :class:`boxbot.hardware.speaker.Speaker`.

    Records the bytes/sample-rate/channels passed to ``play`` and
    holds a configurable volume. ``play`` is a coroutine so the
    AudioPlayer can ``await`` it like the real thing.

    ``play`` blocks for ``play_duration_ms`` so the player's elapsed-vs-
    decoded comparison reaches the natural-completion branch. Tests
    that want to exercise the interrupt path set this lower than the
    decoded length on the next call.
    """

    def __init__(self, *, volume: float = 0.8):
        self._volume = volume
        self.played: list[tuple[bytes, int, int]] = []
        # Sleep this long inside ``play``. Set per-test to bracket
        # the player's 250 ms interrupt threshold either side.
        self.play_duration_ms: int = 0

    def get_volume(self) -> float:
        return self._volume

    def set_volume(self, v: float) -> None:
        self._volume = v

    async def play(self, pcm: bytes, sample_rate: int, channels: int) -> None:
        self.played.append((pcm, sample_rate, channels))
        if self.play_duration_ms > 0:
            import asyncio as _a
            await _a.sleep(self.play_duration_ms / 1000.0)


def _patch_decode(monkeypatch, *, duration_ms: int,
                  source_sr: int = 44100,
                  source_ch: int = 2):
    """Replace ``_decode_to_pcm`` with a stub returning fixed PCM."""
    from boxbot.communication import audio_player

    def fake_decode(abs_path: Path):
        # Roughly the right number of bytes: 24kHz mono int16
        n_samples = int(duration_ms * 24000 / 1000)
        return audio_player._DecodedAudio(
            pcm_bytes=b"\x00\x00" * n_samples,
            duration_ms=duration_ms,
            source_sample_rate=source_sr,
            source_channels=source_ch,
        )

    monkeypatch.setattr(audio_player, "_decode_to_pcm", fake_decode)


@pytest.mark.asyncio
async def test_player_decodes_and_writes_to_speaker(tmp_path, monkeypatch):
    from boxbot.communication.audio_player import AudioPlayer

    f = tmp_path / "x.wav"
    f.write_bytes(b"\x00" * 100)
    _patch_decode(monkeypatch, duration_ms=500)

    speaker = _FakeSpeaker()
    # Mirror real-speaker behaviour: drain takes the decoded length.
    speaker.play_duration_ms = 500
    player = AudioPlayer(speaker)
    result = await player.play_file(f)

    assert speaker.played, "speaker.play was never called"
    pcm, sr, ch = speaker.played[0]
    assert sr == 24000
    assert ch == 1
    assert result.duration_ms == 500
    assert result.file_format == "wav"
    assert result.sample_rate == 44100
    assert result.channels == 2
    assert result.interrupted is False


@pytest.mark.asyncio
async def test_player_detects_interrupted_when_speaker_returns_early(
    tmp_path, monkeypatch,
):
    from boxbot.communication.audio_player import AudioPlayer

    f = tmp_path / "x.wav"
    f.write_bytes(b"\x00" * 100)
    _patch_decode(monkeypatch, duration_ms=2000)

    speaker = _FakeSpeaker()
    # Speaker returns after 100 ms — short of the 2 s decoded length
    # by more than the 250 ms slop, so the player flags interrupt.
    speaker.play_duration_ms = 100
    player = AudioPlayer(speaker)
    result = await player.play_file(f)
    assert result.interrupted is True
    assert result.elapsed_ms < result.duration_ms


@pytest.mark.asyncio
async def test_player_volume_snapshot_and_restore(tmp_path, monkeypatch):
    from boxbot.communication.audio_player import AudioPlayer

    f = tmp_path / "x.wav"
    f.write_bytes(b"\x00" * 10)
    _patch_decode(monkeypatch, duration_ms=100)

    speaker = _FakeSpeaker(volume=0.8)
    speaker.play_duration_ms = 100
    player = AudioPlayer(speaker)
    await player.play_file(f, volume=0.3)

    # Volume was set to 0.3 during play, restored to 0.8 after.
    assert speaker.get_volume() == 0.8


@pytest.mark.asyncio
async def test_player_rejects_unsupported_extension(tmp_path):
    from boxbot.communication.audio_player import AudioPlayer, AudioPlayerError

    f = tmp_path / "x.aiff"
    f.write_bytes(b"\x00")
    speaker = _FakeSpeaker()
    player = AudioPlayer(speaker)
    with pytest.raises(AudioPlayerError, match="unsupported"):
        await player.play_file(f)


@pytest.mark.asyncio
async def test_player_rejects_missing_file(tmp_path):
    from boxbot.communication.audio_player import AudioPlayer, AudioPlayerError

    speaker = _FakeSpeaker()
    player = AudioPlayer(speaker)
    with pytest.raises(AudioPlayerError, match="not found"):
        await player.play_file(tmp_path / "nope.wav")


@pytest.mark.asyncio
async def test_player_rejects_oversized_file(tmp_path):
    from boxbot.communication.audio_player import AudioPlayer, AudioPlayerError

    f = tmp_path / "big.wav"
    f.write_bytes(b"\x00" * 1024)
    speaker = _FakeSpeaker()
    player = AudioPlayer(speaker, max_file_bytes=512)
    with pytest.raises(AudioPlayerError, match="exceeds cap"):
        await player.play_file(f)


@pytest.mark.asyncio
async def test_player_rejects_overlong_decoded_audio(tmp_path, monkeypatch):
    from boxbot.communication.audio_player import AudioPlayer, AudioPlayerError

    f = tmp_path / "long.wav"
    f.write_bytes(b"\x00" * 100)
    _patch_decode(monkeypatch, duration_ms=120_000)  # 2 minutes

    speaker = _FakeSpeaker()
    player = AudioPlayer(speaker, max_duration_seconds=60.0)
    with pytest.raises(AudioPlayerError, match="exceeds"):
        await player.play_file(f)


# ---------------------------------------------------------------------------
# audio.play action handler
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace_root(tmp_path, monkeypatch):
    """Point WORKSPACE_DIR at a temp directory for the duration of the test."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    from boxbot.core import paths
    monkeypatch.setattr(paths, "WORKSPACE_DIR", ws)
    # _sandbox_actions imports it inside the handler — patch there too
    # so the resolved object matches.
    return ws


@pytest.fixture
def fake_voice_session(monkeypatch):
    """Install a mock voice session with an awaitable ``play_audio``."""
    from boxbot.communication import voice as voice_module
    from boxbot.communication.audio_player import PlaybackResult

    session = MagicMock()
    session.play_audio = AsyncMock(return_value=PlaybackResult(
        duration_ms=1000,
        elapsed_ms=1000,
        interrupted=False,
        file_format="wav",
        sample_rate=24000,
        channels=1,
    ))
    monkeypatch.setattr(voice_module, "_voice_session", session)
    return session


@pytest.mark.asyncio
async def test_action_rejects_empty_path(workspace_root, fake_voice_session):
    from boxbot.tools._sandbox_actions import (
        ActionContext, _handle_audio_action,
    )

    ctx = ActionContext()
    result = await _handle_audio_action("audio.play", {"path": ""}, ctx)
    assert result["status"] == "error"
    assert "non-empty" in result["error"]
    assert not fake_voice_session.play_audio.called


@pytest.mark.asyncio
async def test_action_rejects_dotdot(workspace_root, fake_voice_session):
    from boxbot.tools._sandbox_actions import (
        ActionContext, _handle_audio_action,
    )

    ctx = ActionContext()
    result = await _handle_audio_action(
        "audio.play", {"path": "../../etc/passwd"}, ctx,
    )
    assert result["status"] == "error"
    assert "workspace-relative" in result["error"]
    assert not fake_voice_session.play_audio.called


@pytest.mark.asyncio
async def test_action_rejects_absolute_path(workspace_root, fake_voice_session):
    from boxbot.tools._sandbox_actions import (
        ActionContext, _handle_audio_action,
    )

    ctx = ActionContext()
    result = await _handle_audio_action(
        "audio.play", {"path": "/etc/passwd"}, ctx,
    )
    assert result["status"] == "error"
    assert not fake_voice_session.play_audio.called


@pytest.mark.asyncio
async def test_action_rejects_missing_file(workspace_root, fake_voice_session):
    from boxbot.tools._sandbox_actions import (
        ActionContext, _handle_audio_action,
    )

    ctx = ActionContext()
    result = await _handle_audio_action(
        "audio.play", {"path": "audio/nope.wav"}, ctx,
    )
    assert result["status"] == "error"
    assert "not found" in result["error"]


@pytest.mark.asyncio
async def test_action_rejects_unsupported_format(
    workspace_root, fake_voice_session,
):
    from boxbot.tools._sandbox_actions import (
        ActionContext, _handle_audio_action,
    )

    f = workspace_root / "audio" / "x.aiff"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"\x00")

    ctx = ActionContext()
    result = await _handle_audio_action(
        "audio.play", {"path": "audio/x.aiff"}, ctx,
    )
    assert result["status"] == "error"
    assert "unsupported" in result["error"]


@pytest.mark.asyncio
async def test_action_rejects_volume_out_of_range(
    workspace_root, fake_voice_session,
):
    from boxbot.tools._sandbox_actions import (
        ActionContext, _handle_audio_action,
    )

    f = workspace_root / "audio" / "chime.wav"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"\x00")

    ctx = ActionContext()
    result = await _handle_audio_action(
        "audio.play", {"path": "audio/chime.wav", "volume": 5.0}, ctx,
    )
    assert result["status"] == "error"
    assert "between 0.0 and 1.0" in result["error"]


@pytest.mark.asyncio
async def test_action_routes_to_voice_session(
    workspace_root, fake_voice_session,
):
    from boxbot.tools._sandbox_actions import (
        ActionContext, _handle_audio_action,
    )

    f = workspace_root / "audio" / "chime.wav"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"\x00")

    ctx = ActionContext()
    result = await _handle_audio_action(
        "audio.play", {"path": "audio/chime.wav", "volume": 0.5}, ctx,
    )

    assert result["status"] == "ok"
    assert result["duration_ms"] == 1000
    assert result["format"] == "wav"
    fake_voice_session.play_audio.assert_awaited_once()
    args, kwargs = fake_voice_session.play_audio.call_args
    assert args[0].name == "chime.wav"
    assert kwargs["volume"] == 0.5


@pytest.mark.asyncio
async def test_action_surfaces_interrupted_status(
    workspace_root, monkeypatch,
):
    from boxbot.communication import voice as voice_module
    from boxbot.communication.audio_player import PlaybackResult
    from boxbot.tools._sandbox_actions import (
        ActionContext, _handle_audio_action,
    )

    session = MagicMock()
    session.play_audio = AsyncMock(return_value=PlaybackResult(
        duration_ms=30000, elapsed_ms=4000, interrupted=True,
        file_format="mp3", sample_rate=44100, channels=2,
    ))
    monkeypatch.setattr(voice_module, "_voice_session", session)

    f = workspace_root / "music" / "song.mp3"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"\x00")

    ctx = ActionContext()
    result = await _handle_audio_action(
        "audio.play", {"path": "music/song.mp3"}, ctx,
    )

    assert result["status"] == "interrupted"
    assert result["elapsed_ms"] == 4000
    assert result["format"] == "mp3"


@pytest.mark.asyncio
async def test_action_errors_when_no_voice_session(workspace_root, monkeypatch):
    from boxbot.communication import voice as voice_module
    from boxbot.tools._sandbox_actions import (
        ActionContext, _handle_audio_action,
    )

    monkeypatch.setattr(voice_module, "_voice_session", None)

    f = workspace_root / "audio" / "chime.wav"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"\x00")

    ctx = ActionContext()
    result = await _handle_audio_action(
        "audio.play", {"path": "audio/chime.wav"}, ctx,
    )
    assert result["status"] == "error"
    assert "voice session" in result["error"]


@pytest.mark.asyncio
async def test_action_unknown_subaction_errors(
    workspace_root, fake_voice_session,
):
    from boxbot.tools._sandbox_actions import (
        ActionContext, _handle_audio_action,
    )

    ctx = ActionContext()
    result = await _handle_audio_action("audio.bogus", {}, ctx)
    assert result["status"] == "error"
    assert "unknown audio action" in result["error"]
