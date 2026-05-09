"""Decode workspace audio files and play them through the speaker.

The agent calls ``bb.audio.play("audio/chime.wav")``; that routes to an
``audio.play`` action handler in the main process, which builds a path
under ``WORKSPACE_DIR`` and hands it to :class:`AudioPlayer`.

Decoding goes through ``miniaudio`` — a single pure-Python wheel that
covers WAV / FLAC / OGG-Vorbis / MP3 with no system dependency on
ffmpeg. The decoder targets int16 mono at 24 kHz so the speaker's
existing resample ladder doesn't have to deal with anything new (TTS
output is already 24 kHz mono int16; reusing that path keeps the AEC
reference timing aligned).

Interrupted vs. natural completion is decided after the speaker awaits
``play()``: if the wake-word handler called ``stop_playback()`` mid-
playback, the actual elapsed time will be shorter than the decoded
duration. The voice session sets a flag on its end (``_tts_interrupted``)
that the action-handler layer checks to fill in the SDK response.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from boxbot.hardware.speaker import Speaker


logger = logging.getLogger(__name__)


# Decoded PCM target — matches the speaker's TTS path so resampling and
# the AEC reference behave identically for arbitrary audio.
_TARGET_SAMPLE_RATE = 24000
_TARGET_CHANNELS = 1

_SUPPORTED_EXTS = {".wav", ".flac", ".ogg", ".mp3"}


class AudioPlayerError(Exception):
    """Raised for path/format/quota/decoding/playback failures."""


@dataclass
class PlaybackResult:
    duration_ms: int
    elapsed_ms: int
    interrupted: bool
    file_format: str          # "wav" | "flac" | "ogg" | "mp3"
    sample_rate: int          # source sample rate (pre-resample)
    channels: int             # source channel count


class AudioPlayer:
    """Decode an on-disk audio file and play it on the speaker.

    Stateless across calls: no playback queue of its own. Concurrent
    calls serialize at the speaker's ``_playback_lock``.
    """

    def __init__(
        self,
        speaker: "Speaker",
        *,
        max_duration_seconds: float = 300.0,
        max_file_bytes: int = 25 * 1024 * 1024,
    ) -> None:
        self._speaker = speaker
        self._max_duration_seconds = max_duration_seconds
        self._max_file_bytes = max_file_bytes

    async def play_file(
        self,
        abs_path: Path,
        *,
        volume: float | None = None,
    ) -> PlaybackResult:
        """Decode ``abs_path`` and play it through the speaker.

        Args:
            abs_path: Resolved absolute path to the audio file. The
                caller is responsible for confining this to the
                workspace; the player only checks existence + format
                + caps.
            volume: Optional per-call volume override (0.0–1.0). The
                speaker's previous level is restored on completion so
                the next TTS or audio call isn't affected.

        Returns:
            :class:`PlaybackResult` with decoded duration, actual
            elapsed time, and an ``interrupted`` flag derived from the
            elapsed-vs-decoded comparison.

        Raises:
            AudioPlayerError: file missing, unsupported format, file
                too large, decoded audio too long, decode failure.
        """
        ext = abs_path.suffix.lower()
        if ext not in _SUPPORTED_EXTS:
            raise AudioPlayerError(
                f"unsupported audio format {ext!r}; supported: "
                f"{sorted(_SUPPORTED_EXTS)}"
            )
        if not abs_path.is_file():
            raise AudioPlayerError(f"audio file not found: {abs_path}")

        size = abs_path.stat().st_size
        if size > self._max_file_bytes:
            raise AudioPlayerError(
                f"audio file is {size} bytes, exceeds cap "
                f"{self._max_file_bytes} bytes"
            )

        # Decode off the event loop — miniaudio is sync.
        decoded = await asyncio.get_running_loop().run_in_executor(
            None, _decode_to_pcm, abs_path
        )
        pcm_bytes = decoded.pcm_bytes
        decoded_duration_ms = decoded.duration_ms

        if decoded_duration_ms > int(self._max_duration_seconds * 1000):
            raise AudioPlayerError(
                f"audio is {decoded_duration_ms / 1000:.1f}s, exceeds "
                f"cap {self._max_duration_seconds:.0f}s"
            )

        prior_volume: float | None = None
        if volume is not None:
            prior_volume = self._speaker.get_volume()
            self._speaker.set_volume(volume)

        start = time.monotonic()
        try:
            await self._speaker.play(
                pcm_bytes,
                sample_rate=_TARGET_SAMPLE_RATE,
                channels=_TARGET_CHANNELS,
            )
        finally:
            if prior_volume is not None:
                self._speaker.set_volume(prior_volume)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        # If playback returned more than ~250 ms early, treat as
        # interrupted. The 250 ms slop covers HDMI buffer drain plus
        # the small _await_drained sleep at the end of speaker.play().
        interrupted = elapsed_ms + 250 < decoded_duration_ms

        return PlaybackResult(
            duration_ms=decoded_duration_ms,
            elapsed_ms=elapsed_ms,
            interrupted=interrupted,
            file_format=ext.lstrip("."),
            sample_rate=decoded.source_sample_rate,
            channels=decoded.source_channels,
        )


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


@dataclass
class _DecodedAudio:
    pcm_bytes: bytes
    duration_ms: int
    source_sample_rate: int
    source_channels: int


def _decode_to_pcm(abs_path: Path) -> _DecodedAudio:
    """Decode a workspace audio file to int16 mono PCM at 24 kHz.

    miniaudio's :func:`decode_file` handles MP3/WAV/FLAC/OGG natively
    and resamples + downmixes for us in one pass. No ffmpeg required.
    """
    try:
        import miniaudio  # type: ignore[import-untyped]
    except ImportError as e:
        raise AudioPlayerError(
            "miniaudio package not installed — cannot decode audio"
        ) from e

    try:
        # Probe source format/rate/channels first so we can surface
        # them in the result. ``get_file_info`` is cheap and doesn't
        # decode the whole file.
        info = miniaudio.get_file_info(str(abs_path))
        source_sr = int(info.sample_rate)
        source_ch = int(info.nchannels)

        decoded = miniaudio.decode_file(
            str(abs_path),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=_TARGET_CHANNELS,
            sample_rate=_TARGET_SAMPLE_RATE,
        )
    except miniaudio.DecodeError as e:
        raise AudioPlayerError(f"decode failed: {e}") from e
    except Exception as e:
        raise AudioPlayerError(f"could not read audio file: {e}") from e

    # ``decoded.samples`` is an array.array("h") of int16 samples.
    pcm_bytes = bytes(decoded.samples)
    num_frames = decoded.num_frames  # frames at the *target* rate
    duration_ms = int(num_frames * 1000 / _TARGET_SAMPLE_RATE)

    return _DecodedAudio(
        pcm_bytes=pcm_bytes,
        duration_ms=duration_ms,
        source_sample_rate=source_sr,
        source_channels=source_ch,
    )
