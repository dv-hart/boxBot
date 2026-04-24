"""Audio capture with VAD-driven utterance boundary detection.

Accumulates speech segments from the microphone and finalizes them
into Utterance objects when silence exceeds the configured threshold.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable

from boxbot.communication.vad import VoiceActivityDetector
from boxbot.core.config import TurnDetectionConfig

if TYPE_CHECKING:
    from boxbot.hardware.base import AudioChunk

logger = logging.getLogger(__name__)


@dataclass
class Utterance:
    """A finalized utterance ready for STT and diarization."""

    audio: bytes  # complete PCM audio for this utterance (int16 LE mono)
    duration: float  # seconds
    sample_rate: int
    timestamp_start: float
    timestamp_end: float


class AudioCapture:
    """Accumulates speech audio and detects utterance boundaries.

    Registers as a microphone consumer, passes each chunk through the
    VoiceActivityDetector, and packages continuous speech segments into
    Utterance objects when silence exceeds the configured threshold.
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        config: TurnDetectionConfig,
    ) -> None:
        self._vad = vad
        self._config = config
        self._microphone: object | None = None
        self._callback: Callable[[Utterance], Awaitable[None]] | None = None

        # Audio buffer and state
        self._buffer = bytearray()
        self._is_speaking = False
        self._speech_start: float = 0.0
        self._last_chunk_time: float = 0.0
        self._silence_ms: float = 0.0
        self._sample_rate: int = 16000

    async def start(self, microphone: object) -> None:
        """Register as microphone consumer and start capturing.

        Args:
            microphone: Microphone HAL instance with add_consumer/remove_consumer.
        """
        self._microphone = microphone
        microphone.add_consumer(self._on_audio_chunk)  # type: ignore[attr-defined]
        logger.info("AudioCapture started")

    async def stop(self) -> None:
        """Unregister and release resources."""
        if self._microphone is not None:
            self._microphone.remove_consumer(self._on_audio_chunk)  # type: ignore[attr-defined]
            self._microphone = None
        self.reset()
        logger.info("AudioCapture stopped")

    def set_utterance_callback(
        self, callback: Callable[[Utterance], Awaitable[None]]
    ) -> None:
        """Set callback for when an utterance is finalized.

        Args:
            callback: Async callable that receives the completed Utterance.
        """
        self._callback = callback

    def reset(self) -> None:
        """Reset capture state between sessions."""
        self._buffer = bytearray()
        self._is_speaking = False
        self._speech_start = 0.0
        self._last_chunk_time = 0.0
        self._silence_ms = 0.0

    async def _on_audio_chunk(self, chunk: AudioChunk) -> None:
        """Process chunk: VAD -> accumulate/finalize.

        Flow:
        1. Get speech probability from VAD.
        2. If speech: accumulate audio, reset silence counter.
        3. If silence after speech: accumulate (trailing audio), count silence.
        4. If silence exceeds threshold: finalize utterance.
        5. If duration exceeds max: force finalize.
        """
        self._sample_rate = chunk.sample_rate
        speech_prob = await self._vad.process_chunk(chunk)
        chunk_duration_ms = (chunk.frames / chunk.sample_rate) * 1000

        is_speech = speech_prob > self._vad._config.threshold

        if is_speech:
            if not self._is_speaking:
                # Speech just started
                self._is_speaking = True
                self._speech_start = chunk.timestamp
                self._silence_ms = 0.0
                logger.debug("Speech started at %.3f", chunk.timestamp)

            # Accumulate speech audio and reset silence counter
            self._buffer.extend(chunk.data)
            self._silence_ms = 0.0
            self._last_chunk_time = chunk.timestamp

        elif self._is_speaking:
            # Silence after speech — still accumulate for trailing audio
            self._buffer.extend(chunk.data)
            self._silence_ms += chunk_duration_ms
            self._last_chunk_time = chunk.timestamp

            # Check if silence exceeds threshold
            if self._silence_ms >= self._config.silence_threshold:
                await self._finalize_utterance()
                return

        # Check if total utterance duration exceeds hard cap
        if self._is_speaking:
            elapsed = self._last_chunk_time - self._speech_start
            if elapsed >= self._config.max_utterance_duration:
                logger.info(
                    "Utterance hit max duration (%.1fs), force finalizing",
                    elapsed,
                )
                await self._finalize_utterance()

    async def _finalize_utterance(self) -> None:
        """Package accumulated audio into an Utterance and fire callback."""
        if not self._buffer:
            self.reset()
            return

        duration = self._last_chunk_time - self._speech_start
        utterance = Utterance(
            audio=bytes(self._buffer),
            duration=duration,
            sample_rate=self._sample_rate,
            timestamp_start=self._speech_start,
            timestamp_end=self._last_chunk_time,
        )

        logger.info(
            "Utterance finalized: %.2fs (%d bytes)",
            duration,
            len(utterance.audio),
        )

        # Reset state before callback to allow re-entrant capture
        self.reset()

        if self._callback is not None:
            await self._callback(utterance)
