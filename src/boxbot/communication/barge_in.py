"""Barge-in monitor — graduated yielding during TTS playback.

Monitors the microphone for speech while the agent is speaking (TTS
playback). Implements a three-stage model to distinguish accidental
sounds from intentional interruptions:

1. **Ignore** (0–200 ms): Continue at full volume. Filters coughs, brief sounds.
2. **Fade** (200–400 ms): Fade speaker to configured level. Visual cue via LEDs.
3. **Confirm** (400 ms+): Stop playback completely. Capture interrupting speech.

AEC (Acoustic Echo Cancellation) is critical — without it, the mic hears
boxBot's own voice and triggers false barge-in. The XMOS chip on the
ReSpeaker subtracts the reference signal sent via the dual-output ALSA
device. If AEC is not available, barge-in should be disabled.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from boxbot.hardware.base import AudioChunk

if TYPE_CHECKING:
    from boxbot.communication.vad import VoiceActivityDetector
    from boxbot.core.config import BargeInConfig
    from boxbot.hardware.speaker import Speaker

logger = logging.getLogger(__name__)


class BargeInMonitor:
    """Monitors for speech during TTS playback and triggers barge-in.

    Three-stage graduated yielding:
    - Stage 1 (ignore): 0 to ignore_duration ms — full volume, no action
    - Stage 2 (fade): ignore_duration to confirm_duration ms — fade volume
    - Stage 3 (confirm): beyond confirm_duration ms — stop playback
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        speaker: Speaker,
        config: BargeInConfig,
    ) -> None:
        self._vad = vad
        self._speaker = speaker
        self._config = config

        self._interrupt_callback: Callable[[], Awaitable[None]] | None = None
        self._monitoring = False
        self._speech_start: float | None = None
        self._faded = False

    def set_interrupt_callback(
        self, callback: Callable[[], Awaitable[None]]
    ) -> None:
        """Set callback invoked when barge-in is confirmed."""
        self._interrupt_callback = callback

    async def start(self, microphone: Any) -> None:
        """Start monitoring for barge-in by registering as a mic consumer."""
        if not self._config.enabled:
            logger.debug("Barge-in disabled by config")
            return

        self._monitoring = True
        self._speech_start = None
        self._faded = False
        microphone.add_consumer(self._on_audio_chunk, "barge_in")
        logger.debug("Barge-in monitor started")

    async def stop(self) -> None:
        """Stop monitoring."""
        self._monitoring = False
        self._speech_start = None
        self._faded = False
        logger.debug("Barge-in monitor stopped")

    async def remove_from(self, microphone: Any) -> None:
        """Remove this monitor as a consumer from the microphone."""
        microphone.remove_consumer(self._on_audio_chunk)
        self._monitoring = False

    async def _on_audio_chunk(self, chunk: AudioChunk) -> None:
        """Process audio chunk during TTS playback for barge-in detection."""
        if not self._monitoring:
            return

        # Get speech probability from VAD
        speech_prob = await self._vad.process_chunk(chunk)

        if speech_prob >= self._vad._config.threshold:
            # Speech detected
            if self._speech_start is None:
                self._speech_start = time.monotonic()

            speech_duration_ms = (
                (time.monotonic() - self._speech_start) * 1000
            )

            if speech_duration_ms >= self._config.confirm_duration:
                # Stage 3: Confirm — stop playback
                logger.info(
                    "Barge-in confirmed (%.0f ms speech), stopping playback",
                    speech_duration_ms,
                )
                self._monitoring = False
                await self._speaker.stop_playback()
                if self._interrupt_callback:
                    await self._interrupt_callback()
                return

            if (
                speech_duration_ms >= self._config.ignore_duration
                and not self._faded
            ):
                # Stage 2: Fade — reduce volume
                logger.debug(
                    "Barge-in fade triggered (%.0f ms speech)",
                    speech_duration_ms,
                )
                self._faded = True
                await self._speaker.fade_volume(
                    self._config.fade_volume,
                    self._config.fade_duration,
                )
            # Stage 1: Ignore — do nothing (< ignore_duration ms)

        else:
            # No speech — reset the speech timer
            if self._speech_start is not None:
                # Brief silence resets the counter
                self._speech_start = None
                if self._faded:
                    # Restore volume if we faded but speech stopped
                    self._faded = False
                    await self._speaker.fade_volume(
                        1.0, self._config.fade_duration
                    )
