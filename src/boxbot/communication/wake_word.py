"""OpenWakeWord integration for wake word detection.

Listens to audio chunks from the microphone HAL and publishes
WakeWordHeard events when the configured wake word is detected.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

try:
    import openwakeword  # type: ignore[import-untyped]
except ImportError:
    openwakeword = None  # type: ignore[assignment]

from boxbot.core.config import WakeWordConfig
from boxbot.core.events import WakeWordHeard, get_event_bus

if TYPE_CHECKING:
    from boxbot.hardware.base import AudioChunk

logger = logging.getLogger(__name__)


# Built-in wake-word model names → bundled ONNX file inside the
# openwakeword package. Newer openwakeword versions require a real path,
# so we resolve names ourselves.
_BUILTIN_WAKE_WORDS = {
    "alexa": "alexa_v0.1.onnx",
    "hey_jarvis": "hey_jarvis_v0.1.onnx",
    "hey_marvin": "hey_marvin_v0.1.onnx",
    "hey_mycroft": "hey_mycroft_v0.1.onnx",
    "timer": "timer_v0.1.onnx",
    "weather": "weather_v0.1.onnx",
}


def _resolve_builtin_model(name: str) -> str | None:
    """Map a built-in wake word name to its bundled ONNX path."""
    if openwakeword is None:
        return None
    filename = _BUILTIN_WAKE_WORDS.get(name)
    if filename is None:
        return None
    import os
    pkg_dir = os.path.dirname(openwakeword.__file__)
    path = os.path.join(pkg_dir, "resources", "models", filename)
    return path if os.path.exists(path) else None


class WakeWordDetector:
    """Detects wake words in the audio stream using OpenWakeWord."""

    def __init__(self, config: WakeWordConfig) -> None:
        self._config = config
        self._model: openwakeword.Model | None = None  # type: ignore[name-defined]
        self._microphone: object | None = None
        # Integer handle from microphone.add_consumer (see AudioCapture for
        # the rationale — bound methods are not identity-stable).
        self._consumer_handle: int | None = None
        self._bus = get_event_bus()
        # Debounce: OpenWakeWord can fire 2–3 detections in quick
        # succession off a single spoken wake word (observed 2026-04-24:
        # 0.988, 0.946, 0.728 all within ~1 s). Suppress further hits
        # for this many seconds after a confirmed detection.
        self._debounce_seconds: float = 1.5
        self._suppress_until: float = 0.0

    async def start(self, microphone: object) -> None:
        """Load model and register as microphone consumer.

        Args:
            microphone: Microphone HAL instance with add_consumer/remove_consumer.
        """
        if openwakeword is None:
            logger.error(
                "openwakeword is not installed — wake word detection disabled"
            )
            return

        self._microphone = microphone

        # Determine which model to load
        if self._config.model_path:
            wakeword_model_paths = [self._config.model_path]
            logger.info(
                "Loading custom wake word model: %s", self._config.model_path
            )
        else:
            resolved = _resolve_builtin_model(self._config.word)
            if resolved is None:
                logger.error(
                    "Unknown built-in wake word '%s'. Available: %s",
                    self._config.word,
                    ", ".join(sorted(_BUILTIN_WAKE_WORDS)),
                )
                return
            wakeword_model_paths = [resolved]
            logger.info(
                "Loading built-in wake word model: %s (%s)",
                self._config.word, resolved,
            )

        self._model = openwakeword.Model(
            wakeword_model_paths=wakeword_model_paths,
        )
        logger.info("OpenWakeWord model loaded")

        # Register as a microphone consumer
        self._consumer_handle = microphone.add_consumer(  # type: ignore[attr-defined]
            self._on_audio_chunk, name="wake_word",
        )

    async def stop(self) -> None:
        """Unregister from microphone and release model."""
        if self._microphone is not None and self._consumer_handle is not None:
            self._microphone.remove_consumer(self._consumer_handle)  # type: ignore[attr-defined]
            self._consumer_handle = None
            self._microphone = None

        self._model = None
        logger.info("WakeWordDetector stopped")

    async def _on_audio_chunk(self, chunk: AudioChunk) -> None:
        """Process audio chunk through wake word model.

        OpenWakeWord expects 16kHz mono int16 audio. It processes 80ms
        frames internally and returns confidence scores per model.
        """
        if self._model is None:
            return

        # Debounce window: after a confirmed detection, swallow audio
        # for ~1.5 s so the tail of the wake word doesn't fire again.
        now = time.monotonic()
        if now < self._suppress_until:
            return

        # Convert raw PCM bytes to numpy int16 array
        audio_array = np.frombuffer(chunk.data, dtype=np.int16)

        # Feed to model — OpenWakeWord handles internal framing
        predictions = self._model.predict(audio_array)

        # Check each model's confidence against threshold
        for model_name, confidence in predictions.items():
            if confidence > self._config.confidence_threshold:
                logger.info(
                    "Wake word '%s' detected (confidence=%.3f)",
                    model_name,
                    confidence,
                )
                self._suppress_until = now + self._debounce_seconds
                await self._bus.publish(
                    WakeWordHeard(confidence=confidence)
                )
                # Reset model state after detection to avoid repeated triggers
                self._model.reset()
                break
