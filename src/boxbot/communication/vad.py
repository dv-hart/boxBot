"""Silero VAD wrapper for voice activity detection.

Classifies audio chunks as speech or silence, returning a probability
score. Used by AudioCapture to detect utterance boundaries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from boxbot.core.config import VADConfig

if TYPE_CHECKING:
    from boxbot.hardware.base import AudioChunk

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Silero VAD wrapper for speech/silence classification."""

    def __init__(self, config: VADConfig) -> None:
        self._config = config
        self._model: object | None = None

    async def start(self) -> None:
        """Load Silero VAD model."""
        if torch is None:
            logger.error("torch is not installed — VAD disabled")
            return

        self._model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        logger.info("Silero VAD model loaded")

    async def stop(self) -> None:
        """Release model resources."""
        self._model = None
        logger.info("VoiceActivityDetector stopped")

    async def process_chunk(self, chunk: AudioChunk) -> float:
        """Process audio chunk and return speech probability (0.0-1.0).

        Silero VAD expects 16kHz mono audio in 512-sample (32ms) windows.
        Our 1024-frame chunks yield 2 VAD windows; we return the maximum
        probability across windows.

        Args:
            chunk: Audio chunk from the microphone HAL.

        Returns:
            Speech probability between 0.0 and 1.0. Returns 0.0 if the
            model is not loaded.
        """
        if self._model is None or torch is None:
            return 0.0

        # Convert raw PCM bytes (int16 LE) to float32 tensor normalized to [-1, 1]
        audio_int16 = np.frombuffer(chunk.data, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_float)

        # Silero VAD expects 512-sample windows at 16kHz
        window_size = 512
        max_prob = 0.0

        for offset in range(0, len(audio_tensor), window_size):
            window = audio_tensor[offset : offset + window_size]
            if len(window) < window_size:
                # Pad the final partial window with zeros
                window = torch.nn.functional.pad(
                    window, (0, window_size - len(window))
                )
            prob = self._model(window, chunk.sample_rate).item()
            if prob > max_prob:
                max_prob = prob

        return max_prob

    def reset(self) -> None:
        """Reset VAD state between sessions."""
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()
            logger.debug("VAD state reset")
