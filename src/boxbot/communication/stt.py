"""Pluggable speech-to-text with ElevenLabs Scribe as initial provider.

Provides a protocol for STT providers and a concrete ElevenLabs implementation.
Audio is received as raw PCM and converted to WAV for the API.

Usage:
    from boxbot.communication.stt import ElevenLabsSTT

    stt = ElevenLabsSTT(api_key="...", model="scribe_v2")
    result = await stt.transcribe(pcm_bytes, sample_rate=16000)
    print(result.text)
"""

from __future__ import annotations

import io
import logging
import wave
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

try:
    from elevenlabs import AsyncElevenLabs
except ImportError:
    AsyncElevenLabs = None  # type: ignore[assignment, misc]


@dataclass
class WordInfo:
    """Word-level timing and confidence from STT."""

    word: str
    start: float  # seconds
    end: float  # seconds
    confidence: float | None = None


@dataclass
class STTResult:
    """Result from speech-to-text transcription."""

    text: str
    language: str
    words: list[WordInfo] = field(default_factory=list)


@runtime_checkable
class STTProvider(Protocol):
    """Protocol for speech-to-text providers."""

    async def transcribe(
        self, audio: bytes, sample_rate: int, language: str = "en"
    ) -> STTResult: ...


def pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Convert raw PCM data to WAV format in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


class ElevenLabsSTT:
    """ElevenLabs Scribe STT provider."""

    def __init__(self, api_key: str, model: str = "scribe_v2") -> None:
        if AsyncElevenLabs is None:
            raise ImportError(
                "elevenlabs package is required for ElevenLabsSTT. "
                "Install it with: pip install elevenlabs"
            )
        self._client = AsyncElevenLabs(api_key=api_key)
        self._model = model

    async def transcribe(
        self, audio: bytes, sample_rate: int, language: str = "en"
    ) -> STTResult:
        """Transcribe audio using ElevenLabs Scribe.

        Args:
            audio: Raw PCM int16 mono audio bytes.
            sample_rate: Sample rate of the audio (e.g. 16000).
            language: Language code (default "en").

        Returns:
            STTResult with transcribed text and optional word timings.
        """
        wav_bytes = pcm_to_wav(audio, sample_rate)

        logger.debug(
            "Sending %d bytes of audio to ElevenLabs Scribe (model=%s)",
            len(wav_bytes),
            self._model,
        )

        result = await self._client.speech_to_text.convert(
            file=wav_bytes,
            model_id=self._model,
            language_code=language,
        )

        # Parse word-level timing if available
        words: list[WordInfo] = []
        if hasattr(result, "words") and result.words:
            for w in result.words:
                words.append(
                    WordInfo(
                        word=getattr(w, "text", str(w)),
                        start=getattr(w, "start", 0.0),
                        end=getattr(w, "end", 0.0),
                        confidence=getattr(w, "confidence", None),
                    )
                )

        text = result.text if hasattr(result, "text") else str(result)
        language_detected = (
            getattr(result, "language_code", language)
            if hasattr(result, "language_code")
            else language
        )

        logger.debug("Transcription result: %d chars, %d words", len(text), len(words))

        return STTResult(text=text, language=language_detected, words=words)
