"""Pluggable speech-to-text with ElevenLabs Scribe as initial provider.

Provides a protocol for STT providers and a concrete ElevenLabs implementation.
Audio is received as raw PCM and converted to WAV for the API.

Cost tracking: every successful ElevenLabs Scribe call appends one row
to ``cost_log`` via :mod:`boxbot.cost`. ElevenLabs bills Scribe by the
minute of input audio but does not return the duration in the
response, so we measure it from the input PCM
(``num_samples / sample_rate``). Retries that eventually fail are not
recorded; only the final successful response writes a row.

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
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Default channel and sample-width assumptions for the PCM payload.
# boxBot's voice path is mono int16 throughout (mic capture, VAD,
# Scribe submission); these are also the defaults of ``pcm_to_wav``.
_DEFAULT_CHANNELS = 1
_DEFAULT_SAMPLE_WIDTH_BYTES = 2

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
        self,
        audio: bytes,
        sample_rate: int,
        language: str = "en",
        *,
        conversation_id: str | None = None,
    ) -> STTResult: ...


def pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int,
    channels: int = _DEFAULT_CHANNELS,
    sample_width: int = _DEFAULT_SAMPLE_WIDTH_BYTES,
) -> bytes:
    """Convert raw PCM data to WAV format in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def _audio_seconds(
    pcm_bytes: int,
    sample_rate: int,
    channels: int = _DEFAULT_CHANNELS,
    sample_width: int = _DEFAULT_SAMPLE_WIDTH_BYTES,
) -> float:
    """Return the duration in seconds for a raw PCM byte length.

    ``len(pcm) / (sample_rate * channels * bytes_per_sample)``. Returns
    0.0 when any factor is non-positive so a malformed input cannot
    poison the cost row.
    """
    denom = sample_rate * channels * sample_width
    if denom <= 0 or pcm_bytes <= 0:
        return 0.0
    return pcm_bytes / float(denom)


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
        self,
        audio: bytes,
        sample_rate: int,
        language: str = "en",
        *,
        conversation_id: str | None = None,
    ) -> STTResult:
        """Transcribe audio using ElevenLabs Scribe.

        Records one ``cost_log`` row on success. Billable duration is
        measured from the input PCM (``len(audio) /
        (sample_rate * channels * sample_width)``) — Scribe does not
        return the billed duration in its response.

        Args:
            audio: Raw PCM int16 mono audio bytes.
            sample_rate: Sample rate of the audio (e.g. 16000).
            language: Language code (default "en").
            conversation_id: Optional correlation id written to the
                cost row so STT spend can be traced back to a turn.

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

        # Record cost from the *input* audio duration. We measure
        # before submission so even an empty transcript (silence,
        # garbled audio) bills correctly — ElevenLabs charges per
        # minute of input regardless of returned content.
        await _record_stt_cost(
            model=self._model,
            audio_seconds=_audio_seconds(len(audio), sample_rate),
            conversation_id=conversation_id,
        )

        return STTResult(text=text, language=language_detected, words=words)


# ---------------------------------------------------------------------------
# Cost recording helpers
# ---------------------------------------------------------------------------


# Module-level singleton MemoryStore for cost writes. Mirrors the
# pattern used by ``boxbot.tools.builtins.web_search`` and the TTS
# adapter — the voice path is built deep in the call graph and the
# global store is the simplest stable reference.
_cost_store: Any = None


async def _get_cost_store() -> Any:
    """Return a process-wide MemoryStore for appending cost rows."""
    global _cost_store
    if _cost_store is None:
        from boxbot.memory.store import MemoryStore

        _cost_store = MemoryStore()
        await _cost_store.initialize()
    return _cost_store


async def _record_stt_cost(
    *,
    model: str,
    audio_seconds: float,
    conversation_id: str | None,
) -> None:
    """Append one cost_log row for a successful ElevenLabs Scribe call."""
    try:
        from boxbot.cost import from_elevenlabs_stt, record
    except Exception:
        logger.exception("boxbot.cost unavailable; skipping STT cost write")
        return

    try:
        event = from_elevenlabs_stt(
            model=model,
            audio_seconds=audio_seconds,
            correlation_id=conversation_id,
        )
        store = await _get_cost_store()
        await record(store, event)
    except Exception:
        logger.exception(
            "Failed to record STT cost (model=%s seconds=%.2f)",
            model,
            audio_seconds,
        )
