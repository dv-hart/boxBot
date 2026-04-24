"""Pluggable text-to-speech with ElevenLabs streaming as initial provider.

Provides a protocol for TTS providers, a streaming wrapper for speaker
integration and barge-in, and a concrete ElevenLabs implementation.

Usage:
    from boxbot.communication.tts import ElevenLabsTTS, TTSStream

    tts = ElevenLabsTTS(api_key="...", voice_id="...")
    stream = TTSStream(tts, speaker_hal)
    await stream.speak("Hello, how can I help?")
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Hard upper bound on the synthesis + playback pipeline. ElevenLabs
# normally streams first audio in ~200-400 ms; 20 s is more than enough
# headroom for long utterances while catching a stalled HTTP stream.
_TTS_STREAM_TIMEOUT_SECONDS = 20.0

try:
    from elevenlabs import AsyncElevenLabs, VoiceSettings
except ImportError:
    AsyncElevenLabs = None  # type: ignore[assignment, misc]
    VoiceSettings = None  # type: ignore[assignment, misc]


@runtime_checkable
class TTSProvider(Protocol):
    """Protocol for text-to-speech providers."""

    async def synthesize(self, text: str) -> bytes: ...
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]: ...


class TTSStream:
    """Wraps streaming TTS for integration with speaker HAL and barge-in."""

    def __init__(self, tts: TTSProvider, speaker: object) -> None:
        self._tts = tts
        self._speaker = speaker
        self._is_playing = False

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    async def speak(self, text: str) -> None:
        """Stream TTS audio to speaker.

        Cancellation-safe: if the caller cancels this coroutine (e.g.
        barge-in, conversation timeout, user interruption), playback is
        stopped on the speaker and the cancellation is propagated. Also
        bounds synthesis to ``_TTS_STREAM_TIMEOUT_SECONDS`` so a stalled
        upstream HTTP connection can't hang the agent.

        Args:
            text: Text to synthesize and play.
        """
        self._is_playing = True
        try:
            stream = self._tts.synthesize_stream(text)
            await asyncio.wait_for(
                self._speaker.play_stream(stream),  # type: ignore[attr-defined]
                timeout=_TTS_STREAM_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            # Propagate cancellation, but first stop audible output so
            # we don't keep talking over the user.
            try:
                await self._speaker.stop_playback()  # type: ignore[attr-defined]
            except Exception:
                logger.debug("stop_playback on cancel failed", exc_info=True)
            raise
        except asyncio.TimeoutError:
            logger.error(
                "TTS playback exceeded %.0fs — stopping speaker. "
                "Possible stalled upstream stream.",
                _TTS_STREAM_TIMEOUT_SECONDS,
            )
            try:
                await self._speaker.stop_playback()  # type: ignore[attr-defined]
            except Exception:
                logger.debug("stop_playback on timeout failed", exc_info=True)
        finally:
            self._is_playing = False

    async def stop(self) -> None:
        """Stop current speech (barge-in)."""
        self._is_playing = False
        await self._speaker.stop_playback()  # type: ignore[attr-defined]


class ElevenLabsTTS:
    """ElevenLabs TTS provider with streaming support.

    Output format is raw PCM at 24kHz mono int16 — no MP3 decoding needed.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model: str = "eleven_turbo_v2_5",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        optimize_streaming_latency: int = 3,
    ) -> None:
        if AsyncElevenLabs is None:
            raise ImportError(
                "elevenlabs package is required for ElevenLabsTTS. "
                "Install it with: pip install elevenlabs"
            )
        self._client = AsyncElevenLabs(api_key=api_key)
        self._voice_id = voice_id
        self._model = model
        self._stability = stability
        self._similarity_boost = similarity_boost
        self._optimize_streaming_latency = optimize_streaming_latency

    async def synthesize(self, text: str) -> bytes:
        """Synthesize complete audio (non-streaming).

        Args:
            text: Text to synthesize.

        Returns:
            Raw PCM int16 mono audio bytes at 24kHz.
        """
        logger.debug("Synthesizing %d chars with ElevenLabs (non-streaming)", len(text))

        response = await self._client.text_to_speech.convert(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model,
            output_format="pcm_24000",
            voice_settings=VoiceSettings(
                stability=self._stability,
                similarity_boost=self._similarity_boost,
            ),
        )

        # Collect all chunks into a single bytes object
        chunks: list[bytes] = []
        async for chunk in response:
            chunks.append(chunk)

        audio = b"".join(chunks)
        logger.debug("Synthesized %d bytes of PCM audio", len(audio))
        return audio

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream TTS audio chunks.

        Args:
            text: Text to synthesize.

        Yields:
            Raw PCM int16 mono audio chunks at 24kHz.
        """
        logger.debug("Streaming synthesis of %d chars with ElevenLabs", len(text))

        response = self._client.text_to_speech.stream(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model,
            output_format="pcm_24000",
            voice_settings=VoiceSettings(
                stability=self._stability,
                similarity_boost=self._similarity_boost,
            ),
            optimize_streaming_latency=self._optimize_streaming_latency,
        )

        async for chunk in response:
            yield chunk
