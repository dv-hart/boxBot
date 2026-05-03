"""Pluggable text-to-speech with ElevenLabs streaming as initial provider.

Provides a protocol for TTS providers, a streaming wrapper for speaker
integration and barge-in, and a concrete ElevenLabs implementation.

Cost tracking: every successful ElevenLabs TTS call appends one row to
``cost_log`` via :mod:`boxbot.cost`. The billed character count is read
from the ``x-character-count`` HTTP response header — the authoritative
billed unit for ElevenLabs TTS — by going through the SDK's
``with_raw_response`` accessor. We fall back to ``len(text)`` only if
the header is missing (and log a warning), since SSML and Unicode
normalization can shift the billed count.

Usage:
    from boxbot.communication.tts import ElevenLabsTTS, TTSStream

    tts = ElevenLabsTTS(api_key="...", voice_id="...")
    stream = TTSStream(tts, speaker_hal)
    await stream.speak("Hello, how can I help?")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Hard upper bound on the synthesis + playback pipeline. ElevenLabs
# normally streams first audio in ~200-400 ms; 20 s is more than enough
# headroom for long utterances while catching a stalled HTTP stream.
_TTS_STREAM_TIMEOUT_SECONDS = 20.0

# HTTP response header that carries the authoritative billed character
# count for an ElevenLabs TTS request.
_BILLED_CHARS_HEADER = "x-character-count"

try:
    from elevenlabs import AsyncElevenLabs, VoiceSettings
except ImportError:
    AsyncElevenLabs = None  # type: ignore[assignment, misc]
    VoiceSettings = None  # type: ignore[assignment, misc]


@runtime_checkable
class TTSProvider(Protocol):
    """Protocol for text-to-speech providers."""

    async def synthesize(
        self, text: str, *, conversation_id: str | None = None
    ) -> bytes: ...
    async def synthesize_stream(
        self, text: str, *, conversation_id: str | None = None
    ) -> AsyncIterator[bytes]: ...


class TTSStream:
    """Wraps streaming TTS for integration with speaker HAL and barge-in."""

    def __init__(self, tts: TTSProvider, speaker: object) -> None:
        self._tts = tts
        self._speaker = speaker
        self._is_playing = False

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    async def speak(
        self, text: str, *, conversation_id: str | None = None
    ) -> None:
        """Stream TTS audio to speaker.

        Cancellation-safe: if the caller cancels this coroutine (e.g.
        barge-in, conversation timeout, user interruption), playback is
        stopped on the speaker and the cancellation is propagated. Also
        bounds synthesis to ``_TTS_STREAM_TIMEOUT_SECONDS`` so a stalled
        upstream HTTP connection can't hang the agent.

        Args:
            text: Text to synthesize and play.
            conversation_id: Logical conversation id used as the
                ``correlation_id`` on the emitted ``cost_log`` row.
        """
        self._is_playing = True
        try:
            stream = self._tts.synthesize_stream(
                text, conversation_id=conversation_id
            )
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

    async def synthesize(
        self, text: str, *, conversation_id: str | None = None
    ) -> bytes:
        """Synthesize complete audio (non-streaming).

        Records one ``cost_log`` row on success using the billed
        character count from the ``x-character-count`` response header.

        Args:
            text: Text to synthesize.
            conversation_id: Optional correlation id written to the
                cost row so TTS spend can be traced back to a turn.

        Returns:
            Raw PCM int16 mono audio bytes at 24kHz.
        """
        logger.debug("Synthesizing %d chars with ElevenLabs (non-streaming)", len(text))

        # Use the raw client so we can read the response headers; the
        # high-level wrapper hides them. ``convert`` is an async context
        # manager that yields an ``AsyncHttpResponse[AsyncIterator[bytes]]``.
        chunks: list[bytes] = []
        headers: dict[str, str] = {}
        async with self._client.text_to_speech.with_raw_response.convert(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model,
            output_format="pcm_24000",
            voice_settings=VoiceSettings(
                stability=self._stability,
                similarity_boost=self._similarity_boost,
            ),
        ) as response:
            headers = dict(response.headers or {})
            async for chunk in response.data:
                chunks.append(chunk)

        audio = b"".join(chunks)
        logger.debug("Synthesized %d bytes of PCM audio", len(audio))

        await _record_tts_cost(
            model=self._model,
            text=text,
            headers=headers,
            conversation_id=conversation_id,
        )
        return audio

    async def synthesize_stream(
        self, text: str, *, conversation_id: str | None = None
    ) -> AsyncIterator[bytes]:
        """Stream TTS audio chunks.

        Records one ``cost_log`` row when the stream completes
        successfully, using the billed character count from the
        ``x-character-count`` response header.

        Args:
            text: Text to synthesize.
            conversation_id: Optional correlation id written to the
                cost row so TTS spend can be traced back to a turn.

        Yields:
            Raw PCM int16 mono audio chunks at 24kHz.
        """
        logger.debug("Streaming synthesis of %d chars with ElevenLabs", len(text))

        # Use the raw client to expose response headers. ``stream`` is
        # an async context manager yielding an
        # ``AsyncHttpResponse[AsyncIterator[bytes]]``.
        async with self._client.text_to_speech.with_raw_response.stream(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model,
            output_format="pcm_24000",
            voice_settings=VoiceSettings(
                stability=self._stability,
                similarity_boost=self._similarity_boost,
            ),
            optimize_streaming_latency=self._optimize_streaming_latency,
        ) as response:
            headers = dict(response.headers or {})
            async for chunk in response.data:
                yield chunk

        # Only reached when the stream is consumed to completion (no
        # cancellation, no upstream error). Matches "record once on
        # success" semantics.
        await _record_tts_cost(
            model=self._model,
            text=text,
            headers=headers,
            conversation_id=conversation_id,
        )


# ---------------------------------------------------------------------------
# Cost recording helpers
# ---------------------------------------------------------------------------


# Module-level singleton MemoryStore for cost writes. Mirrors the
# pattern used by ``boxbot.tools.builtins.web_search`` — the production
# process already has a MemoryStore initialised, but the voice adapter
# is built deep in the call graph and threading the store through every
# constructor would touch unrelated code. Lazy-init keeps tests fast
# (they can monkeypatch ``_get_cost_store`` if needed) and avoids
# circular imports at module load.
_cost_store: Any = None


async def _get_cost_store() -> Any:
    """Return a process-wide MemoryStore for appending cost rows."""
    global _cost_store
    if _cost_store is None:
        from boxbot.memory.store import MemoryStore

        _cost_store = MemoryStore()
        await _cost_store.initialize()
    return _cost_store


async def _record_tts_cost(
    *,
    model: str,
    text: str,
    headers: dict[str, str],
    conversation_id: str | None,
) -> None:
    """Append one cost_log row for a successful ElevenLabs TTS call.

    Prefers ``x-character-count`` from the response (authoritative
    billed unit). Falls back to ``len(text)`` and logs a warning if the
    header is missing — SSML and Unicode normalization can shift the
    billed count, so the fallback is an estimate only.
    """
    # Import inside the function so a missing ``boxbot.cost`` (e.g. in
    # very early test bootstrapping) never breaks audio playback.
    try:
        from boxbot.cost import from_elevenlabs_tts, record
    except Exception:
        logger.exception("boxbot.cost unavailable; skipping TTS cost write")
        return

    billed_chars = _extract_billed_chars(headers, text)

    try:
        event = from_elevenlabs_tts(
            model=model,
            billed_chars=billed_chars,
            correlation_id=conversation_id,
        )
        store = await _get_cost_store()
        await record(store, event)
    except Exception:
        logger.exception(
            "Failed to record TTS cost (model=%s chars=%d)",
            model,
            billed_chars,
        )


def _extract_billed_chars(headers: dict[str, str], text: str) -> int:
    """Read ``x-character-count`` from response headers.

    Header names are case-insensitive; the SDK normalises them to lower
    case but we re-check defensively. If the header is missing or
    unparseable, fall back to ``len(text)`` and emit a WARNING — the
    fallback is an estimate, not the billed count.
    """
    if not headers:
        logger.warning(
            "ElevenLabs response had no headers (missing %s); using "
            "len(text)=%d as estimated billed chars",
            _BILLED_CHARS_HEADER,
            len(text),
        )
        return len(text)

    raw = headers.get(_BILLED_CHARS_HEADER)
    if raw is None:
        # Try a case-insensitive sweep in case the SDK didn't normalise.
        for k, v in headers.items():
            if k.lower() == _BILLED_CHARS_HEADER:
                raw = v
                break

    if raw is None:
        logger.warning(
            "ElevenLabs response missing %s header; using len(text)=%d as "
            "estimated billed chars",
            _BILLED_CHARS_HEADER,
            len(text),
        )
        return len(text)

    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "ElevenLabs %s header was not an int (%r); using len(text)=%d",
            _BILLED_CHARS_HEADER,
            raw,
            len(text),
        )
        return len(text)
