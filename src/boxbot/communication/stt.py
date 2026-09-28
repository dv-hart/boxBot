"""Pluggable speech-to-text — batch Scribe and Scribe v2 Realtime.

Two protocols, one registry. :class:`STTProvider` is request/response:
hand it a finished utterance, get a transcript. :class:`StreamingSTTProvider`
adds a socket that opens on touch-down and is fed while the user is still
talking, so the upload and most of the transcription happen inside the
hold instead of after it.

Cost tracking: every successful call appends one row to ``cost_log`` via
:mod:`boxbot.cost`. The billable unit differs by mode and neither API
returns it:

- **batch** (:class:`ElevenLabsSTT`) bills per minute of *input audio*,
  measured from the PCM (``num_samples / sample_rate``).
- **realtime** (:class:`ScribeRealtimeSTT`) bills per minute of *open
  connection*, measured socket-open → socket-close. Length of speech is
  irrelevant; a socket left open over a silent minute costs a full
  minute. This is why the socket lives exactly as long as the press.

Retries that eventually fail are not recorded; only a successful
response writes a row.

Usage:
    from boxbot.communication.stt import ElevenLabsSTT

    stt = ElevenLabsSTT(api_key="...", model="scribe_v2")
    result = await stt.transcribe(pcm_bytes, sample_rate=16000)
    print(result.text)

    # Streaming: open on press, feed while held, harvest on release.
    stt = ScribeRealtimeSTT(api_key="...")
    stt.open(sample_rate=16000)          # returns before the socket exists
    stt.feed(pcm_chunk)                  # ... repeatedly, never blocks
    result = await stt.transcribe(b"", 16000)   # commit + bounded wait
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import io
import logging
import re
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from boxbot.core import latency

if TYPE_CHECKING:
    from boxbot.core.config import ApiKeysConfig, STTConfig

logger = logging.getLogger(__name__)

# Scribe emits bracketed annotations for non-speech audio:
# "[background noise]", "[silence]", "[music]", "[laughter]", etc.
# If the transcript is ONLY annotations after stripping, it carries no
# speech and must not reach the agent — a wasted turn on noise costs ~5s
# of latency and can spawn a conversation from silence (see voice
# pipeline logs 2026-06-05 08:39:56 for the symptom).
_SCRIBE_ANNOTATION_RE = re.compile(r"\[[^\]]+\]")


def _is_annotation_only(text: str) -> bool:
    """True when ``text`` is empty or contains only Scribe annotations.

    Examples that return True: "", "[background noise]",
    "[silence] [music]", " [no speech] ".
    Examples that return False: "[Speaker A]: Hello", "Good morning",
    "[background noise] Help" (any real content alongside annotations
    is preserved — we only filter the all-noise case).
    """
    if not text or not text.strip():
        return True
    stripped = _SCRIBE_ANNOTATION_RE.sub("", text).strip()
    return not stripped

# Default channel and sample-width assumptions for the PCM payload.
# boxBot's voice path is mono int16 throughout (mic capture, VAD,
# Scribe submission); these are also the defaults of ``pcm_to_wav``.
_DEFAULT_CHANNELS = 1
_DEFAULT_SAMPLE_WIDTH_BYTES = 2

try:
    from elevenlabs import AsyncElevenLabs
except ImportError:
    AsyncElevenLabs = None  # type: ignore[assignment, misc]

# Realtime symbols, guarded the same way: ``elevenlabs`` is a setup-script
# dep, not a pyproject dep, so the dev venv and the test suite run without
# it. Bound to module globals rather than imported at use site so tests can
# monkeypatch ``_RealtimeEvents`` with a stub namespace.
try:
    from elevenlabs.realtime.connection import RealtimeEvents as _RealtimeEvents
    from elevenlabs.realtime.scribe import AudioFormat as _AudioFormat
    from elevenlabs.realtime.scribe import CommitStrategy as _CommitStrategy
except ImportError:  # pragma: no cover - exercised by absence, not by a test
    _RealtimeEvents = _AudioFormat = _CommitStrategy = None  # type: ignore[assignment, misc]


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


class StreamingSTTUnavailableError(RuntimeError):
    """The streaming session produced no transcript; the audio is intact.

    Raised for every non-transcript outcome — connect failure, an error
    frame, a session that outlived its server-side limit, a commit that
    never came back — because they all mean the same thing to the caller:
    *use the buffered PCM with a batch provider instead*. Silence is not
    one of these; it comes back as an empty :class:`STTResult`.
    """


@runtime_checkable
class StreamingSTTProvider(Protocol):
    """Protocol for STT providers fed while the user is still speaking.

    Lifecycle is one utterance: ``open`` on touch-down, ``feed`` per
    audio chunk, ``transcribe`` (or ``harvest`` + ``close``) on
    touch-up. ``open`` and ``feed`` are deliberately **not** coroutines —
    capture must never be able to block on a network handshake, and a
    sync signature is the only way to guarantee that at the type level.

    Implementations also satisfy :class:`STTProvider`, so a caller that
    never opens a session still type-checks; ``transcribe`` is the
    harvest step for a session that is already open.
    """

    def open(
        self,
        sample_rate: int,
        language: str = "en",
        *,
        conversation_id: str | None = None,
    ) -> None: ...

    def feed(self, pcm: bytes) -> None: ...

    async def harvest(self, *, timeout: float | None = None) -> STTResult: ...

    async def close(self) -> None: ...

    @property
    def is_live(self) -> bool:
        """True between ``open`` and ``close``.

        The caller uses this to decide whether a session still owes a
        harvest — a press that produced no utterance has to be closed by
        somebody, or the connection meter keeps running.
        """
        ...

    @property
    def buffered_audio(self) -> bytes:
        """Everything fed this session, for a batch fallback."""
        ...


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

        _t = time.monotonic()
        result = await self._client.speech_to_text.convert(
            file=wav_bytes,
            model_id=self._model,
            language_code=language,
        )
        _stt_elapsed = time.monotonic() - _t
        # Batch STT: the whole utterance is uploaded after speech ends,
        # then we block on the full transcript. This is the inherent
        # serial cost streaming STT would attack.
        latency.add(conversation_id, "stt", _stt_elapsed)
        logger.debug(
            "ElevenLabs Scribe request took %.0fms (model=%s)",
            _stt_elapsed * 1000,
            self._model,
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

        # Drop pure non-speech annotations (e.g. "[background noise]")
        # before they reach the voice pipeline. The downstream guard in
        # ``VoiceSession._on_utterance`` treats empty text as "no
        # transcript — return to listening", so this both prevents a
        # stray turn AND avoids creating/interrupting a conversation
        # from silence.
        if _is_annotation_only(text):
            logger.info(
                "STT dropped non-speech annotation transcript: %r",
                text,
            )
            text = ""
            words = []

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
# ElevenLabs Scribe v2 Realtime
# ---------------------------------------------------------------------------


# 8192 B = 256 ms at 16 kHz mono s16le. No documented hard maximum, but
# ``RealtimeEvents.CHUNK_SIZE_EXCEEDED`` exists and the SDK's own
# streaming path chunks at 8192 — mirror it rather than invent a number.
_FLUSH_BYTES = 8192

# Prebuffer ceiling, matching ``turn_detection.max_utterance_duration``.
# 32 KB/s, so 60 s bounds the buffer near 1.9 MB.
_MAX_BUFFER_SECONDS = 60.0

# Bounded wait from ``commit()`` to ``COMMITTED_TRANSCRIPT``. This is the
# span W-M compares against batch, so it must end in a value rather than
# hang: on expiry the caller still holds the whole utterance.
_HARVEST_TIMEOUT = 8.0

# Realtime accepts one PCM rate per ``AudioFormat`` member. The whole
# chain — ReSpeaker capture, VAD, diarization — is 16 kHz
# mono s16le, so there is exactly one supported rate and no resampler.
_REALTIME_SAMPLE_RATE = 16000

# Events we subscribe to, by attribute name on ``RealtimeEvents`` (name
# rather than value so the set is readable without the package
# installed). Anything not listed is left to the SDK.
_TERMINAL_EVENTS = (
    "ERROR",
    "AUTH_ERROR",
    "QUOTA_EXCEEDED",
    "RATE_LIMITED",
    "CHUNK_SIZE_EXCEEDED",
    # Sessions have a server-side max life. Terminal rather than
    # reconnect-and-resume: the press is capped at 60 s so this is
    # near-unreachable, and the prebuffer already makes the fallback
    # lossless. Reconnect logic would be untestable code paying for a
    # case that cannot happen.
    "SESSION_TIME_LIMIT_EXCEEDED",
    # The socket went away without an error frame. Every *error* the SDK
    # knows is re-emitted as ERROR (see its _start_message_handler), but
    # a clean close is only ever CLOSE — and without it here, a server
    # hanging up mid-utterance costs the full _HARVEST_TIMEOUT of dead
    # air before the batch fallback starts.
    "CLOSE",
)
_SUBSCRIBED_EVENTS = (
    "COMMITTED_TRANSCRIPT",
    "PARTIAL_TRANSCRIPT",
    "INSUFFICIENT_AUDIO_ACTIVITY",
) + _TERMINAL_EVENTS


class ScribeRealtimeSTT:
    """Scribe v2 Realtime over a websocket, one socket per press.

    **One FIFO, drain gated on connection.** ``feed`` appends to an
    append-only buffer and sets an event; a pump task waits for the
    handshake, then flushes whatever has accumulated in 8 KB slices and
    keeps flushing live. There is no buffering-vs-streaming state
    machine — capture always writes to the same place, so ordering falls
    out for free and none of these cases needs its own code path:

    - handshake still in flight → audio queues, user waits on nothing
    - release before the socket opens → nothing was dropped, it all
      flushes at once
    - connect fails → :attr:`buffered_audio` still holds the utterance
      and ``harvest`` raises :class:`StreamingSTTUnavailableError`, so the
      caller can fall back to batch

    The buffer is append-only rather than a consuming queue precisely so
    the fallback stays possible: bytes already on the wire must still be
    retrievable if the session dies mid-utterance. The 60 s cap bounds
    the cost of that at ~1.9 MB.

    Args:
        api_key: ElevenLabs credential.
        model: Must be a realtime model. ``STTConfig.model`` defaults to
            the batch ``scribe_v2``, which the realtime endpoint rejects,
            so this is validated at construction (= at ``start()``)
            rather than at the first press.
        connect: Test seam. Async callable returning a live connection
            object; defaults to the real SDK handshake.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "scribe_v2_realtime",
        *,
        connect: Callable[[str], Any] | None = None,
    ) -> None:
        if connect is None and (AsyncElevenLabs is None or _RealtimeEvents is None):
            raise ImportError(
                "elevenlabs>=2.65.0 is required for ScribeRealtimeSTT "
                "(earlier versions import fine but have no "
                "elevenlabs.realtime). Install it with: "
                "pip install 'elevenlabs>=2.65.0'"
            )
        if "realtime" not in model:
            raise ValueError(
                f"stt.model must name a realtime model for provider "
                f"'elevenlabs_realtime', got {model!r} — set "
                f"voice.stt.model: scribe_v2_realtime"
            )
        self._api_key = api_key
        self._model = model
        self._connect = connect or self._sdk_connect
        # One client for the life of the provider. Building it per press
        # leaked an httpx client per turn; the realtime handshake reuses
        # this one. None only when a test supplies its own connect seam,
        # in which case the SDK may not even be installed.
        self._client: Any = (
            AsyncElevenLabs(api_key=api_key) if connect is None else None
        )

        # Session state. All of it is reset by ``open``; ``_captured`` is
        # kept after ``close`` so a failed session's audio survives long
        # enough for the caller to read ``buffered_audio``.
        self._captured = bytearray()
        self._sent = 0
        self._max_bytes = 0
        self._live = False
        self._final = False
        self._language = "en"
        self._conversation_id: str | None = None

        self._conn: Any = None
        self._ready = asyncio.Event()  # handshake settled, either way
        self._data = asyncio.Event()  # new bytes in _captured, or _final
        self._events: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._failure: BaseException | None = None
        self._connect_seconds = 0.0
        self._connected_at = 0.0

    # -- lifecycle ---------------------------------------------------------

    def open(
        self,
        sample_rate: int = _REALTIME_SAMPLE_RATE,
        language: str = "en",
        *,
        conversation_id: str | None = None,
    ) -> None:
        """Start buffering and kick off the handshake concurrently.

        Returns before the socket exists — that is the entire point.
        Sync on purpose: nothing here may await, or capture would be
        blocked behind a network round-trip.

        A second ``open`` on a live session is ignored (a replayed touch
        frame, not a new utterance); reopening would drop what has
        already been said.
        """
        if sample_rate != _REALTIME_SAMPLE_RATE:
            raise ValueError(
                f"ScribeRealtimeSTT supports {_REALTIME_SAMPLE_RATE} Hz only, "
                f"got {sample_rate}"
            )
        if self._live:
            logger.debug("open() on a live streaming session, ignoring")
            return

        self._captured = bytearray()
        self._sent = 0
        self._max_bytes = int(
            _MAX_BUFFER_SECONDS
            * sample_rate
            * _DEFAULT_CHANNELS
            * _DEFAULT_SAMPLE_WIDTH_BYTES
        )
        self._live = True
        self._final = False
        self._language = language
        self._conversation_id = conversation_id
        self._conn = None
        self._ready = asyncio.Event()
        self._data = asyncio.Event()
        self._events = asyncio.Queue()
        self._failure = None
        self._connect_seconds = 0.0
        self._connected_at = 0.0

        self._tasks = [
            asyncio.create_task(self._handshake(), name="scribe-connect"),
            asyncio.create_task(self._pump(), name="scribe-pump"),
        ]

    def feed(self, pcm: bytes) -> None:
        """Append captured PCM. Never blocks, never awaits, never raises.

        Silently drops audio past the 60 s cap: the utterance is already
        longer than any turn we intend to bill, and dropping the tail is
        strictly better than an unbounded buffer on a device with 8 GB.
        """
        if not self._live or not pcm:
            return
        room = self._max_bytes - len(self._captured)
        if room <= 0:
            return
        if room < len(pcm):
            logger.warning(
                "Streaming prebuffer hit the %.0fs cap; dropping tail",
                _MAX_BUFFER_SECONDS,
            )
            pcm = pcm[:room]
        self._captured += pcm
        self._data.set()

    async def harvest(self, *, timeout: float | None = None) -> STTResult:
        """Close the FIFO, commit, and wait (bounded) for the transcript.

        Records two latency spans on the live turn tracker: ``stt`` for
        the commit → ``COMMITTED_TRANSCRIPT`` wait (the number W-M
        compares against batch) and ``stt_connect`` for the handshake
        that the prebuffer hid. The handshake happens during the hold,
        before ``latency.begin`` runs at speech-end, so it is measured
        there and reported here — otherwise it would land on no tracker.

        Raises:
            StreamingSTTUnavailableError: on any non-transcript outcome. The
                utterance is still in :attr:`buffered_audio`.
        """
        budget = _HARVEST_TIMEOUT if timeout is None else timeout
        cid = self._conversation_id

        # Flush the remainder before committing. Bounded by the same
        # budget: a wedged socket must not hold the turn open.
        self._final = True
        self._data.set()

        _t = time.monotonic()
        try:
            await asyncio.wait_for(self._drained(), budget)
            if self._conn is None or self._failure is not None:
                raise StreamingSTTUnavailableError(
                    f"streaming session unusable: {self._failure!r}"
                )
            await self._conn.commit()
            text = await self._await_committed(budget - (time.monotonic() - _t))
        except StreamingSTTUnavailableError:
            raise
        except TimeoutError as exc:
            raise StreamingSTTUnavailableError(
                f"no committed transcript within {budget:.1f}s"
            ) from exc
        except Exception as exc:
            raise StreamingSTTUnavailableError(f"streaming STT failed: {exc}") from exc
        finally:
            # Both spans land here, after the drain, because that is the
            # first moment the handshake is guaranteed to have settled.
            latency.add(cid, "stt_connect", self._connect_seconds)
            latency.add(cid, "stt", time.monotonic() - _t)

        # Same non-speech guard the batch path applies: a transcript of
        # pure annotations is silence with extra steps, and downstream
        # treats empty text as "return to listening".
        if _is_annotation_only(text):
            logger.info("Streaming STT dropped annotation-only transcript: %r", text)
            text = ""
        return STTResult(text=text, language=self._language)

    async def close(self) -> None:
        """Close the socket, stop the tasks, and bill the connection.

        Idempotent, and safe after a failed ``open`` — a session that
        never connected is not billed, because nothing was ever open.
        ``_captured`` deliberately survives so the caller can still read
        :attr:`buffered_audio` for a batch fallback.
        """
        if not self._live:
            return
        self._live = False
        self._final = True
        self._data.set()

        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

        conn, self._conn = self._conn, None
        if conn is None:
            return
        with contextlib.suppress(Exception):
            await conn.close()

        # Realtime bills per minute of *connection*, not of speech, so
        # ``_audio_seconds`` is the wrong unit here — the meter runs from
        # handshake to close regardless of what was said.
        seconds = time.monotonic() - self._connected_at
        logger.debug(
            "Realtime socket open for %.2fs (%d bytes fed)",
            seconds,
            len(self._captured),
        )
        await _record_stt_cost(
            model=self._model,
            audio_seconds=seconds,
            conversation_id=self._conversation_id,
            metadata={"billed_unit": "connection_seconds"},
        )

    @property
    def buffered_audio(self) -> bytes:
        """Everything fed this session — the batch-fallback payload."""
        return bytes(self._captured)

    @property
    def is_live(self) -> bool:
        """True between ``open`` and ``close``."""
        return self._live

    async def transcribe(
        self,
        audio: bytes,
        sample_rate: int,
        language: str = "en",
        *,
        conversation_id: str | None = None,
    ) -> STTResult:
        """:class:`STTProvider` adapter: harvest the open session, then close.

        Lets ``voice.py::_on_utterance`` stay byte-for-byte unchanged —
        it already calls ``transcribe(utterance.audio, ...)`` at exactly
        the moment the hold ends. ``audio`` is ignored because the same
        PCM was already streamed during the hold; it is the caller's
        fallback payload, not an input.

        Raises:
            StreamingSTTUnavailableError: including when no session is open,
                which is what a realtime provider configured without the
                push-to-talk wiring looks like.
        """
        if not self._live:
            raise StreamingSTTUnavailableError(
                "no streaming session open — voice.input_mode must be "
                "push_to_talk for provider 'elevenlabs_realtime'"
            )
        try:
            return await self.harvest()
        finally:
            await self.close()

    # -- internals ---------------------------------------------------------

    async def _sdk_connect(self, language: str) -> Any:
        """Open the real websocket. Isolated so tests never touch it."""
        return await self._client.speech_to_text.realtime.connect(
            {
                "model_id": self._model,
                "audio_format": _AudioFormat.PCM_16000,
                "sample_rate": _REALTIME_SAMPLE_RATE,
                "commit_strategy": _CommitStrategy.MANUAL,
                "language_code": language,
            }
        )

    async def _handshake(self) -> None:
        """Connect, subscribe, and release the pump. Never propagates."""
        _t = time.monotonic()
        try:
            conn = await self._connect(self._language)
            for name in _SUBSCRIBED_EVENTS:
                event = getattr(_RealtimeEvents, name, None)
                if event is not None:
                    conn.on(event, functools.partial(self._on_event, name))
            self._conn = conn
            self._connected_at = time.monotonic()
        except Exception as exc:
            self._failure = exc
            logger.warning("Realtime STT connect failed: %s", exc)
        finally:
            self._connect_seconds = time.monotonic() - _t
            # Set last: the pump must not observe a half-built session.
            self._ready.set()

    def _on_event(self, name: str, payload: Any = None) -> None:
        """SDK event sink. Sync, non-blocking, and cannot raise.

        The SDK's ``_emit`` calls handlers synchronously and swallows
        exceptions to a bare ``print()``, so any real work here would
        stall the socket read loop and any exception would vanish. Push
        and return.
        """
        try:
            data = payload if isinstance(payload, dict) else {}
            self._events.put_nowait((name, data))
        except Exception:  # pragma: no cover - unbounded queue, cannot fill
            logger.exception("Dropped realtime STT event %s", name)

    async def _pump(self) -> None:
        """Drain the FIFO to the socket in 8 KB slices, once connected."""
        await self._ready.wait()
        if self._conn is None:
            # Connect failed. Nothing was sent, and ``_captured`` holds
            # the whole utterance for the caller's batch fallback.
            return
        try:
            while True:
                self._data.clear()
                while len(self._captured) - self._sent >= _FLUSH_BYTES:
                    await self._flush(_FLUSH_BYTES)
                if self._final:
                    remainder = len(self._captured) - self._sent
                    if remainder:
                        await self._flush(remainder)
                    return
                await self._data.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failure = exc
            logger.warning("Realtime STT send failed: %s", exc)

    async def _flush(self, size: int) -> None:
        """Send one slice. ``audio_base_64`` is the key the SDK reads."""
        if self._sent == 0:
            # Prebuffer depth at first flush, against the handshake it
            # hid — the pair of numbers W-M needs to decide whether a warm
            # per-conversation socket is worth keeping open.
            logger.info(
                "Realtime STT first flush: %d bytes buffered during a "
                "%.0fms handshake",
                len(self._captured),
                self._connect_seconds * 1000,
            )
        slice_ = bytes(self._captured[self._sent : self._sent + size])
        self._sent += size
        await self._conn.send(
            {"audio_base_64": base64.b64encode(slice_).decode("ascii")}
        )

    async def _drained(self) -> None:
        """Wait for the handshake to settle and the pump's final flush.

        ``return_exceptions`` because a failed handshake or send is not
        this method's business — the caller reads ``_conn`` / ``_failure``
        and decides. Cancellation still propagates, so ``wait_for`` can
        bound it.
        """
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _await_committed(self, budget: float) -> str:
        """Consume events until one settles the utterance.

        ``INSUFFICIENT_AUDIO_ACTIVITY`` is the streaming silence case —
        the same "no transcript" outcome ``_is_annotation_only`` produces
        on the batch path, not an error. Partials are logged and waited
        past; the manual commit strategy means only
        ``COMMITTED_TRANSCRIPT`` is final.
        """
        deadline = time.monotonic() + budget
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            name, payload = await asyncio.wait_for(self._events.get(), remaining)
            if name == "COMMITTED_TRANSCRIPT":
                return str(payload.get("text") or "")
            if name == "INSUFFICIENT_AUDIO_ACTIVITY":
                logger.info("Realtime STT reported no speech activity")
                return ""
            if name in _TERMINAL_EVENTS:
                raise StreamingSTTUnavailableError(f"realtime STT {name}: {payload}")
            logger.debug("Realtime partial: %r", payload.get("text"))

# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


# ``stt.provider`` name → (ApiKeysConfig field holding the credential,
# builder). A new provider is one row here; nothing in the voice path
# changes.
_STT_PROVIDERS: dict[str, tuple[str, Callable[[str, STTConfig], STTProvider]]] = {
    "elevenlabs": (
        "elevenlabs",
        lambda api_key, cfg: ElevenLabsSTT(api_key=api_key, model=cfg.model),
    ),
    "elevenlabs_realtime": (
        "elevenlabs",
        lambda api_key, cfg: ScribeRealtimeSTT(api_key=api_key, model=cfg.model),
    ),
}


# Streaming provider → the batch provider that transcribes its buffered
# PCM when the socket never came good. The prebuffer means a failed
# connect still holds the whole utterance, so a dropped socket costs
# latency, not the user's words.
_STT_BATCH_FALLBACK = {"elevenlabs_realtime": "elevenlabs"}

_BATCH_FALLBACK_MODEL = "scribe_v2"


def create_stt_fallback(
    cfg: STTConfig, api_keys: ApiKeysConfig
) -> STTProvider | None:
    """Build the batch provider that backs up a streaming one.

    Returns None when ``cfg.provider`` is not streaming (nothing to back
    up) or its credential is unset. Goes through the same registry, so
    the valid-provider set still lives in exactly one place.
    """
    batch_name = _STT_BATCH_FALLBACK.get(cfg.provider)
    if batch_name is None:
        return None
    return create_stt(
        cfg.model_copy(update={"provider": batch_name,
                               "model": _BATCH_FALLBACK_MODEL}),
        api_keys,
    )


def create_stt(cfg: STTConfig, api_keys: ApiKeysConfig) -> STTProvider | None:
    """Build the STT provider named by ``cfg.provider``.

    Returns None when that provider's credential is unset — voice
    degrades to no-STT rather than refusing to boot. An *unknown*
    provider name raises: a typo silently running a different engine is
    the worse failure.
    """
    entry = _STT_PROVIDERS.get(cfg.provider)
    if entry is None:
        raise ValueError(
            f"stt.provider must be one of {'/'.join(sorted(_STT_PROVIDERS))}, "
            f"got {cfg.provider!r}"
        )
    key_field, build = entry
    api_key = getattr(api_keys, key_field)
    if not api_key:
        logger.warning(
            "%s API key not configured — STT disabled", cfg.provider
        )
        return None
    return build(api_key, cfg)


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
    metadata: dict | None = None,
) -> None:
    """Append one cost_log row for a successful ElevenLabs Scribe call.

    ``audio_seconds`` carries whatever ElevenLabs actually meters at this
    model's per-minute rate: input audio for batch, socket lifetime for
    realtime. Same arithmetic, different unit, so realtime passes
    ``metadata={"billed_unit": "connection_seconds"}`` to keep the row
    self-describing rather than forking the compute path.
    """
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
            metadata=metadata,
        )
        store = await _get_cost_store()
        await record(store, event)
    except Exception:
        logger.exception(
            "Failed to record STT cost (model=%s seconds=%.2f)",
            model,
            audio_seconds,
        )
