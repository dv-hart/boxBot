"""Push-to-talk capture — utterance boundaries from touch, not from VAD.

Peer of :mod:`boxbot.communication.audio_capture`, same
``set_utterance_callback`` contract, so ``communication/voice.py`` can
hold either one and ``_on_utterance`` never learns which. Only the
boundary source differs: a finger going down opens the buffer, a finger
coming up finalizes it.

That single change deletes three subsystems from the voice path — no
Silero VAD (no torch, no model fetch at boot), no wake word, no AEC.
The mic is live only while held and TTS only plays when it is not, so
BB's own voice can never land in the buffer; there is no echo path to
cancel.

Used on hosts with a physical talk button, where the microphone module publishes
button presses as ``ButtonPressed`` events on the internal event bus.

**The two edges have different owners.** Release is handled here, off
the bus. Press is *not*: ``communication/voice.py`` owns it and calls
:meth:`begin_hold` explicitly, because a press during TTS must stop
playback before the buffer opens. Both handlers used to sit on the bus,
which dispatches concurrently (``asyncio.gather``), so "playback stops
first" was a race nobody could win. Now it is call order.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Awaitable, Callable

from boxbot.communication.utterance import Utterance
from boxbot.core.config import TurnDetectionConfig
from boxbot.core.events import ButtonPressed, get_event_bus

if TYPE_CHECKING:
    from boxbot.hardware.base import AudioChunk, AudioFanout

logger = logging.getLogger(__name__)


class PushToTalkCapture:
    """Accumulates microphone audio for as long as a button is held.

    Registers as a microphone consumer and subscribes to
    :class:`~boxbot.core.events.ButtonPressed` for the *release* edge
    only. :meth:`begin_hold` starts a hold — called by ``voice.py``
    after it has stopped playback, not fired off the bus. Release
    packages everything captured during the hold as an
    :class:`~boxbot.communication.utterance.Utterance` and fires the
    callback.

    Args:
        config: Turn-detection settings. Only ``max_utterance_duration``
            is read — silence thresholds are meaningless when the user
            decides the boundary.
        button_id: Which button owns the microphone. The panel sends one
            id for the whole screen (``"screen"``); presses from any
            other button are ignored so a future KB2040 knob cannot
            start a recording.
    """

    def __init__(
        self,
        config: TurnDetectionConfig,
        button_id: str = "screen",
    ) -> None:
        self._config = config
        self._button_id = button_id
        self._microphone: AudioFanout | None = None
        # Integer handle returned by microphone.add_consumer. We MUST
        # store this because bound methods like self._on_audio_chunk are
        # not identity-stable across accesses, so the handle is the only
        # reliable key for remove_consumer. The event bus has no such
        # problem — it keys on equality, and bound methods compare equal.
        self._consumer_handle: int | None = None
        self._callback: Callable[[Utterance], Awaitable[None]] | None = None
        # Fired when a hold ends without producing an utterance. The
        # press opened a streaming STT socket; if no utterance is coming
        # then no harvest is coming either, and somebody has to close it.
        self._discarded_callback: Callable[[str], Awaitable[None]] | None = None

        # Hold state. ``_holding`` is the whole state machine: chunks are
        # accumulated iff a hold is open.
        self._buffer = bytearray()
        self._holding = False
        self._press_time: float = 0.0
        self._sample_rate: int = 16000

        # Mute: see AudioCapture.mute. Gated at press rather than at the
        # chunk, because a deliberate press is unambiguous intent and a
        # hold that silently records nothing is worse than one that never
        # starts.
        self._muted = False

    @property
    def is_running(self) -> bool:
        """True iff the consumer is attached to a microphone."""
        return self._consumer_handle is not None

    @property
    def is_holding(self) -> bool:
        """True while a press is open and audio is accumulating."""
        return self._holding

    async def start(self, microphone: AudioFanout) -> None:
        """Register as microphone consumer and start listening for touch.

        Args:
            microphone: Microphone HAL instance with add_consumer/remove_consumer.
        """
        if self._consumer_handle is not None:
            logger.debug(
                "PushToTalkCapture already started, skipping re-registration"
            )
            return
        self._microphone = microphone
        self._consumer_handle = microphone.add_consumer(
            self._on_audio_chunk, name="push_to_talk",
        )
        get_event_bus().subscribe(ButtonPressed, self._on_button)
        logger.info(
            "PushToTalkCapture started (consumer=%d, button=%s)",
            self._consumer_handle, self._button_id,
        )

    async def stop(self) -> None:
        """Unregister from the mic and the bus, dropping any open hold.

        A stop mid-hold discards the held audio: the session is going
        away, and half a hold nobody released is not a turn. Logs only
        when a consumer was actually detached, so a stop on an
        already-stopped capture doesn't claim the mic was turned off.
        """
        was_running = self._consumer_handle is not None
        get_event_bus().unsubscribe(ButtonPressed, self._on_button)
        if self._microphone is not None and self._consumer_handle is not None:
            self._microphone.remove_consumer(self._consumer_handle)
            self._consumer_handle = None
            self._microphone = None
        self.reset()
        if was_running:
            logger.info("PushToTalkCapture stopped")

    def set_utterance_callback(
        self, callback: Callable[[Utterance], Awaitable[None]]
    ) -> None:
        """Set callback for when an utterance is finalized.

        Args:
            callback: Async callable that receives the completed Utterance.
        """
        self._callback = callback

    def set_hold_discarded_callback(
        self, callback: Callable[[str], Awaitable[None]]
    ) -> None:
        """Set callback for a hold that ends with nothing to transcribe.

        Fires on every release that will not reach
        ``set_utterance_callback``: a stray touch-up, and a hold that
        captured no audio. The argument is a short reason string for
        logging. Exists so the caller can retire per-press resources
        (the streaming STT socket) that the utterance path would
        otherwise have retired for it.

        Args:
            callback: Async callable that receives the reason.
        """
        self._discarded_callback = callback

    def begin_hold(self) -> None:
        """Open a hold. Sync, non-blocking, never raises.

        The press edge. Called by ``voice.py::_on_button_press`` *after*
        playback has stopped, so a hold can never be open while BB is
        still audible — the property push-to-talk trades AEC for.

        Sync on purpose, like ``StreamingSTTProvider.open``: the first
        audio must be captured on the touch-down, never behind an await.

        A second call with a hold already open is ignored (a replayed
        touch frame, not a new utterance); restarting the buffer would
        silently drop what has already been said.
        """
        if self._holding:
            logger.debug("begin_hold while already holding, ignoring")
            return
        if self._muted:
            logger.debug("begin_hold while muted, ignoring")
            return
        self._buffer = bytearray()
        self._holding = True
        self._press_time = time.monotonic()
        logger.debug("Hold opened at %.3f", self._press_time)

    def reset(self) -> None:
        """Drop any open hold. Resets capture state between sessions."""
        self._buffer = bytearray()
        self._holding = False
        self._press_time = 0.0

    def mute(self) -> None:
        """Ignore presses until unmuted. Drops any open hold.

        Idempotent. Persists across stop/start, matching
        :meth:`AudioCapture.mute` — the ``mute_mic`` tool contract is the
        same regardless of how turns are bounded.
        """
        if self._muted:
            return
        self._muted = True
        self.reset()
        logger.info("PushToTalkCapture muted")

    def unmute(self) -> None:
        """Resume honouring presses. Idempotent."""
        if not self._muted:
            return
        self._muted = False
        logger.info("PushToTalkCapture unmuted")

    @property
    def is_muted(self) -> bool:
        return self._muted

    async def _on_button(self, event: ButtonPressed) -> None:
        """Finalize the utterance on release.

        Release only. The press edge is ``voice.py``'s — see
        :meth:`begin_hold` and this module's docstring. "press",
        "long_press", and anything added later move nothing here.
        """
        if event.button_id != self._button_id:
            return
        if event.action != "release":
            return

        if not self._holding:
            # Stray touch-up, or the duration cap already finalized this
            # hold, or a press that never opened one (muted, or arriving
            # while the session was tearing down). Nothing left to send,
            # and firing an empty utterance would cost an STT call — but
            # the press may still have opened a socket, so say so.
            logger.debug("Release with no open hold, ignoring")
            await self._notify_discarded("release with no open hold")
            return
        # Release is true speech-end, and it is on time.monotonic()
        # like every downstream latency mark.
        await self._finalize_utterance(time.monotonic())

    async def _on_audio_chunk(self, chunk: AudioChunk) -> None:
        """Accumulate PCM while a hold is open; enforce the duration cap."""
        self._sample_rate = chunk.sample_rate
        if not self._holding:
            # No hold: drop it. The panel APK only opens its AudioRecord
            # on touch-down, but a PortAudio backend fans out
            # continuously — dropping here is what makes "the mic is live
            # only while held" true in Python too.
            return

        self._buffer.extend(chunk.data)

        # Hard cap the hold, auto-finalizing what we have. Checked on the
        # chunk path rather than on a timer because the cap exists to
        # bound the buffer and the buffer only grows when chunks arrive.
        # 16 kHz mono s16le = 32 KB/s, so the 60 s default bounds it near
        # 1.9 MB. Auto-finalize rather than drop: a minute of speech is
        # worth transcribing, and a finger stuck down (or a release frame
        # lost on the socket) must not deadlock the turn.
        elapsed = chunk.timestamp - self._press_time
        if elapsed >= self._config.max_utterance_duration:
            logger.info(
                "Hold hit max duration (%.1fs), auto-finalizing", elapsed
            )
            # Closes the hold, so audio until the (possibly never
            # arriving) release is dropped and that release is a no-op.
            await self._finalize_utterance(chunk.timestamp)

    async def _finalize_utterance(self, timestamp_end: float) -> None:
        """Package the held audio into an Utterance and fire the callback.

        ``duration`` is the hold, press to ``timestamp_end``; the audio
        can be shorter when the mic opens late (the panel starts its
        AudioRecord on touch-down, so the first chunk trails the press).
        """
        audio = bytes(self._buffer)
        press_time = self._press_time

        # Reset before the callback so a press during it opens cleanly.
        self.reset()

        if not audio:
            # Held, but nothing arrived: mic never opened, or the hold
            # was shorter than one chunk. Nothing to transcribe.
            logger.debug("Hold produced no audio")
            await self._notify_discarded("hold produced no audio")
            return

        utterance = Utterance(
            audio=audio,
            duration=timestamp_end - press_time,
            sample_rate=self._sample_rate,
            timestamp_start=press_time,
            timestamp_end=timestamp_end,
        )

        logger.info(
            "Utterance finalized: %.2fs held (%d bytes)",
            utterance.duration,
            len(audio),
        )

        if self._callback is not None:
            await self._callback(utterance)

    async def _notify_discarded(self, reason: str) -> None:
        """Tell the caller this hold produced nothing. Never raises."""
        if self._discarded_callback is None:
            return
        try:
            await self._discarded_callback(reason)
        except Exception:
            logger.exception("Hold-discarded callback failed")
