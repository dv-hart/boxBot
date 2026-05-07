"""Voice session state machine — the main voice pipeline orchestrator.

Ties together wake word detection, VAD, audio capture, STT, TTS, and
diarization into a coherent voice interaction lifecycle.

Interruption is wake-word-only: while BB is speaking, the STT consumer
is detached from the mic (so household chatter and BB's own residual
echo cannot reach the transcript pipeline); the wake word handler is
the only path that re-engages STT mid-reply.

States:
    IDLE       → Wake word detector active, waiting for activation
    ACTIVE     → Full voice pipeline running, capturing and processing speech
    SUSPENDED  → Mic processing paused, context retained, waiting for re-activation
    ENDED      → Session cleanup complete, returns to IDLE

Usage:
    session = VoiceSession(microphone, speaker, config)
    await session.start()  # enters IDLE, starts wake word detection

    # Session activates automatically on wake word, or:
    await session.initiate_conversation("Hello!", person_name="Jacob")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any

from boxbot.communication.audio_capture import AudioCapture, Utterance
from boxbot.communication.stt import ElevenLabsSTT, STTResult
from boxbot.communication.tts import ElevenLabsTTS, TTSStream
from boxbot.communication.vad import VoiceActivityDetector
from boxbot.communication.wake_word import WakeWordDetector
from boxbot.core.events import (
    AgentSpeaking,
    AgentSpeakingDone,
    AgentTurnEnded,
    TranscriptReady,
    VoiceSessionEnded,
    WakeWordHeard,
    get_event_bus,
)

if TYPE_CHECKING:
    from boxbot.communication.diarization import SpeakerDiarizer
    from boxbot.core.config import VoiceConfig
    from boxbot.hardware.microphone import Microphone
    from boxbot.hardware.speaker import Speaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_voice_session: VoiceSession | None = None


def get_voice_session() -> VoiceSession | None:
    """Return the global VoiceSession singleton, or None if not initialised."""
    return _voice_session


def set_voice_session(session: VoiceSession | None) -> None:
    """Set the global VoiceSession singleton."""
    global _voice_session
    _voice_session = session


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class VoiceSessionState(Enum):
    """Voice adapter hardware-capture state.

    Post-refactor the voice layer is a thin adapter — these states
    describe whether the mic is actively producing utterances.
    Conversation-level state (listening/thinking/speaking) lives on the
    ``Conversation`` object, not here.

    DORMANT means "mic off, but the agent's room conversation is still
    alive" — entered after the post-response idle window expires. The
    next wake word transitions DORMANT → ACTIVE while preserving the
    voice session id so the agent treats the new transcript as a
    continuation of the same conversation. ConversationEnded
    transitions DORMANT → IDLE for full cleanup.

    ENDED and SUSPENDED are retained as aliases for IDLE so older call
    sites don't crash during migration; M6 removes them.
    """

    IDLE = "idle"
    ACTIVE = "active"
    DORMANT = "dormant"
    SUSPENDED = "idle"   # deprecated alias
    ENDED = "idle"       # deprecated alias


# ---------------------------------------------------------------------------
# VoiceSession
# ---------------------------------------------------------------------------


class VoiceSession:
    """Main voice pipeline controller and state machine.

    Orchestrates all voice components: wake word detection, VAD, audio
    capture, STT, TTS, and diarization. Manages consumer registration
    on the microphone based on the current state.
    """

    def __init__(
        self,
        microphone: Microphone | None,
        speaker: Speaker | None,
        config: VoiceConfig,
    ) -> None:
        self._microphone = microphone
        self._speaker = speaker
        self._config = config

        self._state = VoiceSessionState.ENDED
        self._conversation_id: str = ""

        # Voice pipeline components (built lazily in start())
        self._wake_word: WakeWordDetector | None = None
        self._vad: VoiceActivityDetector | None = None
        self._audio_capture: AudioCapture | None = None
        self._stt: ElevenLabsSTT | None = None
        self._tts: ElevenLabsTTS | None = None
        self._tts_stream: TTSStream | None = None
        self._diarizer: SpeakerDiarizer | None = None

        # Session management
        self._session_task: asyncio.Task[None] | None = None
        self._suspend_timer: asyncio.Task[None] | None = None
        self._active_timeout_task: asyncio.Task[None] | None = None
        self._last_speech_time: float = 0.0

        # Set true when the wake word handler interrupts an in-flight
        # TTS playback. ``speak()``'s finally block checks this so it
        # does not double-publish AgentSpeakingDone.
        self._tts_interrupted: bool = False

        # Set when _activate_session resumed a DORMANT session (i.e.
        # the room conversation on the agent side is still alive). If
        # the post-wake-word grace expires before an utterance arrives,
        # we fall back to DORMANT instead of IDLE so the conversation
        # thread isn't thrown away by an accidental wake word.
        self._activation_was_resume: bool = False

        # Speaker identity mapping (SPEAKER_XX → person name or display label)
        # Populated from voice ReID matches (high-confidence) + agent
        # identify_person calls via SpeakerIdentified events.
        self._speaker_identities: dict[str, str] = {}

        # Stable display-label assignment for anonymous speakers.
        # pyannote produces "SPEAKER_00", "SPEAKER_01", ...; we present
        # them to the agent as "Speaker A", "Speaker B", ... Mapping is
        # session-scoped so labels stay consistent across utterances.
        self._display_labels: dict[str, str] = {}

        # Richer per-speaker identity snapshot for the TranscriptReady
        # event. Keyed by display label (e.g. "Speaker A", "Jacob"). Built
        # fresh per utterance from voice-ReID + any enrollment claim.
        self._latest_speaker_identities: dict[str, dict[str, Any]] = {}

        # Diarizer lazy-loading state
        self._diarizer_loaded: bool = False
        self._diarizer_unload_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> VoiceSessionState:
        """Current session state."""
        return self._state

    @property
    def conversation_id(self) -> str:
        """Current conversation ID, empty if no active session."""
        return self._conversation_id

    def update_speaker_identities(self, identities: dict[str, str]) -> None:
        """Update the speaker identity mapping for transcript attribution."""
        self._speaker_identities.update(identities)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise all voice components and enter IDLE state.

        Starts wake word detection (always active). Other components
        are initialised but only registered as mic consumers when a
        session becomes ACTIVE.
        """
        if self._microphone is None:
            logger.warning("Voice session cannot start: no microphone")
            return

        # Build components
        self._wake_word = WakeWordDetector(self._config.wake_word)
        self._vad = VoiceActivityDetector(self._config.vad)
        self._audio_capture = AudioCapture(
            self._vad, self._config.turn_detection
        )
        self._audio_capture.set_utterance_callback(self._on_utterance)

        # STT
        from boxbot.core.config import get_config

        api_keys = get_config().api_keys
        if api_keys.elevenlabs:
            self._stt = ElevenLabsSTT(
                api_key=api_keys.elevenlabs,
                model=self._config.stt.model,
            )
            self._tts = ElevenLabsTTS(
                api_key=api_keys.elevenlabs,
                voice_id=self._config.tts.voice_id,
                model=self._config.tts.model,
                stability=self._config.tts.stability,
                similarity_boost=self._config.tts.similarity_boost,
                optimize_streaming_latency=self._config.tts.optimize_streaming_latency,
            )
            if self._speaker:
                self._tts_stream = TTSStream(self._tts, self._speaker)
        else:
            logger.warning(
                "ElevenLabs API key not configured — STT/TTS disabled"
            )

        # Diarization (lazy-loaded when person detected)
        try:
            from boxbot.communication.diarization import SpeakerDiarizer

            self._diarizer = SpeakerDiarizer(self._config.diarization)
            self._diarizer_loaded = False
            # Don't call start() here — models will be lazy-loaded
        except Exception:
            logger.warning(
                "Speaker diarization not available", exc_info=True
            )
            self._diarizer = None
            self._diarizer_loaded = False

        # Start VAD model
        await self._vad.start()

        # Start wake word detection
        await self._wake_word.start(self._microphone)

        # Subscribe to events
        bus = get_event_bus()
        bus.subscribe(WakeWordHeard, self._on_wake_word)

        from boxbot.core.events import ConversationEnded, PersonDetected
        bus.subscribe(PersonDetected, self._on_person_detected)
        # Adapter mirrors the agent's room-conversation lifecycle: when
        # the voice conversation ends (silence timeout, explicit end,
        # agent stop), we deactivate mic capture and LEDs.
        bus.subscribe(ConversationEnded, self._on_conversation_ended)
        # Post-response mic-idle window: AgentTurnEnded fires when a
        # turn settles in LISTENING (whether or not BB spoke). We arm a
        # short timer; if no further utterance arrives we go DORMANT so
        # ambient chatter doesn't keep the mic hot indefinitely.
        bus.subscribe(AgentTurnEnded, self._on_agent_turn_ended)
        bus.subscribe(AgentSpeaking, self._on_agent_speaking)

        self._state = VoiceSessionState.IDLE
        set_voice_session(self)
        logger.info("Voice session started (IDLE)")

    async def stop(self) -> None:
        """Shut down all voice components and clean up."""
        # Cancel any active session
        if self._session_task and not self._session_task.done():
            self._session_task.cancel()
            try:
                await self._session_task
            except asyncio.CancelledError:
                pass

        self._cancel_timers()

        # Cancel diarizer unload task
        if self._diarizer_unload_task and not self._diarizer_unload_task.done():
            self._diarizer_unload_task.cancel()

        # Unsubscribe from events
        bus = get_event_bus()
        bus.unsubscribe(WakeWordHeard, self._on_wake_word)

        from boxbot.core.events import ConversationEnded, PersonDetected
        bus.unsubscribe(PersonDetected, self._on_person_detected)
        bus.unsubscribe(ConversationEnded, self._on_conversation_ended)
        bus.unsubscribe(AgentTurnEnded, self._on_agent_turn_ended)
        bus.unsubscribe(AgentSpeaking, self._on_agent_speaking)

        # Stop components
        if self._wake_word:
            await self._wake_word.stop()
        if self._audio_capture:
            await self._audio_capture.stop()
        if self._vad:
            await self._vad.stop()
        if self._diarizer and self._diarizer_loaded:
            await self._diarizer.stop()

        # Stop any playback
        if self._speaker and self._speaker.is_playing:
            await self._speaker.stop_playback()

        self._state = VoiceSessionState.ENDED
        set_voice_session(None)
        logger.info("Voice session stopped")

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    # Post-wake-word grace: if no utterance arrives within this many
    # seconds we deactivate. Guards against false wake-word detections
    # from draining batteries with a hot mic indefinitely.
    _WAKE_WORD_GRACE_SECONDS = 12.0

    # Post-response mic-idle window: after BB's turn ends (whether or
    # not it spoke) the mic stays hot for this long. Past that we go
    # DORMANT so ambient chatter doesn't keep round-tripping transcripts
    # to the agent ("not addressed to me, say nothing" forever). The
    # conversation thread persists; the next wake word continues it.
    _POST_RESPONSE_IDLE_SECONDS = 15.0

    async def _activate_session(
        self, event: WakeWordHeard | None = None
    ) -> None:
        """Turn on audio capture. LED goes to listening.

        Idempotent — calling while ACTIVE just refreshes the grace timer
        and ensures audio capture is re-attached. The wake word is the
        way to re-enable STT after BB finishes (or is interrupted in
        the middle of) a reply; audio_capture is detached during speech
        so re-attachment must happen on every wake event.

        From DORMANT the existing voice session id is preserved so the
        agent routes the next transcript into the same room
        conversation — wake word resumes, it does not restart.
        """
        self._last_speech_time = time.monotonic()
        prev_state = self._state

        if prev_state is VoiceSessionState.IDLE:
            self._state = VoiceSessionState.ACTIVE
            self._conversation_id = f"voice_{uuid.uuid4().hex[:12]}"
            self._activation_was_resume = False
            logger.info(
                "Voice adapter activated (session=%s)", self._conversation_id,
            )
        elif prev_state is VoiceSessionState.DORMANT:
            self._state = VoiceSessionState.ACTIVE
            self._activation_was_resume = True
            logger.info(
                "Voice adapter re-engaged DORMANT → ACTIVE "
                "(session=%s)", self._conversation_id,
            )
        # else: already ACTIVE — idempotent path (refresh grace timer
        # and re-attach capture below).

        # Always (re-)attach the STT/diarization consumer. ``start`` is
        # idempotent: a no-op if the consumer handle is already live.
        if self._microphone and self._audio_capture:
            await self._audio_capture.start(self._microphone)

        # Post-wake-word grace — if a TranscriptReady publishes, the
        # grace is cancelled and AgentTurnEnded later takes over the
        # post-response idle window.
        self._reset_wake_word_grace_timer()

        # Set LED pattern.
        if self._microphone:
            try:
                await self._microphone.set_led_pattern("listening")
            except Exception:
                pass

    def _reset_wake_word_grace_timer(self) -> None:
        """Arm (or reset) the post-wake-word no-utterance grace timer."""
        if self._active_timeout_task is not None:
            self._active_timeout_task.cancel()
        self._active_timeout_task = asyncio.create_task(
            self._wake_word_grace_loop(),
            name=f"voice-grace-{self._conversation_id}",
        )

    async def _wake_word_grace_loop(self) -> None:
        """Deactivate if no utterance arrives within the grace window.

        Cold wake (no prior session): fall through to IDLE — there's no
        conversation to preserve. Resume from DORMANT: fall back to
        DORMANT so an accidental wake word doesn't throw away the
        still-alive room conversation.
        """
        try:
            await asyncio.sleep(self._WAKE_WORD_GRACE_SECONDS)
        except asyncio.CancelledError:
            return
        if self._state is not VoiceSessionState.ACTIVE:
            return
        if self._activation_was_resume:
            logger.info(
                "Voice adapter: no utterance within %.0fs grace — "
                "returning to DORMANT (session=%s preserved)",
                self._WAKE_WORD_GRACE_SECONDS, self._conversation_id,
            )
            await self._enter_dormant()
        else:
            logger.info(
                "Voice adapter: no utterance within %.0fs grace — "
                "deactivating",
                self._WAKE_WORD_GRACE_SECONDS,
            )
            await self._deactivate_session(reason="grace_timeout")

    def _arm_post_response_idle_timer(self) -> None:
        """Arm (or reset) the post-response mic-idle timer.

        Replaces any active timer (e.g. wake-word grace) — once a turn
        has ended, the post-response window is the only thing that
        should be ticking.
        """
        if self._active_timeout_task is not None:
            self._active_timeout_task.cancel()
        self._active_timeout_task = asyncio.create_task(
            self._post_response_idle_loop(),
            name=f"voice-idle-{self._conversation_id}",
        )

    async def _post_response_idle_loop(self) -> None:
        """Go DORMANT if no follow-up utterance arrives within the window."""
        try:
            await asyncio.sleep(self._POST_RESPONSE_IDLE_SECONDS)
        except asyncio.CancelledError:
            return
        if self._state is VoiceSessionState.ACTIVE:
            logger.info(
                "Voice adapter: %.0fs post-response idle — going DORMANT "
                "(mic off, session=%s retained)",
                self._POST_RESPONSE_IDLE_SECONDS,
                self._conversation_id,
            )
            await self._enter_dormant()

    async def _enter_dormant(self) -> None:
        """ACTIVE → DORMANT: stop capture, retain conv_id, LED off.

        The conversation thread on the agent side keeps living (its own
        silence_timeout decides when to actually end it). A wake word
        from DORMANT re-engages the same session id so transcripts flow
        into the existing room conversation; ConversationEnded from
        DORMANT triggers a normal IDLE cleanup via _deactivate_session.
        No VoiceSessionEnded is published here — the session id is
        still alive.
        """
        if self._state is not VoiceSessionState.ACTIVE:
            return

        self._state = VoiceSessionState.DORMANT

        if self._audio_capture:
            await self._audio_capture.stop()
            self._audio_capture.reset()

        if self._vad:
            self._vad.reset()

        self._cancel_timers()

        if self._microphone:
            try:
                await self._microphone.set_led_pattern("off")
            except Exception:
                pass

    async def _deactivate_session(self, *, reason: str = "external") -> None:
        """Turn off audio capture, publish VoiceSessionEnded.

        Called when:
        - A ConversationEnded(channel="voice") arrives (the agent's
          room conversation has terminated — silence, explicit end, etc.)
        - The post-wake-word grace window expires with no speech.

        Runs from both ACTIVE and DORMANT. From DORMANT capture is
        already stopped; this just clears the session id and publishes
        VoiceSessionEnded so perception runs its end-of-session
        enrollment flush.
        """
        if self._state not in (
            VoiceSessionState.ACTIVE, VoiceSessionState.DORMANT,
        ):
            return

        was_active = self._state is VoiceSessionState.ACTIVE
        self._state = VoiceSessionState.IDLE
        ending_session_id = self._conversation_id
        self._conversation_id = ""

        logger.info(
            "Voice adapter deactivated (session=%s, reason=%s, was=%s)",
            ending_session_id, reason,
            "active" if was_active else "dormant",
        )

        # From DORMANT capture is already stopped; only call stop again
        # if we were actually still capturing.
        if was_active and self._audio_capture:
            await self._audio_capture.stop()
            self._audio_capture.reset()

        if was_active and self._vad:
            self._vad.reset()

        self._cancel_timers()

        # Clear session-scoped identity state.
        self._speaker_identities.clear()
        self._display_labels.clear()
        self._latest_speaker_identities.clear()

        # Warm-unload diarizer after a minute of inactivity.
        if self._diarizer_loaded and self._diarizer:
            self._diarizer_unload_task = asyncio.create_task(
                self._unload_diarizer_after_timeout(60.0)
            )

        # Set LED pattern.
        if self._microphone:
            try:
                await self._microphone.set_led_pattern("off")
            except Exception:
                pass

        # Publish VoiceSessionEnded for perception-side enrollment
        # cleanup. If deactivation was triggered by ConversationEnded,
        # the agent's handler is idempotent (re-ending an ended conv
        # is a no-op) so no loop.
        if ending_session_id:
            try:
                bus = get_event_bus()
                await bus.publish(
                    VoiceSessionEnded(conversation_id=ending_session_id)
                )
            except Exception:
                logger.exception(
                    "Failed to publish VoiceSessionEnded for %s",
                    ending_session_id,
                )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_wake_word(self, event: WakeWordHeard) -> None:
        """Handle wake word detection.

        The wake word is the unified re-engagement signal: if BB is
        mid-TTS the playback stops and an ``AgentSpeakingDone(
        interrupted=True)`` event is published (the agent listens for
        this and calls ``Conversation.interrupt()`` to fold partial
        output and clear queued utterances). STT is re-attached either
        way — it's detached during BB's own speech, and the wake word
        is the only path back to a hot mic.
        """
        if (
            self._speaker is not None
            and self._speaker.is_playing
        ):
            logger.info("Wake word during TTS — stopping playback")
            self._tts_interrupted = True
            await self._speaker.stop_playback()
            try:
                await get_event_bus().publish(
                    AgentSpeakingDone(
                        conversation_id=self._conversation_id,
                        interrupted=True,
                    )
                )
            except Exception:
                logger.exception(
                    "Failed to publish AgentSpeakingDone after wake-word interrupt"
                )
        await self._activate_session(event)

    async def _on_conversation_ended(self, event: Any) -> None:
        """React to ConversationEnded — deactivate capture for voice.

        Only voice-channel conversation ends deactivate capture; whatsapp
        and trigger conversations don't touch the mic. Runs from ACTIVE
        or DORMANT — _deactivate_session handles both.
        """
        if getattr(event, "channel", "") == "voice":
            await self._deactivate_session(reason="conversation_ended")

    async def _on_agent_turn_ended(self, event: AgentTurnEnded) -> None:
        """Arm the post-response mic-idle timer.

        Fires once per turn when the room conversation settles in
        LISTENING. Includes turns where BB chose silence — that's the
        whole point: previously a silent turn left the mic hot for
        180s, so ambient chatter would round-trip through the model
        again and again. Now the timer arms regardless.
        """
        if event.channel != "voice":
            return
        if event.conversation_id != self._conversation_id:
            return
        if self._state is not VoiceSessionState.ACTIVE:
            return
        self._arm_post_response_idle_timer()

    async def _on_agent_speaking(self, event: AgentSpeaking) -> None:
        """Cancel the post-response idle timer — BB is speaking.

        Defensive: while BB speaks, audio_capture is detached anyway,
        but a stray timer firing mid-TTS would race with speak()'s
        re-attachment. Easier to just cancel here.
        """
        if event.conversation_id != self._conversation_id:
            return
        if self._active_timeout_task is not None:
            self._active_timeout_task.cancel()
            self._active_timeout_task = None

    async def _on_utterance(self, utterance: Utterance) -> None:
        """Process a finalized utterance: run STT + diarization, publish transcript."""
        if self._state is not VoiceSessionState.ACTIVE:
            return

        # An utterance arrived — cancel any pending mic-idle timer
        # (wake-word grace before the first turn, post-response idle
        # after). AgentTurnEnded re-arms the post-response timer when
        # the agent is done.
        if self._active_timeout_task is not None:
            self._active_timeout_task.cancel()
            self._active_timeout_task = None

        self._last_speech_time = time.monotonic()

        # Set LED pattern to "thinking"
        if self._microphone:
            try:
                await self._microphone.set_led_pattern("thinking")
            except Exception:
                pass

        # Ensure diarizer is loaded if available
        if self._diarizer and not self._diarizer_loaded:
            await self._ensure_diarizer_loaded()

        # Run STT and diarization in parallel
        stt_task = None
        diarize_task = None

        if self._stt:
            stt_task = asyncio.create_task(
                self._stt.transcribe(
                    utterance.audio,
                    utterance.sample_rate,
                    self._config.stt.language,
                    conversation_id=self._conversation_id or None,
                )
            )

        if self._diarizer and self._diarizer_loaded:
            diarize_task = asyncio.create_task(
                self._diarizer.diarize(
                    utterance.audio,
                    utterance.sample_rate,
                )
            )

        # Wait for results
        stt_result: STTResult | None = None
        diarization_result = None

        if stt_task:
            try:
                stt_result = await stt_task
            except Exception:
                logger.exception("STT failed for utterance")

        if diarize_task:
            try:
                diarization_result = await diarize_task
            except Exception:
                logger.exception("Diarization failed for utterance")

        if not stt_result or not stt_result.text.strip():
            # No transcript — go back to listening
            if self._microphone:
                try:
                    await self._microphone.set_led_pattern("listening")
                except Exception:
                    pass
            return

        # Run voice ReID against stored centroids for each diarization
        # segment with an embedding. Results seed enrollment claims and
        # the per-speaker identity block in the TranscriptReady event.
        # Also maps raw pyannote labels (SPEAKER_00) to stable display
        # labels (Speaker A) or known names (Jacob) for the transcript.
        label_map, identity_block = await self._resolve_speaker_identities(
            diarization_result
        )
        # Merge anything derived here with the session-scoped mapping used
        # by the transcript builder. Agent ``identify_person`` calls update
        # ``_speaker_identities`` via SpeakerIdentified events; voice-ReID
        # updates live-match labels.
        for raw_label, display_name in label_map.items():
            self._speaker_identities.setdefault(raw_label, display_name)
        self._latest_speaker_identities = identity_block

        # Build attributed transcript using the merged label map
        transcript = self._build_attributed_transcript(
            stt_result,
            diarization_result,
            speaker_identities=self._speaker_identities if self._speaker_identities else None,
        )

        # Build speaker segments data for the event
        speaker_segments: list[dict[str, Any]] = []
        if diarization_result:
            for seg in diarization_result.segments:
                seg_data: dict[str, Any] = {
                    "speaker": seg.speaker_label,
                    "start": seg.start,
                    "end": seg.end,
                }
                if seg.embedding is not None:
                    seg_data["embedding"] = seg.embedding
                speaker_segments.append(seg_data)

        # Publish TranscriptReady event
        bus = get_event_bus()
        await bus.publish(
            TranscriptReady(
                conversation_id=self._conversation_id,
                transcript=transcript,
                speaker_segments=speaker_segments,
                speaker_identities=identity_block,
                source="voice",
            )
        )

        # Set LED back to listening
        if self._microphone:
            try:
                await self._microphone.set_led_pattern("listening")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Speech output
    # ------------------------------------------------------------------

    async def speak(self, text: str, priority: str = "normal") -> None:
        """Speak text through TTS and the speaker.

        The STT/diarization consumer is detached from the microphone
        for the duration of TTS playback so household chatter and BB's
        residual echo cannot enter the transcript pipeline. Mid-reply
        interruption requires the wake word; the wake-word handler
        re-attaches STT itself in that path.

        On natural completion, STT is re-attached so the user can
        continue the conversation without saying the wake word again.
        The post-response idle timer (armed by AgentTurnEnded) gives
        the user ~15s to follow up; past that the adapter goes DORMANT
        and a wake word is required to resume.

        Args:
            text: The text to speak.
            priority: "normal" or "urgent". Urgent interrupts current audio.
        """
        if not self._tts_stream or not self._speaker:
            logger.warning("Cannot speak: TTS or speaker not available")
            return

        self._tts_interrupted = False

        # If urgent and something is already playing, stop it
        if priority == "urgent" and self._speaker.is_playing:
            await self._speaker.stop_playback()

        # Set LED pattern
        if self._microphone:
            try:
                await self._microphone.set_led_pattern("speaking")
            except Exception:
                pass

        # Publish speaking event — the agent uses this to flip the
        # room conversation into SPEAKING state.
        bus = get_event_bus()
        await bus.publish(
            AgentSpeaking(
                conversation_id=self._conversation_id,
                text=text,
            )
        )

        # Detach STT/diarization: chatter and BB's residual echo cannot
        # enter the transcript. The wake word handler is the only path
        # back to an attached audio_capture.
        if self._audio_capture is not None:
            await self._audio_capture.stop()

        try:
            await self._tts_stream.speak(
                text, conversation_id=self._conversation_id or None
            )
        except Exception:
            logger.exception("TTS playback failed")
        finally:
            # The wake-word handler publishes AgentSpeakingDone with
            # interrupted=True itself when it stops playback; only
            # publish the natural-completion event when that did not
            # happen.
            if not self._tts_interrupted:
                await bus.publish(
                    AgentSpeakingDone(
                        conversation_id=self._conversation_id,
                        interrupted=False,
                    )
                )

                # Natural TTS completion: re-attach STT so the user
                # can continue the conversation without re-saying the
                # wake word. The interrupt path skips this — the
                # wake-word handler re-attaches via _activate_session
                # itself.
                if (
                    self._state == VoiceSessionState.ACTIVE
                    and self._audio_capture is not None
                    and self._microphone is not None
                ):
                    try:
                        await self._audio_capture.start(self._microphone)
                    except Exception:
                        logger.exception(
                            "Failed to re-attach audio_capture after TTS"
                        )

            if self._state == VoiceSessionState.ACTIVE and self._microphone:
                try:
                    await self._microphone.set_led_pattern("listening")
                except Exception:
                    pass

    async def initiate_conversation(
        self, text: str, person_name: str | None = None
    ) -> None:
        """Start a conversation initiated by boxBot (not by wake word).

        Activates the session, speaks the initiating text, then enters
        the normal listening loop.

        Args:
            text: The text to speak to start the conversation.
            person_name: Optional name of the person to address.
        """
        if self._state != VoiceSessionState.IDLE:
            logger.warning(
                "Cannot initiate conversation: session in state %s",
                self._state.value,
            )
            return

        # Activate without wake word
        await self._activate_session()

        # Speak the initiating text
        await self.speak(text)

    # ------------------------------------------------------------------
    # Transcript building
    # ------------------------------------------------------------------

    def _assign_display_label(self, raw_label: str) -> str:
        """Map a raw pyannote label (SPEAKER_00) to a session-stable display
        label (Speaker A). Consistent across utterances in the same session.
        """
        if raw_label in self._display_labels:
            return self._display_labels[raw_label]
        idx = len(self._display_labels)
        # Speaker A, B, C, ..., Z, then AA, AB, ...
        if idx < 26:
            letter = chr(ord("A") + idx)
        else:
            letter = chr(ord("A") + (idx // 26) - 1) + chr(ord("A") + (idx % 26))
        display = f"Speaker {letter}"
        self._display_labels[raw_label] = display
        return display

    async def _resolve_speaker_identities(
        self,
        diarization_result: Any | None,
    ) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
        """Run voice ReID on this utterance's speaker segments.

        Returns:
            (label_map, identity_block)
            label_map: {raw_pyannote_label: display_name} — e.g. either
                {"SPEAKER_00": "Jacob"} (high-confidence match) or
                {"SPEAKER_00": "Speaker A"} (otherwise). Used to rewrite
                the transcript.
            identity_block: {display_name: {person_id, person_name,
                voice_tier, voice_score, source}} — the per-speaker
                identity snapshot surfaced on TranscriptReady.
        """
        label_map: dict[str, str] = {}
        identity_block: dict[str, dict[str, Any]] = {}

        if not diarization_result or not diarization_result.segments:
            return label_map, identity_block

        # Lazy-import perception to avoid circular deps and to gracefully
        # degrade when perception is disabled.
        voice_centroids: dict[str, tuple[str, Any]] = {}
        voice_reid: Any = None
        enrollment: Any = None
        try:
            from boxbot.perception.pipeline import get_pipeline
            from boxbot.perception.voice_reid import VoiceReID

            pipeline = get_pipeline()
            if pipeline is not None:
                store = getattr(pipeline, "_cloud_store", None)
                enrollment = getattr(pipeline, "enrollment", None)
                if store is not None:
                    try:
                        voice_centroids = await store.get_voice_centroids()
                    except Exception:
                        logger.debug(
                            "Could not load voice centroids", exc_info=True
                        )
                voice_reid = VoiceReID()
        except Exception:
            # Pipeline not running — we'll still assign display labels.
            logger.debug(
                "Voice ReID/enrollment unavailable; proceeding label-only",
                exc_info=True,
            )

        # Pick ONE representative embedding per raw speaker label
        # (multiple segments from the same speaker in one utterance share
        # the same identity). Use the first non-None embedding.
        seen_raw: set[str] = set()
        for seg in diarization_result.segments:
            raw = seg.speaker_label
            if raw in seen_raw:
                continue
            embedding = seg.embedding
            if embedding is None:
                # No embedding on this segment — assign label and skip ReID
                display = self._assign_display_label(raw)
                label_map[raw] = display
                identity_block.setdefault(display, {
                    "person_id": None,
                    "person_name": None,
                    "voice_tier": "unknown",
                    "voice_score": 0.0,
                    "source": "unknown",
                })
                seen_raw.add(raw)
                continue
            seen_raw.add(raw)

            # Run voice ReID if we have the machinery
            if voice_reid is not None and voice_centroids:
                try:
                    match = voice_reid.match(embedding, voice_centroids)
                except Exception:
                    logger.debug("Voice ReID match failed", exc_info=True)
                    match = None
            else:
                match = None

            # Decide the display label:
            # - high-confidence match → use the matched person's name
            # - everything else → stable anonymous "Speaker X"
            if match is not None and match.tier == "high" and match.person_name:
                display = match.person_name
            else:
                display = self._assign_display_label(raw)
            label_map[raw] = display

            # Build the identity entry for this speaker
            entry: dict[str, Any] = {
                "person_id": match.person_id if match else None,
                "person_name": match.person_name if match else None,
                "voice_tier": match.tier if match else "unknown",
                "voice_score": float(match.confidence) if match else 0.0,
                "source": "voice_reid_match" if match and match.person_id else "unknown",
            }

            # If enrollment has an agent-established claim for this ref,
            # that trumps ReID — the agent has explicitly said who this is.
            if enrollment is not None:
                claim = enrollment.get_claim(raw)
                if claim and claim.source == "agent_identify" and claim.name:
                    # Prefer the display_label to the person's actual name
                    # in the transcript so we don't confuse the model if it
                    # was told "I'm Carina" but the claim is "Sarah" after a
                    # correction. Keep identity_block pointing to the claim.
                    display = claim.name
                    label_map[raw] = display
                    entry["person_id"] = claim.person_id
                    entry["person_name"] = claim.name
                    entry["source"] = "agent_identify"

            identity_block[display] = entry

            # Buffer the embedding + seed the session claim so commit at
            # session end routes correctly. Uses the raw pyannote label as
            # the stable ref key within the session (not the display label).
            if enrollment is not None:
                try:
                    enrollment.buffer_voice_embedding(raw, embedding)
                    if match is not None:
                        enrollment.on_reid_match(
                            raw,
                            "voice",
                            person_id=match.person_id,
                            person_name=match.person_name,
                            tier=match.tier,
                            score=float(match.confidence),
                        )
                except Exception:
                    logger.debug(
                        "Enrollment buffer/seed failed for ref=%s", raw,
                        exc_info=True,
                    )

        return label_map, identity_block

    @staticmethod
    def _build_attributed_transcript(
        stt_result: STTResult,
        diarization_result: Any | None,
        speaker_identities: dict[str, str] | None = None,
    ) -> str:
        """Align STT transcript with diarization segments.

        If diarization is available, attributes each portion of the
        transcript to a speaker label. When speaker_identities is
        provided, maps diarization labels (e.g. SPEAKER_00) to known
        person names.
        """
        if not diarization_result or not diarization_result.segments:
            # No diarization — return plain transcript
            return stt_result.text

        if not stt_result.words:
            # No word-level timing — can't align, prefix with first speaker
            first_speaker = diarization_result.segments[0].speaker_label
            if speaker_identities and first_speaker in speaker_identities:
                first_speaker = speaker_identities[first_speaker]
            return f"[{first_speaker}]: {stt_result.text}"

        # Align words to speaker segments
        segments = sorted(diarization_result.segments, key=lambda s: s.start)
        lines: list[str] = []
        current_speaker = ""
        current_words: list[str] = []

        for word_info in stt_result.words:
            word_mid = (word_info.start + word_info.end) / 2
            # Find which speaker segment this word belongs to
            speaker = _find_speaker_for_time(segments, word_mid)
            # Map to known identity if available
            if speaker_identities and speaker in speaker_identities:
                speaker = speaker_identities[speaker]

            if speaker != current_speaker:
                # Flush current line
                if current_words:
                    lines.append(
                        f"[{current_speaker}]: {' '.join(current_words)}"
                    )
                current_speaker = speaker
                current_words = [word_info.word]
            else:
                current_words.append(word_info.word)

        # Flush final line
        if current_words:
            lines.append(f"[{current_speaker}]: {' '.join(current_words)}")

        return "\n".join(lines) if lines else stt_result.text

    # ------------------------------------------------------------------
    # Lazy diarizer loading
    # ------------------------------------------------------------------

    async def _ensure_diarizer_loaded(self) -> None:
        """Lazy-load diarizer models if not already loaded."""
        if self._diarizer is None or self._diarizer_loaded:
            return

        # Cancel any pending unload
        if self._diarizer_unload_task and not self._diarizer_unload_task.done():
            self._diarizer_unload_task.cancel()
            self._diarizer_unload_task = None

        try:
            logger.info("Lazy-loading diarization models...")
            await self._diarizer.start()
            self._diarizer_loaded = True
            logger.info("Diarization models loaded")
        except Exception:
            logger.warning("Failed to lazy-load diarization models", exc_info=True)

    async def _on_person_detected(self, event: Any) -> None:
        """Lazy-load diarizer when a person is detected."""
        await self._ensure_diarizer_loaded()

    async def _unload_diarizer_after_timeout(self, timeout: float) -> None:
        """Unload diarizer models after timeout to reclaim RAM."""
        try:
            await asyncio.sleep(timeout)
            if self._diarizer and self._diarizer_loaded:
                await self._diarizer.stop()
                self._diarizer_loaded = False
                logger.info("Diarization models unloaded (warm timeout)")
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Timeout management
    # ------------------------------------------------------------------

    async def _active_timeout_loop(self) -> None:
        """Monitor for silence and trigger session suspension."""
        timeout = self._config.session.active_timeout
        try:
            while self._state == VoiceSessionState.ACTIVE:
                await asyncio.sleep(1.0)
                elapsed = time.monotonic() - self._last_speech_time
                if elapsed >= timeout:
                    logger.info(
                        "Active timeout (%.0fs silence), suspending session",
                        elapsed,
                    )
                    await self._suspend_session()
                    return
        except asyncio.CancelledError:
            pass

    async def _suspend_timeout_loop(self) -> None:
        """Monitor suspend duration and end session if exceeded."""
        timeout = self._config.session.suspend_timeout
        try:
            await asyncio.sleep(timeout)
            if self._state == VoiceSessionState.SUSPENDED:
                logger.info(
                    "Suspend timeout (%ds), ending session", timeout
                )
                await self._end_session()
        except asyncio.CancelledError:
            pass

    def _cancel_timers(self) -> None:
        """Cancel all active timer tasks."""
        for task in (self._active_timeout_task, self._suspend_timer):
            if task and not task.done():
                task.cancel()
        self._active_timeout_task = None
        self._suspend_timer = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_speaker_for_time(
    segments: list[Any], t: float
) -> str:
    """Find which speaker segment contains the given time point."""
    for seg in segments:
        if seg.start <= t <= seg.end:
            return seg.speaker_label
    # If no segment matches, use the nearest one
    if not segments:
        return "SPEAKER_00"
    nearest = min(segments, key=lambda s: min(abs(s.start - t), abs(s.end - t)))
    return nearest.speaker_label
