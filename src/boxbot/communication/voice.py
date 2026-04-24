"""Voice session state machine — the main voice pipeline orchestrator.

Ties together wake word detection, VAD, audio capture, STT, TTS,
diarization, and barge-in monitoring into a coherent voice interaction
lifecycle.

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
from boxbot.communication.barge_in import BargeInMonitor
from boxbot.communication.stt import ElevenLabsSTT, STTResult
from boxbot.communication.tts import ElevenLabsTTS, TTSStream
from boxbot.communication.vad import VoiceActivityDetector
from boxbot.communication.wake_word import WakeWordDetector
from boxbot.core.events import (
    AgentSpeaking,
    AgentSpeakingDone,
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
    """Voice session lifecycle states."""

    IDLE = "idle"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ENDED = "ended"


# ---------------------------------------------------------------------------
# VoiceSession
# ---------------------------------------------------------------------------


class VoiceSession:
    """Main voice pipeline controller and state machine.

    Orchestrates all voice components: wake word detection, VAD, audio
    capture, STT, TTS, diarization, and barge-in. Manages consumer
    registration on the microphone based on the current state.
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
        self._barge_in: BargeInMonitor | None = None

        # Session management
        self._session_task: asyncio.Task[None] | None = None
        self._suspend_timer: asyncio.Task[None] | None = None
        self._active_timeout_task: asyncio.Task[None] | None = None
        self._last_speech_time: float = 0.0

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

        # Barge-in monitor
        if self._speaker and self._vad:
            self._barge_in = BargeInMonitor(
                self._vad, self._speaker, self._config.barge_in
            )
            self._barge_in.set_interrupt_callback(self._on_barge_in)

        # Start VAD model
        await self._vad.start()

        # Start wake word detection
        await self._wake_word.start(self._microphone)

        # Subscribe to events
        bus = get_event_bus()
        bus.subscribe(WakeWordHeard, self._on_wake_word)

        from boxbot.core.events import PersonDetected
        bus.subscribe(PersonDetected, self._on_person_detected)

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

        from boxbot.core.events import PersonDetected
        bus.unsubscribe(PersonDetected, self._on_person_detected)

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

    async def _activate_session(
        self, event: WakeWordHeard | None = None
    ) -> None:
        """Transition from IDLE or SUSPENDED to ACTIVE."""
        if self._state == VoiceSessionState.ACTIVE:
            logger.debug("Session already active, ignoring activation")
            return

        prev_state = self._state
        self._state = VoiceSessionState.ACTIVE
        self._last_speech_time = time.monotonic()

        if prev_state == VoiceSessionState.IDLE:
            # New conversation
            self._conversation_id = f"voice_{uuid.uuid4().hex[:12]}"
            logger.info(
                "Voice session activated (new conversation %s)",
                self._conversation_id,
            )
        else:
            # Resuming from suspended
            logger.info(
                "Voice session re-activated (conversation %s)",
                self._conversation_id,
            )

        self._cancel_timers()

        # Register audio processing consumers
        if self._microphone and self._audio_capture:
            await self._audio_capture.start(self._microphone)

        # Start active timeout monitor
        self._active_timeout_task = asyncio.create_task(
            self._active_timeout_loop(),
            name=f"voice-timeout-{self._conversation_id}",
        )

        # Set LED pattern
        if self._microphone:
            try:
                await self._microphone.set_led_pattern("listening")
            except Exception:
                pass

    async def _suspend_session(self) -> None:
        """Transition from ACTIVE to SUSPENDED (silence timeout)."""
        if self._state != VoiceSessionState.ACTIVE:
            return

        self._state = VoiceSessionState.SUSPENDED
        logger.info(
            "Voice session suspended (conversation %s)",
            self._conversation_id,
        )

        # Remove audio processing consumers (save CPU)
        if self._audio_capture:
            await self._audio_capture.stop()
            self._audio_capture.reset()

        self._cancel_timers()

        # Start suspend timeout
        self._suspend_timer = asyncio.create_task(
            self._suspend_timeout_loop(),
            name=f"voice-suspend-{self._conversation_id}",
        )

        # Set LED pattern
        if self._microphone:
            try:
                await self._microphone.set_led_pattern("idle")
            except Exception:
                pass

    async def _end_session(self) -> None:
        """Transition to ENDED and return to IDLE."""
        prev_state = self._state
        self._state = VoiceSessionState.ENDED

        # Snapshot the conversation id BEFORE we clear it so we can
        # publish a meaningful VoiceSessionEnded event.
        ending_conversation_id = self._conversation_id

        if prev_state != VoiceSessionState.IDLE:
            logger.info(
                "Voice session ended (conversation %s)",
                ending_conversation_id,
            )

        # Clean up
        if self._audio_capture:
            await self._audio_capture.stop()
            self._audio_capture.reset()

        if self._vad:
            self._vad.reset()

        self._cancel_timers()

        self._conversation_id = ""

        # Clear session-scoped identity state
        self._speaker_identities.clear()
        self._display_labels.clear()
        self._latest_speaker_identities.clear()

        # Schedule diarizer unload after warm timeout
        if self._diarizer_loaded and self._diarizer:
            self._diarizer_unload_task = asyncio.create_task(
                self._unload_diarizer_after_timeout(60.0)
            )

        # Set LED pattern
        if self._microphone:
            try:
                await self._microphone.set_led_pattern("off")
            except Exception:
                pass

        # Publish VoiceSessionEnded so downstream systems (enrollment,
        # memory extraction) can flush session-scoped state.
        if prev_state != VoiceSessionState.IDLE and ending_conversation_id:
            try:
                bus = get_event_bus()
                await bus.publish(
                    VoiceSessionEnded(
                        conversation_id=ending_conversation_id,
                    )
                )
            except Exception:
                logger.exception(
                    "Failed to publish VoiceSessionEnded for %s",
                    ending_conversation_id,
                )

        # Return to idle
        self._state = VoiceSessionState.IDLE
        logger.debug("Voice session returned to IDLE")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_wake_word(self, event: WakeWordHeard) -> None:
        """Handle wake word detection."""
        if self._state == VoiceSessionState.IDLE:
            await self._activate_session(event)
        elif self._state == VoiceSessionState.SUSPENDED:
            await self._activate_session(event)
        # If already ACTIVE, ignore (wake word during conversation)

    async def _on_utterance(self, utterance: Utterance) -> None:
        """Process a finalized utterance: run STT + diarization, publish transcript."""
        if self._state != VoiceSessionState.ACTIVE:
            return

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

    async def _on_barge_in(self) -> None:
        """Handle confirmed barge-in during TTS playback."""
        logger.info("Barge-in confirmed, stopping speech")
        bus = get_event_bus()
        await bus.publish(
            AgentSpeakingDone(
                conversation_id=self._conversation_id,
                interrupted=True,
            )
        )
        # Barge-in monitor already stopped playback

    # ------------------------------------------------------------------
    # Speech output
    # ------------------------------------------------------------------

    async def speak(self, text: str, priority: str = "normal") -> None:
        """Speak text through TTS and the speaker.

        Args:
            text: The text to speak.
            priority: "normal" or "urgent". Urgent interrupts current audio.
        """
        if not self._tts_stream or not self._speaker:
            logger.warning("Cannot speak: TTS or speaker not available")
            return

        # If urgent and something is already playing, stop it
        if priority == "urgent" and self._speaker.is_playing:
            await self._speaker.stop_playback()

        # Set LED pattern
        if self._microphone:
            try:
                await self._microphone.set_led_pattern("speaking")
            except Exception:
                pass

        # Publish speaking event
        bus = get_event_bus()
        await bus.publish(
            AgentSpeaking(
                conversation_id=self._conversation_id,
                text=text,
            )
        )

        # Start barge-in monitoring
        if self._barge_in and self._microphone:
            await self._barge_in.start(self._microphone)

        try:
            await self._tts_stream.speak(text)
        except Exception:
            logger.exception("TTS playback failed")
        finally:
            # Stop barge-in monitoring
            if self._barge_in and self._microphone:
                await self._barge_in.remove_from(self._microphone)

            # Publish done event (if not already published by barge-in)
            if not (self._barge_in and not self._barge_in._monitoring):
                await bus.publish(
                    AgentSpeakingDone(
                        conversation_id=self._conversation_id,
                        interrupted=False,
                    )
                )

            # Set LED back to listening if still in active session
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

    def _build_attributed_transcript(
        self,
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
