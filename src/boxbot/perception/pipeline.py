"""Perception pipeline orchestrator.

Async loop that ties together motion detection, person detection, visual
ReID, voice-visual fusion, embedding storage, enrollment, and the state
machine. Owns references to Camera and Hailo HAL modules (injected) and
creates all perception components internally.

Publishes MotionDetected, PersonDetected, PersonIdentified, and
SpeakerIdentified events to the internal event bus.

Usage:
    from boxbot.perception.pipeline import PerceptionPipeline

    pipeline = PerceptionPipeline(camera=camera, hailo=hailo)
    await pipeline.start()
    # ...
    await pipeline.stop()

    # For tool/display access:
    from boxbot.perception.pipeline import get_pipeline
    people = get_pipeline().get_present_people()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from boxbot.core.events import (
    ConversationStarted,
    ConversationEnded,
    MotionDetected,
    PersonDetected,
    PersonIdentified,
    SpeakerIdentified,
    TranscriptReady,
    VoiceSessionEnded,
    get_event_bus,
)
from boxbot.perception.clouds import CloudStore
from boxbot.perception.crops import CropManager
from boxbot.perception.doa import DOATracker
from boxbot.perception.enrollment import EnrollmentManager
from boxbot.perception.fusion import FusionResult, IdentityFusion
from boxbot.perception.motion import MotionDetector
from boxbot.perception.person_detector import PersonDetector
from boxbot.perception.state_machine import PerceptionState, PerceptionStateMachine
from boxbot.perception.visual_reid import VisualReID
from boxbot.perception.voice_reid import VoiceReID

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton for tool/display access
# ---------------------------------------------------------------------------

_pipeline_instance: PerceptionPipeline | None = None


def get_pipeline() -> PerceptionPipeline:
    """Return the running pipeline instance.

    Raises RuntimeError if the pipeline has not been started.
    """
    if _pipeline_instance is None:
        raise RuntimeError(
            "Perception pipeline not started. "
            "Call PerceptionPipeline.start() during system startup."
        )
    return _pipeline_instance


# ---------------------------------------------------------------------------
# Ref counter for assigning unique person refs across a session
# ---------------------------------------------------------------------------

_ref_counter: int = 0


def _next_ref() -> str:
    """Generate the next person ref label."""
    global _ref_counter
    idx = _ref_counter
    _ref_counter += 1
    if idx < 26:
        return f"Person {chr(ord('A') + idx)}"
    return f"Person {idx + 1}"


def _reset_ref_counter() -> None:
    global _ref_counter
    _ref_counter = 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PerceptionPipeline:
    """Async perception pipeline orchestrator.

    Ties together all perception components into a single async loop:
    DORMANT (motion detection) → CHECKING (YOLO) → DETECTED (ReID + tracking)
    → CONVERSATION (voice-visual fusion) → POST_CONVERSATION (embedding
    confirmation).

    Args:
        camera: Camera HAL module (must implement get_lores_frame, capture_frame).
        hailo: Hailo HAL module (must implement infer, inference_session).
        cloud_store: Optional CloudStore instance. Created internally if not provided.
        microphone: Optional Microphone HAL module for DOA access.
        motion_threshold: Motion detection threshold (0-255 range).
        reid_high_threshold: High confidence ReID threshold.
        reid_low_threshold: Low confidence ReID threshold.
        presence_timeout: Seconds without detection → DORMANT.
        heartbeat_interval: YOLO re-check interval in DETECTED state.
        scan_fps: Motion detection frame rate.
        voice_match_threshold: Cosine similarity threshold for voice ReID.
        doa_forward_angle: ReSpeaker angle (degrees) that maps to camera center.
        camera_hfov: Camera horizontal field of view in degrees.
        crop_retention_days: Days to retain crop images in normal mode.
        crop_retention_days_debug: Days to retain crop images in debug mode.
    """

    def __init__(
        self,
        camera: Any,
        hailo: Any,
        cloud_store: CloudStore | None = None,
        microphone: Any = None,
        motion_threshold: float = 12.0,
        reid_high_threshold: float = 0.85,
        reid_low_threshold: float = 0.60,
        presence_timeout: float = 30.0,
        heartbeat_interval: float = 5.0,
        scan_fps: int = 5,
        voice_match_threshold: float = 0.60,
        doa_forward_angle: int = 0,
        camera_hfov: int = 120,
        crop_retention_days: int = 1,
        crop_retention_days_debug: int = 7,
    ) -> None:
        self._camera = camera
        self._hailo = hailo
        self._microphone = microphone

        # Cloud store — created lazily if not injected
        self._cloud_store = cloud_store
        self._owns_cloud_store = cloud_store is None

        # Components (created in start)
        self._motion = MotionDetector(threshold=motion_threshold)
        self._detector = PersonDetector()
        self._reid = VisualReID(
            high_threshold=reid_high_threshold,
            low_threshold=reid_low_threshold,
        )
        self._state_machine = PerceptionStateMachine(
            presence_timeout=presence_timeout,
            heartbeat_interval=heartbeat_interval,
        )
        self._enrollment: EnrollmentManager | None = None

        # Voice-visual fusion components
        self._voice_reid = VoiceReID(threshold=voice_match_threshold)
        self._doa_tracker = DOATracker(
            forward_angle=doa_forward_angle, camera_hfov=camera_hfov,
        )
        self._crop_manager = CropManager(
            retention_days=crop_retention_days,
            debug_retention_days=crop_retention_days_debug,
        )
        self._fusion: IdentityFusion | None = None  # created in start()
        self._session_data: dict = {}  # speaker_label -> session data
        self._conversation_active = False

        self._scan_fps = scan_fps
        self._scan_interval = 1.0 / scan_fps

        self._loop_task: asyncio.Task[None] | None = None
        self._running = False

    # ── Lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize perception components and start the async loop."""
        global _pipeline_instance

        # Initialize cloud store
        if self._cloud_store is None:
            self._cloud_store = CloudStore()
            await self._cloud_store.initialize()

        self._enrollment = EnrollmentManager(self._cloud_store)

        # Create identity fusion (needs cloud store + perception config)
        try:
            from boxbot.core.config import get_config
            perception_config = get_config().perception
        except RuntimeError:
            # Config not loaded (e.g. in tests) — use defaults
            from boxbot.core.config import PerceptionConfig
            perception_config = PerceptionConfig()

        self._fusion = IdentityFusion(
            self._cloud_store, self._voice_reid, self._doa_tracker,
            perception_config,
        )

        # Subscribe to conversation lifecycle events
        bus = get_event_bus()
        bus.subscribe(ConversationStarted, self._on_conversation_started)
        bus.subscribe(ConversationEnded, self._on_conversation_ended)
        bus.subscribe(TranscriptReady, self._on_transcript_ready)
        bus.subscribe(VoiceSessionEnded, self._on_voice_session_ended)

        # Start the main perception loop
        self._running = True
        self._loop_task = asyncio.create_task(
            self._run_loop(), name="perception-pipeline"
        )
        _pipeline_instance = self

        logger.info(
            "Perception pipeline started (scan_fps=%d, motion_threshold=%.1f)",
            self._scan_fps,
            self._motion.threshold,
        )

    async def stop(self) -> None:
        """Stop the perception loop and clean up."""
        global _pipeline_instance

        # Unsubscribe from events
        bus = get_event_bus()
        bus.unsubscribe(ConversationStarted, self._on_conversation_started)
        bus.unsubscribe(ConversationEnded, self._on_conversation_ended)
        bus.unsubscribe(TranscriptReady, self._on_transcript_ready)
        bus.unsubscribe(VoiceSessionEnded, self._on_voice_session_ended)

        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

        if self._owns_cloud_store and self._cloud_store is not None:
            await self._cloud_store.close()

        _pipeline_instance = None
        logger.info("Perception pipeline stopped")

    # ── Public API ─────────────────────────────────────────────────

    def get_present_people(self) -> list[dict]:
        """Get currently detected people for display/tool consumption.

        Returns:
            List of dicts with ref, name, confidence, since fields.
        """
        return self._state_machine.get_present_people()

    @property
    def state(self) -> PerceptionState:
        """Current perception state."""
        return self._state_machine.state

    @property
    def enrollment(self) -> EnrollmentManager | None:
        """Enrollment manager for the identify_person tool."""
        return self._enrollment

    @property
    def cloud_store(self) -> CloudStore | None:
        """Cloud store for embedding persistence."""
        return self._cloud_store

    # ── Main loop ──────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Main perception loop — runs until stopped."""
        bus = get_event_bus()

        while self._running:
            try:
                state = self._state_machine.state

                if state == PerceptionState.DORMANT:
                    await self._step_dormant(bus)

                elif state == PerceptionState.CHECKING:
                    await self._step_checking(bus)

                elif state == PerceptionState.DETECTED:
                    await self._step_detected(bus)

                elif state == PerceptionState.CONVERSATION:
                    await self._step_conversation()

                elif state == PerceptionState.POST_CONVERSATION:
                    # Post-conversation processing is handled by the
                    # _on_conversation_ended event handler; just wait.
                    await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in perception loop")
                # Brief pause before retrying to avoid tight error loops
                await asyncio.sleep(1.0)

    async def _step_dormant(self, bus: Any) -> None:
        """DORMANT: run motion detection at scan_fps."""
        try:
            frame = await self._camera.get_lores_frame()
        except Exception:
            logger.debug("Camera lores frame failed, sleeping")
            await asyncio.sleep(self._scan_interval)
            return

        score = self._motion.detect(frame)
        new_state = self._state_machine.on_motion(score, self._motion.threshold)

        if new_state == PerceptionState.CHECKING:
            await bus.publish(MotionDetected(region="full"))

        await asyncio.sleep(self._scan_interval)

    async def _step_checking(self, bus: Any) -> None:
        """CHECKING: run single YOLO frame to confirm person presence."""
        try:
            frame = await self._camera.capture_frame()
        except Exception:
            logger.warning("Camera capture failed in CHECKING, back to DORMANT")
            self._state_machine.reset()
            return

        detections = await self._run_yolo(frame)
        new_state = self._state_machine.on_person_detected(detections)

        if new_state == PerceptionState.DETECTED:
            # Run ReID on detected persons
            await self._run_reid(frame, detections, bus)

    async def _step_detected(self, bus: Any) -> None:
        """DETECTED: periodic heartbeat checks and timeout monitoring."""
        # Check for presence timeout
        new_state = self._state_machine.check_timeout()
        if new_state == PerceptionState.DORMANT:
            self._motion.reset()
            _reset_ref_counter()
            if self._enrollment:
                self._enrollment.clear_session()
            return

        # Periodic YOLO heartbeat
        if self._state_machine.should_heartbeat():
            try:
                frame = await self._camera.capture_frame()
                detections = await self._run_yolo(frame)
                self._state_machine.on_person_detected(detections)
                self._state_machine.record_heartbeat()

                if detections:
                    await self._run_reid(frame, detections, bus)
            except Exception:
                logger.debug("Heartbeat check failed", exc_info=True)

        # Don't spin — sleep until next heartbeat check
        await asyncio.sleep(1.0)

    # ── Inference helpers ──────────────────────────────────────────

    async def _run_yolo(self, frame: np.ndarray) -> list:
        """Run YOLO person detection on a frame.

        Returns list of Detection objects.
        """
        preprocessed, params = self._detector.preprocess(frame)
        outputs = await self._hailo.infer("yolo", preprocessed)
        return self._detector.postprocess(
            outputs, params, frame.shape[:2], frame
        )

    async def _run_reid(
        self, frame: np.ndarray, detections: list, bus: Any
    ) -> None:
        """Run ReID on detected persons and publish events."""
        if not self._cloud_store:
            return

        centroids = await self._cloud_store.get_centroids()

        for i, det in enumerate(detections):
            # Get or generate ref for this detection
            active_persons = self._state_machine.active_persons
            # Find ref for this detection by matching bbox position
            ref = self._find_ref_for_detection(i, active_persons)

            # Publish PersonDetected event
            await bus.publish(
                PersonDetected(
                    person_ref=ref,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    source="visual",
                )
            )

            # Extract ReID crop if not already in detection
            crop = det.crop
            if crop is None:
                crops = self._detector.extract_reid_crops(frame, [det])
                crop = crops[0] if crops else None

            if crop is None:
                continue

            # Run ReID inference
            try:
                reid_output = await self._hailo.infer("reid", crop)
                # Get first output tensor
                raw_embedding = next(iter(reid_output.values()))
                embedding = self._reid.normalize_embedding(raw_embedding)
            except Exception:
                logger.debug("ReID inference failed for %s", ref, exc_info=True)
                continue

            # Match against centroids
            match = self._reid.match(embedding, centroids)
            self._state_machine.on_identification(ref, match, det.bbox)

            # Buffer embedding for enrollment
            if self._enrollment:
                self._enrollment.buffer_embedding(ref, embedding)

            # Publish PersonIdentified for high-confidence matches
            if match.tier == "high" and match.person_id and match.person_name:
                await bus.publish(
                    PersonIdentified(
                        person_id=match.person_id,
                        person_name=match.person_name,
                        confidence=match.confidence,
                        source="visual",
                    )
                )
                # Update last_seen
                await self._cloud_store.update_last_seen(match.person_id)

    async def _step_conversation(self) -> None:
        """CONVERSATION: no YOLO heartbeats, Hailo freed for other use."""
        await asyncio.sleep(1.0)

    # ── Event handlers ────────────────────────────────────────────

    async def _on_conversation_started(self, event: ConversationStarted) -> None:
        """Handle conversation start — freeze heartbeats."""
        self._state_machine.on_conversation_started()
        self._conversation_active = True
        self._session_data.clear()
        logger.info(
            "Perception: CONVERSATION state (conversation %s)",
            event.conversation_id,
        )

    async def _on_conversation_ended(self, event: ConversationEnded) -> None:
        """Handle conversation end — run post-conversation processing."""
        self._state_machine.on_conversation_ended()
        self._conversation_active = False
        await self._run_post_conversation()
        logger.info(
            "Perception: POST_CONVERSATION complete (conversation %s)",
            event.conversation_id,
        )

    async def _on_voice_session_ended(self, event: VoiceSessionEnded) -> None:
        """Voice session fully ended — commit enrollment buffers.

        A single voice session can span many conversations (each wake
        word → transcript → agent turn is its own ConversationStarted/
        Ended), but the enrollment buffers accumulate across all of
        them. We flush once the voice session itself ends, routing each
        ref's buffered voice + visual embeddings to the person their
        session claim points at, or dropping them if no identity was
        ever resolved.
        """
        if self._enrollment is None:
            return
        try:
            summary = await self._enrollment.commit_session()
            logger.info(
                "Enrollment committed on voice session end (%s): %s",
                event.conversation_id, summary,
            )
        except Exception:
            logger.exception(
                "commit_session failed for voice session %s",
                event.conversation_id,
            )

    async def _on_transcript_ready(self, event: TranscriptReady) -> None:
        """Process speaker segments for voice identity fusion."""
        if not self._fusion or not self._cloud_store:
            return

        bus = get_event_bus()

        for seg in event.speaker_segments:
            speaker_label = seg.get("speaker") or seg.get("speaker_label", "")
            # Embedding must be present in the segment for voice matching
            embedding = seg.get("embedding")
            if embedding is None:
                continue

            # Get DOA angle from microphone if available
            doa_angle: int | None = None
            if self._microphone is not None:
                try:
                    doa_angle = self._microphone.get_doa()
                except Exception:
                    pass

            # Run fusion
            result = await self._fusion.fuse_speaker(
                speaker_label=speaker_label,
                speaker_embedding=embedding,
                active_persons=self._state_machine.active_persons,
                doa_angle=doa_angle,
            )

            # Publish SpeakerIdentified for confirmed matches
            if result.voice_confirmed and result.person_id and result.person_name:
                await bus.publish(
                    SpeakerIdentified(
                        speaker_label=speaker_label,
                        person_id=result.person_id,
                        person_name=result.person_name,
                        confidence=result.confidence,
                        source=result.source,
                    )
                )

    # ── Post-conversation ─────────────────────────────────────────

    async def _run_post_conversation(self) -> None:
        """Run post-conversation embedding confirmation."""
        if not self._fusion or not self._cloud_store:
            self._state_machine.on_post_conversation_done(
                people_still_present=bool(self._state_machine.active_persons)
            )
            return

        try:
            # Build session data for fusion confirmation
            session_data: dict = {}
            if hasattr(self._fusion, "session_speakers"):
                for label, speaker in self._fusion.session_speakers.items():
                    session_data[label] = {
                        "fusion_result": speaker.fusion_result
                        if hasattr(speaker, "fusion_result")
                        else speaker,
                        "voice_embeddings": getattr(
                            speaker, "voice_embeddings", []
                        ),
                        "visual_embeddings": [],
                    }

            results = await self._fusion.confirm_session_embeddings(
                session_data, self._cloud_store
            )

            logger.info(
                "Post-conversation: confirmed %d speakers", len(results)
            )
        except Exception:
            logger.exception("Post-conversation embedding confirmation failed")
        finally:
            if hasattr(self._fusion, "clear_session"):
                self._fusion.clear_session()
            self._session_data.clear()

            # Transition out of POST_CONVERSATION
            people_present = bool(self._state_machine.active_persons)
            self._state_machine.on_post_conversation_done(
                people_still_present=people_present
            )

            if not people_present:
                self._motion.reset()
                _reset_ref_counter()
                if self._enrollment:
                    self._enrollment.clear_session()

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _find_ref_for_detection(
        index: int, active_persons: dict
    ) -> str:
        """Find or assign a ref for a detection at the given index.

        Simple strategy: use existing refs by order, or generate new ones.
        """
        refs = list(active_persons.keys())
        if index < len(refs):
            return refs[index]
        # Generate a new ref
        return _next_ref()
