"""Perception pipeline state machine.

Manages transitions between DORMANT, CHECKING, and DETECTED states
based on motion detection and person detection results. Tracks active
persons and handles presence timeouts and heartbeat scheduling.

The CONVERSATION state is deferred to the voice pipeline phase.

Usage:
    from boxbot.perception.state_machine import PerceptionStateMachine

    sm = PerceptionStateMachine()
    new_state = sm.on_motion(score=15.0, threshold=12.0)
    new_state = sm.on_person_detected(detections)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from boxbot.perception.person_detector import Detection
from boxbot.perception.visual_reid import MatchResult

logger = logging.getLogger(__name__)


class PerceptionState(Enum):
    """States for the perception pipeline."""

    DORMANT = "dormant"
    CHECKING = "checking"
    DETECTED = "detected"
    CONVERSATION = "conversation"
    POST_CONVERSATION = "post_conversation"


@dataclass
class ActivePerson:
    """A person currently tracked by the perception pipeline.

    Attributes:
        ref: Temporary or identified label (e.g., "Person A" or "Jacob").
        match_result: ReID match result, or None if not yet matched.
        last_detected: When this person was last seen in a detection frame.
        bbox: Most recent bounding box, or None.
    """

    ref: str
    match_result: MatchResult | None
    last_detected: datetime
    bbox: tuple[int, int, int, int] | None = None


class PerceptionStateMachine:
    """State machine for the perception pipeline.

    Manages transitions between DORMANT, CHECKING, and DETECTED states.
    Tracks active persons and provides presence timeout and heartbeat
    scheduling.

    Args:
        presence_timeout: Seconds without detection before returning to
            DORMANT from DETECTED.
        heartbeat_interval: Seconds between periodic YOLO re-checks in
            DETECTED state.
    """

    def __init__(
        self,
        presence_timeout: float = 30.0,
        heartbeat_interval: float = 5.0,
    ) -> None:
        self._state = PerceptionState.DORMANT
        self._presence_timeout = presence_timeout
        self._heartbeat_interval = heartbeat_interval
        self._active_persons: dict[str, ActivePerson] = {}
        self._last_detection_time: datetime | None = None
        self._last_heartbeat_time: datetime | None = None

    @property
    def state(self) -> PerceptionState:
        """Current perception state."""
        return self._state

    @property
    def active_persons(self) -> dict[str, ActivePerson]:
        """Currently tracked persons."""
        return self._active_persons

    def on_motion(self, score: float, threshold: float) -> PerceptionState:
        """Handle motion detection result.

        Transitions DORMANT -> CHECKING when motion exceeds threshold.

        Args:
            score: Motion score from MotionDetector.
            threshold: Motion threshold to compare against.

        Returns:
            New perception state.
        """
        if self._state == PerceptionState.DORMANT and score > threshold:
            self._state = PerceptionState.CHECKING
            logger.debug("DORMANT -> CHECKING (motion=%.1f > %.1f)", score, threshold)

        return self._state

    def on_person_detected(
        self, detections: list[Detection]
    ) -> PerceptionState:
        """Handle YOLO detection results.

        CHECKING + persons found -> DETECTED
        CHECKING + no persons -> DORMANT
        DETECTED + persons found -> stays DETECTED (updates tracking)

        Args:
            detections: Person detections from PersonDetector.

        Returns:
            New perception state.
        """
        now = datetime.now(timezone.utc)

        if not detections:
            if self._state == PerceptionState.CHECKING:
                self._state = PerceptionState.DORMANT
                logger.debug("CHECKING -> DORMANT (no persons)")
            return self._state

        # Persons found
        self._last_detection_time = now
        self._last_heartbeat_time = now

        if self._state == PerceptionState.CHECKING:
            self._state = PerceptionState.DETECTED
            logger.debug(
                "CHECKING -> DETECTED (%d person(s))", len(detections)
            )

        # Update or create active person entries
        # Assign refs based on detection order (simple strategy for now)
        existing_refs = set(self._active_persons.keys())
        used_refs: set[str] = set()

        for i, det in enumerate(detections):
            ref = self._assign_ref(i, existing_refs, used_refs)
            used_refs.add(ref)

            if ref in self._active_persons:
                self._active_persons[ref].last_detected = now
                self._active_persons[ref].bbox = det.bbox
            else:
                self._active_persons[ref] = ActivePerson(
                    ref=ref,
                    match_result=None,
                    last_detected=now,
                    bbox=det.bbox,
                )

        return self._state

    def on_identification(
        self,
        ref: str,
        match: MatchResult,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Record identification result for a detected person.

        Args:
            ref: The person ref to update.
            match: ReID match result.
            bbox: Optional updated bounding box.
        """
        if ref in self._active_persons:
            self._active_persons[ref].match_result = match
            if bbox is not None:
                self._active_persons[ref].bbox = bbox
        else:
            # Create entry if not already tracked
            self._active_persons[ref] = ActivePerson(
                ref=ref,
                match_result=match,
                last_detected=datetime.now(timezone.utc),
                bbox=bbox,
            )

    def on_conversation_started(self) -> PerceptionState:
        """Handle conversation start — freeze heartbeats, free Hailo.

        Transitions DETECTED -> CONVERSATION. No-op if not in DETECTED.

        Returns:
            New perception state.
        """
        if self._state == PerceptionState.DETECTED:
            self._state = PerceptionState.CONVERSATION
            logger.debug("DETECTED -> CONVERSATION")
        return self._state

    def on_conversation_ended(self) -> PerceptionState:
        """Handle conversation end — enter post-conversation processing.

        Transitions CONVERSATION -> POST_CONVERSATION.

        Returns:
            New perception state.
        """
        if self._state == PerceptionState.CONVERSATION:
            self._state = PerceptionState.POST_CONVERSATION
            logger.debug("CONVERSATION -> POST_CONVERSATION")
        return self._state

    def on_post_conversation_done(
        self, people_still_present: bool = False
    ) -> PerceptionState:
        """Handle post-conversation processing complete.

        Transitions POST_CONVERSATION -> DETECTED (if people still present)
        or POST_CONVERSATION -> DORMANT.

        Args:
            people_still_present: Whether people are still detected.

        Returns:
            New perception state.
        """
        if self._state == PerceptionState.POST_CONVERSATION:
            if people_still_present:
                self._state = PerceptionState.DETECTED
                self._last_detection_time = datetime.now(timezone.utc)
                self._last_heartbeat_time = datetime.now(timezone.utc)
                logger.debug("POST_CONVERSATION -> DETECTED (people present)")
            else:
                self._state = PerceptionState.DORMANT
                self._active_persons.clear()
                self._last_detection_time = None
                self._last_heartbeat_time = None
                logger.debug("POST_CONVERSATION -> DORMANT")
        return self._state

    def should_heartbeat(self) -> bool:
        """Check if it's time for a periodic YOLO re-check in DETECTED state.

        Returns:
            True if in DETECTED state and heartbeat_interval has elapsed.
        """
        if self._state != PerceptionState.DETECTED:
            return False

        if self._last_heartbeat_time is None:
            return True

        elapsed = (
            datetime.now(timezone.utc) - self._last_heartbeat_time
        ).total_seconds()
        return elapsed >= self._heartbeat_interval

    def record_heartbeat(self) -> None:
        """Record that a heartbeat check was performed."""
        self._last_heartbeat_time = datetime.now(timezone.utc)

    def check_timeout(self) -> PerceptionState:
        """Check if presence has timed out.

        If in DETECTED state and no detection has occurred within
        presence_timeout seconds, transitions back to DORMANT.
        Timeout only applies in DETECTED state — CONVERSATION and
        POST_CONVERSATION have their own lifecycle.

        Returns:
            New perception state.
        """
        if self._state != PerceptionState.DETECTED:
            return self._state

        if self._last_detection_time is None:
            return self._state

        elapsed = (
            datetime.now(timezone.utc) - self._last_detection_time
        ).total_seconds()

        if elapsed >= self._presence_timeout:
            self._state = PerceptionState.DORMANT
            self._active_persons.clear()
            self._last_detection_time = None
            self._last_heartbeat_time = None
            logger.debug(
                "DETECTED -> DORMANT (timeout after %.1fs)", elapsed
            )

        return self._state

    def get_present_people(self) -> list[dict]:
        """Get currently present people for display/tool consumption.

        Returns:
            List of dicts with ref, name, confidence, and since fields.
        """
        people: list[dict] = []
        for ref, person in self._active_persons.items():
            entry: dict = {
                "ref": ref,
                "name": None,
                "confidence": 0.0,
                "since": person.last_detected.isoformat(),
            }
            if person.match_result is not None:
                entry["name"] = person.match_result.person_name
                entry["confidence"] = person.match_result.confidence
            people.append(entry)
        return people

    def reset(self) -> None:
        """Reset to DORMANT state, clearing all tracking."""
        self._state = PerceptionState.DORMANT
        self._active_persons.clear()
        self._last_detection_time = None
        self._last_heartbeat_time = None

    @staticmethod
    def _assign_ref(
        index: int,
        existing_refs: set[str],
        used_refs: set[str],
    ) -> str:
        """Assign a temporary ref for a detection.

        Uses "Person A", "Person B", etc. naming scheme.

        Args:
            index: Detection index.
            existing_refs: Already-assigned refs from previous frames.
            used_refs: Refs assigned so far in this frame.

        Returns:
            A ref string.
        """
        # Simple alphabetical assignment
        letter = chr(ord("A") + index)
        ref = f"Person {letter}"

        # If we overflow past Z, use numbers
        if index >= 26:
            ref = f"Person {index + 1}"

        return ref
