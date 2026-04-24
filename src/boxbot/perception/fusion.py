"""Multi-modal identity fusion — voice + visual + DOA.

Fuses speaker embeddings (voice) with visual ReID and DOA spatial signals
to produce confident person identities. Voice is the authority: it gates
whether visual embeddings are written to the cloud (voice-confirmed rule).

Usage:
    from boxbot.perception.fusion import IdentityFusion

    fusion = IdentityFusion(cloud_store, voice_reid, doa_tracker, config)
    result = await fusion.fuse_speaker(
        speaker_label="SPEAKER_00",
        speaker_embedding=embedding,
        active_persons=state_machine.active_persons,
        doa_angle=45,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from boxbot.core.config import PerceptionConfig
from boxbot.perception.clouds import CloudStore
from boxbot.perception.doa import DOATracker
from boxbot.perception.voice_reid import VoiceReID

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Result of fusing voice, visual, and DOA identity signals.

    Attributes:
        person_id: Matched person's ID, or None if unknown.
        person_name: Matched person's name, or None if unknown.
        confidence: Best match confidence score.
        source: How the identity was determined — "fused", "voice", or "visual".
        voice_confirmed: Whether visual embeddings should be written to cloud.
        speaker_label: Original diarization label (e.g. "SPEAKER_00").
        visual_ref: Perception ref of the associated detection, or None.
    """

    person_id: str | None
    person_name: str | None
    confidence: float
    source: str  # "fused", "voice", "visual"
    voice_confirmed: bool
    speaker_label: str
    visual_ref: str | None


class IdentityFusion:
    """Fuse visual and voice identity signals with DOA spatial association.

    Enforces the voice-gates-vision confirmation rule: visual embeddings are
    only written to a person's cloud when the identity is voice-confirmed.

    Args:
        cloud_store: Embedding cloud storage for centroid lookups.
        voice_reid: Voice re-identification matcher.
        doa_tracker: DOA angle to camera position mapper.
        config: Perception pipeline configuration.
    """

    def __init__(
        self,
        cloud_store: CloudStore,
        voice_reid: VoiceReID,
        doa_tracker: DOATracker,
        config: PerceptionConfig,
    ) -> None:
        self._cloud_store = cloud_store
        self._voice_reid = voice_reid
        self._doa_tracker = doa_tracker
        self._config = config

    async def fuse_speaker(
        self,
        speaker_label: str,
        speaker_embedding: np.ndarray,
        active_persons: dict,
        doa_angle: int | None = None,
    ) -> FusionResult:
        """Fuse voice identity with visual detection and DOA.

        Steps:
        1. Match speaker embedding against voice centroids.
        2. If DOA available, associate speaker with a visual detection.
        3. Check agreement/disagreement between voice and visual signals.

        Args:
            speaker_label: Diarization label (e.g. "SPEAKER_00").
            speaker_embedding: Normalized speaker embedding vector.
            active_persons: {ref: ActivePerson} from the state machine.
            doa_angle: DOA angle in degrees (0-359), or None if unavailable.

        Returns:
            FusionResult with resolved identity and confirmation status.
        """
        # Step 1: Voice matching
        voice_centroids = await self._cloud_store.get_voice_centroids()
        voice_match = self._voice_reid.match(speaker_embedding, voice_centroids)

        # Step 2: DOA-based visual association
        visual_ref: str | None = None
        visual_person_id: str | None = None

        if doa_angle is not None and active_persons:
            # Build detection list from active persons that have bboxes
            from boxbot.perception.person_detector import Detection

            ref_detection_pairs: list[tuple[str, Detection]] = []
            for ref, person in active_persons.items():
                if person.bbox is not None:
                    det = Detection(
                        bbox=person.bbox,
                        confidence=1.0,
                        class_id=0,
                    )
                    ref_detection_pairs.append((ref, det))

            if ref_detection_pairs:
                detections = [det for _, det in ref_detection_pairs]
                best_det = self._doa_tracker.associate_speaker_to_detection(
                    doa_angle, detections
                )
                if best_det is not None:
                    # Find the ref that corresponds to this detection
                    for ref, det in ref_detection_pairs:
                        if det is best_det:
                            visual_ref = ref
                            # Check if this person has a visual match
                            ap = active_persons[ref]
                            if (
                                ap.match_result is not None
                                and ap.match_result.person_id is not None
                            ):
                                visual_person_id = ap.match_result.person_id
                            break

        # Step 3: Fuse signals
        if voice_match.tier == "high":
            # Voice matched — check agreement with visual
            if visual_ref is not None and visual_person_id is not None:
                if visual_person_id == voice_match.person_id:
                    # Voice + visual agree → confirmed fused identity
                    logger.debug(
                        "Fused identity: voice+visual agree on %s (ref=%s)",
                        voice_match.person_name,
                        visual_ref,
                    )
                    return FusionResult(
                        person_id=voice_match.person_id,
                        person_name=voice_match.person_name,
                        confidence=voice_match.confidence,
                        source="fused",
                        voice_confirmed=True,
                        speaker_label=speaker_label,
                        visual_ref=visual_ref,
                    )
                else:
                    # Voice + visual disagree — check DOA
                    if doa_angle is not None and not self._doa_tracker.is_in_fov(
                        doa_angle
                    ):
                        # Speaker is out of FOV — no conflict, trust voice
                        logger.debug(
                            "Voice match %s, visual mismatch but speaker out of FOV",
                            voice_match.person_name,
                        )
                        return FusionResult(
                            person_id=voice_match.person_id,
                            person_name=voice_match.person_name,
                            confidence=voice_match.confidence,
                            source="voice",
                            voice_confirmed=True,
                            speaker_label=speaker_label,
                            visual_ref=None,
                        )
                    else:
                        # In FOV but disagree — trust voice, flag visual as wrong
                        logger.info(
                            "Voice-visual conflict: voice=%s visual=%s, trusting voice",
                            voice_match.person_name,
                            visual_person_id,
                        )
                        return FusionResult(
                            person_id=voice_match.person_id,
                            person_name=voice_match.person_name,
                            confidence=voice_match.confidence,
                            source="voice",
                            voice_confirmed=True,
                            speaker_label=speaker_label,
                            visual_ref=visual_ref,
                        )
            else:
                # Voice match, no visual match (or no visual association)
                logger.debug(
                    "Voice-only identity: %s", voice_match.person_name
                )
                return FusionResult(
                    person_id=voice_match.person_id,
                    person_name=voice_match.person_name,
                    confidence=voice_match.confidence,
                    source="voice",
                    voice_confirmed=True,
                    speaker_label=speaker_label,
                    visual_ref=visual_ref,
                )
        else:
            # No voice match → unknown speaker
            logger.debug(
                "Unknown speaker %s (best voice score=%.2f)",
                speaker_label,
                voice_match.confidence,
            )
            return FusionResult(
                person_id=None,
                person_name=None,
                confidence=voice_match.confidence,
                source="voice",
                voice_confirmed=False,
                speaker_label=speaker_label,
                visual_ref=visual_ref,
            )

    async def confirm_session_embeddings(
        self,
        session_data: dict,
        cloud_store: CloudStore,
    ) -> list[dict]:
        """Write voice-confirmed embeddings to cloud after a conversation.

        Called during POST_CONVERSATION state. For each speaker with
        voice_confirmed=True, writes their visual embeddings to the matched
        person's cloud and adds voice embeddings.

        Args:
            session_data: {speaker_label: {
                "fusion_result": FusionResult,
                "voice_embeddings": list[np.ndarray],
                "visual_embeddings": list[np.ndarray],
            }}
            cloud_store: Embedding cloud storage.

        Returns:
            List of {person_id, visual_added, voice_added} dicts.
        """
        results: list[dict] = []

        for speaker_label, data in session_data.items():
            fusion_result: FusionResult = data["fusion_result"]
            voice_embeddings: list[np.ndarray] = data.get("voice_embeddings", [])
            visual_embeddings: list[np.ndarray] = data.get("visual_embeddings", [])

            if not fusion_result.voice_confirmed or fusion_result.person_id is None:
                continue

            person_id = fusion_result.person_id
            visual_added = 0
            voice_added = 0

            # Write visual embeddings (voice-confirmed)
            for emb in visual_embeddings:
                await cloud_store.add_visual_embedding(
                    person_id, emb, voice_confirmed=True
                )
                visual_added += 1

            # Write voice embeddings
            for emb in voice_embeddings:
                await cloud_store.add_voice_embedding(person_id, emb)
                voice_added += 1

            # Recompute centroids if we added anything
            if visual_added > 0:
                await cloud_store.recompute_centroid(person_id)
            if voice_added > 0:
                await cloud_store.recompute_voice_centroid(person_id)

            logger.info(
                "Confirmed embeddings for %s: %d visual, %d voice",
                fusion_result.person_name,
                visual_added,
                voice_added,
            )
            results.append({
                "person_id": person_id,
                "visual_added": visual_added,
                "voice_added": voice_added,
            })

        return results
