"""Voice re-identification using speaker embeddings.

Matches pyannote speaker embeddings against stored voice centroids using
cosine similarity, with four confidence tiers (``high``, ``medium``,
``low``, ``unknown``). The agent sees the tier + the raw score in its
dynamic context and uses them to decide whether to address the speaker
by name, verify softly, or load the onboarding skill.

Default thresholds (tunable from config in future):

- ``high``     ≥ 0.85    → confident enough to address by name
- ``medium``   0.70-0.85 → probably them; soft-verify before committing
- ``low``      0.60-0.70 → weakly reminds of someone; don't assume
- ``unknown``  < 0.60    → no useful signal; treat as stranger

The medium/low bands give the agent room to verify rather than reinforce
a bad match. Voice tiers are intentionally more generous than the raw
similarity numbers because multiple utterances accumulate confidence
during a session and we'd rather verify-then-commit than ignore-then-drop.

Usage:
    voice_reid = VoiceReID()
    result = voice_reid.match(embedding, centroids)
    # result.tier ∈ {"high","medium","low","unknown"}
    # result.confidence is the raw cosine similarity (-1..1)
"""

from __future__ import annotations

import logging

import numpy as np

from boxbot.perception.visual_reid import MatchResult, VisualReID

logger = logging.getLogger(__name__)


class VoiceReID:
    """Voice re-identification via speaker embedding comparison.

    Matches speaker embeddings against known voice centroids using cosine
    similarity. Reuses ``MatchResult`` from :mod:`visual_reid` for
    symmetry; tier strings are the superset {high, medium, low, unknown}.

    Args:
        high_threshold: Minimum score for a ``"high"`` match.
        medium_threshold: Minimum score for a ``"medium"`` match.
        low_threshold: Minimum score for a ``"low"`` match. Below this
            the match is reported as ``"unknown"``.
    """

    HIGH_THRESHOLD = 0.85
    MEDIUM_THRESHOLD = 0.70
    LOW_THRESHOLD = 0.60

    def __init__(
        self,
        *,
        high_threshold: float | None = None,
        medium_threshold: float | None = None,
        low_threshold: float | None = None,
        # Legacy positional param — old callers passed a single threshold.
        threshold: float | None = None,
    ) -> None:
        if threshold is not None and low_threshold is None:
            low_threshold = threshold
        self._high = high_threshold if high_threshold is not None else self.HIGH_THRESHOLD
        self._medium = medium_threshold if medium_threshold is not None else self.MEDIUM_THRESHOLD
        self._low = low_threshold if low_threshold is not None else self.LOW_THRESHOLD

    def normalize_embedding(self, raw_output: np.ndarray) -> np.ndarray:
        """Normalize raw speaker embedding to unit L2 vector.

        Args:
            raw_output: Raw embedding from pyannote (typically 192-dim).

        Returns:
            L2-normalized float32 embedding.
        """
        vec = raw_output.astype(np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def match(
        self,
        embedding: np.ndarray,
        centroids: dict[str, tuple[str, np.ndarray]],
    ) -> MatchResult:
        """Match speaker embedding against known voice centroids.

        Returns the best-scoring centroid with a tier. When the score is
        below ``low_threshold`` the tier is ``"unknown"`` and person_id /
        person_name are cleared — we don't expose speculative identities
        below the floor. Above the floor, person_id / person_name are set
        even for ``"low"``/``"medium"`` tiers so the agent can verify.
        """
        if not centroids:
            return MatchResult(
                person_id=None,
                person_name=None,
                confidence=0.0,
                tier="unknown",
            )

        best_id: str | None = None
        best_name: str | None = None
        best_score = -1.0

        for person_id, (name, centroid) in centroids.items():
            score = VisualReID.cosine_similarity(embedding, centroid)
            if score > best_score:
                best_score = score
                best_id = person_id
                best_name = name

        if best_score >= self._high:
            tier = "high"
        elif best_score >= self._medium:
            tier = "medium"
        elif best_score >= self._low:
            tier = "low"
        else:
            tier = "unknown"
            best_id = None
            best_name = None

        return MatchResult(
            person_id=best_id,
            person_name=best_name,
            confidence=best_score,
            tier=tier,
        )
