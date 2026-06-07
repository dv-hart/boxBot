"""Visual re-identification using RepVGG-A0 embeddings.

Normalizes raw model output to L2 unit vectors and matches against
known person centroids using cosine similarity. Three confidence tiers:
high (above high_threshold), medium (between thresholds), and unknown
(below low_threshold).

Usage:
    from boxbot.perception.visual_reid import VisualReID, MatchResult

    reid = VisualReID()
    embedding = reid.normalize_embedding(raw_output)
    result = reid.match(embedding, centroids)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of matching an embedding against known centroids.

    Attributes:
        person_id: Matched person's ID, or None if unknown.
        person_name: Matched person's name, or None if unknown.
        confidence: Cosine similarity score of the best match (0-1).
        tier: Confidence tier — "high", "medium", or "unknown".
    """

    person_id: str | None
    person_name: str | None
    confidence: float
    tier: str  # "high", "medium", "unknown"


class VisualReID:
    """Visual re-identification via embedding comparison.

    Normalizes raw model output and matches against known person centroids
    using cosine similarity with configurable thresholds.

    Args:
        high_threshold: Minimum cosine similarity for a "high" confidence match.
        low_threshold: Minimum cosine similarity for a "medium" confidence match.
            Below this, the person is considered unknown.
    """

    HIGH_THRESHOLD = 0.85
    LOW_THRESHOLD = 0.60
    # Cloud-matching tiers (match_cloud), mirroring VoiceReID. confirmed → the
    # "high" tier that can fire triggers; maybe → "medium" (verify/visual).
    CONFIRMED_THRESHOLD = 0.70
    MAYBE_THRESHOLD = 0.55
    CLOUD_TOPK = 3

    def __init__(
        self,
        high_threshold: float = 0.85,
        low_threshold: float = 0.60,
        confirmed_threshold: float | None = None,
        maybe_threshold: float | None = None,
        topk: int | None = None,
    ) -> None:
        self._high_threshold = high_threshold
        self._low_threshold = low_threshold
        self._confirmed = (
            confirmed_threshold if confirmed_threshold is not None
            else self.CONFIRMED_THRESHOLD
        )
        self._maybe = (
            maybe_threshold if maybe_threshold is not None
            else self.MAYBE_THRESHOLD
        )
        self._topk = topk if topk is not None else self.CLOUD_TOPK

    def normalize_embedding(self, raw_output: np.ndarray) -> np.ndarray:
        """Normalize raw model output to unit L2 vector.

        Args:
            raw_output: Raw ReID model output (may be any shape/dtype).

        Returns:
            (512,) float32 L2-normalized embedding.
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
        """Match embedding against known person centroids.

        Args:
            embedding: (512,) float32 normalized embedding.
            centroids: {person_id: (person_name, centroid_vector)} dict.

        Returns:
            MatchResult with best match or unknown.
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
            score = self.cosine_similarity(embedding, centroid)
            if score > best_score:
                best_score = score
                best_id = person_id
                best_name = name

        if best_score >= self._high_threshold:
            tier = "high"
        elif best_score >= self._low_threshold:
            tier = "medium"
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

    def match_cloud(
        self,
        embedding: np.ndarray,
        clouds: dict[str, tuple[str, np.ndarray, np.ndarray]],
        *,
        confirmed_threshold: float | None = None,
        maybe_threshold: float | None = None,
        topk: int | None = None,
    ) -> MatchResult:
        """Match an embedding against per-person visual embedding clouds.

        Mirrors :meth:`VoiceReID.match_cloud`. For each person the score is the
        **provenance-weighted mean of the top-k cosine** similarities to that
        person's cloud — proximity to real stored points (preserving
        per-appearance modes) rather than to a single averaged centroid, with
        higher-provenance points (e.g. hand-confirmed anchors) weighted up.

        Tiers: ``high`` ≥ confirmed_threshold (address by name / fire triggers),
        ``medium`` ≥ maybe_threshold (tentative; verify), else ``unknown``.

        Args:
            embedding: (D,) float32 normalized query embedding.
            clouds: ``{person_id: (name, embeddings (N,D), weights (N,))}``.
            confirmed_threshold / maybe_threshold / topk: overrides.
        """
        confirmed = (
            confirmed_threshold if confirmed_threshold is not None
            else self._confirmed
        )
        maybe = maybe_threshold if maybe_threshold is not None else self._maybe
        k_top = topk if topk is not None else self._topk

        if not clouds:
            return MatchResult(
                person_id=None, person_name=None, confidence=0.0,
                tier="unknown",
            )

        q = embedding.astype(np.float32)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn

        best_id: str | None = None
        best_name: str | None = None
        best_score = -1.0

        for person_id, (name, embs, weights) in clouds.items():
            if embs.size == 0:
                continue
            sims = embs @ q  # cloud already L2-unit → cosine
            k = min(k_top, sims.shape[0])
            # Indices of the top-k most-similar points for this person.
            top_idx = np.argpartition(sims, -k)[-k:]
            top_sims = sims[top_idx]
            top_w = weights[top_idx]
            w_sum = float(top_w.sum())
            if w_sum <= 0:
                score = float(top_sims.mean())
            else:
                score = float((top_sims * top_w).sum() / w_sum)
            if score > best_score:
                best_score = score
                best_id = person_id
                best_name = name

        if best_score >= confirmed:
            tier = "high"
        elif best_score >= maybe:
            tier = "medium"
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

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        If both vectors are L2-normalized, this is simply dot(a, b).
        Falls back to the full formula for safety.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity in [-1, 1].
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def compute_centroid(embeddings: list[np.ndarray]) -> np.ndarray:
        """Compute L2-normalized centroid from a list of embeddings.

        Args:
            embeddings: List of embedding vectors (should be L2-normalized).

        Returns:
            L2-normalized centroid vector.
        """
        mean = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        return mean
