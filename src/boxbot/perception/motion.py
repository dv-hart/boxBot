"""CPU motion detection using frame differencing.

Runs on 320x240 grayscale frames at 5-10 FPS during DORMANT state.
Uses Gaussian blur to reduce noise before computing the absolute
difference between consecutive frames. Returns a motion score
(mean absolute difference) that the pipeline compares against a
configurable threshold.

Usage:
    from boxbot.perception.motion import MotionDetector

    detector = MotionDetector(threshold=12.0)
    score = detector.detect(grayscale_frame)
    if score > detector.threshold:
        # transition to CHECKING state
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MotionDetector:
    """Frame-differencing motion detector for low-res grayscale input.

    Compares consecutive frames after Gaussian blur to produce a scalar
    motion score. The first frame always returns 0.0 (no reference).

    Args:
        threshold: Motion score above which motion is considered detected.
        blur_kernel: Size of the Gaussian blur kernel (must be odd).
    """

    def __init__(self, threshold: float = 12.0, blur_kernel: int = 21) -> None:
        self._threshold = threshold
        self._blur_kernel = blur_kernel
        self._prev_frame: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> float:
        """Process a grayscale frame and return motion score.

        Args:
            frame: (H, W) uint8 grayscale frame.

        Returns:
            Motion score (mean absolute difference, 0-255 range).
            Compare against threshold to determine motion.
        """
        blurred = cv2.GaussianBlur(
            frame, (self._blur_kernel, self._blur_kernel), 0
        )

        if self._prev_frame is None:
            self._prev_frame = blurred
            return 0.0

        diff = cv2.absdiff(self._prev_frame, blurred)
        score = float(np.mean(diff))

        self._prev_frame = blurred
        return score

    @property
    def threshold(self) -> float:
        """Current motion detection threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    def reset(self) -> None:
        """Clear previous frame (e.g., on state reset)."""
        self._prev_frame = None
