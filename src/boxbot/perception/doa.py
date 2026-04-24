"""Direction of Arrival (DOA) tracker for ReSpeaker microphone array.

Maps ReSpeaker DOA angles (0-359 degrees) to camera FOV positions and
associates speakers with detected persons by spatial proximity.

Usage:
    from boxbot.perception.doa import DOATracker

    tracker = DOATracker(forward_angle=0, camera_hfov=120)
    camera_x = tracker.angle_to_camera_x(angle=30)
    detection = tracker.associate_speaker_to_detection(angle=30, detections=dets)
"""

from __future__ import annotations

import logging

from boxbot.perception.person_detector import Detection

logger = logging.getLogger(__name__)

# Frame width used for bbox normalization (main stream resolution).
_FRAME_WIDTH = 1280


class DOATracker:
    """Maps ReSpeaker DOA angles to camera FOV positions.

    The ReSpeaker reports direction of arrival as 0-359 degrees. This class
    converts those angles to normalized camera x-coordinates and associates
    speakers with person detections.

    Args:
        forward_angle: ReSpeaker angle (degrees) corresponding to camera center.
        camera_hfov: Camera horizontal field of view in degrees.
    """

    def __init__(
        self,
        forward_angle: float = 0,
        camera_hfov: float = 120,
    ) -> None:
        self._forward_angle = forward_angle
        self._camera_hfov = camera_hfov

    def _angular_difference(self, angle: float) -> float:
        """Compute shortest signed angular difference from forward angle.

        Args:
            angle: DOA angle in degrees (0-359).

        Returns:
            Signed difference in degrees (-180 to 180). Positive = right of center.
        """
        diff = (angle - self._forward_angle) % 360
        if diff > 180:
            diff -= 360
        return diff

    def angle_to_camera_x(self, angle: float) -> float:
        """Map DOA angle to normalized camera x-coordinate.

        Args:
            angle: DOA angle in degrees (0-359).

        Returns:
            Normalized x-coordinate where -1.0 = left edge, 0.0 = center,
            1.0 = right edge. Values outside [-1, 1] indicate out-of-FOV.
        """
        diff = self._angular_difference(angle)
        half_fov = self._camera_hfov / 2.0
        return diff / half_fov

    def is_in_fov(self, angle: float) -> bool:
        """Check whether a DOA angle falls within the camera FOV.

        Args:
            angle: DOA angle in degrees (0-359).

        Returns:
            True if the angle is within +/- half the camera HFOV of forward.
        """
        diff = self._angular_difference(angle)
        half_fov = self._camera_hfov / 2.0
        return abs(diff) <= half_fov

    def associate_speaker_to_detection(
        self,
        angle: float,
        detections: list[Detection],
    ) -> Detection | None:
        """Find the detection closest to a DOA angle.

        Computes the normalized camera x-position for the DOA angle, then
        finds the detection whose bounding box center x (normalized to
        [-1, 1]) is closest. Returns None if the speaker is out of FOV
        or there are no detections.

        Args:
            angle: DOA angle in degrees (0-359).
            detections: List of person detections with bounding boxes.

        Returns:
            The closest Detection, or None if out of FOV or no detections.
        """
        if not detections or not self.is_in_fov(angle):
            return None

        camera_x = self.angle_to_camera_x(angle)
        best_det: Detection | None = None
        best_dist = float("inf")

        for det in detections:
            x1, _y1, x2, _y2 = det.bbox
            # Bbox center x, normalized to [-1, 1] using frame width
            center_px = (x1 + x2) / 2.0
            center_norm = (center_px / _FRAME_WIDTH) * 2.0 - 1.0

            dist = abs(camera_x - center_norm)
            if dist < best_dist:
                best_dist = dist
                best_det = det

        return best_det
