"""YOLO pre/post-processing for person detection.

Model-agnostic — the HAL does inference, this module handles preprocessing
input frames and parsing output tensors. Designed for yolov5s-personface
running on Hailo-8L with on-chip NMS.

Usage:
    from boxbot.perception.person_detector import PersonDetector

    detector = PersonDetector()
    preprocessed, params = detector.preprocess(rgb_frame)
    # ... HAL inference ...
    detections = detector.postprocess(outputs, params, original_shape, frame)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single object detection result.

    Attributes:
        bbox: (x1, y1, x2, y2) in original image coordinates.
        confidence: Detection confidence score (0-1).
        class_id: Class index — 0=person, 1=face for yolov5s-personface.
        crop: Person crop resized for ReID input, or None.
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    crop: np.ndarray | None = None


@dataclass
class LetterboxParams:
    """Parameters for remapping letterboxed coordinates back to original space."""

    scale: float
    pad_x: float
    pad_y: float


class PersonDetector:
    """YOLO pre/post-processing for person and face detection.

    Handles letterbox preprocessing, output parsing for on-chip NMS format,
    coordinate remapping, and ReID crop extraction.

    Args:
        input_size: YOLO model input dimensions (H, W).
        confidence_threshold: Minimum confidence for detections.
    """

    PERSON_CLASS = 0  # Person class index (verify on-device)
    FACE_CLASS = 1  # Face class index (verify on-device)
    REID_INPUT_SIZE = (128, 256)  # (H, W) for RepVGG-A0 ReID model

    def __init__(
        self,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.5,
    ) -> None:
        self._input_size = input_size
        self._confidence_threshold = confidence_threshold

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        self._confidence_threshold = value

    def preprocess(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, LetterboxParams]:
        """Letterbox and normalize frame for YOLO input.

        Args:
            frame: (H, W, 3) RGB uint8 frame.

        Returns:
            (preprocessed, letterbox_params)
            preprocessed: (1, H, W, 3) float32 normalized [0, 1]
            letterbox_params: For coordinate remapping in postprocess.
        """
        target_h, target_w = self._input_size
        src_h, src_w = frame.shape[:2]

        # Compute scale to fit within target while maintaining aspect ratio
        scale = min(target_w / src_w, target_h / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Compute padding to center the resized image
        pad_x = (target_w - new_w) / 2.0
        pad_y = (target_h - new_h) / 2.0

        top = int(round(pad_y - 0.1))
        bottom = int(round(pad_y + 0.1))
        left = int(round(pad_x - 0.1))
        right = int(round(pad_x + 0.1))

        # Pad with gray (114) — standard YOLO letterbox fill
        letterboxed = cv2.copyMakeBorder(
            resized,
            top,
            target_h - new_h - top,
            left,
            target_w - new_w - left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # Normalize to [0, 1] float32 and add batch dimension
        preprocessed = letterboxed.astype(np.float32) / 255.0
        preprocessed = np.expand_dims(preprocessed, axis=0)

        params = LetterboxParams(scale=scale, pad_x=pad_x, pad_y=pad_y)
        return preprocessed, params

    def postprocess(
        self,
        outputs: dict[str, np.ndarray],
        letterbox_params: LetterboxParams,
        original_shape: tuple[int, int],
        frame: np.ndarray | None = None,
    ) -> list[Detection]:
        """Parse YOLO NMS output and extract detections.

        The on-chip NMS output shape is (2, 5, 80):
        - Dim 0 (2): class grouping — index 0 = person, index 1 = face
        - Dim 1 (5): values per detection — x1, y1, x2, y2, confidence
          (pixel coordinates in 640x640 letterboxed space)
        - Dim 2 (80): max detections per class

        NOTE: This interpretation needs on-device verification. The exact
        output layout may differ depending on the Hailo model compiler
        configuration.

        Args:
            outputs: Raw model output dict from HAL inference.
            letterbox_params: From preprocess() for coordinate remapping.
            original_shape: (H, W) of original input frame.
            frame: Original frame for crop extraction (optional).

        Returns:
            List of Detection objects (person class only, above threshold).
        """
        # Extract the NMS output tensor — try common output names
        raw = None
        for key in outputs:
            tensor = outputs[key]
            if tensor.shape == (2, 5, 80) or (
                len(tensor.shape) == 3 and tensor.shape[1] == 5
            ):
                raw = tensor
                break

        if raw is None:
            # Fall back to the first output
            raw = next(iter(outputs.values()))
            if raw.ndim != 3:
                logger.warning(
                    "Unexpected YOLO output shape: %s. Expected (2, 5, 80).",
                    raw.shape,
                )
                return []

        orig_h, orig_w = original_shape
        scale = letterbox_params.scale
        pad_x = letterbox_params.pad_x
        pad_y = letterbox_params.pad_y

        detections: list[Detection] = []

        # Iterate over class groups (0=person, 1=face)
        num_classes = raw.shape[0]
        for class_id in range(num_classes):
            class_data = raw[class_id]  # (5, max_detections)

            for det_idx in range(class_data.shape[1]):
                x1_lb = class_data[0, det_idx]
                y1_lb = class_data[1, det_idx]
                x2_lb = class_data[2, det_idx]
                y2_lb = class_data[3, det_idx]
                confidence = class_data[4, det_idx]

                if confidence < self._confidence_threshold:
                    continue

                # Skip zero-area detections (padding in NMS output)
                if x1_lb == 0 and y1_lb == 0 and x2_lb == 0 and y2_lb == 0:
                    continue

                # Remap from letterboxed 640x640 to original image coordinates
                x1 = int((x1_lb - pad_x) / scale)
                y1 = int((y1_lb - pad_y) / scale)
                x2 = int((x2_lb - pad_x) / scale)
                y2 = int((y2_lb - pad_y) / scale)

                # Clamp to original image bounds
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))

                # Skip degenerate boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = None
                if frame is not None and class_id == self.PERSON_CLASS:
                    crop = self._extract_single_crop(frame, (x1, y1, x2, y2))

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(confidence),
                        class_id=class_id,
                        crop=crop,
                    )
                )

        # Filter to person-class only for the return value
        person_detections = [
            d for d in detections if d.class_id == self.PERSON_CLASS
        ]
        return person_detections

    def extract_reid_crops(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> list[np.ndarray]:
        """Extract person crops resized for ReID model input.

        Args:
            frame: Original (H, W, 3) RGB frame.
            detections: Person detections with bbox coordinates.

        Returns:
            List of (1, 128, 256, 3) float32 normalized crops,
            one per detection.
        """
        crops: list[np.ndarray] = []
        for det in detections:
            crop = self._extract_single_crop(frame, det.bbox)
            if crop is not None:
                crops.append(crop)
        return crops

    def _extract_single_crop(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray | None:
        """Extract and resize a single person crop for ReID.

        Args:
            frame: Original (H, W, 3) RGB frame.
            bbox: (x1, y1, x2, y2) bounding box.

        Returns:
            (1, 128, 256, 3) float32 normalized [0, 1] crop, or None if
            the crop is degenerate.
        """
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return None

        # Resize to ReID input: (H=128, W=256) — the model expects 256x128
        # which is (W, H) = (256, 128), so numpy shape is (128, 256, 3)
        reid_h, reid_w = self.REID_INPUT_SIZE
        resized = cv2.resize(cropped, (reid_w, reid_h), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] float32 and add batch dimension
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)
