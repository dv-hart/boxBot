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


def _extract_per_class_detections(
    outputs: "object",
) -> list | None:
    """Pull the inner per-class detection list out of Hailo's output dict.

    The actual structure (confirmed via the one-shot diagnostic on
    ``yolov5s_personface_h8l``) is:

        {<single key>: [              # outer list = batch dim, len 1
            [ndarray, ndarray, ...],  # inner list = per-class
        ]}

    Returns the inner per-class list, or None if the structure doesn't
    match (in which case the caller skips this frame; the diagnostic
    dump will already have surfaced the format change).
    """
    if not isinstance(outputs, dict) or not outputs:
        return None
    raw = next(iter(outputs.values()))
    if not isinstance(raw, list) or len(raw) == 0:
        return None
    batch0 = raw[0]
    if not isinstance(batch0, list):
        return None
    return batch0


def _outputs_have_detections(outputs: "object") -> bool:
    """Return True if any nested ndarray has at least one row.

    Walks the same nested-list shape the diagnostic dump describes;
    used to delay the one-shot diagnostic until we have a frame with
    real values to inspect.
    """
    if isinstance(outputs, dict):
        return any(_outputs_have_detections(v) for v in outputs.values())
    if isinstance(outputs, (list, tuple)):
        return any(_outputs_have_detections(v) for v in outputs)
    shape = getattr(outputs, "shape", None)
    if shape is not None and len(shape) >= 1:
        try:
            return int(shape[0]) > 0
        except Exception:
            return False
    return False


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
        # One-shot diagnostic flag: log the actual structure of the
        # Hailo outputs the very first time postprocess() runs, then
        # never again (per-frame logging would flood the log at 5 fps).
        # See ``_log_outputs_diagnostics``.
        self._diagnostics_logged = False

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
        outputs: dict[str, "object"],
        letterbox_params: LetterboxParams,
        original_shape: tuple[int, int],
        frame: np.ndarray | None = None,
    ) -> list[Detection]:
        """Parse YOLOv5s-personface NMS-on-chip output.

        Layout confirmed on-device (HailoRT 4.23, ``yolov5s_personface_h8l``):

            outputs[<single key>] = [
                [                                  # batch dim, len 1
                    np.ndarray(shape=(N0, 5)),     # class 0: person
                    np.ndarray(shape=(N1, 5)),     # class 1: face
                ]
            ]

        Each detection row is ``[y_min, x_min, y_max, x_max, score]`` with
        coordinates **normalized to [0, 1]** in the model's letterboxed
        input frame (640×640). Per-class arrays contain only valid
        detections — no zero-padding to a fixed length.

        Args:
            outputs: Raw model output dict from HAL inference.
            letterbox_params: From preprocess() for coordinate remapping.
            original_shape: (H, W) of original input frame.
            frame: Original frame for crop extraction (optional).

        Returns:
            List of Detection objects (person class only, above threshold).
        """
        # One-shot diagnostic dump — kept after the postprocessor was
        # rewritten so future HEF swaps surface a clear "structure
        # changed" signal in the log instead of silent zero detections.
        if not self._diagnostics_logged:
            self._diagnostics_frames_seen = (
                getattr(self, "_diagnostics_frames_seen", 0) + 1
            )
            has_data = _outputs_have_detections(outputs)
            if has_data or self._diagnostics_frames_seen >= 50:
                self._log_outputs_diagnostics(outputs)
                self._diagnostics_logged = True

        per_class = _extract_per_class_detections(outputs)
        if per_class is None:
            return []

        orig_h, orig_w = original_shape
        scale = letterbox_params.scale
        pad_x = letterbox_params.pad_x
        pad_y = letterbox_params.pad_y
        input_h, input_w = self._input_size

        detections: list[Detection] = []

        for class_id, class_dets in enumerate(per_class):
            if class_dets is None:
                continue
            shape = getattr(class_dets, "shape", None)
            if shape is None or len(shape) != 2 or shape[1] < 5 or shape[0] == 0:
                continue
            for row in class_dets:
                y_min_n = float(row[0])
                x_min_n = float(row[1])
                y_max_n = float(row[2])
                x_max_n = float(row[3])
                confidence = float(row[4])

                if confidence < self._confidence_threshold:
                    continue

                # Normalized → letterboxed pixel space (640×640).
                x1_lb = x_min_n * input_w
                y1_lb = y_min_n * input_h
                x2_lb = x_max_n * input_w
                y2_lb = y_max_n * input_h

                # Letterboxed → original frame.
                x1 = int((x1_lb - pad_x) / scale)
                y1 = int((y1_lb - pad_y) / scale)
                x2 = int((x2_lb - pad_x) / scale)
                y2 = int((y2_lb - pad_y) / scale)

                # Clamp to original image bounds (the model can return
                # boxes that bleed slightly outside [0, 1] for subjects
                # that fill the frame — we saw -0.0149 / 1.0230 in the
                # diagnostic dump).
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = None
                if frame is not None and class_id == self.PERSON_CLASS:
                    crop = self._extract_single_crop(frame, (x1, y1, x2, y2))

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        class_id=class_id,
                        crop=crop,
                    )
                )

        # Filter to person-class only for the return value
        person_detections = [
            d for d in detections if d.class_id == self.PERSON_CLASS
        ]
        return person_detections

    def _log_outputs_diagnostics(
        self, outputs: dict[str, "object"]
    ) -> None:
        """One-shot dump of the Hailo output structure.

        Recurses through nested lists/tuples (HailoRT wraps NMS-on-chip
        outputs as ``[batch][class][detections]``) until it hits an
        ndarray or scalar. For each ndarray we log shape, dtype, and
        the first-row sample so we can read off the field order and
        coordinate system (normalized vs pixel).
        """
        try:
            lines = ["Hailo perception output structure (one-shot):"]
            for key, value in outputs.items():
                lines.append(f"  key={key!r}")
                self._describe_value(value, indent="    ", lines=lines)
            logger.warning("\n".join(lines))
        except Exception:
            logger.exception(
                "Failed to log Hailo outputs diagnostics; postprocess will "
                "still attempt to handle the data."
            )

    def _describe_value(
        self,
        value: "object",
        *,
        indent: str,
        lines: list[str],
        max_depth: int = 6,
    ) -> None:
        """Recursive structural dump of one output value.

        Stops at ``max_depth`` to avoid runaway recursion on
        pathological inputs. At each list/tuple level we log the
        length and recurse into each element. At each ndarray we log
        shape, dtype, and the first-row sample (capped at 8 fields).
        """
        if max_depth <= 0:
            lines.append(f"{indent}<max recursion depth reached>")
            return
        if isinstance(value, (list, tuple)):
            lines.append(
                f"{indent}type={type(value).__name__} len={len(value)}"
            )
            for i, item in enumerate(value):
                lines.append(f"{indent}[{i}]")
                self._describe_value(
                    item,
                    indent=indent + "  ",
                    lines=lines,
                    max_depth=max_depth - 1,
                )
            return
        # ndarray-ish: shape + dtype + sample
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None:
            sample = ""
            try:
                arr = np.asarray(value)
                if arr.size > 0 and arr.ndim >= 1:
                    flat_first = arr[0]
                    if hasattr(flat_first, "tolist"):
                        sample_vals = flat_first.tolist()
                    else:
                        sample_vals = [flat_first]
                    if isinstance(sample_vals, list):
                        sample_vals = sample_vals[:8]
                    sample = f" sample[0]={sample_vals}"
            except Exception:
                pass
            lines.append(
                f"{indent}type={type(value).__name__} "
                f"shape={shape} dtype={dtype}{sample}"
            )
            return
        # Scalar / other
        lines.append(f"{indent}type={type(value).__name__} value={value!r}")

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
