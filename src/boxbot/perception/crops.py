"""Person crop image retention for debugging and auditing.

Saves person detection crops to disk with sidecar JSON metadata.
Supports configurable retention periods with automatic pruning.

Usage:
    from boxbot.perception.crops import CropManager

    manager = CropManager()
    path = manager.save_crop(image, "Person A", "emb-123", "Jacob", 0.92, True)
    manager.prune_expired()
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np

from boxbot.core.paths import PERCEPTION_CROPS_DIR

logger = logging.getLogger(__name__)


class CropManager:
    """Manages person crop image retention with metadata.

    Saves crops organized by date (YYYY-MM-DD directories) with
    sidecar JSON metadata for each image.

    Args:
        base_path: Root directory for crop storage. Defaults to the
            project-root-anchored data/perception/crops.
        retention_days: Days to retain crops in normal mode.
        debug_retention_days: Days to retain crops in debug mode.
    """

    def __init__(
        self,
        base_path: str | Path | None = None,
        retention_days: int = 1,
        debug_retention_days: int = 7,
    ) -> None:
        self._base_path = (
            Path(base_path) if base_path is not None else PERCEPTION_CROPS_DIR
        )
        self._retention_days = retention_days
        self._debug_retention_days = debug_retention_days

    def save_crop(
        self,
        image: np.ndarray,
        ref: str,
        embedding_id: str,
        label: str,
        confidence: float,
        voice_confirmed: bool,
    ) -> str:
        """Save a person crop image with metadata.

        Args:
            image: Crop image as numpy array (H, W, 3) uint8 BGR or RGB.
            ref: Person reference label (e.g., "Person A").
            embedding_id: Associated embedding UUID.
            label: Person name or label.
            confidence: Match confidence score.
            voice_confirmed: Whether the identity was voice-confirmed.

        Returns:
            Path to the saved crop image.
        """
        now = datetime.now(timezone.utc)
        date_dir = self._base_path / now.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        crop_id = str(uuid.uuid4())
        image_path = date_dir / f"{crop_id}.jpg"
        meta_path = date_dir / f"{crop_id}.json"

        # Save image
        cv2.imwrite(str(image_path), image)

        # Save metadata
        metadata = {
            "ref": ref,
            "embedding_id": embedding_id,
            "label": label,
            "confidence": confidence,
            "voice_confirmed": voice_confirmed,
            "timestamp": now.isoformat(),
        }
        meta_path.write_text(json.dumps(metadata, indent=2))

        logger.debug("Saved crop %s for %s (confidence=%.2f)", crop_id, label, confidence)
        return str(image_path)

    def prune_expired(self, debug_mode: bool = False) -> int:
        """Delete crops older than the retention period.

        Args:
            debug_mode: Use extended retention period if True.

        Returns:
            Number of files deleted.
        """
        retention = self._debug_retention_days if debug_mode else self._retention_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention)
        deleted = 0

        if not self._base_path.exists():
            return 0

        for date_dir in sorted(self._base_path.iterdir()):
            if not date_dir.is_dir():
                continue

            # Parse date from directory name
            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue

            if dir_date < cutoff:
                # Delete all files in this directory
                for f in date_dir.iterdir():
                    f.unlink()
                    deleted += 1
                date_dir.rmdir()
                logger.debug("Pruned crop directory: %s (%d files)", date_dir.name, deleted)

        if deleted > 0:
            logger.info("Pruned %d expired crop files", deleted)
        return deleted

    def latest_for_ref(
        self,
        ref: str,
        *,
        max_age_minutes: int = 30,
    ) -> Path | None:
        """Find the most recent crop saved for a given speaker ref.

        Scans today's and yesterday's crop directories for sidecar JSON
        whose ``ref`` matches. Returns the newest image path, or None
        if no crop within the age window is found.

        Used by identify_person to attach the speaker's face to the
        tool result on first-meeting outcomes, so the agent can note
        appearance details into person memory.
        """
        if not self._base_path.exists():
            return None

        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - max_age_minutes * 60

        # Scan today's + yesterday's dirs (crops are tiny, fast to walk)
        date_dirs: list[Path] = []
        for delta in (0, 1):
            d = (now - timedelta(days=delta)).strftime("%Y-%m-%d")
            p = self._base_path / d
            if p.exists():
                date_dirs.append(p)

        best_path: Path | None = None
        best_mtime = 0.0
        for date_dir in date_dirs:
            for meta_path in date_dir.glob("*.json"):
                try:
                    meta = json.loads(meta_path.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if meta.get("ref") != ref:
                    continue
                img_path = meta_path.with_suffix(".jpg")
                if not img_path.exists():
                    continue
                mtime = img_path.stat().st_mtime
                if mtime < cutoff:
                    continue
                if mtime > best_mtime:
                    best_mtime = mtime
                    best_path = img_path
        return best_path

    def get_crop_metadata(self, crop_path: str) -> dict | None:
        """Read sidecar JSON metadata for a crop image.

        Args:
            crop_path: Path to the crop JPEG file.

        Returns:
            Metadata dict, or None if not found.
        """
        meta_path = Path(crop_path).with_suffix(".json")
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read crop metadata: %s", meta_path)
            return None
