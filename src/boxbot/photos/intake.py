"""Async photo intake pipeline.

Processes incoming photos when the system is idle:
  1. Resize to configurable max resolution (default 1920x1080)
  2. Person detection via YOLO + ReID (stub — Hailo integration)
  3. Small model tagging (stub — description + tags)
  4. Embed description with MiniLM
  5. Store image + metadata in SQLite

Photos are queued and processed sequentially. The pipeline is fault-tolerant:
a failure on one photo does not block the rest. Live perception always
preempts photo intake via a shared Hailo semaphore.

Usage:
    from boxbot.photos.intake import IntakePipeline

    pipeline = IntakePipeline(store)
    await pipeline.start()

    photo_id = await pipeline.enqueue(file_path, source="whatsapp", sender="Jacob")
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from boxbot.core.config import get_config
from boxbot.memory.embeddings import embed
from boxbot.photos.store import PhotoStore

logger = logging.getLogger(__name__)


# Module-level accessor so sandbox action handlers (bb.photos.ingest)
# can reach the running pipeline without DI plumbing. Mirrors the
# display manager / WhatsApp client singleton pattern.
_intake_pipeline_instance: "IntakePipeline | None" = None


def get_intake_pipeline() -> "IntakePipeline | None":
    """Return the process-wide IntakePipeline instance, or None if unset."""
    return _intake_pipeline_instance


def set_intake_pipeline(pipeline: "IntakePipeline | None") -> None:
    """Register the process-wide IntakePipeline instance."""
    global _intake_pipeline_instance
    _intake_pipeline_instance = pipeline


# How long staged inbound files (e.g. tmp/inbound/whatsapp/…) may sit
# before the janitor reaps them. Files the agent actively ingested are
# deleted right after the pipeline copies them; this only catches the
# ones the agent looked at and decided not to keep.
_INBOUND_TTL_SECONDS = 7 * 24 * 3600
_INBOUND_SWEEP_INTERVAL = 6 * 3600


@dataclass
class IntakeItem:
    """An item in the intake queue awaiting processing."""

    file_path: Path
    source: str  # "whatsapp", "camera", "upload"
    sender: str | None
    future: asyncio.Future[str]
    caption: str | None = None
    delete_source: bool = False


@dataclass
class IntakeResult:
    """Result from processing a single photo through the intake pipeline."""

    photo_id: str
    filename: str
    description: str
    tags: list[str]
    people: list[dict[str, Any]]
    width: int
    height: int
    orientation: str
    file_size: int


class IntakePipeline:
    """Async pipeline for processing incoming photos.

    Photos are enqueued and processed sequentially in the background.
    The pipeline respects system idle state and Hailo NPU contention.
    """

    def __init__(self, store: PhotoStore) -> None:
        self._store = store
        self._queue: asyncio.Queue[IntakeItem] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._janitor_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background processing loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        self._janitor_task = asyncio.create_task(
            self._inbound_janitor_loop(),
            name="photo-intake-janitor",
        )
        logger.info("Photo intake pipeline started")

    async def stop(self) -> None:
        """Stop the background processing loop gracefully."""
        self._running = False
        if self._janitor_task is not None:
            self._janitor_task.cancel()
            try:
                await self._janitor_task
            except (asyncio.CancelledError, Exception):
                pass
            self._janitor_task = None
        if self._task:
            # Put a sentinel to unblock the queue.get()
            sentinel_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
            sentinel_future.set_result("")
            self._queue.put_nowait(
                IntakeItem(
                    file_path=Path("/dev/null"),
                    source="__stop__",
                    sender=None,
                    future=sentinel_future,
                )
            )
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
        logger.info("Photo intake pipeline stopped")

    async def _inbound_janitor_loop(self) -> None:
        """Periodically prune stale files under tmp/inbound/.

        Files the agent ingests are deleted by the pipeline as soon as
        the bytes are copied into the photo store. This loop catches
        the rest — inbound images the agent looked at and decided not
        to keep — and prevents the staging dir from growing without
        bound. Runs every ``_INBOUND_SWEEP_INTERVAL`` seconds.
        """
        # First sweep at startup catches anything left over from a
        # previous run that crashed mid-ingest.
        while self._running:
            try:
                await self._sweep_inbound_dir()
            except Exception:
                logger.exception("Inbound janitor sweep failed")
            try:
                await asyncio.sleep(_INBOUND_SWEEP_INTERVAL)
            except asyncio.CancelledError:
                return

    async def _sweep_inbound_dir(self) -> None:
        """Delete inbound staging files older than ``_INBOUND_TTL_SECONDS``."""
        try:
            tmp_dir = Path(get_config().sandbox.tmp_dir)
        except Exception:
            tmp_dir = Path("/var/lib/boxbot-sandbox/tmp")
        inbound_root = tmp_dir / "inbound"
        if not inbound_root.exists():
            return

        cutoff = datetime.now(timezone.utc).timestamp() - _INBOUND_TTL_SECONDS
        reaped = 0
        for path in inbound_root.rglob("*"):
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    reaped += 1
            except OSError as e:
                logger.debug("Janitor skip %s: %s", path, e)
        if reaped:
            logger.info("Inbound janitor: reaped %d stale files", reaped)

    async def enqueue(
        self,
        file_path: str | Path,
        *,
        source: str = "upload",
        sender: str | None = None,
        caption: str | None = None,
        delete_source: bool = False,
    ) -> str:
        """Add a photo to the intake queue for processing.

        Args:
            file_path: Path to the image file to process.
            source: Origin — "whatsapp", "camera", or "upload".
            sender: Person name if from messaging.
            caption: Optional human-supplied description (e.g. WhatsApp
                caption). Seeds the photo description so search works
                even before the small-model tagger fills it in.
            delete_source: If True, delete ``file_path`` after the
                pipeline has copied the bytes into the photo store. Used
                for staged inbound files (tmp/inbound/whatsapp/…) so we
                don't leave duplicates lying around.

        Returns:
            The photo ID once processing is complete.

        Raises:
            RuntimeError: If the pipeline is not running.
            FileNotFoundError: If the file does not exist.
            ValueError: If storage quota would be exceeded.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Photo file not found: {file_path}")

        # Check storage quota before queueing
        storage_info = await self._store.get_storage_info()
        if storage_info.used_percent >= 98.0:
            raise ValueError(
                f"Photo storage is at {storage_info.used_percent}% of quota "
                f"({storage_info.used_bytes / (1024**3):.1f} GB / "
                f"{storage_info.quota_bytes / (1024**3):.1f} GB). "
                f"Cannot save new photos. Ask the user what to remove."
            )

        loop = asyncio.get_event_loop()
        future: asyncio.Future[str] = loop.create_future()

        item = IntakeItem(
            file_path=file_path,
            source=source,
            sender=sender,
            future=future,
            caption=caption,
            delete_source=delete_source,
        )
        await self._queue.put(item)
        logger.debug("Enqueued photo for intake: %s", file_path)

        return await future

    async def process_photo(
        self,
        file_path: str | Path,
        *,
        source: str = "upload",
        sender: str | None = None,
        caption: str | None = None,
    ) -> str:
        """Process a photo synchronously (bypasses the queue).

        Useful for direct processing without the background loop.

        Args:
            file_path: Path to the image file to process.
            source: Origin — "whatsapp", "camera", or "upload".
            sender: Person name if from messaging.
            caption: Optional human-supplied description.

        Returns:
            The photo ID.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Photo file not found: {file_path}")

        # Check storage quota
        storage_info = await self._store.get_storage_info()
        if storage_info.used_percent >= 98.0:
            raise ValueError(
                f"Photo storage is at {storage_info.used_percent}% of quota "
                f"({storage_info.used_bytes / (1024**3):.1f} GB / "
                f"{storage_info.quota_bytes / (1024**3):.1f} GB). "
                f"Cannot save new photos. Ask the user what to remove."
            )

        result = await self._run_pipeline(file_path, source, sender, caption=caption)
        return result.photo_id

    # ------------------------------------------------------------------
    # Background processing
    # ------------------------------------------------------------------

    async def _process_loop(self) -> None:
        """Main processing loop — pulls items from the queue."""
        while self._running:
            try:
                item = await self._queue.get()

                # Check for stop sentinel
                if item.source == "__stop__":
                    break

                try:
                    result = await self._run_pipeline(
                        item.file_path, item.source, item.sender,
                        caption=item.caption,
                    )
                    if not item.future.done():
                        item.future.set_result(result.photo_id)
                    if item.delete_source:
                        try:
                            item.file_path.unlink(missing_ok=True)
                        except OSError as e:
                            logger.warning(
                                "Could not delete intake source %s: %s",
                                item.file_path, e,
                            )
                except Exception as e:
                    logger.exception("Intake pipeline failed for %s", item.file_path)
                    if not item.future.done():
                        item.future.set_exception(e)
                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Unexpected error in intake loop")

    async def _run_pipeline(
        self,
        file_path: Path,
        source: str,
        sender: str | None,
        *,
        caption: str | None = None,
    ) -> IntakeResult:
        """Execute the full intake pipeline on a single photo.

        Steps:
          1. Resize image
          2. Person detection (stub)
          3. Small model tagging (stub)
          4. Embed description
          5. Store to disk and database
        """
        config = get_config()

        # Step 1: Resize and get image info
        max_res = config.photos.max_image_resolution
        resized_image, width, height, orientation = await self._resize_image(
            file_path, max_width=max_res[0], max_height=max_res[1]
        )

        # Step 2: Person detection (stub — will integrate with Hailo)
        people = await self._detect_people(resized_image)

        # Step 3: Small model tagging (stub — will call Claude small model)
        description, tags, new_tags = await self._generate_tags(
            resized_image, people
        )

        # Caption fallback: if the small-model tagger gave us nothing
        # (current behaviour — it's a stub), use the human-supplied
        # caption so search at least has something to match against.
        # Once the tagger lands, the caption can be passed in as a hint.
        if not description and caption:
            description = caption

        # Step 4: Embed the description
        description_embedding: np.ndarray | None = None
        if description:
            description_embedding = embed(description)

        # Step 5: Store the image file
        storage_path = Path(config.photos.storage_path)
        now = datetime.now(timezone.utc)
        relative_dir = f"{now.year}/{now.month:02d}"
        dest_dir = storage_path / relative_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        file_id = uuid4().hex[:12]
        dest_filename = f"{file_id}.jpg"
        dest_path = dest_dir / dest_filename
        relative_path = f"{relative_dir}/{dest_filename}"

        # Write the resized image
        await asyncio.to_thread(self._save_image, resized_image, dest_path)

        file_size = dest_path.stat().st_size

        # Step 6: Store metadata in database
        photo_id = await self._store.create_photo(
            filename=relative_path,
            source=source,
            sender=sender,
            description=description,
            orientation=orientation,
            width=width,
            height=height,
            file_size=file_size,
            in_slideshow=True,
            tags=tags,
            people=people,
            embedding=description_embedding,
        )

        logger.info(
            "Processed photo %s: %dx%d %s, %d tags, %d people",
            photo_id, width, height, orientation, len(tags), len(people),
        )

        return IntakeResult(
            photo_id=photo_id,
            filename=relative_path,
            description=description,
            tags=tags,
            people=people,
            width=width,
            height=height,
            orientation=orientation,
            file_size=file_size,
        )

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    async def _resize_image(
        self,
        file_path: Path,
        *,
        max_width: int = 1920,
        max_height: int = 1080,
    ) -> tuple[Any, int, int, str]:
        """Resize an image to fit within max dimensions, preserving aspect ratio.

        Returns:
            Tuple of (PIL Image, width, height, orientation).
        """
        from PIL import Image

        def _do_resize() -> tuple[Any, int, int, str]:
            img = Image.open(file_path)

            # Auto-rotate based on EXIF orientation
            try:
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass

            # Convert to RGB if needed (handles RGBA, palette, etc.)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            orig_w, orig_h = img.size

            # Only downscale, never upscale
            if orig_w > max_width or orig_h > max_height:
                ratio = min(max_width / orig_w, max_height / orig_h)
                new_w = int(orig_w * ratio)
                new_h = int(orig_h * ratio)
                img = img.resize((new_w, new_h), Image.LANCZOS)

            w, h = img.size

            if w > h:
                orientation = "landscape"
            elif h > w:
                orientation = "portrait"
            else:
                orientation = "square"

            return img, w, h, orientation

        return await asyncio.to_thread(_do_resize)

    async def _detect_people(
        self, image: Any
    ) -> list[dict[str, Any]]:
        """Detect and identify people in the image.

        STUB: Will integrate with Hailo NPU for YOLO detection + OSNet ReID.
        Currently returns an empty list.

        When implemented:
          1. Run YOLOv8n on Hailo to detect person bounding boxes
          2. For each detection, extract OSNet embedding
          3. Match against known person clouds from perception system
          4. Return list of {label, person_id, bbox_x, bbox_y, bbox_w, bbox_h}

        Hailo contention: This step yields to live perception via a shared
        priority semaphore. If a person is detected by the live pipeline
        during photo processing, intake pauses until the NPU is available.
        """
        # TODO: Integrate with boxbot.hardware.hailo for person detection
        # TODO: Integrate with perception system for ReID matching
        return []

    async def _generate_tags(
        self, image: Any, people: list[dict[str, Any]]
    ) -> tuple[str, list[str], list[str]]:
        """Generate description and tags using the small model.

        STUB: Will call the small Claude model with the image, person
        annotations, and existing tag library.

        Returns:
            Tuple of (description, all_tags, new_tags_created).

        When implemented, the small model receives:
          - The image
          - Person annotations: ["Jacob (left)", "unknown person (center)"]
          - The existing tag library
        And returns structured output:
          {"description": "...", "tags": [...], "new_tags": [...]}
        """
        # TODO: Call small model for description + tagging
        # For now, return empty description and no tags
        description = ""
        tags: list[str] = []
        new_tags: list[str] = []
        return description, tags, new_tags

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_image(image: Any, dest_path: Path) -> None:
        """Save a PIL Image to disk as JPEG."""
        image.save(
            str(dest_path),
            format="JPEG",
            quality=85,
            optimize=True,
        )
