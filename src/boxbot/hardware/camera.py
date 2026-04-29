"""Pi Camera Module 3 Wide NoIR interface via picamera2.

Provides dual-stream video (low-res for motion detection, main for
perception) and full-resolution still capture for the photo library.
All picamera2 calls are wrapped in ``run_in_executor`` since the library
is blocking.

Hardware: Pi Camera Module 3 Wide NoIR (IMX708, 120 deg FOV, 12MP)
Interface: CSI-2 ribbon cable
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from boxbot.hardware.base import (
    HardwareModule,
    HardwareUnavailableError,
    HealthStatus,
)

logger = logging.getLogger(__name__)


# Module-level accessor so sandbox action handlers (and anything else
# outside the startup closure) can reach the shared Camera instance
# without a full DI framework. Set once by ``boxbot.core.main``.
_camera_instance: "Camera | None" = None


def get_camera() -> "Camera | None":
    """Return the running Camera, or None if the HAL is not up yet."""
    return _camera_instance


def set_camera(cam: "Camera | None") -> None:
    """Publish (or clear) the Camera instance for global access."""
    global _camera_instance
    _camera_instance = cam


class Camera(HardwareModule):
    """Pi Camera Module 3 Wide NoIR via picamera2.

    Starts a dual-stream preview: low-res (320x240 grayscale) for CPU
    motion detection and main (1280x720 RGB) for YOLO / ReID.  Full
    12MP still capture is available on demand via ``capture_photo()``.
    """

    name = "camera"

    def __init__(
        self,
        rotation: int = 180,
        main_resolution: tuple[int, int] = (1280, 720),
        lores_resolution: tuple[int, int] = (320, 240),
        scan_fps: int = 5,
    ) -> None:
        super().__init__()
        self._rotation = rotation
        self._main_resolution = main_resolution
        self._lores_resolution = lores_resolution
        self._scan_fps = scan_fps

        # Set by start(), typed as Any to avoid import at module level
        self._picam2: Any = None
        self._still_config: Any = None
        self._preview_config: Any = None

    # ── Lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize picamera2 with dual-stream preview configuration."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._start_sync)
            self._started = True
            await self._emit_health(HealthStatus.OK)
            logger.info(
                "Camera started: main=%s lores=%s rotation=%d",
                self._main_resolution,
                self._lores_resolution,
                self._rotation,
            )
        except Exception as exc:
            await self._emit_health(HealthStatus.ERROR, str(exc))
            raise HardwareUnavailableError(
                f"Camera not available: {exc}"
            ) from exc

    def _start_sync(self) -> None:
        """Blocking picamera2 initialization (runs in executor)."""
        from libcamera import Transform  # type: ignore[import-untyped]
        from picamera2 import Picamera2  # type: ignore[import-untyped]

        self._picam2 = Picamera2()

        # Build transform for rotation
        transform = Transform()
        if self._rotation == 180:
            transform = Transform(hflip=True, vflip=True)
        elif self._rotation == 90:
            transform = Transform(hflip=False, vflip=True, transpose=True)
        elif self._rotation == 270:
            transform = Transform(hflip=True, vflip=False, transpose=True)

        # Dual-stream preview configuration
        self._preview_config = self._picam2.create_preview_configuration(
            main={"size": self._main_resolution, "format": "RGB888"},
            lores={"size": self._lores_resolution, "format": "YUV420"},
            transform=transform,
        )
        self._picam2.configure(self._preview_config)

        # Prepare still configuration for photo capture
        self._still_config = self._picam2.create_still_configuration(
            main={"format": "RGB888"},
            transform=transform,
        )

        self._picam2.start()

    async def stop(self) -> None:
        """Stop camera and release resources."""
        if self._picam2 is not None:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self._picam2.stop)
                await loop.run_in_executor(None, self._picam2.close)
            except Exception:
                logger.exception("Error stopping camera")
            finally:
                self._picam2 = None
        self._started = False
        await self._emit_health(HealthStatus.STOPPED)

    # ── Frame capture ──────────────────────────────────────────────

    async def get_lores_frame(self) -> np.ndarray:
        """Get the latest low-resolution frame for motion detection.

        Extracts the Y plane from YUV420 (grayscale) and crops to the
        configured lores resolution.  picamera2 pads YUV420 planes to
        multiples of 32/16, so the raw buffer is larger than requested.

        Returns:
            (H, W) uint8 grayscale numpy array.
        """
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, self._picam2.capture_array, "lores"
        )
        # YUV420 layout: Y plane is the first (H * W) bytes, but
        # picamera2 returns the full padded buffer.  We need the
        # configured resolution, not the padded one.
        h, w = self._lores_resolution[1], self._lores_resolution[0]
        # capture_array("lores") returns shape (padded_h, padded_w) for
        # YUV420 — we take the Y plane rows and columns we need.
        y_plane: np.ndarray = raw[:h, :w]
        return y_plane

    async def capture_frame(self) -> np.ndarray:
        """Capture a frame from the main stream.

        Returns:
            (H, W, 3) RGB uint8 numpy array at main_resolution.
        """
        loop = asyncio.get_event_loop()
        frame: np.ndarray = await loop.run_in_executor(
            None, self._picam2.capture_array, "main"
        )
        return frame

    async def capture_photo(self) -> np.ndarray:
        """Capture a full-resolution still image.

        Temporarily switches to the still configuration (up to 12MP),
        captures one frame, then returns to the preview configuration.
        Mode switch takes ~100-200ms.

        Returns:
            (H, W, 3) RGB uint8 numpy array at full sensor resolution.
        """
        loop = asyncio.get_event_loop()
        photo: np.ndarray = await loop.run_in_executor(
            None, self._capture_photo_sync
        )
        return photo

    def _capture_photo_sync(self) -> np.ndarray:
        """Blocking still capture with mode switch (runs in executor)."""
        frame = self._picam2.switch_mode_and_capture_array(
            self._still_config, "main"
        )
        return frame

    # ── Properties ─────────────────────────────────────────────────

    @property
    def main_resolution(self) -> tuple[int, int]:
        """Current main stream resolution (width, height)."""
        return self._main_resolution

    @property
    def photo_resolution(self) -> tuple[int, int]:
        """Still capture resolution (width, height).

        Returns the sensor's native max resolution.  The actual output
        depends on the still configuration built by picamera2.
        """
        if self._picam2 is not None:
            props = self._picam2.camera_properties
            size = props.get("PixelArraySize", (4608, 2592))
            return (size[0], size[1])
        return (4608, 2592)

    @property
    def is_streaming(self) -> bool:
        """Whether the low-res stream is active."""
        return self._started and self._picam2 is not None

    @property
    def is_available(self) -> bool:
        """Whether the camera is connected and responsive."""
        if self._picam2 is None:
            # Try to detect without fully initializing
            try:
                from picamera2 import Picamera2  # type: ignore[import-untyped]

                cameras = Picamera2.global_camera_info()
                return len(cameras) > 0
            except Exception:
                return False
        return True

    async def health_check(self) -> HealthStatus:
        """Check camera health by verifying the stream is active."""
        if not self._started:
            return HealthStatus.STOPPED
        if self._picam2 is None:
            return HealthStatus.ERROR
        return HealthStatus.OK
