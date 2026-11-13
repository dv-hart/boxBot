"""Pi Camera Module 3 Wide NoIR interface via picamera2.

Provides dual-stream video (low-res for motion detection, main for
perception) and full-resolution still capture for the photo library.
All picamera2 calls run on a **dedicated single-thread executor** owned
by this module, never the loop's default executor. picamera2 has no
cancellable wait: a wedged ``capture_array`` blocks its thread forever,
and ``asyncio.wait_for`` only abandons the await, not the thread. On
the shared default pool that strands one worker per timed-out capture
until the pool is dead — observed 2026-09-03: eight stranded workers in
30 s, after which Hailo inference, voice embedding, TTS decode and the
graceful-shutdown path all queued forever behind them. With a private
executor a stall can only ever cost this module its own worker, and
the watchdog restart recycles it.

Hardware: Pi Camera Module 3 Wide NoIR (IMX708, 120 deg FOV, 12MP)
Interface: CSI-2 ribbon cable
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
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
        colour_gains: tuple[float, float] | None = None,
        colour_correction_matrix: tuple[float, ...] | None = None,
        saturation: float = 1.0,
        tuning_file: str | None = None,
        capture_timeout: float = 5.0,
        photo_timeout: float = 15.0,
        watchdog_interval: float = 30.0,
        watchdog_stale: float = 60.0,
        restart_timeout: float = 30.0,
        max_fps: int = 15,
        min_fps: int = 5,
        abandon_limit: int = 20,
        shutdown_callback: Any = None,
    ) -> None:
        super().__init__()
        self._rotation = rotation
        # Sensor frame-rate clamp. Without FrameDurationLimits the
        # preview config lets AE run the 1536x864 mode at its 120 fps
        # floor in daylight (measured 2026-11-12: 67→83°C in 75 s, 52%
        # CPU with nothing consuming frames). Perception needs ≤15 fps.
        self._max_fps = max(1, int(max_fps))
        self._min_fps = max(1, min(int(min_fps), self._max_fps))
        # Every abandoned pipeline handle leaks its worker thread and
        # dma_heap FDs (646 abandons → 1413 threads, EMFILE on
        # 2026-11-03). Past this budget the process asks for a clean
        # restart instead of dying slowly.
        self._abandon_limit = int(abandon_limit)
        self._abandoned = 0
        self._shutdown_callback = shutdown_callback
        self._main_resolution = main_resolution
        self._lores_resolution = lores_resolution
        self._scan_fps = scan_fps
        self._colour_gains = colour_gains
        self._colour_correction_matrix = colour_correction_matrix
        self._saturation = saturation
        self._tuning_file = tuning_file
        self._capture_timeout = capture_timeout
        self._photo_timeout = photo_timeout
        self._watchdog_interval = watchdog_interval
        self._watchdog_stale = watchdog_stale
        self._restart_timeout = restart_timeout

        # Set by start(), typed as Any to avoid import at module level
        self._picam2: Any = None
        self._still_config: Any = None
        self._preview_config: Any = None

        # Stall watchdog state. libcamera can stop completing requests
        # without any exception or kernel error (observed 2026-07-26:
        # frames froze mid-stream after ~6 days uptime; every capture
        # blocked forever and perception went silently blind). The
        # watchdog probes when frames go stale and restarts the
        # picamera2 pipeline in-process.
        self._last_frame_time: float | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._restart_lock = asyncio.Lock()

        # Private executor for every blocking picamera2 call. One
        # worker: the camera is a single serial device, and a single
        # worker means a wedged capture strands exactly one thread —
        # later captures queue behind it and time out without spawning
        # more. ``_recycle_executor`` swaps in a fresh pool whenever a
        # thread is known to be stuck.
        self._executor = self._new_executor()

    # ── Executor management ────────────────────────────────────────

    @staticmethod
    def _new_executor() -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera")

    def _recycle_executor(self) -> None:
        """Abandon the current executor (and any stranded worker in it).

        Pending work items are cancelled; the stuck thread is left to
        die with the process. ``shutdown(wait=False)`` never blocks.
        Callers awaiting a cancelled item see ``HardwareUnavailableError``
        from ``_run_bounded`` — never a bare ``CancelledError``.
        """
        old = self._executor
        self._executor = self._new_executor()
        old.shutdown(wait=False, cancel_futures=True)
        self._abandoned += 1
        if self._abandon_limit > 0 and self._abandoned == self._abandon_limit:
            logger.critical(
                "Camera abandoned %d pipeline workers since start — leaked "
                "threads/FDs past budget; requesting graceful restart",
                self._abandoned,
            )
            if self._shutdown_callback is not None:
                try:
                    self._shutdown_callback()
                except Exception:
                    logger.exception("Camera shutdown callback failed")

    async def _run_bounded(self, fn: Any, *args: Any, timeout: float) -> Any:
        """Run ``fn`` on the camera executor with a hard timeout.

        Raises ``asyncio.TimeoutError`` if it does not return in time.
        The worker thread is then presumed stranded; callers decide
        whether to recycle the executor (restart/stop do, captures
        leave it for the watchdog).
        """
        loop = asyncio.get_event_loop()
        fut = loop.run_in_executor(self._executor, fn, *args)
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.CancelledError:
            # ``_recycle_executor`` cancels queued work items. That
            # cancellation must surface to the caller as a hardware
            # error, not as CancelledError — the latter unwinds the
            # caller's task (perception's ``_run_loop`` broke on it and
            # went blind for 26 days, 2026-10-17). A cancellation of the
            # *awaiting task* itself is re-raised untouched.
            task = asyncio.current_task()
            cancelling = getattr(task, "cancelling", None)
            self_cancelled = bool(cancelling()) if cancelling else task is None
            if fut.cancelled() and not self_cancelled:
                raise HardwareUnavailableError(
                    "camera executor recycled while capture was queued"
                ) from None
            raise

    # ── Lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize picamera2 with dual-stream preview configuration."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(self._executor, self._start_sync)
            self._started = True
            self._last_frame_time = time.monotonic()
            if self._watchdog_task is None:
                self._watchdog_task = asyncio.create_task(
                    self._watchdog_loop(), name="camera-watchdog"
                )
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

        # Tuning-file override. libcamera auto-selects by sensor name —
        # for a NoIR sensor that means the *_noir tuning, whose colour
        # tables assume no IR-cut filter. With a filter retrofitted, the
        # standard tuning (e.g. "imx708_wide.json") is the correct one;
        # load_tuning_file resolves the platform dir (pisp/vc4) itself.
        tuning = None
        if self._tuning_file:
            tuning = Picamera2.load_tuning_file(self._tuning_file)

        self._picam2 = Picamera2(tuning=tuning)

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
            controls={
                "FrameDurationLimits": (
                    int(1_000_000 / self._max_fps),
                    int(1_000_000 / self._min_fps),
                ),
            },
        )
        self._picam2.configure(self._preview_config)

        # Prepare still configuration for photo capture
        self._still_config = self._picam2.create_still_configuration(
            main={"format": "RGB888"},
            transform=transform,
        )

        self._picam2.start()

        # Colour correction (NoIR sensor — see HardwareCameraConfig docstring).
        # Both gains and CCM must be set together; otherwise AWB stays on.
        controls: dict[str, Any] = {}
        if self._colour_gains is not None and self._colour_correction_matrix is not None:
            controls["AwbEnable"] = False
            controls["ColourGains"] = tuple(self._colour_gains)
            controls["ColourCorrectionMatrix"] = tuple(self._colour_correction_matrix)
        if self._saturation != 1.0:
            controls["Saturation"] = float(self._saturation)
        if controls:
            self._picam2.set_controls(controls)
            logger.info("Camera controls applied: %s", controls)

    async def stop(self) -> None:
        """Stop camera and release resources."""
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None
        if self._picam2 is not None:
            # Bounded: a wedged pipeline must not be able to hang the
            # process's shutdown sequence. Recycle first so teardown
            # gets a live worker even if a capture is currently stuck.
            self._recycle_executor()
            picam2, self._picam2 = self._picam2, None
            for op in (picam2.stop, picam2.close):
                try:
                    await self._run_bounded(op, timeout=self._capture_timeout)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Camera stop: %s() timed out; abandoning handle",
                        op.__name__,
                    )
                    self._recycle_executor()
                except Exception:
                    logger.exception("Error stopping camera (%s)", op.__name__)
        # Drop the executor; a fresh (idle, threadless) one keeps a
        # later start() valid without special-casing a closed pool.
        self._recycle_executor()
        self._started = False
        await self._emit_health(HealthStatus.STOPPED)

    # ── Frame capture ──────────────────────────────────────────────

    async def _run_capture(
        self, fn: Any, *args: Any, timeout: float
    ) -> np.ndarray:
        """Run a blocking capture on the camera executor with a hard timeout.

        A timed-out capture strands the executor's single worker
        (picamera2 has no cancellable wait). Later captures queue
        behind it and time out too — cheaply, without spawning threads
        — until the watchdog's pipeline restart recycles the executor.
        Raising here keeps callers' error paths live instead of hanging
        them for the sandbox's full 30s budget.
        """
        try:
            result: np.ndarray = await self._run_bounded(
                fn, *args, timeout=timeout
            )
        except asyncio.TimeoutError:
            await self._emit_health(
                HealthStatus.DEGRADED, f"capture timed out after {timeout:.0f}s"
            )
            raise HardwareUnavailableError(
                f"camera capture timed out after {timeout:.0f}s"
            ) from None
        self._last_frame_time = time.monotonic()
        return result

    async def get_lores_frame(self) -> np.ndarray:
        """Get the latest low-resolution frame for motion detection.

        Extracts the Y plane from YUV420 (grayscale) and crops to the
        configured lores resolution.  picamera2 pads YUV420 planes to
        multiples of 32/16, so the raw buffer is larger than requested.

        Returns:
            (H, W) uint8 grayscale numpy array.
        """
        raw = await self._run_capture(
            self._picam2.capture_array, "lores",
            timeout=self._capture_timeout,
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
        return await self._run_capture(
            self._picam2.capture_array, "main",
            timeout=self._capture_timeout,
        )

    async def capture_photo(self) -> np.ndarray:
        """Capture a full-resolution still image.

        Temporarily switches to the still configuration (up to 12MP),
        captures one frame, then returns to the preview configuration.
        Mode switch takes ~100-200ms.

        Returns:
            (H, W, 3) RGB uint8 numpy array at full sensor resolution.
        """
        return await self._run_capture(
            self._capture_photo_sync,
            timeout=self._photo_timeout,
        )

    def _capture_photo_sync(self) -> np.ndarray:
        """Blocking still capture with mode switch (runs in executor)."""
        frame = self._picam2.switch_mode_and_capture_array(
            self._still_config, "main"
        )
        return frame

    # ── Stall watchdog ─────────────────────────────────────────────

    @property
    def frame_age(self) -> float | None:
        """Seconds since the last successful capture, or None pre-start."""
        if self._last_frame_time is None:
            return None
        return time.monotonic() - self._last_frame_time

    async def _watchdog_loop(self) -> None:
        """Probe when frames go stale; restart the pipeline on a wedge.

        Perception stops grabbing frames during CONVERSATION state, so
        staleness alone is not a fault — the probe capture is what
        distinguishes "idle" from "wedged". A successful probe refreshes
        ``_last_frame_time``, so probes fire at most once per interval
        and only when nothing else has captured recently.
        """
        while True:
            await asyncio.sleep(self._watchdog_interval)
            try:
                if not self._started:
                    continue
                if self._picam2 is None:
                    # A previous restart's re-init failed and left no
                    # handle. Skipping here parked the watchdog for good
                    # (2026-11-03, EMFILE) — retry instead.
                    logger.error(
                        "Camera watchdog: no pipeline handle (last restart "
                        "failed) — retrying restart"
                    )
                    await self._restart()
                    continue
                age = self.frame_age
                if age is None or age < self._watchdog_stale:
                    continue
                try:
                    await self.get_lores_frame()
                    continue  # probe ok — camera alive, just idle
                except Exception:
                    logger.error(
                        "Camera watchdog: frames stale for %.0fs and probe "
                        "failed — restarting camera pipeline",
                        age,
                    )
                await self._restart()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Camera watchdog iteration failed")

    async def _restart(self) -> None:
        """Tear down and re-init picamera2 in-process.

        Best effort: teardown of a wedged pipeline can itself block, so
        it runs under the same hard timeout and a stuck close is
        abandoned. If the old handle never released the device, re-init
        fails and the watchdog retries next interval.

        Every step runs on a freshly recycled executor: the worker that
        wedged on the stale capture is abandoned up front, and any
        teardown op that times out strands the new worker, so it is
        recycled again before re-init. Re-init is bounded too — an
        unbounded ``_start_sync`` here would hold ``_restart_lock``
        forever and kill the watchdog (2026-09-03).
        """
        async with self._restart_lock:
            old = self._picam2
            self._picam2 = None
            self._recycle_executor()
            if old is not None:
                for op in (old.stop, old.close):
                    try:
                        await self._run_bounded(
                            op, timeout=self._capture_timeout
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Camera restart: %s() timed out; abandoning "
                            "old pipeline handle", op.__name__,
                        )
                        self._recycle_executor()
                    except Exception:
                        logger.warning(
                            "Camera restart: %s() failed; abandoning old "
                            "pipeline handle", op.__name__, exc_info=True,
                        )
            try:
                await self._run_bounded(
                    self._start_sync, timeout=self._restart_timeout
                )
            except asyncio.TimeoutError:
                self._recycle_executor()
                await self._emit_health(
                    HealthStatus.ERROR,
                    f"restart timed out after {self._restart_timeout:.0f}s",
                )
                logger.error(
                    "Camera restart: init timed out after %.0fs; will retry",
                    self._restart_timeout,
                )
                return
            except Exception as exc:
                await self._emit_health(HealthStatus.ERROR, str(exc))
                logger.exception("Camera restart failed; will retry")
                return
            self._last_frame_time = time.monotonic()
            await self._emit_health(HealthStatus.OK, "recovered by watchdog")
            logger.info("Camera watchdog: pipeline restarted successfully")

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
        """Check camera health by verifying frames are actually flowing."""
        if not self._started:
            return HealthStatus.STOPPED
        if self._picam2 is None:
            return HealthStatus.ERROR
        # The watchdog probes whenever frames go stale, so a healthy
        # camera never exceeds stale + interval (+ probe timeout) even
        # when perception is idle. Older than that = wedged and not yet
        # recovered.
        age = self.frame_age
        if age is not None and age > (
            self._watchdog_stale + 2 * self._watchdog_interval
        ):
            return HealthStatus.ERROR
        return HealthStatus.OK
