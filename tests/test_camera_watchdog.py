"""Camera capture timeouts + stall watchdog.

The Jul 26 failure mode: libcamera silently stops completing requests —
no exception, no kernel error — and every ``capture_array`` call blocks
forever. These tests fake picamera2 to verify:

- captures fail fast with HardwareUnavailableError instead of hanging
- successful captures refresh the frame timestamp
- the watchdog probes on staleness and restarts a wedged pipeline
- health_check reports ERROR once frames are stale beyond recovery time
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from boxbot.hardware.base import HardwareUnavailableError, HealthStatus
from boxbot.hardware.camera import Camera


class FakePicam2:
    """Stand-in for picamera2 with a switchable 'wedged' mode."""

    def __init__(self) -> None:
        self.wedged = False
        self.stopped = False
        self.closed = False
        self._unwedge = threading.Event()

    def capture_array(self, stream: str) -> np.ndarray:
        if self.wedged:
            # Simulate the real stall: block until torn down.
            self._unwedge.wait(timeout=30)
        if stream == "lores":
            return np.zeros((256, 320), dtype=np.uint8)
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    def switch_mode_and_capture_array(self, config, stream: str) -> np.ndarray:
        if self.wedged:
            self._unwedge.wait(timeout=30)
        return np.zeros((2592, 4608, 3), dtype=np.uint8)

    def stop(self) -> None:
        self.stopped = True
        self._unwedge.set()

    def close(self) -> None:
        self.closed = True


def make_camera(**kwargs) -> tuple[Camera, FakePicam2]:
    cam = Camera(
        rotation=0,
        capture_timeout=kwargs.pop("capture_timeout", 0.2),
        photo_timeout=kwargs.pop("photo_timeout", 0.2),
        watchdog_interval=kwargs.pop("watchdog_interval", 0.1),
        watchdog_stale=kwargs.pop("watchdog_stale", 0.3),
        **kwargs,
    )
    fake = FakePicam2()
    cam._picam2 = fake
    cam._started = True
    cam._last_frame_time = time.monotonic()
    return cam, fake


@pytest.mark.asyncio
async def test_capture_updates_frame_time():
    cam, _ = make_camera()
    cam._last_frame_time = time.monotonic() - 100
    frame = await cam.capture_frame()
    assert frame.shape == (720, 1280, 3)
    assert cam.frame_age is not None and cam.frame_age < 1.0


@pytest.mark.asyncio
async def test_lores_frame_cropped_to_configured_resolution():
    cam, _ = make_camera()
    frame = await cam.get_lores_frame()
    assert frame.shape == (240, 320)


@pytest.mark.asyncio
async def test_wedged_capture_raises_instead_of_hanging():
    cam, fake = make_camera()
    fake.wedged = True
    start = time.monotonic()
    with pytest.raises(HardwareUnavailableError, match="timed out"):
        await cam.capture_frame()
    assert time.monotonic() - start < 2.0
    fake._unwedge.set()  # release the stranded executor thread


@pytest.mark.asyncio
async def test_wedged_photo_raises():
    cam, fake = make_camera()
    fake.wedged = True
    with pytest.raises(HardwareUnavailableError, match="timed out"):
        await cam.capture_photo()
    fake._unwedge.set()


@pytest.mark.asyncio
async def test_watchdog_restarts_wedged_pipeline(monkeypatch):
    cam, fake = make_camera()

    new_fake = FakePicam2()

    def fake_start_sync() -> None:
        cam._picam2 = new_fake

    monkeypatch.setattr(cam, "_start_sync", fake_start_sync)

    fake.wedged = True
    cam._last_frame_time = time.monotonic() - 100  # stale

    cam._watchdog_task = asyncio.create_task(cam._watchdog_loop())
    try:
        for _ in range(100):
            await asyncio.sleep(0.05)
            if cam._picam2 is new_fake:
                break
        assert cam._picam2 is new_fake, "watchdog did not restart pipeline"
        assert fake.stopped or fake.closed
        assert cam.frame_age is not None and cam.frame_age < 5.0
        # Camera usable again after recovery
        frame = await cam.capture_frame()
        assert frame.shape == (720, 1280, 3)
    finally:
        cam._watchdog_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cam._watchdog_task
        cam._watchdog_task = None
        fake._unwedge.set()


@pytest.mark.asyncio
async def test_watchdog_probe_skips_healthy_idle_camera():
    cam, fake = make_camera()
    cam._last_frame_time = time.monotonic() - 100  # stale but not wedged

    cam._watchdog_task = asyncio.create_task(cam._watchdog_loop())
    try:
        await asyncio.sleep(0.3)
        # Probe succeeded — timestamp refreshed, pipeline untouched
        assert cam._picam2 is fake
        assert not fake.stopped
        assert cam.frame_age is not None and cam.frame_age < 5.0
    finally:
        cam._watchdog_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cam._watchdog_task
        cam._watchdog_task = None


@pytest.mark.asyncio
async def test_health_check_errors_on_stale_frames():
    cam, _ = make_camera()
    assert await cam.health_check() == HealthStatus.OK
    cam._last_frame_time = time.monotonic() - 100  # >> stale + 2*interval
    assert await cam.health_check() == HealthStatus.ERROR


@pytest.mark.asyncio
async def test_health_check_stopped_when_not_started():
    cam = Camera(rotation=0)
    assert await cam.health_check() == HealthStatus.STOPPED
