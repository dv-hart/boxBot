"""ReSpeaker 4-Mic Array v2.0 (XMOS XVF3000) interface via sounddevice.

Provides 6-channel 16kHz S16_LE capture with single-channel extraction,
consumer fan-out for audio chunks, LED ring control via USB vendor commands,
and direction-of-arrival (DOA) reading.

Hardware: Seeed ReSpeaker 4-Mic Array v2.0 (USB PID 0x0018)
Interface: USB Audio + USB Vendor (pyusb)
Channels: 6 (4 raw mic + 2 processed), output defaults to ch 0 (beamformed)
LEDs: 12x APA102 RGB via USB vendor commands
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from typing import Any, Awaitable, Callable

from boxbot.hardware.base import (
    AudioChunk,
    HardwareModule,
    HardwareUnavailableError,
    HealthStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import sounddevice as sd

    _HAS_SOUNDDEVICE = True
except ImportError:
    sd = None  # type: ignore[assignment]
    _HAS_SOUNDDEVICE = False

try:
    import usb.core
    import usb.util

    _HAS_PYUSB = True
except ImportError:
    usb = None  # type: ignore[assignment]
    _HAS_PYUSB = False


# ---------------------------------------------------------------------------
# ReSpeaker USB constants
# ---------------------------------------------------------------------------

_RESPEAKER_VENDOR_ID = 0x2886
_RESPEAKER_PRODUCT_ID = 0x0018

# USB vendor request types
_USB_REQUEST_TYPE_WRITE = 0x40
_USB_REQUEST_TYPE_READ = 0xC0
_USB_REQUEST = 0x00

# LED ring geometry
_LED_COUNT = 12

# ---------------------------------------------------------------------------
# Seeed pixel_ring protocol
# ---------------------------------------------------------------------------
#
# The XMOS XVF3000 firmware on this board exposes a command-based protocol
# for LED and parameter access. All LED commands are issued as vendor OUT
# control transfers with the command ID in wValue and a fixed wIndex
# pointing at the pixel_ring interface. Parameter reads use a parameter-ID
# encoding on top of a vendor IN control transfer.
#
# Reference: https://github.com/respeaker/pixel_ring (Seeed vendor lib)
# Empirically validated on-device 2026-04-23; see
# docs/plans/led-state-diagnostic.md for probe results.

_PIXEL_RING_IFACE = 0x1C   # wIndex for all pixel_ring commands
_CMD_TRACE = 0             # firmware-driven reactive mode
_CMD_MONO = 1              # all 12 LEDs one color (4-byte RGBA)
_CMD_LISTEN = 2            # optional dir param
_CMD_SPEAK = 3
_CMD_THINK = 4
_CMD_SPIN = 5
_CMD_SHOW = 6              # custom per-LED (48 bytes RGBA * 12)
_CMD_SET_BRIGHTNESS = 0x20

_PARAM_DOAANGLE_ID = 21

# Consumer callback type: async callable receiving AudioChunk
AudioConsumer = Callable[[AudioChunk], Awaitable[None]]


# ---------------------------------------------------------------------------
# LED pattern definitions
# ---------------------------------------------------------------------------

# Base colors for patterns (R, G, B)
_COLOR_OFF = (0, 0, 0)
_COLOR_IDLE = (20, 15, 8)  # warm amber, very dim
_COLOR_LISTENING = (0, 120, 200)  # bright blue
_COLOR_THINKING = (180, 100, 20)  # amber-coral (boxbot theme)
_COLOR_SPEAKING = (40, 180, 80)  # green
_COLOR_ERROR = (200, 20, 10)  # red

# Pattern types
_PATTERN_STATIC = "static"
_PATTERN_PULSE = "pulse"
_PATTERN_CHASE = "chase"

_PATTERN_CONFIG: dict[str, dict[str, Any]] = {
    "off": {"type": _PATTERN_STATIC, "color": _COLOR_OFF},
    "idle": {"type": _PATTERN_PULSE, "color": _COLOR_IDLE, "speed": 1.5},
    "listening": {"type": _PATTERN_PULSE, "color": _COLOR_LISTENING, "speed": 2.0},
    "thinking": {"type": _PATTERN_CHASE, "color": _COLOR_THINKING, "speed": 3.0},
    "speaking": {"type": _PATTERN_PULSE, "color": _COLOR_SPEAKING, "speed": 1.0},
    # "doa" is defined but currently unused — DOA-as-listening-overlay is
    # future work (will sample get_doa() at ~5 Hz and blend into the
    # listening pulse). get_doa() itself is available for other consumers.
    "doa": {"type": _PATTERN_STATIC, "color": _COLOR_LISTENING},
    "error": {"type": _PATTERN_PULSE, "color": _COLOR_ERROR, "speed": 4.0},
}


class Microphone(HardwareModule):
    """ReSpeaker 4-Mic Array v2.0 via sounddevice + pyusb.

    Captures 6-channel 16kHz audio, extracts the configured output channel,
    and fans out AudioChunk instances to registered async consumers. Also
    provides LED ring control and direction-of-arrival reading.
    """

    name = "microphone"

    def __init__(
        self,
        config: Any | None = None,
        *,
        device_name: str = "ReSpeaker",
        sample_rate: int = 16000,
        capture_channels: int = 6,
        output_channel: int = 0,
        chunk_duration_ms: int = 64,
        doa_enabled: bool = True,
        led_brightness: float = 0.5,
    ) -> None:
        super().__init__()
        # Accept either a config object or individual kwargs
        if config is not None:
            device_name = config.device_name
            sample_rate = config.sample_rate
            capture_channels = config.capture_channels
            output_channel = config.output_channel
            chunk_duration_ms = config.chunk_duration_ms
            doa_enabled = config.doa_enabled
            led_brightness = config.led_brightness
        self._device_name = device_name
        self._sample_rate = sample_rate
        self._capture_channels = capture_channels
        self._output_channel = output_channel
        self._chunk_duration_ms = chunk_duration_ms
        self._doa_enabled = doa_enabled
        self._led_brightness = max(0.0, min(1.0, led_brightness))

        # Computed
        self._chunk_frames = int(sample_rate * chunk_duration_ms / 1000)

        # Runtime state
        self._stream: Any = None  # sd.InputStream
        self._device_index: int | None = None
        self._usb_device: Any = None  # usb.core.Device
        self._loop: asyncio.AbstractEventLoop | None = None
        # Consumers are keyed by a stable integer handle returned from
        # add_consumer(). This avoids the bound-method identity pitfall:
        # ``obj.method is obj.method`` is False, so using the callable
        # itself as the key silently breaks remove_consumer().
        self._consumers: list[tuple[int, AudioConsumer, str]] = []
        self._next_consumer_id: int = 1
        self._animation_task: asyncio.Task[None] | None = None
        self._current_pattern: str = "off"
        self._pattern_params: dict[str, Any] = {}
        self._animation_running: bool = False
        # USB write failure counter. First failure logs at WARNING with
        # the exception, subsequent failures at DEBUG to avoid log spam
        # when the udev rule is missing or the device is unplugged.
        self._usb_write_failure_count: int = 0
        # Liveness: the audio-thread callback updates this each chunk;
        # the watchdog task compares against wall time and restarts the
        # stream if callbacks stop firing. Prevents the silent-death
        # failure mode where sounddevice stalls and nothing notices.
        self._last_chunk_monotonic: float = 0.0
        self._watchdog_task: asyncio.Task[None] | None = None
        self._stream_restart_count: int = 0

    # ── Lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize audio stream, USB device for LEDs/DOA, and start animation."""
        self._loop = asyncio.get_event_loop()

        # Find sounddevice device index
        self._device_index = self._find_device_index()
        if self._device_index is None:
            await self._emit_health(
                HealthStatus.ERROR, "ReSpeaker device not found in audio devices"
            )
            raise HardwareUnavailableError(
                f"Audio device matching '{self._device_name}' not found"
            )

        # Initialize USB for LEDs + DOA
        self._init_usb()

        # Open audio stream
        try:
            self._open_stream()
        except Exception as exc:
            await self._emit_health(HealthStatus.ERROR, str(exc))
            raise HardwareUnavailableError(
                f"Failed to open audio stream: {exc}"
            ) from exc

        # Start LED animation loop
        self._animation_running = True
        self._animation_task = asyncio.create_task(self._animation_loop())

        # Start stream liveness watchdog
        self._last_chunk_monotonic = time.monotonic()
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

        # Set initial LED state
        await self.set_led_pattern("idle")

        self._started = True
        await self._emit_health(HealthStatus.OK)
        logger.info(
            "Microphone started: device=%s rate=%d ch=%d->%d chunk=%dms",
            self._device_name,
            self._sample_rate,
            self._capture_channels,
            self._output_channel,
            self._chunk_duration_ms,
        )

    async def stop(self) -> None:
        """Stop audio stream, LED animation, and release USB handle."""
        # Stop watchdog
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None

        # Stop animation
        self._animation_running = False
        if self._animation_task is not None:
            self._animation_task.cancel()
            try:
                await self._animation_task
            except asyncio.CancelledError:
                pass
            self._animation_task = None

        # Turn off LEDs
        self._set_all_leds_raw(_COLOR_OFF)

        # Stop audio stream
        self._close_stream()

        # Release USB
        if self._usb_device is not None:
            try:
                usb.util.dispose_resources(self._usb_device)
            except Exception:
                logger.exception("Error releasing USB device")
            finally:
                self._usb_device = None

        self._started = False
        self._loop = None
        await self._emit_health(HealthStatus.STOPPED)

    # ── Stream management ─────────────────────────────────────────

    def _open_stream(self) -> None:
        """Open (or reopen) the sounddevice InputStream.

        Called from start() and from the watchdog when a stalled stream
        is detected. Runs synchronously — sounddevice start is fast.
        """
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._capture_channels,
            dtype="int16",
            device=self._device_index,
            blocksize=self._chunk_frames,
            callback=self._audio_callback,
        )
        self._stream.start()

    def _close_stream(self) -> None:
        """Stop and close the current stream, swallowing errors."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("Error stopping audio stream")
            finally:
                self._stream = None

    async def _watchdog_loop(self) -> None:
        """Monitor the audio stream and restart it if callbacks stall.

        sounddevice has no built-in recovery: if the underlying ALSA/USB
        stream wedges (hub disconnect, driver glitch), the callback
        simply stops firing and no exception surfaces. This watchdog
        compares wall time against the last observed chunk timestamp
        and forces a reopen if chunks haven't arrived for a while.
        """
        # Stall threshold is generous (5s >> any legitimate gap at 64ms
        # cadence) so we don't fight against brief scheduler hiccups.
        stall_seconds = 5.0
        poll_interval = 2.0
        try:
            while self._started:
                await asyncio.sleep(poll_interval)
                if not self._started:
                    return
                elapsed = time.monotonic() - self._last_chunk_monotonic
                if elapsed < stall_seconds:
                    continue

                self._stream_restart_count += 1
                logger.error(
                    "Audio stream appears stalled (%.1fs since last "
                    "chunk). Restarting stream (restart #%d).",
                    elapsed, self._stream_restart_count,
                )
                try:
                    await self._emit_health(
                        HealthStatus.DEGRADED,
                        f"audio stream stalled for {elapsed:.1f}s",
                    )
                except Exception:
                    pass

                # Reopen the stream. Do it on the event loop thread —
                # sounddevice close/open are blocking but fast (<100ms);
                # running them inline is simpler than juggling an
                # executor + re-registration.
                self._close_stream()
                try:
                    self._open_stream()
                    self._last_chunk_monotonic = time.monotonic()
                    logger.info(
                        "Audio stream reopened after stall (restart #%d)",
                        self._stream_restart_count,
                    )
                    try:
                        await self._emit_health(HealthStatus.OK)
                    except Exception:
                        pass
                except Exception:
                    logger.exception(
                        "Failed to reopen audio stream after stall; "
                        "will retry on next watchdog tick",
                    )
        except asyncio.CancelledError:
            return

    # ── Consumer fan-out ──────────────────────────────────────────

    def add_consumer(self, callback: AudioConsumer, name: str = "") -> int:
        """Register an async callback to receive audio chunks.

        Args:
            callback: Async callable that receives AudioChunk.
            name: Human-readable name for logging.

        Returns:
            A handle id. Pass this to ``remove_consumer`` to unregister.
            Callers MUST store this id — bound methods are not
            identity-stable across accesses, so the callable itself is
            not a reliable key.
        """
        handle = self._next_consumer_id
        self._next_consumer_id += 1
        display = name or repr(callback)
        self._consumers.append((handle, callback, display))
        logger.debug(
            "Audio consumer added: %s [id=%d] (total: %d)",
            display, handle, len(self._consumers),
        )
        return handle

    def remove_consumer(self, handle: int) -> bool:
        """Remove a previously registered consumer by handle.

        Args:
            handle: The integer handle returned from ``add_consumer``.

        Returns:
            True if a consumer was removed; False if the handle was
            unknown (caller logic bug — should never happen if handles
            are stored correctly).
        """
        for i, (h, _cb, name) in enumerate(self._consumers):
            if h == handle:
                self._consumers.pop(i)
                logger.debug(
                    "Audio consumer removed: %s [id=%d] (total: %d)",
                    name, handle, len(self._consumers),
                )
                return True
        logger.warning(
            "remove_consumer called with unknown handle %d — consumer "
            "list unchanged (total: %d)",
            handle, len(self._consumers),
        )
        return False

    def _audio_callback(
        self,
        indata: Any,  # numpy ndarray, shape (frames, channels)
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """sounddevice InputStream callback (runs in audio thread).

        Extracts the configured output channel and dispatches to the
        async event loop via call_soon_threadsafe.
        """
        # Always stamp liveness, even with no consumers and even on
        # status-flagged chunks — the watchdog only cares whether the
        # stream is producing callbacks at all.
        now = time.monotonic()
        self._last_chunk_monotonic = now

        if status:
            logger.warning("Audio stream status: %s", status)

        if not self._consumers or self._loop is None:
            return

        # Extract the mono output channel (int16)
        mono = indata[:, self._output_channel].copy()
        pcm_bytes = mono.tobytes()

        chunk = AudioChunk(
            data=pcm_bytes,
            timestamp=now,
            sample_rate=self._sample_rate,
            channels=1,
            frames=frames,
        )

        # Dispatch to async loop from the audio thread
        self._loop.call_soon_threadsafe(
            self._loop.create_task, self._dispatch_chunk(chunk)
        )

    async def _dispatch_chunk(self, chunk: AudioChunk) -> None:
        """Distribute an audio chunk to all registered consumers.

        Each consumer is called concurrently. Slow or failing consumers
        do not block others.
        """
        if not self._consumers:
            return

        async def _safe_deliver(
            callback: AudioConsumer, name: str, chunk: AudioChunk
        ) -> None:
            try:
                await callback(chunk)
            except Exception:
                logger.exception("Error in audio consumer %s", name)

        # Snapshot the consumer list: a consumer that unregisters itself
        # during delivery must not mutate the iterable we're awaiting on.
        snapshot = list(self._consumers)
        await asyncio.gather(
            *(_safe_deliver(cb, name, chunk) for _h, cb, name in snapshot)
        )

    # ── LED ring ──────────────────────────────────────────────────

    async def set_led_pattern(
        self, pattern: str, params: dict[str, Any] | None = None
    ) -> None:
        """Set the LED ring pattern.

        Args:
            pattern: One of: off, idle, listening, thinking, speaking, doa, error
            params: Optional overrides (e.g. {"angle": 90} for doa pattern).
        """
        if pattern not in _PATTERN_CONFIG:
            logger.warning("Unknown LED pattern: %s", pattern)
            return
        self._current_pattern = pattern
        self._pattern_params = params or {}

        # For static patterns, apply immediately
        config = _PATTERN_CONFIG[pattern]
        if config["type"] == _PATTERN_STATIC:
            if pattern == "doa" and "angle" in self._pattern_params:
                self._set_doa_leds(
                    self._pattern_params["angle"], config["color"]
                )
            else:
                self._set_all_leds_raw(config["color"])

    async def _animation_loop(self) -> None:
        """Internal loop that drives animated LED patterns at ~30 FPS."""
        frame_interval = 1.0 / 30.0
        frame_count = 0

        while self._animation_running:
            try:
                config = _PATTERN_CONFIG.get(self._current_pattern)
                if config is None or config["type"] == _PATTERN_STATIC:
                    await asyncio.sleep(frame_interval)
                    frame_count += 1
                    continue

                t = frame_count * frame_interval
                speed = config.get("speed", 1.0)
                color = config["color"]

                if config["type"] == _PATTERN_PULSE:
                    self._render_pulse(color, t, speed)
                elif config["type"] == _PATTERN_CHASE:
                    self._render_chase(color, t, speed)

                frame_count += 1
                await asyncio.sleep(frame_interval)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in LED animation loop")
                await asyncio.sleep(1.0)

    def _render_pulse(
        self, base_color: tuple[int, int, int], t: float, speed: float
    ) -> None:
        """Render a breathing/pulse effect on all LEDs."""
        import math

        # Sinusoidal brightness oscillation between 0.1 and 1.0
        factor = 0.55 + 0.45 * math.sin(t * speed * 2 * math.pi)
        brightness = factor * self._led_brightness
        color = (
            int(base_color[0] * brightness),
            int(base_color[1] * brightness),
            int(base_color[2] * brightness),
        )
        self._set_all_leds_raw(color)

    def _render_chase(
        self, base_color: tuple[int, int, int], t: float, speed: float
    ) -> None:
        """Render a rotating chase pattern around the ring."""
        import math

        position = (t * speed) % _LED_COUNT
        colors: list[tuple[int, int, int]] = []
        for i in range(_LED_COUNT):
            # Distance from the "head" of the chase
            dist = min(abs(i - position), _LED_COUNT - abs(i - position))
            # Fade based on distance, tail length ~3 LEDs
            factor = max(0.0, 1.0 - dist / 3.0)
            factor = factor * factor  # quadratic falloff
            brightness = factor * self._led_brightness
            colors.append((
                int(base_color[0] * brightness),
                int(base_color[1] * brightness),
                int(base_color[2] * brightness),
            ))
        self._set_leds_raw(colors)

    def _set_doa_leds(
        self, angle: int, color: tuple[int, int, int]
    ) -> None:
        """Light the LED closest to the given DOA angle, with neighbors dimmed."""
        # LEDs are evenly spaced at 30-degree intervals (360 / 12)
        primary_led = round(angle / 30.0) % _LED_COUNT
        colors: list[tuple[int, int, int]] = []
        for i in range(_LED_COUNT):
            dist = min(abs(i - primary_led), _LED_COUNT - abs(i - primary_led))
            if dist == 0:
                factor = self._led_brightness
            elif dist == 1:
                factor = self._led_brightness * 0.3
            else:
                factor = 0.0
            colors.append((
                int(color[0] * factor),
                int(color[1] * factor),
                int(color[2] * factor),
            ))
        self._set_leds_raw(colors)

    def _set_all_leds_raw(self, color: tuple[int, int, int]) -> None:
        """Set all LEDs to the same color via USB vendor commands."""
        self._set_leds_raw([color] * _LED_COUNT)

    def _set_leds_raw(self, colors: list[tuple[int, int, int]]) -> None:
        """Set individual LED colors via the pixel_ring SHOW command.

        Packs all 12 LEDs into a single 48-byte RGBA payload (R, G, B,
        brightness per LED) and sends it as one vendor OUT control
        transfer with wValue=_CMD_SHOW and wIndex=_PIXEL_RING_IFACE.

        Args:
            colors: List of (R, G, B) tuples, one per LED. Shorter
                lists are zero-padded; extra entries are ignored.
        """
        if self._usb_device is None:
            return
        payload = bytearray()
        for (r, g, b) in colors[:_LED_COUNT]:
            # RGBA: alpha byte doubles as per-LED brightness (0xFF = full).
            payload += bytes([r & 0xFF, g & 0xFF, b & 0xFF, 0xFF])
        # Zero-pad to the full 12-LED frame if the caller passed fewer.
        while len(payload) < _LED_COUNT * 4:
            payload += bytes([0, 0, 0, 0])
        try:
            self._usb_device.ctrl_transfer(
                _USB_REQUEST_TYPE_WRITE,
                _USB_REQUEST,
                _CMD_SHOW,
                _PIXEL_RING_IFACE,
                bytes(payload),
                8000,
            )
        except Exception as e:
            self._log_usb_write_failure(e)

    def _log_usb_write_failure(self, exc: BaseException) -> None:
        """Log a USB LED write failure, throttled to avoid log spam.

        The first failure per process lifetime logs at WARNING with the
        exception type/message so a misconfigured udev rule or
        disconnected device is loud. All subsequent failures log at
        DEBUG. Animation runs at ~30 FPS, so uncapped WARNING would
        flood logs quickly.
        """
        self._usb_write_failure_count += 1
        if self._usb_write_failure_count == 1:
            logger.warning(
                "ReSpeaker LED USB write failed (%s: %s). Check that "
                "/etc/udev/rules.d/60-respeaker.rules is installed and "
                "that the user is in the 'plugdev' group. Subsequent "
                "failures this session will be logged at DEBUG.",
                type(exc).__name__,
                exc,
            )
        else:
            logger.debug(
                "ReSpeaker LED USB write failed (#%d): %s",
                self._usb_write_failure_count,
                exc,
            )

    # ── DOA ────────────────────────────────────────────────────────

    def get_doa(self) -> int | None:
        """Read direction of arrival from the XMOS chip.

        Uses the parameter-ID encoding: wValue = (read flag 0x80) |
        (int flag 0x40) | DOAANGLE param id. The 8-byte response is
        int32 LE value followed by int32 LE max; we return the value.

        Returns:
            Angle in degrees (0-359), or None if DOA is unavailable.
        """
        if not self._doa_enabled or self._usb_device is None:
            return None

        # Read flag | int flag | param id
        cmd = 0x80 | 0x40 | _PARAM_DOAANGLE_ID
        try:
            data = self._usb_device.ctrl_transfer(
                _USB_REQUEST_TYPE_READ,
                _USB_REQUEST,
                cmd,
                0,
                8,
                8000,
            )
            if len(data) >= 4:
                # Response: int32 LE value + int32 LE max (we want first)
                angle = struct.unpack_from("<i", bytes(data))[0]
                if 0 <= angle < 360:
                    return angle
            return None
        except Exception:
            logger.debug("Failed to read DOA", exc_info=True)
            return None

    # ── Health ─────────────────────────────────────────────────────

    async def health_check(self) -> HealthStatus:
        """Check microphone health by verifying device connectivity."""
        if not self._started:
            return HealthStatus.STOPPED

        # Check if audio stream is still active
        if self._stream is None or not self._stream.active:
            return HealthStatus.ERROR

        # Check if USB device is still connected (for LEDs/DOA)
        if self._usb_device is None and _HAS_PYUSB:
            # USB gone but audio still works -> degraded
            return HealthStatus.DEGRADED

        return HealthStatus.OK

    # ── Properties ─────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Whether the ReSpeaker is connected and visible to sounddevice."""
        if not _HAS_SOUNDDEVICE:
            return False
        return self._find_device_index() is not None

    @property
    def sample_rate(self) -> int:
        """Configured sample rate in Hz."""
        return self._sample_rate

    @property
    def chunk_frames(self) -> int:
        """Number of audio frames per chunk."""
        return self._chunk_frames

    @property
    def consumer_count(self) -> int:
        """Number of registered audio consumers."""
        return len(self._consumers)

    # ── Internal helpers ───────────────────────────────────────────

    def _find_device_index(self) -> int | None:
        """Find the sounddevice index for the ReSpeaker by name."""
        if not _HAS_SOUNDDEVICE:
            return None

        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if (
                    self._device_name.lower() in dev["name"].lower()
                    and dev["max_input_channels"] >= self._capture_channels
                ):
                    return i
        except Exception:
            logger.debug("Error querying audio devices", exc_info=True)
        return None

    def _init_usb(self) -> None:
        """Initialize the USB vendor interface for LEDs and DOA.

        Non-fatal: if pyusb is unavailable or the device is not found,
        LED and DOA methods become no-ops.
        """
        if not _HAS_PYUSB:
            logger.info("pyusb not available; LED ring and DOA disabled")
            return

        try:
            self._usb_device = usb.core.find(
                idVendor=_RESPEAKER_VENDOR_ID, idProduct=_RESPEAKER_PRODUCT_ID
            )
            if self._usb_device is None:
                logger.warning(
                    "ReSpeaker USB device not found (VID=%04x PID=%04x); "
                    "LED ring and DOA disabled",
                    _RESPEAKER_VENDOR_ID,
                    _RESPEAKER_PRODUCT_ID,
                )
                return
            logger.info(
                "ReSpeaker USB interface initialized for LEDs + DOA "
                "(LED writes and DOA require a working udev rule; "
                "see scripts/setup.sh)"
            )
        except Exception:
            logger.warning("Failed to initialize ReSpeaker USB interface", exc_info=True)
            self._usb_device = None
