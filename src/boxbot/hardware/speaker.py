"""Speaker output via a persistent sounddevice OutputStream.

The stream is opened once at ``start()`` and closed only at ``stop()``.
A dedicated writer thread continuously feeds the stream: when the playback
queue has speech PCM it writes that, otherwise it writes silence. This
keeps the HDMI audio link in its active state at all times, which:

  * eliminates the ~100 ms cold-open cutoff that eats the first phoneme
  * eliminates the pop/click caused by the HDMI codec muting on link close
  * makes barge-in instant (just drop queued chunks; silence resumes)

Hardware: Waveshare 8 ohm 5W speaker via HDMI audio.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from boxbot.hardware.base import (
    HardwareModule,
    HardwareUnavailableError,
    HealthStatus,
)

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd  # type: ignore[import-untyped]

    _SD_AVAILABLE = True
except ImportError:
    sd = None  # type: ignore[assignment]
    _SD_AVAILABLE = False


# Sample rates HDMI audio reliably supports on the Pi 5 (vc4-hdmi).
# Anything below 32 kHz gets rejected with PaErrorCode -9994, so the
# persistent stream always runs at 48 kHz and we resample inputs.
_HDMI_TARGET_RATE = 48000
_BLOCK_FRAMES = 1024  # ~21 ms at 48 kHz — unit of feed to the stream


def _resample_int16(
    samples: np.ndarray, src_rate: int, dst_rate: int
) -> np.ndarray:
    """Linear-interpolation resample for int16 mono PCM.

    Dependency-free. Adequate for TTS speech where the source rate
    (24 kHz) is well above speech bandwidth.
    """
    if src_rate == dst_rate or samples.size == 0:
        return samples
    duration = samples.size / src_rate
    n_out = int(duration * dst_rate)
    src_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    resampled = np.interp(dst_x, src_x, samples.astype(np.float32))
    return resampled.astype(np.int16)


class Speaker(HardwareModule):
    """Speaker output via a persistent sounddevice OutputStream.

    Caller-facing API is the same as the prior non-persistent speaker:
    ``play`` for complete buffers, ``play_stream`` for async iterators
    (e.g. ElevenLabs chunks), ``stop_playback`` for barge-in. The
    difference is that a single OutputStream stays open for the module's
    lifetime, with a writer thread producing silence when nothing else
    is queued.
    """

    name = "speaker"

    # Leave 1 dBFS headroom so the DAC never sees inter-sample peaks at
    # the rail after resampling.
    _CEILING_DBFS = -1.0

    def __init__(
        self,
        config: Any | None = None,
        *,
        device_name: str = "boxbot_speaker",
        sample_rate: int = 24000,
        default_volume: float = 1.0,
        gain_db: float = 6.0,
    ) -> None:
        super().__init__()
        if config is not None:
            device_name = config.device_name
            sample_rate = config.sample_rate
            default_volume = config.default_volume
            gain_db = getattr(config, "gain_db", gain_db)
        self._device_name = device_name
        # The configured ``sample_rate`` is the *expected* input rate that
        # callers will pass to play()/play_stream(). Everything is resampled
        # to 48 kHz internally before reaching the stream.
        self._source_rate = sample_rate
        self._volume = max(0.0, min(1.0, default_volume))
        self._gain_db = gain_db

        self._device_index: int | None = None
        self._stream: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Playback queue drained by the writer thread. Each entry is a
        # block-sized int16 ndarray shaped (frames, 1).
        self._queue: deque[np.ndarray] = deque()
        self._queue_lock = threading.Lock()
        self._silence_block = np.zeros((_BLOCK_FRAMES, 1), dtype=np.int16)

        # Writer thread control.
        self._writer_thread: threading.Thread | None = None
        self._shutdown = threading.Event()

        # Async-side "queue is drained" signal so callers of play()/
        # play_stream() can await actual completion.
        self._drained_event: asyncio.Event | None = None

        # Coordinate concurrent play() / play_stream() calls.
        self._playback_lock: asyncio.Lock | None = None

        # Fade task reference (barge-in graduated fade).
        self._fade_task: asyncio.Task[None] | None = None

    # ── Lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        if not _SD_AVAILABLE:
            await self._emit_health(
                HealthStatus.ERROR, "sounddevice not installed"
            )
            raise HardwareUnavailableError(
                "sounddevice package not installed"
            )

        self._device_index = self._find_device()
        if self._device_index is None:
            await self._emit_health(
                HealthStatus.ERROR,
                f"device '{self._device_name}' not found",
            )
            raise HardwareUnavailableError(
                f"Speaker device '{self._device_name}' not found. "
                f"Available devices: {sd.query_devices()}"
            )

        self._loop = asyncio.get_running_loop()
        self._drained_event = asyncio.Event()
        self._drained_event.set()  # starts idle
        self._playback_lock = asyncio.Lock()
        self._shutdown.clear()

        self._stream = sd.OutputStream(
            samplerate=_HDMI_TARGET_RATE,
            channels=1,
            dtype="int16",
            device=self._device_index,
            blocksize=_BLOCK_FRAMES,
            latency="high",
        )
        self._stream.start()

        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="speaker-writer", daemon=True
        )
        self._writer_thread.start()

        self._started = True
        await self._emit_health(HealthStatus.OK)
        logger.info(
            "Speaker started: device=%s (index=%d) stream_rate=%d "
            "source_rate=%d volume=%.2f gain=%.1fdB",
            self._device_name,
            self._device_index,
            _HDMI_TARGET_RATE,
            self._source_rate,
            self._volume,
            self._gain_db,
        )

    async def stop(self) -> None:
        """Drop any queued audio and tear down the writer + stream."""
        self._shutdown.set()
        with self._queue_lock:
            self._queue.clear()
        if self._fade_task is not None and not self._fade_task.done():
            self._fade_task.cancel()
            self._fade_task = None
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=1.0)
            self._writer_thread = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("Error closing OutputStream")
            self._stream = None
        self._device_index = None
        self._started = False
        await self._emit_health(HealthStatus.STOPPED)

    # ── Device discovery ───────────────────────────────────────────

    def _find_device(self) -> int | None:
        """Find the output device index matching the configured name.

        Prefers ALSA "plug" devices (which do format/rate conversion) over
        raw ``hw:`` devices. On the Pi, the raw vc4-hdmi device rejects
        every sample format via PortAudio while the plug-wrapped ``hdmi``
        device accepts int16/int32/float32 at any rate.
        """
        if not _SD_AVAILABLE:
            return None
        try:
            devices = sd.query_devices()
        except Exception:
            logger.exception("Failed to query audio devices")
            return None

        if isinstance(devices, dict):
            devices = [devices]

        target = self._device_name.lower()
        substring_match: int | None = None
        raw_hw_match: int | None = None
        exact_match: int | None = None

        for i, dev in enumerate(devices):
            name = dev.get("name", "")
            name_lower = name.lower()
            if dev.get("max_output_channels", 0) <= 0:
                continue
            if target not in name_lower:
                continue
            if name_lower == target:
                exact_match = i
                break
            if "hw:" in name_lower or "(hw:" in name_lower:
                if raw_hw_match is None:
                    raw_hw_match = i
            else:
                if substring_match is None:
                    substring_match = i

        if exact_match is not None:
            return exact_match
        if substring_match is not None:
            return substring_match
        return raw_hw_match

    # ── Writer thread ──────────────────────────────────────────────

    def _writer_loop(self) -> None:
        """Pump blocks from the queue into the stream; fill with silence
        when the queue is empty.

        ``stream.write`` blocks until the DAC has room, so the silence
        loop does not spin the CPU.
        """
        drained_signalled = True  # starts in idle/drained state
        while not self._shutdown.is_set():
            block: np.ndarray | None = None
            with self._queue_lock:
                if self._queue:
                    block = self._queue.popleft()
            if block is not None:
                drained_signalled = False
                try:
                    self._stream.write(block)
                except Exception:
                    logger.exception("stream.write failed on speech block")
            else:
                if not drained_signalled and self._loop is not None:
                    # Queue just drained — tell async side so callers of
                    # play()/play_stream() can return.
                    self._loop.call_soon_threadsafe(self._mark_drained)
                    drained_signalled = True
                try:
                    self._stream.write(self._silence_block)
                except Exception:
                    logger.exception(
                        "stream.write failed on silence block"
                    )

    def _mark_drained(self) -> None:
        if self._drained_event is not None:
            self._drained_event.set()

    def _mark_busy(self) -> None:
        if self._drained_event is not None:
            self._drained_event.clear()

    # ── Playback ───────────────────────────────────────────────────

    async def play(
        self,
        audio_data: bytes,
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> None:
        """Play a complete audio buffer. Returns when the audio has drained."""
        if not self._started or self._playback_lock is None:
            logger.warning("Speaker not started, ignoring play()")
            return
        async with self._playback_lock:
            samples = np.frombuffer(audio_data, dtype=np.int16).copy()
            self._enqueue_samples(samples, sample_rate, channels)
            await self._await_drained()

    async def play_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> None:
        """Play audio from an async stream (for TTS).

        Each chunk is gain-limited, resampled to the stream rate, and
        pushed onto the writer thread's queue as it arrives. Returns
        when the last queued block has been written to the stream.
        """
        if not self._started or self._playback_lock is None:
            logger.warning("Speaker not started, ignoring play_stream()")
            return
        async with self._playback_lock:
            try:
                async for chunk in audio_stream:
                    if self._shutdown.is_set():
                        break
                    if not chunk:
                        continue
                    samples = np.frombuffer(chunk, dtype=np.int16).copy()
                    if samples.size == 0:
                        continue
                    self._enqueue_samples(samples, sample_rate, channels)
                await self._await_drained()
            except asyncio.CancelledError:
                await self.stop_playback()
                raise

    def _enqueue_samples(
        self, samples: np.ndarray, sample_rate: int, channels: int
    ) -> None:
        """Apply gain/volume, downmix, resample, split into blocks, enqueue.

        Blocks are split at ``_BLOCK_FRAMES`` boundaries but the trailing
        sub-block is **not** padded — ``stream.write()`` accepts any size,
        and padding with silence would insert an audible chop at every
        chunk boundary.
        """
        scaled = self._apply_volume(samples)

        if channels > 1:
            # Average multi-channel input down to mono.
            scaled = (
                scaled.reshape(-1, channels).mean(axis=1).astype(np.int16)
            )

        if sample_rate != _HDMI_TARGET_RATE:
            scaled = _resample_int16(scaled, sample_rate, _HDMI_TARGET_RATE)

        if scaled.size == 0:
            return

        shaped = scaled.reshape(-1, 1)
        total = shaped.shape[0]
        blocks: list[np.ndarray] = []
        for offset in range(0, total, _BLOCK_FRAMES):
            end = min(offset + _BLOCK_FRAMES, total)
            blocks.append(shaped[offset:end].copy())

        self._mark_busy()
        with self._queue_lock:
            self._queue.extend(blocks)

    async def _await_drained(self) -> None:
        """Wait for the writer queue to drain, then for the stream's own
        buffer to finish draining (so the caller's await matches what
        the listener actually hears)."""
        if self._drained_event is None:
            return
        await self._drained_event.wait()
        latency = 0.05
        if self._stream is not None:
            try:
                latency = float(self._stream.latency)
            except Exception:
                pass
        await asyncio.sleep(latency + 0.02)

    async def stop_playback(self) -> None:
        """Barge-in: drop any queued PCM. Writer resumes silence on its own."""
        with self._queue_lock:
            self._queue.clear()
        self._mark_drained()

    # ── Volume ─────────────────────────────────────────────────────

    def set_volume(self, level: float) -> None:
        self._volume = max(0.0, min(1.0, level))
        logger.debug("Speaker volume set to %.2f", self._volume)

    def get_volume(self) -> float:
        return self._volume

    async def fade_volume(self, target: float, duration_ms: int) -> None:
        """Smoothly fade volume to a target level.

        Note: volume is applied at enqueue time, so a fade only affects
        audio queued *after* the fade starts. For a true audible fade of
        already-queued audio, call this before pushing new chunks.
        """
        target = max(0.0, min(1.0, target))
        if duration_ms <= 0:
            self.set_volume(target)
            return
        if self._fade_task is not None and not self._fade_task.done():
            self._fade_task.cancel()
        self._fade_task = asyncio.ensure_future(
            self._fade_loop(target, duration_ms)
        )
        await self._fade_task

    async def _fade_loop(self, target: float, duration_ms: int) -> None:
        steps = 30
        start_volume = self._volume
        step_duration = duration_ms / steps / 1000.0
        for i in range(1, steps + 1):
            t = i / steps
            smooth_t = t * t * (3.0 - 2.0 * t)
            self._volume = start_volume + (target - start_volume) * smooth_t
            await asyncio.sleep(step_duration)
        self._volume = target

    def _apply_volume(self, samples: np.ndarray) -> np.ndarray:
        """Apply gain + soft limiter + volume to int16 PCM samples.

        Pipeline: samples * 10^(gain_db/20) → tanh soft-limit to ceiling
        → * volume. The soft-limit matters for TTS: ElevenLabs output
        peaks near 0 dBFS but speech has ~20 dB crest factor, so the
        RMS is 15–20 dB below peak. Raising gain then limiting brings
        perceived loudness up without clipping transient peaks.
        """
        if self._volume <= 0.0:
            return np.zeros_like(samples)

        x = samples.astype(np.float32)
        if self._gain_db != 0.0:
            g = float(10.0 ** (self._gain_db / 20.0))
            ceiling = 32767.0 * float(
                10.0 ** (self._CEILING_DBFS / 20.0)
            )
            x = ceiling * np.tanh(x * g / ceiling)

        if self._volume < 1.0:
            x = x * self._volume

        return x.clip(-32768, 32767).astype(np.int16)

    # ── Properties ─────────────────────────────────────────────────

    @property
    def is_playing(self) -> bool:
        """Whether real audio (not silence) is currently queued or playing."""
        with self._queue_lock:
            return len(self._queue) > 0

    @property
    def is_available(self) -> bool:
        if not _SD_AVAILABLE:
            return False
        return self._find_device() is not None

    async def health_check(self) -> HealthStatus:
        if not self._started:
            return HealthStatus.STOPPED
        if self._stream is None or self._device_index is None:
            return HealthStatus.ERROR
        if self._writer_thread is None or not self._writer_thread.is_alive():
            return HealthStatus.ERROR
        if self._find_device() is None:
            return HealthStatus.ERROR
        return HealthStatus.OK
