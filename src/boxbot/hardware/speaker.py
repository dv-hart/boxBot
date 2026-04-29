"""Speaker output via persistent sounddevice OutputStreams.

The primary stream feeds HDMI audio (the audible output). An optional
secondary stream feeds the **AEC reference channel** on the ReSpeaker
USB playback device — this is what the XMOS chip subtracts from the mic
input so BB doesn't hear itself when speaking. Without this reference,
TTS playback gets captured by the mic, transcribed, and fed back as a
fake user turn.

Each stream is opened once at ``start()`` and closed only at ``stop()``.
A dedicated writer thread per stream continuously feeds it: when the
playback queue has speech PCM it writes that, otherwise it writes silence.
This keeps the audio link in its active state at all times, which:

  * eliminates the ~100 ms cold-open cutoff that eats the first phoneme
  * eliminates the pop/click caused by the HDMI codec muting on link close
  * makes barge-in instant (just drop queued chunks; silence resumes)
  * keeps the AEC reference timing aligned with what the speaker emits

Hardware: Waveshare 8 ohm 5W speaker via HDMI audio. Optional AEC
reference path: ReSpeaker 4-Mic Array USB playback (XMOS XVF3000).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from boxbot.core.paths import CALIBRATION_DIR
from boxbot.hardware.base import (
    HardwareInitFatal,
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

# AEC reference path runs at the rate the XMOS chip natively expects.
# Block size is the AEC-rate equivalent of one HDMI block so the two
# writer loops tick in roughly the same cadence.
_AEC_TARGET_RATE = 16000
_AEC_BLOCK_FRAMES = max(
    1, int(_BLOCK_FRAMES * _AEC_TARGET_RATE / _HDMI_TARGET_RATE)
)
# ReSpeaker USB playback is 2-channel (left/right). The XMOS firmware
# uses both as the AEC reference; we duplicate mono to stereo.
_AEC_CHANNELS = 2


def _list_output_device_names() -> list[str]:
    """Return a flat list of output-capable device names (best effort)."""
    if not _SD_AVAILABLE:
        return []
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    if isinstance(devices, dict):
        devices = [devices]
    return [
        d.get("name", "")
        for d in devices
        if d.get("max_output_channels", 0) > 0
    ]


def _load_calibrated_aec_delay() -> int:
    """Read the AEC reference delay (in AEC-rate samples) from disk.

    The calibration file is written by ``scripts/calibrate_aec.py``. It
    records the measured HDMI playback latency at the mic-capture rate
    (typically 16 kHz, matching ``_AEC_TARGET_RATE``). If the file is
    missing, malformed, or specifies a different sample rate, return 0
    and let AEC reference run uncompensated.
    """
    path = CALIBRATION_DIR / "aec_delay_samples.json"
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return 0
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read AEC calibration at %s: %s", path, e)
        return 0

    delay = data.get("delay_samples")
    sr = data.get("sample_rate")
    if not isinstance(delay, int) or delay < 0:
        logger.warning(
            "AEC calibration at %s has invalid delay_samples=%r; ignoring",
            path, delay,
        )
        return 0
    if sr != _AEC_TARGET_RATE:
        # Rescale: delay seconds is invariant; samples scales with rate.
        try:
            scaled = int(round(delay * _AEC_TARGET_RATE / float(sr)))
        except (TypeError, ZeroDivisionError):
            logger.warning(
                "AEC calibration at %s has unusable sample_rate=%r; ignoring",
                path, sr,
            )
            return 0
        logger.info(
            "AEC calibration is at %d Hz; rescaling delay %d → %d samples "
            "for the %d Hz reference stream",
            sr, delay, scaled, _AEC_TARGET_RATE,
        )
        delay = scaled
    return delay


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
        aec_reference_device: str | None = "ReSpeaker",
        aec_required: bool = True,
        aec_discovery_retries: int = 5,
        aec_discovery_retry_delay: float = 0.2,
    ) -> None:
        super().__init__()
        if config is not None:
            device_name = config.device_name
            sample_rate = config.sample_rate
            default_volume = config.default_volume
            gain_db = getattr(config, "gain_db", gain_db)
            aec_reference_device = getattr(
                config, "aec_reference_device", aec_reference_device
            )
            aec_required = getattr(config, "aec_required", aec_required)
            aec_discovery_retries = getattr(
                config, "aec_discovery_retries", aec_discovery_retries
            )
            aec_discovery_retry_delay = getattr(
                config, "aec_discovery_retry_delay", aec_discovery_retry_delay
            )
        self._device_name = device_name
        # The configured ``sample_rate`` is the *expected* input rate that
        # callers will pass to play()/play_stream(). Everything is resampled
        # to 48 kHz internally before reaching the stream.
        self._source_rate = sample_rate
        self._volume = max(0.0, min(1.0, default_volume))
        self._gain_db = gain_db
        self._aec_reference_device = aec_reference_device
        self._aec_required = aec_required
        self._aec_discovery_retries = max(1, aec_discovery_retries)
        self._aec_discovery_retry_delay = max(0.0, aec_discovery_retry_delay)

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

        # AEC reference path (optional). Same source PCM, different stream
        # at 16 kHz stereo aimed at the ReSpeaker USB playback device. The
        # XMOS chip subtracts this signal from the mic capture.
        self._aec_device_index: int | None = None
        self._aec_stream: Any = None
        self._aec_queue: deque[np.ndarray] = deque()
        self._aec_silence_block = np.zeros(
            (_AEC_BLOCK_FRAMES, _AEC_CHANNELS), dtype=np.int16
        )
        self._aec_writer_thread: threading.Thread | None = None
        # Number of AEC-rate frames to prepend to every chunk on the AEC
        # reference queue. Loaded from data/calibration/aec_delay_samples.json
        # at start(). Compensates for the difference between HDMI playback
        # latency (deep buffer + DAC + analog) and the low-latency USB AEC
        # reference path so the XMOS chip sees the reference signal at the
        # same time the mic captures the actual speaker output.
        self._aec_delay_samples: int = 0

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

        # HDMI latency: 200 ms target. This is *deliberately* deeper than
        # the natural AEC reference path latency (~100 ms on this
        # hardware) so that we can pad-delay the reference signal up to
        # match HDMI emission timing — the only knob we have is "delay
        # the reference further", so HDMI must be the slower of the two.
        # Cold-open artifacts (~100 ms first-phoneme cutoff) are
        # prevented by the persistent writer thread feeding silence at
        # all times, so we no longer rely on the buffer for that.
        self._stream = sd.OutputStream(
            samplerate=_HDMI_TARGET_RATE,
            channels=1,
            dtype="int16",
            device=self._device_index,
            blocksize=_BLOCK_FRAMES,
            latency=0.20,
        )
        self._stream.start()

        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="speaker-writer", daemon=True
        )
        self._writer_thread.start()

        # Optional AEC reference path. Failure here must not take the
        # speaker down — HDMI-only playback still works, just without
        # echo cancellation (BB hears itself).
        self._open_aec_reference_stream()

        self._started = True
        await self._emit_health(HealthStatus.OK)
        logger.info(
            "Speaker started: device=%s (index=%d) stream_rate=%d "
            "source_rate=%d volume=%.2f gain=%.1fdB aec_ref=%s",
            self._device_name,
            self._device_index,
            _HDMI_TARGET_RATE,
            self._source_rate,
            self._volume,
            self._gain_db,
            "on" if self._aec_stream is not None else "off",
        )

    def _open_aec_reference_stream(self) -> None:
        """Open the secondary stream that feeds the XMOS AEC reference.

        Behaviour:

        - If ``aec_reference_device`` is ``None``, AEC is explicitly
          disabled — log + return. Caller has opted in to BB hearing
          itself (e.g. dev box, external echo canceller).
        - Otherwise, try to find the device with retries (USB
          enumeration on cold boot is sometimes slow enough that the
          first ``query_devices()`` doesn't list the ReSpeaker output).
        - If found, open the stream. If the stream open itself fails,
          that's a hardware problem, not a discovery one — escalate
          based on ``aec_required``.
        - If not found after all retries:
            * ``aec_required=True`` (default): raise
              :class:`HardwareInitFatal`. Boxbot refuses to start; the
              alternative is silently feeding BB's TTS back as user
              input, which derails every conversation.
            * ``aec_required=False``: log warning and continue with
              HDMI-only output.
        """
        if self._aec_reference_device is None:
            logger.info(
                "AEC reference disabled (aec_reference_device=None) — "
                "BB will hear its own voice unless an external echo "
                "canceller is in the loop."
            )
            return

        # Retry the device lookup. Each retry forces a fresh PortAudio
        # device enumeration so we pick up devices that USB hadn't
        # finished announcing on the first probe.
        last_seen_devices: list[str] = []
        for attempt in range(1, self._aec_discovery_retries + 1):
            try:
                self._aec_device_index = self._find_aec_reference_device(
                    self._aec_reference_device
                )
            except Exception:
                logger.exception(
                    "Failed to query devices for AEC reference (attempt %d/%d)",
                    attempt, self._aec_discovery_retries,
                )
                self._aec_device_index = None
            if self._aec_device_index is not None:
                break
            last_seen_devices = _list_output_device_names()
            if attempt < self._aec_discovery_retries:
                logger.info(
                    "AEC reference '%s' not in device list (attempt %d/%d). "
                    "Visible: %s. Re-querying after %.0f ms.",
                    self._aec_reference_device,
                    attempt,
                    self._aec_discovery_retries,
                    last_seen_devices,
                    self._aec_discovery_retry_delay * 1000,
                )
                _force_portaudio_reinit()
                time.sleep(self._aec_discovery_retry_delay)

        if self._aec_device_index is None:
            msg = (
                f"AEC reference device '{self._aec_reference_device}' "
                f"not found after {self._aec_discovery_retries} attempts. "
                f"Visible output devices: {last_seen_devices}"
            )
            if self._aec_required:
                raise HardwareInitFatal(
                    msg
                    + ". Set hardware.speaker.aec_required=false to boot "
                    "without AEC (BB will hear itself). On the boxBot "
                    "reference hardware, ensure the ReSpeaker USB array "
                    "is plugged in and recognised by `aplay -l`."
                )
            logger.warning(
                "%s — continuing without AEC because aec_required=false. "
                "BB will hear itself.",
                msg,
            )
            return

        try:
            self._aec_stream = sd.OutputStream(
                samplerate=_AEC_TARGET_RATE,
                channels=_AEC_CHANNELS,
                dtype="int16",
                device=self._aec_device_index,
                blocksize=_AEC_BLOCK_FRAMES,
                latency="low",
            )
            self._aec_stream.start()
        except Exception as e:
            self._aec_stream = None
            self._aec_device_index = None
            msg = (
                f"Found AEC reference device '{self._aec_reference_device}' "
                f"but failed to open stream: {e}"
            )
            if self._aec_required:
                raise HardwareInitFatal(msg) from e
            logger.warning(
                "%s — continuing without AEC because aec_required=false.",
                msg,
            )
            return

        # Load calibrated playback-latency delay (if any) so we can
        # offset each enqueued chunk on the reference path to match
        # HDMI emission timing.
        self._aec_delay_samples = _load_calibrated_aec_delay()

        self._aec_writer_thread = threading.Thread(
            target=self._aec_writer_loop,
            name="speaker-aec-writer",
            daemon=True,
        )
        self._aec_writer_thread.start()
        logger.info(
            "AEC reference stream open: device='%s' (index=%d) "
            "rate=%d ch=%d block=%d delay=%d samples (%.1f ms)",
            self._aec_reference_device,
            self._aec_device_index,
            _AEC_TARGET_RATE,
            _AEC_CHANNELS,
            _AEC_BLOCK_FRAMES,
            self._aec_delay_samples,
            self._aec_delay_samples / _AEC_TARGET_RATE * 1000.0,
        )

    async def stop(self) -> None:
        """Drop any queued audio and tear down both writers + streams."""
        self._shutdown.set()
        with self._queue_lock:
            self._queue.clear()
            self._aec_queue.clear()
        if self._fade_task is not None and not self._fade_task.done():
            self._fade_task.cancel()
            self._fade_task = None
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=1.0)
            self._writer_thread = None
        if self._aec_writer_thread is not None:
            self._aec_writer_thread.join(timeout=1.0)
            self._aec_writer_thread = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("Error closing OutputStream")
            self._stream = None
        if self._aec_stream is not None:
            try:
                self._aec_stream.stop()
                self._aec_stream.close()
            except Exception:
                logger.exception("Error closing AEC reference OutputStream")
            self._aec_stream = None
        self._device_index = None
        self._aec_device_index = None
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

    def _find_aec_reference_device(self, target: str) -> int | None:
        """Find an output device matching ``target`` for the AEC reference.

        Unlike ``_find_device`` (which prefers ALSA "plug" wrappers over
        raw "hw:" devices for the *audible* output), the AEC reference
        path is happiest pushing int16 through a ``plughw:`` device that
        ALSA converts on the fly to whatever format the XMOS chip
        actually wants (S24_3LE on this hardware). So: prefer ``plughw``
        first, then anything else, then raw ``hw:`` last.
        """
        if not _SD_AVAILABLE:
            return None
        devices = sd.query_devices()
        if isinstance(devices, dict):
            devices = [devices]
        target_lower = target.lower()
        plughw_match: int | None = None
        plain_match: int | None = None
        raw_hw_match: int | None = None
        for i, dev in enumerate(devices):
            if dev.get("max_output_channels", 0) <= 0:
                continue
            name_lower = dev.get("name", "").lower()
            if target_lower not in name_lower:
                continue
            if "plughw" in name_lower:
                if plughw_match is None:
                    plughw_match = i
            elif "hw:" in name_lower or "(hw:" in name_lower:
                if raw_hw_match is None:
                    raw_hw_match = i
            else:
                if plain_match is None:
                    plain_match = i
        if plughw_match is not None:
            return plughw_match
        if plain_match is not None:
            return plain_match
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

    def _aec_writer_loop(self) -> None:
        """Pump blocks from the AEC queue into the secondary stream;
        fill with silence when the queue is empty.

        Independent of HDMI drain signalling — callers await the HDMI
        side because that's what the listener actually hears. The AEC
        stream only needs to keep timing aligned with HDMI; whether
        it's fully drained when the caller returns is irrelevant.
        """
        while not self._shutdown.is_set():
            block: np.ndarray | None = None
            with self._queue_lock:
                if self._aec_queue:
                    block = self._aec_queue.popleft()
            if block is None:
                block = self._aec_silence_block
            try:
                self._aec_stream.write(block)
            except Exception:
                logger.exception(
                    "AEC reference stream.write failed"
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

        Splits one chunk into HDMI-rate blocks (audible playback) and,
        if the AEC reference stream is up, AEC-rate stereo blocks. Both
        queues are appended together so the writer threads see the same
        audio at roughly the same time — important for echo cancellation,
        which compares the reference to the captured mic signal in
        time-aligned windows.

        Blocks are split at their respective block-frame boundaries but
        the trailing sub-block is **not** padded — ``stream.write()``
        accepts any size, and padding with silence would insert an
        audible chop at every chunk boundary.
        """
        scaled = self._apply_volume(samples)

        if channels > 1:
            # Average multi-channel input down to mono.
            scaled = (
                scaled.reshape(-1, channels).mean(axis=1).astype(np.int16)
            )

        # Mono source at the original rate, post gain/volume.
        mono_source = scaled

        if sample_rate != _HDMI_TARGET_RATE:
            hdmi_samples = _resample_int16(
                mono_source, sample_rate, _HDMI_TARGET_RATE
            )
        else:
            hdmi_samples = mono_source

        if hdmi_samples.size == 0:
            return

        shaped = hdmi_samples.reshape(-1, 1)
        total = shaped.shape[0]
        hdmi_blocks: list[np.ndarray] = []
        for offset in range(0, total, _BLOCK_FRAMES):
            end = min(offset + _BLOCK_FRAMES, total)
            hdmi_blocks.append(shaped[offset:end].copy())

        aec_blocks: list[np.ndarray] = []
        if self._aec_stream is not None:
            if sample_rate != _AEC_TARGET_RATE:
                aec_mono = _resample_int16(
                    mono_source, sample_rate, _AEC_TARGET_RATE
                )
            else:
                aec_mono = mono_source
            if aec_mono.size > 0:
                # Duplicate mono → stereo (XMOS uses both channels as
                # the reference signal).
                aec_stereo = np.repeat(
                    aec_mono.reshape(-1, 1), _AEC_CHANNELS, axis=1
                )
                a_total = aec_stereo.shape[0]
                for offset in range(0, a_total, _AEC_BLOCK_FRAMES):
                    end = min(offset + _AEC_BLOCK_FRAMES, a_total)
                    aec_blocks.append(aec_stereo[offset:end].copy())

        # Prepend calibrated silence to delay the AEC reference path so it
        # arrives at the XMOS chip in lockstep with what the mic captures
        # from the speaker. Without this, the reference signal arrives
        # tens of ms early (HDMI buffer is much deeper than the USB AEC
        # ref stream) and the adaptive filter can't subtract cleanly.
        delay_blocks: list[np.ndarray] = []
        if aec_blocks and self._aec_delay_samples > 0:
            remaining = self._aec_delay_samples
            while remaining > 0:
                n = min(_AEC_BLOCK_FRAMES, remaining)
                if n == _AEC_BLOCK_FRAMES:
                    delay_blocks.append(self._aec_silence_block.copy())
                else:
                    delay_blocks.append(
                        np.zeros((n, _AEC_CHANNELS), dtype=np.int16)
                    )
                remaining -= n

        self._mark_busy()
        with self._queue_lock:
            self._queue.extend(hdmi_blocks)
            if aec_blocks:
                if delay_blocks:
                    self._aec_queue.extend(delay_blocks)
                self._aec_queue.extend(aec_blocks)

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
        """Barge-in: drop any queued PCM. Writer resumes silence on its own.

        Drains both the audible (HDMI) and AEC reference queues so the
        XMOS chip stops being told "BB is talking" the moment we cut TTS.
        """
        with self._queue_lock:
            self._queue.clear()
            self._aec_queue.clear()
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


def _force_portaudio_reinit() -> None:
    """Tear down and re-initialize PortAudio so ``query_devices`` picks
    up devices that arrived (or got hidden) since first init.

    PortAudio caches its device list at initialization time. On the
    boxBot reference hardware we've seen the ReSpeaker drop off the
    output-device list between two restarts even though the kernel
    sees it consistently — likely a USB enumeration race when the box
    is hot-rebooted. ``_terminate`` + ``_initialize`` is the only
    public-ish hook sounddevice exposes for forcing a re-enumeration.
    """
    if not _SD_AVAILABLE:
        return
    try:
        sd._terminate()  # type: ignore[attr-defined]
        sd._initialize()  # type: ignore[attr-defined]
    except Exception:
        # Don't let a private-API change kill the boot path; the next
        # query_devices() will still return *something*, possibly the
        # same cached list. The retry loop will then exhaust naturally.
        logger.exception("PortAudio re-init failed")
