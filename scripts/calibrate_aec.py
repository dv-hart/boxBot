#!/usr/bin/env python3
"""Calibrate the AEC reference delay for boxBot — and verify it works.

Two modes:

**Calibration (default):** plays a chirp via the speaker with the AEC
reference path *disabled* so XMOS doesn't try to cancel anything,
captures it via the ReSpeaker mic, cross-correlates to measure the
HDMI playback latency, and writes the result to
``data/calibration/aec_delay_samples.json``. The Speaker module reads
that file on startup and pre-pads the AEC reference stream so the
XMOS chip sees the reference at the same wall-clock instant the mic
captures the speaker output.

**Validation (``--validate``):** plays the same chirp but with the AEC
reference path *enabled* (using the saved calibration). Captures and
reports residual energy on channel 0. If alignment is good, channel 0
should show a dramatically lower correlation peak than the calibration
baseline because the XMOS chip is now subtracting the reference
signal cleanly. Use this to confirm the calibration actually works.

Run from the repo root on the device::

    python3 scripts/calibrate_aec.py            # measure + save
    python3 scripts/calibrate_aec.py --validate # test with AEC enabled
    python3 scripts/calibrate_aec.py --dry-run  # measure but don't write
    python3 scripts/calibrate_aec.py --volume 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the project package importable regardless of how the script is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from boxbot.core.paths import CALIBRATION_DIR  # noqa: E402
from boxbot.hardware.speaker import Speaker  # noqa: E402

try:
    import sounddevice as sd  # noqa: E402
except ImportError as e:
    raise SystemExit(
        "sounddevice is required. Install with: pip install sounddevice"
    ) from e


logger = logging.getLogger("calibrate_aec")


# Calibration signal parameters.
SAMPLE_RATE = 16000        # mic capture rate
CHIRP_DURATION_S = 1.5
CHIRP_F0 = 200.0
CHIRP_F1 = 6000.0
SILENCE_BEFORE_S = 0.5     # quiet window inside the playback buffer
SILENCE_AFTER_S = 1.5
PLAY_LEAD_TIME_S = 0.3     # gap between starting capture and starting playback


def _make_chirp(volume: float) -> np.ndarray:
    """Generate a Hann-windowed linear chirp, int16 mono at SAMPLE_RATE."""
    n = int(CHIRP_DURATION_S * SAMPLE_RATE)
    t = np.linspace(0.0, CHIRP_DURATION_S, n, endpoint=False)
    k = (CHIRP_F1 - CHIRP_F0) / CHIRP_DURATION_S
    phase = 2.0 * np.pi * (CHIRP_F0 * t + 0.5 * k * t * t)
    window = np.hanning(n)
    sig = np.sin(phase) * window * float(volume)
    return (sig * 32767.0).astype(np.int16)


def _make_noise(volume: float, duration_s: float = 3.0) -> np.ndarray:
    """Generate band-limited white noise, int16 mono at SAMPLE_RATE.

    More speech-like than a chirp, so the XMOS AEC's voice-activity-
    gated adaptive filter will actually engage. Used by --validate to
    make the cancellation comparison meaningful.
    """
    n = int(duration_s * SAMPLE_RATE)
    rng = np.random.default_rng(seed=42)  # reproducible across runs
    raw = rng.standard_normal(n).astype(np.float32)
    # Band-limit ~200 Hz – 6 kHz with a simple FIR via FFT shaping.
    spec = np.fft.rfft(raw)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    spec[(freqs < 200.0) | (freqs > 6000.0)] = 0
    band = np.fft.irfft(spec, n).astype(np.float32)
    band /= max(np.max(np.abs(band)), 1e-9)
    # Brief fade in/out so the speaker doesn't click.
    fade = int(0.03 * SAMPLE_RATE)
    if fade * 2 < n:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        band[:fade] *= ramp
        band[-fade:] *= ramp[::-1]
    return (band * float(volume) * 32767.0).astype(np.int16)


def _find_respeaker_input_index() -> tuple[int, int]:
    """Return (device_index, max_input_channels) for the ReSpeaker mic."""
    devices = sd.query_devices()
    if isinstance(devices, dict):
        devices = [devices]
    for i, dev in enumerate(devices):
        name = dev.get("name", "").lower()
        max_in = dev.get("max_input_channels", 0)
        if "respeaker" in name and max_in >= 1:
            return i, max_in
    raise RuntimeError(
        "ReSpeaker input device not found. Available devices:\n"
        + "\n".join(
            f"  [{i}] {d.get('name')} (in={d.get('max_input_channels')})"
            for i, d in enumerate(devices)
        )
    )


def _read_current_padding() -> int:
    """Read the saved AEC delay (in samples) from disk, 0 if absent."""
    path = CALIBRATION_DIR / "aec_delay_samples.json"
    try:
        return int(json.loads(path.read_text()).get("delay_samples", 0))
    except Exception:
        return 0


def _cross_correlate_lag(
    captured: np.ndarray, chirp: np.ndarray
) -> tuple[int, float]:
    """Return (best_lag_samples, peak_value) for `chirp` inside `captured`.

    `best_lag_samples` is the index in `captured` where the chirp's
    leading sample best aligns. Peak value is the (signed) correlation
    at that lag, useful for quality scoring across channels.
    """
    cap_f = captured.astype(np.float32) / 32768.0
    chp_f = chirp.astype(np.float32) / 32768.0
    cap_f -= cap_f.mean()
    chp_f -= chp_f.mean()
    if cap_f.size < chp_f.size:
        return 0, 0.0
    corr = np.correlate(cap_f, chp_f, mode="valid")
    peak_idx = int(np.argmax(np.abs(corr)))
    return peak_idx, float(corr[peak_idx])


async def _run_chirp_capture(
    chirp: np.ndarray,
    full: np.ndarray,
    *,
    speaker_kwargs: dict,
    mic_idx: int,
    n_channels: int,
) -> np.ndarray:
    """Open the speaker with the given kwargs, play `full`, capture mics.

    Returns the captured audio as a (frames, n_channels) int16 ndarray.
    """
    capture_buffer: list[np.ndarray] = []

    def _audio_cb(indata, frames, time_info, status):  # type: ignore[no-untyped-def]
        if status:
            logger.debug("capture status: %s", status)
        capture_buffer.append(indata.copy())

    speaker = Speaker(**speaker_kwargs)
    await speaker.start()
    try:
        stream = sd.InputStream(
            device=mic_idx,
            channels=n_channels,
            samplerate=SAMPLE_RATE,
            dtype="int16",
            callback=_audio_cb,
        )
        stream.start()
        try:
            await asyncio.sleep(PLAY_LEAD_TIME_S)
            await speaker.play(full.tobytes(), SAMPLE_RATE, 1)
            await asyncio.sleep(0.4)
        finally:
            stream.stop()
            stream.close()
    finally:
        await speaker.stop()

    if not capture_buffer:
        return np.zeros((0, n_channels), dtype=np.int16)
    return np.concatenate(capture_buffer)


async def _calibrate(args: argparse.Namespace) -> int:
    # In validate mode use band-limited noise instead of a chirp: it
    # looks much more like speech to the XMOS AEC's voice-activity-gated
    # adaptive filter, which is what triggers actual cancellation. The
    # chirp gives a cleaner correlation peak for raw timing alignment
    # but the filter rarely engages on a sweeping pure tone.
    if args.validate:
        chirp = _make_noise(args.volume, duration_s=3.0)
    else:
        chirp = _make_chirp(args.volume)
    pre = np.zeros(int(SILENCE_BEFORE_S * SAMPLE_RATE), dtype=np.int16)
    post = np.zeros(int(SILENCE_AFTER_S * SAMPLE_RATE), dtype=np.int16)
    full = np.concatenate([pre, chirp, post])

    mic_idx, n_channels = _find_respeaker_input_index()
    logger.info(
        "ReSpeaker input: index=%d, channels=%d", mic_idx, n_channels,
    )

    # Build Speaker with the project's hardware config for device_name
    # and gain, but explicitly disable the AEC reference path. We want to
    # measure the HDMI-only path's latency without XMOS attempting to
    # subtract anything.
    speaker_device = "boxbot_speaker"
    speaker_gain_db = 6.0
    speaker_volume = 1.0
    speaker_rate = 24000
    try:
        from boxbot.core.config import load_config

        cfg = load_config(str(_REPO_ROOT / "config" / "config.yaml"))
        speaker_device = cfg.hardware.speaker.device_name
        speaker_gain_db = getattr(cfg.hardware.speaker, "gain_db", speaker_gain_db)
        speaker_volume = getattr(
            cfg.hardware.speaker, "default_volume", speaker_volume,
        )
        speaker_rate = getattr(cfg.hardware.speaker, "sample_rate", speaker_rate)
        logger.info(
            "Using speaker config: device=%s gain=%.1fdB vol=%.2f rate=%d",
            speaker_device, speaker_gain_db, speaker_volume, speaker_rate,
        )
    except Exception as e:
        logger.warning(
            "Could not load config (%s); using defaults device=%s",
            e, speaker_device,
        )

    # Both calibration and validation modes run with AEC reference ENABLED.
    # The ReSpeaker XVF3000 surfaces the reference signal it actually
    # received as channel 5 of the input stream (the "AEC reference
    # loopback"). Measuring chirp lag on channel 5 vs the speaker-captured
    # chirp on channel 0 directly gives us how much the reference is
    # mis-aligned, in real samples. This is more accurate than measuring
    # only HDMI latency (which is what the previous version did) because
    # the AEC reference USB path also has its own ~100 ms of latency.
    aec_dev: str | None = "ReSpeaker"
    try:
        from boxbot.core.config import load_config as _lc

        _cfg2 = _lc(str(_REPO_ROOT / "config" / "config.yaml"))
        aec_dev = getattr(
            _cfg2.hardware.speaker, "aec_reference_device", aec_dev,
        )
    except Exception:
        pass

    mode = "VALIDATE" if args.validate else "CALIBRATE"
    current_padding = _read_current_padding()
    logger.info(
        "%s mode: starting with saved AEC pad = %d samples (%.1f ms)",
        mode, current_padding, current_padding / SAMPLE_RATE * 1000.0,
    )

    common_kwargs = dict(
        device_name=speaker_device,
        sample_rate=speaker_rate,
        default_volume=speaker_volume,
        gain_db=speaker_gain_db,
    )

    # Validate mode runs an extra "AEC OFF" baseline pass first, so we
    # can attribute any reduction in channel 0 to the AEC subtraction
    # (rather than to noise variation, beamformer gain changes, etc.).
    baseline_ch0_peak: float | None = None
    if args.validate:
        logger.info("─── pass 1/2: AEC OFF baseline ───")
        baseline = await _run_chirp_capture(
            chirp, full,
            speaker_kwargs={
                **common_kwargs,
                "aec_reference_device": None,
                "aec_required": False,
            },
            mic_idx=mic_idx, n_channels=n_channels,
        )
        if baseline.size == 0:
            logger.error("No audio captured during baseline pass")
            return 1
        _, baseline_ch0_peak = _cross_correlate_lag(baseline[:, 0], chirp)
        logger.info(
            "Baseline channel 0 peak (AEC OFF): %.3f", baseline_ch0_peak,
        )
        await asyncio.sleep(0.5)
        logger.info("─── pass 2/2: AEC ON with current calibration ───")

    captured = await _run_chirp_capture(
        chirp, full,
        speaker_kwargs={
            **common_kwargs,
            "aec_reference_device": aec_dev,
            "aec_required": True,
        },
        mic_idx=mic_idx, n_channels=n_channels,
    )
    if captured.size == 0:
        logger.error("No audio captured")
        return 1
    logger.info(
        "Captured %d frames across %d channel(s)",
        captured.shape[0], captured.shape[1],
    )

    # Per-channel correlation snapshot.
    per_channel: list[tuple[int, int, float]] = []
    for ch in range(captured.shape[1]):
        lag, peak = _cross_correlate_lag(captured[:, ch], chirp)
        logger.info(
            "  ch %d: peak=%.3f at lag=%d samples (%.1f ms)",
            ch, peak, lag, lag / SAMPLE_RATE * 1000.0,
        )
        per_channel.append((ch, lag, peak))

    # On the ReSpeaker XVF3000 with AEC engaged:
    #  - channel 0 is the AEC-processed/beamformed output
    #  - channel 5 is the AEC reference loopback (what XMOS *received*)
    #  - channels 1-4 are raw mics (post-microphone, pre-AEC subtraction)
    #
    # For alignment we compare the reference (ch5) to one of the raw mic
    # channels (the strongest of 1-4), since the raw mics give us a clean
    # picture of when the speaker output reached the array — not muddled
    # by any partial cancellation.
    raw_mic_choice = max(per_channel[1:5], key=lambda r: abs(r[2]))
    raw_ch, raw_lag, raw_peak = raw_mic_choice
    ref_lag, ref_peak = (
        per_channel[5][1], per_channel[5][2]
    ) if captured.shape[1] >= 6 else (0, 0.0)

    # Channel 0 (AEC-processed) — primary cancellation indicator.
    ch0_lag, ch0_peak = per_channel[0][1], per_channel[0][2]

    logger.info(
        "Raw mic (ch %d): chirp at %.1f ms, peak=%.3f",
        raw_ch, raw_lag / SAMPLE_RATE * 1000.0, raw_peak,
    )
    logger.info(
        "AEC ref loopback (ch 5): chirp at %.1f ms, peak=%.3f",
        ref_lag / SAMPLE_RATE * 1000.0, ref_peak,
    )
    logger.info(
        "AEC-processed (ch 0): chirp at %.1f ms, peak=%.3f",
        ch0_lag / SAMPLE_RATE * 1000.0, ch0_peak,
    )

    if abs(ref_peak) < 1.0:
        logger.error(
            "Channel 5 (AEC reference loopback) is silent — the XMOS "
            "chip isn't receiving any reference signal. Verify the "
            "ReSpeaker is plugged in, that the AEC reference output "
            "stream actually opened (check Speaker startup log), and "
            "that channel 5 on this firmware is the loopback (some "
            "ReSpeaker firmware revisions surface the reference on a "
            "different channel)."
        )
        return 1

    # Misalignment = how much later the speaker-captured signal arrived
    # at the mic compared to when XMOS received the reference. Positive
    # means we need MORE pre-pad on the AEC reference path.
    misalign_samples = raw_lag - ref_lag
    misalign_ms = misalign_samples / SAMPLE_RATE * 1000.0
    logger.info(
        "Misalignment (raw mic − reference): %d samples (%.1f ms)",
        misalign_samples, misalign_ms,
    )

    new_padding = max(0, current_padding + misalign_samples)
    logger.info(
        "Current pad %d → new pad %d (Δ=%+d samples, %+.1f ms)",
        current_padding, new_padding,
        new_padding - current_padding,
        (new_padding - current_padding) / SAMPLE_RATE * 1000.0,
    )

    # Effectiveness reporting only makes sense when we have an AEC-OFF
    # baseline on the same channel. That's only true in --validate mode.
    if args.validate and baseline_ch0_peak is not None:
        if abs(baseline_ch0_peak) < 1e-6:
            logger.warning(
                "Baseline channel 0 peak ~0 — couldn't measure attenuation."
            )
        else:
            ratio = abs(ch0_peak) / abs(baseline_ch0_peak)
            db = 20.0 * np.log10(ratio) if ratio > 0 else float("-inf")
            logger.info(
                "AEC effectiveness: ch0 OFF=%.3f → ON=%.3f "
                "(ratio %.3f, %.1f dB attenuation)",
                baseline_ch0_peak, ch0_peak, ratio, -db,
            )
            if ratio < 0.3:
                logger.info("✓ AEC is cancelling well (>10 dB).")
            elif ratio < 0.7:
                logger.info(
                    "~ Partial cancellation (3-10 dB). Re-run "
                    "`calibrate_aec.py` (no --validate) to tune the "
                    "delay; misalignment delta should shrink toward 0."
                )
            else:
                logger.warning(
                    "✗ Barely any cancellation (<3 dB). Likely the "
                    "alignment delta is still off, or the XMOS adaptive "
                    "filter hasn't engaged. Try a longer or more "
                    "speech-like signal, or recalibrate."
                )

    if args.validate:
        logger.info(
            "VALIDATE: not writing. Run without --validate to save the "
            "adjusted padding (%d samples).", new_padding,
        )
        return 0

    out = {
        "delay_samples": new_padding,
        "delay_ms": new_padding / SAMPLE_RATE * 1000.0,
        "sample_rate": SAMPLE_RATE,
        "raw_mic_channel": raw_ch,
        "raw_mic_lag_samples": raw_lag,
        "ref_loopback_lag_samples": ref_lag,
        "misalignment_samples": misalign_samples,
        "previous_pad_samples": current_padding,
        "channel0_peak": ch0_peak,
        "raw_mic_peak": raw_peak,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "chirp": {
            "duration_s": CHIRP_DURATION_S,
            "f0_hz": CHIRP_F0,
            "f1_hz": CHIRP_F1,
            "volume": args.volume,
        },
    }

    if args.dry_run:
        logger.info("DRY RUN — would write:\n%s", json.dumps(out, indent=2))
        return 0

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CALIBRATION_DIR / "aec_delay_samples.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    logger.info("Wrote calibration → %s", out_path)
    if abs(misalign_samples) > 50:
        logger.info(
            "Misalignment was non-trivial — re-run %s to confirm "
            "convergence.", Path(__file__).name,
        )
    logger.info(
        "Restart boxbot for the new delay to take effect "
        "(it is read once at Speaker.start()).",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate the AEC reference delay for boxBot.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run the measurement but don't write the result file.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help=(
            "Test mode: play the chirp WITH AEC reference enabled and "
            "the saved calibration applied. Reports channel 0 residual "
            "vs the calibration baseline. Does not modify the saved "
            "calibration."
        ),
    )
    parser.add_argument(
        "--volume", type=float, default=0.7,
        help="Chirp amplitude scale, 0.0–1.0 (default: 0.7).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not (0.0 < args.volume <= 1.0):
        parser.error("--volume must be in (0.0, 1.0]")

    return asyncio.run(_calibrate(args))


if __name__ == "__main__":
    raise SystemExit(main())
