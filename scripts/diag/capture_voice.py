#!/usr/bin/env python3
"""Capture 6-channel wake-word-gated utterances for voice-ID diagnosis.

Records the FULL 6-channel ReSpeaker stream for each spoken utterance so
the speaker-embedding pipeline can be analysed offline (channel choice,
diarization fragmentation, clip-length sensitivity, noise robustness,
speaker separability) WITHOUT re-recording. See
``scripts/diag/analyze_voice.py`` for the analyses.

Faithful to production where it matters: the same Silero VAD
(``AudioCapture`` + ``VoiceActivityDetector``) drives utterance
boundaries, and the same OpenWakeWord model + threshold gates each
trial. The one deliberate divergence is that we keep all six channels
instead of extracting ``output_channel`` — which is exactly the choice
under test.

Run this with the main boxbot process STOPPED (the ReSpeaker capture
device is exclusive). One condition (quiet / dishwasher / etc.) per
invocation; the script walks a fixed command list and prompts before
each trial.

Usage (on the Pi, in the project venv):
    ./scripts/restart-boxbot.sh stop      # or however you stop it
    python3 scripts/diag/capture_voice.py \
        --speaker jacob --condition dishwasher_close

    # second speaker, same conditions:
    python3 scripts/diag/capture_voice.py \
        --speaker sarah  --condition quiet_close

Output: ``data/voice_diag/<speaker>/<condition>/trialNN_<command>.wav``
(6-channel int16) plus a ``.json`` sidecar of metadata per utterance.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Make the boxbot package importable when run from the project root.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import sounddevice as sd  # noqa: E402

from boxbot.communication.audio_capture import AudioCapture, Utterance  # noqa: E402
from boxbot.communication.vad import VoiceActivityDetector  # noqa: E402
from boxbot.core.config import get_config  # noqa: E402
from boxbot.hardware.base import AudioChunk  # noqa: E402

SAMPLE_RATE = 16000
CAPTURE_CHANNELS = 6
CHUNK_MS = 64  # match the HAL's chunk size
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_MS / 1000)
RING_SECONDS = 90  # how much 6ch audio to keep for slicing
PAD_SECONDS = 0.2  # context kept either side of the VAD boundary

# Default command list — short, natural, household-realistic utterances.
# Three per pass (one per treatment per person, per the experiment plan);
# re-run the same condition to stack more samples. Override with
# --commands-file (one command per line).
DEFAULT_COMMANDS = [
    "hey jarvis what's on my calendar for today",
    "hey jarvis pull up the picture of the kids at the pumpkin patch",
    "hey jarvis what's the weather looking like this afternoon",
]


@dataclass
class TrialMeta:
    speaker: str
    condition: str
    command: str
    trial_index: int
    duration_s: float
    sample_rate: int
    channels: int
    n_frames: int
    wake_fired: bool
    wake_confidence: float
    captured_at: float


def _find_device(name: str) -> int:
    """Return the sounddevice index for a >=6ch input matching ``name``."""
    for idx, dev in enumerate(sd.query_devices()):
        if name.lower() in dev["name"].lower() and dev["max_input_channels"] >= CAPTURE_CHANNELS:
            return idx
    raise RuntimeError(
        f"No input device matching '{name}' with >= {CAPTURE_CHANNELS} channels. "
        f"Available: {[d['name'] for d in sd.query_devices()]}"
    )


class SixChannelMic:
    """Minimal 6ch capture that mimics the HAL's mono fan-out.

    Dispatches channel-0 mono ``AudioChunk``s to registered consumers
    (so the production VAD / wake-word run exactly as they would live),
    while retaining every 6-channel frame in a timestamped ring buffer
    for later slicing.
    """

    def __init__(self, device_index: int, loop: asyncio.AbstractEventLoop) -> None:
        self._device_index = device_index
        self._loop = loop
        self._consumers: list = []
        self._stream = None
        # ring of (monotonic_ts, ndarray[(frames, 6)] int16)
        self._ring: deque = deque(maxlen=int(RING_SECONDS * 1000 / CHUNK_MS))

    def add_consumer(self, callback, name: str = "") -> int:
        self._consumers.append(callback)
        return len(self._consumers) - 1

    def remove_consumer(self, handle: int) -> bool:  # pragma: no cover - parity
        return True

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CAPTURE_CHANNELS,
            dtype="int16",
            device=self._device_index,
            blocksize=CHUNK_FRAMES,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def slice(self, t_start: float, t_end: float) -> np.ndarray:
        """Concatenate ring-buffer frames within [t_start-pad, t_end+pad]."""
        lo, hi = t_start - PAD_SECONDS, t_end + PAD_SECONDS
        frames = [f for (ts, f) in list(self._ring) if lo <= ts <= hi]
        if not frames:
            return np.zeros((0, CAPTURE_CHANNELS), dtype=np.int16)
        return np.concatenate(frames, axis=0)

    def _callback(self, indata, frames, time_info, status) -> None:
        now = time.monotonic()
        if status:
            print(f"[audio] stream status: {status}", file=sys.stderr)
        # Retain the full 6ch frame (copy — sounddevice reuses the buffer).
        self._ring.append((now, indata.copy()))
        if not self._consumers:
            return
        mono = indata[:, 0].copy()
        chunk = AudioChunk(
            data=mono.tobytes(),
            timestamp=now,
            sample_rate=SAMPLE_RATE,
            channels=1,
            frames=frames,
        )
        for cb in self._consumers:
            self._loop.call_soon_threadsafe(self._loop.create_task, cb(chunk))


class WakeGate:
    """Runs OpenWakeWord on ch0 mono, recording the latest detection.

    Reused faithfully (same model + threshold as production); we only
    need to know *whether* the wake word fired during an utterance, so
    this records the last-fire timestamp/confidence rather than driving
    an event bus.
    """

    def __init__(self, word: str, threshold: float, model_path: str | None) -> None:
        import openwakeword

        from boxbot.communication.wake_word import _resolve_builtin_model

        self._threshold = threshold
        paths = [model_path] if model_path else [_resolve_builtin_model(word)]
        self._model = openwakeword.Model(wakeword_model_paths=paths)
        self.last_fire_ts: float = 0.0
        self.last_conf: float = 0.0

    async def __call__(self, chunk: AudioChunk) -> None:
        audio = np.frombuffer(chunk.data, dtype=np.int16)
        preds = self._model.predict(audio)
        conf = max(preds.values()) if preds else 0.0
        if conf > self._threshold:
            self.last_fire_ts = chunk.timestamp
            self.last_conf = float(conf)
            self._model.reset()


def _write_wav(path: Path, audio_6ch: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(CAPTURE_CHANNELS)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio_6ch.astype("<i2").tobytes())


async def run(args: argparse.Namespace) -> int:
    cfg = get_config().voice
    loop = asyncio.get_running_loop()

    if args.commands_file:
        commands = [
            line.strip()
            for line in Path(args.commands_file).read_text().splitlines()
            if line.strip()
        ]
    else:
        commands = DEFAULT_COMMANDS

    device_index = _find_device(args.device)
    print(f"[capture] device='{args.device}' index={device_index}")

    mic = SixChannelMic(device_index, loop)
    vad = VoiceActivityDetector(cfg.vad)
    capture = AudioCapture(vad, cfg.turn_detection)

    wake = WakeGate(
        word=cfg.wake_word.word,
        threshold=cfg.wake_word.confidence_threshold,
        model_path=cfg.wake_word.model_path,
    )
    mic.add_consumer(wake, name="wake")

    # One utterance per trial; an Event hands the finalized utterance back
    # to the trial loop.
    pending: dict = {"utt": None, "event": asyncio.Event()}

    async def on_utterance(utt: Utterance) -> None:
        pending["utt"] = utt
        pending["event"].set()

    capture.set_utterance_callback(on_utterance)
    await capture.start(mic)
    mic.start()

    out_root = Path(args.out_dir) / args.speaker / args.condition
    run_id = time.strftime("%H%M%S")  # distinguishes repeated passes
    print(
        f"[capture] speaker={args.speaker} condition={args.condition} "
        f"run={run_id} -> {out_root}\n"
    )

    saved = 0
    try:
        for i, command in enumerate(commands, start=1):
            slug = command.replace("hey jarvis ", "").replace(" ", "_")[:40]
            print(f"--- Trial {i}/{len(commands)} [{args.condition}] ---")
            print(f'    Say: "{command}"')
            await loop.run_in_executor(None, input, "    Press Enter when ready, then speak... ")

            capture.reset()
            pending["utt"] = None
            pending["event"].clear()
            try:
                await asyncio.wait_for(pending["event"].wait(), timeout=args.timeout)
            except asyncio.TimeoutError:
                print("    [!] No utterance detected before timeout — skipping.\n")
                continue

            utt: Utterance = pending["utt"]
            audio_6ch = mic.slice(utt.timestamp_start, utt.timestamp_end)
            wake_fired = utt.timestamp_start <= wake.last_fire_ts <= utt.timestamp_end + PAD_SECONDS

            stem = f"trial{i:02d}_{slug}_{run_id}"
            wav_path = out_root / f"{stem}.wav"
            _write_wav(wav_path, audio_6ch)

            meta = TrialMeta(
                speaker=args.speaker,
                condition=args.condition,
                command=command,
                trial_index=i,
                duration_s=round(utt.duration, 3),
                sample_rate=SAMPLE_RATE,
                channels=CAPTURE_CHANNELS,
                n_frames=int(audio_6ch.shape[0]),
                wake_fired=bool(wake_fired),
                wake_confidence=round(wake.last_conf, 3) if wake_fired else 0.0,
                captured_at=time.time(),
            )
            wav_path.with_suffix(".json").write_text(json.dumps(asdict(meta), indent=2))
            saved += 1
            warn = "" if wake_fired else "  (wake word did NOT fire)"
            print(
                f"    saved {wav_path.name}  {utt.duration:.2f}s  "
                f"{audio_6ch.shape[0]} frames{warn}\n"
            )
    finally:
        await capture.stop()
        mic.stop()

    print(f"[capture] done — {saved}/{len(commands)} utterances saved under {out_root}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--speaker", required=True, help="Speaker label, e.g. jacob")
    p.add_argument(
        "--condition",
        required=True,
        help="Acoustic condition label, e.g. quiet_close, dishwasher_far",
    )
    p.add_argument("--device", default="ReSpeaker", help="Input device name substring")
    p.add_argument("--out-dir", default="data/voice_diag", help="Output root")
    p.add_argument("--commands-file", default=None, help="File of commands, one per line")
    p.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for an utterance per trial",
    )
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
