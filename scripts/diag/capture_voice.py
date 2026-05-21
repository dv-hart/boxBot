#!/usr/bin/env python3
"""Continuous 6-channel utterance recorder for voice-ID diagnosis.

Runs as a hands-off daemon using the same gating as the live voice
adapter: OpenWakeWord arms capture, then the production Silero VAD
finalizes the utterance — and that utterance is submitted (all six
channels + a wall-clock-timestamped metadata sidecar). Speech without a
preceding wake word is ignored. No prompts, no per-trial typing.

Workflow:
    1. Stop boxbot (the capture device is exclusive).
    2. Start this recorder (typically in the background).
    3. Just talk — say the wake word + a command, naturally, under each
       acoustic condition. Jot down "time / who / condition" per batch.
    4. Stop the recorder. Hand the notes to Claude, who aligns each
       utterance's timestamp to a (speaker, condition) and runs
       ``analyze_voice.py``.

All six channels are kept (production extracts only ``output_channel=0``,
which is itself under test). Output:
    data/voice_diag/_inbox/utt_<epoch_ms>.wav   (6ch int16)
    data/voice_diag/_inbox/utt_<epoch_ms>.json  (timestamp, duration, …)

Usage (on the Pi, in the project venv, boxbot stopped):
    python3 scripts/diag/capture_voice.py
    # or in the background:
    nohup python3 scripts/diag/capture_voice.py > logs/voice_capture.log 2>&1 &
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import signal
import sys
import time
import wave
from collections import deque
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
ARM_WINDOW = 8.0  # seconds a wake word "arms" capture (covers "BB" ... pause ... command)


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
    (so the production VAD / wake-word run exactly as they would live)
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
    need to know *whether* the wake word fired during an utterance.
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

    device_index = _find_device(args.device)
    out_dir = Path(args.out_dir) / "_inbox"
    print(f"[capture] device='{args.device}' index={device_index}")
    print(f"[capture] writing utterances to {out_dir}")

    mic = SixChannelMic(device_index, loop)
    vad = VoiceActivityDetector(cfg.vad)
    capture = AudioCapture(vad, cfg.turn_detection)
    wake = WakeGate(
        word=cfg.wake_word.word,
        threshold=cfg.wake_word.confidence_threshold,
        model_path=cfg.wake_word.model_path,
    )
    mic.add_consumer(wake, name="wake")

    count = {"n": 0}

    async def on_utterance(utt: Utterance) -> None:
        # Wake-word gate — submit only utterances the wake word started,
        # exactly like the live voice adapter. An utterance counts if a
        # wake-word detection landed within it or in the arming window
        # just before it; stray speech / background talk is ignored.
        fired = wake.last_fire_ts
        armed = fired > 0 and (utt.timestamp_start - ARM_WINDOW) <= fired <= (utt.timestamp_end + PAD_SECONDS)
        clock = datetime.datetime.now().strftime("%H:%M:%S")
        if not armed:
            print(f"[{clock}]      (ignored {utt.duration:.2f}s — no wake word)", flush=True)
            return
        audio_6ch = mic.slice(utt.timestamp_start, utt.timestamp_end)
        if audio_6ch.shape[0] == 0:
            return
        now = time.time()
        epoch_ms = int(now * 1000)
        wake_conf = wake.last_conf
        wake.last_fire_ts = 0.0  # consume — a fresh wake is needed for the next clip
        wav_path = out_dir / f"utt_{epoch_ms}.wav"
        _write_wav(wav_path, audio_6ch)
        meta = {
            "epoch_ms": epoch_ms,
            "local_time": datetime.datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "duration_s": round(utt.duration, 3),
            "sample_rate": SAMPLE_RATE,
            "channels": CAPTURE_CHANNELS,
            "n_frames": int(audio_6ch.shape[0]),
            "wake_fired": True,
            "wake_confidence": round(wake_conf, 3),
            # speaker / condition filled in later by alignment from notes.
            "speaker": None,
            "condition": None,
        }
        wav_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        count["n"] += 1
        print(f"[{clock}] #{count['n']:>3} saved {wav_path.name}  "
              f"{utt.duration:.2f}s  wake={wake_conf:.2f}", flush=True)

    capture.set_utterance_callback(on_utterance)
    await capture.start(mic)
    mic.start()

    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    print("[capture] recording — speak naturally. Ctrl-C or SIGTERM to stop.\n", flush=True)
    await stop_event.wait()

    print(f"\n[capture] stopping — {count['n']} utterances saved to {out_dir}")
    await capture.stop()
    mic.stop()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--device", default="ReSpeaker", help="Input device name substring")
    p.add_argument("--out-dir", default="data/voice_diag", help="Output root")
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
