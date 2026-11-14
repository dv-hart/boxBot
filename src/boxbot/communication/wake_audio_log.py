"""Wake-word audio archive: clips around every hit and near miss.

The wake-word detector only ever logged the detections that cleared
the threshold, so misses were invisible and nothing could be re-scored
or retrained. This module keeps a short ring buffer of the mono audio
the detector hears and writes a WAV + JSON sidecar for:

- **hits** — score cleared ``confidence_threshold`` (the published wake)
- **near misses** — score rose above ``near_miss_floor`` but never
  cleared the threshold before decaying again

Each clip spans ``pre_seconds`` before the peak to ``post_seconds``
after it, so the wake word itself (and the start of any command) is in
the file. Sidecars carry the peak score, threshold, level, and — for
hits — the outcome the voice adapter reports afterwards (first
transcript, or no utterance in the grace window). That outcome is the
label: a hit followed by nothing is a false accept; a near miss the
owner recognises as a real attempt is a false reject. Both feed
``scripts/diag/analyze_wake.py`` and, later, a custom verifier.

Layout (``data/wake_audio/``, Pi-local, gitignored)::

    hits/20261114_083012_p0.71.wav
    hits/20261114_083012_p0.71.json
    near_misses/20261114_091500_p0.48.wav
    ...

Retention: files older than ``retention_days`` and anything past
``max_mb`` (oldest first) are pruned at start and every few hours.
Clips are ~3.5 s of 16 kHz mono (~110 KB), so a busy day costs a few MB.

File IO runs in the default executor so the audio fan-out never blocks
on the SD card.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import math
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from boxbot.core.paths import WAKE_AUDIO_DIR

logger = logging.getLogger(__name__)

HITS_SUBDIR = "hits"
NEAR_MISSES_SUBDIR = "near_misses"

# A near-miss episode closes once the score has sat below the floor for
# this long. Long enough to ride out the per-frame jitter of one spoken
# word, short enough that two attempts a few seconds apart stay separate.
_EPISODE_DECAY_SECONDS = 0.6
_PRUNE_INTERVAL_SECONDS = 6 * 3600


def _level_dbfs(pcm: np.ndarray) -> tuple[float, float]:
    """(rms_dbfs, peak_dbfs) of int16 PCM. -120 for silence."""
    if pcm.size == 0:
        return -120.0, -120.0
    x = pcm.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    to_db = lambda v: 20.0 * math.log10(v) if v > 0 else -120.0  # noqa: E731
    return round(to_db(rms), 1), round(to_db(peak), 1)


@dataclass
class _PendingClip:
    kind: str  # "hit" | "near_miss"
    center_ts: float  # monotonic timestamp of the peak / detection
    score: float
    due_ts: float  # write once the ring holds audio up to here
    wall_time: float  # time.time() at the peak, for the filename
    annotations: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None  # set once written (sidecar path)


class WakeAudioLog:
    """Ring buffer + clip writer for wake-word hits and near misses."""

    def __init__(
        self,
        *,
        threshold: float,
        near_miss_floor: float = 0.3,
        pre_seconds: float = 2.5,
        post_seconds: float = 1.0,
        retention_days: int = 30,
        max_mb: int = 500,
        root: Path | None = None,
        model_name: str = "",
    ) -> None:
        self._threshold = float(threshold)
        self._floor = float(near_miss_floor)
        self._pre = float(pre_seconds)
        self._post = float(post_seconds)
        self._retention_days = int(retention_days)
        self._max_bytes = int(max_mb) * 1024 * 1024
        self._root = Path(root) if root is not None else WAKE_AUDIO_DIR
        self._model_name = model_name

        # Ring of (monotonic_ts_of_chunk_start, pcm bytes, frames). Sized
        # from the first chunk's duration; 64 ms chunks → ~70 entries.
        self._ring: deque[tuple[float, bytes, int]] = deque()
        self._ring_seconds = self._pre + self._post + 1.5
        self._sample_rate = 16000

        # Near-miss episode state.
        self._ep_active = False
        self._ep_peak = 0.0
        self._ep_peak_ts = 0.0
        self._ep_peak_wall = 0.0
        self._ep_last_above_ts = 0.0

        self._pending: list[_PendingClip] = []
        self._last_hit: _PendingClip | None = None
        self._inflight: set[asyncio.Future[Any]] = set()
        self._last_prune = 0.0

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def observe(self, chunk: Any, score: float | None) -> None:
        """Feed one mono chunk and the detector's score for it.

        ``score`` is None while the detector is in its post-hit debounce
        (no prediction ran); the audio still lands in the ring so the
        hit's post-roll is captured.
        """
        ts = float(chunk.timestamp)
        self._sample_rate = int(chunk.sample_rate)
        self._ring.append((ts, bytes(chunk.data), int(chunk.frames)))
        # Trim by time, not count, so the ring is right for any chunk size.
        cutoff = ts - self._ring_seconds
        while len(self._ring) > 1 and self._ring[0][0] < cutoff:
            self._ring.popleft()

        if score is not None:
            self._track_episode(ts, float(score))

        self._flush_due(ts)
        self._maybe_prune()

    def on_hit(self, ts: float, score: float) -> None:
        """A detection cleared the threshold at ``ts``; archive it."""
        clip = _PendingClip(
            kind="hit",
            center_ts=float(ts),
            score=float(score),
            due_ts=float(ts) + self._post,
            wall_time=time.time(),
        )
        self._pending.append(clip)
        self._last_hit = clip
        # The same word must not also be filed as a near miss.
        self._reset_episode()

    def annotate_last_hit(self, **fields: Any) -> None:
        """Attach outcome fields to the most recent hit's sidecar.

        Called by the voice adapter once it knows what followed the
        wake: the first transcript, or nothing within the grace window.
        Safe before or after the clip has been written.
        """
        clip = self._last_hit
        if clip is None:
            return
        clip.annotations.update(fields)
        if clip.path is not None:
            self._submit(self._rewrite_sidecar, clip)

    # ------------------------------------------------------------------
    # Episode tracking (near misses)
    # ------------------------------------------------------------------

    def _track_episode(self, ts: float, score: float) -> None:
        if score >= self._floor:
            if not self._ep_active:
                self._ep_active = True
                self._ep_peak = score
                self._ep_peak_ts = ts
                self._ep_peak_wall = time.time()
            elif score > self._ep_peak:
                self._ep_peak = score
                self._ep_peak_ts = ts
                self._ep_peak_wall = time.time()
            self._ep_last_above_ts = ts
            return
        if self._ep_active and (ts - self._ep_last_above_ts) >= _EPISODE_DECAY_SECONDS:
            # Episode over without a hit (on_hit() resets the episode, so
            # anything still active here never cleared the threshold).
            peak, peak_ts, wall = self._ep_peak, self._ep_peak_ts, self._ep_peak_wall
            self._reset_episode()
            if peak < self._threshold:
                logger.info(
                    "Wake word near miss (peak=%.3f, threshold=%.2f)",
                    peak, self._threshold,
                )
                self._pending.append(_PendingClip(
                    kind="near_miss",
                    center_ts=peak_ts,
                    score=peak,
                    due_ts=max(peak_ts + self._post, ts),
                    wall_time=wall,
                ))

    def _reset_episode(self) -> None:
        self._ep_active = False
        self._ep_peak = 0.0
        self._ep_peak_ts = 0.0
        self._ep_last_above_ts = 0.0

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _flush_due(self, now_ts: float) -> None:
        if not self._pending:
            return
        still: list[_PendingClip] = []
        for clip in self._pending:
            if now_ts >= clip.due_ts:
                pcm = self._slice(clip.center_ts - self._pre, clip.center_ts + self._post)
                self._submit(self._write_clip, clip, pcm)
            else:
                still.append(clip)
        self._pending = still

    def _slice(self, t0: float, t1: float) -> np.ndarray:
        parts = [data for (ts, data, _f) in self._ring if t0 <= ts <= t1]
        if not parts:
            return np.zeros(0, dtype=np.int16)
        return np.frombuffer(b"".join(parts), dtype=np.int16)

    def _write_clip(self, clip: _PendingClip, pcm: np.ndarray) -> None:
        subdir = HITS_SUBDIR if clip.kind == "hit" else NEAR_MISSES_SUBDIR
        out_dir = self._root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.fromtimestamp(clip.wall_time).strftime("%Y%m%d_%H%M%S")
        # Explicit string, not with_suffix(): "p0.47" has a "." in it.
        stem = f"{stamp}_p{clip.score:.2f}"
        wav_path = out_dir / f"{stem}.wav"
        n = 1
        while wav_path.exists():  # two clips in one second (double hit)
            n += 1
            wav_path = out_dir / f"{stem}_{n}.wav"
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self._sample_rate)
            w.writeframes(pcm.astype("<i2").tobytes())

        rms_db, peak_db = _level_dbfs(pcm)
        meta: dict[str, Any] = {
            "kind": clip.kind,
            "fired": clip.kind == "hit",
            "score": round(clip.score, 4),
            "threshold": self._threshold,
            "near_miss_floor": self._floor,
            "epoch_ms": int(clip.wall_time * 1000),
            "local_time": _dt.datetime.fromtimestamp(clip.wall_time).isoformat(timespec="seconds"),
            "pre_seconds": self._pre,
            "post_seconds": self._post,
            "sample_rate": self._sample_rate,
            "duration_s": round(pcm.size / self._sample_rate, 3),
            "rms_dbfs": rms_db,
            "peak_dbfs": peak_db,
            "model": self._model_name,
            # Filled by the voice adapter (hits) or a human (both):
            #   outcome: "transcript" | "no_transcript" | "no_utterance"
            #   label:   "positive" | "negative" | null
            "outcome": None,
            "label": None,
        }
        meta.update(clip.annotations)
        clip.path = wav_path.with_name(wav_path.stem + ".json")
        clip.path.write_text(json.dumps(meta, indent=2))

    def _rewrite_sidecar(self, clip: _PendingClip) -> None:
        if clip.path is None or not clip.path.exists():
            return
        try:
            meta = json.loads(clip.path.read_text())
        except Exception:
            meta = {}
        meta.update(clip.annotations)
        clip.path.write_text(json.dumps(meta, indent=2))

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    def _maybe_prune(self) -> None:
        now = time.monotonic()
        if now - self._last_prune < _PRUNE_INTERVAL_SECONDS:
            return
        self._last_prune = now
        self._submit(self.prune)

    def prune(self) -> int:
        """Delete clips past retention or over the size cap. Returns count."""
        files: list[tuple[float, int, Path]] = []
        for sub in (HITS_SUBDIR, NEAR_MISSES_SUBDIR):
            d = self._root / sub
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if p.suffix in (".wav", ".json") and p.is_file():
                    st = p.stat()
                    files.append((st.st_mtime, st.st_size, p))
        if not files:
            return 0
        files.sort()  # oldest first
        cutoff = time.time() - self._retention_days * 86400
        total = sum(sz for _m, sz, _p in files)
        removed = 0
        for mtime, size, path in files:
            if mtime < cutoff or total > self._max_bytes:
                try:
                    path.unlink()
                    removed += 1
                    total -= size
                except OSError:
                    logger.debug("prune: could not remove %s", path, exc_info=True)
        if removed:
            logger.info("Wake audio archive: pruned %d files", removed)
        return removed

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _submit(self, fn: Any, *args: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            fn(*args)  # no loop (tests / shutdown): run inline
            return
        fut = loop.run_in_executor(None, fn, *args)
        self._inflight.add(fut)

        def _done(f: asyncio.Future[Any]) -> None:
            self._inflight.discard(f)
            if not f.cancelled() and f.exception() is not None:
                logger.warning(
                    "Wake audio archive write failed: %s", f.exception()
                )

        fut.add_done_callback(_done)

    async def drain(self) -> None:
        """Wait for in-flight writes (tests, shutdown)."""
        if self._inflight:
            await asyncio.gather(*list(self._inflight), return_exceptions=True)
