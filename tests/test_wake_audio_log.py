"""Wake-word audio archive: clips for hits and near misses."""

from __future__ import annotations

import json
import os
import time
import wave

import numpy as np
import pytest

from boxbot.communication.wake_audio_log import WakeAudioLog
from boxbot.hardware.base import AudioChunk

CHUNK = 0.064  # 64 ms chunks, like the HAL


def chunk(ts: float, value: int = 0, frames: int = 1024) -> AudioChunk:
    data = np.full(frames, value, dtype=np.int16).tobytes()
    return AudioChunk(data=data, timestamp=ts, sample_rate=16000, channels=1, frames=frames)


def _files(root, sub):
    d = root / sub
    return sorted(d.iterdir()) if d.is_dir() else []


def make_log(tmp_path, **kw):
    defaults = dict(threshold=0.5, near_miss_floor=0.3, pre_seconds=1.0,
                    post_seconds=0.5, root=tmp_path, model_name="t")
    defaults.update(kw)
    return WakeAudioLog(**defaults)


async def feed(log, t0, seconds, score=0.0, value=0):
    """Stream `seconds` of chunks starting at t0; returns the next timestamp."""
    t = t0
    n = int(round(seconds / CHUNK))
    for _ in range(n):
        log.observe(chunk(t, value), score)
        t += CHUNK
    return t


@pytest.mark.asyncio
async def test_hit_writes_clip_with_pre_and_post_roll(tmp_path):
    log = make_log(tmp_path)
    t = await feed(log, 100.0, 2.0, score=0.01, value=1000)
    log.on_hit(t, 0.83)
    # Nothing is written until the post-roll has streamed in.
    log.observe(chunk(t, 1000), None)
    await log.drain()
    assert _files(tmp_path, "hits") == []
    t = await feed(log, t + CHUNK, 0.7, score=None)
    await log.drain()

    files = _files(tmp_path, "hits")
    wavs = [f for f in files if f.suffix == ".wav"]
    sidecars = [f for f in files if f.suffix == ".json"]
    assert len(wavs) == 1 and len(sidecars) == 1
    assert "_p0.83" in wavs[0].name
    with wave.open(str(wavs[0])) as w:
        dur = w.getnframes() / w.getframerate()
    # pre 1.0 + post 0.5, within a chunk or two.
    assert 1.3 <= dur <= 1.7
    meta = json.loads(sidecars[0].read_text())
    assert meta["kind"] == "hit" and meta["fired"] is True
    assert meta["score"] == 0.83 and meta["threshold"] == 0.5
    assert meta["outcome"] is None
    assert meta["rms_dbfs"] > -40  # 1000/32768 ≈ -30 dBFS


@pytest.mark.asyncio
async def test_near_miss_written_after_score_decays(tmp_path):
    log = make_log(tmp_path)
    t = await feed(log, 50.0, 1.5, score=0.02)
    t = await feed(log, t, 0.3, score=0.42)  # rises above floor, under threshold
    t = await feed(log, t, 0.2, score=0.47)  # peak
    assert _files(tmp_path, "near_misses") == []
    t = await feed(log, t, 1.5, score=0.01)  # decay + post-roll
    await log.drain()

    files = _files(tmp_path, "near_misses")
    assert [f.suffix for f in files] == [".json", ".wav"]
    meta = json.loads(files[0].read_text())
    assert meta["kind"] == "near_miss" and meta["fired"] is False
    assert meta["score"] == 0.47
    assert "_p0.47" in files[1].name


@pytest.mark.asyncio
async def test_below_floor_writes_nothing(tmp_path):
    log = make_log(tmp_path)
    await feed(log, 10.0, 3.0, score=0.29)
    await log.drain()
    assert not (tmp_path / "near_misses").exists()
    assert not (tmp_path / "hits").exists()


@pytest.mark.asyncio
async def test_hit_is_not_also_filed_as_near_miss(tmp_path):
    log = make_log(tmp_path)
    t = await feed(log, 10.0, 1.5, score=0.02)
    t = await feed(log, t, 0.2, score=0.45)  # climbing…
    log.on_hit(t, 0.9)                        # …then clears the threshold
    t = await feed(log, t, 2.0, score=0.0)
    await log.drain()
    assert len(_files(tmp_path, "hits")) == 2
    assert _files(tmp_path, "near_misses") == []


@pytest.mark.asyncio
async def test_annotate_before_and_after_write(tmp_path):
    log = make_log(tmp_path)
    t = await feed(log, 10.0, 1.5)
    log.on_hit(t, 0.7)
    log.annotate_last_hit(outcome="no_utterance")  # before the clip exists
    t = await feed(log, t, 1.0, score=None)
    await log.drain()
    sidecar = [f for f in _files(tmp_path, "hits") if f.suffix == ".json"][0]
    assert json.loads(sidecar.read_text())["outcome"] == "no_utterance"

    log.annotate_last_hit(outcome="transcript", text="hey jarvis what time is it")
    await log.drain()
    meta = json.loads(sidecar.read_text())
    assert meta["outcome"] == "transcript"
    assert meta["text"].startswith("hey jarvis")


@pytest.mark.asyncio
async def test_annotate_without_hit_is_noop(tmp_path):
    log = make_log(tmp_path)
    log.annotate_last_hit(outcome="transcript")  # nothing to annotate
    await log.drain()
    assert not (tmp_path / "hits").exists()


def test_prune_by_age_and_size(tmp_path):
    log = make_log(tmp_path, retention_days=7, max_mb=1)
    hits = tmp_path / "hits"
    hits.mkdir()
    old = hits / "old.wav"
    old.write_bytes(b"x" * 10)
    os.utime(old, (time.time() - 10 * 86400,) * 2)
    big_a = hits / "a.wav"
    big_a.write_bytes(b"x" * 700_000)
    os.utime(big_a, (time.time() - 100,) * 2)
    big_b = hits / "b.wav"
    big_b.write_bytes(b"x" * 700_000)  # a + b > 1 MB → oldest (a) goes

    removed = log.prune()
    assert removed == 2
    assert not old.exists() and not big_a.exists() and big_b.exists()


# ---------------------------------------------------------------------------
# Detector integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detector_feeds_archive_and_annotates(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    import boxbot.communication.wake_word as ww
    from boxbot.core.config import WakeWordConfig

    fake_oww = MagicMock()
    monkeypatch.setattr(ww, "openwakeword", fake_oww)
    monkeypatch.setattr(ww, "_resolve_builtin_model", lambda name: "/fake.onnx")

    cfg = WakeWordConfig(confidence_threshold=0.5, record_audio=True)
    log = make_log(tmp_path, threshold=0.5)
    det = ww.WakeWordDetector(cfg, audio_log=log)
    await det.start(MagicMock())

    t = 10.0
    det._model.predict.return_value = {"hey_jarvis": 0.05}
    for _ in range(20):
        await det._on_audio_chunk(chunk(t)); t += CHUNK
    det._model.predict.return_value = {"hey_jarvis": 0.9}
    await det._on_audio_chunk(chunk(t)); t += CHUNK
    det.annotate_last_hit(outcome="transcript", text="what time is it")
    # Post-roll arrives during the debounce window (predict not called).
    det._suppress_until = float("inf")
    for _ in range(12):
        await det._on_audio_chunk(chunk(t)); t += CHUNK
    model = det._model
    await det.stop()

    sidecars = [f for f in _files(tmp_path, "hits") if f.suffix == ".json"]
    assert len(sidecars) == 1
    meta = json.loads(sidecars[0].read_text())
    assert meta["score"] == 0.9 and meta["outcome"] == "transcript"
    # 20 chunks at 0.05 + hit + 12 post chunks: predict ran 21 times.
    assert model.predict.call_count == 21
