#!/usr/bin/env python3
"""Summarise the wake-word audio archive (``data/wake_audio/``).

The live detector writes a clip + JSON sidecar for every hit and every
near miss (see ``boxbot/communication/wake_audio_log.py``). The voice
adapter stamps hits with what followed: ``transcript``, ``no_transcript``
or ``no_utterance``. This script turns that archive into numbers:

  summary   — counts, score histogram (hits vs near misses), hit
              outcomes, level stats. Run anywhere with numpy.
  review    — the near misses and no-utterance hits a human should
              listen to and label. Edit the sidecar's ``label`` to
              "positive" (a real attempt) or "negative". Labelled clips
              are the training set for an openwakeword custom verifier.
  rescore   — replay every clip through openwakeword (same frame size
              as live) and report peak score per clip, so a different
              threshold / model / verifier can be evaluated on exactly
              the audio the box heard. Pi only (needs openwakeword).

    python3 scripts/diag/analyze_wake.py summary
    python3 scripts/diag/analyze_wake.py review
    python3 scripts/diag/analyze_wake.py rescore --threshold 0.5 [--model-path X]

Clips are mono 16 kHz, pre+post roll around the detector's peak.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def load(root: Path) -> list[dict]:
    clips: list[dict] = []
    for sub in ("hits", "near_misses"):
        d = root / sub
        if not d.is_dir():
            continue
        for sidecar in sorted(d.glob("*.json")):
            try:
                meta = json.loads(sidecar.read_text())
            except Exception:
                continue
            meta["_json"] = sidecar
            meta["_wav"] = sidecar.with_suffix(".wav")
            clips.append(meta)
    clips.sort(key=lambda m: m.get("epoch_ms", 0))
    return clips


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm, sr


def histogram(scores: list[float]) -> str:
    if not scores:
        return "  (none)"
    counts, _ = np.histogram(scores, bins=BINS)
    lines = []
    for lo, hi, c in zip(BINS[:-1], BINS[1:], counts):
        bar = "#" * int(c)
        lines.append(f"  {lo:.1f}–{min(hi, 1.0):.1f}  {c:4d} {bar}")
    return "\n".join(lines)


def cmd_summary(clips: list[dict]) -> int:
    hits = [c for c in clips if c.get("kind") == "hit"]
    near = [c for c in clips if c.get("kind") == "near_miss"]
    print(f"clips: {len(clips)}  hits: {len(hits)}  near misses: {len(near)}")
    if clips:
        print(f"range: {clips[0].get('local_time')} → {clips[-1].get('local_time')}")
        thr = sorted({c.get('threshold') for c in clips})
        print(f"threshold(s) in archive: {thr}")
    print("\nhit scores:")
    print(histogram([c["score"] for c in hits]))
    print("\nnear-miss peaks:")
    print(histogram([c["score"] for c in near]))

    outcomes = Counter(c.get("outcome") or "unlabelled" for c in hits)
    print("\nhit outcomes:")
    for k, v in outcomes.most_common():
        print(f"  {k:15s} {v}")
    spk = Counter()
    for c in hits:
        for s in c.get("speakers") or []:
            spk[s] += 1
    if spk:
        print("\nfirst-utterance speakers after a hit:")
        for k, v in spk.most_common():
            print(f"  {k:15s} {v}")

    labels = Counter(c.get("label") or "unlabelled" for c in clips)
    print("\nhuman labels:")
    for k, v in labels.most_common():
        print(f"  {k:15s} {v}")

    def lvl(cs):
        xs = [c["rms_dbfs"] for c in cs if isinstance(c.get("rms_dbfs"), (int, float))]
        return f"median {np.median(xs):.1f} dBFS (n={len(xs)})" if xs else "n/a"
    print(f"\nlevel: hits {lvl(hits)} | near misses {lvl(near)}")

    if hits:
        false_accepts = [c for c in hits if c.get("outcome") == "no_utterance"]
        print(f"\nfalse-accept proxy (hit, nothing heard in grace): "
              f"{len(false_accepts)}/{len(hits)}")
    return 0


def cmd_review(clips: list[dict]) -> int:
    todo = [c for c in clips if c.get("label") is None and (
        c.get("kind") == "near_miss" or c.get("outcome") == "no_utterance")]
    print(f"{len(todo)} clips to listen to and label (edit \"label\" in the sidecar):\n")
    for c in todo:
        tag = "NEAR" if c.get("kind") == "near_miss" else "HIT "
        print(f"{tag} {c.get('local_time')}  score={c.get('score'):.2f}  "
              f"rms={c.get('rms_dbfs')}  {c['_wav'].relative_to(_ROOT) if c['_wav'].is_relative_to(_ROOT) else c['_wav']}")
    print("\nAlso check hits whose first transcript was clearly not addressed to BB:")
    for c in clips:
        if c.get("kind") == "hit" and c.get("outcome") == "transcript" and c.get("label") is None:
            print(f"HIT  {c.get('local_time')}  score={c.get('score'):.2f}  "
                  f"{c.get('speakers')}  {c.get('text', '')[:70]!r}")
    return 0


def cmd_rescore(clips: list[dict], threshold: float, model_path: str | None,
                verifier: str | None, verifier_threshold: float) -> int:
    import openwakeword  # Pi venv only

    from boxbot.communication.wake_word import _resolve_builtin_model
    from boxbot.core.config import get_config, load_config

    load_config()
    cfg = get_config().voice.wake_word
    path = model_path or cfg.model_path or _resolve_builtin_model(cfg.word)
    kwargs = {}
    if verifier:
        kwargs = {"custom_verifier_models": {Path(path).stem: verifier},
                  "custom_verifier_threshold": verifier_threshold}
    model = openwakeword.Model(wakeword_model_paths=[path], **kwargs)
    frame = 1280  # 80 ms, openwakeword's native frame

    rows = []
    for c in clips:
        if not c["_wav"].exists():
            continue
        pcm, _sr = read_wav(c["_wav"])
        model.reset()
        peak = 0.0
        for off in range(0, len(pcm) - frame + 1, frame):
            preds = model.predict(pcm[off:off + frame])
            peak = max(peak, max(preds.values(), default=0.0))
        rows.append((c, peak))

    print(f"model={Path(path).name} verifier={verifier or '-'} threshold={threshold}\n")
    agg = Counter()
    for c, peak in rows:
        would_fire = peak > threshold
        key = (c.get("kind"), c.get("label") or c.get("outcome") or "?", would_fire)
        agg[key] += 1
    print(f"{'kind':10s} {'label/outcome':16s} {'fires@thr':>9s}  n")
    for (kind, lab, fires), n in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        print(f"{kind:10s} {lab:16s} {str(fires):>9s}  {n}")

    pos = [p for c, p in rows if c.get("label") == "positive"]
    neg = [p for c, p in rows if c.get("label") == "negative"]
    if pos or neg:
        print("\nlabelled clips, recall / false-accept by threshold:")
        for t in (0.3, 0.4, 0.5, 0.6, 0.7):
            r = sum(p > t for p in pos) / len(pos) if pos else float("nan")
            fa = sum(p > t for p in neg)
            print(f"  thr {t:.1f}: recall {r:5.1%} (n={len(pos)})  false accepts {fa}/{len(neg)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["summary", "review", "rescore"])
    ap.add_argument("--root", default=str(_ROOT / "data" / "wake_audio"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--verifier", default=None, help="custom verifier .pkl")
    ap.add_argument("--verifier-threshold", type=float, default=0.1)
    a = ap.parse_args()
    clips = load(Path(a.root))
    if a.cmd == "summary":
        return cmd_summary(clips)
    if a.cmd == "review":
        return cmd_review(clips)
    return cmd_rescore(clips, a.threshold, a.model_path, a.verifier, a.verifier_threshold)


if __name__ == "__main__":
    raise SystemExit(main())
