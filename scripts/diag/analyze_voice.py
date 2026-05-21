#!/usr/bin/env python3
"""Analyse 6-channel utterance captures to diagnose voice-ID failure.

Consumes the WAV+JSON pairs produced by ``capture_voice.py`` and runs
five analyses, each mapped to a hypothesis about why same-speaker
cosine similarity is so low:

  1. Channel verification  — per-channel RMS / noise-floor / SNR.
     Which of the 6 ReSpeaker channels is actually beamformed/denoised,
     and is production's ``output_channel=0`` the right pick?
  2. Whole vs fragments    — embed the whole utterance vs the pyannote
     diarization fragments. Is diarization shredding single-speaker
     utterances into weak sub-second embeddings?
  3. Duration sweep        — same-speaker cosine vs clip length
     (1.0/1.5/2.0/3.0 s windows). Can we get a clean read on short clips?
  4. Noise impact          — same-speaker cosine quiet vs noisy, per
     channel. Does the beamformed channel survive the dishwasher?
  5. Separability          — within-speaker vs cross-speaker cosine
     distributions -> EER -> an honest speaker_threshold.

Run on the Pi (the only host with torch + pyannote installed):
    HF_TOKEN=... python3 scripts/diag/analyze_voice.py \
        --in-dir data/voice_diag --channel 0

Embeds with the SAME model + ``window="whole"`` as production
(``pyannote/wespeaker-voxceleb-resnet34-LM``) so numbers transfer
directly to the live pipeline.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from boxbot.core.config import get_config, load_config  # noqa: E402

load_config()  # reads config/config.yaml (same as production startup)

DURATION_BUCKETS = [1.0, 1.5, 2.0, 3.0]


# --------------------------------------------------------------------------
# IO + signal helpers
# --------------------------------------------------------------------------
def load_utterances(in_dir: Path) -> list[dict]:
    """Load every WAV+JSON pair under in_dir into utterance records."""
    utts = []
    for wav_path in sorted(in_dir.rglob("*.wav")):
        meta_path = wav_path.with_suffix(".json")
        if not meta_path.exists():
            print(f"[warn] no metadata for {wav_path}, skipping", file=sys.stderr)
            continue
        meta = json.loads(meta_path.read_text())
        if not meta.get("speaker") or not meta.get("condition"):
            continue  # unlabeled capture — align it to notes first
        with wave.open(str(wav_path), "rb") as w:
            nch = w.getnchannels()
            sr = w.getframerate()
            raw = w.readframes(w.getnframes())
        audio = np.frombuffer(raw, dtype="<i2").reshape(-1, nch).astype(np.float32) / 32768.0
        utts.append({"meta": meta, "audio": audio, "sr": sr, "path": wav_path})
    return utts


def frame_energy(x: np.ndarray, sr: int, win_ms: float = 25.0) -> np.ndarray:
    n = max(1, int(sr * win_ms / 1000))
    trimmed = x[: len(x) - len(x) % n]
    if len(trimmed) == 0:
        return np.array([np.sqrt(np.mean(x**2) + 1e-12)])
    frames = trimmed.reshape(-1, n)
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12)


def snr_db(x: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Return (rms, noise_floor_rms, snr_db) from energy percentiles."""
    e = frame_energy(x, sr)
    noise = float(np.percentile(e, 10))
    speech = float(np.percentile(e, 90))
    rms = float(np.sqrt(np.mean(x**2) + 1e-12))
    return rms, noise, 20.0 * np.log10((speech + 1e-9) / (noise + 1e-9))


def best_window(x: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """Highest-energy contiguous window of the given length (avoids silence)."""
    n = int(sr * seconds)
    if len(x) <= n:
        return x
    e = frame_energy(x, sr, win_ms=50.0)
    step = max(1, int(sr * 0.05))
    win_frames = max(1, n // step)
    if len(e) <= win_frames:
        return x[:n]
    csum = np.cumsum(np.insert(e, 0, 0))
    sums = csum[win_frames:] - csum[:-win_frames]
    start = int(np.argmax(sums)) * step
    return x[start : start + n]


def cosine_matrix(embs: np.ndarray) -> np.ndarray:
    u = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return u @ u.T


def pairwise_stats(embs: np.ndarray) -> tuple[float, float, float, int]:
    if len(embs) < 2:
        return float("nan"), float("nan"), float("nan"), 0
    g = cosine_matrix(embs)
    iu = np.triu_indices(len(embs), 1)
    v = g[iu]
    return float(v.min()), float(v.mean()), float(v.max()), len(v)


# --------------------------------------------------------------------------
# Embedding (production-faithful)
# --------------------------------------------------------------------------
class Embedder:
    def __init__(self) -> None:
        import torch
        from pyannote.audio import Inference, Model

        self._torch = torch
        cfg = get_config().voice.diarization
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        print(f"[embed] loading {cfg.embedding_model}")
        model = Model.from_pretrained(cfg.embedding_model, token=token)
        self._inf = Inference(model, window="whole")
        self._diar_model = cfg.model
        self._token = token
        self._pipeline = None

    def embed(self, mono: np.ndarray, sr: int) -> np.ndarray | None:
        if len(mono) == 0:
            return None
        wf = self._torch.tensor(mono.astype(np.float32)).unsqueeze(0)
        arr = np.array(self._inf({"waveform": wf, "sample_rate": sr}))
        if not np.all(np.isfinite(arr)):
            return None
        return arr

    def pipeline(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline

            print(f"[embed] loading diarization pipeline {self._diar_model}")
            self._pipeline = Pipeline.from_pretrained(self._diar_model, token=self._token)
        return self._pipeline

    def diarize_fragments(self, mono: np.ndarray, sr: int) -> list[tuple[float, float, np.ndarray]]:
        wf = self._torch.tensor(mono.astype(np.float32)).unsqueeze(0)
        result = self.pipeline()({"waveform": wf, "sample_rate": sr})
        diar = getattr(result, "speaker_diarization", result)
        out = []
        for seg, _track, label in diar.itertracks(yield_label=True):
            crop = mono[int(seg.start * sr) : int(seg.end * sr)]
            emb = self.embed(crop, sr)
            if emb is not None:
                out.append((seg.start, seg.end, label, emb))
        return out


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------
def a1_channels(utts: list[dict]) -> int:
    print("\n=== [1] Per-channel RMS / noise-floor / SNR ===")
    nch = utts[0]["audio"].shape[1]
    rows = np.zeros((nch, 3))
    for u in utts:
        for c in range(nch):
            rms, noise, snr = snr_db(u["audio"][:, c], u["sr"])
            rows[c] += [rms, noise, snr]
    rows /= len(utts)
    print(f"  {'ch':>3} {'rms':>10} {'noise':>10} {'SNR_dB':>8}")
    for c in range(nch):
        print(f"  {c:>3} {rows[c,0]:>10.4f} {rows[c,1]:>10.4f} {rows[c,2]:>8.1f}")
    best = int(np.argmax(rows[:, 2]))
    print(f"  -> highest mean SNR: channel {best} "
          f"(production uses output_channel=0)")
    return best


def a2_whole_vs_fragments(utts: list[dict], emb: Embedder, channel: int) -> None:
    print("\n=== [2] Whole utterance vs diarization fragments (ch%d) ===" % channel)
    whole_by_spk: dict[str, list[np.ndarray]] = defaultdict(list)
    frag_by_spk: dict[str, list[np.ndarray]] = defaultdict(list)
    split_counts = []
    for u in utts:
        spk = u["meta"]["speaker"]
        mono = u["audio"][:, channel]
        w = emb.embed(mono, u["sr"])
        if w is not None:
            whole_by_spk[spk].append(w)
        frags = emb.diarize_fragments(mono, u["sr"])
        n_spk = len({f[2] for f in frags})
        split_counts.append((n_spk, len(frags), u["meta"]["command"]))
        for _s, _e, _lbl, fe in frags:
            frag_by_spk[spk].append(fe)
    print("  same-speaker pairwise cosine:")
    for spk in sorted(whole_by_spk):
        wmin, wmean, wmax, wn = pairwise_stats(np.array(whole_by_spk[spk]))
        fmin, fmean, fmax, fn = pairwise_stats(np.array(frag_by_spk.get(spk, [])))
        print(f"    {spk:>8}  WHOLE mean={wmean:.3f} [{wmin:.2f},{wmax:.2f}] n={wn}")
        print(f"    {spk:>8}  FRAG  mean={fmean:.3f} [{fmin:.2f},{fmax:.2f}] n={fn}")
    multi = sum(1 for n, _f, _c in split_counts if n > 1)
    print(f"  single-speaker utterances split into >1 'speaker': "
          f"{multi}/{len(split_counts)}")
    avg_frags = np.mean([f for _n, f, _c in split_counts])
    print(f"  avg fragments per utterance: {avg_frags:.1f}")


def a3_duration(utts: list[dict], emb: Embedder, channel: int) -> None:
    print("\n=== [3] Same-speaker cosine vs clip length (ch%d) ===" % channel)
    print(f"  {'len_s':>6}  {'mean':>6} {'min':>6} {'max':>6} {'pairs':>6}")
    for L in DURATION_BUCKETS:
        by_spk: dict[str, list[np.ndarray]] = defaultdict(list)
        for u in utts:
            mono = best_window(u["audio"][:, channel], u["sr"], L)
            e = emb.embed(mono, u["sr"])
            if e is not None:
                by_spk[u["meta"]["speaker"]].append(e)
        allmins, allmeans, allmaxs, n = [], [], [], 0
        for spk, embs in by_spk.items():
            mn, me, mx, c = pairwise_stats(np.array(embs))
            if c:
                allmins.append(mn); allmeans.append(me); allmaxs.append(mx); n += c
        if allmeans:
            print(f"  {L:>6.1f}  {np.mean(allmeans):>6.3f} {np.min(allmins):>6.3f} "
                  f"{np.max(allmaxs):>6.3f} {n:>6}")


def a4_noise(utts: list[dict], emb: Embedder, channels: list[int]) -> None:
    print("\n=== [4] Noise impact: same-speaker cosine by condition x channel ===")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for u in utts:
        cond = u["meta"]["condition"]
        kind = "noisy" if any(k in cond for k in ("dishwash", "tv", "noise")) else "quiet"
        groups[(u["meta"]["speaker"], kind)].append(u)
    print(f"  {'speaker':>8} {'cond':>6} {'ch':>3} {'mean':>6} {'n':>4}")
    for (spk, kind), us in sorted(groups.items()):
        for c in channels:
            embs = [emb.embed(u["audio"][:, c], u["sr"]) for u in us]
            embs = [e for e in embs if e is not None]
            _mn, me, _mx, n = pairwise_stats(np.array(embs))
            print(f"  {spk:>8} {kind:>6} {c:>3} {me:>6.3f} {n:>4}")


def a5_separability(utts: list[dict], emb: Embedder, channel: int) -> None:
    print("\n=== [5] Within- vs cross-speaker separability + threshold (ch%d) ===" % channel)
    embs, labels = [], []
    for u in utts:
        e = emb.embed(u["audio"][:, channel], u["sr"])
        if e is not None:
            embs.append(e)
            labels.append(u["meta"]["speaker"])
    if len({*labels}) < 2:
        print("  need >=2 speakers for separability — skipping.")
        return
    embs = np.array(embs)
    g = cosine_matrix(embs)
    same, cross = [], []
    for i, j in itertools.combinations(range(len(embs)), 2):
        (same if labels[i] == labels[j] else cross).append(g[i, j])
    same, cross = np.array(same), np.array(cross)
    print(f"  within-speaker: mean={same.mean():.3f} sd={same.std():.3f} n={len(same)}")
    print(f"  cross-speaker:  mean={cross.mean():.3f} sd={cross.std():.3f} n={len(cross)}")
    # EER sweep
    best_t, best_gap = 0.0, 1e9
    for t in np.linspace(-0.2, 1.0, 121):
        frr = float(np.mean(same < t))   # genuine rejected
        far = float(np.mean(cross >= t))  # impostor accepted
        if abs(frr - far) < best_gap:
            best_gap, best_t, best_eer = abs(frr - far), float(t), (frr + far) / 2
    far1 = next((float(t) for t in np.linspace(1.0, -0.2, 121)
                 if np.mean(cross >= t) <= 0.01), float("nan"))
    prod_t = getattr(get_config().perception, "speaker_threshold", "n/a")
    print(f"  EER threshold ~ {best_t:.3f} (EER={best_eer:.2%})")
    print(f"  threshold for <=1% false-accept: {far1:.3f}")
    print(f"  (production speaker_threshold = {prod_t})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--in-dir", default="data/voice_diag")
    p.add_argument("--channel", type=int, default=0, help="Primary channel to analyse")
    p.add_argument("--noise-channels", default="0", help="Comma list for the noise sweep")
    p.add_argument("--skip-diarization", action="store_true",
                   help="Skip analysis 2 (heavy pyannote pipeline)")
    args = p.parse_args()

    utts = load_utterances(Path(args.in_dir))
    if not utts:
        print(f"no utterances found under {args.in_dir}", file=sys.stderr)
        return 2
    speakers = sorted({u["meta"]["speaker"] for u in utts})
    conds = sorted({u["meta"]["condition"] for u in utts})
    print(f"loaded {len(utts)} utterances | speakers={speakers} | conditions={conds}")

    best_ch = a1_channels(utts)

    emb = Embedder()
    if not args.skip_diarization:
        a2_whole_vs_fragments(utts, emb, args.channel)
    a3_duration(utts, emb, args.channel)
    a4_noise(utts, emb, [int(c) for c in args.noise_channels.split(",")])
    a5_separability(utts, emb, args.channel)
    print(f"\n[done] (channel-1 analysis suggested ch{best_ch} as cleanest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
