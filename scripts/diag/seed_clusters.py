#!/usr/bin/env python3
"""Seed voice clouds from vetted diagnostic clips + calibrate thresholds.

Embeds the labelled captures from ``capture_voice.py`` (ch0, whole
utterance, the same wespeaker path the live matcher uses) and writes them
into each person's voice cloud in ``perception.db`` via ``CloudStore`` —
prepopulating Jacob and Carina so recognition works from boot. Creates
the Carina person record if missing.

Then reports the **clip-to-cloud** calibration: leave-one-out genuine
scores (mean top-k cosine to the rest of your own cloud) vs cross-speaker
impostor scores, with suggested confirmed/maybe thresholds. These are the
real numbers to lock `voice_confirmed_threshold` / `voice_maybe_threshold`
to (the earlier 0.55/0.44 were clip-to-clip estimates).

Run on the Pi, boxbot stopped:
    python3 scripts/diag/seed_clusters.py            # seed + calibrate
    python3 scripts/diag/seed_clusters.py --dry-run  # calibrate only, no writes
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from analyze_voice import Embedder, load_utterances  # noqa: E402

from boxbot.perception.clouds import CloudStore  # noqa: E402

TOPK = 3


def _topk_mean(query: np.ndarray, cloud: np.ndarray, k: int = TOPK) -> float:
    sims = cloud @ query
    k = min(k, sims.shape[0])
    return float(np.sort(sims)[-k:].mean())


def calibrate(by_speaker: dict[str, np.ndarray]) -> None:
    """Report leave-one-out genuine vs cross-speaker impostor scores."""
    print("\n=== clip-to-cloud calibration ===")
    genuine: list[float] = []
    impostor: list[float] = []
    for spk, embs in by_speaker.items():
        for i in range(len(embs)):
            rest = np.delete(embs, i, axis=0)
            if len(rest):
                genuine.append(_topk_mean(embs[i], rest))
        for other, oembs in by_speaker.items():
            if other == spk:
                continue
            for i in range(len(embs)):
                impostor.append(_topk_mean(embs[i], oembs))
    g, im = np.array(genuine), np.array(impostor)
    print(f"  genuine  (leave-one-out): mean {g.mean():.3f} sd {g.std():.3f} "
          f"min {g.min():.3f} n {len(g)}")
    if len(im):
        print(f"  impostor (cross-speaker): mean {im.mean():.3f} sd {im.std():.3f} "
              f"max {im.max():.3f} n {len(im)}")
    # Suggested thresholds: capture ~80% / ~98% of genuine, sanity-check FAR.
    conf = float(np.percentile(g, 20))
    maybe = float(np.percentile(g, 2))
    print(f"  suggested confirmed (80% genuine capture): {conf:.3f}")
    print(f"  suggested maybe     (98% genuine capture): {maybe:.3f}")
    if len(im):
        far_conf = float((im >= conf).mean())
        far_maybe = float((im >= maybe).mean())
        print(f"  impostor false-accept @confirmed: {far_conf:.1%}  "
              f"@maybe: {far_maybe:.1%}")


async def seed(by_speaker: dict[str, np.ndarray], dry_run: bool) -> None:
    if dry_run:
        print("\n[dry-run] skipping DB writes")
        return
    store = CloudStore()
    await store.initialize()
    try:
        for spk, embs in by_speaker.items():
            person = await store.get_person_by_name(spk.capitalize())
            if person is None:
                person = await store.get_person_by_name(spk)
            if person is None:
                pid = await store.create_person(spk.capitalize())
                print(f"  created person {spk.capitalize()} ({pid})")
            else:
                pid = person["id"]
            added = 0
            for e in embs:
                if await store.add_voice_embedding(pid, e):
                    added += 1
            await store.recompute_voice_centroid(pid)  # legacy-centroid backcompat
            print(f"  {spk}: +{added} voice embeddings → person {pid}")
    finally:
        await store.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--in-dir", default="data/voice_diag")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="calibrate only, no DB writes")
    args = p.parse_args()

    utts = load_utterances(Path(args.in_dir))
    if not utts:
        print(f"no labelled utterances under {args.in_dir}", file=sys.stderr)
        return 2

    emb = Embedder()
    by_speaker: dict[str, list[np.ndarray]] = defaultdict(list)
    for u in utts:
        e = emb.embed(u["audio"][:, args.channel], u["sr"])
        if e is None:
            continue
        v = e.astype(np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        by_speaker[u["meta"]["speaker"]].append(v)
    stacked = {s: np.stack(v) for s, v in by_speaker.items() if v}
    print(f"embedded {sum(len(v) for v in stacked.values())} clips: "
          + ", ".join(f"{s}={len(v)}" for s, v in stacked.items()))

    calibrate(stacked)
    asyncio.run(seed(stacked, args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
