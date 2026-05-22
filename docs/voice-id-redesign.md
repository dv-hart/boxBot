# Voice ID redesign (2026-05)

## Why

Person voice-ID was effectively non-functional. Root cause, measured with
`scripts/diag/` on real captures (see memory `voice-visual-id-overhaul`):

- The `speaker_threshold` was **0.75**, but same-speaker cosine on this
  far-field 16 kHz front-end tops out ~0.6. Genuine matches could never
  clear the bar — *the threshold, not the embeddings, was the bug.*
- Cross-speaker cosine sits at ~0.20, so there is a wide, clean margin
  (Jacob↔Carina EER = 0% at threshold 0.36). Recognition is very doable
  at the right operating point.
- Speaker diarization fragmented each single-speaker utterance into ~1.6
  weak sub-second embeddings (whole-utterance 0.61 vs fragments 0.41) and
  cost 3.6–4.5 s of latency for no benefit in the common case.
- The enrolled "centroid" was a flat mean of a polluted, multi-modal
  cloud — a phantom point close to no real clip.

## Operating model

**Attribution is per-utterance, never per-session.** Each VAD utterance
gets one whole-utterance embedding and is matched independently against
each person's *cloud*. There is no sticky "this conversation is Jacob"
flag — so when the mic stays open and Carina or the kids speak, those
clips match their own cloud (or nobody), never the person who started
the conversation.

**The cloud, not a centroid.** A person is the set of their embeddings.
Match score = mean of the **top-3 cosine** to that cloud (proximity to
real points, preserving near/far/noisy modes instead of averaging them
into a phantom). Cheap at household scale; ANN indexes only needed at
thousands of users.

**Thresholds calibrated to the genuine distribution, bounded by the
impostor gap** (clip-to-clip proxy; re-measured clip-to-cloud after
seeding):

| Tier | cosine | meaning | source |
|---|---|---|---|
| confirmed | ≥ 0.55 | address by name, act on identity | ~80th pct genuine capture; ~0 false-accept |
| maybe | 0.44–0.55 | tentative; verify / use visual / context | ~98th pct genuine capture |
| unknown | < 0.44 | unidentified speaker | — |

## Enrollment (admission to a cloud) — strict, decoupled from matching

Matching is liberal; admission is strict. Admit an utterance's embedding
to person X only if **intrinsic quality** passes AND an **identity
source** confirms it:

- **Intrinsic quality** (absolute, no feedback loop):
  - duration ≥ ~2 s, SNR above floor;
  - **single-speaker check** — split into ~1 s sub-windows, embed each,
    require they agree (intra-clip cosine high). Kills "two voices bled
    into one clip" without ever consulting the cloud. This is the cheap
    stand-in for diarization's one useful job (detecting overlap).
- **Identity source** (one of):
  - **agent_identify** — "this is X" binds to the *current speaker's*
    utterance(s) only (the contiguous run that voice-matches each other),
    never the whole session;
  - **voice + visual agree** — voice ≥ 0.55 to X *and* DOA-associated
    visual ReID = X.

No coherence-to-cloud gate (v1) — single-anchor coherence is a
poison-lock (a bad first clip would reject all later good clips). Intrinsic
quality + majority-consensus at enrollment + isolation-based eviction
keep the cloud clean without it.

**Visual is bootstrapped by voice:** a voice-confirmed utterance (≥0.55)
with a DOA-associated visual detection enrolls the *visual* embedding —
"voice teaches vision." (Wiring this through `identify_person`/fusion is
Stage 2; it fixes the dropped-visual bug.)

## Retention — cap + isolation/age eviction

When a cloud exceeds its cap, evict by:

```
evict_priority = isolation + λ·age        (λ small → isolation dominates)
isolation = 1 − mean(top-k cosine to nearest neighbours in the cloud)
```

Pulls true outliers first (isolated points: contaminated/mis-enrolled),
then old outliers, then just-old — your ordering. Uses distance to
**neighbours**, not to the global mean, so legit far/dishwasher clips
(which have neighbours) are preserved while genuine junk (isolated) is
shed. Evict in a batch to a low-water mark (~90% of cap) to avoid churn.

## Re-clustering — deferred

A per-person "looks off" flag (bimodal cloud, repeated clean clips that
conflict with the core) is left as a cheap hook for a later dream-cycle
hygiene pass. Not built in v1; eviction handles routine cleanliness.

## Diarization — off for now

Single speaker per VAD utterance is assumed; embed the whole utterance
directly with the same wespeaker model (`window="whole"`), skip the
diarization pipeline (no fragmentation, no 3.6–4.5 s latency, faster
boot). Re-enable later only when the single-speaker check detects
genuine overlap.

## Prepopulation

Seed Jacob + Carina clouds from the vetted diagnostic clips
(`data/voice_diag/_inbox/`, ch0, whole-utterance, same embedding path as
live) so recognition works from boot. Creates the Carina person record.

## Build stages

- **Stage 1 (foundation, offline-validatable):** cloud storage +
  top-3 matching (`clouds.py`, `voice_reid.py`), config thresholds,
  isolation/age eviction, `scripts/diag/seed_clusters.py` (seed +
  clip-to-cloud percentile calibration to lock 0.55/0.44 from real data).
  Does not change live behaviour yet.
- **Stage 2 (live path):** diarization-off whole-utterance embedding
  (`voice.py`, `diarization.py`), per-utterance attribution + tiers
  (`fusion.py`, `pipeline.py`), enrollment rewrite with intrinsic gates
  and per-utterance admission (`enrollment.py`), DOA→visual binding.
  Needs on-device speaking to verify.

## File map

| File | Change |
|---|---|
| `core/config.py` | `DiarizationConfig.enabled`; `PerceptionConfig.voice_confirmed_threshold` (0.55), `voice_maybe_threshold` (0.44), `voice_cloud_topk` (3) |
| `perception/clouds.py` | `get_voice_clouds()`; isolation/age `_enforce_voice_cap` |
| `perception/voice_reid.py` | `match_cloud()` top-k + config-driven tiers |
| `scripts/diag/seed_clusters.py` | seed clouds + calibrate |
| `communication/voice.py`, `diarization.py` | Stage 2: whole-utterance, no pipeline |
| `perception/fusion.py`, `pipeline.py` | Stage 2: per-utterance tiers/attribution |
| `perception/enrollment.py` | Stage 2: intrinsic gates, per-utterance admission |
