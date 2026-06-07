# Person-ID overhaul: bootstrap, maintenance & reconciliation

Status: **approved (2026-06-07)**, building in phases. Extends
`docs/voice-id-redesign.md` (completes the visual half of Stage 2 + adds the
reconciliation layer it deferred).

## Desired end-state
BB recognizes household members by face+voice, knows who's speaking, the
embedding clouds **grow from everyday interaction**, mistakes **self-correct**,
and duplicate/contaminated records get **cleaned up** — no manual babysitting.
Named-person triggers (`person="Jacob"`) fire reliably as a consequence.

## Why the current system can't do this
- Visual identification is dead: ReID runs but live-vs-stored cosine ≈0.35 vs a
  0.85 threshold; only **1** visual embedding enrolled (vs 42 voice, bulk-seeded).
- Root cause: enrollment admits **only** on explicit `identify_person`
  (`commit_session` gates on `claim.source == "agent_identify"`). Passive
  recognition never teaches the system, so the cloud never grows and decays as
  appearance drifts. The post-conversation `confirm_session_embeddings` path is
  vestigial (`IdentityFusion` has no `session_speakers` attr → always empty).
- Visual still uses single-centroid matching + naive oldest-first eviction;
  voice was migrated to the cloud/top-k model but visual was not.

## Design principles
Provenance-tagged **liberal admission** (not a strict gate) · **voice
bootstraps vision** · **always-on isolation janitor** · **nightly dream-cycle
reconciliation** · `agent_identify` = ground-truth anchor **and** correction
lever, with corrections that **propagate**.

## Locked decisions (2026-06-07)
1. **Generalize `voice_confirmed` → `provenance`** column (back-compat flag derived).
2. **Admission rules:**
   - *Visual* admits on **voice-confirmed (≥0.55) + DOA-associated detection in
     FOV** (bootstrap; vision need not already know the face). provenance=`voice_doa`.
   - *Voice* admits on **voice+visual agreement** (voice=X and visual ReID=X).
     provenance=`voice_visual_agree`.
   - *`identify_person`* is canonical for both; overrides claims; corrections
     propagate (evict/re-tag bad admits). provenance=`agent_identify`.
3. **Thresholds:** ship a sensible **middle-road** default, calibrate later.
4. **Person-merge:** **audit-only** for now (flag, never auto-merge).
5. **Reconcile home:** fold into the existing **nightly dream trigger**.

Provenance tiers (strongest→weakest), with weights for matching:
`agent_identify > voice_visual_agree > voice_doa > visual_reid > context_inferred > seed > legacy`

---

## Foundation (prereq for all phases)
- **Schema** (`clouds.py` + `_migrate`): `visual_embeddings`/`voice_embeddings`
  gain `provenance TEXT NOT NULL DEFAULT 'legacy'`, `confidence REAL`,
  `source_ref TEXT`. Keep `crop_path`; keep `voice_confirmed` derived. New
  `id_corrections` audit table (ref, from→to person, ts, source). Backfill
  existing rows to `seed`/`legacy`.
- **Config** (`PerceptionConfig`): `visual_confirmed_threshold`,
  `visual_maybe_threshold`, `visual_cloud_topk`, `visual_admit_min_confidence`,
  provenance weights, reconcile knobs (audit_only, gates, caps). Middle-road
  initial thresholds.

## PHASE NOW — runtime growth + self-clean + cloud matching
- **N1 Cloud matching:** `get_visual_clouds()`; `visual_reid.match_cloud()`
  (provenance-weighted top-k mean cosine + confirmed/maybe tiers);
  `_run_reid` uses them, publishes `PersonIdentified` on high tier. Stop reading
  `centroids` table (leave dormant).
- **N2 Admission (per decision 2):** add claim sources `voice_doa`,
  `voice_visual_agree`. Visual admits on voice-confirmed+DOA+FOV; voice admits
  on voice+visual agreement; `agent_identify` overrides. Quality gates:
  detection confidence, crop area, DOA-ambiguity guard (skip when two
  detections are ~equidistant in angle).
- **N3 Visual isolation+age eviction:** rewrite `_enforce_visual_cap` to mirror
  `_enforce_voice_cap`, provenance-aware (never evict `agent_identify`; shed
  low-provenance outliers first). **Ships with N2.**
- **N4 Wire crop capture:** call `save_crop` in `_run_reid`/at admission, linked
  to `embedding_id` + provenance (currently never called).
- **N5 Tests + deploy + verify:** unit (cloud shape, weighted tiers, isolation
  picks outliers, voice_doa/voice_visual_agree admit); on-device — converse a
  few times, watch visual cloud grow, ReID tier climb, `PersonIdentified`
  publish, named trigger fire.

## PHASE NEXT — dream-cycle ID reconciliation (audit-only first)
New `perception/reconcile.py::run_id_reconcile(...)`, folded into the nightly
dream trigger. Mirrors memory dream (`gather → judge → apply`,
`audit_only=True` default, confidence gate, async batch via poller). Four jobs:
1. **Outlier eviction** (deterministic) — global isolation pass per cloud.
2. **Retroactive self-labeling** (embedding-first, semi-supervised) — match
   low-confidence/unknown stored crops against **high-provenance anchors only**;
   promote when top1≫top2, ≥ promote_threshold, and context-consistent
   (who was voice-confirmed/present at that timestamp).
3. **Duplicate-person merge** (LLM + embeddings + **crop images**) — detect
   same-human records (name edit-distance e.g. Eric/Erik + cloud proximity +
   co-presence) → multimodal `MERGE_PERSONS` judge. **Audit-only.**
4. **Cross-modal conflict resolution** (LLM-assisted, crops attached) — reconcile
   voice-vs-visual contradictions over a window, weighing provenance + context.

De-risk: `audit_only` default, high confidence gate (≥0.85), **prefer FLAG over
destroy**, never touch `agent_identify` anchors, log to `id_corrections` for
undo, batch-to-low-water, self-label off anchors only (no feedback loop). Cold
start: no anchors → no-op beyond outlier eviction (hard dependency: NOW seeds
anchors before NEXT helps).

## PHASE OPTIONAL — utterance-time speaker crop
- **O1** Speaker-synchronized capture: per utterance, use DOA to pick the bbox,
  capture a fresh frame, run YOLO+ReID → embedding+crop time-aligned to the
  voice (spends a little Hailo during CONVERSATION, currently freed).
- **O2** Correction propagation + prompt completeness: inject `[Jacob (0.9)]` /
  `[Unknown_1]` per turn; "that's Carina" → `identify_person` → runtime
  propagate (evict/re-tag recent wrong admits, log to `id_corrections`).
- **O3** DOA ambiguity UX: overlapping speakers → mark identity ambiguous, agent
  can ask, rather than guessing.

## Sequencing
Foundation → NOW (one PR, deploy+verify) → NEXT (reconcile, audit-only ~1 week,
then enable apply) → OPTIONAL. Thresholds: ship middle-road, calibrate from
accumulated data (visual analog of `scripts/diag/seed_clusters.py`) as follow-up.
