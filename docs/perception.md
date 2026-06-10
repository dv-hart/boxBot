# Perception Pipeline

## Overview

boxBot identifies who is present through fused visual and audio signals.
The key insight: **voice teaches vision**. Speaker identification (via
pyannote diarization) provides reliable labels that bootstrap the visual
re-identification system over time, eliminating the need for explicit
face enrollment.

The pipeline is **event-driven, not continuous**. At rest, the system
runs minimal motion detection. When someone appears, it takes a snapshot
for identification. During conversation, the Hailo NPU is completely
free — all active work shifts to CPU-based audio processing and network
API calls.

People recognized by perception are **not users**. Anyone in the room
can talk to BB — physical presence is sufficient trust. "User" status
(WhatsApp access, remote messaging) requires admin-approved registration
via the separate code-based flow. See
[user-registration.md](user-registration.md).

## State Machine

```
┌──────────────────────────────────────────────────────────────┐
│  DORMANT                                                      │
│  Camera: 5-10 FPS low-res (320×240) to CPU                   │
│  CPU: frame differencing (~1ms per frame)                     │
│  Hailo: fully idle, models loaded but not executing           │
│                                                               │
│  Trigger: motion_score > threshold                            │
├────────────────────────────────────────────────────┬─────────┤
│  ▼ motion detected                                 │ timeout │
├────────────────────────────────────────────────────┤ (no     │
│  CHECKING                                          │ motion) │
│  Camera: single full-res frame                     │    │    │
│  Hailo: YOLOv5s-personface detection (~26ms)        │    │    │
│  Result: person? → DETECTED                        │    │    │
│  Result: not a person → back to DORMANT            │◄───┘    │
├────────────────────────────────────────────────────┤         │
│  ▼ person confirmed                                │         │
├────────────────────────────────────────────────────┤         │
│  DETECTED                                          │         │
│  Hailo: ReID embedding on person crop(s) (<1ms)    │         │
│  CPU: centroid comparison → tentative label        │         │
│  Event: person_detected(label, confidence)         │         │
│  Begin lazy-loading pyannote models (background)   │         │
│                                                    │         │
│  If wake trigger pending for this person → fire    │         │
│  If wake word heard → CONVERSATION                 │         │
│  Periodic re-check: YOLO every ~5s (presence)      │         │
│  No person for 30s → DORMANT                       │         │
├────────────────────────────────────────────────────┤         │
│  ▼ wake word / speech                              │         │
├────────────────────────────────────────────────────┤         │
│  CONVERSATION                                      │         │
│  Hailo: one final ReID shot, then FREE             │         │
│  CPU: pyannote diarization on audio stream         │         │
│  CPU: DOA tracking per utterance (ReSpeaker)       │         │
│  Network: ElevenLabs STT, Claude API               │         │
│                                                    │         │
│  Output: attributed transcript injected into       │         │
│          agent context (see Conversation Input)    │         │
│  Idle timeout → POST_CONVERSATION                  │         │
├────────────────────────────────────────────────────┤         │
│  ▼ conversation ends                               │         │
├────────────────────────────────────────────────────┤         │
│  POST_CONVERSATION                                 │         │
│  Commit nothing here — embedding commits happen at │         │
│  voice-session end (EnrollmentManager session)     │         │
│  Prune expired crop images                         │         │
│  Unload pyannote models (or keep warm briefly)     │         │
│  → DORMANT                                         │         │
└──────────────────────────────────────────────────────────────┘
```

## Two-Step Idle Detection

The idle detection uses a CPU → Hailo cascade. This isn't about saving
compute (YOLO at 1 FPS is ~0.07% of Hailo capacity) — it's about
**lower latency** and **clean state transitions**.

### Step 1: CPU Motion Detection

Frame differencing on low-res grayscale frames. Runs at 5-10 FPS on a
single A76 core at negligible cost (~1ms per frame).

```
current_gray = grayscale(frame_320x240)
delta = absdiff(current_gray, previous_gray)
motion_score = mean(delta)
if motion_score > threshold → trigger Step 2
```

Gaussian blur before differencing reduces false triggers from sensor
noise. Threshold tuning: start at 10-15 on uint8 (0-255 range). Too
low triggers on lighting changes; too high misses slow movement.

**Why not just YOLO at 1 FPS?** Motion detection at 5-10 FPS catches
a person entering the room within ~100-200ms. YOLO at 1 FPS has up to
1000ms worst-case latency. The faster detection matters for
person-triggered wake events ("remind Jacob when he walks in").

### Step 2: Hailo Person Detection

A single full-resolution frame is sent to YOLOv5s-personface on the
Hailo. Takes ~26ms. If no person is found (pet, shadow, lighting change),
the system returns to DORMANT. If a person is found, bounding boxes are
extracted and passed to ReID. Note: YOLOv5s-personface detects both
person and face classes — face detections are available for future
face-based ReID but are not used in v1.

**Compute cost at 1 FPS equivalent:**
```
YOLOv5s-personface: ~7.4 GFLOPs per inference
Hailo-8L: 13 TOPS capacity
Utilization: ~0.06%
Inference time: ~26ms per frame
Power: ~1.52W (barely above 1.5W idle)
```

### Edge Case: Person Standing Still

If someone enters the room (triggering motion → YOLO → DETECTED) and
then sits still, motion detection stops firing. The DETECTED state runs
a periodic YOLO heartbeat every ~5 seconds to confirm continued
presence. This heartbeat stops after the person leaves (no detection
for 30s → DORMANT).

## Visual Re-Identification (ReID)

### Model

**RepVGG-A0** (person ReID variant):
- 2.5M parameters, 512-dim embeddings
- RepVGG architecture: multi-branch training, single-path inference
  (structural re-parameterization makes it fast on accelerators)
- Sub-millisecond inference on Hailo-8L (<1ms per crop)
- 256×128 input resolution (standard person ReID crop size)
- Pre-compiled HEF: `repvgg_a0_person_reid_512.hef`

**Other options considered:**

| Model | Params | Embedding | Accuracy (Market-1501) | Notes |
|-------|--------|-----------|----------------------|-------|
| OSNet-x0.25 | 0.2M | 512-dim | ~73% mAP | Ultralight, lower accuracy |
| OSNet-AIN-x1.0 | 2.2M | 512-dim | ~86% mAP | Good domain shift handling, but Hailo compilation issues |
| **RepVGG-A0** | **2.5M** | **512-dim** | **~85% mAP** | **Selected — sub-ms on Hailo, proven HEF available** |
| MobileNetV3 + ReID head | 5.4M | 512-dim | ~78% mAP | General-purpose backbone, less ReID-optimized |
| ResNet18 + ReID head | 11M | 512-dim | ~80% mAP | Proven but larger, no ReID-specific design |
| LightMBN | 3.3M | 1536-dim | ~91% mAP | Upgrade path if accuracy insufficient |
| BoT (ResNet50) | 23.5M | 2048-dim | ~94% mAP | Overkill for on-demand inference at household scale |

**Potential improvement:** Fine-tune on home-environment data collected
by BB itself. Export accumulated labeled crops, fine-tune on a GPU
machine, deploy updated HEF. This is an offline process, not real-time.

### Embedding Clouds

Each known person accumulates a cloud of 512-dim ReID embedding vectors.
New detections are compared against these clouds to estimate identity.

**Matching:** Centroid-based cosine similarity.

```
For each known person:
  similarity = cosine(new_embedding, person_centroid)

Best match above high_threshold (0.85) → confident identification
Best match in range (0.60-0.85) → tentative (logged, not acted on)
No match above 0.60 → unknown person
```

If two people's centroids are close enough to cause ambiguity, fall back
to k-NN voting among stored embeddings for disambiguation.

### Confirmation Rule: Voice Gates Vision

**Visual-only identification can read from the cloud but never writes
to it.** New visual embeddings are only added to a person's cloud when
confirmed by one of:

1. **Voice confirmation** — pyannote diarization matches the speaker to
   a known voice profile during the same session
2. **Agent confirmation** — the agent learns identity through
   conversation context and calls `identify_person`

This prevents misidentification cascading. Without this rule:
- BB misidentifies Jacob as Erik (visual-only)
- Adds the embedding to Erik's cloud
- Erik's centroid drifts toward Jacob's appearance
- Future misidentifications become more likely

With the rule, the visual cloud only grows from confirmed data points.
Visual matching can still fire wake triggers (reading the cloud), but
it never poisons the cloud with unconfirmed data.

### Confidence Tiers

```
Visual-only, high confidence (>0.85):
  → Fire person-triggered wake events
  → Provide tentative label in context
  → Do NOT add embedding to cloud

Visual + voice agreement:
  → Full confidence
  → Add visual embedding to cloud (confirmed)
  → Attribute conversation turns

Visual + voice conflict:
  → Check DOA (see below)
  → If DOA shows speaker outside FOV: no conflict, different people
  → If DOA shows speaker in FOV: trust voice, flag for review
  → Do NOT add visual embedding to conflicting cloud

Visual-only, medium confidence (0.60-0.85):
  → Tentative label (internal/debug only)
  → Do NOT fire triggers
  → Wait for voice confirmation
```

### Embedding Pruning

Clouds are capped to allow natural drift over time:

```
MAX_VISUAL_EMBEDDINGS = 200 per person
MAX_VOICE_EMBEDDINGS = 50 per person
PRUNING_STRATEGY = isolation + age (anchors never evicted)
```

When the cap is exceeded, embeddings are evicted in one batch down to
~90% of cap, choosing the highest `isolation + λ·age` first — isolated
points (likely contamination) go before merely-old ones, and
`agent_identify` anchors are never evicted. This lets the cloud
gradually adapt to appearance changes (new haircut, glasses, seasonal
clothing) without manual re-enrollment. The same janitor re-enforces
the caps after a person merge combines two clouds.

Centroids are recomputed whenever the cloud changes.

### Crop Image Retention

Person crops used for ReID are stored temporarily for debugging:

```
Normal mode: 1 day retention
Debug mode:  7 days retention
```

Crops are stored with metadata: timestamp, embedding vector ID, assigned
label, confidence score, voice-confirmed flag. This lets developers
audit misidentifications ("why did BB think this was Jacob?").

Embeddings themselves persist indefinitely (a few KB per person). Only
the images expire.

## Speaker Identification

### pyannote.audio

Speaker identification uses **pyannote.audio** for diarization and
speaker embedding extraction. pyannote runs on CPU during conversations.

**Why pyannote (not a standalone speaker embedding model)?**

The multi-speaker scenario is the deciding factor. When two people
talk to BB simultaneously (e.g., a couple discussing their calendar),
pyannote provides:
- **Diarization** — who spoke when, with timestamps
- **Overlapping speech handling** — detects when two people talk at once
- **Speaker embeddings** — extracted as part of diarization, reused for
  voice identification (no separate model needed)
- **Per-utterance attribution** — feeds directly into attributed
  conversation input for the agent

A standalone speaker embedding model (ECAPA-TDNN, wespeaker) would
handle single-speaker verification but cannot diarize multi-speaker
audio.

**Resource footprint:**
- Models: ~300-500MB RAM (lazy-loaded, see below)
- CPU: well within Pi 5's capability for real-time audio diarization
- PyTorch dependency: accepted tradeoff for accuracy and multi-speaker
  support
- Hailo: not used for audio — Hailo handles vision, CPU handles audio

**Lazy loading:** pyannote models are loaded when a person is detected
(DETECTED state), before conversation begins. The ~2-3 second load time
is hidden behind the natural delay between someone appearing and
speaking. By the time the wake word fires, models are warm.

### Voice Profiles

Each person stores multiple voice embeddings from pyannote (not just one
centroid). Voice varies across sessions (energy level, health, time of
day), so multiple reference embeddings capture this range.

Matching uses cosine similarity against stored embeddings, same approach
as visual ReID. Threshold: ~0.55-0.65 (voice embeddings are generally
more discriminative than visual, so the threshold can be tighter).

## Direction of Arrival (DOA)

The ReSpeaker XVF3000 4-mic array provides direction-of-arrival
estimation for the dominant voice source.

### Hardware Layout

```
     ┌──────────────────────────┐
     │   [ReSpeaker 4-Mic Array]│  ← sits on top of case
     │       ◉ (camera lens)    │  ← top center, forward-facing
     │                          │
     │       box front          │
     └──────────────────────────┘
```

### Calibration

Minimal. One config value:

```yaml
perception:
  doa_forward_angle: 0    # ReSpeaker angle that maps to camera center
  camera_hfov: 120        # Pi Camera Module 3 Wide horizontal FOV
```

`doa_forward_angle` is determined by how the mic array's USB cable is
oriented relative to the box front. Set once during box assembly. At
1-3 meters distance, the few centimeters of parallax between mic and
camera is negligible.

Mapping DOA angle to camera position:

```
camera_x_pct = (doa_angle - forward_angle) / (hfov / 2)
# -1.0 = left edge, 0.0 = center, +1.0 = right edge
# Values outside [-1, 1] = speaker is outside camera FOV
```

### DOA Use Cases

**1. Bootstrap voice-to-vision association (multi-person):**
When multiple people are in frame and one is speaking, DOA indicates
which bounding box corresponds to the speaker. The crop nearest the DOA
direction gets labeled with the speaker's voice ID.

**2. Conflict resolution:**
Visual ReID says "Jacob" is in frame. Voice says "Sarah" is speaking.
DOA shows the speaker is at 90° — outside the camera's FOV. Resolution:
Jacob is visible but not speaking. Sarah is speaking from beside the
box. No conflict — they're different people. Without DOA, this would
appear as a visual-voice disagreement.

**3. Speaker-outside-FOV detection:**
If `|camera_x_pct| > 1.0`, the speaker is not visible. Their voice
embeddings are still captured and matched, but no visual embedding
is associated with them for this utterance.

## Conversation Input Format

The perception pipeline's output is **not a tool call** — it's injected
directly into the agent's conversation input as attributed text. The
agent never sees raw embeddings, distances, or confidence scores.

**Single known speaker:**
```
[Jacob]: Hey BB, what's on the calendar tomorrow?
```

**Multiple known speakers:**
```
[Jacob]: Hey BB, what's on our calendar tomorrow?
[Sarah]: Oh also check if the plumber confirmed.
[Jacob]: Right, and remind me about the dentist.
```

**Unknown speaker present:**
```
[Jacob]: Hey BB, this is my friend Erik.
[Person B]: Hey, nice to meet you!
```

The agent sees `[Person B]` (the diarization label) and can engage
naturally. If Person B identifies themselves or Jacob introduces them,
the agent calls `identify_person` to create a named record.

**Presence header (injected into the dynamic context):**
```
[Present: Jacob (confirmed), Person B (new)]
```

Built from the perception pipeline's tracked people
(`boxbot.perception.presence.format_presence_line`) and injected into
the dynamic system context for **voice and trigger** conversations
(WhatsApp/Signal have no notion of room presence). Three tiers:

- `(confirmed)` — high-confidence identification (voice-confirmed or a
  high-tier visual match)
- `(likely)` — named, but only a medium/low-tier match
- `(new)` — no match in any known profile; shown by session ref
  (e.g. `Person B`)

The header is rebuilt every turn, so it is always current at the
moment the agent thinks.

**Mid-conversation updates.** When the tracked-person set changes
while a voice conversation is active, an update line is injected into
the thread as a user-style turn:

```
[Presence update: Jacob (confirmed), Sarah (likely)]
```

Updates are event-driven (PersonDetected / PersonIdentified — no
polling) and debounced: a change must be stable for ~7 s (aligned with
the DETECTED-state YOLO heartbeat of 5 s) before it is announced, and
updates are rate-limited to at most one per 30 s per conversation, so
tracking flicker never reaches the agent. Two known limits: visual
heartbeats pause while perception is in the CONVERSATION state (Hailo
is freed), so updates mostly fire between utterances; and departures
have no dedicated event — they surface on the next person event or
the next turn's header.

## The `identify_person` Tool

The single identity gateway. One tool, four actions — the agent never
touches embeddings directly:

```
action: "identify" (default) | "rename" | "merge" | "list_flags"

identify:   name + ref      — session speaker → person (create/confirm/
                              correct via session claims)
rename:     name + new_name — rename an existing record ("call me Jake").
                              Metadata only; errors if new_name belongs
                              to another person (that's a merge).
merge:      name + duplicate_name — merge duplicate records of the SAME
                              human ("Erik" → "Eric"). Destructive; the
                              agent must confirm with the people
                              involved before calling it.
list_flags: (no params)     — read the latest nightly reconcile audit
                              (duplicate-person candidates etc.).
```

### `action="identify"` Behavior

**If `name` matches an existing person record:**
Link `ref`'s embeddings (voice + associated visual) to that person's
record. This handles the case where BB doesn't recognize someone
(cold, different context) but the person identifies themselves in
conversation.

Example: Person A says "It's me, Jacob" → agent calls
`identify_person(name="Jacob", ref="Person A")` → backend merges
Person A's session embeddings into Jacob's cloud.

**If `name` does not match any existing record:**
Create a new person record with that name. Associate `ref`'s
embeddings with the new record.

Example: Jacob introduces his friend → agent calls
`identify_person(name="Erik", ref="Person B")` → backend creates
"Erik" and stores Person B's voice and visual embeddings.

### What the Backend Does

On receiving an `identify_person` call:

1. Look up all voice embeddings tagged with `ref` from the current
   diarization session
2. Look up all visual crops associated with `ref` via DOA/timing from
   the current session
3. If `name` exists: merge into existing person's cloud
4. If `name` is new: create person record, store as initial cloud
5. Recompute centroids
6. Return confirmation to agent: "Created person 'Erik'" or
   "Identified Person A as Jacob"

The agent never sees or handles embeddings. Token cost: one short tool
call.

### Rename and Merge

`action="rename"` updates the person's name in place — embeddings are
untouched. `action="merge"` moves the duplicate's visual + voice
embeddings into the surviving record (the per-person caps are then
re-enforced by the standard eviction janitor) and soft-keeps the
duplicate row with a `merged_into` pointer, so "It's Erik" still
resolves to Eric afterwards.

Both actions re-point everything keyed on the person:

- **photo person tags** (`photo_people.person_id` + label) — re-pointed
- **active person-condition triggers** (name-keyed) — re-pointed
- **in-session enrollment claims** — re-pointed, so the voice-session
  commit can't write embeddings to a stale/merged-away person
- **live speaker/presence maps** (agent + voice session) — refreshed
  via the `PersonRenamed` event
- **memory records** — deliberately NOT rewritten (names live inside
  free text); the tool response tells the agent to save a memory noting
  the change

### Reconcile Flags and the Merge Nudge

The nightly identity reconcile (`perception/reconcile.py`, runs in the
dream window) flags person-record pairs that are probably the same
human (small name edit distance and/or high centroid similarity). The
audit report is persisted to `data/perception/id-reconcile-latest.json`;
the agent reads it with `identify_person(action="list_flags")`. Newly
flagged pairs also create a one-line `[id-reconcile]` to-do (deduped by
description, suppressed once completed), so the `[To-do: N]` status
line nudges the agent to investigate. Merging itself is never
automatic — the agent confirms with the household first, then calls
`action="merge"`.

### What This Tool Does NOT Do

- **Query who is present** — presence is injected into conversation
  context automatically via the `[Present: ...]` header
- **Return embeddings or distances** — the agent operates at the
  semantic level only
- **Trigger perception** — perception runs autonomously; this tool only
  names or links people

## Person-Triggered Wake Events

The perception pipeline enables a key capability: waking the agent when
a specific person is detected, without requiring them to speak.

### Flow

```
1. Agent schedules: "Remind Jacob about doctor at 3pm"
   (via manage_tasks tool with person="Jacob" trigger condition)

2. Perception runs continuously:
   DORMANT → motion → CHECKING → YOLO → person detected

3. ReID embedding compared to known centroids:
   cosine_sim(embedding, Jacob_centroid) = 0.89 → confident match

4. Scheduler checks: pending trigger for "Jacob"? → YES

5. Agent wakes → speaks: "Hey Jacob, your doctor appointment is at 3pm"

6. Cooldown: same trigger won't re-fire for configurable period
   (default 30 minutes) unless explicitly reset
```

### Confidence Requirements

Person-triggered wake events require **high visual confidence (>0.85)**
since there's no voice confirmation at this point (the person hasn't
spoken yet). This is a read-only operation on the cloud — no new
embeddings are written.

If visual confidence is only medium (0.60-0.85), the trigger is held
until either:
- The person speaks (voice confirmation raises confidence), or
- They leave and the trigger stays pending for next detection

This prevents BB from delivering a private reminder to the wrong person.

## Data Storage

### Person Record

```
persons table (SQLite):
  id:              uuid
  name:            str
  first_seen:      datetime
  last_seen:       datetime
  is_user:         bool        # admin-approved via registration flow
  merged_into:     uuid | null # soft merge tombstone — set when this
                               # record was merged into another person;
                               # name lookups follow the chain

visual_embeddings table:
  id:              uuid
  person_id:       uuid (FK → persons)
  embedding:       blob (512 × float32 = 2KB)
  timestamp:       datetime
  voice_confirmed: bool
  crop_path:       str | null  # expires per retention policy

voice_embeddings table:
  id:              uuid
  person_id:       uuid (FK → persons)
  embedding:       blob (192 × float32 = 768 bytes)
  timestamp:       datetime

centroids table:
  person_id:       uuid (FK → persons)
  visual_centroid: blob (512 × float32)
  voice_centroid:  blob (192 × float32)
  updated_at:      datetime
```

At household scale (3-10 people, 200 visual + 50 voice embeddings
each), the entire embedding database fits in ~1-2MB. Centroid comparison
is O(n_persons) per detection — effectively instant.

### Storage Locations

```
data/
  perception/
    perception.db        # SQLite: persons, embeddings, centroids
    crops/               # temporary crop images (retention policy)
      2026-02-21/
        {uuid}.jpg       # 100-250px person crops
    models/              # Hailo HEF files
      yolov5s_personface_h8l.hef
      repvgg_a0_person_reid_512.hef
```

## Privacy

- All inference runs on-device (Hailo NPU + CPU)
- Camera frames are processed and immediately discarded — only
  embeddings (abstract float vectors) are stored long-term
- Crop images are retained briefly for debugging, then deleted
- Audio is processed by pyannote on-device; only transcripts are sent
  to the STT API (ElevenLabs)
- No perception data leaves the box except explicit API calls
- Person records are local — no cloud sync of identity data
