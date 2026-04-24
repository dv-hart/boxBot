# perception/

The perception pipeline — identifies who is present and what they're
saying. Uses voice identification to bootstrap visual re-identification,
eliminating the need for explicit face enrollment.

See [docs/perception.md](../../../docs/perception.md) for the full
design, state machine, and rationale.

## Architecture

```
DORMANT ──► Motion Detect (CPU, 5-10 FPS, frame diff)
                    │
                    ▼ motion detected
CHECKING ──► Person Detection (Hailo, YOLOv5s-personface, ~26ms)
                    │
                    ▼ person confirmed
DETECTED ──► ReID Embedding (Hailo, RepVGG-A0, <1ms)
             ──► Centroid Match (CPU) → tentative label
             ──► Lazy-load pyannote
                    │
                    ▼ wake word / speech
CONVERSATION ──► pyannote Diarization (CPU)
               ──► Speaker Embedding Match (CPU)
               ──► DOA Tracking (ReSpeaker)
               ──► Hailo: FREE
                    │
                    ▼ conversation ends
POST ──► Store confirmed embeddings, recompute centroids
       ──► Prune expired crops → DORMANT
```

The Hailo NPU does a few seconds of work per interaction (detection +
ReID), then sits idle for the entire conversation. Voice identification
runs on CPU. The two workloads are cleanly separated.

## Files

### `state_machine.py`
The perception state machine: DORMANT → CHECKING → DETECTED →
CONVERSATION → POST_CONVERSATION. Manages transitions, timeouts, and
periodic re-checks. Emits events to the event bus on state changes.

### `motion.py`
CPU-based motion detection using frame differencing on low-res
grayscale frames. Runs at 5-10 FPS during DORMANT state. Triggers
CHECKING state when motion exceeds threshold. Configurable sensitivity
with Gaussian blur pre-filtering to reduce noise.

### `person_detector.py`
YOLOv5s-personface detection running on Hailo. Invoked on-demand when
motion is detected (not continuously). Returns bounding boxes for
detected persons and faces. A single inference takes ~26ms. Face
detections are available for future face-based ReID.

### `visual_reid.py`
RepVGG-A0 visual re-identification running on Hailo:
- Generates 512-dim embedding vectors from person crops (256x128 input)
- Sub-millisecond inference on Hailo-8L (<1ms per crop)
- Compares against known centroids using cosine similarity
- Returns tentative label with confidence score
- Does NOT write to embedding clouds (see confirmation rule below)

### `speaker_id.py`
Speaker identification using pyannote.audio:
- **Diarization** — who spoke when, with timestamps, handles
  overlapping speech from multiple speakers
- **Speaker embeddings** — extracted as part of diarization, matched
  against known voice profiles via cosine similarity
- **Lazy loading** — models loaded on person detection (DETECTED state),
  warm by the time conversation starts (~2-3s load time)
- **Multi-speaker** — properly attributes utterances when multiple
  people talk to BB simultaneously

### `doa.py`
Direction of arrival tracking using the ReSpeaker XVF3000 4-mic array:
- Maps DOA angle to camera FOV position
- Determines if speaker is within or outside camera FOV
- Associates voice with correct bounding box in multi-person scenes
- Resolves visual-voice conflicts (speaker beside BB, different
  person in camera)

### `fusion.py`
Cross-modal identity fusion and the confirmation gate:
- Combines visual ReID and speaker ID results
- **Confirmation rule:** visual embeddings are only stored in a
  person's cloud when confirmed by voice or agent context
- Voice-confirmed: pyannote matches speaker to known voice profile
  during the same session → visual embedding added to cloud
- Agent-confirmed: agent calls `identify_person` tool → backend
  adds associated visual embeddings to the named person's cloud
- Visual-only matches can read clouds (for triggers) but never write
- Handles conflict resolution using DOA

### `clouds.py`
Embedding cloud management:
- Per-person storage of visual (512-dim) and voice (192-dim) embeddings
- Centroid computation and updates
- Pruning: oldest-first, capped at 200 visual / 50 voice per person
- Natural drift: dropping old embeddings lets the cloud adapt to
  appearance changes over time
- Crop image retention: 1 day (normal) / 7 days (debug mode)

### `enrollment.py`
Person record creation and identification. Called by the `identify_person`
tool when the agent names or identifies someone:
- `create_person(name, ref)` — new person record, stores ref's
  embeddings from current session
- `link_person(name, ref)` — links ref's embeddings to existing person
  record (e.g., "It's me, Jacob" when voice doesn't match)
- The agent provides semantic labels; this module handles all embedding
  bookkeeping

### `models/`
Model files and conversion scripts for Hailo deployment:
- YOLOv5s-personface detection (`yolov5s_personface_h8l.hef` — pre-compiled,
  from `/usr/share/hailo-models/`)
- RepVGG-A0 person ReID (`repvgg_a0_person_reid_512.hef` — 256x128 input,
  512-dim output)
- pyannote models are managed by the pyannote library (PyTorch,
  downloaded on first use, cached)

## Embedding Database

Stored in `data/perception/perception.db` (SQLite):
- `persons` — name, first_seen, last_seen, is_user, whatsapp_number
- `visual_embeddings` — 512-dim float vectors, timestamped, with
  voice_confirmed flag
- `voice_embeddings` — 192-dim float vectors, timestamped
- `centroids` — precomputed per person, updated on cloud changes

At household scale (3-10 people), centroid comparison is O(n_persons)
per detection — effectively instant. The full database fits in ~1-2MB.

## Conversation Input

The pipeline's output is injected into the agent's conversation input
as attributed text. The agent never sees embeddings, distances, or
confidence scores:

```
[Present: Jacob (confirmed), Person B (new)]

[Jacob]: Hey BB, what's on the calendar tomorrow?
[Person B]: Hey, nice to meet you!
```

If the agent learns Person B's name, it calls `identify_person` to
create the record. The backend handles all embedding storage.

## Privacy

- All inference runs on-device (Hailo NPU + CPU)
- Frames are processed and discarded — only embeddings are stored
- Crop images retained briefly for debugging, then deleted
- Audio diarization runs on-device via pyannote
- Only transcripts are sent to the STT API (not raw audio embeddings)
- No perception data leaves the box except explicit API calls
