# Plan: Onboarding + person-ID bootstrap (voice + visual)

## Goal
BB should recognize household members and gracefully handle the three identity
events that matter: **first meeting**, **confirmation via normal conversation**,
and **correction** when it gets it wrong. All three should be the same code
path, not bespoke tools layered on top of each other.

Behavioral target:
- Unrecognized speaker → BB greets, asks their name, captures a visual reference
  from the camera (bbox-cropped on whoever is speaking)
- Known speaker → BB addresses them by name, and if it got it right, that
  correctness is silently reinforced (their cloud of embeddings grows)
- BB got it wrong ("I'm not Carina, I'm Sarah") → the correction naturally
  redirects this session's embeddings to Sarah and away from Carina, tightening
  both centroids

The design principle is that `identify_person(name, speaker_label)` is the
*only* tool the agent needs for identity events. Its behavior depends on what
the system already believes about `speaker_label` in this session.

## Current state (what already works)

### Voice + transcript path
- Diarization is live (pyannote), segments carry speaker labels
- `voice.py:_build_attributed_transcript` (lines 616-668) produces `[Name]: text` format
- `SpeakerIdentified` event updates label → name mapping (`agent.py:299-312`)
- `TranscriptReady` carries `speaker_segments` with embeddings, but those don't propagate to the agent — only the attributed text string does

### Perception
- `perception/voice_reid.py:match()` — tier `"high"` / `"unknown"` vs known centroids
- `perception/visual_reid.py:match()` — `"high"` / `"medium"` / `"unknown"`
- `perception/enrollment.py:buffer_embedding(ref, embedding)` — queues visual embeddings under a session-local label
- `perception/enrollment.py:identify(name, ref)` — commits buffered visual embeddings to a named person
- **Gap**: no voice-embedding equivalent (buffer / commit / centroid)
- **Gap**: no session-claim state tracking what BB *told the user* — so corrections can't distinguish "new name for this speaker" from "rename the whole person"

### DOA ↔ camera
- `perception/doa.py:angle_to_camera_x(angle)` — 0-359° → normalized `[-1, 1]`
- `perception/doa.py:associate_speaker_to_detection(angle, detections)` — picks nearest person-bbox
- `perception/pipeline.py` fusion uses DOA to cross-check voice→visual match — so the math and cross-check exist; what's missing is exposing them to a tool

### Camera
- `hardware/camera.py:capture_frame()` (149-159) — current main stream, 1280×720 RGB
- `hardware/camera.py:capture_photo()` (161-182) — 12MP still (mode-switch overhead)
- **Gap**: no public "latest YOLO detections for this frame" helper (YOLO runs async inside perception pipeline)

### Tools
- `identify_person` (`tools/builtins/identify_person.py:50-104`) — takes `name` + `ref`, commits buffered visual embeddings to a person. No voice-embedding mode, no session-claim awareness, no camera capture.

## Unified identity lifecycle

The heart of this plan is that `identify_person` becomes smart about context.
It does one of five things based on the state of `speaker_label` and `name`:

| State at call time | Action | Example |
|---|---|---|
| No prior claim, name doesn't exist | **Create** new person; seed centroids from buffered embeddings | First meeting with Brian |
| No prior claim, name exists | **Confirm** match; add buffered embeddings to existing person | Agent just says `identify_person(Jacob, S0)` on a verified `high`-match speaker |
| Prior claim matches name | **No-op** (already correct) | Agent re-acknowledges mid-conversation |
| Prior claim differs, name exists | **Correct**: redirect pending commit from old → existing new person | "I'm Sarah, not Carina" |
| Prior claim differs, name doesn't exist | **Rename**: the session's speaker was misnamed; if the prior claim was from a voice-ReID match with existing stored embeddings, this is ambiguous. Default: treat as new person (conservative). Admin can merge later if needed | "Call me Bri" where Bri doesn't exist yet |

### Session-claim state (new)

`EnrollmentManager` gains session-scoped state:

```python
class SessionClaim(TypedDict):
    person_id: int
    name: str
    source: Literal["voice_reid_match", "visual_reid_match", "agent_identify", "unknown"]
    match_tier: Literal["high", "medium", "unknown"] | None
    established_at: float  # monotonic

class EnrollmentManager:
    session_claims: dict[str, SessionClaim]            # speaker_label -> claim
    session_voice_buffer: dict[str, list[Embedding]]   # speaker_label -> embeddings
    session_visual_buffer: dict[str, list[Embedding]]  # speaker_label -> embeddings
```

At session start, for each speaker that voice ReID matches, a `SessionClaim` is
seeded with `source="voice_reid_match"` and the appropriate tier. Agent calls
to `identify_person` overwrite the claim with `source="agent_identify"`.

### Commit semantics (end of session)

When the session ends, for every `speaker_label`:
- If a claim exists, commit buffered voice + visual embeddings to that person
- Provenance is recorded on each embedding: `(session_id, speaker_label, match_tier_at_commit, source_that_established_claim)`
- If no claim exists (speaker was `unknown` and agent never identified them) → embeddings are dropped, NOT stored as a floating "Person X" record

This last point matters: we don't create orphan person records for unacknowledged strangers. A stranger who never got introduced leaves no trace in the identity store, which matches the privacy principle.

### Reinforcement + correction in the same code path

"Hi Carina" (mid-session agent utterance) is derived from the session claim. If
the user doesn't correct it, the claim stands → commit reinforces Carina's
centroid. If the user corrects, the claim updates → commit goes to Sarah
instead. The distinction is entirely in session-claim state; the commit code
doesn't know (or need to know) which happened.

### Cross-session miscorrection (out of scope for V1)

"You've been getting my name wrong for a week" is not fixable by redirecting
the current session's embeddings. Options:
- **Embedding decay** (already designed per project memory: access-based
  retention, 6mo for person) — wrong embeddings quietly age out over time
- **Admin merge/reattribute CLI** — out-of-band, outside the conversation,
  operates on provenance-tagged embeddings
- **Per-embedding confidence tagging** — low-confidence commits (medium-tier
  matches) can be preferentially pruned when corrections pile up

V1 relies on the first two; adds provenance tagging so option 3 is available
later. No new CLI this round.

## Proposed conversation flow

```
Session starts; diarization creates SPEAKER_00, captures voice embeddings.
   ↓
VoiceReID.match(E_v) → "unknown"
   ↓
EnrollmentManager.session_claims[SPEAKER_00] = {source: "voice_reid_match", person_id: None, match_tier: "unknown"}
   ↓
TranscriptReady event enriched with unknown_speaker_labels: ["SPEAKER_00"]
   ↓
Agent context includes: "A speaker (SPEAKER_00) is present but not recognized."
   ↓
Agent: "Hi — I don't think we've met. What's your name?"
   ↓
Speaker: "I'm Brian"
   ↓
Agent calls capture_speaker_reference(speaker_label="SPEAKER_00")
   ↓
Main process:
   1. Read DOA angle from microphone
   2. Look up latest YOLO detections from pipeline
   3. associate_speaker_to_detection(doa, detections) → best bbox
   4. camera.capture_frame() → full frame
   5. Crop bbox → image
   6. Buffer visual embedding from the bbox crop into session_visual_buffer[SPEAKER_00]
   7. Return {image_base64, bbox, detection_confidence, doa_angle}
   ↓
Agent sees the image (Opus 4.7 vision block) in its next turn context
   ↓
Agent calls identify_person(name="Brian", speaker_label="SPEAKER_00")
   ↓
EnrollmentManager.identify():
   - session_claims[SPEAKER_00].source != "agent_identify" → agent is establishing canonical claim
   - person "Brian" doesn't exist → CREATE
   - session_claims[SPEAKER_00] = {person_id: <new>, name: "Brian", source: "agent_identify", ...}
   ↓
Agent: "Nice to meet you, Brian. I'll remember you."

[Session ends after some conversation]
   ↓
EnrollmentManager.commit_session():
   - For SPEAKER_00: commit buffered voice + visual embeddings to Brian
   - Tag provenance on each: (session_id, "SPEAKER_00", "unknown" tier at start, "agent_identify" source)
```

### Correction variant

```
Session starts; voice ReID matches SPEAKER_00 → Carina (tier: "high")
   ↓
session_claims[SPEAKER_00] = {person_id: carina.id, name: "Carina", source: "voice_reid_match", match_tier: "high"}
   ↓
Transcript: "[Carina]: hey BB, I'm home"
   ↓
Agent: "Hi Carina, welcome back"
   ↓
Speaker: "I'm actually Sarah — you always get this wrong"
   ↓
Agent calls identify_person(name="Sarah", speaker_label="SPEAKER_00")
   ↓
EnrollmentManager.identify():
   - prior claim: Carina (voice_reid_match, tier=high)
   - target: Sarah (exists)
   - CORRECTION: session_claims[SPEAKER_00] = {person_id: sarah.id, name: "Sarah", source: "agent_identify", ...}
   - Log correction event for telemetry
   ↓
Agent: "Sorry Sarah, got that wrong. I'll remember."

[Session ends]
   ↓
commit_session():
   - SPEAKER_00's buffered embeddings commit to Sarah, NOT Carina
   - Carina's centroid unchanged
   - Sarah's centroid tightened (or created if first confirmation)
```

The cross-session problem (Carina's cloud still contains past wrong embeddings
that matched SPEAKER_00) is out of scope; decay will handle it over weeks.

## New / changed APIs

### `src/boxbot/perception/enrollment.py`
- Add `session_claims` and `session_voice_buffer`
- New: `buffer_voice_embedding(speaker_label, embedding)`
- Rework `identify(name, speaker_label)`:
  - Returns `IdentifyOutcome` enum: `CREATED | CONFIRMED | NO_OP | CORRECTED | RENAMED`
  - Updates `session_claims`
  - Does NOT immediately commit embeddings
- New: `commit_session(session_id)` — flushes buffers to person clouds, tags provenance
- New: `on_voice_reid_match(speaker_label, person_id, tier)` — seeds claim from perception layer
- Keep legacy `buffer_embedding(ref, embedding)` (visual) but route through the new internals

### `src/boxbot/perception/clouds.py` (or equivalent centroid store)
- New: `add_voice_embedding(person_id, embedding, provenance)`
- New: `recompute_voice_centroid(person_id)`
- Modify: `add_visual_embedding` to accept `provenance` dict
- Symmetric to existing visual methods

### `src/boxbot/tools/builtins/identify_person.py`
- Add `speaker_label` param (replaces / supplements the old `ref` name)
- Route through the reworked `EnrollmentManager.identify`
- Response surfaces `IdentifyOutcome` so the agent knows whether it created, confirmed, corrected, etc. — useful for natural language feedback ("got it, I'll remember you" vs "sorry about that")

### `src/boxbot/tools/builtins/capture_speaker_reference.py` (new)
- Input: `speaker_label`
- Logic: DOA → detection lookup → camera capture → crop → buffer embedding + return image
- Output: image base64 + bbox + metadata (so agent can comment naturally: "Got it — I see you in the blue shirt")
- Fallback: if DOA is None (ReSpeaker USB issue) and only one person is visible → use that bbox. If multiple and no DOA → return `{ok: false, reason: "no_speaker_localization"}` and let the agent handle verbally.

### `src/boxbot/core/events.py`
- Enrich `TranscriptReady` with `unknown_speaker_labels: list[str]`
- Populated in `voice.py` from voice-ReID results on the utterance's segments

### `src/boxbot/core/agent.py`
- Include `unknown_speaker_labels` in the turn context so the agent is aware
- When `capture_speaker_reference` returns an image, surface it as an image content block in the next turn (pairs with Opus 4.7 vision)

## System prompt additions

In the system prompt builder (`agent.py:65-94` base, `_build_system_prompt`
~544-628 for dynamic pieces), add a short identity section:

```
## Identity awareness

Every utterance you see is attributed: "[Carina]: ..." or "[Unknown_1]: ...".
"Unknown" means voice ReID didn't find a match — introduce yourself, ask their
name, and call capture_speaker_reference(speaker_label) while they're still in
frame, then identify_person(name, speaker_label).

When a person corrects an identity ("I'm not Carina, I'm Sarah"), just call
identify_person(name="Sarah", speaker_label) with the corrected name. The
system handles the redirection automatically. Apologize naturally, then
continue. Don't over-explain; don't promise complex fixes.

Never expose internal labels (SPEAKER_00, Unknown_1) to the user.
```

## Files touched

- `src/boxbot/core/events.py` — enrich `TranscriptReady`
- `src/boxbot/communication/voice.py` — populate `unknown_speaker_labels`
- `src/boxbot/core/agent.py` — context injection, image-block handling on tool results
- `src/boxbot/perception/enrollment.py` — session claims, commit semantics, voice buffer
- `src/boxbot/perception/clouds.py` — voice centroid APIs, provenance on adds
- `src/boxbot/perception/pipeline.py` — public `get_recent_detections()`
- `src/boxbot/tools/builtins/capture_speaker_reference.py` (new)
- `src/boxbot/tools/builtins/identify_person.py` — accept `speaker_label`, return outcome enum
- `src/boxbot/tools/__init__.py` — register new tool

## Explicitly NOT done in V1 (deferred)

- Cross-session embedding reattribution ("you've had me wrong for a week")
- Admin merge CLI for splitting/joining person records
- Person-user linkage column on the `users` table (WhatsApp-registered users → person records)
- Multi-unknown-speaker simultaneous enrollment (supported accidentally via the design, but not a priority flow)

## Open questions

1. Where is the "most recent YOLO detections" cache held in `perception/pipeline.py`? Needs a short code-read before implementing `get_recent_detections()`.
2. Voice ReID's `"unknown"` threshold is 0.60 — too eager or too loose? Ship and tune from real data.
3. `session_id` currently... is there one? `voice.py` uses `_conversation_id` — reuse that as the session id for provenance.
4. When `capture_speaker_reference` produces a visual embedding, should it go into `session_visual_buffer` immediately, or wait for `identify_person`? Buffer immediately — that way if the agent never calls `identify_person`, we just drop buffers on session end (no orphan record, matches the privacy principle).

## Effort

Roughly a full day. Biggest unknowns: pipeline-state detection cache shape, and whether the commit-at-session-end model plays nicely with the existing async enrollment flow.
