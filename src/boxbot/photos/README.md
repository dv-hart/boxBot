# photos/

Photo management — ingesting, tagging, searching, and displaying the
household photo library. Photos arrive via WhatsApp (or camera capture),
pass through an async intake pipeline, and are stored locally with rich
metadata for search and slideshow display.

## Architecture

Photos are stored locally in `data/photos/` with metadata in SQLite
(`data/photos/photos.db`). The intake pipeline handles resizing, person
detection, and AI-powered description/tagging. A shared search backend
serves both the `search_photos` tool and the `boxbot_sdk.photos` API.

```
Photo arrives (WhatsApp, camera, future upload)
  │
  ▼
Intake Queue
  │
  ▼ (when system is idle)
Intake Pipeline
  ├── 1. Resize to max resolution (default 1920x1080)
  ├── 2. YOLO person detection (Hailo NPU)
  ├── 3. ReID matching for each detected person
  ├── 4. Small model: description + tags + household people
  │      (receives person annotations, existing tag library, and —
  │       with photos.infer_people — sender, caption, household roster)
  ├── 5. Embed description (MiniLM, 384-dim)
  └── 6. Store image + metadata in SQLite
           │
           ├── File: data/photos/YYYY/MM/{uuid}.jpg
           └── DB:   data/photos/photos.db
```

## Files

### `manager.py`
Photo storage, metadata CRUD, and lifecycle management:
- Store photos in organized directory structure (`data/photos/YYYY/MM/`)
- Metadata CRUD: description, tags, people, slideshow membership
- Tag library management: merge, rename, delete tags
- Soft delete with configurable retention (default 30 days)
- Restore soft-deleted photos
- Storage quota enforcement (configurable % of disk)
- Optional cloud backup to AWS S3 (disabled by default)

### `intake.py`
Async intake pipeline — processes incoming photos when the system is idle:

1. **Resize** — scale to configurable max resolution (default 1920x1080),
   preserving aspect ratio. Determines orientation (landscape, portrait,
   square) and stores dimensions. The decoder is bounded by
   `Image.MAX_IMAGE_PIXELS` (decompression-bomb guard) and by
   `photos.max_ingest_bytes` (source-size cap), both enforced before the
   bytes reach PIL
2. **Person detection** — delegated to an injected detector
   (`PeopleDetector`) that wraps the perception pipeline's Hailo YOLOv8n +
   OSNet ReID and matches detected boxes against known person clouds. The
   seam is **optional**: when no detector is wired (e.g. dev, or before
   the on-device adapter is enabled) the pipeline stores photos without
   person annotations. Detector errors are swallowed — a photo is worth
   keeping even if detection hiccups
3. **Tagging** — send the (downscaled) image to the small model with any
   person annotations and the existing tag library. The model returns a
   1-2 sentence description and lowercase tags, preferring existing tags.
   With `photos.infer_people` (default on) and a non-empty roster it also
   names household people — see "People inference" below.
   Any failure (no API key, API error) degrades to the human-supplied
   caption so search still has something to match
4. **Embedding** — embed the description text with MiniLM (384-dim) for
   hybrid search
5. **Storage** — re-check quota, save resized image to disk, write
   metadata to SQLite, add to slideshow by default

**Status:** resize, tagging (incl. people inference), embedding, and
storage are live. ReID person detection runs only when an on-device
`PeopleDetector` is injected at construction — wiring the
perception-backed adapter (and validating Hailo contention against the
live pipeline) is on-device follow-up work. People labels do not depend
on it.

**Backpressure & quota:** `enqueue` rejects oversize sources, refuses to
queue beyond `photos.max_intake_queue_depth`, and checks quota up front;
the worker re-checks quota at write time (a backlog can cross the line
between enqueue and write). Quota blocks at `photos.quota_block_percent`.

**Hailo priority (planned):** Live perception should always preempt photo
intake. The on-device detector adapter must acquire the shared Hailo lock
so intake yields the NPU to live perception. (The shared-semaphore /
priority-preemption layer is still v1 — lock fairness only.)

**Queue processing:** Multiple incoming photos are processed sequentially
from the intake queue. The pipeline is fault-tolerant — a failure on one
photo does not block the rest.

### `people.py`
Name / photo-id / note validation for the people handlers, and the
mapping from a name onto the perception person record (case-insensitive;
the perception spelling and `person_id` win when a record exists, and a
person without one is still allowed).

### `backfill.py`
`python3 -m boxbot.photos.backfill [--apply] [--limit N] [--photo ID]` —
re-runs people inference over existing photos. Dry-run by default; one
small-model call per photo; idempotent; skips human-confirmed photos;
never touches descriptions or tags. No caption exists at backfill time,
so labels come out `likely` at best and are meant to be reviewed
(`bb.photos.unconfirmed()`).

### `search.py`
Shared search backend used by both the `search_photos` tool and
`boxbot_sdk.photos.search()` — one function (`search_photos`), one set
of defaults (`DEFAULT_LIMIT`).

**Scoring** (query given), per photo:

| Signal | Weight | Notes |
|---|---|---|
| vector | 0.6 x cosine(query, description) | negatives clipped to 0 |
| keyword | 0.4 x (0.5 + 0.5 x normalised BM25) | FTS5 over description, tags, people; any literal hit earns >= 0.2 |
| person | 1.0 x match | per name: confirmed/certain label 1.0, likely label 0.8, name in tags/description 0.6 |

- A **person name is a soft match** across people labels, tags and
  description — never a hard filter on `photo_people`. Names come from
  `people=[...]` or from a roster/label name appearing in the query.
  With `people=[...]` a photo must mention at least one of the names
  somewhere. `unsure` suggestions never match.
- **Relevance floor** — `photos.search_min_score` (default 0.2). Results
  below it are dropped, so an unrelated query returns nothing. A keyword
  hit (>= 0.2) and a name match (>= 0.6) always clear it; a meaning-only
  match needs cosine >= ~0.33. No query and no people = newest first, no
  floor.
- **Query sanitising** — word tokens only, request filler ("photos",
  "of", "show", ...) removed, each term quoted and OR-ed, so user
  punctuation cannot produce an FTS syntax error. A real FTS fault is
  logged with a traceback.
- **Hard filters** — tags (AND), date range, source, slideshow.
- **Empty result → `hint`, not a broadened result set.** The hint names
  the single filter whose removal finds photos ("without tags: 3").
  Broadening silently is the unsafe choice: with `show=True` the relaxed
  set would go straight onto the screen as "photos of X".
- **Rows are slim** — id, short description, tags, people (+confidence),
  created date, source, sender, score. The full record is `get`.
- **`show=True`** dispatches the `picture` display with the result ids in
  the same call.

**FTS index** — standalone FTS5 table `photos_fts(photo_id UNINDEXED,
description, tags, people)`, maintained from `PhotoStore._reindex_photo`
on every write that touches those fields. `PhotoStore.initialize()`
migrates the old description-only external-content table (drops its
triggers, recreates, rebuilds once).

**DRY architecture:**
```
src/boxbot/photos/search.py                        <- search backend
    │
    ├── src/boxbot/tools/builtins/search_photos.py <- core tool
    │     Agent calls directly during conversation.
    │
    └── src/boxbot/sdk/photos.py                   <- SDK function
          Sandbox scripts call boxbot_sdk.photos.search().
          Emits JSON action -> main process routes to search backend.
```

### Slideshow selection
Slideshow mode for the `picture` display is driven by the
`in_slideshow` flag (set via `add_to_slideshow` / `remove_from_slideshow`,
default on at intake). When the display manager switches to `picture`
with no explicit `image_ids`, it selects the slideshow-enabled set
(active, not soft-deleted, newest first) and rotates through it on a
timer — see `DisplayManager._load_slideshow_ids` and
`_maybe_start_slideshow` in `src/boxbot/displays/manager.py`. Transition
effects and rendering are handled by the `picture` display module.

### `maintenance.py`
Daily background job (extends the system-wide maintenance schedule):
- **Purge expired soft-deletes** — delete files and DB records for photos
  where `deleted_at` is older than the retention window
- **Recalculate storage usage** — update cached quota metrics
- **Rebuild FTS index** — keep full-text search consistent

## Intake Pipeline Details

### Person Detection in Photos

The intake pipeline reuses the perception system's models (YOLOv8n +
OSNet-AIN-x1.0) for static image analysis. This is a different call path
from real-time perception — it's a batch operation on a single image.

**Identified person:** If the ReID embedding matches a known person cloud
(cosine similarity above threshold), the photo is tagged with that
person's ID and name.

**Unknown person:** If no match is found, the person is tagged as
"unknown person" with bounding box coordinates stored. The agent can
later update the label via the SDK if told who it is. No new person
records are created from photos — person creation remains the domain of
the live perception pipeline and the `identify_person` tool.

**Important:** Photos do not write to person embedding clouds. The
perception principle "visual-only matching can read clouds but never
writes to them" extends to photos. Photo-based embeddings are used for
matching only, then discarded.

### Small Model Tagging

The small model receives:
- The image
- Person annotations from ReID, when a detector is wired
- The existing tag library (full list of current tags)
- With people inference: the sender's name, the caption, and the
  household roster

It returns (forced tool call `describe_photo`) a 1-2 sentence
description, lowercase tags (preferring the library), and `people`.
New tags are created in the tag library automatically.

### People inference (caption + household roster)

No face registration. The tagger names household people from context:
the caption first, then appearance against a short roster.

**Roster** — table `people_notes(name, note, person_id)` in `photos.db`:
one row per household person with a one-line appearance/context note
(age, hair, glasses, usual context). It lives in the photo store because the
tagger is its only reader; a small validated table beats scraping
free-text memories, and a person the box has never seen or heard still
needs an entry. The agent edits it through `bb.photos.roster()` /
`set_roster()` / `remove_from_roster()`. On write the name is matched to
a perception person case-insensitively (their spelling and `person_id`
are stored); identity renames/merges carry over via `repoint_person`.
An empty roster means nobody is named.

**Rules** (prompt `_TAGGER_PEOPLE`, enforced again in code by
`_parse_people`):
- Only roster names — the tool schema's `name` is an enum of the roster
  and anything else is dropped. Strangers are never named.
- `certain` only when the **caption itself** names the person;
  otherwise it is lowered to `likely`. Appearance is `likely` at best.
- `unsure` is kept as a suggestion (`PhotoRecord.suggestions`), not as a
  person: not searchable, not in the FTS index.
- Caption, sender, roster notes and stored notes are untrusted data:
  flattened to one bounded line, angle brackets neutralised, wrapped in
  delimiters, and the prompt says to treat them as evidence only.

**Storage** — `photo_people` gains `confidence`
(`confirmed | certain | likely | unsure`), `source`
(`human | tagger | detector`) and `basis`; `photos.people_confirmed_at`
marks a human-reviewed photo.

**Confirmation** — `bb.photos.unconfirmed()` lists unreviewed photos
with `likely`/`unsure` labels; `confirm_people(id)` accepts the labels
as they stand; `set_people(id, names=[...])` replaces them (add, remove,
correct — no detected slot needed). Both mark the photo confirmed.
`PhotoStore.apply_inferred_people` — the only path the tagger writes
through — refuses a confirmed photo, so human labels are never
overwritten.

**Privacy** — labels and roster are household data and stay in
`photos.db`. Nothing new leaves the box: the one existing small-model
tagging call per photo now also carries the roster and the caption.

### Sender Context

The sender's identity is recorded as metadata (`sender`) and passed to
the tagger as context (it helps read captions like "me and the kids"),
but is never by itself evidence that the sender is in the photo.

## Tag Library

Tags are stored in a dedicated `tags` table — a flat, managed vocabulary.
The small model receives the full tag list during tagging and prefers
existing tags over creating new ones.

### Properties
- **Flat** — no hierarchy, no nesting
- **Normalized** — lowercase, trimmed, deduplicated on create
- **Case-insensitive** — `COLLATE NOCASE` on the name column

### Management (via SDK)
- **Merge** — reassign all photos from source tag to target tag, then
  delete the source. Use for consolidating synonyms ("sunsets" into
  "sunset")
- **Rename** — change a tag's name across all photos
- **Delete** — remove a tag and its associations (photos remain, only
  the tag link is removed)

Tag management is available only through the SDK
(`boxbot_sdk.photos.merge_tags()`, etc.), not as a direct agent tool.
The agent can run tag cleanup via `execute_script` when appropriate.

## Soft Delete

Photos deleted via the SDK or agent are **soft-deleted**: the `deleted_at`
timestamp is set but the file and metadata are preserved.

- **Retention window:** Configurable, default 30 days
- **Hidden from search:** Soft-deleted photos are excluded from default
  search results and slideshow rotation
- **Visible with flag:** The `search_photos` tool and SDK support
  `include_deleted=true` to surface soft-deleted photos
- **Restorable:** `photos.restore(photo_id)` clears `deleted_at` and
  returns the photo to active status
- **Quota impact:** Soft-deleted photos **count against the storage
  quota**. This prevents restore-after-delete from unexpectedly exceeding
  the quota
- **Purge:** The daily maintenance job permanently deletes photos where
  `deleted_at` is older than the retention window (file removed from disk,
  DB record deleted)

## Storage Quota

Photo storage is capped at a configurable percentage of disk capacity
(default 50%). This prevents photos from consuming all available space
on the Pi's SD card.

### What's Counted
- All image files in `data/photos/` (active + soft-deleted)
- The quota is measured against the filesystem where `data/photos/`
  resides (handles external storage mounts correctly)

### Quota Enforcement
- Checked during the intake pipeline before storing a new photo
- If quota would be exceeded, the photo is **rejected** with a clear
  message to the agent:
  ```
  Photo storage is at 98% of quota (15.2 GB / 16.0 GB).
  Cannot save new photos. Ask the user what to remove.
  ```
- The agent relays this to the user and can suggest deletions based on
  search results (oldest photos, least-viewed tags, etc.)
- Storage info is available via SDK: `photos.storage_info()`

### Configuration
```yaml
photos:
  max_storage_percent: 50          # % of disk for photo storage
```

## Database Schema

All photo metadata is stored in `data/photos/photos.db` (separate from
the memory database to simplify quota management).

```sql
CREATE TABLE photos (
    id              TEXT PRIMARY KEY,    -- UUID
    filename        TEXT NOT NULL,       -- relative path: YYYY/MM/{uuid}.jpg
    source          TEXT NOT NULL,       -- 'whatsapp', 'camera', 'upload'
    sender          TEXT,                -- person name if from messaging
    description     TEXT,                -- small-model generated, 1-2 sentences
    orientation     TEXT,                -- 'landscape', 'portrait', 'square'
    width           INTEGER,            -- pixels (after resize)
    height          INTEGER,            -- pixels (after resize)
    file_size       INTEGER NOT NULL,   -- bytes
    in_slideshow    INTEGER DEFAULT 1,  -- 1 = included in slideshow rotation
    created_at      TEXT NOT NULL,       -- ISO 8601
    deleted_at      TEXT,                -- soft delete timestamp, NULL = active
    updated_at      TEXT NOT NULL,
    embedding       BLOB                -- 384-dim float32 (MiniLM, on description)
);

CREATE TABLE tags (
    id              INTEGER PRIMARY KEY,
    name            TEXT UNIQUE NOT NULL COLLATE NOCASE,
    created_at      TEXT NOT NULL
);

CREATE TABLE photo_tags (
    photo_id        TEXT REFERENCES photos(id),
    tag_id          INTEGER REFERENCES tags(id),
    PRIMARY KEY (photo_id, tag_id)
);

CREATE TABLE photo_people (
    id              INTEGER PRIMARY KEY,
    photo_id        TEXT REFERENCES photos(id),
    person_id       TEXT,                -- FK to persons.id if identified, NULL if unknown
    label           TEXT NOT NULL,       -- "Jacob", "unknown person"
    bbox_x          REAL,               -- normalized 0-1 coordinates
    bbox_y          REAL,
    bbox_w          REAL,
    bbox_h          REAL
);

-- Full-text search on descriptions
CREATE VIRTUAL TABLE photos_fts USING fts5(
    description,
    content='photos', content_rowid='rowid'
);

-- Indexes
CREATE INDEX idx_photos_deleted ON photos(deleted_at);
CREATE INDEX idx_photos_slideshow ON photos(in_slideshow) WHERE deleted_at IS NULL;
CREATE INDEX idx_photos_created ON photos(created_at);
CREATE INDEX idx_photo_people_photo ON photo_people(photo_id);
CREATE INDEX idx_photo_people_person ON photo_people(person_id);
```

## Storage Locations

- **Images:** `data/photos/YYYY/MM/{uuid}.jpg` (gitignored)
- **Database:** `data/photos/photos.db` (gitignored)
- **Cloud backup (optional):** AWS S3 bucket, encrypted at rest, sync
  on configurable schedule. Disabled by default

## Privacy

- Photos are stored locally on the device
- Person detection and ReID run on-device (Hailo NPU)
- Descriptions are generated via the small Claude model (API call with
  the image — same privacy boundary as all Claude API calls)
- No photo data leaves the box except explicit agent actions (sending
  a photo via WhatsApp) and the tagging API call
- Optional S3 backup is encrypted at rest and disabled by default
