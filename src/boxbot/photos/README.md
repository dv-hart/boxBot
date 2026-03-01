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
  ├── 4. Small model: description + tags
  │      (receives person annotations + existing tag library)
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
   square) and stores dimensions
2. **Person detection** — run YOLOv8n on Hailo to detect people in the
   image. For each detected person bounding box, extract OSNet embedding
   and match against known person clouds from the perception system
3. **Tagging** — send the image to the small model with person
   annotations and the existing tag library. Model returns a 1-2 sentence
   description and selected/created tags
4. **Embedding** — embed the description text with MiniLM (384-dim) for
   hybrid search
5. **Storage** — save resized image to disk, write metadata to SQLite,
   add to slideshow by default

**Idle scheduling:** The intake pipeline is async and waits for the system
to be idle (no active conversation, no pending perception tasks). This
ensures the Hailo NPU is available for person detection without contending
with live perception.

**Hailo priority:** Live perception always preempts photo intake. If a
person is detected during photo processing, intake yields the Hailo and
resumes when it becomes available. Implemented via a shared semaphore
with priority levels.

**Queue processing:** Multiple incoming photos are processed sequentially
from the intake queue. The pipeline is fault-tolerant — a failure on one
photo does not block the rest.

### `search.py`
Shared search backend used by both the `search_photos` tool and the
`boxbot_sdk.photos.search()` function. Same DRY pattern as the memory
search backend.

**Search pipeline:**
```
Query: "beach photos with Jacob"
  │
  ├─ 1. Hybrid retrieval on descriptions
  │     Vector cosine similarity (MiniLM embeddings)
  │     + SQLite FTS5 BM25 keyword matching
  │     Combined score: 0.6 x vector + 0.4 x BM25
  │     → candidate set
  │
  ├─ 2. Structured filters (applied on candidates)
  │     Tags (AND logic), people, date range, source, slideshow membership
  │
  └─ 3. Return ranked results with metadata
```

**DRY architecture:**
```
src/boxbot/photos/search.py               <- search backend
    │
    ├── src/boxbot/tools/search_photos.py  <- core tool (8th tool)
    │     Agent calls directly during conversation.
    │
    └── src/boxbot/sdk/photos.py           <- SDK function
          Sandbox scripts call boxbot_sdk.photos.search().
          Emits JSON action -> main process routes to search backend.
```

### `slideshow.py`
Photo selection logic for the `picture` display's slideshow mode:
- Select photos from the slideshow-enabled set (active, not soft-deleted)
- Rotation strategies: random, chronological, tag-based, seasonal
- Provide ordered photo list to the display module
- Respect configured timing (seconds per photo)

Note: transition effects and rendering are handled by the `picture`
display module, not this file. This module only handles photo selection.

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
- Person annotations from ReID: `["Jacob (left)", "unknown person (center)"]`
- The existing tag library (full list of current tags)

**Instructions to the model:**
- Write a 1-2 sentence description of the image. Include identified
  people by name and describe the scene, activity, and setting. Keep it
  concise but specific enough to be useful for search.
- Select tags from the existing library that apply. Prefer existing tags.
  Create new tags only when nothing in the library fits. New tags should
  be lowercase, singular, and general enough for reuse (e.g., "beach"
  not "myrtle-beach-trip-2026").

**Model output (structured):**
```json
{
  "description": "Jacob and an unknown person standing on a rocky beach at sunset. Ocean waves visible in the background.",
  "tags": ["beach", "sunset", "outdoor"],
  "new_tags": ["rocky"]
}
```

New tags are created in the tag library automatically. The model's
preference for existing tags keeps the library stable over time while
still allowing growth.

### Sender Context (WhatsApp)

When a photo arrives via WhatsApp, the sender's identity is recorded as
metadata (`sender` field) but is **not** used to assume who appears in
the photo. The sender might be sending a photo of other people, a
landscape, or a screenshot. Person identification relies solely on
the ReID pipeline.

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
