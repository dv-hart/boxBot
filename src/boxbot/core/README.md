# core/

The central nervous system of boxBot. This module owns the agent lifecycle,
configuration, and task scheduling.

## Files

### `main.py`
Application entry point. Initializes all subsystems, starts the agent loop.
Handles graceful shutdown and signal handling.

### `agent.py`
The Claude Agent SDK integration. This is where the agent is instantiated with
its system prompt, tools, skills, and conversation management. Responsibilities:
- Build the agent's system prompt from config, active memories, and current
  context (who is present, time of day, pending tasks)
- Register core tools (always loaded) and relevant skills (per-conversation)
- Route work to the large model (conversations, reasoning) or small model
  (tagging, classification, extraction)
- Manage conversation turns and context window
- Trigger memory extraction before compaction and at conversation end

### `config.py`
Configuration loading and validation. Reads from `config/config.yaml` and
environment variables (`.env`). Uses Pydantic models for validation.
All configuration is centralized here — no module reads config files directly.

### `scheduler.py`
The agent's task management and wake/sleep lifecycle. Manages two related
but distinct subsystems — **triggers** (event-driven wake conditions) and
**to-do items** (persistent action items) — plus the wake cycle that ties
them together.

This is the agent's *internal planning system*, distinct from the family
calendar. Calendar events (appointments, family schedules) are managed by
calendar skills and displayed separately. The scheduler manages when the
agent wakes, what it needs to do, and what it's tracking.

#### Triggers

A trigger is a set of conditions that, when all are met, wake the agent
with specific instructions. Conditions are AND'd — every specified
condition must be satisfied for the trigger to fire.

**Conditions** (combine any subset; all specified must be met to fire):

| Condition | Field | Description |
|-----------|-------|-------------|
| Point-in-time | `fire_at` | Absolute ISO datetime. Condition met when current time ≥ value. Stays met once passed |
| Timer | `fire_after` | Relative duration (`"30m"`, `"2h"`, `"1d"`). Converted to absolute `fire_at` on creation. Max 24 hours |
| Recurring | `cron` | Cron expression. Mutually exclusive with `fire_at`/`fire_after`. Each occurrence arms the trigger; resets after firing |
| Person | `person` | Person name. Condition met when this person is currently detected by perception |

**Condition evaluation:**
- Time conditions (`fire_at`, resolved `fire_after`): once current time
  passes the threshold, the condition is **met and stays met**. The
  trigger then waits for any remaining conditions (typically a person)
- Person conditions: met when the person is currently detected by
  perception. Transient — the person must be present at evaluation time
- Recurring + person: each cron occurrence arms a new window. If the
  person doesn't appear before the next cron occurrence, the current
  window expires and a new one begins on the next occurrence
- A trigger with only a time condition fires immediately when the time
  arrives. A trigger with only a person condition fires the next time
  that person is detected. A trigger with both waits for the time to
  pass, then waits for the person

**Examples:**
- `fire_at="15:00"` — fire at 3 PM today
- `fire_after="30m"` — fire in 30 minutes
- `person="Jacob"` — fire when Jacob is detected
- `fire_after="30m", person="Jacob"` — after 30 minutes, fire when
  Jacob is next detected (time condition met first, then wait for person)
- `fire_at="15:00", person="Jacob"` — at/after 3 PM, fire when Jacob
  appears (if Jacob is present at 3 PM, fires immediately)
- `cron="0 7 * * *"` — fire every day at 7 AM
- `cron="0 7 * * *", person="Jacob"` — every day starting at 7 AM,
  fire when Jacob is first detected

**Trigger fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique identifier |
| `description` | string | Human-readable summary (shown in lists) |
| `instructions` | string | What the agent should do when fired (delivered in wake context) |
| `fire_at` | datetime | Absolute fire time (null for person-only or recurring) |
| `cron` | string | Cron expression (null for non-recurring) |
| `person` | string | Person name condition (null for time-only) |
| `for_person` | string | Who this task relates to — for context and delivery routing (optional) |
| `todo_id` | UUID | Optional FK to a linked to-do item |
| `status` | enum | `active`, `fired`, `expired`, `cancelled` |
| `source` | enum | `config`, `agent`, `conversation` |
| `created_at` | datetime | When created |
| `expires` | datetime | When the trigger stops being active |
| `last_fired` | datetime | Last fire timestamp (recurring triggers) |
| `fire_count` | int | Number of times fired (recurring triggers) |

**Expiry rules:**
- Person-only triggers: configurable default (default 7 days)
- Timer triggers: `fire_at` + person expiry window if a person condition
  exists (max 24h base + person wait window)
- Point-in-time triggers: `fire_at` + person expiry window if a person
  condition exists
- Recurring triggers: do not expire (cancel explicitly)
- Any trigger can set an explicit `expires` datetime override

#### To-Do Items

Persistent action items the agent tracks. The list view is lightweight
(descriptions only); the agent fetches full notes via `get` when it
decides to work on an item. This keeps the to-do list cheap to scan
while preserving rich context for execution.

**To-do fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique identifier |
| `description` | string | Brief summary (shown in list view) |
| `notes` | string | Detailed context, background, instructions (loaded on demand via `get`) |
| `for_person` | string | Who this relates to (optional) |
| `due_date` | date | Soft deadline (optional) |
| `status` | enum | `pending`, `completed`, `cancelled` |
| `created_at` | datetime | When created |
| `completed_at` | datetime | When completed (null until done) |
| `source` | enum | `agent`, `conversation` |

**Design notes:**
- No priority levels. The agent decides what to work on based on context,
  due dates, and its own judgment
- Notes carry rich context: "refer to conversation on Feb 20 about the
  menu", "don't forget to check dietary restrictions", "books are on the
  kitchen counter." Loaded only when the agent works on the item
- A to-do without any linked trigger is tracked passively — the agent
  reviews it during wake cycles or when a user asks "what's on my list?"

#### Trigger ↔ To-Do Relationship

Triggers and to-dos are separate but linked:
- A trigger can reference a `todo_id`. When the trigger fires, the
  agent works on the linked to-do item
- The agent creates to-do items independently for persistent tracking
  ("return library books", "plan birthday party") — no trigger needed
- **Triggers do not auto-complete linked to-dos.** When a trigger fires,
  the agent executes the instructions, then explicitly decides whether
  the to-do is complete. This handles partial completion, follow-ups,
  and cases where delivery fails
- A single to-do can be referenced by multiple triggers (e.g., a time
  trigger that falls back to a person trigger)

#### Wake Cycle

The agent's sleep/wake lifecycle. When sleeping, boxBot shows idle
displays (photo slideshow, clock, weather) and listens for wake words
and person detection events only.

**Wake sequence (trigger-fired):**
```
1. Trigger conditions all met → scheduler emits `trigger_fired` event
2. Agent wakes with context injection:
   - Trigger that fired: description + instructions
   - Who is present (from perception)
   - To-do count + active trigger count
   - Relevant memories (standard injection)
3. Agent executes trigger instructions
4. Agent reviews to-do list for actionable items
5. Agent updates state:
   - Complete to-do items that are done
   - Create new triggers for deferred delivery
     (e.g., person not present → person trigger)
   - Cancel stale triggers
6. Update displays (weather, calendar, etc.)
7. Idle timeout (configurable, default 300s) → sleep state
```

**Wake sequence (recurring / wake cycle):**
```
1. Cron trigger fires at configured time (e.g., 7 AM daily)
2. Agent wakes with wake cycle instructions from config
3. Execute instructions: check weather, review calendar, etc.
4. Review to-do list — act on actionable items
5. Create triggers for deferred items (person delivery, etc.)
6. Update displays with fresh data
7. Idle timeout → sleep state
```

**Conversation start injection:**
When a conversation starts naturally (wake word, WhatsApp — not
trigger-fired), inject a lightweight status line:
```
[To-do: 3 items | Triggers: 2 active]
```
The agent can call `manage_tasks(action="list")` if it wants details.
This keeps injection cost minimal while giving the agent enough to
decide whether the to-do list is relevant to the current conversation.

**Config-seeded recurring triggers:**
The `wake_cycle` config defines recurring triggers seeded into the
database on first boot. After initial seeding, the runtime database
(`data/scheduler/scheduler.db`) is authoritative — the agent can
modify, cancel, or create new recurring triggers at runtime. Config
is re-seeded only if the database does not exist (fresh install or
manual reset).

#### Scheduler Process

Background async task that monitors trigger conditions:

- **Time-based triggers:** maintains a priority queue sorted by next
  `fire_at`. Checks the head of the queue every 60 seconds. When
  current time ≥ `fire_at`:
  - No person condition → fire immediately
  - Person condition exists → mark time condition as met, add trigger
    to the person-watch list
- **Person-watch list:** subscribes to `person_identified` events from
  the perception pipeline. When a person is identified, checks if any
  active triggers have a matching person condition with all other
  conditions already met → fire
- **Recurring triggers:** after firing, compute next occurrence from
  the cron expression, reset condition states for the new window
- **Expiry sweep:** runs daily, marks expired triggers as
  `status: expired`

#### Storage

SQLite database at `data/scheduler/scheduler.db`. Two tables:

**`triggers` table:**
```sql
CREATE TABLE triggers (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    instructions TEXT NOT NULL,
    fire_at TEXT,          -- ISO datetime (resolved from fire_after on creation)
    cron TEXT,             -- cron expression (recurring only)
    person TEXT,           -- person name condition
    for_person TEXT,       -- who this relates to (context/delivery)
    todo_id TEXT,          -- FK to todos.id (optional)
    status TEXT NOT NULL DEFAULT 'active',
    source TEXT NOT NULL DEFAULT 'agent',
    created_at TEXT NOT NULL,
    expires TEXT,          -- ISO datetime
    last_fired TEXT,       -- ISO datetime (recurring)
    fire_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (todo_id) REFERENCES todos(id)
);
```

**`todos` table:**
```sql
CREATE TABLE todos (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    notes TEXT,            -- detailed context, loaded on demand
    for_person TEXT,
    due_date TEXT,         -- ISO date
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    completed_at TEXT,     -- ISO datetime
    source TEXT NOT NULL DEFAULT 'agent'
);
```

### `events.py`
Internal event bus for decoupled communication between subsystems. Events
include:
- `person_detected` — perception pipeline detected a person in frame
- `person_identified` — perception pipeline identified a known person
- `wake_word_heard` — wake word triggered
- `button_pressed` — physical input from KB2040
- `whatsapp_message` — incoming WhatsApp message
- `trigger_fired` — a trigger's conditions are all met (carries full
  trigger details: id, description, instructions, conditions, for_person)
- `conversation_started` — agent began a conversation
- `conversation_ended` — agent finished a conversation
