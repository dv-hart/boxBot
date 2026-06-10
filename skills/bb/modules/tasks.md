# bb.tasks — triggers (wake conditions) and to-do items

`bb.tasks` shares one backend with the `manage_tasks` core tool — a
write here is visible to the tool and vice-versa. Reach for the SDK
when you're composing task management with other SDK calls in one
script (batch operations, conditional logic); the `manage_tasks` tool
is fine for one-off edits.

Two subsystems:

- **Triggers** — wake conditions. Conditions AND together: every
  specified condition must hold for the trigger to fire.
- **To-dos** — persistent action items. Lightweight descriptions up
  front, detailed `notes` loaded on demand via `get()`.

## API

### Create a trigger

```python
import boxbot_sdk as bb

trigger_id = bb.tasks.create_trigger(
    description="Dentist reminder for Jacob",
    instructions="Remind Jacob about his 3:30 dentist appointment",
    fire_at="2026-06-21T15:00:00",     # point in time (ISO)
    # fire_after="30m",                # or a timer: "30m", "2h", "1d" (max 24h)
    # cron="0 8 * * 1-5",              # or recurring
    # person="Jacob",                  # or presence: fires when Jacob is SEEN
    #                                  #   ("*" = any person, no ID needed)
    for_person="Jacob",                # context only — not a condition
    # todo_id="d_…",                   # link to a to-do (does NOT auto-complete it)
)
```

At least one condition (`fire_at`, `fire_after`, `cron`, `person`) is
required. Compound example — "in 30 minutes, when you next see Jacob":
pass both `fire_after="30m"` and `person="Jacob"`. Time conditions stay
met once reached; person conditions are transient (the person must be
present).

### Create a to-do

```python
todo_id = bb.tasks.create_todo(
    description="Return library books",
    notes="Three books, due Saturday. The Pratchett one is Erik's.",
    for_person="Jacob",
    due_date="2026-06-13",
)
```

### List / inspect

```python
for t in bb.tasks.list_triggers(status="active"):     # active|expired|cancelled
    print(t.id, t.description, t.fire_at)

for d in bb.tasks.list_todos(status="pending"):       # pending|completed|cancelled
    print(d.id, d.description, d.due_date)

item = bb.tasks.get("d_a1b2c3")    # TodoRecord or TriggerRecord, full detail
print(item.notes)                  # to-do notes load here, not in list_todos
```

`list_*` return record objects (`TriggerRecord` / `TodoRecord`) with
properties `id`, `description`, `status`, `created_at`, plus
trigger-side `instructions` / `fire_at` / `fire_after` / `cron` /
`person` / `for_person` and todo-side `notes` / `for_person` /
`due_date`. Ids are prefixed: `t_…` triggers, `d_…` to-dos.

### Complete / cancel

```python
bb.tasks.complete("d_a1b2c3")    # to-dos only
bb.tasks.cancel("t_9f8e7d")      # triggers or to-dos
```

Completing a to-do does not touch any trigger linked to it, and a
firing trigger does not complete its to-do — close both ends yourself.

## Error semantics

All writes (`create_trigger`, `create_todo`, `complete`, `cancel`)
raise `bb.ActionError` if the main process rejects the call (unknown
id, invalid condition, store error). A task that didn't persist never
looks like it did.

## When NOT to use it

- Calendar events → the `calendar` integration
  (`bb.integrations.get("calendar", …)`). The scheduler is the agent's
  *internal* planning, not the household calendar.
- Facts to remember → `bb.memory`. A trigger is for *waking up and
  acting*, not for recall.
- Checking back on a pending package install → that works well though:
  `fire_after="2h"` with instructions to run `bb.packages.status(id)`.
