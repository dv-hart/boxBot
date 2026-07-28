# bb.tasks — triggers and to-dos

Same backend as the `manage_tasks` tool; writes are visible to both.
Use the SDK when batching task edits with other SDK calls in one
script. Use the tool for one-off edits.

- **Triggers** — wake conditions. Conditions AND together: all must
  hold to fire.
- **To-dos** — persistent action items. Descriptions up front,
  `notes` loaded on demand via `get()`.

## Create a trigger

```python
import boxbot_sdk as bb

trigger_id = bb.tasks.create_trigger(
    description="Dentist reminder for Jacob",
    instructions="Remind Jacob about his 3:30 dentist appointment",
    fire_at="2026-06-21T15:00:00",     # point in time (ISO)
    # fire_after="30m",                # timer: "30m", "2h", "1d" (max 24h)
    # cron="0 8 * * 1-5",              # recurring
    # person="Jacob",                  # fires when Jacob is SEEN
    #                                  #   ("*" = any person, no ID needed)
    # entity="binary_sensor.front_door_person",  # Home Assistant entity
    # entity_state="on",               #   default "on"
    for_person="Jacob",                # context only, not a condition
    # todo_id="d_…",                   # link a to-do (no auto-complete)
)
```

At least one of `fire_at` / `fire_after` / `cron` / `person` /
`entity` is required.

Compound = pass several. "In 30 minutes, when you next see Jacob" =
`fire_after="30m"` + `person="Jacob"`. Time conditions stay met once
reached; person and entity conditions are transient (must hold *now*).

`person` vs `entity`: `person` is BB's own camera recognizing someone
in the room. `entity` is any Home Assistant sensor anywhere — needs
`HOME_ASSISTANT_URL` / `HOME_ASSISTANT_TOKEN` secrets and the outbound
WebSocket bridge. "Tell me when someone approaches the front door" =
`entity="binary_sensor.front_door_person"` (Alarm.com person
detection; `_vehicle` / `_animal` / `_package` variants exist per
camera). Find entity ids via the home skill's
`get_states(domain="binary_sensor")`.

## Create a to-do

```python
todo_id = bb.tasks.create_todo(
    description="Return library books",
    notes="Three books, due Saturday. The Pratchett one is Erik's.",
    for_person="Jacob",
    due_date="2026-06-13",
)
```

## List and inspect

```python
for t in bb.tasks.list_triggers(status="active"):     # active|expired|cancelled
    print(t.id, t.description, t.fire_at)

for d in bb.tasks.list_todos(status="pending"):       # pending|completed|cancelled
    print(d.id, d.description, d.due_date)

item = bb.tasks.get("d_a1b2c3")    # TriggerRecord | TodoRecord, full detail
print(item.notes)                  # notes load here, not in list_todos
```

`TriggerRecord` / `TodoRecord`: `id`, `description`, `status`,
`created_at`; trigger-side `instructions`, `fire_at`, `fire_after`,
`cron`, `person`, `for_person`; todo-side `notes`, `for_person`,
`due_date`. Id prefixes: `t_` triggers, `d_` to-dos.

## Complete and cancel

```python
bb.tasks.complete("d_a1b2c3")    # to-dos only
bb.tasks.cancel("t_9f8e7d")      # triggers or to-dos
```

Completing a to-do does not touch its linked trigger. A firing trigger
does not complete its to-do. Close both ends yourself.

## Errors

`create_trigger`, `create_todo`, `complete`, `cancel` raise
`bb.ActionError` on rejection (unknown id, invalid condition, store
error). A task that did not persist never looks like it did.

## Not this

- Calendar events → `bb.integrations.get("calendar", …)`. The
  scheduler is *your* planning, not the household calendar.
- Facts to remember → `bb.memory`. A trigger is for waking up and
  acting, not recall.
- Checking a pending package install → this is a good fit:
  `fire_after="2h"` with instructions to run `bb.packages.status(id)`.
