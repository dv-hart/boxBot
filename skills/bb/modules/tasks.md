# bb.tasks — triggers and to-dos

Same backend as the `manage_tasks` tool; writes are visible to both.
Use the SDK when batching task edits with other SDK calls in one
script. Use the tool for one-off edits.

- **Triggers** — wake conditions. Conditions AND together: all must
  hold to fire.
- **To-dos** — persistent action items. Descriptions up front,
  `notes` loaded on demand via `get()`.

## Create a trigger — fire_at / fire_after / cron / person / entity (AND logic)

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
    # entity="binary_sensor.front_door_person",  # events-bridge sensor id
    # entity_state="on",               #   default "on"
    # rearm_after_s=60,                # "whenever", not "next time"
    for_person="Jacob",                # context only, not a condition
    # todo_id="d_…",                   # link a to-do (no auto-complete)
)
```

At least one of `fire_at` / `fire_after` / `cron` / `person` /
`entity` is required.

Condition triggers are **one-shot** by default: first fire completes
them. "Text me *whenever* someone's at the front door" needs
`rearm_after_s` — the trigger stays active and fires again each time
the condition is met after that many seconds (cooldown; small floor
always applies, `0` = as often as possible). Requires `person` or
`entity`; invalid with `cron` (cron already recurs). Re-arming
triggers default to 30-day expiry instead of 7 — renew or set
`expires` for longer.

Compound = pass several. "In 30 minutes, when you next see Jacob" =
`fire_after="30m"` + `person="Jacob"`. Time conditions stay met once
reached; person and entity conditions are transient (must hold *now*).

`person` vs `entity`: `person` is BB's own camera recognizing someone
in the room. `entity` is a sensor id published by whichever events
bridge this box runs — only ids the bridge publishes can fire. "Tell
me when someone approaches the front door" =
`entity="binary_sensor.front_door_person"`, `entity_state="on"`
(camera detection sensors: `_person` / `_vehicle` / `_animal` /
`_package` variants per camera). The HA bridge needs
`HOME_ASSISTANT_URL` / `HOME_ASSISTANT_TOKEN` secrets; find entity ids
via the home skill's `get_states(domain="binary_sensor")`. Do NOT reach
for the `home_assistant` integration to set up a camera alert — the
trigger alone is the whole job.

## Run a script instead of waking me

`run_script` / `run_integration` = fire a sandboxed script, no model
call, no tokens. Split the work: **condition → trigger; logic →
script; communication → escalate to me.**

`run_script` is the default choice — a plain workspace `.py` with the
full `bb.*` surface. Author → test → schedule, no new concepts:

```python
bb.workspace.write("scripts/nightly_lockup.py", '''
import boxbot_sdk as bb
problems = []
ha = lambda **kw: bb.integrations.get("home_assistant", **kw)["output"]
for lock in ha(action="get_states", domain="lock"):
    if lock["state"] != "locked":
        problems.append(f"{lock['entity_id']} is {lock['state']}")
# ... garage close, arm-stay checks ...
if problems:
    bb.escalate("; ".join(problems))      # wakes me with the details
''')
# run it once NOW via execute_script to verify, then:
bb.tasks.create_trigger(
    description="Nightly lock-up",
    instructions="Lock-up check hit a problem — decide what to tell Jacob.",
    cron="0 22 * * *",
    run_script="scripts/nightly_lockup.py",   # path validated NOW
)
```

`run_integration` is the same contract for a **registered integration**
— use it when the job needs manifest-validated inputs or its own
credentials (see [integrations](integrations.md)):

```python
bb.tasks.create_trigger(
    description="Morning weather check",
    instructions="Forecast pulled — decide whether it's worth mentioning.",
    cron="0 7 * * *",
    run_integration="weather",
    run_inputs={"forecast_days": 1},     # validated NOW against the manifest
)
```

Outcomes (both kinds):

| Run | Result |
|---|---|
| exits clean, no `escalate` | silent, zero tokens |
| `bb.escalate("<why>")` called | I wake: `instructions` + script output |
| raises / times out / `status != ok` | I wake: `instructions` + the error |

So `instructions` is **escalation context**, not the scene — the script
is the scene. Write it for the failure case.

Scripts cannot reach a person: `message` stays agent-gated. A script
that needs a human calls `bb.escalate("Back door open 14 min")` and
I decide what to say. Also denied to unattended scripts: all
`tasks.*` (no self-replicating triggers). Cron recurrence re-arms on its own;
`run_inputs` reach the script via `bb.integration.inputs()`.

Time windows, hold durations, OR-logic: put them **in the script**.
Trigger conditions stay a simple AND.

## Create a to-do — create_todo(description, notes, for_person, due_date)

```python
todo_id = bb.tasks.create_todo(
    description="Return library books",
    notes="Three books, due Saturday. The Pratchett one is Erik's.",
    for_person="Jacob",
    due_date="2026-06-13",
)
```

## List and inspect — list_triggers(status) / list_todos(status) / get(id)

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

## Complete and cancel — complete(to-dos) / cancel(triggers or to-dos)

```python
bb.tasks.complete("d_a1b2c3")    # to-dos only
bb.tasks.cancel("t_9f8e7d")      # triggers or to-dos
```

Completing a to-do does not touch its linked trigger. A firing trigger
does not complete its to-do. Close both ends yourself.

## Errors — create_trigger/create_todo/complete/cancel raise ActionError

`create_trigger`, `create_todo`, `complete`, `cancel` raise
`bb.ActionError` on rejection (unknown id, invalid condition, store
error). A task that did not persist never looks like it did.

## Not this — calendar events→integrations, durable facts→memory

- Calendar events → `bb.integrations.get("calendar", …)`. The
  scheduler is *your* planning, not the household calendar.
- Facts to remember → `bb.memory`. A trigger is for waking up and
  acting, not recall.
- Checking a pending package install → this is a good fit:
  `fire_after="2h"` with instructions to run `bb.packages.status(id)`.
