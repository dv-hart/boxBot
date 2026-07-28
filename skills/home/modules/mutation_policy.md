# Mutation policy — blocked in V1

The `home_assistant` integration refuses `call_service` in three
domains:

- `alarm_control_panel.*` — arm/disarm
- `lock.*` — lock/unlock
- `cover.*` — garage doors, gates, motorized blinds

Refusal happens in the script before the HTTP call. HA never sees the
request.

```
{"status": "error", "error": "service calls in domain 'lock' are blocked
 in V1 — alarm/lock/cover mutations need the confirmation gate (not yet
 implemented). State reads (get_state) on these entities still work."}
```

## Still works

`get_state` on blocked-domain entities is **always** allowed. BB can
answer "is the alarm armed?", "are the front doors locked?", "is the
garage open?" It just can't change the answer.

## Why these three

These are where acting on a hallucinated — or successfully
prompt-injected — request causes real physical or security harm:

- Disarming the alarm while nobody is home, invisibly.
- Unlocking a door at 3 a.m. on a misheard wake word.
- Closing a garage on a child, a pet, or a person. Covers are too
  broad to evaluate by domain, so V1 blocks all of them, benign blinds
  included.

Refusing costs little. Mis-acting costs a lot.

## Not blocked

- `light.*`, `switch.*`, `scene.*`, `script.*`, `climate.*`,
  `media_player.*`, `fan.*`, and the boring long tail.
- Reads on **any** entity, alarm/lock/cover included.
- `list_services` still lists gated services. Discovery is not the
  threat.

A boring domain wired to a dangerous effect — a smart plug driving a
space heater in a child's room — slips past the denylist. Note those
in memory so BB handles them carefully.

## V2 — confirmation gate

Mirrors the existing package-install approval flow:

1. The manifest declares gated actions:
   ```yaml
   confirmations:
     - alarm_arm_home
     - alarm_arm_away
     - alarm_disarm
     - unlock
     - open_cover
     - close_cover
   ```
2. `call_service` on a gated action pauses, emits an approval request
   (admin text YES, or screen tap), proceeds only on explicit approval.
3. Denial closes the call with a clean error. Timeout (~30s) does too.

That replaces the domain denylist with per-action gates. Until then:
reads on, writes off.

## Talking about it

> "I can see the alarm is armed-home and the front door is locked. I
> can't change those from BB yet — that needs the confirmation step
> I'm working on. You can do it from the Alarm.com app or the HA UI in
> the meantime."

Don't pretend the limit isn't there. Don't route around it via
`script.*` — a script wrapping `lock.unlock` succeeds server-side and
the gate never sees it. The integration cannot see through HA's
automation layer. Note specific user scripts in memory.
