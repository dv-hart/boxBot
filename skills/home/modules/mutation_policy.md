# Mutation policy — what's blocked in V1 and why

The `home_assistant` integration refuses `call_service` for three
domains:

- `alarm_control_panel.*` — arming and disarming
- `lock.*` — locking and unlocking
- `cover.*` — garage doors, gates, motorized blinds

Refusal happens in the integration's script before the HTTP call —
HA never sees the request. The error returned looks like:

```
{"status": "error", "error": "service calls in domain 'lock' are blocked
 in V1 — alarm/lock/cover mutations need the confirmation gate (not yet
 implemented). State reads (get_state) on these entities still work."}
```

## What still works

State reads (`get_state`) on blocked-domain entities are **always**
allowed. BB can answer:

- "Is the alarm armed?"
- "Are the front doors locked?"
- "Is the garage open?"

It just can't change any of those answers yet.

## Why these three

These are the domains where an agent acting on a hallucinated request
— or following a successfully prompt-injected one — could create real
physical or security harm:

- **Disarming the alarm** while no one is home, then doing nothing
  visible to the user.
- **Unlocking a door** at 3 a.m. on a misheard wake word.
- **Closing a garage** on a child, a pet, or a person — covers in
  general are too broad to evaluate by domain alone, so V1 takes the
  conservative path and blocks them all (even blinds, which are
  benign).

The cost of refusing is small ("tell the user BB can see but not yet
change"); the cost of mis-acting is large.

## What ISN'T blocked

- `light.*`, `switch.*`, `scene.*`, `script.*`, `climate.*`,
  `media_player.*`, `fan.*`, and the long tail of "boring" domains.
- Reads (`get_state`, `get_states`) on **any** entity, including
  alarm/lock/cover.
- `list_services` returns gated services in its listing — discovery
  is not the threat.

If something boring is wired to a dangerous physical effect (a smart
plug controlling a space heater wired into a child's room), the V1
denylist won't catch it. Document those edge cases in memory so BB
treats them carefully.

## V2 — confirmation gate

The eventual design mirrors the package-install approval pattern that
already exists for `bb.packages.install`:

1. The integration manifest declares which actions need confirmation:
   ```yaml
   confirmations:
     - alarm_arm_home
     - alarm_arm_away
     - alarm_disarm
     - unlock
     - open_cover
     - close_cover
   ```
2. When `call_service` is invoked for a confirmation-gated action,
   the runner pauses the call, emits an approval request (admin
   WhatsApp YES or screen-tap), and only proceeds on explicit
   approval.
3. Denial closes the call with a clean error; timeout (e.g. 30s)
   also closes it.

V2 unblocks the domain denylist by replacing it with per-action gates
+ a clear confirmation UX. Until then: reads on, writes off.

## Talking to the user about it

When the user asks BB to disarm the alarm or unlock a door, the right
response is something like:

> "I can see the alarm is armed-home and the front door is locked. I
> can't change those from BB yet — that needs the confirmation step
> I'm working on. You can do it from the Alarm.com app or the HA UI
> in the meantime."

Don't pretend the limit isn't there. Don't try to work around it by,
e.g., calling `script.*` if the script wraps a `lock.unlock` call —
the script will succeed (HA processes it server-side) and the gate
won't catch it. That's an edge case to document in memory for the
specific scripts users set up; the integration can't see through
HA's automation layer.
