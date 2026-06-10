# bb.packages — request PyPI packages, humans approve out-of-band

The sandbox cannot install packages — seccomp blocks subprocesses, pip
is owner-only, site-packages is read-only. The only path is a
**request → human approval → main-process install** flow, and nothing
you do in the sandbox can short-circuit the human.

## The flow

1. `bb.packages.request("name", reason="…")` validates the spec,
   durably queues a **pending** request, and messages every registered
   admin on their channel with reply instructions
   (`approve pkg <id>` / `deny pkg <id>`).
2. The call **returns immediately** — approval can take hours. Do not
   wait in-script.
3. An admin replies; the message router (not the agent) handles it.
   On approve, the main process pip-installs into the sandbox venv and
   the request becomes `installed` (or `failed`).
4. You check back later with `status()` / `list()`. Once `installed`,
   the package is importable in your *next* `execute_script` run.

Lifecycle: `pending → approved → installed | failed`, or
`pending → denied`.

## API

```python
import boxbot_sdk as bb

req = bb.packages.request(
    "google-api-python-client",          # bare name, or exact pin "name==1.2.3"
    reason="Gmail integration needs the API client",
)
# → {"id": "ab12cd34", "package": …, "status": "pending",
#    "requested_at": …, "duplicate": False, "admins_notified": 1}

req = bb.packages.status("ab12cd34")
# → the same record, with current "status" and, when resolved,
#   "resolved_by", "resolved_at", and "note" (deny reason / pip error tail)

bb.packages.list()                  # all requests, newest first
bb.packages.list("pending")         # filter: pending|approved|installed|failed|denied
```

### The right pattern for waiting

Don't poll in a loop — set yourself a trigger and move on:

```python
req = bb.packages.request("feedparser", reason="RSS skill needs a parser")
bb.tasks.create_trigger(
    description=f"Check package request {req['id']}",
    instructions=(
        f"Run bb.packages.status('{req['id']}'). If installed, finish "
        "building the RSS skill; if denied, tell Jacob what the "
        "alternative is."
    ),
    fire_after="2h",
)
```

## Error semantics — "no" is not an error

- `request()` raises `bb.ActionError` for **system** problems: invalid
  spec (URLs, local paths, extras, `>=` ranges are all rejected — only
  a bare PyPI name or an exact `name==version` pin is accepted),
  missing reason, store failure.
- A human denying the request is **data, not an exception**: it shows
  up as `status() == "denied"`, with the admin's reason in `note`.
  Never report a denial as a malfunction — and never report a
  malfunction as a denial.
- `duplicate: True` on `request()` means an identical pending request
  already existed; it's returned instead and the admins were not
  re-pinged. `admins_notified: 0` means nobody received the message
  (no admins registered or channel down) — the request still waits in
  the queue.

## What you cannot do (by design)

- Approve your own request — there is no SDK action for approval;
  it exists only as an inbound admin message on Signal/WhatsApp, and
  only senders whose registered role is **admin** are honoured.
- Sneak pip options or URLs through the package name — the spec is
  validated against a strict regex before anything reaches pip.
- Install without the request — pip is unreachable from the sandbox at
  the OS level.
