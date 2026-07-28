# bb.packages — request PyPI installs, humans approve

The sandbox cannot install packages: seccomp blocks subprocesses, pip
is owner-only, site-packages is read-only. Only path is
**request → human approval → main-process install**. Nothing you do in
the sandbox short-circuits the human.

## Flow

1. `bb.packages.request("name", reason="…")` validates the spec, queues
   a **pending** request, messages every admin with reply instructions
   (`approve pkg <id>` / `deny pkg <id>`).
2. Returns **immediately**. Approval can take hours. Never wait
   in-script.
3. An admin replies. The message router handles it — not you. On
   approve, the main process pip-installs into the sandbox venv;
   the request becomes `installed` or `failed`.
4. Check back with `status()` / `list()`. Once `installed`, the package
   imports in your *next* `execute_script` run.

Lifecycle: `pending → approved → installed | failed`, or
`pending → denied`.

## API

```python
import boxbot_sdk as bb

req = bb.packages.request(
    "google-api-python-client",     # bare name, or exact pin "name==1.2.3"
    reason="Gmail integration needs the API client",
)
# → {"id": "ab12cd34", "package": …, "status": "pending",
#    "requested_at": …, "duplicate": False, "admins_notified": 1}

req = bb.packages.status("ab12cd34")
# same record + current "status"; when resolved also "resolved_by",
# "resolved_at", "note" (deny reason or pip error tail)

bb.packages.list()              # all, newest first
bb.packages.list("pending")     # pending|approved|installed|failed|denied
```

## Waiting: set a trigger, do not poll

```python
req = bb.packages.request("feedparser", reason="RSS skill needs a parser")
bb.tasks.create_trigger(
    description=f"Check package request {req['id']}",
    instructions=(
        f"Run bb.packages.status('{req['id']}'). If installed, finish "
        "building the RSS skill; if denied, tell Jacob the alternative."
    ),
    fire_after="2h",
)
```

## Errors — "no" is not an error

- `request()` raises `bb.ActionError` for **system** problems: invalid
  spec (URLs, local paths, extras, `>=` ranges all rejected — only a
  bare PyPI name or exact `name==version`), missing reason, store
  failure.
- A denial is **data**: `status() == "denied"` with the admin's reason
  in `note`. Never report a denial as a malfunction, or a malfunction
  as a denial.
- `duplicate: True` — an identical pending request already existed;
  admins were not re-pinged. `admins_notified: 0` — nobody got the
  message (no admins, or channel down); the request still queues.

## Cannot, by design

- Approve your own request. There is no SDK approval action — only an
  inbound admin message on Signal/WhatsApp, honored only from senders
  whose registered role is **admin**.
- Sneak pip options or URLs through the package name. Strict regex runs
  before anything reaches pip.
- Install without the request. Pip is unreachable at the OS level.
