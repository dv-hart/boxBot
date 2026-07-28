# bb.secrets — write-only credential vault

Store a value once (usually when the user pastes a key). Scripts and
integrations receive only the values they declared, as
`BOXBOT_SECRET_<NAME>` env vars injected at launch. Never as Python
strings you can read back.

**Use when:** the user pastes an API key, OAuth refresh token, or
webhook secret; an integration manifest declares `secrets: [...]` and
the runner reports one missing; a one-off `execute_script` needs a
credential — pass `secrets=[...]` on the tool call, **not** `env_vars`
(that would require you to know the value).

**Not for:** non-sensitive config → the manifest's `inputs:` or the
data source's `params`. Anything you need to *read* — this is one-way.
Values over 8 KB, or more than 64 stored; the store is deliberately
small.

## Lifecycle

```python
import boxbot_sdk as bb

bb.secrets.store("POLYGON_API_KEY", "pk_live_…")
# → {"status": "ok", "name": "POLYGON_API_KEY", "previous": "created"}

bb.secrets.list()          # names + timestamps. No values.
# → {"status": "ok",
#    "secrets": [{"name": "POLYGON_API_KEY", "stored_at": "2026-05-02T…Z"}]}

if bb.secrets.has("POLYGON_API_KEY"):
    ...

bb.secrets.delete("OLD_API_KEY")
# → {"status": "ok", "name": "OLD_API_KEY"}
# Raises bb.ActionError if the name isn't stored — a delete that
# removed nothing fails loudly.
```

`store` and `delete` raise `bb.ActionError` on rejection (bad name
shape, oversized value, store full, name absent). `list`, `has`, `use`
return the shapes above.

## Naming

SCREAMING_SNAKE_CASE, `^[A-Z][A-Z0-9_]*$`, ≤64 chars. Same shape
integrations declare in manifests, so one name works end to end.
`bb.secrets.store("polygon_api_key", …)` errors.

## Reaching a secret from a script

Integration script — the manifest declares `secrets: [...]`, the runner
injects:

```python
# integrations/polygon/script.py
import os
api_key = os.environ.get("BOXBOT_SECRET_POLYGON_API_KEY", "")
if not api_key:
    return_output({"error": "POLYGON_API_KEY not stored"})
```

A declared-but-absent secret logs a warning and launches anyway. Your
script sees a missing env var and should surface a helpful error.

Ad-hoc `execute_script` — pass `secrets=["POLYGON_API_KEY"]` on the
tool call; unknown names are skipped silently:

```python
import os
key = os.environ["BOXBOT_SECRET_POLYGON_API_KEY"]
```

Do **not** call `bb.secrets.use("…")` and route the result through
`env_vars`. `use()` returns the env-var *name* only, as a diagnostic.

## What you can see

Names, stored-at timestamps, and whether a secret was reachable for a
given call (observable in your script's own error handling). **Never
values** — once stored, a value returns through no SDK call, only
through env vars in the launched subprocess.

## Conversation-start hint

```
[To-do: 3 items | Triggers: 1 active | Secrets: 7 stored]
```

`0` means no credentials on file. Non-zero and you need specifics:
`bb.secrets.list()`.

## Storage

`data/credentials/secrets.json`, mode `0600`, owned by the main-process
user. `boxbot-sandbox` has no read. Unencrypted at rest — same
protection class as the `.env` beside it. If that stops being enough,
the fix is filesystem encryption (LUKS on `/data`), not per-file crypto.
