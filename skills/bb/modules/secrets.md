# bb.secrets — store credentials, hand them to scripts and integrations

`bb.secrets` is a write-only credential vault. The agent stores values
once (typically when the user pastes a key into chat), and scripts and
integrations receive only the specific values they declared they need
— never as Python strings the agent can read back, only as
`BOXBOT_SECRET_<NAME>` environment variables injected at launch.

## When to use it

- The user pastes an API key, OAuth refresh token, webhook secret, or
  any other credential the agent should keep on file.
- An integration manifest declares `secrets: [...]` and the runner
  reports the secret is missing — store the value and rerun.
- A one-off `execute_script` call needs a credential. Use the
  `secrets=[...]` parameter on `execute_script`; do **not** pass the
  value through `env_vars` (the agent would have to know the value).

## When NOT to use it

- Plain config that isn't sensitive — the manifest's `inputs:` field
  or the data source's `params` is a better home.
- Anything the agent should be able to *read*. `bb.secrets` is
  one-way: store and forget. If the agent needs to see a value to
  reason about it, it doesn't belong here.
- Long values (>8 KB) or many-of-the-same (>64 stored). The store is
  intentionally small; if you need more, the design has drifted.

## Lifecycle

```python
import boxbot_sdk as bb

# Store (write-only).
bb.secrets.store("POLYGON_API_KEY", "pk_live_…")
# → {"status": "ok", "name": "POLYGON_API_KEY", "previous": "created"}

# Inventory — names + when they were stored. No values.
bb.secrets.list()
# → {"status": "ok",
#    "secrets": [{"name": "POLYGON_API_KEY", "stored_at": "2026-05-02T…Z"}]}

# Quick existence check.
if bb.secrets.has("POLYGON_API_KEY"):
    ...

# Delete when rotating or removing an account.
bb.secrets.delete("OLD_API_KEY")
# → {"status": "ok"|"missing", "name": "OLD_API_KEY"}
```

## Naming

Names are SCREAMING_SNAKE_CASE — `^[A-Z][A-Z0-9_]*$`, ≤64 chars. Same
shape integrations declare in their manifests, so the same name works
end-to-end. `secrets.store("polygon_api_key", …)` errors; use
`POLYGON_API_KEY`.

## Reaching a secret from a script

### From an integration script

The integration's manifest declares `secrets: [...]`; the runner
injects `BOXBOT_SECRET_<NAME>` at launch.

```python
# integrations/polygon/script.py
import os
api_key = os.environ.get("BOXBOT_SECRET_POLYGON_API_KEY", "")
if not api_key:
    return_output({"error": "POLYGON_API_KEY not stored"})
```

If a declared secret isn't on file, the runner logs a warning and
launches anyway — your script sees an empty/missing env var and
should surface a helpful error.

### From an ad-hoc execute_script call

Pass `secrets=[NAMES]` on the tool call. The tool resolves names
against the store and injects the values for the duration of the
script. Unknown names are silently skipped.

```python
# tool call args:
#   secrets: ["POLYGON_API_KEY"]
import os
key = os.environ["BOXBOT_SECRET_POLYGON_API_KEY"]
```

The agent never sees the value. Do **not** call
`bb.secrets.use("…")` and pass the result through `env_vars`: that
defeats the point — `use()` returns the env-var name only iff the
secret exists, as a diagnostic.

## What the agent can and can't see

- ✅ Names (`list`, `has`).
- ✅ Stored-at timestamps.
- ✅ Whether a stored secret is reachable for a given script call
  (via the `secrets=` parameter's silent-skip behaviour, observable
  in the script's own error handling).
- ❌ Values. Once stored, the value never returns through any
  SDK call — only through env vars in the launched subprocess.

## Conversation start hint

The status line injected at conversation start now includes a count:

```
[To-do: 3 items | Triggers: 1 active | Secrets: 7 stored]
```

If the count is `0`, you don't have credentials on file yet. If it's
non-zero and you need to know what's available, call
`bb.secrets.list()`.

## Storage

Values live at `data/credentials/secrets.json`, mode `0600`, owned by
the main-process user. The `boxbot-sandbox` user has no read on the
file. Values are unencrypted at rest — same protection class as
`.env`, which sits next to it. If that protection class becomes
inadequate, the right fix is filesystem-level encryption (LUKS on
`/data`), not per-file crypto.
