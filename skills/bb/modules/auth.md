# bb.auth — registered users, registration codes, admin notify

`bb.auth` is an RPC façade onto the main-process `AuthManager`. It
reads user/admin state, mints single-use registration codes, and
messages every admin. No raw secrets ever cross this surface — channel
credentials stay in the main process.

For the full onboarding playbook (first boot, inviting users), load
the `onboarding` skill; this doc is the API reference.

## API

### Who is registered?

```python
import boxbot_sdk as bb

users = bb.auth.list_users()
# → [{"id": …, "name": "Jacob", "phone": "+1503…", "role": "admin",
#     "created_at": …, "last_seen": …}, …]
```

An empty list means no admin has been bootstrapped yet — the canonical
"is BB set up?" signal.

### First-admin bootstrap

```python
code = bb.auth.generate_bootstrap_code()   # 6 digits, 10 min, single use
```

Only works while **no admin exists** (the auth layer enforces this;
afterwards it raises). Surface the code **on the 7" screen** via
`switch_display` — the security property is *physical presence*. Never
relay it over voice or any messaging channel. The human texts the code
to BB from their phone and becomes the first admin.

### Inviting a new user (admin-initiated)

```python
code = bb.auth.generate_registration_code()
```

Requires the current conversation's sender to be a registered
**admin** on a messaging channel — the main process resolves who you
are from conversation context; the sandbox cannot pass an arbitrary
inviter. Reply to the inviting admin with the code so they can forward
it out-of-band. The new user texts it to BB and registers as a
standard user. Codes: 6 digits, 10-minute expiry, single use, rate
limited (3/hour per admin).

### Message every admin

```python
bb.auth.notify_admins("Heads up: a new user just registered.")
```

Delivers to each admin on their **registered channel** — every user
row carries a `channel` field ("signal" or "whatsapp") and the main
process resolves the right outbound client per admin. Use sparingly:
security notifications and "new user joined" pings, not general chat
(use the normal outputs array to talk to people).

## Error semantics

The writes (`generate_bootstrap_code`, `generate_registration_code`,
`notify_admins`) raise `bb.ActionError` on rejection — bootstrap
attempted after an admin exists, a non-admin trying to mint an invite,
rate limit hit, empty notification text. `list_users` returns the list
shape above and only raises if the auth manager itself is unavailable.

## What this module will never do

- Hand you a code *path* that skips the human: codes always travel
  out-of-band (screen for bootstrap, admin's phone for invites).
- Register a user directly. Registration only happens when an unknown
  number texts a valid code — that path lives in the message router,
  not in the SDK.
- Reveal channel credentials or message content.
