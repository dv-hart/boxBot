# bb.auth — users, registration codes, admin notify

RPC façade onto the main-process `AuthManager`. Reads user/admin
state, mints single-use registration codes, messages every admin. No
raw secrets cross this surface — channel credentials stay in the main
process.

Full onboarding playbook: the `onboarding` skill. This is the API
reference.

## Who is registered

```python
import boxbot_sdk as bb

users = bb.auth.list_users()
# → [{"id": …, "name": "Jacob", "phone": "+1503…", "role": "admin",
#     "created_at": …, "last_seen": …}, …]
```

Empty list = no admin bootstrapped yet. That is the canonical "is BB
set up?" signal.

## First-admin bootstrap

```python
code = bb.auth.generate_bootstrap_code()   # 6 digits, 10 min, single use
```

Works only while **no admin exists**; raises afterward. Show the code
**on the 7" screen** via `switch_display` — the security property is
*physical presence*. Never speak it or send it over any messaging
channel. The human texts it to BB and becomes the first admin.

## Invite a user (admin-initiated)

```python
code = bb.auth.generate_registration_code()
```

The current conversation's sender must be a registered **admin** on a
messaging channel. The main process resolves who you are from
conversation context; the sandbox cannot name an arbitrary inviter.
Reply to that admin with the code; they forward it out-of-band. The new
user texts it to BB and registers as a standard user.

6 digits, 10-minute expiry, single use, 3/hour per admin.

## Message every admin

```python
bb.auth.notify_admins("Heads up: a new user just registered.")
```

Delivers to each admin on their registered `channel` ("signal" or
"whatsapp"); the main process picks the right outbound client. Use for
security notifications and "new user joined" — not general chat.

## Errors

`generate_bootstrap_code`, `generate_registration_code`,
`notify_admins` raise `bb.ActionError` on rejection: bootstrap after an
admin exists, non-admin minting an invite, rate limit, empty text.
`list_users` raises only if the auth manager is unavailable.

## Never

- Hand you a code path that skips the human. Codes always travel
  out-of-band — screen for bootstrap, admin's phone for invites.
- Register a user directly. Registration only happens when an unknown
  number texts a valid code, in the message router.
- Reveal channel credentials or message content.
