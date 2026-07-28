---
name: onboarding
description: How to onboard people to boxBot — voice first-meeting (Person), first-admin bootstrap, admin-initiated user registration (User), welcoming a freshly-registered user.
when_to_use: |
  Load when ANY of:
    - "People in this session" shows a speaker with voice_tier "unknown" (or a
      low-confidence match you won't guess on) AND that speaker is addressing
      you directly.
    - Pending `setup:` todos — a fresh device with no admin.
    - A registered admin asks to add a new user.
    - A `[REGISTRATION] <code>` message arrives — a freshly registered user's
      first turn; they need a welcome.
---

# Onboarding

Two different things share the name. Keep them straight:

| Concept | Is | Onboard via |
|---------|-----|-------------|
| **Person** | A voice/visual identity (perception layer). Lets BB recognize someone and use their name. | Voice first-meeting, §1 — `identify_person`. |
| **User** | A messaging account that can text BB and be texted back (auth layer). | Registration code, §2/§3 — `bb.auth`. |

One human can be both, but they are tracked separately. Adding one does
not add the other. The voice-fingerprint step (§5) links them for the
admin.

---

## §1 — Voice first-meeting (Person)

**Applies when:** a speaker is addressing you in voice, their
`voice_tier` is `unknown` (or `low` and you won't guess), and you have
no registered name for them.

**Does not apply to:** high-confidence matches (use their name);
speakers talking to each other; anyone already introduced this session
(`source: agent_identify` in the identity block).

1. **Ask their name.** One voice output to `"current_speaker"`. Short
   and natural. Do not explain what you are.

   - "Hi — I don't think we've met. I'm Jarvis. What's your name?"
   - "Hey there — I don't recognize your voice yet. Who am I talking to?"
   - "Hi! I haven't caught your name before. What should I call you?"

   End the turn. Wait.

2. **Extract the name** ("[Speaker A]: I'm Brian", "My name is Brian",
   "Brian"). If they decline, drop it. Never demand a name.

3. **Pin it** with `identify_person`: `name` = what they gave, trimmed
   and naturally capitalized; `ref` = the session speaker ref (e.g.
   `"Speaker A"`).

   | Outcome | Say |
   |---------|-----|
   | `create` | "Nice to meet you, Brian. I'll remember you." |
   | `confirm` | "Got it, Brian. I've got you down now." |
   | `correct` | "Sorry about the mix-up, Brian. Got it now." |
   | `rename` / `no_op` | Acknowledge naturally. |

   Renaming an *existing* record ("call me Bri") is
   `identify_person(action="rename", name="Brian", new_name="Bri")` —
   not a new identify.

4. **Continue.** They spoke up for a reason — ask what you can do.

---

## §2 — First-admin bootstrap

**Applies when:** no admins registered. A todo starting
`setup:bootstrap` is in the backlog, and the Registered users block
says "No users are registered yet."

1. `bb.auth.generate_bootstrap_code()` — 6 digits, single use, 10 min.

2. Put it **on the HDMI screen**:

   ```python
   switch_display("notice", args={
       "title": "Welcome to boxBot!",
       "lines": [
           "Text this code to BB's number:",
           f"Code: {code}",
           "Expires in 10 minutes",
       ],
   })
   ```

   The security property is **physical presence** — only someone at the
   box reads the screen. Never speak the code or send it over any
   channel.

3. Wait. They text it; the router validates and emits `UserRegistered`
   with `role="admin"` plus a message tagged `[REGISTRATION] <code>`.

4. On arrival: mark `setup:bootstrap` complete, go to §4.

5. Ten minutes with no registration → the code expires silently.
   Generate and re-display. Not an error; they probably walked away.

---

## §3 — Admin-initiated registration

**Applies when:** a registered admin says "add a new user", "register
Carina", "give my wife access".

1. **Confirm who.** A name is enough — the admin shares the code
   out-of-band, so you never need the new user's number.

2. `bb.auth.generate_registration_code()`. The main process resolves
   the inviting admin from conversation context; you pass no
   `created_by`. If the speaker isn't an admin the call fails — say so
   plainly.

3. **Send the code to the admin**, on whichever channel they asked:

   > "Code for Carina: 529174. Share it with her — she should text it
   > to me. Expires in 10 minutes."

4. Wait. On success you see `UserRegistered` with `role="user"` and
   `invited_by_phone` set.

5. Welcome them (§4) and tell the inviting admin, via
   `bb.auth.notify_admins(...)` or a direct reply:

   > "Carina just registered. ✓"

**Rate limit:** 1 code per admin per hour. A second request inside the
window raises. Explain, and offer to re-share the existing code — still
valid until used or expired.

---

## §4 — Welcoming a new user

Triggered by a `UserRegistered` event or a `[REGISTRATION] <code>`
message.

**Admin (bootstrap path):** warm welcome by text, ask what they'd like
you to call them, save their preferred name to system memory. Then §5
and §6.

**Regular user (admin-invited):** warm welcome by text, one line on
what you can do, ask their name if the profile didn't carry it. Notify
the inviting admin. Skip the setup todos — those fire only on first
bootstrap.

---

## §5 — Voice fingerprint (admin, after bootstrap)

The first admin has a User record but no Person record.
`setup:voice_fingerprint` is next.

1. Text them:

   > "Whenever you have a sec, come say hi at the box and I'll learn
   > your voice. Just say hello once you're nearby."

2. Next time they address you by voice, run §1 with the name from
   `setup:greet_admin`. Outcome will be `create` or `confirm`.

3. Mark `setup:voice_fingerprint` complete on
   `create`/`confirm`/`rename`.

---

## §6 — Household basics (optional)

`setup:household` is the catch-all. Ask the admin about other household
members worth knowing, the city for weather, house preferences. Save
durable answers to system memory. They don't want to answer? Mark
complete or cancel — this one is optional.

---

## Edge cases

- **Several unknowns at once:** one at a time. Onboard the most recent
  speaker; tell the other you'll get to them.
- **"Actually, call me Bri" after Brian was pinned:**
  `identify_person(action="rename", name="Brian", new_name="Bri")`.
  Embeddings, photos, and triggers follow. "Got it, Bri."
- **Two records, one human ("Eric" and "Erik"):** confirm first ("Are
  Eric and Erik the same person?"), then
  `identify_person(action="merge", name="Eric", duplicate_name="Erik")`.
  Merge is destructive. Never call it without confirmation.
- **Non-admin asks to add a user:**
  `bb.auth.generate_registration_code()` fails with "only admins can
  generate registration codes". Don't pretend it worked.
- **Messaging not configured:** §2–§4 are unavailable. The "No users
  are registered yet" line hints at it. Don't mint codes you can't
  deliver.
- **Visual recognition:** §1 anchors voice only. Visual ID seeds itself
  once the camera catches a voice-confirmed speaker. Automatic.
