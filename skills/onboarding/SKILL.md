---
name: onboarding
description: How to onboard new people to boxBot — voice first-meeting (Person), first-admin bootstrap, admin-initiated user registration (User), and welcoming a freshly-registered user.
when_to_use: |
  Load this when ANY of:
    - The "People in this session" block shows a speaker with voice_tier
      "unknown" (or a low-confidence match you don't want to guess on)
      AND that speaker is addressing you directly.
    - There are pending `setup:` todos (a fresh device with no admin yet).
    - A registered admin (voice or WhatsApp) asks to add a new user.
    - You receive a `[REGISTRATION] <code>` message — that's a freshly
      registered user's first turn and they need a welcome.
---

# Onboarding

There are two distinct things called "onboarding" in boxBot — keep them
straight before you act:

| Concept | What it represents | How to onboard |
|---------|--------------------|----------------|
| **Person** | A voice/visual identity (perception layer). Lets BB recognize someone walking by, address them by name. | Voice first-meeting procedure (§1) — uses `identify_person`. |
| **User** | A WhatsApp account that can message BB and receive messages back (auth layer). | Registration code procedure (§2 / §3) — uses `bb.auth`. |

A Person and a User can refer to the same human, but they're tracked
separately. Adding one does not add the other. The "voice fingerprint"
step in setup is what links them for the admin.

---

## §1 — Voice first-meeting (Person creation)

### When this procedure applies

- A speaker is addressing you in a voice conversation.
- Their `voice_tier` in the identity block is `unknown` (or `low` and
  you don't want to guess).
- You do not yet have a registered name for them.

**Do NOT run this procedure for:**

- High-confidence matches — address those people by name directly.
- Speakers who are talking to *each other* and not to you.
- Someone who's already been introduced this session (check the
  identity block — if `source` is `agent_identify` you've already pinned
  them).

### The procedure

1. **Warmly acknowledge and ask their name.** Single voice output
   directed at `"current_speaker"`. Keep it short and natural — don't
   launch into an explanation of what you are. Pick phrasing that fits
   the moment:

   - "Hi — I don't think we've met. I'm Jarvis. What's your name?"
   - "Hey there — I don't recognize your voice yet. Who am I talking to?"
   - "Hi! I haven't caught your name before. What should I call you?"

   End your turn there. Wait for the reply.

2. **Extract the name from the reply** ("[Speaker A]: I'm Brian", "My
   name is Brian", "Brian"). If they decline gracefully, drop it — do
   NOT demand a name.

3. **Pin the identity with `identify_person`.**

   - `name`: the name they gave, trimmed and capitalised naturally.
   - `ref`: the session speaker ref (display name like `"Speaker A"`).

   Tool outcomes:

   - `create`: first meeting, new person record.
     "Nice to meet you, Brian. I'll remember you."
   - `confirm`: a person with that name already existed; linked.
     "Got it, Brian. I've got you down now."
   - `correct`: you had a different belief; updated.
     "Sorry about the mix-up, Brian. Got it now."
   - `rename` / `no_op`: rare; just acknowledge naturally.

4. **Continue the conversation normally.** They reached out for some
   reason — ask what you can help with.

---

## §2 — First-admin bootstrap (initial setup)

### When this applies

There are no admins registered yet. The seeded backlog will surface a
todo whose description starts with `setup:bootstrap`. You'll also see
"No users are registered yet" in the **Registered users** block.

### The procedure

1. **Mint the bootstrap code** via `bb.auth.generate_bootstrap_code()`.
   It returns a 6-digit numeric code. Single-use, expires in 10 minutes.

2. **Surface the code on the HDMI screen** with `switch_display`:

   ```python
   switch_display("notice", args={
       "title": "Welcome to boxBot!",
       "lines": [
           "Text this code to BB's WhatsApp number:",
           f"Code: {code}",
           "Expires in 10 minutes",
       ],
   })
   ```

   The security property is **physical presence** — only someone at the
   box can read the screen. Do NOT relay the code via voice or any
   messaging channel.

3. **Wait.** The user texts the code from their phone. The router
   validates it and emits a `UserRegistered` event with `role="admin"`
   and a `WhatsAppMessage` tagged `[REGISTRATION] <code>`.

4. **When the registration lands**, mark `setup:bootstrap` complete and
   move to §4 (welcome the new user).

5. **If 10 minutes pass with no registration**, the code expires
   silently. Generate a new one and re-display it. Don't error out —
   the human may simply have walked away briefly.

---

## §3 — Admin-initiated user registration (adding more users)

### When this applies

A registered admin says (via WhatsApp or voice) something like "add a
new user", "register Carina", "give my wife access".

### The procedure

1. **Confirm who they want to add.** A name is enough — you don't need
   the new user's phone number, the admin shares the code out-of-band.

2. **Generate a code** via `bb.auth.generate_registration_code()`.
   The main process resolves the inviting admin from the conversation
   context — you don't pass `created_by`. If the call fails because the
   speaker isn't an admin, tell them so plainly.

3. **Send the code back to the admin.** WhatsApp text reply (or a voice
   line if they asked at the box). Format:

   > "Code for Carina: 529174. Share it with her — she should text it
   > to me. Expires in 10 minutes."

4. **Wait.** When the new user texts the code, you'll see a
   `UserRegistered` event with `role="user"` and `invited_by_phone`
   set to the admin's phone.

5. **Welcome the new user (§4) and notify the inviting admin** via
   `bb.auth.notify_admins(...)` or a direct reply to that admin:

   > "Carina just registered. ✓"

### Rate limit

Admins can mint 1 code per hour. If they ask for a second code within
that window, the call raises — explain to them and offer to share the
existing code again (which is still valid until used or expired).

---

## §4 — Welcoming a freshly registered user

### Triggered by

- A `UserRegistered` event, OR
- A `[REGISTRATION] <code>` message in the conversation thread.

### What to do

1. **For an admin (bootstrap path):** warm welcome via WhatsApp, ask
   what they'd like you to be called by them. When they answer, save
   their preferred name to system memory. Then move to
   `setup:voice_fingerprint` (§5) and `setup:household` (§6).

2. **For a regular user (admin-invited):** warm welcome via WhatsApp,
   brief one-line description of what you can do, ask their name if it
   wasn't passed through (most WhatsApp profiles include it). Notify
   the inviting admin. Don't run the setup todos — those only fire on
   first-admin bootstrap.

---

## §5 — Voice fingerprint linking (admin-only, after bootstrap)

After the first admin registers via WhatsApp, `setup:voice_fingerprint`
becomes the next todo. They have a User record but no Person record yet.

1. **Send a WhatsApp message** inviting them to come say hi at the box:

   > "Whenever you have a sec, come say hi at the box and I'll learn
   > your voice. Just say hello once you're nearby."

2. **The next time they address you in a voice conversation**, run §1
   (the voice first-meeting procedure) using the name they gave during
   `setup:greet_admin`. The `identify_person` outcome will be `create`
   (new Person) or `confirm` (Person already existed).

3. **Mark `setup:voice_fingerprint` complete** on a successful
   `create`/`confirm`/`rename` outcome.

---

## §6 — Household basics (optional)

`setup:household` is the catch-all. Ask the admin (whichever channel
they prefer) about other household members worth knowing, the city to
use for weather, and any house preferences. Save anything durable to
system memory via the existing memory tool. If they don't want to
answer, mark complete or cancel — this todo is optional.

---

## Quick edge-case reference

- **Multiple unknowns at once (voice):** handle one at a time. Onboard
  the most recent speaker; tell the other you'll get to them next.
- **"Actually, call me Bri" after Brian was pinned:** call
  `identify_person(name="Bri", ref=…)` — the tool reports `correct` /
  `rename`. Acknowledge: "Got it, Bri."
- **Admin asks to add a user but isn't actually admin:** the
  `bb.auth.generate_registration_code()` call fails with "only admins
  can generate registration codes". Don't pretend it worked — tell
  them plainly they're not an admin.
- **WhatsApp not configured (no phone number ID / token):** §2-§4 are
  unavailable. The "No users are registered yet" line in the
  **Registered users** block hints at this. Don't try to mint codes
  you can't deliver.
- **Visual recognition fills in over time:** §1 only anchors voice.
  Visual ID gets seeded once the camera catches a voice-confirmed
  speaker. Nothing for you to do — happens automatically.
