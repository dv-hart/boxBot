# User Registration & Authentication

## Concepts

### People vs Users vs Admins

**Person** — anyone boxBot recognizes via perception (visual ReID +
speaker diarization). Has embeddings stored locally. May or may not
have remote access. A regular visitor who BB greets by name but who
can't message it remotely.

**User** — a person with an authenticated remote communication channel.
In the default implementation, this is a WhatsApp number. Users can
message BB, receive reminders, share photos, etc. Forks can add other
channels (Telegram, Signal, SMS) by implementing the same interface.

**Admin** — a user with elevated permissions. Can register new users,
approve package installs, promote other users to admin, and modify
settings. At least one admin must exist for the system to operate.

The relationship:

```
Person (perception only)
  └── User (+ remote channel)
        └── Admin (+ elevated permissions)
```

A person becomes a user through registration. A user becomes an admin
through promotion by an existing admin. These are one-way escalations
that require explicit human approval.

## WhatsApp Is a Privileged Channel

BB's WhatsApp number is for **user ↔ BB communication only**. It is
not a proxy for other services.

| Goes through WhatsApp | Goes through Skills (sandbox) |
|-----------------------|-------------------------------|
| User sends BB a message | Checking email (IMAP/Gmail API) |
| BB sends reminders, alerts | Calendar sync |
| Admin registers new users | RSS feeds, news |
| Admin approves packages | Any "inbox" style service |
| Photo sharing | API integrations |
| User registration codes | |

If a user wants BB to check their email, that's a **skill** running
in the sandbox with its own credentials and approved packages. BB may
relay the results via WhatsApp or voice ("you have 3 unread emails"),
but the fetching itself never touches the messaging path. This keeps
the privileged channel clean and prevents confusion between BB messages
and proxied content.

## Message Handling: Unknown Numbers

Messages from unregistered numbers are **hard blocked**:
- No response of any kind (silent drop)
- No error message, no "who are you?", no "this number is not registered"
- No information leakage — the sender cannot determine if BB's number
  is active, if it's a bot, or anything else
- Optionally logged for admin review (see `registration.log_blocked`)

The only exception: a message containing a **valid, unexpired
registration code** (see Registration Flow below).

## Registration Flow

### First Admin (Bootstrap)

The first admin is registered during initial device setup. This is the
only registration that doesn't require an existing admin.

```
1. User runs setup for the first time
         │
         ▼
2. BB displays on the 7" screen:
   ┌──────────────────────────────────┐
   │  Welcome to boxBot!              │
   │                                  │
   │  To register as admin, text      │
   │  this code to:                   │
   │                                  │
   │  +1 (555) 123-4567              │
   │                                  │
   │  Code: 847291                    │
   │  Expires in 10 minutes           │
   └──────────────────────────────────┘
         │
         ▼
3. User texts "847291" to BB's WhatsApp number
         │
         ▼
4. BB verifies:
   - Code matches? ✓
   - Code not expired? ✓
   - Code not already used? ✓
         │
         ▼
5. BB registers the sender's phone number as primary admin
   BB replies via WhatsApp: "You're registered as admin.
   What should I call you?"
         │
         ▼
6. User replies with their name
   BB saves: {phone: "+1...", name: "Jacob", role: "admin"}
         │
         ▼
7. Setup code is destroyed (single-use)
   BB speaks: "Setup complete. Hi Jacob."
```

**Security properties:**
- Code is displayed on the physical screen — requires physical presence
- Code is single-use — cannot be replayed
- Code expires after 10 minutes — cannot be used later if photographed
- Only one bootstrap registration is allowed — once an admin exists,
  this flow is permanently disabled
- If the code expires without being used, BB generates a new one

### Adding New Users (Admin-Initiated)

Only admins can initiate new user registration. The flow is designed
so that BB never contacts an unknown number — the admin is the
gatekeeper who shares the code.

```
1. Admin (via WhatsApp or voice): "Add a new user"
         │
         ▼
2. BB generates a registration code
   BB sends to ADMIN via WhatsApp:
   "Registration code: 529174
    Share this with the person you want to add.
    They should text it to me.
    Expires in 10 minutes. Single use."
         │
         ▼
3. Admin shares the code with the new user
   (in person, text, call — however they want)
         │
         ▼
4. New user texts the code to BB's WhatsApp number
         │
         ▼
5. BB verifies:
   - Valid, unexpired, unused code? ✓
   - Sender number not already registered? ✓
         │
         ▼
6. BB registers the sender as a standard user
   BB replies: "Welcome! What should I call you?"
   BB notifies admin: "New user registered: +1234567890"
         │
         ▼
7. Code is destroyed
   Admin can optionally link the user to a person profile:
   "That's Carina" → BB links user to person_02
```

**Why the admin never provides the phone number directly:**
- Typos could register the wrong person
- BB never initiates contact with unknown numbers (anti-spam)
- The code proves the new user has BB's number AND the code — both
  pieces must come together
- The admin controls distribution — they decide who gets the code

### Promoting to Admin

```
1. Admin: "Make Carina an admin"
         │
         ▼
2. BB sends confirmation to admin via WhatsApp:
   "Promote Carina (+1234567890) to admin?
    Reply YES to confirm."
         │
         ▼
3. Admin replies "YES"
         │
         ▼
4. BB updates Carina's role to admin
   BB notifies Carina: "You've been promoted to admin by Jacob."
```

### Removing Users

```
1. Admin: "Remove Carina"
         │
         ▼
2. BB sends confirmation to admin:
   "Remove Carina (+1234567890)? They will no longer be
    able to message me. Reply YES to confirm."
         │
         ▼
3. Admin replies "YES"
         │
         ▼
4. BB removes user from whitelist
   Future messages from that number → silent drop
   Person profile (perception) is NOT deleted — BB will
   still recognize them visually/vocally
```

### Demoting Admins

Admins can demote other admins but cannot demote themselves (prevents
accidental lockout). The primary admin (first registered) cannot be
demoted by other admins — only by physical button reset.

## Registration Codes

### Properties

- **6 digits** — easy to type on a phone, hard enough to brute-force
  with rate limiting
- **Single-use** — destroyed after successful registration or expiry
- **Time-limited** — 10 minutes default (configurable)
- **One active code at a time** — generating a new code invalidates
  the previous one
- **Rate limited** — max 3 code generations per hour per admin
- **Not sequential** — cryptographically random, no pattern to predict

### Brute-Force Protection

With 6 digits (1,000,000 possibilities) and rate limiting:
- Max 5 attempts per phone number per 10-minute window
- After 5 failed attempts: number is temporarily blocked (1 hour)
- After 3 temporary blocks: number is permanently blocked (admin can
  unblock)
- BB notifies admin of repeated failed attempts

```
Unknown number texts "123456"
         │
         ▼
  ┌─ Valid code? ─┐
  │               │
  │  No ──► Is there even an active code?
  │         ├─ No  ──► Silent drop (no response)
  │         └─ Yes ──► Increment attempt counter
  │                    If attempts > 5: temp block
  │                    Still silent drop (no response)
  │
  │  Yes ──► Register user, destroy code
  │
  └───────────────┘
```

**Critical:** failed attempts produce **no response**. The sender
cannot determine whether they got the wrong code, whether a code
exists, or whether BB is even active. This prevents enumeration.

## Linking Users to People

Users (WhatsApp accounts) and people (perception profiles) are
initially independent. They get linked through:

1. **Explicit linking:** Admin says "Carina's WhatsApp is the one
   ending in 7890" → BB links the user account to person_02
2. **Automatic linking:** When a registered user is physically present
   and speaks, BB can correlate the voice/visual profile with the
   user account (if the user has interacted via WhatsApp recently
   and the timing aligns)
3. **Enrollment linking:** If a person goes through perception
   enrollment (camera + voice capture) while their phone is registered,
   BB links them

Once linked, BB can do things like: Carina messages "remind Jacob
to take out the chicken" → BB knows Jacob's visual profile → when
Jacob walks by, BB sees him and delivers the reminder.

## Fork-Friendly Design

The user registration system is built on an abstract `Channel` interface.
WhatsApp is the default implementation, but forks can add:
- Telegram
- Signal
- SMS
- Custom channels

Each channel must implement:
- `send_message(user_id, text)` — send a message
- `receive_message()` → `{sender_id, text, media}` — incoming webhook
- `verify_sender(sender_id)` — confirm sender identity
- Sender identity must be cryptographically verified by the channel
  provider (WhatsApp/Telegram/Signal all do this)

The registration flow (code generation, verification, user management)
is channel-agnostic. Only the message transport changes.

## Storage

User records are stored in the local database (not in config files):

```
User:
  id: uuid
  phone: str             # WhatsApp number (E.164 format)
  name: str              # display name
  role: "admin" | "standard"
  person_id: uuid | null # link to perception profile
  registered_at: datetime
  registered_by: uuid    # admin who initiated registration
  channel: "whatsapp"    # which communication channel
  blocked: bool          # temporarily or permanently blocked
  last_active: datetime
```

The `config/whatsapp.yaml` file is for API credentials only, not user
management. Users are managed through the registration flow and stored
in the database.
