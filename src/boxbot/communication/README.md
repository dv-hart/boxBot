# communication/

All external communication channels. This is the boundary between boxBot
and the outside world — every path here is authenticated and intentional.

For the full voice pipeline design — turn detection, barge-in, session
management, and audio architecture — see
[docs/voice-pipeline.md](../../../docs/voice-pipeline.md).

## Channels

1. **Voice** — direct speech via microphone and speaker (always available)
2. **WhatsApp** — registered users only (default remote channel)
3. **Buttons** — physical input from the KB2040 controller

There is no web UI, no REST API, no open network listener.

**WhatsApp is a privileged channel.** It is for user-to-BB communication
only — not a proxy for other services. Email checking, calendar sync,
and other "inbox" features are **skills** that run in the sandbox with
their own credentials. BB may relay results via WhatsApp or voice, but
the fetching never touches the messaging path.

## Key Concepts

**Person** — anyone BB recognizes via perception. May not have remote
access.

**User** — a person with a registered remote channel (WhatsApp number).
Can message BB, receive reminders, etc.

**Admin** — a user with elevated permissions. Can register new users,
approve packages, promote admins.

See [docs/user-registration.md](../../../docs/user-registration.md) for
the full registration flow.

## Files

### `voice.py`
Voice adapter and pipeline orchestrator. Coordinates wake word, VAD,
STT, TTS, and diarization. The conversation state machine itself
lives in `boxbot.core.conversation`; voice is now a thin I/O layer
that produces transcripts and speaks the agent's outputs. Mid-reply
interruption is wake-word-gated: STT detaches while BB is speaking
and the wake word is the only path back.

See [docs/voice-pipeline.md](../../../docs/voice-pipeline.md) for the
full design.

### `wake_word.py`
OpenWakeWord integration. Listens for "BB" (bee-bee) on CPU with
configurable confidence threshold. Always active during IDLE and
SESSION SUSPENDED states.

### `vad.py`
Silero VAD wrapper. Neural network-based voice activity detection
that distinguishes speech from residual noise after XMOS hardware
filtering. Provides the speech/silence signal that drives the
silence persistence model for turn detection.

### `stt.py`
STT provider interface and ElevenLabs Scribe implementation.
Provider-agnostic — alternative engines (Whisper, Deepgram,
AssemblyAI) implement the same `STTProvider` protocol.

### `tts.py`
TTS provider interface and ElevenLabs implementation. Supports
streaming playback — BB starts speaking before the full agent
response is generated. Manages audio output to the speaker via HAL.

### `audio_capture.py`
ReSpeaker audio stream management. Handles the raw audio interface
with the XMOS XVF3000 — sample rate, channel configuration, and
buffer management. Provides cleaned mono audio to downstream
consumers (VAD, STT, diarization).

### `diarization.py`
pyannote integration for speaker diarization ("who spoke when").
Shared between the voice pipeline (speaker attribution in
transcripts) and the perception pipeline (speaker identification
and voice cloud management). During conversation, both systems
consume the same diarization output.

### `whatsapp.py`
WhatsApp Business API integration:
- Receives messages via webhook
- First check: is the sender a registered user? If not → silent drop
- Only exception: message contains a valid registration code
- Sends messages and media (photos) to registered users
- Message types: text, image (with caption), voice note
- Routes incoming messages to the agent as conversation events
- Handles message queuing for when the agent is sleeping (process on wake)
- Validates webhook request signatures (prevents forged webhooks)

### `channels.py`
Abstract `Channel` interface for fork-friendly extensibility:
- `send_message(user_id, text)` — send outgoing message
- `receive_message()` → `{sender_id, text, media}` — incoming message
- `verify_sender(sender_id)` — confirm sender identity
- WhatsApp is the default implementation; forks can add Telegram,
  Signal, SMS, etc. by implementing this interface
- The registration flow, user management, and message routing are
  channel-agnostic — only the transport changes

### `auth.py`
User registration and authentication:

**Registration codes:**
- 6-digit, cryptographically random, single-use, time-limited (10 min)
- Only admins can generate codes
- Code sent to the admin via WhatsApp — admin shares with new user
  out-of-band
- BB never initiates contact with unknown numbers

**First admin bootstrap:**
- During initial setup, code displayed on the 7" screen
- User texts code to BB's WhatsApp → registered as primary admin
- Bootstrap flow disabled permanently after first admin registers

**User management:**
- Register: admin generates code → new user texts code → registered
- Promote: admin requests → confirmation → promoted
- Remove: admin requests → confirmation → removed
- Demote: admin can demote other admins (not self, not primary)

**Brute-force protection:**
- Failed attempts produce NO response (no information leakage)
- Rate limiting: 5 attempts per number per 10-minute window
- Temp block after 5 failures (1 hour), permanent after 3 temp blocks
- Admin notified of repeated failures

**Permission levels:**
- `admin` — register/remove users, approve packages, promote admins,
  change settings, all standard permissions
- `standard` — send/receive messages, interact with skills, share photos

### `users.py`
User storage and lookup:
- SQLite-backed user records (not config files)
- CRUD operations on user accounts
- Link/unlink users to perception person profiles
- User record schema: id, phone, name, role, person_id, channel,
  registered_at, registered_by, blocked, last_active

## Message Flow (WhatsApp)

```
Incoming WhatsApp message
         │
         ▼
  Webhook signature valid?
  ├─ No  ──► Reject (HTTP 403)
  └─ Yes ──► Extract sender phone number
              │
              ▼
        Registered user?
        ├─ Yes ──► Route to agent with user identity + person profile
        │          Agent processes, may respond via WhatsApp/voice
        │
        └─ No  ──► Contains valid registration code?
                   ├─ Yes ──► Register user, welcome message
                   └─ No  ──► Silent drop (no response, no error)
                              Increment attempt counter
                              Rate limit check → temp/perm block
```

## What Does NOT Go Through WhatsApp

These are **skills** running in the sandbox, not WhatsApp features:
- Email checking (IMAP, Gmail API)
- Calendar sync (Google Calendar, CalDAV)
- RSS/news feeds
- Social media monitoring
- Any third-party inbox or notification service

The agent may relay results to users via WhatsApp (or voice), but the
data fetching runs in the sandbox with its own approved packages and
credentials. This separation keeps the privileged messaging channel
clean and prevents mixing of protocols.
