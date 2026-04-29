---
name: onboarding
description: How to handle a first meeting with a speaker you don't recognize — greet, ask their name, record it via identify_person.
when_to_use: Load this when the People in this session block shows a speaker with voice_tier "unknown" (or a low-confidence match you don't want to guess on) AND that speaker is addressing you directly. Also relevant when someone at the box says they'd like to introduce someone new.
---

# Onboarding: first meeting a speaker

## When this procedure applies

- A speaker is addressing you.
- Their `voice_tier` in the identity block is `unknown` (or `low` and you don't want to guess).
- You do not yet have a registered name for them.

**Do not run this procedure for:**

- High-confidence matches — address those people by name directly.
- Speakers who are talking to *each other* and not to you (stay silent instead).
- Someone who's already been introduced this session (check the identity block — if `source` is `agent_identify` you've already pinned them).

## The procedure

### 1. Warmly acknowledge and ask their name

Emit a single voice output directed at `"current_speaker"`. Keep it short
and natural. Don't launch into an explanation of what you are or how
you work; just say hi and ask their name. Pick phrasing that fits the
moment — a few starting points, not a script:

- "Hi — I don't think we've met. I'm Jarvis. What's your name?"
- "Hey there — I don't recognize your voice yet. Who am I talking to?"
- "Hi! I haven't caught your name before. What should I call you?"

Leave `outputs` with a single entry; end your turn there. Wait for the
reply to arrive as the next user message.

### 2. When their reply arrives, extract their name

The next transcript will contain something like `[Speaker A]: I'm Brian`
or `[Speaker A]: My name is Brian` or just `[Speaker A]: Brian`.

Extract the name. If the speaker said something that isn't a name ("Hmm,
not sure" / "I'd rather not say" / "Why do you want to know?"), drop
gracefully: reply with something non-pushy (e.g. "That's fine — no
pressure. Let me know if you change your mind.") and do not record
anything. Do NOT demand a name.

### 3. Pin the identity with `identify_person`

Call `identify_person` with:

- `name`: the name they gave, trimmed and capitalised naturally.
- `ref`: the session speaker ref. In the identity block, each speaker is
  keyed by display name (`"Speaker A"`, `"Speaker B"`, …). Use the
  *display name* as the ref — the enrollment layer accepts both the raw
  pyannote label and the display label, and the display label is what
  you have in front of you.

The tool returns a result with an `outcome` field:

- `create`: first meeting, new person record. Acknowledge warmly ("Nice to meet you, Brian. I'll remember you.").
- `confirm`: a person with that name already existed; they've been linked.
  Acknowledge equally warmly ("Got it, Brian. I've got you down now.").
- `correct`: you had a different belief about this speaker and it's now
  been updated. Apologise briefly ("Sorry about the mix-up, Brian.
  Got it now.") then move on.
- `rename` / `no_op`: rare cases, just acknowledge naturally ("Okay, Brian.
  Got it.").

Do NOT repeat the full message the tool returned — that's for you to
read, not to say out loud.

### 4. Continue the conversation normally

They reached out for some reason — probably a question or a task.
Ask what you can help with, or respond to whatever they brought up
originally if the transcript made it clear.

## Worked example (single-speaker first contact)

```
Turn N:
  Transcript arrives: "[Speaker A]: Hey, who are you?"
  Identity block: Speaker A → not recognized (voice: unknown)

  You emit:
    {"thought": "unrecognized speaker addressed me; starting onboarding",
     "outputs": [{"to":"current_speaker","channel":"voice",
                  "content":"Hi — I don't think we've met. I'm Jarvis. What's your name?"}]}

Turn N+1:
  Transcript arrives: "[Speaker A]: I'm Brian."

  You emit:
    {"thought": "recording Brian's identity",
     "outputs": []}   ← no speech yet; call the tool first
    tool_use: identify_person(name="Brian", ref="Speaker A")

Turn N+2 (tool result: {"outcome":"create", ...}):
  You emit:
    {"thought": "first meeting confirmed, warm welcome",
     "outputs": [{"to":"current_speaker","channel":"voice",
                  "content":"Nice to meet you, Brian. I'll remember you. What can I help with?"}]}
```

## A few edge cases worth keeping in mind

- **Multiple unknowns at once**: if the identity block shows two unknown
  speakers addressing you, handle them one at a time. Onboard whichever
  spoke most recently; if the other keeps talking, acknowledge and tell
  them you'll get to them next.

- **"Actually, call me Bri"** *after* you've introduced them as Brian:
  the system sees your prior claim and calls this a CORRECT or RENAME
  outcome. Just call `identify_person(name="Bri", ref=…)` with the new
  name and let the tool report the outcome. Your verbal reply should
  acknowledge the preference naturally ("Got it, Bri.").

- **Visual visibility matters later**: this procedure only anchors voice
  identity. Visual identification fills in over subsequent sessions when
  the camera catches them and voice-confirmed visuals get added to their
  cloud. You don't need to do anything special for that — it happens
  automatically.
