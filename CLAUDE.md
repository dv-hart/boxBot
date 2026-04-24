# boxBot — Project Principles

## What Is boxBot?
An open-source Claude agent that lives in an elegant wooden box, built on a
Raspberry Pi 5 with the Claude Agent SDK. It sees, hears, remembers, and
communicates — acting as an ambient household assistant that recognizes the
people around it and proactively helps them.

## Core Design Principles

### 1. Security First
All communication with the outside world flows through **structured,
authenticated paths only**:
- **Voice** — direct conversation via microphone and speaker
- **Camera** — passive perception, never streamed or shared externally
- **Button inputs** — physical interaction on the box
- **WhatsApp** — registered users only; unknown numbers are hard blocked
  (silent drop, no response, no information leakage)

There is **no open network listener, no web UI, no SSH-by-default**. The
only open port is the WhatsApp webhook (required by the API).

**User registration** uses single-use, time-limited codes:
- First admin: code displayed on screen during setup (physical presence)
- New users: admin generates code, shares it out-of-band, new user texts
  it to BB. BB never initiates contact with unknown numbers
- See [docs/user-registration.md](docs/user-registration.md) for details

**WhatsApp is a privileged channel, not a proxy.** Email, calendar, RSS,
and other "inbox" services are **skills** running in the sandbox with their
own credentials. BB may relay results via WhatsApp, but data fetching never
touches the messaging path.

### 2. Slim Core, Flexible Extensions
The repository stays focused on the core agent loop, hardware abstraction, and
plugin interfaces. Features are added through three extension systems:
- **Tools** — always-loaded core capabilities (switch display, run script,
  send message, manage memory/tasks)
- **Skills** — modular, optional capabilities the agent can invoke (weather
  lookup, reminders, home control, etc.)
- **Displays** — swappable screen layouts the agent rotates through or selects
  contextually

Tools are the agent's hands — always attached, always available. Skills are
items it picks up — modular, swappable, contributed by the community. Both
use simple base classes. Skills and displays use auto-discovery so
contributors can add them without touching core code.

### 3. Tools, SDK, and Skills

Three layers, each with a different loading strategy and context cost:

**Tools** (`src/boxbot/tools/`) — 9 always-loaded capabilities for
real-time, conversational operations:
- `execute_script` — run Python in the sandbox (gateway to the SDK)
- `speak` — say something through the speaker (real-time TTS)
- `switch_display` — change what's on the 7" screen (with optional
  display-specific `args`)
- `send_message` — WhatsApp to a whitelisted user
- `identify_person` — query who is present or recently seen
- `manage_tasks` — manage triggers (wake conditions) and to-do list
- `search_memory` — search, summarize, or retrieve stored memories
- `search_photos` — search and retrieve photos from the photo library
- `web_search` — search the web or fetch a URL, filtered by the small
  model (content firewall against prompt injection)

**SDK** (`src/boxbot/sdk/`) — a constrained, immutable Python API
pre-installed in the sandbox venv. Agent scripts import it to perform
complex or infrequent operations without dedicated tools:
- `boxbot_sdk.display` — create displays via declarative builder (the
  agent describes **what** to show using building blocks, never writes
  raw render code; the main process generates validated display classes)
- `boxbot_sdk.skill` — create skills via builder
- `boxbot_sdk.packages` — request package installation (user approval)
- `boxbot_sdk.memory` — save/search/invalidate memories (shares backend
  with `search_memory` tool)
- `boxbot_sdk.secrets` — store and use credentials (write-only, agent
  cannot view stored secrets after storage)
- `boxbot_sdk.photos` — manage photo library (tagging, slideshow
  curation, tag library, soft delete/restore; search shares backend
  with `search_photos` tool)
- `boxbot_sdk.tasks` — manage triggers (wake conditions) and to-do
  items within scripts (shares backend with `manage_tasks` tool)

The SDK communicates with the main process through structured JSON on
stdout — the `execute_script` tool parses these actions and applies them
safely. The agent never directly writes code that runs in the main process.

**Skills** (`src/boxbot/skills/` + `skills/`) — modular capabilities
loaded per-conversation based on relevance:
- Declares name, description, and parameter schema
- Can be built-in, user-installed, or agent-created (via SDK)
- Skill logic runs as sandboxed Python scripts
- Only relevant skills are injected, conserving context

### 4. Sandbox & Agent Authoring
The agent operates through a **sandbox** — a separate venv
(`data/sandbox/venv`) isolated from the main application. Sandbox
security is enforced at the **OS level** (not Python level):
- Scripts run as `boxbot-sandbox` user with minimal filesystem permissions
- **seccomp** blocks `execve`/`fork` — no subprocess spawning of any kind
- `.env` is mode `0600` owned by `boxbot` — sandbox cannot read secrets
- `site-packages` is read-only — sandbox cannot install packages
- The agent can only interact with boxBot through the **immutable SDK**
- The SDK is **declarative for displays**: building blocks only, no raw
  render code. Main process generates validated display classes
- Agent-created **skills auto-activate** (their logic runs sandboxed)
- Agent-created **displays require user confirmation** (runs in main process)
- **Package installation requires out-of-band human approval** — physical
  screen tap or WhatsApp reply from admin. The sandbox can only emit
  requests; there is no way to spoof approval
- See [docs/sandbox.md](docs/sandbox.md) for full details

### 5. Two-Model Architecture
boxBot uses two Claude models to balance capability and cost:
- **Large model** (`BOXBOT_MODEL_LARGE`) — conversations, reasoning, memory
  extraction, skill/display authoring, complex decision-making
- **Small model** (`BOXBOT_MODEL_SMALL`) — photo tagging, intent
  classification, structured data extraction, wake-word transcript filtering,
  web search filtering (content firewall), routine pre-processing that feeds
  into the large model's context

The small model handles high-frequency, low-stakes tasks. The large model
handles everything that requires judgment or creativity.

### 6. Memory-Driven Continuity
Three stores give the agent persistent knowledge:
- **System memory** — always-loaded household facts and standing instructions
  (`data/memory/system.md`, ~2-4 KB, auto-updated with guardrails)
- **Fact memories** — typed (person, household, methodology, operational),
  extracted from conversations, searchable via hybrid vector + keyword
  retrieval with small-model reranking
- **Conversation log** — ultra-compact summaries of every conversation,
  rolling 2-month window

Memories are **contextually injected** at conversation start based on who
is speaking and what they said. The `search_memory` tool provides three
modes: `lookup` (ranked results), `summary` (synthesized answers), and
`get` (full record by ID). **Access-based retention** keeps useful memories
alive and lets unused ones fade.

See [docs/memory.md](docs/memory.md) for the full design.

The agent should feel like it *knows* you, not like every conversation starts
from zero.

### 7. Multimodal Person Identification
The agent recognizes household members through fused visual and audio signals:
- **Visual ReID** — appearance embeddings clustered per person, run on Hailo
- **Speaker diarization** — voice embeddings for persistent speaker identity
- **Fusion** — audio confirms visual identity; voice teaches the visual system
  new appearances over time

This enables contextual awareness: the agent knows *who* it's talking to and
can act on person-specific tasks (e.g., relay a message from one family member
to another when they walk by).

### 8. Triggers, To-Do List, and Wake Cycle
The agent has a trigger-driven wake/sleep lifecycle and a persistent
to-do list:
- **Triggers** — event-driven wake conditions with AND logic. Types:
  point-in-time (`fire_at`), timer (`fire_after`, max 24h), recurring
  (`cron`), and person detection (`person`). Compound triggers combine
  conditions: "after 30 minutes, remind Jacob when you see him" =
  `fire_after` + `person`, both must be met
- **To-do list** — persistent action items the agent tracks. Lightweight
  list (descriptions only) with detailed notes loaded on demand. Reviewed
  during wake cycles and available during conversations
- **Wake cycle** — configurable recurring triggers (seeded from config)
  that wake the agent to check the to-do list, update displays, and
  act on pending items. Config seeds on first boot; agent can modify
  at runtime
- **Sleep state** — when idle, boxBot shows idle displays and listens
  only for wake words and person detection events
- **Conversation injection** — `[To-do: N items | Triggers: N active]`
  injected at conversation start so the agent can decide whether to
  consult its task list

### 9. Privacy by Design
- All person identification runs **on-device** (Hailo NPU + CPU)
- Photos and embeddings are stored **locally** (optional encrypted cloud backup)
- No telemetry, no data leaves the box except explicit API calls (Claude,
  WhatsApp, weather, etc.)
- Camera frames are processed and discarded — never stored unless the user
  explicitly saves a photo

### 10. One Conversation Abstraction
The agent sees text, not voice. Voice and WhatsApp are **I/O adapters**
that produce attributed transcripts and deliver outputs; the agent
processes conversations independent of transport.

- **`Conversation`** (`src/boxbot/core/conversation.py`) owns the full
  lifecycle of one logical interaction: its thread of messages, its
  participants, the I/O channels it covers, the in-flight generation
  task, and its state (`LISTENING` / `THINKING` / `SPEAKING` / `ENDED`).
- **Per-conversation serialization, cross-conversation parallelism.**
  One generation runs at a time within a conversation, but a voice
  conversation in the room runs fully in parallel with a WhatsApp
  conversation with a different person. No global conversation lock.
- **New input cancels in-flight generation.** If a user speaks while
  the agent is thinking or mid-TTS, the current generation is
  cancelled, partial delivery is folded into the thread as an
  interrupted assistant turn, and a fresh generation starts with the
  updated thread. The model sees what was actually delivered and
  decides what to do next.
- **Silence timeout lives on the conversation**, not the voice
  transport. The voice adapter is a thin layer: wake word activates
  mic capture, a `ConversationEnded(channel="voice")` event
  deactivates it. LEDs are a pure function of the voice-room
  conversation's state — no independent timers drift from the
  conversation's reality.
- **Conversation keys:** `voice:room` (one per physical room),
  `whatsapp:<phone>` (per sender), `trigger:<id>:<uuid>` (one-shot
  per firing).

## Architecture Boundaries

```
┌─────────────────────────────────────────────────┐
│           Claude Agent (large model)             │
│       conversations · reasoning · authoring      │
├─────────────────────┬───────────────────────────┤
│  Tools (9, always)  │  Small model (async)      │
│  execute_script     │  photo tagging            │
│  speak              │  intent classification    │
│  switch_display     │  structured extraction    │
│  send_message       │  transcript filtering     │
│  identify_person    │  memory search reranking  │
│  manage_tasks       │  web search filtering     │
│  search_memory      │                           │
│  search_photos      │                           │
│  web_search         │                           │
├─────────────────────┴───────────────────────────┤
│  Sandbox (isolated venv)                         │
│  ┌─────────────────────────────────────────────┐ │
│  │ boxbot_sdk (immutable, declarative)         │ │
│  │ display · skill · packages · memory         │ │
│  │ secrets · photos · tasks                    │ │
│  ├─────────────────────────────────────────────┤ │
│  │ agent scripts · skill scripts · user pkgs   │ │
│  └─────────────────────────────────────────────┘ │
├──────────┬──────────┬──────────┬────────────────┤
│  Skills  │ Displays │  Memory  │   Scheduler    │
│ (modular)│ (screen) │ (recall) │  (lifecycle)   │
├──────────┴──────────┴──────────┴────────────────┤
│              Communication Layer                 │
│         (Voice I/O · WhatsApp · Buttons)         │
├─────────────────────────────────────────────────┤
│              Perception Pipeline                 │
│    (Person Detection · ReID · Speaker ID · VAD)  │
├─────────────────────────────────────────────────┤
│            Hardware Abstraction Layer             │
│   (Camera · Mic · Speaker · Screen · Hailo NPU)  │
└─────────────────────────────────────────────────┘
```

## Development Guidelines

### Python
- Use `python3` explicitly — this system does not have `python` on PATH
- Target Python 3.11+ (Raspberry Pi OS Bookworm default)
- Use `pyproject.toml` for project metadata and dependencies
- Type hints on all public interfaces

### Code Style
- Keep modules small and focused — one responsibility per file
- Hardware access goes through the HAL, never direct GPIO/I2C in business logic
- All configuration via YAML files, never hardcoded
- Secrets (API keys, WhatsApp tokens) go in `.env`, never committed

### Testing
- Unit tests mock hardware; integration tests run on-device
- Skills and displays must include at least one test

### Contributing
- New skills go in `skills/` as standalone packages
- New displays go in `displays/` as standalone packages
- Core changes require discussion in an issue first

## Hardware Reference
- **Compute:** Raspberry Pi 5 (8 GB)
- **AI Accelerator:** Raspberry Pi AI HAT+ (13 TOPS Hailo-8L)
- **Display:** 7" HDMI LCD (H), 1024x600, IPS
- **Camera:** Pi Camera Module 3 Wide NoIR (12MP, 120 degree)
- **Microphone:** ReSpeaker XMOS XVF3000 4-Mic USB Array
- **Speaker:** Waveshare 8 ohm 5W
- **Input:** Adafruit KB2040 (RP2040) for button/knob input
- **Power:** Official Pi 27W PD Supply (5.1V/5A)
