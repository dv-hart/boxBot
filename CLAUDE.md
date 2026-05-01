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

Three layers, each with a different loading strategy and context cost.
The big shift (2026-04): most capability lives in the `bb` Python
package (sandbox SDK), reached through `execute_script`. Only genuinely
hot-path or security-sensitive operations remain as standalone tools.

**Tools** (`src/boxbot/tools/`) — 7 always-loaded tools:
- `execute_script` — primary gateway: run Python in the sandbox with
  the `bb` package (photos, camera, workspace, display, memory, tasks,
  skill, secrets, packages, calendar). Composes many ops per turn.
- `switch_display` — change what's on the 7" screen (hot-path singleton)
- `identify_person` — register / correct a speaker's identity; returns
  a cropped still of the speaker on new-person outcomes so the agent
  can note appearance into memory
- `manage_tasks` — triggers + to-dos (hot-path, every-turn)
- `search_memory` — memory lookup (hot-path, every-turn)
- `web_search` — web fetch + small-model content firewall
- `load_skill` — progressive-disclosure entry point (loads a skill's
  SKILL.md and optional sub-files into context on demand)

Speech and WhatsApp replies flow through the structured output
(`response_text` / `outputs` array) rather than tools — see
`output_dispatcher` and the `tools/registry.py` module docstring.

**SDK** (`src/boxbot/sdk/`, importable as `boxbot_sdk` or just `bb`
inside `execute_script`) — the constrained, immutable Python API:
- `bb.workspace` — filesystem-backed notebook (notes, CSVs, saved
  images). Path-safe, quota-capped, grep-searchable. The "now look it
  up" counterpart to memory's "rings a bell."
- `bb.camera` — `capture` / `capture_cropped` stills; images attach
  straight to the tool result so the agent sees pixels
- `bb.photos` — search, get, view (attaches pixels), show_on_screen
- `bb.display` — declarative block-based display builder + preview
- `bb.memory` — save/search/invalidate (shares backend with tool)
- `bb.tasks` — triggers + to-dos (shares backend with tool)
- `bb.skill` — create new skills at runtime
- `bb.secrets` — write-only credential storage
- `bb.packages` — request package install (human approval required)
- `bb.calendar` — Google Calendar read/write

The SDK communicates with the main process through structured JSON on
stdout + stdin (streaming, bidirectional). `execute_script` reads
action lines line-by-line, dispatches to per-module handlers, writes
JSON replies back to the sandbox, and collects image attachments into
a multimodal tool result. The agent never writes code that runs in the
main process.

**Self-documentation:** the `skills/bb/` skill is the agent's map to
the `bb` package. Its `SKILL.md` is injected into the system prompt at
Level 1 (metadata) and loaded at Level 2 (module index) on demand;
per-module docs (`skills/bb/modules/*.md`) load at Level 3 as needed.

**Skills** (`src/boxbot/skills/` + `skills/`) — modular capabilities
loaded per-conversation based on relevance:
- Declares name, description, and parameter schema
- Can be built-in, user-installed, or agent-created (via SDK)
- Skill logic runs as sandboxed Python scripts
- Only relevant skills are injected, conserving context

### 4. Sandbox & Agent Authoring
The agent operates through a **sandbox** — a separate venv (default
`/var/lib/boxbot-sandbox/venv`, override via ``sandbox.runtime_dir`` in
config) isolated from the main application. The sandbox lives outside
the project tree on purpose: this is open-source, and the
``boxbot-sandbox`` system user can't traverse a 0700 home directory
just to reach the venv. Sandbox security is enforced at the **OS
level** (not Python level):
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

Alongside memory there is the **agent workspace** (`data/workspace/`,
exposed as `bb.workspace`) — a filesystem-backed notebook the agent
owns. Memory = "rings a bell" (recognize that something is relevant);
workspace = "now look it up" (read the detail from a file). A 40-item
Pokémon list lives in `workspace/notes/people/erik/pokemon.md`; memory
just knows the file exists. This separation keeps memory retrieval
sharp (no giant records, no tiny records diluting search) and gives
the agent a real persistent scratch space for notes, CSVs that drive
displays, drafts, and images it captured or saved.

See [docs/memory.md](docs/memory.md) for the memory design and
[skills/bb/modules/workspace.md](skills/bb/modules/workspace.md) for
the workspace API.

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
┌──────────────────────────────────────────────────┐
│           Claude Agent (large model)              │
│      conversations · reasoning · authoring        │
├──────────────────────┬───────────────────────────┤
│  Tools (7)           │  Small model (async)      │
│  execute_script      │  photo tagging            │
│  switch_display      │  intent classification    │
│  identify_person     │  structured extraction    │
│  manage_tasks        │  transcript filtering     │
│  search_memory       │  memory search reranking  │
│  web_search          │  web search filtering     │
│  load_skill          │                           │
│  (+ structured       │                           │
│   outputs[voice,     │                           │
│   text] for speech)  │                           │
├──────────────────────┴───────────────────────────┤
│  Sandbox (isolated venv, streaming IO)            │
│  ┌──────────────────────────────────────────────┐ │
│  │ boxbot_sdk / bb  (immutable, declarative)    │ │
│  │ workspace · camera · photos · display        │ │
│  │ memory · tasks · skill · secrets             │ │
│  │ packages · calendar                          │ │
│  ├──────────────────────────────────────────────┤ │
│  │ agent scripts · skill scripts · user pkgs    │ │
│  └──────────────────────────────────────────────┘ │
│  Image attachments from sandbox → multimodal      │
│  tool result (workspace/photos/crops/tmp only)    │
├──────────┬──────────┬──────────┬─────────────────┤
│  Skills  │ Displays │  Memory  │   Scheduler     │
│ (modular,│ (screen) │ (recall) │  (lifecycle)    │
│  skills/bb = bb-package reference)                │
├──────────┴──────────┴──────────┴─────────────────┤
│              Communication Layer                  │
│         (Voice I/O · WhatsApp · Buttons)          │
├──────────────────────────────────────────────────┤
│              Perception Pipeline                  │
│    (Person Detection · ReID · Speaker ID · VAD)   │
├──────────────────────────────────────────────────┤
│            Hardware Abstraction Layer             │
│   (Camera · Mic · Speaker · Screen · Hailo NPU)   │
└──────────────────────────────────────────────────┘
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

## Deploying to the Pi

There is **one canonical deploy path**. Follow it unless you have a
specific reason to use the fast-iter alternative.

### Canonical: `scripts/deploy.sh`

```bash
# From the dev machine, after committing your work:
./scripts/deploy.sh                     # default target pi@boxbot.local
./scripts/deploy.sh other@host          # override target
```

What it does, in order:

1. **Pre-flight**: refuses to deploy unless the working tree is clean,
   you're on `main`, and `main` isn't behind `origin/main`. These
   checks exist because everything that runs on the Pi should be
   reproducible from a git SHA.
2. **`git push origin main`** — no-op if already pushed.
3. **SSH to Pi → `git fetch && git pull --ff-only origin main`** —
   refuses non-fast-forward merges, so the Pi can never silently
   diverge.
4. **Warns** if `scripts/setup-sandbox.sh` changed in this deploy. The
   operator runs `sudo bash scripts/setup-sandbox.sh` manually after
   the deploy because it needs sudo and may want attention.
5. **`scripts/restart-boxbot.sh` on the Pi** — SIGTERMs the running
   boxbot, waits up to 15s for clean exit, SIGKILLs if needed, then
   spawns a fresh one with a new dated log under `logs/`.
6. **Tails the startup log** so you see whether it came up cleanly.

If any step fails, the script exits non-zero and the Pi is left in a
recoverable state (last good commit + still-running old boxbot until
restart actually fires).

### Pi-only state that is never deployed

The deploy script never touches these — they live only on the Pi:

- `.env` (API keys, secrets) — gitignored, mode 0600.
- `config/config.yaml`, `config/whatsapp.yaml` — gitignored runtime
  config. Templates `*.example.yaml` are in git for reference. To
  change runtime config, SSH in and edit on the Pi directly.
- `data/**` — memory DB, perception DB, photos, workspace, scheduler
  DB, credentials. All Pi-local.
- `logs/**` — runtime logs.

### Fast-iter alternative: `scripts/deploy-to-pi.sh`

Pure rsync of the working tree. **Does not commit, does not push, does
not restart boxbot.** Use only when iterating tight on a change you
may not keep. Acknowledges the audit-trail bypass.

```bash
./scripts/deploy-to-pi.sh
ssh pi@boxbot.local 'cd software/boxBot && bash scripts/restart-boxbot.sh'
```

After the iteration is done, **always** finish with the canonical path
(commit + `scripts/deploy.sh`) so the Pi's git state matches reality.

### One-time Pi reconcile

If the Pi's local git is behind what's actually deployed (e.g. after a
period of fast-iter rsync deploys), reconcile once:

```bash
ssh pi@boxbot.local 'cd software/boxBot && git fetch origin && git reset --hard origin/main'
```

Safe because `data/`, `.env`, `logs/`, `config/config.yaml` are all
gitignored and untouched by `git reset`.

## Hardware Reference
- **Compute:** Raspberry Pi 5 (8 GB)
- **AI Accelerator:** Raspberry Pi AI HAT+ (13 TOPS Hailo-8L)
- **Display:** 7" HDMI LCD (H), 1024x600, IPS
- **Camera:** Pi Camera Module 3 Wide NoIR (12MP, 120 degree)
- **Microphone:** ReSpeaker XMOS XVF3000 4-Mic USB Array
- **Speaker:** Waveshare 8 ohm 5W
- **Input:** Adafruit KB2040 (RP2040) for button/knob input
- **Power:** Official Pi 27W PD Supply (5.1V/5A)
