# Architecture

## System Overview

boxBot is structured as a layered system where each layer has clear
responsibilities and communicates through defined interfaces.

```
┌─────────────────────────────────────────────────────┐
│              Agent Layer (large model)               │
│  Claude Agent SDK · System Prompt · Conversation     │
│  Management · Memory Injection · Tool Dispatch       │
├─────────────────────┬───────────────────────────────┤
│  Tools (9, always)  │  Small Model (async)          │
│  execute_script     │  Photo tagging                │
│  speak              │  Intent classification        │
│  switch_display     │  Structured extraction        │
│  send_message       │  Transcript filtering         │
│  identify_person    │  Memory search reranking      │
│  manage_tasks       │  Web search filtering         │
│  search_memory      │                               │
│  search_photos      │                               │
│  web_search         │                               │
├─────────────────────┴───────────────────────────────┤
│  Sandbox (isolated venv)                             │
│  ┌─────────────────────────────────────────────────┐ │
│  │  boxbot_sdk (immutable, declarative)            │ │
│  │  display · skill · packages · memory            │ │
│  │  secrets · photos · tasks                       │ │
│  ├─────────────────────────────────────────────────┤ │
│  │  agent scripts · skill scripts · user packages  │ │
│  └─────────────────────────────────────────────────┘ │
├────────────┬────────────┬───────────┬───────────────┤
│   Skills   │  Displays  │  Memory   │  Scheduler    │
│ (modular)  │  (screen)  │  (recall) │  (lifecycle)  │
├────────────┴────────────┴───────────┴───────────────┤
│              Communication Layer                     │
│     Voice I/O · WhatsApp · Button Input · Auth       │
├─────────────────────────────────────────────────────┤
│              Perception Pipeline                     │
│  Person Detection · Visual ReID · Speaker ID · VAD   │
│  Fusion · Enrollment                                 │
├─────────────────────────────────────────────────────┤
│           Hardware Abstraction Layer                  │
│  Camera · Mic · Speaker · Screen · Buttons · Hailo   │
└─────────────────────────────────────────────────────┘
```

## Two-Model Architecture

boxBot routes work to two Claude models based on task complexity:

| Model | Used For | Characteristics |
|-------|----------|-----------------|
| **Large** (`BOXBOT_MODEL_LARGE`) | Conversations, reasoning, memory extraction, skill/display authoring, complex decisions | Higher cost, deeper reasoning |
| **Small** (`BOXBOT_MODEL_SMALL`) | Photo tagging, intent classification, structured extraction, transcript filtering, web search filtering | Lower cost, faster, high-frequency |

The small model handles high-frequency pre-processing that feeds into the
large model's context. For example: the small model classifies whether a
wake-word transcript is actually directed at boxBot or background speech,
tags incoming photos, extracts structured data from raw inputs, and filters
web search results for prompt injection and relevance before they reach the
large model.

## Tools, SDK, and Skills

The agent has three layers of callable capabilities, each with different
loading strategies and context costs:

**Tools** (9, always loaded) — real-time, conversational operations that
need immediate response. These are the only items in the agent's tool list:
`execute_script`, `speak`, `switch_display`, `send_message`,
`identify_person`, `manage_tasks`, `search_memory`, `search_photos`,
`web_search`.

**SDK** (`boxbot_sdk`, pre-installed in sandbox venv) — a constrained,
immutable Python API for complex or infrequent operations. Agent scripts
import it via `execute_script`. The SDK communicates with the main process
through structured JSON on stdout. Key modules:
- `display` — declarative display builder (agent describes **what** to show
  using building blocks; main process generates validated render code)
- `skill` — skill builder
- `packages` — package installation with user approval
- `memory` — save/search/invalidate memories (shares backend with
  `search_memory` tool)
- `secrets` — store and use credentials (write-only, agent cannot view
  stored secrets)
- `photos` — photo management
- `tasks` — trigger and to-do management within scripts

**Skills** (modular, per-conversation) — domain-specific capabilities
loaded based on relevance. Can be built-in, user-installed, or
agent-created via the SDK. Skill logic runs in the sandbox.

This design keeps the always-loaded context slim (9 tools) while giving
the agent full access to boxBot's capabilities through the SDK when needed.

## Sandbox

Agent-written scripts and skill logic run in an isolated sandbox:

```
Main Process (main venv)          Sandbox (sandbox venv)
┌──────────────────────┐          ┌──────────────────────┐
│ boxBot application   │  invoke  │ boxbot_sdk           │
│ Hardware drivers     │ ──────►  │ Agent scripts        │
│ Display rendering    │  stdout  │ Skill scripts        │
│ Agent conversation   │ ◄──────  │ User-installed pkgs  │
│                      │ (JSON)   │                      │
│ Applies SDK actions  │          │ ⛔ No hardware access │
│ ⛔ Agent can't modify│          │ ⛔ No boxbot imports  │
└──────────────────────┘          └──────────────────────┘
```

Key sandbox rules:
- Separate venv with pre-installed packages (`config/sandbox-requirements.txt`)
  plus the immutable `boxbot_sdk`
- Network access: allowed (skills need to call APIs)
- Filesystem: read project (read-only), write to `data/sandbox/` and `skills/`
- Cannot read `.env` or write to `src/boxbot/`
- Timeout: 30s default, memory limit: 256MB
- Package installation requires user approval (voice/button confirm)
- Display creation is declarative — agent uses building blocks, never writes
  raw render code

See [sandbox.md](sandbox.md) for the full design including the authoring
trust model (why skills auto-activate but displays need confirmation).

## Event-Driven Core

Subsystems communicate via an internal event bus (see `core/events.py`).
This decouples modules — the perception pipeline doesn't know about skills,
the display manager doesn't know about WhatsApp, etc.

Key events:
| Event | Source | Consumers |
|-------|--------|-----------|
| `motion_detected` | Perception | Perception (triggers YOLO check) |
| `person_detected` | Perception | Agent, Display, Scheduler |
| `person_identified` | Perception | Agent, Memory, Scheduler |
| `wake_word_heard` | Communication | Agent, Display, Perception |
| `conversation_started` | Agent | Display, Memory, Perception |
| `conversation_ended` | Agent | Memory, Scheduler, Perception |
| `button_pressed` | Hardware | Agent, Display |
| `whatsapp_message` | Communication | Agent |
| `trigger_fired` | Scheduler | Agent |
| `display_switch` | Agent/Manager | Display |

## Perception Pipeline

The perception pipeline is event-driven, not continuous. At rest it runs
minimal CPU-based motion detection. When someone appears, it takes a
snapshot for identification. During conversation, the Hailo NPU is free.

Voice teaches vision: speaker identification (pyannote diarization)
provides reliable labels that bootstrap visual ReID over time. Visual
embeddings are only added to a person's cloud when confirmed by voice
or agent context — visual-only matching can read clouds but never writes
to them.

Identities are injected directly into the agent's conversation input as
attributed text (`[Jacob]: ...`, `[Person B]: ...`). The agent never
sees raw embeddings. The `identify_person` tool lets the agent name or
identify people it learns about through conversation.

See [perception.md](perception.md) for the full design, state machine,
models, and data storage.

## Data Flow: Voice Conversation

```
1. Motion detected → YOLO confirms person → ReID tentative label
2. Person speaks → wake word ("BB") detected → session starts
3. VAD (Silero) detects speech → audio accumulates
4. Silence persists (800ms default) → utterance finalized
5. Parallel: pyannote diarizes + ElevenLabs Scribe transcribes
6. Attributed transcript: [Jacob]: "..." with context + memories
7. Agent decides: respond / stay quiet / wait for more
8. If responding: streaming TTS → speaker playback
   (barge-in monitored: ignore → fade → stop)
9. 30s silence → session suspended (mic off, context retained)
10. 2-3 min timeout → session ended:
    store voice-confirmed visual embeddings,
    extract memories, recompute centroids
```

See [voice-pipeline.md](voice-pipeline.md) for the full design
including turn detection, barge-in, session management, and
multi-speaker handling.

## Data Flow: WhatsApp Message

```
1. Webhook receives message ──► Whitelist check
2. If authorized: route to agent with sender identity
3. Agent processes ──► Response text
4. WhatsApp API ──► Send reply
5. Memory extraction
6. If message includes task: add to scheduler
```

## Data Flow: Trigger-Fired Wake

```
1. Scheduler monitors trigger conditions (time, person, compound)
2. All conditions AND-satisfied → emit trigger_fired event
3. Agent wakes with context:
   - Trigger: description + instructions
   - Present people (from perception)
   - To-do count + active trigger count
   - Relevant memories (standard injection)
4. Agent executes trigger instructions
5. Agent reviews to-do list for actionable items
6. Agent updates state:
   - Complete to-do items that are done
   - Create person triggers for deferred delivery
   - Cancel stale triggers
7. Update displays with fresh data
8. Idle timeout → sleep state
```

Recurring wake cycle triggers (config-seeded) follow the same flow.
The agent's wake cycle instructions tell it to check weather, review
the to-do list, update displays, etc. — all through the same
trigger-fired mechanism.

## Concurrency Model

boxBot uses Python asyncio as the main concurrency model:
- Hardware polling loops run as async tasks
- Perception inference is async (submitted to Hailo, results awaited)
- API calls (Claude, STT, TTS, WhatsApp) are async HTTP
- The event bus is async (publish/subscribe)
- SQLite access via aiosqlite

CPU-bound work (embedding similarity search, image preprocessing) runs
in a thread pool executor to avoid blocking the event loop.

## Security Boundaries

```
                    ┌─── TRUSTED ZONE ───┐
                    │                     │
                    │  Agent · Tools      │
                    │  Skills · Memory    │
                    │  Display · HAL      │
                    │  Perception         │
                    │                     │
                    └────────┬────────────┘
                             │
              ┌──── AUTH BOUNDARY ────┐
              │                       │
              │  Voice (physical)     │ ◄── Implicit trust (you're in the room)
              │  Buttons (physical)   │ ◄── Implicit trust
              │  WhatsApp (whitelist) │ ◄── Explicit user auth
              │                       │
              └───────────┬───────────┘
                          │
           ┌── CONTENT FIREWALL ──┐
           │                      │
           │  Web content         │ ◄── Small model filters, summarizes,
           │  (via web_search)    │     and strips injection attempts.
           │                      │     Plain text output only — no tools,
           │                      │     no SDK, no boxBot access
           └──────────┬───────────┘
                      │
           ┌── UNTRUSTED ──┐
           │  Everything   │
           │  else         │
           └───────────────┘
```
