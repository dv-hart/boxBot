# Implementation spec — 2026-04-23 boxBot work

**Authoritative for subagent execution. Read in full before writing code.** This spec captures architecture decisions made after live research of the Anthropic docs. It contradicts common priors — read it carefully.

## Non-negotiable architecture constraints

### 1. boxBot uses the `anthropic` Python SDK directly. DO NOT migrate to `claude-agent-sdk` or `claude-code-sdk`.

- Current call path: `anthropic.AsyncAnthropic().messages.create(...)` in `src/boxbot/core/agent.py:719-725`.
- `claude-code-sdk` appears in `pyproject.toml` but is **not imported anywhere**. Treat it as dead weight; do not adopt it.
- The Claude Agent SDK is literally "Claude Code as a library" — bundles the Claude Code binary, built-in tools are Read/Write/Edit/Bash/Glob/Grep. It is opinionated toward filesystem/coding agents. boxBot is a hardware-facing ambient assistant with bespoke tools (`speak`, `switch_display`, `identify_person`, etc.). Wrong abstraction. Cherry-pick ideas if useful, but do not import `claude_agent_sdk`.

### 2. DO NOT use Anthropic's `/v1/skills` API. Build a custom filesystem skill loader.

- Anthropic's API-level Skills are tightly bound to Anthropic's code execution container (`code-execution-2025-08-25` + `skills-2025-10-02` + `files-api-2025-04-14` betas). The container has no network, no arbitrary package install, no hardware access. Useless for boxBot's local hardware tools.
- We build our own loader: see §5. Filesystem-based, injected into system prompt, with a `load_skill` tool for progressive disclosure.

### 3. Opus 4.7 breaking-change rules

The current model is `claude-opus-4-7`. Any of these in a `messages.create()` call will **400**:

- `temperature=...`  — removed
- `top_p=...`  — removed
- `top_k=...`  — removed
- `thinking={"type": "enabled", "budget_tokens": N}`  — removed

Use instead:
- No sampling params — prompting is the only control lever on 4.7
- `thinking` is **off by default**; set `thinking={"type": "adaptive"}` ONLY if you explicitly want thinking. For boxBot's conversational use case, leave thinking OFF (latency matters).
- `output_config={"effort": "high"}` is the default; don't set unless you have reason to.
- `max_tokens`: bump from 4096 → **8192** for the work in this spec. Opus 4.7's token counting shifted vs Sonnet 4; give headroom.

Today's code at `agent.py:718-725` doesn't pass any banned params. Keep it that way.

---

## 4. Structured outputs for decide_response — the NEW gate

**Use native `output_config.format` with JSON schema.** Not a tool. GA on Opus 4.7, no beta header.

### The schema (pin once per session — mutation invalidates the messages cache)

```python
DECIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "respond": {
            "type": "boolean",
            "description": "True if BB should speak this turn; False for silent listening.",
        },
        "response_text": {
            "type": ["string", "null"],
            "description": "If respond is true, the exact words BB says. If false, null.",
        },
        "reason": {
            "type": "string",
            "description": "One sentence: why respond or stay silent.",
        },
        "addressed_to": {
            "enum": ["me", "other_person", "group", "ambiguous"],
            "description": "Who the most recent utterance was directed at.",
        },
        "urgency": {
            "enum": ["immediate", "normal", "defer", "none"],
            "description": "How time-sensitive any action is.",
        },
        "silent_context_note": {
            "type": ["string", "null"],
            "description": "If respond=false, optional private note capturing something worth remembering. <=1 sentence. Null if nothing to note.",
        },
    },
    "required": ["respond", "reason", "addressed_to", "urgency"],
    "additionalProperties": False,
}
```

### Call shape

```python
response = await self._client.messages.create(
    model=config.models.large,
    max_tokens=8192,
    system=system_prompt_blocks,      # list of text blocks; last has cache_control 1h
    messages=messages,
    tools=tool_definitions,           # list; last tool has cache_control 1h
    output_config={"format": {"type": "json_schema", "schema": DECIDE_SCHEMA}},
    cache_control={"type": "ephemeral"},  # top-level auto-cache of last messages block, 5m
)
```

### How this interacts with tool use

The schema constrains **text content only**. Tool-use blocks are separate and not affected. During tool-use turns the model typically emits tool_use without text — no issue. When the model produces final text (end_turn), that text conforms to the schema.

**If the model DOES emit text during a tool-use turn**, the text must be valid per schema. In practice, on tool_use turns the model emits a minimal valid object (e.g. `{"respond": false, "reason": "running tools", "addressed_to": "me", "urgency": "none"}`) or no text. Handle both gracefully: only act on the decision when `stop_reason == "end_turn"`.

### Parsing

- Final text block on `stop_reason == "end_turn"` is JSON. `json.loads()` it.
- Handle `stop_reason == "refusal"` (schema may not match — fall back to a canned apology response, log the event).
- Handle `stop_reason == "max_tokens"` (truncation — bump max_tokens or break the response).
- If JSON parse fails (should not happen under constrained decoding, but defensive), log ERROR and route the raw text to TTS as a fallback.

### The `speak` tool goes away as a primary speech mechanism

`src/boxbot/tools/builtins/speak.py` currently lets the agent speak via tool call. Under the new model, speech flows through `response_text` in the decision. The `speak` tool should be **removed from the always-loaded tool list** (`get_tools()`). Keep the file for now in case we want mid-turn explicit speech later; just don't register it.

---

## 5. Prompt caching strategy — Opus 4.7

### The rules that matter

- **4096 tokens minimum** cacheable prefix on Opus 4.7. Below that, `cache_control` silently no-ops.
- **Max 4 breakpoints per request.**
- Render order: `tools` → `system` → `messages`. A breakpoint captures everything before it in the render order.
- **20-block lookback window** in message history — past 20 blocks, a new breakpoint won't find old caches.
- Changing `output_config.format` schema invalidates the messages cache. Pin the schema once per session.

### Breakpoint placement (3 of 4 used)

1. **Tools list** — put `cache_control={"type": "ephemeral", "ttl": "1h"}` on the LAST tool definition in the `tools` list. Tools + system above it get cached together with 1h TTL. Tools rarely change across turns.
2. **System prompt** — the system must be a **list** of text blocks, not a single string. Put `cache_control={"type": "ephemeral", "ttl": "1h"}` on the LAST system block. System prompt must be ≥4K tokens for this to engage.
3. **Messages rolling cache** — top-level `cache_control={"type": "ephemeral"}` on the request itself (5m default TTL). Auto-caches the last cacheable message block.

Reserve 1 breakpoint spare for 20-block-lookback refreshes in long sessions (future work).

### System-prompt block structure

The base system prompt (static persona, etiquette, skills index, capabilities) should be one big block with the 1h cache marker. Dynamic per-conversation context (who is present, memories, status line) should be a SEPARATE, later block WITHOUT a 1h marker so it can vary without killing the static cache.

```python
system = [
    {
        "type": "text",
        "text": _render_static_system_prompt(skill_index),  # persona + etiquette + skills + capabilities
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    },
    {
        "type": "text",
        "text": _render_dynamic_context(conversation_context),  # who's present, memories, time, etc.
        # NO cache_control — this varies per conversation
    },
]
```

### Verification

After a second turn with same prefix, check `response.usage.cache_read_input_tokens > 0`. If zero, audit for silent invalidators: changing schema, changing tool order, timestamps/UUIDs in the static prompt, non-deterministic JSON serialization.

---

## 6. System prompt structure (the static part)

Organize into named sections so the skill loader and other work can plug in without conflicts. Subagents implementing the refactor must create helpers like `_prompt_persona()`, `_prompt_etiquette()`, `_prompt_skills_index()`, `_prompt_capabilities()`, `_prompt_identity()` — each returning a string — and compose them in order.

### Required sections (static, cached 1h)

- **Persona** — name, wake word, core identity (existing `_BASE_SYSTEM_PROMPT` content, reworded for clarity).
- **Etiquette** (new, for multi-speaker) — see §7.
- **Skills index** (new) — auto-injected from the skill loader. Placeholder if loader isn't available. See §9.
- **Capabilities** — tool overview + SDK reference.

### Dynamic section (not cached, separate block)

- Who is present (from perception)
- Current time
- Schedule status line
- Injected memories

### Dummy placeholder for skill loader

If `boxbot.skills.loader` import fails, `_prompt_skills_index()` returns empty string. No crash.

---

## 7. Etiquette section — multi-speaker prompt content

Inject this into the static system prompt. Tune later based on observation; this is v1.

```
## When to speak

You hear every utterance in the room. Diarization labels each speaker: for
example, "[Jacob]: ..." or "[Unknown_1]: ...".

Every turn you produce a structured decision: respond (true/false),
response_text (what to say if responding), reason, addressed_to, urgency,
and silent_context_note (private observation if staying silent).

Respond when:
- You are directly addressed ("BB ...", "Jarvis ...", a direct question)
- You have urgent, useful information (timer expired, important message, factual correction)
- There is a clear action you should take from what was said ("add that to my list")

Stay silent when:
- People are talking to each other, not to you
- They are thinking out loud or processing
- You already answered and they are confirming among themselves

When silent, you are still listening and tracking context. The
silent_context_note field is for jotting down anything worth later recall
(e.g. "Sarah mentioned a dentist appointment for 3pm Thursday").

When uncertain, prefer silence. When addressed by name, always respond.
Never expose internal labels like SPEAKER_00 or Unknown_1 to the user.
```

---

## 8. Silent-turn logging

When `respond == false`, persist to a new SQLite table. Purpose: feed post-conversation memory extraction with ambient observations.

### Schema

New table in `data/memory/memory.db` (or new file — implementer's call; reuse memory.db for locality):

```sql
CREATE TABLE IF NOT EXISTS silent_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    reason TEXT NOT NULL,
    addressed_to TEXT NOT NULL,
    urgency TEXT NOT NULL,
    silent_context_note TEXT,
    transcript_snippet TEXT,
    speakers_present TEXT
);
CREATE INDEX IF NOT EXISTS idx_silent_turns_conv ON silent_turns(conversation_id);
```

### Integration

New module `src/boxbot/core/silent_log.py` exporting:
- `log_silent_turn(conversation_id, decision_dict, transcript_snippet, speakers)`
- `get_silent_turns(conversation_id) -> list[dict]` (for extraction)

Integration: in `_run_conversation` where the decision is parsed, if `respond == false`, call `log_silent_turn`.

---

## 9. Custom skill loader (replaces "just use Skills API")

### Why custom

Anthropic's `/v1/skills` requires the code execution sandbox and is useless for boxBot's local hardware tools. We build our own, simpler, filesystem-based system.

### Directory layout

```
skills/                         # repo-level, like displays/
  hal-audio/
    SKILL.md
    examples/play-tone.py
  hal-camera/
    SKILL.md
  calendar-how-to/
    SKILL.md
```

Each skill is a folder. `SKILL.md` has YAML frontmatter + markdown body.

### SKILL.md format

```markdown
---
name: hal-audio
description: How to play audio or capture microphone data from sandbox scripts.
when_to_use: Scripts that need to play sounds, capture raw audio, or control the LED ring programmatically.
---

# HAL: Audio

Body in markdown. Full reference, examples, code.

See also: `examples/play-tone.py` (subpath).
```

### Loader module

New: `src/boxbot/skills/loader.py`

```python
@dataclass(frozen=True)
class SkillMeta:
    name: str
    description: str
    when_to_use: str
    root_path: Path

def discover_skills(root: Path = Path("skills")) -> list[SkillMeta]:
    """Scan skills/ at startup. Read frontmatter only."""

def get_skill_index() -> str:
    """Render the one-line-per-skill index as system prompt text.
    If no skills or loader fails, returns empty string (safe)."""

def load_skill(name: str, subpath: str | None = None) -> str:
    """Read full SKILL.md body (frontmatter stripped) or a specific sub-file.
    subpath is relative to the skill directory."""
```

### Index format (injected into system prompt)

```
## Available skills

You have on-demand access to skills — topic-specific guidance loaded via the
load_skill tool. Read the full body only when the task calls for it.

- hal-audio: Scripts that need to play sounds, capture raw audio, or control the LED ring programmatically.
- hal-camera: Scripts that need to capture images or inspect the camera state.
- calendar-how-to: Formatting and querying the calendar via the calendar SDK.

To read a full skill, call load_skill(name="hal-audio").
To read a sub-file of a skill, call load_skill(name="hal-audio", subpath="examples/play-tone.py").
```

### New tool: `load_skill`

Create `src/boxbot/tools/builtins/load_skill.py`:

```python
{
    "name": "load_skill",
    "description": "Load a specific skill's body or sub-file for on-demand guidance. Call when a skill's index description matches the current task.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Skill name from the index."},
            "subpath": {
                "type": "string",
                "description": "Optional relative path to a sub-file within the skill directory.",
            },
        },
        "required": ["name"],
    },
}
```

Add to the tool registry. With `load_skill` + removal of `speak`, the tool count stays at 9.

### Seed skill

Implementer seeds **one** skill as demonstration: `skills/hal-sandbox-ref/SKILL.md` — describes how to use `boxbot_sdk` within sandboxed scripts (display builder, memory, secrets, photos, tasks). Content can be condensed from `src/boxbot/sdk/README.md`.

### Safety

- `load_skill` must validate `subpath` doesn't escape the skill directory (no `..`, absolute paths).
- Skills are READ-ONLY from the agent's perspective. No write API.

---

## 10. Image content in tool results (future onboarding prep — stub only)

Subagents DO NOT need to build this for this round, but they should structure code to allow it. In the future, `capture_speaker_reference` will return an image. The agent.py tool-result handling should support image content blocks alongside text:

```python
{
    "type": "tool_result",
    "tool_use_id": tu.id,
    "content": [
        {"type": "text", "text": "Captured speaker bbox (detection confidence 0.91)"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
    ],
}
```

Adding an image invalidates the messages cache from that point; system+tools 1h caches survive. This is acceptable for rare onboarding captures.

Current `_process_tool_calls` in `agent.py` wraps results as strings. Leave room in the API design so future additions don't require ripping it up: return `list | str` from the tool, let the formatter handle both.

---

## 11. File-by-file summary

### `src/boxbot/core/agent.py`
- Add `DECIDE_SCHEMA` constant
- Refactor `_agent_loop` to pass `output_config.format` + top-level `cache_control`
- Refactor system prompt construction into named section helpers
- Replace base persona with an updated version that includes etiquette section (see §7)
- Pull skill index via `get_skill_index()` with try/except import guard
- Parse end_turn response as decision JSON; route `response_text` to TTS when `respond=true`; call `log_silent_turn` when false
- Bump `max_tokens` to 8192
- Remove `speak` from registered tools (don't import the tool's registration in `tools/__init__.py` registry wiring)
- Handle `stop_reason`: `end_turn` → parse; `tool_use` → continue loop; `refusal` → canned apology; `max_tokens` → log + break; `stop_sequence` (N/A) → log

### `src/boxbot/core/silent_log.py` (new)
- Table creation on import
- `log_silent_turn(conversation_id, decision, transcript_snippet, speakers)`
- `get_silent_turns(conversation_id)`

### `src/boxbot/hardware/microphone.py`
- Replace `_set_leds_raw` with Seeed pixel_ring command protocol (cmd_id in wValue, wIndex=0x1C)
- Replace `get_doa` with parameter-ID encoding (`cmd = 0x80 | 0x40 | 21`, parse int32 LE)
- Promote USB write failures to WARNING (first occurrence per session), throttled thereafter
- Add new constants: `_PIXEL_RING_IFACE = 0x1C`, command IDs (TRACE=0, MONO=1, LISTEN=2, SPEAK=3, THINK=4, SPIN=5, SHOW=6, SET_BRIGHTNESS=0x20), `_PARAM_DOAANGLE_ID = 21`
- Keep existing `_PATTERN_CONFIG`; the renderers (`_render_pulse`, `_render_chase`) still compute per-LED colors; internally map those colors to `CMD_SHOW` with a 48-byte payload (12 LEDs × 4 bytes RGBA)

### `src/boxbot/skills/loader.py` (new)
- `SkillMeta` dataclass
- `discover_skills(root)`
- `get_skill_index()`
- `load_skill(name, subpath=None)` — with path-escape guard

### `src/boxbot/skills/__init__.py` (new or updated)
- Export loader functions

### `src/boxbot/tools/builtins/load_skill.py` (new)
- Wrapper around `loader.load_skill` returning text

### `src/boxbot/tools/registry.py` or `tools/__init__.py`
- Register `load_skill`
- Unregister `speak`

### `skills/hal-sandbox-ref/SKILL.md` (new — seed)
- YAML frontmatter + condensed SDK reference

### `scripts/setup.sh`
- Append block that installs `/etc/udev/rules.d/60-respeaker.rules` (idempotent — check if file exists first) and reloads udev

---

## 12. Testing expectations

Each subagent must:
- Run `python3 -c "from boxbot.core.agent import BoxBotAgent"` (or equivalent import path) without error
- Existing tests in `tests/` should still pass (run `pytest tests/ -x` from repo root)
- For new modules, add at least one smoke test file under `tests/`

If a test would require live hardware or network (camera, USB, Anthropic API), skip it or use mocks.

---

## 13. Common pitfalls to avoid

- Don't import `claude_agent_sdk` or `claude_code_sdk`.
- Don't use `output_format=` — that's deprecated. Use `output_config.format`.
- Don't pass `temperature`, `top_p`, `top_k`, or `budget_tokens` — 4.7 400s.
- Don't put timestamps, UUIDs, or `datetime.now()` in the STATIC system prompt block (you will invalidate the 1h cache). Put them in the dynamic block.
- Don't re-create the schema dict on every call — that's a new object reference but not a caching issue; still, define once at module scope.
- Don't forget `additionalProperties: false` on every object in the schema — the API enforces it.
- Don't hand-edit generated JSON from the model — use `json.loads()`.
- Don't assume `response.content[-1].text` is always the final text — iterate blocks and find the last `type == "text"` block.
- Don't block the event loop with sync sqlite in silent_log — use `aiosqlite` (already a project dep).
