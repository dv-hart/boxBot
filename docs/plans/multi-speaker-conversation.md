# Plan: Multi-speaker conversation handling (silent listening + structured response)

## Goal
Make BB feel like a polite third party, not a chatbot.

- Wake word starts a listening session
- Every turn, BB **decides** whether to respond (`respond: bool`), even when LLMs are always-going-to-emit-output
- In a 3-way conversation, BB listens while humans talk to each other, jumps in only when addressed or when it has a clear, time-sensitive action
- Diarization + person metadata flows continuously so BB attributes utterances correctly and knows who is speaking to whom

Example scenario — already in the user's head, captured here for reference:
1. Jacob: "BB, what's on the calendar tomorrow?" → BB responds
2. Jacob and Sarah discuss it among themselves for 60s → BB silent, still listening
3. Sarah: "BB, add a 3pm dentist" → BB responds with action

## Current state

- **Turn loop**: uses raw `anthropic.AsyncAnthropic().messages.create()` at `core/agent.py:719-725`. Not the Agent SDK. Single-turn; when `stop_reason != "tool_use"`, conversation ends (`agent.py:758-760`). No "silent" path.
- **Transcript attribution**: diarization + `[Name]: text` format already wired (`voice.py:_build_attributed_transcript` 616-668). All diarized utterances go through during ACTIVE — not just wake-word-triggered ones.
- **Active timeout**: 30s of silence suspends (`config.py:331`, loop at `voice.py:710-726`). `_last_speech_time` is reset on *any* finalized utterance (`voice.py:413`), so — critical — human-human chat *does* keep the session alive. Good news: the timer fix I previously assumed is needed turns out to be already correct.
- **Barge-in**: detected at `voice.py:510-520`. Session stays ACTIVE through a barge-in. Solid.
- **System prompt**: dynamic construction at `agent.py:65-94` + `_build_system_prompt` (~544-628). Says "You know when to speak up and when to stay quiet" but no explicit multi-speaker guidance or decision scaffolding.

## Proposed architecture

### 1. Structured decision output — the core change

Every agent turn produces a structured decision before any tool calls. Two viable paths:

**Option A (recommended): "decide_response" tool, always called first**

Register a tool like:
```json
{
  "name": "decide_response",
  "description": "ALWAYS call this first in every turn. Decide whether to speak.",
  "input_schema": {
    "type": "object",
    "properties": {
      "respond": {"type": "boolean"},
      "reason": {"type": "string", "description": "1 sentence: why respond or stay silent"},
      "addressed_to": {"enum": ["me", "other_person", "group", "ambiguous"]},
      "urgency": {"enum": ["immediate", "normal", "defer", "none"]}
    },
    "required": ["respond", "reason", "addressed_to"]
  }
}
```
Agent loop:
- If `respond=false`, terminate the turn after this tool returns. Do NOT call `speak()`. Record `reason` to a silent-log so we can inspect BB's restraint decisions.
- If `respond=true`, proceed to normal tool use / final text, which flows into `speak()`.
- Enforce "first tool must be decide_response" in the prompt; log an error if violated.

Advantages: no API changes, prompt-caching friendly, backwards-compatible, trivially testable (the decision is a discrete tool call, not buried in the text stream).

**Option B: JSON-schema'd `response_format`**

Newer API; requires a `response_format` parameter if we upgrade the Anthropic SDK version. Cleaner boundary but every turn's output becomes JSON, and tool-use mid-structured-output is awkward. Defer.

### 2. Transcript flow — no pipeline change needed

Every diarized utterance is already forwarded to the agent as a new turn. What changes is the *prompt* and the *response gate*, not the transport. Keep `voice.py` as-is.

One small enrichment: include a hint about recent wake-word status in the turn context. Either:
- Add `wake_word_mentioned: bool` to `TranscriptReady` — simple regex scan on the transcript for "BB", "hey BB", etc.
- Let the agent decide from the text alone. Given Opus 4.7, this is probably fine without a flag. Start flag-free; add if needed.

### 3. System prompt additions (the real behaviour change)

In `_build_system_prompt`, add a "conversation etiquette" section that's persona-appropriate (warm, not robotic):

```
## When to speak

You receive every utterance in the room, attributed by speaker name like
"[Jacob]: ..." and "[Sarah]: ...". Before every response, you must call
decide_response() first.

Respond when:
- You are directly addressed ("BB, ...", "Jarvis, ...", direct question to you)
- You have information someone needs RIGHT NOW (timer expired, incoming
  important message, factual correction to something just said)
- There's a clear action you should take given what was discussed (someone
  said "BB, book it" or "add that to my list")

Stay silent when:
- People are talking to each other, not to you
- The group is thinking or processing
- Something was said but it's not your place to weigh in
- You already answered and they're confirming among themselves

When silent, you are still listening and tracking context. You may be
addressed in the next turn. Treat silent turns as attentive listening, not
as ignoring.

When uncertain, prefer silence over interjection.
```

Add it once; cache it via prompt caching (see model-upgrade plan).

### 4. Persona-per-person nuance (later)

The CLAUDE.md / project memory notes that per-person interjection thresholds come from person memories. That's fine — the prompt above is the default; individual person memories can override ("Jacob prefers you to jump in with reminders quickly; Sarah prefers minimal interruption"). No new mechanism needed — this is just prompt text pulled from memory.

### 5. Session lifecycle — small tweak

Current suspend behavior: 30s of silence → SUSPEND → 3min → END. The fact that `_last_speech_time` updates on human-human chat means human-dense conversations stay ACTIVE for as long as people keep talking. This is good.

One gap: what if BB is silent (correctly) and then the room falls silent too? The 30s timer suspends. When the room wakes back up, a new wake word is needed. That's acceptable behavior — but worth verifying in practice that 30s is long enough given typical conversational pauses (turn-based discussion is usually <5s gaps; 30s is generous).

**Potential refinement**: extend the timer while there's been human speech in the last ~60s, even if the most recent 30s is quiet. Not a day-one change; consider after a week of real use.

## Files touched

- `src/boxbot/core/agent.py`
  - Register `decide_response` tool in the tool set
  - Enforce "first tool call" semantic in agent loop
  - Add prompt section (or refactor `_build_system_prompt` to compose a new "etiquette" block)
  - Log silent-turn reasons for later review
- `src/boxbot/tools/builtins/decide_response.py` (new, thin — just records the decision and returns an acknowledgement)
- `src/boxbot/tools/__init__.py` — register the new tool
- `src/boxbot/core/events.py` — optional: add `AgentDecidedSilence` event for downstream consumers (memory extraction wants to know BB was attentive-but-silent, for example)

## Open questions

1. **Enforcement**: if the agent forgets to call `decide_response` first, what happens? Options: (a) silently coerce to "respond=true" and log a warning, (b) reject the turn and re-prompt. Start with (a) to avoid cascading failures; metric on how often it happens.
2. **Should `decide_response` also be called mid-turn after a tool result?** No — only at turn boundaries. A turn that decided to respond proceeds to completion.
3. **Silent-log noise**: every non-response becomes a log line with the agent's reason. That's useful for tuning but spammy. Log to DB (silent_turns table) instead of stdout — query-friendly.
4. **Wake-word-plus-silence edge case**: someone says "BB ..." and then stops (misfire). BB hears "[Jacob]: BB" as a single utterance. `decide_response` will probably go `respond=true, addressed_to=me` and BB will say something like "Yes?" — which may be the right behavior, or may be annoying. Let real use decide.
5. **Model cost**: every turn costs an extra small tool call. With prompt caching on Opus 4.7, the marginal cost is cache-hits on the system prompt + a few output tokens for the tool call JSON. Probably ~$0.005/turn. Acceptable.
6. **Latency**: tool-call overhead adds ~200-500ms per turn. Tolerable since BB already has ~3-4s latency budget and the silence path is immediate (no speak required).

## Testing
- Two-voice simulation: script two utterances where one addresses BB, one doesn't. Assert BB responds only to the first.
- Three-party scripted conversation: 10 turns, 4 addressed-to-BB, 6 between humans. Assert BB responds to the 4 and stays silent on the 6.
- Edge: wake word without follow-up → assert BB says "Yes?" or similar acknowledgement, not silent.
- Edge: addressed-but-abstract ("BB, what do you think?") → assert BB responds rather than deflecting to silence.

Mock the SDK call with a deterministic fake that returns canned decide_response tool calls; no need to hit real Claude for unit tests.

## Effort
Half a day for the tool + prompt + wiring. Another half-day for tests and log analysis plumbing. Real validation needs a week of household use; plan on prompt tweaks at day-3 and day-7.
