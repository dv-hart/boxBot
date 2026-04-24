# Plan: Upgrade large model to Opus 4.7 (for vision + reasoning)

## Status quo
- Large model: `claude-sonnet-4-20250514` (Sonnet 4, May 2025) — `src/boxbot/core/config.py:435`
- Small model: `claude-haiku-4-5-20251001` — `src/boxbot/core/config.py:436`
- Env var overrides: `BOXBOT_MODEL_LARGE`, `BOXBOT_MODEL_SMALL` (`config.py:95,97`)
- Single client + single call site: `AsyncAnthropic` in `core/agent.py:177-179`; `messages.create(...)` in `_agent_loop` at `core/agent.py:719-725`
- No image content is sent to Claude today. Photo-tagging has a stub (`photos/intake.py:391-414`) with `# TODO: Call small model for description + tagging` (line 409)
- `identify_person` tool (`tools/builtins/identify_person.py`) uses perception embeddings only — does not invoke Claude for visual reasoning

## Why Opus 4.7
The core argument is that identity bootstrap (separate plan) will capture **crops of speakers** and send them to the agent. That upgrades the large-model role from "reason over text" to "reason over crops + text in a live conversation." Opus 4.7 has the strongest visual reasoning in the lineup and is a drop-in replacement for Sonnet 4 via env var. Reasoning/tool-orchestration improvements also help the multi-speaker "decide to speak" logic (separate plan) where a single bad call is more visible than a latency win.

Cost trade-off is real (~5-10x Sonnet). Mitigations:
- Small model stays Haiku for high-frequency tasks (intent classification, photo tagging, transcript filtering)
- Large-model invocations are per-conversation-turn, not per-utterance (agent already gates on wake word / addressed turns — especially after the multi-speaker plan lands)
- Prompt caching: the large system prompt + persona is stable across turns — cache it explicitly

## Change set

### Minimum (env-var only)
One line in the Pi's environment:
```
BOXBOT_MODEL_LARGE=claude-opus-4-7
```
Restart boxBot. Verify `BoxBotAgent started (model: claude-opus-4-7)` in logs.

### Recommended accompaniment
1. **Prompt caching** at the system prompt boundary. `agent.py:719-725` passes `system=...` — wrap it as a cache-control block: `system=[{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]`. Opus is expensive; caching the persona+tools prompt is essentially free and cuts per-turn cost dramatically when turns come in bursts (multi-speaker rooms).
2. **Config surfacing**: in `config/config.example.yaml`, bump the documented default and add a comment noting the cost profile. Don't change the hard-coded default in `config.py:435` until we're confident; env var is the escape hatch.
3. **Vision plumbing** (pairs with onboarding plan): the `content` field in `messages.create` accepts mixed text+image blocks. When the agent receives a captured crop via a tool result, surface it as an image block in the next user message, not as a base64 string in text. Image handling goes in `core/agent.py` near the tool-result formatting (inside `_agent_loop`).

## Files touched
- `.env` on Pi (or `config/config.yaml`) — env var override
- `config/config.example.yaml` — documentation
- `src/boxbot/core/agent.py` — add cache_control, later add image content handling

## Open questions
1. Do we want a hybrid? e.g., route "visual identification" turns to Opus but keep "pure conversation" on Sonnet. Adds routing complexity; recommend deferring until we have cost data from pure Opus.
2. Prompt cache TTL is 5 minutes — is that long enough for typical conversation cadence? Almost certainly yes; SDK will refresh automatically.
3. Opus 4.7 may handle tool-use differently around stop_reason edge cases. Run the existing conversation tests (`tests/test_communication.py`) post-upgrade.

## Effort
15 minutes (env var + restart + log verify). Caching is another 30 minutes. Real work is in the downstream plans (vision blocks belong with the onboarding flow).
