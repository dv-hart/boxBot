# Prefetch: context-aware injection + cache-stable turns

Findings from the 2026-08-31 16:44Z multi-device voice session
(conv_a812808e6903, 5 turns):

- `onboarding` skill body (≈1.6k tok) injected 4× into one thread —
  fan-out lanes never see `already_loaded`; only the hot path dedups.
- Skills lane cap `_MAX_SKILLS=1` let onboarding crowd out
  tier0-support on every turn (both matched: speaker unresolved →
  onboarding; panel task → tier0-support).
- Thermostat hallucination (`bb.panel.thermostat.set(...)`): the SDK
  selector chooses on **H2 headings only** — `## Lifecycle` (the
  signatures section) carries no signal, so signatures were never in
  context; and the lock turn's bundle tracking overclaimed the whole
  module (empty `sdk_sections` → whole-module fallback key), so dedup
  gutted the thermostat turn's own bundle.
- Fan-out ran on mid-conversation follow-ups ("Stay.") — blocked the
  reply path (gen start +3.2s/+3.7s) to re-select content already in
  thread.
- Every turn-1 Luna call has cache_read=0: per-turn dynamics (clock,
  presence, bundle) render into the SYSTEM message, busting the prefix
  from token 0. Within-turn calls cache fine (~95%+).

## 1. Dedup at injection, every path

- `_prefetch_context_for_text`: run `_drop_already_loaded` on fan-out
  bundles too (hot path already does, inside `lookup`). Also filter
  `bundle.memories` against already-injected memory ids.
- Feed context to the selectors: skills lane candidates exclude
  already-loaded skills; sdk lane menu omits already-loaded section
  keys (and every section of a fully-loaded module). The instructions
  already tell the model to skip them — now it actually knows them.
- `_MAX_SKILLS` 1 → 2 (crowding-out fix; dedup makes repeats free).

## 2. Honest tracking granularity

- `_track_prefetch_injected`: record the whole-module key ONLY when
  the whole module was injected (`sdk_sections[m] == [m]`). A splice
  with empty/missing `sdk_sections` (legacy cache format) records
  nothing — never a claim broader than what entered context.
- `_drop_already_loaded` whole-module check then stays valid as-is.

## 3. Heading pass over `skills/bb/modules/*.md`

The SDK selector menu is `- <key>: <H2 heading>` — headings ARE the
selection signal. Every H2 must state what the section contains
(functions, verbs, decision it settles). Style: longer, concrete,
no "(read this)"/"API"/"Patterns"/"Lifecycle":

- panel: `## Lifecycle` → `## Call signatures — state/devices/lock/
  unlock/light/thermostat/arm/disarm`
- `## Auth — disarm and unlock (read this)` → `## Disarm & unlock
  auth — code collected over Signal, no credential in scripts`
- Same treatment across audio/auth/camera/display/integrations/
  memory/packages/photos/secrets/skill/tasks/workspace.

Content unchanged unless a heading demands a tiny reflow. Doc mtimes
change → content fingerprint → hot bundles auto-rebuild on deploy.

## 4. Cache-stable turns (append, don't bust)

Move ALL per-turn dynamics out of the system message so the prefix
(static prompt + tools + prior thread) is byte-stable across turns:

- `_prompt_dynamic_context` output (clock, presence, channel, to-do/
  trigger counts) and the rendered prefetch bundle ride as text blocks
  in the LATEST user message instead.
- Both loops (`_agent_loop_openai` + anthropic). Nothing earlier in
  the message list may mutate (probe-verified: any prefix change
  → 0% cached).
- Expected: turn-1 cache_read jumps from 0 to ~90% of input on every
  follow-up turn (~$0.007/session on the measured 5-turn session);
  extraction's prefix reuse also improves (rides the same thread).
- Verify by replaying the exact 16:44 test sequence: lock →
  thermostat read → thermostat lower → arm → stay; compare cost rows.

## 5. Voice follow-ups: hot-only, no fan-out

First voice turn of a conversation: unchanged (draft-warmed hot
lookup, fan-out on miss). Follow-up turns (conversation has ≥1 prior
assistant turn):

- Hot lookup still runs — local ONNX embed ~150ms, no model calls;
  carries live device state + fresh memory pick (the zero-tool-turn
  input for "…now lock the door" after chat).
- On hot MISS: inject nothing. No selector fan-out mid-conversation.
  Fallback = the agent's own search_memory / load_skill.
- Voice channel only; text channels keep current behavior (async
  cadence, latency invisible).

## Out of scope (tracked separately)

- Voice ReID intermittency (session ran as [Speaker A]; the reason
  onboarding matched at all).
- $6.80 phantom STT connection-seconds row (stale-timestamp
  accounting bug).
- TTS stall retry.

## Order & verification

1+2+5 together (prefetch behavior; unit tests on dedup/tracking/skip).
3 next (docs only; verify hot rebuild + a thermostat command lands
`panel.thermostat(...)` first try). 4 last, alone (highest blast
radius; verify cross-turn cache_read > 0 and no behavior drift before
piling on). Deploy after each stage; replay the 5-turn session at the
end and compare against: lock 6.1s/$0.0025, thermostat-lower
12.2s/$0.0048 (5 calls), stay 7.4s/$0.0052, session Luna ≈$0.021.
