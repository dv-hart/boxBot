# Code Review Remediation Plan — June 2026

Tracking doc for the findings from the 2026-06-09 full-project review.
Work is split into independent workstreams sized for one subagent each.
Status values: `todo` / `in-progress` / `review` / `done`.

**Guiding principle (from the review):** boxBot is agent-led. A
capability only counts as done when (a) the agent has a tool or `bb`
SDK path to it, (b) the docs the agent reads describe it accurately,
and (c) failures surface as actionable errors the agent can act on.
Silent failure is the enemy — half these findings are features that
fail quietly enough that neither the agent nor the operator notices.

---

## WS1 — Revive the long-lived SandboxRunner

**Status:** done (2026-06-10) · **Size:** XS (hours) · **Depends on:** nothing

The persistent sandbox server has never run: the composed code
(`_SECCOMP_PROLOGUE + _SERVER_CODE`) fails to compile because
`src/boxbot/tools/_sandbox_server.py` has `from __future__ import
annotations` after its docstring, and the prologue is prepended above
it (`sandbox_runner.py:138`). Every `execute_script` call silently
falls back to per-call subprocess spawning.

**Tasks**
- [x] Remove the `__future__` import from `_sandbox_server.py` (quote
      any annotations that need it) or strip it during composition.
- [x] Add a unit test that `compile()`s the composed server code —
      this exact failure mode must never silently return.
- [x] Make the fallback loud: when runner startup fails, log at
      WARNING with the underlying error (it's currently invisible).
- [x] Integration test: two `execute_script` calls in one conversation
      reuse the warm runner (assert the runner PID is stable).

**Acceptance:** warm-runner path verified live on dev; compile test in CI.

---

## WS2 — Dream cycle correctness

**Status:** done (2026-06-10) · **Size:** S (half day) · **Depends on:** nothing

**Tasks**
- [x] **Watermark bug** (`memory/dream.py:1373-1413`): only advance
      `DREAM_WATERMARK_KEY` when batch submission succeeded *or* there
      were no pairs to submit. Today the `except` at :1383 swallows the
      failure and :1413 advances anyway — memories in that window are
      never dedup-scanned. The comment at :1410-12 already describes
      the correct behavior; make the code match it.
- [x] **Poller retry loop** (`memory/dream_poller.py` ~:225): when
      `apply_dream_result` throws, mark the row `failed` with the error
      message (mirror the timeout path at ~:179). Today it stays
      `submitted` and re-throws identically on every boot, forever.
- [x] **Stale comments** (`core/agent.py:983`, `:1080`): they claim
      audit-only is the default; `config.py:554` says `dream_audit_only
      = False` (apply-mode). Fix the comments to match reality.
- [x] **"operational" memory type** — resolve the three-way split:
      `sdk/_validators.py:49` accepts it, extraction rejects it,
      `docs/memory.md` still documents it. **Recommendation: drop it
      everywhere** (extraction already has; retrieval budget is already
      0). Remove from validators, docs/memory.md, and any remaining
      references; migrate or archive existing operational rows.

**Acceptance:** tests for watermark-on-failure and poller-failure
paths; grep for "operational" returns only changelog/history mentions.

---

## WS3 — SDK error-semantics standardization + packages flow

**Status:** done (2026-06-10) · **Size:** M (1-2 days) · **Depends on:** decision D2

**The rule (decided):** *mutating* SDK calls **raise** on
`status != ok` (with the dispatcher's message in the exception);
*read* calls return raw response dicts with documented shapes. This is
what `bb.memory.save()` already does — extend it everywhere.

**Tasks**
- [x] Centralize the raise-on-error helper in `sdk/_transport.py`
      (e.g. `dispatch_or_raise()`), migrate write paths in:
      `integrations` (create/update/delete/save), `secrets`
      (store/delete), `workspace` writes, `packages`, `auth` writes.
      Read paths keep raw returns; document the response shape in each
      docstring.
- [x] **Implement the missing `packages` handler.** There is no
      `@action_handler("packages")` in `tools/_sandbox_actions.py`, so
      `bb.packages.request()` gets "unknown action" and silently wraps
      it as `PackageResult(approved=False)`. Wire the full documented
      flow: sandbox emits request → main process queues it → out-of-band
      approval (screen tap and/or admin WhatsApp YES per D2) → pip
      install into sandbox venv by the main process → result reported.
      Interim milestone if approval UX is deferred: handler exists,
      request is durably queued + admin notified, and the SDK returns
      `pending` honestly instead of fake `denied`.
- [x] Fix `packages.request()` to raise on transport/dispatch errors —
      "denied by human" and "system broke" must be distinguishable.
- [x] **Write the three missing module docs**:
      `skills/bb/modules/tasks.md`, `auth.md`, `packages.md`. The
      agent can call these today but can't load reference docs.
- [x] Update existing module docs (`integrations.md`, `secrets.md`,
      `workspace.md`, `memory.md`) to state the raise-vs-raw rule.

**Acceptance:** one documented error model across all `bb` modules;
`bb.packages.request()` end-to-end test (mock approval); 12 module
docs covering all 12 SDK modules.

---

## WS4 — Implement `memory_query` display source

**Status:** done (2026-06-10) · **Size:** S (half day) · **Depends on:** nothing

**Purpose** (already documented in `docs/display-system.md:449`,
`skills/bb/modules/display.md:369`, `displays/README.md:74` — the code
just never caught up): a display data source that re-runs a memory
search on every refresh, so a display can surface live memory content
without the agent rebuilding it. Canonical examples: a standing
"household reminders" panel (`query="household reminders and standing
instructions", refresh=300`), a project status board
(`query="kitchen renovation"`). It's the bridge that lets memory
drive the screen.

**Tasks**
- [x] Implement `MemoryQuerySource.fetch()`
      (`displays/data_sources.py:396-411`) against the shared backend
      in `memory/search.py` (lookup mode).
- [x] **Skip small-model reranking on display refresh** (hybrid
      vector+BM25 score only) — a 5-minute refresh loop must not make
      Haiku calls. Confirm with cost logging.
- [x] Display-ready output shape, binding-friendly:
      `{results: [{text, type, age}], count, query}` — document in
      `skills/bb/modules/display.md` (replace the current example if
      the shape differs).
- [x] Graceful empty/error behavior consistent with other sources
      (cached data on failure, `_fetch_error` set).
- [x] Tests: query → results shape; empty query; backend error.

**Acceptance:** a real display bound to `{reminders.results}` renders
actual memories on dev; no model-call cost per refresh.

---

## WS5 — Presence & identity (perception)

**Status:** done (2026-06-10) · **Size:** M-L (2-3 days) · **Depends on:** decision D3

Design intent (confirmed 2026-06-09): **boxBot should always know who
is talking and, where possible, who is present.** Visual ID was
non-functional until the 2026-06-07 overhaul; now that it works, the
presence surface the docs promised can become real.

**Tasks**
- [x] **`[Present: …]` header** — documented in `perception.md`
      (~:388) but generated nowhere. Implement: at conversation start
      (and on mid-conversation arrivals/departures), build a presence
      block from the perception pipeline's `active_persons` +
      confidence (`confirmed` / `likely` / `new person`), inject into
      the conversation thread. Voice conversations first; trigger
      conversations second; WhatsApp N/A.
- [x] **Agent-reachable rename.** Today "actually, call me Jake"
      via `identify_person` RENAME creates a *new* person instead of
      renaming. Add a true rename path (per D3: extend
      `identify_person` or new `bb.people` SDK module).
- [x] **Agent-reachable merge.** Reconcile flags duplicates
      (Eric/Erik) audit-only, but nothing consumes the flag. Add:
      (a) a way for the agent to *read* the reconcile report
      (duplicate_persons list with evidence), and (b) a merge
      operation (move embeddings + memories person-refs from loser to
      winner, soft-keep the loser row with `merged_into`). Merge is
      destructive — require the agent to confirm with the humans
      involved before calling it; say so in the tool description.
- [x] **Acceptance test = resolve Eric/Erik** on the live device via
      the agent path, not manual DB surgery.
- [x] Delete the vestigial `confirm_session_embeddings` post-
      conversation path (`perception/pipeline.py:713-754` +
      `fusion.py:234-300`) — `IdentityFusion` has no
      `session_speakers`, so it always runs on empty data;
      `EnrollmentManager.commit_session()` is the real path.
- [x] Update `docs/perception.md` to match what ships.

**Acceptance:** presence header visible in conversation logs; Eric/Erik
merged by the agent; vestigial path gone; docs accurate.

---

## WS6 — Integrations hardening

**Status:** done (2026-06-10) · **Size:** S-M (1 day) · **Depends on:** nothing

**Tasks**
- [x] **TOCTOU crash** (`integrations/persist.py:84-95`): concurrent
      creates of the same name race past `exists()`; loser gets an
      unhandled `FileExistsError`. Catch it and return the documented
      `{"status": "exists"}`.
- [x] **Log redaction**: integration *inputs* are logged verbatim —
      an agent passing an API key as an input (instead of via
      `bb.secrets`) leaks it into logs. Redact values for keys
      matching `token|key|secret|password|auth`.
- [x] **Smarter truncation**: 32 KB head-only truncation can cut the
      actual error off the end. Keep head + tail (errors usually end
      with the stack trace).
- [x] **Surface display-source failures**: `IntegrationSource` logs
      failures at `debug` and returns `{}` — displays silently render
      placeholders. Raise to WARNING and expose staleness so a display
      can show "data unavailable" (renderer already has stale checks).
- [x] **Doc updates** (`skills/bb/modules/integrations.md`): the
      300-second timeout ceiling and its rationale; concurrent runs of
      one integration are NOT serialized (matters for token-refresh
      scripts like calendar); `return_output()` last-call-wins — early
      error returns must be followed by `sys.exit(0)`; setup notes for
      the three built-ins (calendar `GOOGLE_CALENDAR_TOKEN_JSON`,
      home_assistant URL/token, weather).

**Acceptance:** concurrent-create test; redaction test; a display with
a failing integration shows degraded state instead of silent
placeholders.

---

## WS7 — Documentation drift batch

**Status:** done (2026-06-10) · **Size:** XS (hours) · **Depends on:** WS3, WS5
(land last so it documents final state)

- [x] CLAUDE.md: tool count 7 → **10**; add `message`, `mute_mic`,
      `search_photos` to the list (decision D1: keep all 10).
- [x] `tools/builtins/load_skill.py:8-15`: docstring claims the tool
      is "intentionally not registered" — it is registered
      (`registry.py:61`). Fix.
- [x] `docs/memory.md`: remove operational type (after WS2).
- [x] `docs/perception.md`: align with WS5 outcome.
- [x] MEMORY.md / project memory: mark findings fixed as workstreams
      land (`code_review_2026_06_09.md`).
- [x] Sweep: re-run a doc-vs-code spot check on `skills/bb/modules/*`
      after WS3/WS4 land.

---

## Open decisions (all resolved 2026-06-10)

- **D1 — RESOLVED: keep all 10.** keep all 10 registered tools and fix CLAUDE.md?
  *(Recommended: yes — `message`, `mute_mic`, `search_photos` are all
  real and in use.)*
- **D2 — RESOLVED: admin messaging reply (channel-agnostic; household uses Signal).** which out-of-band path ships first —
  screen tap, admin WhatsApp YES, or both? Determines WS3 scope.
- **D3 — RESOLVED: extended identify_person with an action param.** extend `identify_person` with an
  `action` parameter (`identify` / `rename` / `merge`) vs. a new
  `bb.people` SDK module. *(Recommended: extend `identify_person` —
  identity is hot-path and security-sensitive, which is the stated
  bar for standalone tools; keeps one tool as the single identity
  gateway.)*
- **D4 — RESOLVED: dropped (was already lifecycle step 7; validator+docs were drift).** drop entirely *(recommended)* or
  re-add to extraction?

## Suggested execution order

1. **WS1** (hours, unblocks warm-runner perf for everything else)
2. **WS2** (silent data loss, clear fixes)
3. **WS4** ∥ **WS6** (independent, parallelizable subagents)
4. **WS3** (needs D2)
5. **WS5** (needs D3; largest design surface)
6. **WS7** (documents the settled state)

Each workstream is sized for one subagent with this doc section as its
brief. WS3 and WS5 should get a design pass before implementation.
