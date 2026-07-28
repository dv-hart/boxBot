# tools/

The always-loaded tools registered with the Claude agent. Ten of them,
kept slim on purpose — every tool costs context on every turn.

Parameter schemas live in the tool classes. Read the code, not a copy
of it: each `builtins/<name>.py` carries `name`, `description`, and
`parameters` as class attributes, plus a module docstring for anything
non-obvious. This README is the layer map and the invariants.

## Three layers

| Layer | Loaded | Cost | For |
|-------|--------|------|-----|
| **Tools** (`tools/`) | Always | Fixed, slim | Hot-path or security-sensitive: messaging, display switching, identity, tasks, memory/photo lookup |
| **SDK** (`sdk/`, via `execute_script`) | On use | Zero when unused | Complex, infrequent, composable: workspace, camera, displays, integrations, skills, packages |
| **Skills** (`skills/`) | Per-conversation | Only what's relevant | Domain guidance, user-extensible |

The rule of thumb: a dedicated tool has to earn its permanent context
cost with frequency or with a security property. Everything else goes
through `execute_script` + `bb`.

## The invariant: message is the only human path

The model's text output is constrained to a private internal-notes JSON
shape (`output_dispatcher.INTERNAL_NOTES_SCHEMA`). By construction,
nothing in the model's text reaches a person. To speak or text, the
model must call `message(to, channel, content)`.

Every other tool *does* something — creates a reminder, searches
memory, switches a display, runs a script. None of them speak. No
`message` call means silence.

Multiple `message` calls per turn are expected: an interim
acknowledgement alongside `execute_script`, multi-recipient deliveries,
voice to the room plus text to someone absent.

## The ten

| Tool | Why it's a tool, not the SDK |
|------|------------------------------|
| `message` | The only human path. Security-critical, every turn. |
| `execute_script` | The gateway to everything else. |
| `switch_display` | Single action, mid-conversation. "Show me the weather" shouldn't need a script. |
| `identify_person` | Identity gateway. The agent supplies semantic labels; the backend owns all embedding bookkeeping. Merge is destructive. |
| `manage_tasks` | Every-turn frequency. Compound trigger creation must be one atomic call. |
| `search_memory` | Every-turn recall. Shares `memory/search.py` with injection and `bb.memory`. |
| `search_photos` | High-frequency conversational lookup. Shares `photos/search.py` with `bb.photos`. |
| `mute_mic` | Real-time voice control. Must land on the same turn as the decision. |
| `web_search` | Untrusted input needs the small-model firewall (below). |
| `load_skill` | Progressive disclosure entry point. |

`speak.py` and `send_message.py` remain on disk for reference. They are
not imported and not registered — `message` replaced both.

## identify_person is not a query tool

Who is present is injected into the conversation as attributed text
(`[Jacob]: …`, `[Person B]: …`) plus a `[Present: …]` header. The agent
never asks "who is here?" — it already knows. `identify_person` is for
*writing* identity: first meetings, corrections, renames, merges.

## web_search is a content firewall

Web content is the only channel where an adversary can inject arbitrary
text into the agent's context. So the large model never sees raw web
content.

`web_search` delegates to a small-model agent (`BOXBOT_MODEL_SMALL`)
whose only tools are web search and URL fetch. It strips boilerplate,
discards prompt-injection attempts, flags conflicting sources, and
returns a plain-text summary with citations.

Defense in depth:

1. The small model filters injection attempts.
2. Its output is plain text — it cannot call tools or emit SDK actions.
3. The large model applies its own judgment to the summary.

The small agent has **no** boxBot access: no SDK, no memory, no
messaging, no `execute_script`. Results are ephemeral; nothing persists
unless the agent later saves a fact deliberately.

Its system prompt is `SMALL_AGENT_SYSTEM_PROMPT` in `web_search.py` —
hardcoded, not modifiable by the large model. Read it there.

Token economics: pages run 10k+ tokens; summaries run 200–500. The
firewall is also the context budget. 30s timeout on the
search-and-summarize loop; on timeout it returns what it has, flagged
incomplete.

No SDK counterpart. Sandbox scripts that need web data use
`requests`/`httpx` directly — they already have network access. The
filter protects the large model's context, not the sandbox.

## Files

- `base.py` — the `Tool` base class: `name`, `description`,
  `parameters` (JSON schema), `async execute(**kwargs)`. Returns a
  string, or a list of content blocks for multimodal results.
- `registry.py` — singleton registry. `get_tools()` for agent init,
  `get_tool(name)` for lookup. Always loads all tools; no conditional
  discovery.
- `sandbox_runner.py` — the conversation-scoped long-lived sandbox
  subprocess. Python state persists across turns within a conversation.
- `_sandbox_actions.py` — action protocol: parses
  `__BOXBOT_SDK_ACTION__:` lines, dispatches per module, accumulates
  image attachments.
- `_sandbox_server.py`, `_tool_context.py` — transport and
  current-conversation plumbing.
- `builtins/` — one module per tool.

## What lives in the SDK instead

| Operation | Module | Why not a tool |
|-----------|--------|----------------|
| Create display | `bb.display` | Infrequent, multi-step |
| Create skill | `bb.skill` | Infrequent, multi-step |
| Install package | `bb.packages` | Infrequent, human-approved |
| Manage photos | `bb.photos` | Multi-step: tags, slideshow, soft delete |
| Manage memory | `bb.memory` | Extraction is mostly automatic |
| Notes and files | `bb.workspace` | Composable with everything else |
| Camera stills | `bb.camera` | Usually composed with a save or a note |
| External data | `bb.integrations` | Composable; calendar lives here |
| Batch task edits | `bb.tasks` | The tool covers one-off edits |

Photo **search** keeps a tool because users ask for photos constantly.
Photo **management** stays SDK-only because it composes.
