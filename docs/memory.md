# Memory System

## Overview

boxBot's memory gives it persistent, contextual recall across
conversations. Every interaction should feel like a continuation, not a
fresh start. The agent remembers what you've told it, what it's observed,
and what it's learned from its own actions.

The memory system has three components:
- **Three stores** with different loading strategies and lifecycles
- **Four memory types** for categorization and retrieval
- **A unified search backend** shared across the `search_memory` tool,
  the `boxbot_sdk`, and automatic injection at conversation start

## Three-Store Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  SYSTEM MEMORY (data/memory/system.md)                      │
│  Always loaded in system prompt. Core household facts,      │
│  standing instructions, operational notes.                   │
│  Updated by extraction agent when warranted.                │
│  ~2-4 KB budget.                                            │
├─────────────────────────────────────────────────────────────┤
│  FACT MEMORIES (data/memory/memory.db → memories table)     │
│  Typed, tagged, searchable. Retrieved contextually based    │
│  on who is present and what is being discussed.             │
│  Four types: person, household, methodology, operational.   │
│  Managed by access-based retention.                         │
├─────────────────────────────────────────────────────────────┤
│  CONVERSATION LOG (data/memory/memory.db → conversations)   │
│  Ultra-compact summaries of every conversation. Rolling     │
│  2-month window. Searchable for context and continuity.     │
│  Format: max info, min tokens.                              │
└─────────────────────────────────────────────────────────────┘
```

### System Memory

A structured markdown file loaded in full into the agent's system
prompt for every conversation. Contains information the agent should
**always know** without needing to search.

```markdown
## Household
- Jacob (admin) and Carina (admin) — primary owners
- Erik (child, 1st grade) — loves Pokemon, Monster Trucks
- Zara (child) — loves space, princesses

## Standing Instructions
- Regularly update and organize the slideshow
- Carina is still learning my functionality — explain features patiently

## Operational Notes
- [entries managed by the agent]
```

System memory uses **named sections** so the update mechanism can target
specific sections (`update section "Household"`) rather than doing
free-form text edits. See [System Memory Guardrails](#system-memory-guardrails).

### Fact Memories

Individual pieces of knowledge extracted from conversations, stored in
SQLite with full-text search and vector embeddings. Each fact is typed
(see [Memory Types](#memory-types)), tagged with relevant people, and
indexed for hybrid retrieval.

Facts are the core of the memory system. They persist based on
**access-based retention** — facts that keep being relevant survive;
unused facts fade and are eventually archived.

### Conversation Log

Ultra-compact summaries of every conversation, optimized for maximum
information at minimum token cost.

**Format principles:**
- Preserve all key information: who, what, decisions, actions, outcomes
- Strip filler, pleasantries, and verbose phrasing
- Use shorthand: names, abbreviations, comma-separated lists

**Examples:**
```
Discussed pizza for dinner. Carina: BBQ chicken. Jacob: chicken pesto.
Agreed on pesto. Texted shopping list to Carina.

Erik asked about Pokemon types. Explained fire/water/grass triangle.
Erik wants to trade cards at school tomorrow.

Carina asked how to set a reminder. Walked through manage_tasks.
Created recurring 8 AM school prep reminder for weekdays.
```

Conversations have a **rolling 2-month retention window**. Older
summaries are archived but remain searchable with `include_archived`.

## Memory Types

| Type | Purpose | Example | Retention |
|------|---------|---------|-----------|
| **person** | Facts about a specific individual | "Erik goes to GISC on Murray Blvd, 1st grade" | 6 months |
| **household** | Shared environment facts not tied to one person | "The WiFi password is on the fridge" | 6 months |
| **methodology** | Agent's learned approaches and lessons | "Weather API requires zip code, not city name" | 3 months |
| **operational** | Actions the agent took, activity log | "Removed 8 old slideshow photos, added 4 from Carina on 1/1/2026" | 2 months |

**Notes:**
- **person** memories are not siloed per person. If Carina tells BB
  about dinner plans, Jacob can ask about the topic and BB will say
  "Carina and I discussed a chicken salad for tonight."
- **household** covers facts that don't belong to a specific person.
  Without this type, "the family car is a blue Subaru" would need to be
  awkwardly tagged to every household member.
- **methodology** memories are global — they capture the agent's own
  learning regardless of which person was involved in the conversation.
- **operational** memories serve as an activity log, useful for
  "when did I last update the slideshow?" queries.

## Database Schema

```sql
-- Fact memories
CREATE TABLE memories (
    id                TEXT PRIMARY KEY,    -- uuid
    type              TEXT NOT NULL,       -- person | household | methodology | operational
    content           TEXT NOT NULL,       -- the full memory content
    summary           TEXT NOT NULL,       -- one-line summary for injection
    person            TEXT,                -- primary person (null for household/methodology)
    people            TEXT NOT NULL,       -- JSON list of all people involved/mentioned
    tags              TEXT NOT NULL,       -- JSON list of topic tags
    source_conversation TEXT,              -- FK → conversations.id (provenance)
    created_at        TEXT NOT NULL,       -- ISO 8601
    last_relevant_at  TEXT NOT NULL,       -- updated on access/injection/filter pass
    status            TEXT NOT NULL DEFAULT 'active',  -- active | archived | invalidated
    invalidated_by    TEXT,                -- conversation ID that caused invalidation
    superseded_by     TEXT,                -- FK → memories.id (replacement memory)
    embedding         BLOB,               -- 384-dim float32 (MiniLM)

    FOREIGN KEY (source_conversation) REFERENCES conversations(id),
    FOREIGN KEY (superseded_by) REFERENCES memories(id)
);

CREATE VIRTUAL TABLE memories_fts USING fts5(
    content, summary, tags, person, people,
    content='memories', content_rowid='rowid'
);

-- Conversation log
CREATE TABLE conversations (
    id                TEXT PRIMARY KEY,    -- uuid
    channel           TEXT NOT NULL,       -- voice | whatsapp
    participants      TEXT NOT NULL,       -- JSON list of people
    started_at        TEXT NOT NULL,       -- ISO 8601
    summary           TEXT NOT NULL,       -- ultra-compact summary
    topics            TEXT NOT NULL,       -- JSON list of topic tags
    accessed_memories TEXT NOT NULL,       -- JSON list of memory IDs injected
    embedding         BLOB,               -- 384-dim float32 (MiniLM)
);

CREATE VIRTUAL TABLE conversations_fts USING fts5(
    summary, topics, participants,
    content='conversations', content_rowid='rowid'
);

-- System memory version history
CREATE TABLE system_memory_versions (
    version           INTEGER PRIMARY KEY AUTOINCREMENT,
    content           TEXT NOT NULL,       -- full file content at this version
    updated_at        TEXT NOT NULL,       -- ISO 8601
    updated_by        TEXT NOT NULL,       -- "extraction_agent" or conversation ID
    change_summary    TEXT NOT NULL        -- what changed
);
```

## The `search_memory` Tool

A core tool (always loaded) with three modes. Shares its search backend
with the `boxbot_sdk.memory.search()` function and the automatic
injection system.

### Parameters

```
search_memory(
    mode:       "lookup" | "summary" | "get"
    query:      str | null       -- required for lookup and summary
    memory_id:  str | null       -- required for get
    types:      list[str] | null -- optional filter: person, household, methodology, operational
    person:     str | null       -- optional filter by person name
    include_conversations: bool = true   -- include conversation log in results
    include_archived:      bool = false  -- include archived memories
)
```

### Lookup Mode

Returns the most relevant facts and conversations for a query, each
with a contextual title, one-sentence summary, and relevance reason
generated by the small model.

**Pipeline:**
```
Query: "Erik school pickup"
  │
  ├─ 1. Hybrid search
  │     Vector cosine similarity (MiniLM embeddings)
  │     + SQLite FTS5 BM25 keyword matching
  │     Combined score: 0.6 × vector + 0.4 × BM25 (configurable)
  │     → top 30 fact candidates + top 10 conversation candidates
  │
  ├─ 2. Small model reranking (5-6 parallel Haiku calls)
  │     Batch candidates into groups of ~6
  │     Each call: structured output per candidate
  │     Output: relevant (bool), title, summary, relevance_reason
  │
  └─ 3. Return top 10 facts + top 3 conversations
```

**Return format:**
```json
{
  "facts": [
    {
      "id": "mem-223",
      "type": "person",
      "person": "Erik",
      "title": "Erik's school schedule",
      "summary": "Erik attends GISC on Murray Blvd, pickup at 3:15 PM",
      "relevance": "Directly answers the school pickup question"
    }
  ],
  "conversations": [
    {
      "id": "conv-42",
      "date": "2026-01-18",
      "participants": ["Jacob"],
      "title": "Erik's schedule discussion",
      "summary": "Jacob asked about Erik's weekly schedule changes",
      "relevance": "Recent conversation covering Erik's school routine"
    }
  ]
}
```

### Summary Mode

Answers a natural language question by searching, filtering in parallel,
and synthesizing an answer from relevant memories.

**Pipeline:**
```
Question: "What kind of food does Erik like?"
  │
  ├─ 1. Hybrid search → top 30 candidates
  │
  ├─ 2. Parallel filtering (5-6 Haiku calls)
  │     Each group evaluated for relevance to the question
  │     Returns: relevant memories with extracted key snippets
  │
  ├─ 3. Synthesis (1 Haiku call)
  │     Input: question + all relevant snippets from step 2
  │     Output: natural language answer with source citations
  │
  └─ 4. Return answer + source memory IDs
```

**Return format:**
```json
{
  "answer": "Erik primarily enjoys kid-friendly foods: pizza (especially pepperoni), chicken nuggets, mac & cheese, and hot dogs. He recently tried tacos and liked them. He doesn't like spicy food but will eat carrots.",
  "sources": ["mem-123", "mem-456", "conv-42"]
}
```

Summary mode is better for broad questions ("what does Erik like to
eat?"). Lookup mode is better for finding specific references ("Erik
school pickup time").

### Get Mode

Retrieves the full record for a specific memory by ID. No small model
involvement — direct database lookup. Use this when lookup mode returns
a summary and the agent needs the complete content.

**Return format:**
```json
{
  "id": "mem-223",
  "type": "person",
  "person": "Erik",
  "content": "Erik goes to German International School (GISC) on Murray Blvd. He is in 1st grade. He takes the bus in the morning and gets picked up at 3:15 PM on weekdays. Fridays are early dismissal at 12:30 PM.",
  "summary": "Erik attends GISC on Murray Blvd, 1st grade, pickup 3:15 PM",
  "tags": ["school", "schedule", "transportation"],
  "people": ["Erik"],
  "source_conversation": "conv-38",
  "created_at": "2026-01-10T14:30:00",
  "last_relevant_at": "2026-02-15T09:00:00",
  "status": "active"
}
```

### DRY Architecture: Shared Backend

One search backend, three interfaces:

```
src/boxbot/memory/search.py              ← search backend
    │                                       (hybrid retrieval + Haiku reranking)
    │
    ├── src/boxbot/tools/search_memory.py   ← core tool
    │     Agent calls directly during conversation.
    │     Returns structured results in tool response.
    │
    ├── src/boxbot/memory/retrieval.py      ← injection system
    │     Called at conversation start with person + first utterance.
    │     Formats results into system prompt injection block.
    │
    └── src/boxbot/sdk/memory.py            ← SDK function
          Sandbox scripts call boxbot_sdk.memory.search().
          Emits JSON action → main process routes to search backend.
```

The tool and injection system call the backend directly (main process).
The SDK function emits a structured JSON request on stdout, resolved by
the `execute_script` tool via the same backend.

## Memory Injection

At the start of every conversation, relevant memories are automatically
injected into the agent's system prompt. The injection uses the same
search backend as `search_memory`, informed by **who is speaking** and
**what they said** (the first utterance).

### Flow

```
1. User speaks → wake word → STT → first utterance text
2. Injection system calls search backend:
     query  = first utterance text
     person = identified speaker(s)
3. Backend returns relevant memories (hybrid search + Haiku reranking)
4. Results formatted into injection block with memory IDs
5. Agent receives: system memory + injected memories + user utterance
```

For **scheduled/person-triggered wakes** (agent initiates), the task
description provides the query context instead of a user utterance.

For **WhatsApp messages**, the message content and sender identity
provide the query context.

In all cases, the injection step is identical: call the shared search
backend with context, format results into the system prompt.

### Injection Format

Injected memories include IDs so the extraction agent can reference
them for invalidation when contradictions are detected.

```
[Active Memories]
#541 (person/Jacob): Jacob has no known food allergies
#223 (person/Erik): Erik's school pickup is 3:15 PM on weekdays
#890 (operational): Updated slideshow 1/15/2026 — added 5 photos from Carina
#102 (household): Family car is a blue 2022 Subaru Outback

[Recent Conversations]
#conv-42 (1/18, Jacob, voice): Discussed weekend plans. Jacob wants aquarium.
  Checked hours — open 9-5 Sat.
#conv-39 (1/15, Carina, whatsapp): Carina sent 5 vacation photos. Added to
  slideshow rotation.
```

### Token Budget

| Source | Budget | Priority |
|--------|--------|----------|
| System memory | Always loaded (full file) | Highest |
| Person-tagged facts | Up to 10 memories | High — facts about present people |
| Topic-matched facts | Up to 5 memories | Medium — matched to utterance |
| Recent conversations | Up to 3 summaries | Lower — recent context |

Total injection budget: ~2000 tokens (excluding system memory). If the
agent needs more context during conversation, it calls `search_memory`
explicitly.

## Memory Extraction

After every conversation, a background extraction agent processes the
full transcript and produces a single structured output that
simultaneously:
- Summarizes the conversation (for the conversation log)
- Extracts new fact memories
- Detects contradictions with injected memories (for invalidation)
- Proposes system memory updates (when warranted)
- Deduplicates against existing memories

### When Extraction Runs

- **On conversation end** — primary trigger, processes full transcript
- **On context compaction** — if the conversation is long enough to
  require compaction, extract memories before compacting

### Input to Extraction Agent

The extraction agent receives:
1. **Full conversation transcript** with speaker attribution
2. **Accessed memories** — IDs and content of all memories injected
   into this conversation (enables contradiction detection)
3. **Similar existing memories** — for each candidate fact, the top-5
   most similar existing memories by vector cosine similarity (enables
   deduplication, provided in a second pass after initial extraction)

### Structured Output Format

The extraction agent returns a single structured response:

```json
{
  "conversation_summary": {
    "participants": ["Jacob", "Carina"],
    "channel": "voice",
    "topics": ["dinner", "groceries", "slideshow"],
    "summary": "Discussed dinner. Carina: BBQ chicken. Jacob: chicken pesto. Agreed on pesto. Texted shopping list to Carina. Jacob asked to update slideshow with vacation photos."
  },
  "extracted_memories": [
    {
      "type": "person",
      "person": "Jacob",
      "content": "Jacob prefers chicken pesto pizza",
      "summary": "Jacob's pizza preference: chicken pesto",
      "tags": ["food", "preference"],
      "action": "create"
    },
    {
      "type": "operational",
      "person": null,
      "content": "Sent grocery shopping list to Carina via WhatsApp on 2/21/2026: mozzarella, pesto, chicken breast, pizza dough",
      "summary": "Sent grocery list to Carina 2/21/2026",
      "tags": ["groceries", "whatsapp"],
      "action": "create"
    },
    {
      "type": "person",
      "person": "Erik",
      "content": "Erik's favorite foods: pizza (pepperoni), chicken nuggets, mac & cheese, hot dogs. Recently tried tacos — liked them. Doesn't like spicy food. Will eat carrots.",
      "summary": "Erik likes pizza, nuggets, mac & cheese. No spicy food.",
      "tags": ["food", "preference"],
      "action": "update",
      "existing_memory_id": "mem-123",
      "reason": "Adding new info: Erik tried tacos and liked them"
    }
  ],
  "invalidations": [
    {
      "memory_id": "mem-541",
      "reason": "Jacob explicitly stated 'I'm not allergic to anything'",
      "replacement": {
        "type": "person",
        "person": "Jacob",
        "content": "Jacob has no food allergies",
        "summary": "Jacob has no known allergies",
        "tags": ["health", "allergies"]
      }
    }
  ],
  "system_memory_updates": [
    {
      "section": "Standing Instructions",
      "action": "add_entry",
      "content": "When updating slideshow, prioritize recent family photos over landscapes"
    }
  ]
}
```

### Deduplication

Before inserting extracted memories, the system checks for near-
duplicates. For each candidate memory, the top-5 most similar existing
memories (by vector cosine similarity) are provided to the extraction
agent. It decides:

| Action | When | Effect |
|--------|------|--------|
| **create** | No similar memory exists | Insert new memory |
| **update** | Similar memory exists but candidate adds info | Merge content into existing, update embedding |
| **skip** | Exact or near-exact duplicate | No action |

### Invalidation

When the extraction agent detects a contradiction between an injected
memory and the conversation content, it produces an invalidation:

1. The existing memory's status is set to `invalidated`
2. `invalidated_by` records the conversation ID
3. A replacement memory is created with the corrected information
4. `superseded_by` on the old memory links to the replacement

**Invalidation is always a post-conversation operation.** During the
conversation, the agent already knows the correction — it's in the
transcript. The memory store only needs to update for the *next*
conversation.

The invalidated memory is preserved (soft delete) for audit trail. The
chain old → `invalidated_by` → `superseded_by` traces the evolution of
the agent's knowledge over time.

### System Memory Updates

The extraction agent can propose updates to system memory when it
encounters information important enough to be always-loaded. Updates
target a specific **section** with one of three actions:

| Action | Effect |
|--------|--------|
| `set` | Replace the entire section content |
| `add_entry` | Add a bullet point to the section |
| `remove_entry` | Remove a specific bullet point |

System memory updates are subject to
[guardrails](#system-memory-guardrails).

## System Memory Guardrails

The agent auto-updates its system memory without requiring human
approval. Security relies on structural guardrails, not agent
perfection.

| Guardrail | Enforcement |
|-----------|-------------|
| **Size cap** | Updates rejected if total file would exceed ~4 KB |
| **Version history** | Last 20 versions stored in `system_memory_versions` table. Admin can rollback via command |
| **Section schema** | Predefined sections (Household, Standing Instructions, Operational Notes). Agent cannot create arbitrary sections or delete the template structure |
| **No secrets** | Content validated against common patterns (API keys, tokens, passwords). Secrets belong in the SDK secret store, not system memory |

If the agent writes incorrect information to system memory, the damage
is bounded (size cap, section structure) and recoverable (version
history). The worst case is a few KB of wrong text that an admin can
rollback.

**Secret store:** The `boxbot_sdk` provides a separate secret store
(`boxbot_sdk.secrets`) for API keys and credentials. The agent can store
a secret and use it for API calls but cannot view the value again after
storage. Secrets must never be stored in system memory or fact memories.
See the SDK documentation for details.

## Decay & Retention

Memory lifecycle uses **access-based retention**: memories that keep
being useful survive, unused memories fade.

### How It Works

Every memory tracks a `last_relevant_at` timestamp, updated when:
- The memory is **created**
- The memory **passes the small model filter** during a search
- The memory is **injected** into a conversation
- The memory is **explicitly retrieved** via get mode

Each memory type has a **retention window**:

| Type | Retention Window |
|------|-----------------|
| person | 6 months |
| household | 6 months |
| methodology | 3 months |
| operational | 2 months |
| conversation | 2 months |

### Daily Maintenance Job

A background job runs daily:

1. Memories where `now - last_relevant_at > retention_window` →
   status set to **archived**
2. Archived memories are excluded from injection and default searches
3. Archived memories remain searchable with `include_archived=true`
4. If an archived memory surfaces in an explicit search and passes the
   small model filter → **auto-unarchived** (it's relevant again)

No importance scores, no decay curves, no multiplication factors. The
model is simple: *was this useful recently?* The small model filter is
the gatekeeper — if a memory keeps being relevant to queries, it stays
alive. If nobody ever needs it, it fades.

### Conversations

Conversation log entries follow the same `last_relevant_at` logic with
a fixed 2-month retention window. Conversations older than 2 months are
archived. Facts extracted from those conversations persist independently
(linked via `source_conversation` for provenance).

## Storage Management

### Size-Based Cap

The memory store has a configurable **size cap** (default: 50 MB for
the full `memory.db`). When storage exceeds **99%** of the cap:

1. Sort **archived** memories by `last_relevant_at` ascending
   (oldest first)
2. Delete archived memories until storage drops to **70%** of the cap
3. If still over 70% after clearing all archives, evict the oldest
   **active** memories with the least recent `last_relevant_at`

This provides a burst buffer: normal operation stays well under the cap,
but spikes (many conversations in a short period) are absorbed and
cleaned on the next maintenance pass.

### What Gets Counted

The size cap covers the entire `memory.db` file:
- `memories` table (facts + embeddings)
- `conversations` table (summaries + embeddings)
- `system_memory_versions` table (version history)
- FTS5 index data

System memory (`data/memory/system.md`) is not counted against the cap
— it has its own ~4 KB budget enforced separately.

## Embedding Strategy

Memory search uses **hybrid retrieval**: vector similarity for semantic
matching combined with SQLite FTS5 BM25 for keyword matching. This
requires text embeddings for every memory and conversation summary.

### On-Device Embeddings

**Model:** `all-MiniLM-L6-v2` (or equivalent small sentence transformer)
- 22M parameters, 384-dimensional embeddings
- Runs on CPU in ~10-50ms per embedding
- No API calls, no network dependency
- Quality is sufficient when combined with BM25 + small model reranking

**Embedding generation:** Synchronous at memory creation time. Each
memory gets its embedding immediately on insert, ensuring it's
searchable right away. At ~50ms per embedding, this adds negligible
latency to the extraction pipeline.

**Hybrid scoring:** Vector cosine similarity and BM25 scores are
normalized and combined with configurable weights (default: 0.6 vector,
0.4 BM25). The combined score ranks candidates before the small model
reranking step.

### Why Not API-Based Embeddings?

- Adds ~200ms network latency to every search
- Adds cost (small but cumulative with frequent searches)
- Makes memory search dependent on network availability
- On-device quality is "good enough" — the small model reranker handles
  precision after the initial retrieval

## Privacy

- All memory storage is local (SQLite on device)
- Embeddings are abstract float vectors, not reconstructible to text
- System memory and fact memories never leave the device except via
  explicit agent actions (e.g., sending a message via WhatsApp)
- Conversation transcripts are summarized and discarded — only compact
  summaries are stored
- The secret store encrypts credentials at rest
- No memory telemetry or cloud sync

## Files

### `search.py`
Unified search backend: hybrid vector + BM25 retrieval with small model
reranking. Used by the `search_memory` tool, injection system, and SDK.

### `store.py`
Memory persistence layer: SQLite CRUD, FTS5 indexing, embedding storage,
status management (active/archived/invalidated).

### `extraction.py`
Post-conversation memory extraction agent. Produces structured output:
conversation summary, new facts, invalidations, system memory updates.
Handles deduplication via similarity check against existing memories.

### `retrieval.py`
Memory injection at conversation start. Calls the search backend with
person + utterance context, formats results into the system prompt
injection block with memory IDs.

### `maintenance.py`
Daily background job: retention window enforcement, archival, storage
cap eviction, auto-unarchival of accessed archived memories, FTS5 index
rebuild.

### `embeddings.py`
On-device text embedding generation using MiniLM. Provides `embed()`
used by `store.py` on memory creation and by `search.py` for query
embedding at search time.
