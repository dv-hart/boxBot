# memory/

The memory system — gives boxBot persistent, contextual recall across
conversations. See [docs/memory.md](../../docs/memory.md) for the full
design specification.

## Architecture

Three stores, one search backend:

```
System Memory (data/memory/system.md)     ← always in system prompt
Fact Memories (memory.db → memories)      ← contextual retrieval
Conversation Log (memory.db → conversations) ← rolling 2-month window
```

Four memory types: `person`, `household`, `methodology`, `operational`.

The `search_memory` core tool, `boxbot_sdk.memory.search()`, and the
injection system all share the same search backend (`search.py`).

## Files

| File | Purpose |
|------|---------|
| `search.py` | Unified search backend (hybrid vector + BM25, small model reranking) |
| `store.py` | SQLite persistence (CRUD, FTS5, embeddings, status management) |
| `extraction.py` | Post-conversation extraction agent (structured output) |
| `retrieval.py` | Memory injection at conversation start |
| `maintenance.py` | Daily retention, archival, storage cap eviction |
| `embeddings.py` | On-device text embeddings (MiniLM) |

## Quick Reference

- **search_memory tool**: 3 modes — `lookup` (ranked results), `summary`
  (synthesized answer), `get` (full record by ID)
- **Injection**: automatic at conversation start using person + first
  utterance as search context
- **Extraction**: background agent after every conversation — extracts
  facts, invalidates contradictions, updates system memory
- **Retention**: access-based (person/household: 6mo, methodology: 3mo,
  operational/conversation: 2mo)
- **Storage cap**: size-based, evict oldest archived at >99%, down to 70%
