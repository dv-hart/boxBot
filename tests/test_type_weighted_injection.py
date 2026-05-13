"""Tests for type-weighted memory injection (lifecycle step 5).

Before: ``inject_memories`` returned top-15 by hybrid score across all
types — a cluster of near-identical operational entries could occupy
every slot, evicting load-bearing methodology and person memories.

After: each type gets a guaranteed budget. With operational pinned at
0, those entries never crowd the block.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from boxbot.memory.search import SearchCandidate


def _mk_memory(
    cid: str, mtype: str, summary: str = "x", person: str | None = None
) -> SearchCandidate:
    return SearchCandidate(
        id=cid,
        source="memory",
        type=mtype,
        person=person,
        content=summary,
        summary=summary,
        vector_score=0.9,
        bm25_score=0.5,
        combined_score=0.7,
        metadata={},
    )


@pytest.mark.asyncio
async def test_operational_budget_zero_keeps_them_out_of_block():
    """Even when operational dominates hybrid search, the per-type
    budget of 0 prevents them from appearing in the injection block."""
    from boxbot.memory.retrieval import inject_memories

    # 20 operational hits, 1 methodology hit. Pre-step-5, methodology
    # would be drowned. Step 5 budget=0 for operational means
    # methodology survives.
    candidates = [
        _mk_memory(f"op{i:02d}", "operational", f"op log {i}")
        for i in range(20)
    ] + [
        _mk_memory("meth01", "methodology", "use bb.integrations.get('calendar')"),
    ]

    class FakeStore:
        async def update_memory_relevance(self, _id): pass

    async def fake_search(*args, **kwargs):
        return candidates

    with patch("boxbot.memory.retrieval.hybrid_search", new=fake_search):
        block, ids = await inject_memories(
            FakeStore(), person="Jacob", utterance="what's up",
        )

    # No operational IDs surfaced
    assert all(not i.startswith("op") for i in ids)
    # Methodology is in the block
    assert "meth01" in ids
    assert "use bb.integrations.get" in block


@pytest.mark.asyncio
async def test_each_type_receives_its_budget_exactly():
    """If the candidate pool has more memories of a type than its
    budget, only the budget-many top-ranked are taken."""
    from boxbot.memory.retrieval import TYPE_BUDGETS, inject_memories

    candidates = (
        [_mk_memory(f"p{i:02d}", "person", f"person {i}", person="Jacob")
         for i in range(10)]
        + [_mk_memory(f"h{i:02d}", "household", f"household {i}")
           for i in range(10)]
        + [_mk_memory(f"m{i:02d}", "methodology", f"meth {i}")
           for i in range(10)]
    )

    class FakeStore:
        async def update_memory_relevance(self, _id): pass

    async def fake_search(*args, **kwargs):
        return candidates

    with patch("boxbot.memory.retrieval.hybrid_search", new=fake_search):
        block, ids = await inject_memories(
            FakeStore(), person="Jacob", utterance="hi",
        )

    by_prefix = {"p": 0, "h": 0, "m": 0}
    for mid in ids:
        if mid[0] in by_prefix:
            by_prefix[mid[0]] += 1
    assert by_prefix["p"] == TYPE_BUDGETS["person"]
    assert by_prefix["h"] == TYPE_BUDGETS["household"]
    assert by_prefix["m"] == TYPE_BUDGETS["methodology"]


@pytest.mark.asyncio
async def test_within_a_type_ranking_is_preserved():
    """Hybrid search returns ranked output; we take the top-N within
    each type bucket. The order within candidates simulates ranking."""
    from boxbot.memory.retrieval import inject_memories

    candidates = [
        _mk_memory(f"m{i:02d}", "methodology", f"meth {i}")
        for i in range(10)
    ]

    class FakeStore:
        async def update_memory_relevance(self, _id): pass

    async def fake_search(*args, **kwargs):
        return candidates

    with patch("boxbot.memory.retrieval.hybrid_search", new=fake_search):
        block, ids = await inject_memories(
            FakeStore(), person="Jacob", utterance="hi",
        )

    # Methodology budget is 4 → first four candidates by order
    assert ids[:4] == ["m00", "m01", "m02", "m03"]


@pytest.mark.asyncio
async def test_empty_type_bucket_does_not_break_injection():
    """If a type has zero candidates, the rest still surface."""
    from boxbot.memory.retrieval import inject_memories

    candidates = [
        _mk_memory("m01", "methodology", "use the SDK"),
    ]

    class FakeStore:
        async def update_memory_relevance(self, _id): pass

    async def fake_search(*args, **kwargs):
        return candidates

    with patch("boxbot.memory.retrieval.hybrid_search", new=fake_search):
        block, ids = await inject_memories(
            FakeStore(), person="Jacob", utterance="hi",
        )

    assert ids == ["m01"]
    assert "use the SDK" in block


@pytest.mark.asyncio
async def test_custom_budgets_override_defaults():
    """Tests + future tuning can override the default budgets."""
    from boxbot.memory.retrieval import inject_memories

    candidates = [
        _mk_memory(f"p{i:02d}", "person", f"p {i}", person="Jacob")
        for i in range(5)
    ]

    class FakeStore:
        async def update_memory_relevance(self, _id): pass

    async def fake_search(*args, **kwargs):
        return candidates

    with patch("boxbot.memory.retrieval.hybrid_search", new=fake_search):
        block, ids = await inject_memories(
            FakeStore(), person="Jacob", utterance="hi",
            type_budgets={"person": 2},
        )
    assert len(ids) == 2
