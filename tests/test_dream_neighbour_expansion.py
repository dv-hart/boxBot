"""Tests for nearest-neighbour expansion in find_near_duplicates
(lifecycle step 8).

Before: dedup compared only memories inside the CandidateSet (today's
new + 6 sampled revisits). A back-catalogue dupe was caught only if
both members happened to land in the same window — probability ~7%
per night on a 90-memory corpus. The audit log on the Pi shows 10
straight nights with zero pairs flagged.

After: for each new-today memory, the dream phase scores against
every active memory and pairs above the threshold. Back-catalogue
dupes get caught the next night the agent creates anything similar.
"""
from __future__ import annotations

import math

import pytest

# Reuse the helpers from the existing dream-phase test module.
from tests.test_dream_phase import _seed, _set_embedding, _unit


@pytest.mark.asyncio
async def test_neighbour_expansion_catches_backlog_dupes(tmp_path, monkeypatch):
    """Two near-identical memories created on different days: under
    the old narrow-pool logic, neither's pair would surface unless
    both landed in the same CandidateSet. Under the new logic, the
    newer memory's neighbour search finds the older one."""
    from pathlib import Path as _Path
    from boxbot.memory.store import MemoryStore
    from boxbot.memory.dream import (
        CandidateSet,
        find_near_duplicates,
    )

    store = MemoryStore(db_path=_Path(tmp_path) / "mem.db")
    await store.initialize()
    try:
        # Old memory (lives in active pool, NOT in CandidateSet)
        v_old = _unit([1.0, 0.0, 0.0])
        old_id = await _seed(
            store, type_="methodology",
            content="calendar via integrations",
            summary="calendar via bb.integrations.get",
            embedding_vec=v_old,
        )

        # New memory created today (in new_today), embedding near the
        # old one — they're effectively duplicates.
        v_new = _unit([0.97, 0.243, 0.0])
        new_id = await _seed(
            store, type_="methodology",
            content="use bb.integrations.get for calendar",
            summary="use bb.integrations.get for calendar",
            embedding_vec=v_new,
        )

        new_mem = await store.get_memory_no_touch(new_id)

        # CandidateSet contains ONLY the new memory — the old one is
        # not surfaced via revisits. The neighbour expansion should
        # still find it in the active pool.
        cands = CandidateSet(new_today=[new_mem], revisits=[])
        pairs = await find_near_duplicates(store, cands)

        pair_ids = {(p.memory_id_a, p.memory_id_b) for p in pairs}
        # Pair-ID ordering is normalised — check both orderings.
        assert (
            (old_id, new_id) in pair_ids
            or (new_id, old_id) in pair_ids
        )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_neighbour_expansion_respects_threshold(tmp_path, monkeypatch):
    """Memories below the cosine threshold must NOT be paired even
    if they're the new memory's nearest neighbour."""
    from pathlib import Path as _Path
    from boxbot.memory.store import MemoryStore
    from boxbot.memory.dream import (
        CandidateSet,
        NEAR_DUP_THRESHOLD,
        find_near_duplicates,
    )

    store = MemoryStore(db_path=_Path(tmp_path) / "mem.db")
    await store.initialize()
    try:
        # Orthogonal vectors — cosine = 0 → far below threshold
        v_old = _unit([1.0, 0.0, 0.0])
        await _seed(
            store, type_="methodology",
            content="x", summary="x", embedding_vec=v_old,
        )
        v_new = _unit([0.0, 1.0, 0.0])
        new_id = await _seed(
            store, type_="methodology",
            content="y", summary="y", embedding_vec=v_new,
        )
        new_mem = await store.get_memory_no_touch(new_id)
        cands = CandidateSet(new_today=[new_mem], revisits=[])
        pairs = await find_near_duplicates(store, cands)
        assert pairs == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_co_injected_pairs_surfaced_by_neighbour_expansion(
    tmp_path, monkeypatch,
):
    """Neighbour expansion surfaces a co-injected pair rather than skipping
    it. The old exclusion let two contradictory memories that kept being
    injected together evade dedup forever; co-injection is now a reason to
    adjudicate, with the enriched judge as the precision gate."""
    from pathlib import Path as _Path
    from boxbot.memory.store import MemoryStore
    from boxbot.memory.dream import (
        CandidateSet,
        find_near_duplicates,
    )

    store = MemoryStore(db_path=_Path(tmp_path) / "mem.db")
    await store.initialize()
    try:
        v_a = _unit([1.0, 0.0, 0.0])
        v_b = _unit([0.99, 0.141, 0.0])  # very close
        old_id = await _seed(
            store, type_="methodology",
            content="a", summary="a", embedding_vec=v_a,
        )
        new_id = await _seed(
            store, type_="methodology",
            content="b", summary="b", embedding_vec=v_b,
        )

        # Both memories were accessed in one conversation — no longer
        # exempts the pair.
        await store.create_conversation_stub(
            conversation_id="conv_shared",
            channel="voice",
            participants=["Jacob"],
        )
        await store.update_conversation(
            "conv_shared",
            summary="",
            topics=[],
            accessed_memories=[old_id, new_id],
        )

        new_mem = await store.get_memory_no_touch(new_id)
        cands = CandidateSet(new_today=[new_mem], revisits=[])
        pairs = await find_near_duplicates(store, cands)
        assert len(pairs) == 1
        assert {pairs[0].memory_id_a, pairs[0].memory_id_b} == {old_id, new_id}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_dedup_pair_cap_enforced(tmp_path, monkeypatch):
    """Generate more near-dup pairs than MAX_DEDUP_PAIRS_DEFAULT and
    confirm the output is capped, retaining highest-confidence first."""
    from pathlib import Path as _Path
    from boxbot.memory.store import MemoryStore
    from boxbot.memory.dream import (
        CandidateSet,
        MAX_DEDUP_PAIRS_DEFAULT,
        find_near_duplicates,
    )

    store = MemoryStore(db_path=_Path(tmp_path) / "mem.db")
    await store.initialize()
    try:
        # Anchor + many close neighbours so the expansion produces
        # > MAX_DEDUP_PAIRS_DEFAULT pairs.
        v_anchor = _unit([1.0, 0.0, 0.0])
        anchor_id = await _seed(
            store, type_="methodology",
            content="anchor", summary="anchor",
            embedding_vec=v_anchor,
        )
        # Create MAX+10 neighbours, each slightly different from anchor
        # but well above threshold. Vary the angle so cosines differ
        # slightly — gives the cap something to actually rank by.
        n_neighbours = MAX_DEDUP_PAIRS_DEFAULT + 10
        ids = []
        for i in range(n_neighbours):
            theta = 0.05 + i * 0.0005   # very small angle
            v = _unit([math.cos(theta), math.sin(theta), 0.0])
            nid = await _seed(
                store, type_="methodology",
                content=f"near {i}", summary=f"near {i}",
                embedding_vec=v,
            )
            ids.append(nid)

        # Use the anchor as today's new memory; everything else is
        # "old". The expansion will find every neighbour above
        # threshold.
        anchor_mem = await store.get_memory_no_touch(anchor_id)
        cands = CandidateSet(new_today=[anchor_mem], revisits=[])
        pairs = await find_near_duplicates(store, cands)

        assert len(pairs) <= MAX_DEDUP_PAIRS_DEFAULT
        # Pairs are sorted by cosine desc
        cosines = [p.cosine for p in pairs]
        assert cosines == sorted(cosines, reverse=True)
    finally:
        await store.close()


def test_dream_audit_only_default_is_apply_mode():
    """Step 8 flips the default. Tests pin the new contract."""
    from boxbot.core.config import MemoryConfig
    cfg = MemoryConfig()
    assert cfg.dream_audit_only is False
