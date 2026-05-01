"""Tests for the dream-phase consolidation pipeline (PR1).

Mocks the Anthropic client following the same shape as
``tests/test_extraction_pipeline.py``. Covers gather/cluster/find-pairs,
batch shape, audit-only vs apply gates, DreamPoller resume, and the
undo script's idempotency.
"""

from __future__ import annotations

import asyncio
import json
import random
import subprocess
import sys
import tempfile
import types
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Anthropic client stub (mirrors test_extraction_pipeline.py)
# ---------------------------------------------------------------------------


class _StubBlock:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _StubMessage:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage


class _StubBatch:
    def __init__(self, batch_id, status="in_progress"):
        self.id = batch_id
        self.processing_status = status


class _StubResultEntry:
    def __init__(self, custom_id, result):
        self.custom_id = custom_id
        self.result = result


class _StubResult:
    def __init__(self, type_, message=None, error=None):
        self.type = type_
        self.message = message
        self.error = error


class _AsyncIter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class FakeAnthropicClient:
    def __init__(self):
        self._next_id = 0
        self._batches: dict[str, _StubBatch] = {}
        self._results: dict[str, list[_StubResultEntry]] = {}
        self.create_calls: list[dict] = []
        self.messages = types.SimpleNamespace(
            batches=types.SimpleNamespace(
                create=self._create,
                retrieve=self._retrieve,
                results=self._results_iter,
            )
        )

    async def _create(self, *, requests):
        self._next_id += 1
        bid = f"msgbatch_dream_{self._next_id}"
        self._batches[bid] = _StubBatch(bid, status="in_progress")
        self.create_calls.append({"id": bid, "requests": requests})
        return _StubBatch(bid, status="in_progress")

    async def _retrieve(self, batch_id):
        return self._batches[batch_id]

    async def _results_iter(self, batch_id):
        return _AsyncIter(self._results.get(batch_id, []))

    def end_with_decisions(
        self,
        batch_id: str,
        entries: list[tuple[str, dict]],
    ):
        """entries: list of (custom_id, dedup_decision_payload)."""
        out: list[_StubResultEntry] = []
        for custom_id, payload in entries:
            msg = _StubMessage(
                content=[_StubBlock(
                    type="tool_use", name="dedup_decision", input=payload,
                )]
            )
            out.append(_StubResultEntry(custom_id, _StubResult("succeeded", message=msg)))
        self._results[batch_id] = out
        self._batches[batch_id] = _StubBatch(batch_id, status="ended")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def fresh_store(tmp_path):
    # Pre-import core to break the package import cycle.
    import boxbot.core  # noqa: F401
    from boxbot.memory.store import MemoryStore

    db_path = tmp_path / "memory.db"
    sys_mem_path = tmp_path / "system.md"
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    with patch("boxbot.memory.store.SYSTEM_MEMORY_PATH", sys_mem_path), \
         patch("boxbot.memory.dream.WORKSPACE_DIR", workspace_dir):
        store = MemoryStore(db_path=db_path)
        await store.initialize()
        yield store
        await store.close()


@pytest.fixture
def fake_client():
    return FakeAnthropicClient()


def _set_embedding(store, mem_id, vec):
    """Helper: directly set a memory's embedding to a known vector."""
    blob = vec.astype(np.float32).tobytes()
    return store.db.execute(
        "UPDATE memories SET embedding = ? WHERE id = ?", (blob, mem_id),
    )


def _unit(v: list[float]) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    n = np.linalg.norm(arr)
    if n > 0:
        arr = arr / n
    return arr


async def _seed(store, *, content, summary, person=None,
                tags=None, type_="person", embedding_vec=None,
                created_at=None, last_relevant_at=None):
    """Create a memory and optionally override its embedding/timestamps."""
    mid = await store.create_memory(
        type=type_,
        content=content,
        summary=summary,
        person=person,
        tags=tags or [],
    )
    if embedding_vec is not None:
        await _set_embedding(store, mid, embedding_vec)
    sets = []
    params = []
    if created_at is not None:
        sets.append("created_at = ?")
        params.append(created_at)
    if last_relevant_at is not None:
        sets.append("last_relevant_at = ?")
        params.append(last_relevant_at)
    if sets:
        params.append(mid)
        await store.db.execute(
            f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", params,
        )
        await store.db.commit()
    return mid


# ---------------------------------------------------------------------------
# gather_candidates
# ---------------------------------------------------------------------------


class TestGatherCandidates:
    @pytest.mark.asyncio
    async def test_new_today_picks_up_recent(self, fresh_store):
        from boxbot.memory.dream import gather_candidates

        midnight = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        # New: created today
        new_id = await _seed(
            fresh_store, content="new fact", summary="new",
            created_at=(midnight + timedelta(hours=2)).isoformat(),
        )
        # Old: created yesterday
        old_id = await _seed(
            fresh_store, content="old fact", summary="old",
            created_at=(midnight - timedelta(days=1)).isoformat(),
        )
        cand = await gather_candidates(
            fresh_store, rng=random.Random(0),
        )
        new_today_ids = {m.id for m in cand.new_today}
        assert new_id in new_today_ids
        assert old_id not in new_today_ids

    @pytest.mark.asyncio
    async def test_revisit_pool_proportions_statistical(self, fresh_store):
        """Over many runs, all three pools should each contribute revisits."""
        from boxbot.memory.dream import gather_candidates

        midnight = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0,
        )

        # Seed a healthy distribution:
        # 5 used-today (older creation, recent last_relevant_at)
        used_today_ids = []
        for i in range(5):
            mid = await _seed(
                fresh_store, content=f"used today {i}",
                summary=f"u{i}",
                created_at=(midnight - timedelta(days=10)).isoformat(),
                last_relevant_at=(midnight + timedelta(hours=3)).isoformat(),
            )
            used_today_ids.append(mid)

        # 10 not-used-today, varying ages
        not_used_ids = []
        for i in range(10):
            mid = await _seed(
                fresh_store, content=f"old fact {i}",
                summary=f"o{i}",
                created_at=(midnight - timedelta(days=10 + i * 3)).isoformat(),
                last_relevant_at=(
                    midnight - timedelta(days=8)
                ).isoformat(),
            )
            not_used_ids.append(mid)

        pool_origin_counts = {"used_today": 0, "age_decayed": 0, "uniform": 0}
        runs = 50
        for seed in range(runs):
            cand = await gather_candidates(
                fresh_store, rng=random.Random(seed),
            )
            for m in cand.revisits:
                origin = cand.revisit_pool_origin.get(m.id)
                if origin in pool_origin_counts:
                    pool_origin_counts[origin] += 1

        # All three pools should have contributed at least once.
        assert pool_origin_counts["used_today"] > 0
        assert pool_origin_counts["age_decayed"] > 0
        assert pool_origin_counts["uniform"] > 0

        # The used-today pool should dominate (3 picks per run vs 2+1).
        assert pool_origin_counts["used_today"] > pool_origin_counts["uniform"]

    @pytest.mark.asyncio
    async def test_revisits_never_overlap_new_today(self, fresh_store):
        from boxbot.memory.dream import gather_candidates

        midnight = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        # New today AND used today
        nid = await _seed(
            fresh_store, content="new used", summary="nu",
            created_at=(midnight + timedelta(hours=2)).isoformat(),
            last_relevant_at=(midnight + timedelta(hours=3)).isoformat(),
        )
        # Old, used today (eligible for revisit)
        rid = await _seed(
            fresh_store, content="old used", summary="ou",
            created_at=(midnight - timedelta(days=10)).isoformat(),
            last_relevant_at=(midnight + timedelta(hours=4)).isoformat(),
        )
        cand = await gather_candidates(
            fresh_store, rng=random.Random(0),
        )
        revisit_ids = {m.id for m in cand.revisits}
        assert nid not in revisit_ids
        # rid is the only valid revisit candidate so it should be picked
        assert rid in revisit_ids


# ---------------------------------------------------------------------------
# cluster_candidates
# ---------------------------------------------------------------------------


class TestClusterCandidates:
    @pytest.mark.asyncio
    async def test_cluster_threshold_and_size(self, fresh_store):
        from boxbot.memory.dream import (
            CandidateSet, cluster_candidates, CLUSTER_THRESHOLD,
        )

        # Three nearly-identical embeddings → one cluster.
        v_a = _unit([1.0, 0.0, 0.0])
        v_b = _unit([0.95, 0.31, 0.0])  # very close to v_a
        v_c = _unit([0.9, 0.43, 0.0])   # close to v_a/v_b
        # Far: cosine to others < 0.7
        v_far = _unit([0.0, 0.0, 1.0])

        ids = []
        for i, (vec, content) in enumerate(
            [(v_a, "a"), (v_b, "b"), (v_c, "c"), (v_far, "far")]
        ):
            ids.append(await _seed(
                fresh_store, content=content, summary=content,
                embedding_vec=vec,
            ))

        # Pull memories back so we have real Memory dataclasses
        mems = []
        for mid in ids:
            mems.append(await fresh_store.get_memory_no_touch(mid))

        cand = CandidateSet(new_today=mems, revisits=[])
        clusters = await cluster_candidates(cand)
        # Expect one cluster of size 3, far one alone (excluded — size 1 not returned)
        assert any(len(c.memory_ids) == 3 for c in clusters)
        assert all(ids[3] not in c.memory_ids for c in clusters)

    @pytest.mark.asyncio
    async def test_below_threshold_no_cluster(self, fresh_store):
        from boxbot.memory.dream import (
            CandidateSet, cluster_candidates,
        )

        v_a = _unit([1.0, 0.0, 0.0])
        v_b = _unit([0.0, 1.0, 0.0])
        ids = [
            await _seed(fresh_store, content="a", summary="a", embedding_vec=v_a),
            await _seed(fresh_store, content="b", summary="b", embedding_vec=v_b),
        ]
        mems = [await fresh_store.get_memory_no_touch(i) for i in ids]
        clusters = await cluster_candidates(
            CandidateSet(new_today=mems, revisits=[]),
        )
        assert clusters == []


# ---------------------------------------------------------------------------
# find_near_duplicates
# ---------------------------------------------------------------------------


class TestFindNearDuplicates:
    @pytest.mark.asyncio
    async def test_pairs_above_threshold(self, fresh_store):
        from boxbot.memory.dream import (
            CandidateSet, find_near_duplicates, NEAR_DUP_THRESHOLD,
        )

        v_a = _unit([1.0, 0.0, 0.0])
        v_b = _unit([0.97, 0.243, 0.0])  # cos ≈ 0.97, well above 0.85
        v_c = _unit([0.0, 1.0, 0.0])     # orthogonal — not a pair

        ids = [
            await _seed(fresh_store, content="a", summary="a", embedding_vec=v_a),
            await _seed(fresh_store, content="b", summary="b", embedding_vec=v_b),
            await _seed(fresh_store, content="c", summary="c", embedding_vec=v_c),
        ]
        mems = [await fresh_store.get_memory_no_touch(i) for i in ids]
        cand = CandidateSet(new_today=mems, revisits=[])
        pairs = await find_near_duplicates(fresh_store, cand)

        assert len(pairs) == 1
        assert {pairs[0].memory_id_a, pairs[0].memory_id_b} == {ids[0], ids[1]}
        assert pairs[0].cosine >= NEAR_DUP_THRESHOLD

    @pytest.mark.asyncio
    async def test_co_injected_pairs_skipped(self, fresh_store):
        """Pairs that were both injected in the same conversation are NOT flagged."""
        from boxbot.memory.dream import (
            CandidateSet, find_near_duplicates,
        )

        v_a = _unit([1.0, 0.0, 0.0])
        v_b = _unit([0.97, 0.243, 0.0])
        id_a = await _seed(
            fresh_store, content="a", summary="a", embedding_vec=v_a,
        )
        id_b = await _seed(
            fresh_store, content="b", summary="b", embedding_vec=v_b,
        )
        # Create a conversation that lists BOTH as accessed memories.
        await fresh_store.create_conversation(
            channel="voice",
            participants=["Jacob"],
            summary="combined",
            accessed_memories=[id_a, id_b],
        )

        mems = [
            await fresh_store.get_memory_no_touch(id_a),
            await fresh_store.get_memory_no_touch(id_b),
        ]
        pairs = await find_near_duplicates(
            fresh_store, CandidateSet(new_today=mems, revisits=[]),
        )
        assert pairs == []

    @pytest.mark.asyncio
    async def test_operational_memories_excluded(self, fresh_store):
        from boxbot.memory.dream import (
            CandidateSet, find_near_duplicates,
        )

        v_a = _unit([1.0, 0.0, 0.0])
        v_b = _unit([0.97, 0.243, 0.0])
        id_a = await _seed(
            fresh_store, type_="operational",
            content="op a", summary="op a", embedding_vec=v_a,
        )
        id_b = await _seed(
            fresh_store, type_="operational",
            content="op b", summary="op b", embedding_vec=v_b,
        )
        mems = [
            await fresh_store.get_memory_no_touch(id_a),
            await fresh_store.get_memory_no_touch(id_b),
        ]
        pairs = await find_near_duplicates(
            fresh_store, CandidateSet(new_today=mems, revisits=[]),
        )
        assert pairs == []


# ---------------------------------------------------------------------------
# submit_dream_batch
# ---------------------------------------------------------------------------


class TestSubmitDreamBatch:
    @pytest.mark.asyncio
    async def test_batch_shape_and_pending_row(self, fresh_store, fake_client):
        from boxbot.memory.dream import (
            DEDUP_TOOL, NearDupPair, submit_dream_batch,
        )

        v_a = _unit([1.0, 0.0, 0.0])
        v_b = _unit([0.97, 0.243, 0.0])
        v_c = _unit([0.96, 0.28, 0.0])
        v_d = _unit([0.94, 0.34, 0.0])
        a = await _seed(fresh_store, content="a", summary="a", embedding_vec=v_a)
        b = await _seed(fresh_store, content="b", summary="b", embedding_vec=v_b)
        c = await _seed(fresh_store, content="c", summary="c", embedding_vec=v_c)
        d = await _seed(fresh_store, content="d", summary="d", embedding_vec=v_d)

        pairs = [
            NearDupPair(memory_id_a=a, memory_id_b=b, cosine=0.97),
            NearDupPair(memory_id_a=c, memory_id_b=d, cosine=0.95),
        ]
        batch_id = await submit_dream_batch(
            fake_client, fresh_store, pairs, candidate_ids=[a, b, c, d],
        )

        # Anthropic was called once with two requests.
        assert len(fake_client.create_calls) == 1
        requests = fake_client.create_calls[0]["requests"]
        assert len(requests) == 2
        for req in requests:
            assert req["custom_id"].startswith("dream-")
            tools = req["params"]["tools"]
            assert tools[0]["name"] == DEDUP_TOOL["name"]
            tool_choice = req["params"]["tool_choice"]
            assert tool_choice == {"type": "tool", "name": "dedup_decision"}

        # pending_dreams row written.
        row = await fresh_store.get_pending_dream(batch_id)
        assert row is not None
        assert row["status"] == "submitted"
        assert row["request_types"] == {"dedup": 2}
        assert set(row["candidate_ids"]) == {a, b, c, d}

    @pytest.mark.asyncio
    async def test_no_pairs_raises(self, fresh_store, fake_client):
        from boxbot.memory.dream import submit_dream_batch
        with pytest.raises(ValueError):
            await submit_dream_batch(fake_client, fresh_store, [])


# ---------------------------------------------------------------------------
# apply_dream_result
# ---------------------------------------------------------------------------


class TestApplyDreamResult:
    @pytest.mark.asyncio
    async def test_audit_only_does_not_mutate(self, fresh_store):
        from boxbot.memory.dream import (
            NearDupPair, apply_dream_result,
        )

        a = await _seed(fresh_store, content="aa", summary="aa")
        b = await _seed(fresh_store, content="bb", summary="bb")
        pair = NearDupPair(memory_id_a=a, memory_id_b=b, cosine=0.9)
        custom_id = "dream-test-0001"
        payload = {
            "decision": "merge_into_a",
            "merged_content": "aabb",
            "merged_summary": "aabb",
            "evidence": [a, b],
            "confidence": 0.95,
            "notes": "obvious dup",
        }
        msg = _StubMessage(content=[_StubBlock(
            type="tool_use", name="dedup_decision", input=payload,
        )])
        entries = [_StubResultEntry(custom_id, _StubResult("succeeded", message=msg))]

        result = await apply_dream_result(
            fresh_store, entries,
            batch_id="msgbatch_test",
            pairs_by_custom_id={custom_id: pair},
            audit_only=True,
        )
        assert result.audit_only is True
        assert result.applied_merges == 1  # counted, not actually applied

        # Memories still active, no mutations.
        mem_a = await fresh_store.get_memory_no_touch(a)
        mem_b = await fresh_store.get_memory_no_touch(b)
        assert mem_a.status == "active"
        assert mem_b.status == "active"
        assert mem_a.consolidated_by is None
        assert mem_b.consolidated_by is None
        assert mem_a.content == "aa"  # unchanged
        assert mem_b.content == "bb"

    @pytest.mark.asyncio
    async def test_apply_real_above_threshold(self, fresh_store):
        from boxbot.memory.dream import (
            NearDupPair, apply_dream_result,
        )

        a = await _seed(fresh_store, content="aa", summary="aa")
        b = await _seed(fresh_store, content="bb", summary="bb")
        pair = NearDupPair(memory_id_a=a, memory_id_b=b, cosine=0.9)
        custom_id = "dream-test-0001"
        payload = {
            "decision": "merge_into_a",
            "merged_content": "Combined: aa + bb",
            "merged_summary": "Combined",
            "evidence": [a, b],
            "confidence": 0.95,
            "notes": "obvious dup",
        }
        msg = _StubMessage(content=[_StubBlock(
            type="tool_use", name="dedup_decision", input=payload,
        )])
        entries = [_StubResultEntry(custom_id, _StubResult("succeeded", message=msg))]

        result = await apply_dream_result(
            fresh_store, entries,
            batch_id="msgbatch_apply",
            pairs_by_custom_id={custom_id: pair},
            audit_only=False,
        )
        assert result.applied_merges == 1
        mem_a = await fresh_store.get_memory_no_touch(a)
        mem_b = await fresh_store.get_memory_no_touch(b)
        assert mem_a.status == "active"
        assert mem_a.content == "Combined: aa + bb"
        assert mem_a.consolidated_by == "msgbatch_apply"
        assert mem_b.status == "invalidated"
        assert mem_b.superseded_by == a
        assert mem_b.consolidated_by == "msgbatch_apply"

    @pytest.mark.asyncio
    async def test_low_confidence_skipped(self, fresh_store):
        from boxbot.memory.dream import (
            NearDupPair, apply_dream_result,
        )

        a = await _seed(fresh_store, content="aa", summary="aa")
        b = await _seed(fresh_store, content="bb", summary="bb")
        pair = NearDupPair(memory_id_a=a, memory_id_b=b, cosine=0.9)
        payload = {
            "decision": "merge_into_a",
            "merged_content": "x",
            "merged_summary": "x",
            "evidence": [a, b],
            "confidence": 0.5,  # below threshold
            "notes": "",
        }
        msg = _StubMessage(content=[_StubBlock(
            type="tool_use", name="dedup_decision", input=payload,
        )])
        entries = [_StubResultEntry("c1", _StubResult("succeeded", message=msg))]

        result = await apply_dream_result(
            fresh_store, entries,
            batch_id="batch_low",
            pairs_by_custom_id={"c1": pair},
            audit_only=False,
        )
        assert result.applied_merges == 0
        assert result.skipped_low_confidence == 1
        mem_a = await fresh_store.get_memory_no_touch(a)
        mem_b = await fresh_store.get_memory_no_touch(b)
        assert mem_a.status == "active"
        assert mem_b.status == "active"
        assert mem_a.consolidated_by is None

    @pytest.mark.asyncio
    async def test_distinct_or_unsure_no_op(self, fresh_store):
        from boxbot.memory.dream import (
            NearDupPair, apply_dream_result,
        )

        a = await _seed(fresh_store, content="aa", summary="aa")
        b = await _seed(fresh_store, content="bb", summary="bb")
        entries = []
        for i, decision in enumerate(["distinct", "unsure"]):
            payload = {
                "decision": decision,
                "evidence": [a, b],
                "confidence": 0.95,
                "notes": "",
            }
            msg = _StubMessage(content=[_StubBlock(
                type="tool_use", name="dedup_decision", input=payload,
            )])
            entries.append(_StubResultEntry(
                f"c{i}", _StubResult("succeeded", message=msg),
            ))
        result = await apply_dream_result(
            fresh_store, entries,
            batch_id="batch_distinct",
            pairs_by_custom_id={},
            audit_only=False,
        )
        assert result.applied_merges == 0
        assert result.skipped_unsure_or_distinct == 2

    @pytest.mark.asyncio
    async def test_evidence_must_cite_both(self, fresh_store):
        """If evidence doesn't list both memory IDs, skip the merge."""
        from boxbot.memory.dream import (
            NearDupPair, apply_dream_result,
        )

        a = await _seed(fresh_store, content="aa", summary="aa")
        b = await _seed(fresh_store, content="bb", summary="bb")
        pair = NearDupPair(memory_id_a=a, memory_id_b=b, cosine=0.9)
        payload = {
            "decision": "merge_into_a",
            "merged_content": "x",
            "merged_summary": "x",
            "evidence": [a],  # missing b!
            "confidence": 0.95,
            "notes": "",
        }
        msg = _StubMessage(content=[_StubBlock(
            type="tool_use", name="dedup_decision", input=payload,
        )])
        entries = [_StubResultEntry("c1", _StubResult("succeeded", message=msg))]
        result = await apply_dream_result(
            fresh_store, entries,
            batch_id="b",
            pairs_by_custom_id={"c1": pair},
            audit_only=False,
        )
        assert result.applied_merges == 0
        mem_b = await fresh_store.get_memory_no_touch(b)
        assert mem_b.status == "active"


# ---------------------------------------------------------------------------
# DreamPoller resume + lifecycle
# ---------------------------------------------------------------------------


class TestDreamPoller:
    @pytest.mark.asyncio
    async def test_resume_on_boot(self, fresh_store, fake_client, tmp_path):
        from boxbot.memory.dream_poller import DreamPoller

        # Pre-existing submitted row from prior boot.
        await fresh_store.create_pending_dream(
            batch_id="msgbatch_pre",
            candidate_ids=["mid1"],
            request_types={"dedup": 1},
            summary="pre-existing",
        )
        fake_client._batches["msgbatch_pre"] = _StubBatch(
            "msgbatch_pre", status="in_progress",
        )

        poller = DreamPoller(fresh_store, fake_client, audit_only=True)
        await poller.start()
        try:
            # Poller should have seeded next_check for the pre-existing row.
            assert "msgbatch_pre" in poller._next_check
        finally:
            await poller.stop()

    @pytest.mark.asyncio
    async def test_full_lifecycle_audit_only(
        self, fresh_store, fake_client, tmp_path,
    ):
        from boxbot.memory.dream_poller import DreamPoller

        # Seed two memories that the dream batch will attempt to merge.
        a = await _seed(fresh_store, content="aa", summary="aa")
        b = await _seed(fresh_store, content="bb", summary="bb")

        # Simulate a submitted batch.
        await fresh_store.create_pending_dream(
            batch_id="msgbatch_done",
            candidate_ids=[a, b],
            request_types={"dedup": 1},
        )
        fake_client.end_with_decisions(
            "msgbatch_done",
            [(
                "dream-x-0000",
                {
                    "decision": "merge_into_a",
                    "merged_content": "merged",
                    "merged_summary": "merged",
                    "evidence": [a, b],
                    "confidence": 0.95,
                    "notes": "",
                },
            )],
        )

        # Patch WORKSPACE_DIR so the dream-log write is sandboxed.
        with patch(
            "boxbot.memory.dream.WORKSPACE_DIR", tmp_path / "workspace",
        ):
            poller = DreamPoller(fresh_store, fake_client, audit_only=True)
            await poller.start()
            try:
                # Force immediate poll
                poller._next_check["msgbatch_done"] = 0.0
                await poller._sweep_once()
            finally:
                await poller.stop()

        # In audit_only mode, memories must be unchanged.
        mem_a = await fresh_store.get_memory_no_touch(a)
        mem_b = await fresh_store.get_memory_no_touch(b)
        assert mem_a.status == "active"
        assert mem_b.status == "active"

        # pending_dreams row marked applied.
        row = await fresh_store.get_pending_dream("msgbatch_done")
        assert row["status"] == "applied"

        # Cost recorded.
        cost = await fresh_store.cost_summary(days=1)
        assert "dream" in cost
        assert cost["dream"] > 0

    @pytest.mark.asyncio
    async def test_full_lifecycle_apply_mode(
        self, fresh_store, fake_client, tmp_path,
    ):
        from boxbot.memory.dream_poller import DreamPoller

        a = await _seed(fresh_store, content="aa", summary="aa")
        b = await _seed(fresh_store, content="bb", summary="bb")
        await fresh_store.create_pending_dream(
            batch_id="msgbatch_apply",
            candidate_ids=[a, b],
            request_types={"dedup": 1},
        )
        fake_client.end_with_decisions(
            "msgbatch_apply",
            [(
                "dream-y-0000",
                {
                    "decision": "merge_into_a",
                    "merged_content": "MERGED",
                    "merged_summary": "MERGED",
                    "evidence": [a, b],
                    "confidence": 0.95,
                    "notes": "",
                },
            )],
        )

        with patch(
            "boxbot.memory.dream.WORKSPACE_DIR", tmp_path / "workspace",
        ):
            poller = DreamPoller(fresh_store, fake_client, audit_only=False)
            await poller.start()
            try:
                poller._next_check["msgbatch_apply"] = 0.0
                await poller._sweep_once()
            finally:
                await poller.stop()

        # The poller doesn't have pairs_by_custom_id (lost across boots),
        # so it relies on evidence to identify the pair. Check that the
        # mutation went through.
        mem_a = await fresh_store.get_memory_no_touch(a)
        mem_b = await fresh_store.get_memory_no_touch(b)
        # The decision was merge_into_a → a is keeper, b is invalidated.
        assert mem_a.consolidated_by == "msgbatch_apply"
        assert mem_b.status == "invalidated"
        assert mem_b.superseded_by == a


# ---------------------------------------------------------------------------
# run_dream_cycle (integration)
# ---------------------------------------------------------------------------


class TestRunDreamCycle:
    @pytest.mark.asyncio
    async def test_no_pairs_no_batch(self, fresh_store, fake_client, tmp_path):
        """If there's nothing to dedup, no batch is submitted."""
        from boxbot.memory.dream import run_dream_cycle

        # One memory only → can't pair.
        await _seed(fresh_store, content="lonely", summary="lonely")
        with patch(
            "boxbot.memory.dream.WORKSPACE_DIR", tmp_path / "workspace",
        ):
            summary = await run_dream_cycle(
                fresh_store, fake_client, audit_only=True,
            )
        assert summary["batch_id"] is None
        assert summary["near_dup_pairs"] == 0
        # No batch submitted to Anthropic.
        assert fake_client.create_calls == []

    @pytest.mark.asyncio
    async def test_with_pair_submits_batch_and_writes_log(
        self, fresh_store, fake_client, tmp_path,
    ):
        from boxbot.memory.dream import run_dream_cycle

        v_a = _unit([1.0, 0.0, 0.0])
        v_b = _unit([0.97, 0.243, 0.0])  # cos > 0.85
        await _seed(fresh_store, content="a", summary="a", embedding_vec=v_a)
        await _seed(fresh_store, content="b", summary="b", embedding_vec=v_b)
        workspace = tmp_path / "workspace"
        with patch("boxbot.memory.dream.WORKSPACE_DIR", workspace):
            summary = await run_dream_cycle(
                fresh_store, fake_client, audit_only=True,
            )
        assert summary["batch_id"] is not None
        assert summary["near_dup_pairs"] >= 1
        # Dream log written.
        log_files = list(
            (workspace / "notes" / "system" / "dream-log").glob("*.md")
        )
        assert log_files, "Dream log not written"
        text = log_files[0].read_text()
        assert "Dream cycle" in text
        assert "audit-only" in text


# ---------------------------------------------------------------------------
# Undo script
# ---------------------------------------------------------------------------


class TestUndoScript:
    @pytest.mark.asyncio
    async def test_undo_idempotent(self, tmp_path):
        """Running undo twice on the same batch is a no-op the second time.

        We invoke the undo script's core ``_undo`` coroutine directly
        rather than spawning a subprocess so we (a) keep dependency
        loading consistent with the test environment and (b) can
        reset the MemoryStore singleton patches between invocations.
        """
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "undo_last_dream.py"
        assert script_path.exists()

        # Set up a store with applied dream batch.
        import boxbot.core  # noqa: F401
        from boxbot.memory.store import MemoryStore

        db_path = tmp_path / "memory.db"
        sys_mem_path = tmp_path / "system.md"
        batch_id = "msgbatch_undo_test"
        with patch("boxbot.memory.store.SYSTEM_MEMORY_PATH", sys_mem_path):
            store = MemoryStore(db_path=db_path)
            await store.initialize()
            try:
                a = await store.create_memory(
                    type="person", content="a", summary="a",
                )
                b = await store.create_memory(
                    type="person", content="b", summary="b",
                )
                # Simulate a real dream apply: invalidate b, stamp both
                # with consolidated_by.
                await store.create_pending_dream(
                    batch_id=batch_id,
                    candidate_ids=[a, b],
                    request_types={"dedup": 1},
                )
                await store.invalidate_memory(
                    b, invalidated_by=batch_id, superseded_by=a,
                )
                await store.set_dream_audit_fields(
                    a, consolidated_by=batch_id,
                )
                await store.set_dream_audit_fields(
                    b, consolidated_by=batch_id,
                )
                await store.mark_dream_applied(batch_id)
            finally:
                await store.close()

        # Load the script module so we can call its ``_undo`` coroutine.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_undo_script", str(script_path),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Patch the script's MemoryStore default path via DB_PATH override.
        # The script's _undo() uses MemoryStore() with no args, which
        # picks up DB_PATH from the store module — patch that.
        with patch("boxbot.memory.store.DB_PATH", db_path), \
             patch("boxbot.memory.store.SYSTEM_MEMORY_PATH", sys_mem_path):
            rc1 = await mod._undo(batch_id=None, dry_run=False)
            assert rc1 == 0
            rc2 = await mod._undo(batch_id=None, dry_run=False)
            assert rc2 == 0

        # Verify b is now active again.
        with patch("boxbot.memory.store.SYSTEM_MEMORY_PATH", sys_mem_path):
            store = MemoryStore(db_path=db_path)
            await store.initialize()
            try:
                mems = await store.list_memories(
                    limit=10, status=None,
                )
                # The Memory dataclass list filtered by status default
                # excludes archived/invalidated; pull all explicitly.
                cur = await store.db.execute(
                    "SELECT id, status, consolidated_by, content FROM memories"
                )
                rows = await cur.fetchall()
                by_content = {r["content"]: dict(r) for r in rows}
                mem_b = by_content.get("b")
                assert mem_b is not None
                assert mem_b["status"] == "active"
                assert mem_b["consolidated_by"] is None
                mem_a = by_content.get("a")
                assert mem_a["consolidated_by"] is None
                # pending_dreams row marked failed (so it can't be undone again).
                row = await store.get_pending_dream(batch_id)
                assert row["status"] == "failed"
            finally:
                await store.close()
