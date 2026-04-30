"""End-to-end tests for the new batch-driven extraction pipeline.

Covers: pending_extractions CRUD, transcript search, batch poller resume
on boot, parse + apply success path, and per-request error handling.
The Anthropic client is mocked so tests run offline.
"""

from __future__ import annotations

import asyncio
import json
import types
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Mocks of the Anthropic SDK shape we depend on
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
    """In-memory stand-in for ``anthropic.AsyncAnthropic`` that records
    submitted batches and returns programmable retrieve/results."""

    def __init__(self):
        self._next_id = 0
        self._batches: dict[str, _StubBatch] = {}
        self._results: dict[str, list[_StubResultEntry]] = {}
        self.create_calls: list[dict] = []
        # Build the namespaced API surface
        self.messages = types.SimpleNamespace(
            batches=types.SimpleNamespace(
                create=self._create,
                retrieve=self._retrieve,
                results=self._results_iter,
            )
        )

    async def _create(self, *, requests):
        self._next_id += 1
        bid = f"msgbatch_test_{self._next_id}"
        self._batches[bid] = _StubBatch(bid, status="in_progress")
        self.create_calls.append({"id": bid, "requests": requests})
        return _StubBatch(bid, status="in_progress")

    async def _retrieve(self, batch_id):
        return self._batches[batch_id]

    async def _results_iter(self, batch_id):
        return _AsyncIter(self._results.get(batch_id, []))

    # -- test helpers --

    def end_with_success(self, batch_id, custom_id, payload, *, usage=None):
        msg = _StubMessage(
            content=[
                _StubBlock(type="tool_use", name="emit_extraction", input=payload),
            ],
            usage=usage,
        )
        self._results[batch_id] = [
            _StubResultEntry(custom_id, _StubResult("succeeded", message=msg)),
        ]
        self._batches[batch_id] = _StubBatch(batch_id, status="ended")

    def end_with_error(self, batch_id, custom_id, error_payload):
        self._results[batch_id] = [
            _StubResultEntry(
                custom_id,
                _StubResult("errored", error=error_payload),
            ),
        ]
        self._batches[batch_id] = _StubBatch(batch_id, status="ended")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def fresh_store(tmp_path):
    from boxbot.memory.store import MemoryStore
    from unittest.mock import patch
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path=db_path)
    sys_mem_path = tmp_path / "system.md"
    with patch("boxbot.memory.store.SYSTEM_MEMORY_PATH", sys_mem_path):
        await store.initialize()
        yield store
        await store.close()


@pytest.fixture
def fake_client():
    return FakeAnthropicClient()


@pytest.fixture
def sample_payload():
    """A realistic extraction-result payload."""
    return {
        "conversation_summary": {
            "topics": ["food", "diet"],
            "summary": "Jacob said he's vegetarian.",
        },
        "extracted_memories": [
            {
                "type": "person",
                "person": "Jacob",
                "content": "Jacob is vegetarian as of 2026-04-29.",
                "summary": "Jacob is vegetarian",
                "tags": ["food", "diet"],
            }
        ],
        "invalidations": [],
        "system_memory_updates": [],
    }


# ---------------------------------------------------------------------------
# Pending extraction CRUD
# ---------------------------------------------------------------------------


class TestPendingExtractions:
    @pytest.mark.asyncio
    async def test_create_and_get(self, fresh_store):
        await fresh_store.create_pending_extraction(
            conversation_id="conv_aaa",
            transcript="hello",
            accessed_memory_ids=["m1", "m2"],
            channel="whatsapp",
            participants=["Jacob"],
            started_at="2026-04-29T10:00:00",
        )
        row = await fresh_store.get_pending_extraction("conv_aaa")
        assert row is not None
        assert row.status == "queued"
        assert row.accessed_memory_ids == ["m1", "m2"]
        assert row.transcript == "hello"

    @pytest.mark.asyncio
    async def test_status_transitions(self, fresh_store):
        await fresh_store.create_pending_extraction(
            conversation_id="conv_aaa",
            transcript="hi",
            accessed_memory_ids=[],
            channel="voice",
            participants=["Jacob"],
            started_at="2026-04-29T10:00:00",
        )
        await fresh_store.mark_pending_submitted("conv_aaa", "msgbatch_1")
        row = await fresh_store.get_pending_extraction("conv_aaa")
        assert row.status == "submitted"
        assert row.batch_id == "msgbatch_1"
        assert row.attempts == 1

        await fresh_store.mark_pending_applied("conv_aaa")
        assert (await fresh_store.get_pending_extraction("conv_aaa")).status == "applied"

        await fresh_store.mark_pending_failed("conv_aaa", "test error")
        row = await fresh_store.get_pending_extraction("conv_aaa")
        assert row.status == "failed"
        assert row.error == "test error"

    @pytest.mark.asyncio
    async def test_list_by_status(self, fresh_store):
        for cid, status_action in [
            ("a", None),  # leave queued
            ("b", "submit"),
            ("c", "submit"),
            ("d", "apply"),
        ]:
            await fresh_store.create_pending_extraction(
                conversation_id=cid, transcript="x",
                accessed_memory_ids=[], channel="voice",
                participants=["Jacob"],
                started_at=f"2026-04-29T10:00:0{ord(cid[0])-ord('a')}",
            )
            if status_action == "submit":
                await fresh_store.mark_pending_submitted(cid, f"b_{cid}")
            elif status_action == "apply":
                await fresh_store.mark_pending_submitted(cid, f"b_{cid}")
                await fresh_store.mark_pending_applied(cid)

        queued = await fresh_store.list_pending_extractions(status="queued")
        submitted = await fresh_store.list_pending_extractions(status="submitted")
        applied = await fresh_store.list_pending_extractions(status="applied")
        assert {r.conversation_id for r in queued} == {"a"}
        assert {r.conversation_id for r in submitted} == {"b", "c"}
        assert {r.conversation_id for r in applied} == {"d"}

    @pytest.mark.asyncio
    async def test_purge_expired_transcripts(self, fresh_store):
        await fresh_store.create_pending_extraction(
            conversation_id="conv_old",
            transcript="old text",
            accessed_memory_ids=[],
            channel="voice", participants=["Jacob"],
            started_at="2026-04-15T10:00:00",
        )
        # Force expiry
        await fresh_store.db.execute(
            "UPDATE pending_extractions SET transcript_purge_at='2000-01-01' "
            "WHERE conversation_id=?",
            ("conv_old",),
        )
        await fresh_store.db.commit()
        purged = await fresh_store.purge_expired_transcripts()
        assert purged == 1
        assert (await fresh_store.get_transcript("conv_old")) is None
        # Row still present, only transcript nulled
        row = await fresh_store.get_pending_extraction("conv_old")
        assert row is not None
        assert row.transcript is None

    @pytest.mark.asyncio
    async def test_search_transcripts(self, fresh_store):
        await fresh_store.create_pending_extraction(
            conversation_id="conv1",
            transcript="we discussed the dining percentage",
            accessed_memory_ids=[],
            channel="whatsapp", participants=["Jacob"],
            started_at=datetime.utcnow().isoformat(),
        )
        await fresh_store.create_pending_extraction(
            conversation_id="conv2",
            transcript="totally different topic about cars",
            accessed_memory_ids=[],
            channel="whatsapp", participants=["Jacob"],
            started_at=datetime.utcnow().isoformat(),
        )
        hits = await fresh_store.search_transcripts("dining")
        assert len(hits) == 1
        assert hits[0][0] == "conv1"
        assert "dining" in hits[0][2]


# ---------------------------------------------------------------------------
# Cost log
# ---------------------------------------------------------------------------


class TestCostLog:
    @pytest.mark.asyncio
    async def test_record_and_summarize(self, fresh_store):
        await fresh_store.record_cost(
            purpose="extraction", model="claude-sonnet-4-6",
            input_tokens=1000, output_tokens=200, is_batch=True,
            cost_usd=0.0027,
        )
        await fresh_store.record_cost(
            purpose="extraction", model="claude-sonnet-4-6",
            input_tokens=2000, output_tokens=500, is_batch=True,
            cost_usd=0.0067,
        )
        await fresh_store.record_cost(
            purpose="rerank", model="claude-haiku-4-5",
            input_tokens=500, output_tokens=100, is_batch=False,
            cost_usd=0.0010,
        )
        summary = await fresh_store.cost_summary(days=7)
        assert summary == pytest.approx(
            {"extraction": 0.0094, "rerank": 0.0010}, rel=1e-6,
        )


# ---------------------------------------------------------------------------
# Extraction prompt + parsing
# ---------------------------------------------------------------------------


class TestExtractionParser:
    def test_parse_full_payload(self, sample_payload):
        from boxbot.memory.extraction import parse_extraction_result
        msg = _StubMessage(
            content=[_StubBlock(type="tool_use", name="emit_extraction",
                                input=sample_payload)]
        )
        result = parse_extraction_result(msg)
        assert result.conversation_summary.summary == "Jacob said he's vegetarian."
        assert len(result.extracted_memories) == 1
        assert result.extracted_memories[0].person == "Jacob"
        assert result.extracted_memories[0].action == "create"

    def test_parse_missing_tool_call_raises(self):
        from boxbot.memory.extraction import parse_extraction_result
        msg = _StubMessage(content=[_StubBlock(type="text", text="oops")])
        with pytest.raises(ValueError):
            parse_extraction_result(msg)

    def test_parse_invalidation_with_replacement(self):
        from boxbot.memory.extraction import parse_extraction_result
        payload = {
            "conversation_summary": {"topics": [], "summary": "x"},
            "invalidations": [
                {
                    "memory_id": "mem-541",
                    "reason": "explicit retraction",
                    "replacement": {
                        "type": "person", "person": "Jacob",
                        "content": "Jacob eats meat.", "summary": "Jacob eats meat",
                        "tags": ["food"],
                    },
                }
            ],
        }
        msg = _StubMessage(
            content=[_StubBlock(type="tool_use", name="emit_extraction",
                                input=payload)]
        )
        result = parse_extraction_result(msg)
        assert len(result.invalidations) == 1
        inv = result.invalidations[0]
        assert inv.memory_id == "mem-541"
        assert inv.replacement is not None
        assert inv.replacement.content == "Jacob eats meat."

    def test_cost_compute_batch_discount(self):
        from boxbot.memory.extraction import compute_cost
        # 10K input + 2K output sonnet, batch
        # = (10K * $3) + (2K * $15) per MTok = $0.030 + $0.030 = $0.060 standard
        # batch = 50% off = $0.030
        c = compute_cost(
            "claude-sonnet-4-6",
            input_tokens=10_000, output_tokens=2_000,
            is_batch=True,
        )
        assert c == pytest.approx(0.030, rel=1e-6)


# ---------------------------------------------------------------------------
# Batch poller — submission, polling, success path
# ---------------------------------------------------------------------------


class TestBatchPoller:
    @pytest.mark.asyncio
    async def test_submit_marks_row_submitted(self, fresh_store, fake_client):
        from boxbot.memory.batch_poller import BatchPoller

        await fresh_store.create_pending_extraction(
            conversation_id="conv_x",
            transcript="[Jacob]: hi\n[boxBot]: hello",
            accessed_memory_ids=[],
            channel="voice",
            participants=["Jacob"],
            started_at="2026-04-29T10:00:00",
        )
        poller = BatchPoller(fresh_store, fake_client)
        row = await fresh_store.get_pending_extraction("conv_x")
        await poller.submit(row)

        # Row should be marked submitted with the fake's batch id.
        row = await fresh_store.get_pending_extraction("conv_x")
        assert row.status == "submitted"
        assert row.batch_id.startswith("msgbatch_test_")
        # Anthropic was called exactly once with our custom_id.
        assert len(fake_client.create_calls) == 1
        req = fake_client.create_calls[0]["requests"][0]
        assert req["custom_id"] == "conv_x"

    @pytest.mark.asyncio
    async def test_full_lifecycle_success(
        self, fresh_store, fake_client, sample_payload,
    ):
        """Submit, end the batch with success, run one sweep, verify
        memories were created and the row is marked applied."""
        from boxbot.memory.batch_poller import BatchPoller

        await fresh_store.create_pending_extraction(
            conversation_id="conv_x",
            transcript="[Jacob]: I'm vegetarian.\n[boxBot]: noted.",
            accessed_memory_ids=[],
            channel="whatsapp",
            participants=["Jacob"],
            started_at="2026-04-29T10:00:00",
        )
        poller = BatchPoller(fresh_store, fake_client)
        row = await fresh_store.get_pending_extraction("conv_x")
        await poller.submit(row)
        batch_id = (await fresh_store.get_pending_extraction("conv_x")).batch_id

        # Provide a successful result with usage info.
        usage = MagicMock()
        usage.input_tokens = 800
        usage.output_tokens = 150
        usage.cache_read_input_tokens = 0
        usage.cache_creation_input_tokens = 0
        fake_client.end_with_success(batch_id, "conv_x", sample_payload, usage=usage)

        # Force the poller's next_check time to "now" so a single sweep
        # immediately polls. We do this by zeroing out _next_check.
        poller._next_check["conv_x"] = 0.0
        await poller._sweep_once()

        # Row should be marked applied.
        row = await fresh_store.get_pending_extraction("conv_x")
        assert row.status == "applied", row.error

        # A memory should exist.
        memories = await fresh_store.list_memories(limit=5)
        assert any(m.person == "Jacob" and "vegetarian" in m.content
                   for m in memories), [m.content for m in memories]

        # Cost should be recorded.
        summary = await fresh_store.cost_summary(days=7)
        assert "extraction" in summary
        assert summary["extraction"] > 0

    @pytest.mark.asyncio
    async def test_errored_result_marks_failed(
        self, fresh_store, fake_client,
    ):
        from boxbot.memory.batch_poller import BatchPoller

        await fresh_store.create_pending_extraction(
            conversation_id="conv_err",
            transcript="x",
            accessed_memory_ids=[], channel="voice",
            participants=["Jacob"], started_at="2026-04-29T10:00:00",
        )
        poller = BatchPoller(fresh_store, fake_client)
        row = await fresh_store.get_pending_extraction("conv_err")
        await poller.submit(row)
        batch_id = (await fresh_store.get_pending_extraction("conv_err")).batch_id

        fake_client.end_with_error(batch_id, "conv_err", {"type": "api_error"})

        poller._next_check["conv_err"] = 0.0
        await poller._sweep_once()

        row = await fresh_store.get_pending_extraction("conv_err")
        assert row.status == "failed"
        assert "errored" in (row.error or "")

    @pytest.mark.asyncio
    async def test_resume_on_boot(
        self, fresh_store, fake_client, sample_payload,
    ):
        """Simulate a crash: queued + submitted rows from a prior boot.
        Poller.start should re-submit queued and pick up submitted."""
        from boxbot.memory.batch_poller import BatchPoller

        # A queued (un-submitted) row from prior boot
        await fresh_store.create_pending_extraction(
            conversation_id="conv_q",
            transcript="queued transcript",
            accessed_memory_ids=[], channel="voice",
            participants=["Jacob"], started_at="2026-04-29T10:00:00",
        )
        # A submitted row (mid-flight) from prior boot
        await fresh_store.create_pending_extraction(
            conversation_id="conv_s",
            transcript="submitted transcript",
            accessed_memory_ids=[], channel="voice",
            participants=["Jacob"], started_at="2026-04-29T11:00:00",
        )
        # Pre-create the batch in the fake so retrieve works.
        # (BatchPoller.start will not re-submit conv_s since it already
        # has status submitted, but it tracks it for polling.)
        await fresh_store.mark_pending_submitted("conv_s", "msgbatch_pre_existing")
        fake_client._batches["msgbatch_pre_existing"] = _StubBatch(
            "msgbatch_pre_existing", status="in_progress",
        )

        poller = BatchPoller(fresh_store, fake_client)
        await poller.start()
        try:
            # conv_q should now be submitted
            q_row = await fresh_store.get_pending_extraction("conv_q")
            assert q_row.status == "submitted"
            assert q_row.batch_id.startswith("msgbatch_test_")

            # conv_s should still be tracked (polled by the loop)
            s_row = await fresh_store.get_pending_extraction("conv_s")
            assert s_row.status == "submitted"
            assert s_row.batch_id == "msgbatch_pre_existing"
        finally:
            await poller.stop()


# ---------------------------------------------------------------------------
# Transcript search via search_memories
# ---------------------------------------------------------------------------


class TestTranscriptSearch:
    @pytest.mark.asyncio
    async def test_get_by_conversation_id(self, fresh_store):
        from boxbot.memory.search import search_memories

        await fresh_store.create_pending_extraction(
            conversation_id="conv_t",
            transcript="[Jacob]: how did we decide on dining?\n[boxBot]: 41% Q1",
            accessed_memory_ids=[],
            channel="whatsapp",
            participants=["Jacob"],
            started_at=datetime.utcnow().isoformat(),
        )
        result = await search_memories(
            fresh_store, mode="transcript", conversation_id="conv_t",
        )
        assert "transcript" in result
        assert "41%" in result["transcript"]
        assert result["channel"] == "whatsapp"

    @pytest.mark.asyncio
    async def test_substring_search(self, fresh_store):
        from boxbot.memory.search import search_memories

        await fresh_store.create_pending_extraction(
            conversation_id="conv_a",
            transcript="we talked about expenses",
            accessed_memory_ids=[],
            channel="whatsapp", participants=["Jacob"],
            started_at=datetime.utcnow().isoformat(),
        )
        await fresh_store.create_pending_extraction(
            conversation_id="conv_b",
            transcript="we talked about kids",
            accessed_memory_ids=[],
            channel="whatsapp", participants=["Jacob"],
            started_at=datetime.utcnow().isoformat(),
        )
        result = await search_memories(
            fresh_store, mode="transcript", query="expenses",
        )
        assert "matches" in result
        assert len(result["matches"]) == 1
        assert result["matches"][0]["conversation_id"] == "conv_a"

    @pytest.mark.asyncio
    async def test_purged_returns_error(self, fresh_store):
        from boxbot.memory.search import search_memories

        await fresh_store.create_pending_extraction(
            conversation_id="conv_p",
            transcript="x",
            accessed_memory_ids=[],
            channel="whatsapp", participants=["Jacob"],
            started_at="2026-04-29T10:00:00",
        )
        # Force purge
        await fresh_store.db.execute(
            "UPDATE pending_extractions SET transcript_purge_at='2000-01-01' "
            "WHERE conversation_id=?",
            ("conv_p",),
        )
        await fresh_store.db.commit()
        await fresh_store.purge_expired_transcripts()

        result = await search_memories(
            fresh_store, mode="transcript", conversation_id="conv_p",
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# inject_memories returns ids
# ---------------------------------------------------------------------------


class TestInjectionReturnsIds:
    @pytest.mark.asyncio
    async def test_returns_tuple_with_ids(self, fresh_store):
        from boxbot.memory.retrieval import inject_memories

        # Seed a memory so injection has something to find.
        mid = await fresh_store.create_memory(
            type="person", person="Jacob",
            content="Jacob loves chicken pesto pizza.",
            summary="Jacob's pizza preference: chicken pesto",
            tags=["food", "preference"],
        )
        block, ids = await inject_memories(
            fresh_store, person="Jacob", utterance="what should I eat tonight?",
        )
        assert isinstance(block, str)
        assert isinstance(ids, list)
        assert mid in ids

    @pytest.mark.asyncio
    async def test_empty_when_no_results(self, fresh_store):
        from boxbot.memory.retrieval import inject_memories
        block, ids = await inject_memories(
            fresh_store, person="Nobody", utterance="completely unrelated",
        )
        assert block == ""
        assert ids == []
