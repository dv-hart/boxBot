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


class TestExtractionPromptContract:
    """Lock in the new lifecycle rules so future prompt edits don't
    silently regress them. These are the behaviours we want the model
    to follow — if a rule needs to change, update both the prompt and
    this test, intentionally."""

    def test_prompt_directs_state_assertions_to_todos(self):
        """Open-issue state ('reauth pending', 'X is broken') must be
        steered to todos, not memory. This was the root of the
        'calendar still down' memory loop."""
        from boxbot.memory.extraction import EXTRACTION_SYSTEM_PROMPT
        text = EXTRACTION_SYSTEM_PROMPT.lower()
        assert "todo" in text
        assert "in-flight" in text or "in flight" in text or "open issue" in text
        # Must explicitly call out the bad patterns we kept seeing
        assert "reauth" in text or "token expired" in text
        assert "broken" in text and "pending" in text

    def test_prompt_restricts_operational_to_workspace_pointers(self):
        """Operational was the noise bucket. Restrict it to pointing
        at workspace artifacts the agent just created."""
        from boxbot.memory.extraction import EXTRACTION_SYSTEM_PROMPT
        text = EXTRACTION_SYSTEM_PROMPT.lower()
        assert "operational" in text
        assert "workspace" in text
        # Activity-log mention warning the model NOT to use that pattern
        assert "activity log" in text or "activity-log" in text or "play-by-play" in text

    def test_prompt_separates_history_from_memory(self):
        """History (what BB did) lives in conversation log + workspace,
        not in memory. The prompt must make this distinction
        explicitly."""
        from boxbot.memory.extraction import EXTRACTION_SYSTEM_PROMPT
        text = EXTRACTION_SYSTEM_PROMPT.lower()
        assert "conversation log" in text or "conversation-log" in text
        assert "rings a bell" in text or "recognition" in text

    def test_prompt_keeps_invalidation_rules_conservative(self):
        """Until step 4 lands the full injection block, the extraction
        model can only see memory IDs. Invalidation rules must remain
        conservative."""
        from boxbot.memory.extraction import EXTRACTION_SYSTEM_PROMPT
        text = EXTRACTION_SYSTEM_PROMPT.lower()
        assert "only invalidate memories listed" in text
        assert "do not invalidate based on inference" in text

    def test_prompt_conversation_summary_is_a_receipt(self):
        """The conversation_summary must be a topic index entry, not a
        recap of conclusions. This is the earworm guard: summaries get
        injected into future runs, so a summary that restates the
        agent's own claims becomes a self-replicating wrong belief."""
        from boxbot.memory.extraction import EXTRACTION_SYSTEM_PROMPT
        text = EXTRACTION_SYSTEM_PROMPT.lower()
        # The receipt framing must be present and explicit
        assert "receipt" in text
        assert "earworm" in text or "self-replicating" in text
        # Must steer toward topic, away from conclusion/assertion
        assert "topic" in text
        assert "editorialise" in text or "editorialize" in text

    def test_summary_schema_description_says_receipt_not_recap(self):
        """The tool schema's summary field description must also carry
        the receipt framing — the model reads the schema, not just the
        system prompt."""
        from boxbot.memory.extraction import EXTRACTION_TOOL
        desc = (
            EXTRACTION_TOOL["input_schema"]["properties"]
            ["conversation_summary"]["properties"]["summary"]["description"]
        ).lower()
        assert "receipt" in desc
        assert "discussed" in desc
        # Explicitly warns against restating conclusions/claims
        assert "concluded" in desc or "assert" in desc


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
            fresh_store, person="Jacob", utterance="chicken pesto pizza",
        )
        assert isinstance(block, str)
        assert isinstance(ids, list)
        assert mid in ids

    @pytest.mark.asyncio
    async def test_name_alone_does_not_inject(self, fresh_store):
        """Nothing filters this path, so it must not fill on a shared
        token. Without an embedder the speaker's name matched every
        memory about them at the top of a normalized score."""
        from boxbot.memory.retrieval import inject_memories

        for content in ("Jacob's car insurance renews in March.",
                        "Jacob keeps the spare key under the blue pot."):
            await fresh_store.create_memory(
                type="person", person="Jacob",
                content=content, summary=content,
            )
        _block, ids = await inject_memories(
            fresh_store, person="Jacob", utterance="what should I eat tonight?",
        )
        assert ids == []

    @pytest.mark.asyncio
    async def test_empty_when_no_results(self, fresh_store):
        from boxbot.memory.retrieval import inject_memories
        block, ids = await inject_memories(
            fresh_store, person="Nobody", utterance="completely unrelated",
        )
        assert block == ""
        assert ids == []


# ---------------------------------------------------------------------------
# Thread-append front-end (OpenAI loop)
# ---------------------------------------------------------------------------


def _thread_reply(payload: dict) -> str:
    """Wrap an extraction payload the way the live model returns it:
    internal-notes JSON with the payload as a string in ``thought``."""
    return json.dumps({
        "thought": json.dumps(payload),
        "observations": [],
        "final_turn": True,
    })


_MINIMAL_PAYLOAD = {
    "conversation_summary": {
        "topics": ["locks"],
        "summary": "Discussed locking the front door.",
    },
    "extracted_memories": [],
    "invalidations": [],
    "system_memory_updates": [],
}


class TestBuildThreadExtractionMessage:
    def test_contains_metadata_policy_and_output_contract(self):
        from boxbot.memory.extraction import build_thread_extraction_message

        msg = build_thread_extraction_message(
            injected_memories_block="[Active Memories]\n- mem_1: Jacob likes pizza",
            channel="voice",
            participants=["BB", "Jacob"],
            started_at="2026-08-29T21:00:00",
        )
        assert "channel=voice" in msg
        assert "BB, Jacob" in msg
        assert "mem_1" in msg
        # Shared policy rode along (spot-check the earworm + todo rules)
        assert "RECEIPT" in msg
        assert "Todos own in-flight state" in msg
        # Output contract: notes shape, payload in thought, no tools
        assert "`thought`" in msg
        assert "Do NOT call any tools" in msg
        assert "conversation_summary" in msg

    def test_no_transcript_section(self):
        """The conversation is already in the (cached) prompt above —
        including a transcript would double it."""
        from boxbot.memory.extraction import build_thread_extraction_message

        msg = build_thread_extraction_message(
            injected_memories_block="",
            channel="voice",
            participants=[],
            started_at="2026-08-29T21:00:00",
        )
        assert "[Transcript]" not in msg
        assert "(none injected)" in msg


class TestParseThreadExtractionContent:
    def test_notes_wrapped_payload(self):
        from boxbot.memory.extraction import parse_thread_extraction_content

        payload = dict(_MINIMAL_PAYLOAD)
        payload["invalidations"] = [
            {"memory_id": "mem_0412", "reason": "lock verified working"},
        ]
        result = parse_thread_extraction_content(_thread_reply(payload))
        assert result.conversation_summary.topics == ["locks"]
        assert len(result.invalidations) == 1
        assert result.invalidations[0].memory_id == "mem_0412"

    def test_payload_directly_at_top_level(self):
        from boxbot.memory.extraction import parse_thread_extraction_content

        result = parse_thread_extraction_content(json.dumps(_MINIMAL_PAYLOAD))
        assert result.conversation_summary.summary.startswith("Discussed")

    def test_thought_already_decoded_to_object(self):
        from boxbot.memory.extraction import parse_thread_extraction_content

        content = json.dumps({
            "thought": _MINIMAL_PAYLOAD, "observations": [], "final_turn": True,
        })
        result = parse_thread_extraction_content(content)
        assert result.conversation_summary.topics == ["locks"]

    def test_markdown_fenced_reply(self):
        from boxbot.memory.extraction import parse_thread_extraction_content

        fenced = "```json\n" + json.dumps(_MINIMAL_PAYLOAD) + "\n```"
        result = parse_thread_extraction_content(fenced)
        assert result.conversation_summary.topics == ["locks"]

    @pytest.mark.parametrize("bad", [
        "",
        "The door is locked.",
        json.dumps({"thought": "no json here", "observations": []}),
        json.dumps(["not", "an", "object"]),
        json.dumps({"observations": ["missing thought and summary"]}),
    ])
    def test_unparseable_replies_raise(self, bad):
        from boxbot.memory.extraction import parse_thread_extraction_content

        with pytest.raises(ValueError):
            parse_thread_extraction_content(bad)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=content, tool_calls=None),
            ),
        ]
        self.usage = types.SimpleNamespace(
            prompt_tokens=7000,
            completion_tokens=90,
            prompt_tokens_details=types.SimpleNamespace(cached_tokens=6700),
        )


def _thread_ctx() -> dict:
    return {
        "model": "gpt-5.6-luna",
        "system_prompt": "You are boxBot.",
        "tools": [],
        "response_format": {"type": "json_schema"},
        "effort_kwargs": {},
    }


def _agent_with_fakes(completion=None, create_side_effect=None):
    """Minimal BoxBotAgent surface for the thread-extraction path
    (same object.__new__ pattern as TestPostConversationTriggerSkip in
    test_trigger_extraction_skip.py)."""
    from boxbot.core import agent as agent_mod

    agent = object.__new__(agent_mod.BoxBotAgent)
    store = MagicMock()
    store.create_pending_extraction = AsyncMock()
    store.get_pending_extraction = AsyncMock(return_value="row")
    store.mark_pending_applied = AsyncMock()
    store.get_conversation = AsyncMock(return_value=None)
    store.create_conversation = AsyncMock()
    store.update_conversation = AsyncMock()
    store.create_memory = AsyncMock(return_value="mem_new")
    store.invalidate_memory = AsyncMock()
    store.update_system_memory = AsyncMock()
    # boxbot.cost.record writes straight through store.db
    store.db.execute = AsyncMock()
    store.db.commit = AsyncMock()
    agent._memory_store = store
    agent._batch_poller = MagicMock()
    agent._batch_poller.submit = AsyncMock()

    client = MagicMock()
    if create_side_effect is not None:
        client.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    else:
        client.chat.completions.create = AsyncMock(return_value=completion)
    agent._openai_client = client
    agent._ensure_openai_client = lambda: client
    return agent, store, client


_THREAD_MESSAGES = [
    {"role": "user", "content": "[Jacob]: Can you lock the door?"},
    {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
]


class TestTryThreadExtraction:
    @pytest.mark.asyncio
    async def test_success_applies_and_marks_row(self, mock_config):
        payload = dict(_MINIMAL_PAYLOAD)
        payload["invalidations"] = [
            {"memory_id": "mem_0412", "reason": "contradicted"},
        ]
        agent, store, client = _agent_with_fakes(
            completion=_FakeCompletion(_thread_reply(payload)),
        )

        applied = await agent._try_thread_extraction(
            conversation_id="conv_t1",
            channel="voice",
            participants=["BB", "Jacob"],
            started_at="2026-08-29T21:00:00",
            messages=list(_THREAD_MESSAGES),
            accessed_memory_ids=["mem_0412"],
            injected_memories_block="[Active Memories]\n- mem_0412: stale",
            ctx=_thread_ctx(),
        )

        assert applied is True
        store.mark_pending_applied.assert_awaited_once_with("conv_t1")
        store.invalidate_memory.assert_awaited_once()
        # Cost row appended (boxbot.cost.record → store.db.execute;
        # params run (timestamp, purpose, provider, model, ...)).
        store.db.execute.assert_awaited_once()
        cost_params = store.db.execute.call_args.args[1]
        assert cost_params[1] == "extraction"
        assert cost_params[2] == "openai"

        # The request replayed the conversation's exact shape: same
        # tools/response_format from ctx, tool_choice=none, and the
        # extraction instruction appended as the final user message.
        call = client.chat.completions.create.call_args.kwargs
        assert call["tool_choice"] == "none"
        assert call["response_format"] == {"type": "json_schema"}
        assert call["messages"][0]["role"] == "system"
        assert call["messages"][-1]["role"] == "user"
        assert "memory extractor" in call["messages"][-1]["content"]

    @pytest.mark.asyncio
    async def test_api_failure_returns_false(self, mock_config):
        agent, store, _ = _agent_with_fakes(
            create_side_effect=RuntimeError("azure stall"),
        )
        applied = await agent._try_thread_extraction(
            conversation_id="conv_t2",
            channel="voice",
            participants=["BB"],
            started_at="2026-08-29T21:00:00",
            messages=list(_THREAD_MESSAGES),
            accessed_memory_ids=[],
            injected_memories_block="",
            ctx=_thread_ctx(),
        )
        assert applied is False
        store.mark_pending_applied.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unparseable_reply_returns_false(self, mock_config):
        agent, store, _ = _agent_with_fakes(
            completion=_FakeCompletion("Sure, the door is locked!"),
        )
        applied = await agent._try_thread_extraction(
            conversation_id="conv_t3",
            channel="voice",
            participants=["BB"],
            started_at="2026-08-29T21:00:00",
            messages=list(_THREAD_MESSAGES),
            accessed_memory_ids=[],
            injected_memories_block="",
            ctx=_thread_ctx(),
        )
        assert applied is False
        store.mark_pending_applied.assert_not_awaited()


class TestPostConversationThreadRouting:
    @pytest.mark.asyncio
    async def test_thread_success_skips_batch(self, mock_config):
        agent, store, _ = _agent_with_fakes(
            completion=_FakeCompletion(_thread_reply(_MINIMAL_PAYLOAD)),
        )
        await agent._post_conversation(
            conversation_id="conv_t4",
            channel="voice",
            person_name="Jacob",
            messages=list(_THREAD_MESSAGES),
            accessed_memory_ids=[],
            started_at="2026-08-29T21:00:00",
            openai_thread_ctx=_thread_ctx(),
        )
        # Durable row still written first, then applied live — batch
        # never submitted.
        store.create_pending_extraction.assert_awaited_once()
        store.mark_pending_applied.assert_awaited_once()
        agent._batch_poller.submit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_thread_failure_falls_back_to_batch(self, mock_config):
        agent, store, _ = _agent_with_fakes(
            create_side_effect=RuntimeError("azure stall"),
        )
        await agent._post_conversation(
            conversation_id="conv_t5",
            channel="voice",
            person_name="Jacob",
            messages=list(_THREAD_MESSAGES),
            accessed_memory_ids=[],
            started_at="2026-08-29T21:00:00",
            openai_thread_ctx=_thread_ctx(),
        )
        store.create_pending_extraction.assert_awaited_once()
        agent._batch_poller.submit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_ctx_goes_straight_to_batch(self, mock_config):
        agent, store, client = _agent_with_fakes(
            completion=_FakeCompletion(_thread_reply(_MINIMAL_PAYLOAD)),
        )
        await agent._post_conversation(
            conversation_id="conv_t6",
            channel="voice",
            person_name="Jacob",
            messages=list(_THREAD_MESSAGES),
            accessed_memory_ids=[],
            started_at="2026-08-29T21:00:00",
        )
        client.chat.completions.create.assert_not_awaited()
        agent._batch_poller.submit.assert_awaited_once()


# ---------------------------------------------------------------------------
# Idle-window thread extraction for persistent text threads
# ---------------------------------------------------------------------------


def _idle_messages():
    return [
        {"role": "user", "content": "[Jacob]: We switched to oat milk."},
        {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
        {"role": "user", "content": "[Jacob]: Carina starts her new job Monday."},
        {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
    ]


class _FakeConv:
    def __init__(self, conversation_id="conv_w1", messages=None,
                 lifecycle_mode="persistent"):
        self.conversation_id = conversation_id
        self.channel = "whatsapp"
        self.lifecycle_mode = lifecycle_mode
        self.thread = list(messages or _idle_messages())
        self.participants = {"Jacob"}
        self.accessed_memory_ids = []
        self.injected_memories_block = ""
        self.is_ended = False

    def started_at_iso(self):
        return "2026-10-03T09:00:00"


def _idle_agent(mock_config):
    import asyncio

    agent, store, client = _agent_with_fakes(
        completion=_FakeCompletion(_thread_reply(_MINIMAL_PAYLOAD)),
    )
    agent._index_lock = asyncio.Lock()
    agent._conversations = {}
    agent._thread_extraction_ctx = {}
    agent._thread_extracted_upto = {}
    agent._idle_extraction_tasks = {}
    return agent, store, client


def _last_extraction_message(client) -> str:
    call = client.chat.completions.create.await_args
    return call.kwargs["messages"][-1]["content"]


class TestThreadExtractionMessageIncremental:
    def test_prior_turns_note(self):
        from boxbot.memory.extraction import build_thread_extraction_message

        full = build_thread_extraction_message(
            injected_memories_block="", channel="whatsapp",
            participants=["BB", "Jacob"], started_at="t",
        )
        assert "Incremental pass" not in full
        partial = build_thread_extraction_message(
            injected_memories_block="", channel="whatsapp",
            participants=["BB", "Jacob"], started_at="t",
            prior_extracted_turns=4,
        )
        assert "first 4 messages" in partial
        assert "conversation_summary still covers the whole thread" in partial


class TestIdleThreadExtraction:
    @pytest.mark.asyncio
    async def test_idle_pass_extracts_then_only_the_delta(self, mock_config):
        agent, store, client = _idle_agent(mock_config)
        conv = _FakeConv()
        agent._conversations[conv.conversation_id] = conv
        agent._thread_extraction_ctx[conv.conversation_id] = _thread_ctx()

        await agent._idle_thread_extraction(conv.conversation_id, 0)
        assert client.chat.completions.create.await_count == 1
        assert "Incremental pass" not in _last_extraction_message(client)
        assert agent._thread_extracted_upto[conv.conversation_id] == 4
        # Nothing queued for the batch path from an idle pass.
        store.create_pending_extraction.assert_not_awaited()
        agent._batch_poller.submit.assert_not_awaited()

        # Only an assistant turn landed since: nothing new from a human.
        conv.thread.append(
            {"role": "assistant", "content": [{"type": "text", "text": "{}"}]}
        )
        await agent._idle_thread_extraction(conv.conversation_id, 0)
        assert client.chat.completions.create.await_count == 1

        # A human turn landed: the next pass is incremental.
        conv.thread.append({"role": "user", "content": "[Jacob]: Also, dentist Friday."})
        conv.thread.append(
            {"role": "assistant", "content": [{"type": "text", "text": "{}"}]}
        )
        await agent._idle_thread_extraction(conv.conversation_id, 0)
        assert client.chat.completions.create.await_count == 2
        assert "first 4 messages" in _last_extraction_message(client)
        assert agent._thread_extracted_upto[conv.conversation_id] == 7

    @pytest.mark.asyncio
    async def test_idle_pass_needs_a_live_conversation_and_ctx(self, mock_config):
        agent, store, client = _idle_agent(mock_config)
        await agent._idle_thread_extraction("ghost", 0)  # not indexed
        conv = _FakeConv()
        agent._conversations[conv.conversation_id] = conv
        await agent._idle_thread_extraction(conv.conversation_id, 0)  # no ctx
        conv.is_ended = True
        agent._thread_extraction_ctx[conv.conversation_id] = _thread_ctx()
        await agent._idle_thread_extraction(conv.conversation_id, 0)  # ended
        client.chat.completions.create.assert_not_awaited()

    def test_arm_only_for_persistent_openai_threads(self, mock_config):
        import asyncio

        async def _run():
            agent, _store, _client = _idle_agent(mock_config)
            mock_config.memory.thread_extraction_idle_seconds = 60
            transient = _FakeConv("conv_v1", lifecycle_mode="transient")
            agent._thread_extraction_ctx["conv_v1"] = _thread_ctx()
            agent._arm_idle_thread_extraction(transient)
            assert "conv_v1" not in agent._idle_extraction_tasks

            persistent = _FakeConv("conv_w2")
            agent._arm_idle_thread_extraction(persistent)  # no ctx yet
            assert "conv_w2" not in agent._idle_extraction_tasks
            agent._thread_extraction_ctx["conv_w2"] = _thread_ctx()
            agent._arm_idle_thread_extraction(persistent)
            first = agent._idle_extraction_tasks["conv_w2"]
            agent._arm_idle_thread_extraction(persistent)  # re-arm cancels
            second = agent._idle_extraction_tasks["conv_w2"]
            await asyncio.sleep(0)
            assert first is not second and first.cancelled()
            agent._cancel_idle_thread_extraction("conv_w2")
            await asyncio.sleep(0)
            assert second.cancelled()

        asyncio.run(_run())


class TestCloseAfterIdleExtraction:
    @pytest.mark.asyncio
    async def test_fully_extracted_thread_does_nothing_at_close(self, mock_config):
        agent, store, client = _idle_agent(mock_config)
        msgs = _idle_messages()
        await agent._post_conversation(
            conversation_id="conv_w1", channel="whatsapp", person_name="Jacob",
            messages=msgs, accessed_memory_ids=[], started_at="t",
            openai_thread_ctx=_thread_ctx(), already_extracted_upto=len(msgs),
        )
        client.chat.completions.create.assert_not_awaited()
        store.create_pending_extraction.assert_not_awaited()
        agent._batch_poller.submit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_close_with_ctx_extracts_incrementally(self, mock_config):
        agent, store, client = _idle_agent(mock_config)
        msgs = _idle_messages()
        await agent._post_conversation(
            conversation_id="conv_w1", channel="whatsapp", person_name="Jacob",
            messages=msgs, accessed_memory_ids=[], started_at="t",
            openai_thread_ctx=_thread_ctx(), already_extracted_upto=2,
        )
        assert client.chat.completions.create.await_count == 1
        assert "first 2 messages" in _last_extraction_message(client)
        agent._batch_poller.submit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_close_without_ctx_batches_only_the_delta(self, mock_config):
        agent, store, client = _idle_agent(mock_config)
        msgs = _idle_messages()
        await agent._post_conversation(
            conversation_id="conv_w1", channel="whatsapp", person_name="Jacob",
            messages=msgs, accessed_memory_ids=[], started_at="t",
            openai_thread_ctx=None, already_extracted_upto=2,
        )
        client.chat.completions.create.assert_not_awaited()
        transcript = store.create_pending_extraction.await_args.kwargs["transcript"]
        assert "already extracted" in transcript
        assert "oat milk" not in transcript
        assert "new job Monday" in transcript
        agent._batch_poller.submit.assert_awaited_once()
