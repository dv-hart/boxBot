"""Tests for Haiku-backed reranking and summary synthesis (search.py).

Covers §1 (rerank_with_haiku) and §2 (summary mode synthesis) from
docs/plans/memory-roadmap-post-phase-a.md. The Anthropic client is
mocked using the same _StubBlock / _StubMessage shape as
test_extraction_pipeline.py.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Mock client (mirrors the shape used in test_extraction_pipeline.py)
# ---------------------------------------------------------------------------


class _StubBlock:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _StubMessage:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage


def _usage(input_tokens=500, output_tokens=120, cache_read=0, cache_write=0):
    u = MagicMock()
    u.input_tokens = input_tokens
    u.output_tokens = output_tokens
    u.cache_read_input_tokens = cache_read
    u.cache_creation_input_tokens = cache_write
    return u


class FakeAnthropicClient:
    """Programmable stand-in for ``anthropic.AsyncAnthropic``.

    Tests pre-load a queue of (tool_name, payload) tuples; each
    ``messages.create`` call pops the next one and returns a stub
    message wrapping it. If the queue is empty, raises so the test
    fails loudly instead of silently returning empty.
    """

    def __init__(self):
        self._queue: list[tuple[str, dict]] = []
        self.create_calls: list[dict] = []
        self.messages = types.SimpleNamespace(create=self._create)

    def queue_response(self, tool_name: str, payload: dict, *, usage=None):
        self._queue.append((tool_name, payload, usage or _usage()))

    async def _create(self, **kwargs):
        self.create_calls.append(kwargs)
        if not self._queue:
            raise RuntimeError(
                "FakeAnthropicClient ran out of queued responses; "
                "test under-provisioned its mock."
            )
        tool_name, payload, usage = self._queue.pop(0)
        return _StubMessage(
            content=[
                _StubBlock(type="tool_use", name=tool_name, input=payload),
            ],
            usage=usage,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def fresh_store(tmp_path):
    from unittest.mock import patch

    from boxbot.memory.store import MemoryStore

    db_path = tmp_path / "memory.db"
    sys_mem_path = tmp_path / "system.md"
    store = MemoryStore(db_path=db_path)
    with patch("boxbot.memory.store.SYSTEM_MEMORY_PATH", sys_mem_path):
        await store.initialize()
        yield store
        await store.close()


@pytest.fixture
def fake_client():
    return FakeAnthropicClient()


def _make_candidate(
    id_: str,
    *,
    source: str = "memory",
    type_: str = "person",
    person: str | None = "Erik",
    content: str = "",
    summary: str = "",
    score: float = 0.5,
):
    from boxbot.memory.search import SearchCandidate

    return SearchCandidate(
        id=id_,
        source=source,
        type=type_,
        person=person,
        content=content or summary,
        summary=summary,
        vector_score=score,
        bm25_score=score,
        combined_score=score,
        metadata={"started_at": "2026-04-01T00:00:00", "status": "active"},
    )


# ---------------------------------------------------------------------------
# §1 — Rerank
# ---------------------------------------------------------------------------


class TestRerank:
    @pytest.mark.asyncio
    async def test_all_relevant_returns_in_score_order_with_titles(
        self, fresh_store, fake_client,
    ):
        from boxbot.memory.search import rerank_with_haiku

        # Build 3 candidates with descending combined_score.
        cands = [
            _make_candidate("m-a", summary="Erik likes pizza", score=0.9),
            _make_candidate("m-b", summary="Erik plays soccer", score=0.7),
            _make_candidate("m-c", summary="Erik's bedtime is 9", score=0.5),
        ]
        fake_client.queue_response(
            "rerank_candidates",
            {
                "judgments": [
                    {
                        "candidate_id": "m-a",
                        "relevant": True,
                        "title": "Erik food preference",
                        "summary": "Erik likes pizza.",
                        "relevance_reason": "Directly mentions a food fact.",
                    },
                    {
                        "candidate_id": "m-b",
                        "relevant": True,
                        "title": "Erik sports",
                        "summary": "Erik plays soccer.",
                        "relevance_reason": "Hobby context.",
                    },
                    {
                        "candidate_id": "m-c",
                        "relevant": True,
                        "title": "Erik bedtime",
                        "summary": "Erik's bedtime is 9 PM.",
                        "relevance_reason": "Routine.",
                    },
                ]
            },
        )

        results = await rerank_with_haiku(
            cands, "Erik habits",
            client=fake_client, store=fresh_store,
        )
        assert len(results) == 3
        # Original combined_score order preserved.
        assert [r.id for r in results] == ["m-a", "m-b", "m-c"]
        # Model-supplied titles populated.
        assert results[0].title == "Erik food preference"
        assert results[1].relevance == "Hobby context."
        assert results[2].summary == "Erik's bedtime is 9 PM."

    @pytest.mark.asyncio
    async def test_irrelevant_candidates_filtered_out(
        self, fresh_store, fake_client,
    ):
        from boxbot.memory.search import rerank_with_haiku

        cands = [
            _make_candidate("m-1", summary="Erik likes pizza", score=0.9),
            _make_candidate("m-2", summary="Server logs from 2019", score=0.8),
            _make_candidate("m-3", summary="Erik's school is GISC", score=0.7),
            _make_candidate("m-4", summary="Random IT note", score=0.6),
        ]
        fake_client.queue_response(
            "rerank_candidates",
            {
                "judgments": [
                    {
                        "candidate_id": "m-1",
                        "relevant": True,
                        "title": "Pizza preference",
                        "summary": "Erik likes pizza.",
                        "relevance_reason": "Food fact about Erik.",
                    },
                    {
                        "candidate_id": "m-2",
                        "relevant": False,
                        "title": "",
                        "summary": "",
                        "relevance_reason": "Not about Erik.",
                    },
                    {
                        "candidate_id": "m-3",
                        "relevant": True,
                        "title": "Erik school",
                        "summary": "Erik attends GISC.",
                        "relevance_reason": "School fact about Erik.",
                    },
                    {
                        "candidate_id": "m-4",
                        "relevant": False,
                        "title": "",
                        "summary": "",
                        "relevance_reason": "Off-topic.",
                    },
                ]
            },
        )

        results = await rerank_with_haiku(
            cands, "Erik facts",
            client=fake_client, store=fresh_store,
        )
        ids = [r.id for r in results]
        assert ids == ["m-1", "m-3"]
        assert all(r.relevance for r in results)

    @pytest.mark.asyncio
    async def test_records_cost_with_purpose_rerank(
        self, fresh_store, fake_client,
    ):
        from boxbot.memory.search import rerank_with_haiku

        cands = [_make_candidate("m-x", summary="x", score=0.5)]
        fake_client.queue_response(
            "rerank_candidates",
            {
                "judgments": [
                    {
                        "candidate_id": "m-x",
                        "relevant": True,
                        "title": "x",
                        "summary": "x",
                        "relevance_reason": "y",
                    },
                ]
            },
            usage=_usage(input_tokens=700, output_tokens=80),
        )

        await rerank_with_haiku(
            cands, "test query",
            client=fake_client, store=fresh_store,
        )
        summary = await fresh_store.cost_summary(days=7)
        assert "rerank" in summary
        assert summary["rerank"] > 0

    @pytest.mark.asyncio
    async def test_empty_candidates_short_circuits(
        self, fresh_store, fake_client,
    ):
        from boxbot.memory.search import rerank_with_haiku

        results = await rerank_with_haiku(
            [], "query", client=fake_client, store=fresh_store,
        )
        assert results == []
        # No API calls made.
        assert fake_client.create_calls == []

    @pytest.mark.asyncio
    async def test_batches_into_groups_of_six(
        self, fresh_store, fake_client,
    ):
        from boxbot.memory.search import rerank_with_haiku

        # 13 candidates -> ceil(13/6) = 3 batches -> 3 parallel API calls.
        cands = [
            _make_candidate(f"m-{i}", summary=f"fact {i}", score=1.0 - 0.05 * i)
            for i in range(13)
        ]
        # Queue 3 responses, one per batch. Each says all candidates relevant.
        for batch_idx in range(3):
            start = batch_idx * 6
            end = min(start + 6, 13)
            fake_client.queue_response(
                "rerank_candidates",
                {
                    "judgments": [
                        {
                            "candidate_id": f"m-{i}",
                            "relevant": True,
                            "title": f"t{i}",
                            "summary": f"s{i}",
                            "relevance_reason": "r",
                        }
                        for i in range(start, end)
                    ]
                },
            )

        results = await rerank_with_haiku(
            cands, "q", client=fake_client, store=fresh_store,
        )
        assert len(fake_client.create_calls) == 3
        # All 13 surfaced, in original combined_score order.
        assert [r.id for r in results] == [f"m-{i}" for i in range(13)]


# ---------------------------------------------------------------------------
# Shared filter helper (rank vs summarize)
# ---------------------------------------------------------------------------


class TestFilterCandidatesShared:
    @pytest.mark.asyncio
    async def test_rank_intent_uses_rerank_tool(
        self, fresh_store, fake_client,
    ):
        from boxbot.memory.search import filter_candidates

        cands = [_make_candidate("m-1", summary="x", score=0.5)]
        fake_client.queue_response(
            "rerank_candidates",
            {
                "judgments": [
                    {
                        "candidate_id": "m-1",
                        "relevant": True,
                        "title": "T", "summary": "S",
                        "relevance_reason": "R",
                    }
                ]
            },
        )
        judgments = await filter_candidates(
            fake_client, cands, "q", intent="rank", store=fresh_store,
            cost_purpose="rerank",
        )
        assert len(judgments) == 1
        # The rank tool was selected.
        call = fake_client.create_calls[0]
        assert call["tool_choice"]["name"] == "rerank_candidates"
        assert "title" in judgments[0]

    @pytest.mark.asyncio
    async def test_summarize_intent_uses_filter_tool(
        self, fresh_store, fake_client,
    ):
        from boxbot.memory.search import filter_candidates

        cands = [_make_candidate("m-1", summary="x", score=0.5)]
        fake_client.queue_response(
            "filter_candidates",
            {
                "judgments": [
                    {
                        "candidate_id": "m-1",
                        "relevant": True,
                        "snippet": "Erik likes pizza.",
                    }
                ]
            },
        )
        judgments = await filter_candidates(
            fake_client, cands, "q", intent="summarize", store=fresh_store,
            cost_purpose="summary",
        )
        assert len(judgments) == 1
        call = fake_client.create_calls[0]
        assert call["tool_choice"]["name"] == "filter_candidates"
        assert judgments[0]["snippet"] == "Erik likes pizza."

    @pytest.mark.asyncio
    async def test_unknown_intent_raises(self, fake_client):
        from boxbot.memory.search import filter_candidates

        with pytest.raises(ValueError):
            await filter_candidates(
                fake_client,
                [_make_candidate("m-1", summary="x")],
                "q",
                intent="bogus",
            )


# ---------------------------------------------------------------------------
# §2 — Summary mode end-to-end
# ---------------------------------------------------------------------------


class TestSummaryMode:
    @pytest.mark.asyncio
    async def test_filter_then_synthesize_happy_path(
        self, fresh_store, fake_client,
    ):
        from boxbot.memory.search import _search_summary
        from unittest.mock import patch

        # Seed 3 memories.
        await fresh_store.create_memory(
            type="person", person="Erik",
            content="Erik likes chicken pesto pizza most.",
            summary="Erik likes chicken pesto pizza",
            tags=["food"],
        )
        await fresh_store.create_memory(
            type="person", person="Erik",
            content="Erik also enjoys mac and cheese.",
            summary="Erik likes mac and cheese",
            tags=["food"],
        )
        await fresh_store.create_memory(
            type="household",
            content="The car is a Subaru.",
            summary="Family car: Subaru",
            tags=["car"],
        )

        async def fake_hybrid_search(*args, **kwargs):
            from boxbot.memory.search import SearchCandidate
            return [
                SearchCandidate(
                    id="mem-1", source="memory", type="person",
                    person="Erik",
                    content="Erik likes chicken pesto pizza most.",
                    summary="Erik likes chicken pesto pizza",
                    combined_score=0.9,
                    metadata={"status": "active"},
                ),
                SearchCandidate(
                    id="mem-2", source="memory", type="person",
                    person="Erik",
                    content="Erik also enjoys mac and cheese.",
                    summary="Erik likes mac and cheese",
                    combined_score=0.8,
                    metadata={"status": "active"},
                ),
                SearchCandidate(
                    id="mem-3", source="memory", type="household",
                    person=None,
                    content="The car is a Subaru.",
                    summary="Family car: Subaru",
                    combined_score=0.6,
                    metadata={"status": "active"},
                ),
            ]

        # Filter judgment: keep mem-1, mem-2; drop mem-3.
        fake_client.queue_response(
            "filter_candidates",
            {
                "judgments": [
                    {
                        "candidate_id": "mem-1",
                        "relevant": True,
                        "snippet": "chicken pesto pizza",
                    },
                    {
                        "candidate_id": "mem-2",
                        "relevant": True,
                        "snippet": "mac and cheese",
                    },
                    {
                        "candidate_id": "mem-3",
                        "relevant": False,
                        "snippet": "",
                    },
                ]
            },
        )
        # Synthesis call.
        fake_client.queue_response(
            "synthesize_answer",
            {
                "answer": "Erik likes chicken pesto pizza and mac and cheese.",
                "source_ids": ["mem-1", "mem-2"],
            },
        )

        with patch("boxbot.memory.search.hybrid_search", fake_hybrid_search):
            result = await _search_summary(
                fresh_store, "what does Erik like to eat?",
                types=None, person=None,
                include_conversations=True, include_archived=False,
                client=fake_client,
            )

        assert result["answer"] == (
            "Erik likes chicken pesto pizza and mac and cheese."
        )
        assert set(result["sources"]) == {"mem-1", "mem-2"}
        # 1 filter + 1 synthesis = 2 client calls.
        assert len(fake_client.create_calls) == 2

    @pytest.mark.asyncio
    async def test_summary_records_cost(self, fresh_store, fake_client):
        from boxbot.memory.search import _search_summary
        from unittest.mock import patch

        async def fake_hybrid_search(*args, **kwargs):
            from boxbot.memory.search import SearchCandidate
            return [
                SearchCandidate(
                    id="mem-1", source="memory", type="person",
                    person="Erik", content="Erik likes pizza.",
                    summary="Erik likes pizza",
                    combined_score=0.9,
                    metadata={"status": "active"},
                ),
            ]

        fake_client.queue_response(
            "filter_candidates",
            {
                "judgments": [
                    {
                        "candidate_id": "mem-1",
                        "relevant": True,
                        "snippet": "pizza",
                    },
                ]
            },
            usage=_usage(input_tokens=400, output_tokens=60),
        )
        fake_client.queue_response(
            "synthesize_answer",
            {"answer": "Erik likes pizza.", "source_ids": ["mem-1"]},
            usage=_usage(input_tokens=200, output_tokens=40),
        )

        with patch("boxbot.memory.search.hybrid_search", fake_hybrid_search):
            await _search_summary(
                fresh_store, "what does Erik eat?",
                types=None, person=None,
                include_conversations=True, include_archived=False,
                client=fake_client,
            )

        summary = await fresh_store.cost_summary(days=7)
        assert "summary" in summary
        assert summary["summary"] > 0

    @pytest.mark.asyncio
    async def test_summary_no_relevant_returns_empty(
        self, fresh_store, fake_client,
    ):
        from boxbot.memory.search import _search_summary
        from unittest.mock import patch

        async def fake_hybrid_search(*args, **kwargs):
            from boxbot.memory.search import SearchCandidate
            return [
                SearchCandidate(
                    id="mem-x", source="memory", type="household",
                    person=None, content="Random fact",
                    summary="Random",
                    combined_score=0.4,
                    metadata={"status": "active"},
                ),
            ]

        fake_client.queue_response(
            "filter_candidates",
            {
                "judgments": [
                    {
                        "candidate_id": "mem-x",
                        "relevant": False,
                        "snippet": "",
                    },
                ]
            },
        )

        with patch("boxbot.memory.search.hybrid_search", fake_hybrid_search):
            result = await _search_summary(
                fresh_store, "unrelated question",
                types=None, person=None,
                include_conversations=True, include_archived=False,
                client=fake_client,
            )
        assert result["answer"] == "No relevant memories found."
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_summary_no_candidates_short_circuits(
        self, fresh_store, fake_client,
    ):
        from boxbot.memory.search import _search_summary
        from unittest.mock import patch

        async def fake_hybrid_search(*args, **kwargs):
            return []

        with patch("boxbot.memory.search.hybrid_search", fake_hybrid_search):
            result = await _search_summary(
                fresh_store, "question",
                types=None, person=None,
                include_conversations=True, include_archived=False,
                client=fake_client,
            )
        assert result["answer"] == "No relevant memories found."
        assert result["sources"] == []
        # No client calls made.
        assert fake_client.create_calls == []


# ---------------------------------------------------------------------------
# Lookup mode end-to-end with rerank
# ---------------------------------------------------------------------------


class TestLookupModeWithRerank:
    @pytest.mark.asyncio
    async def test_lookup_passes_through_rerank(
        self, fresh_store, fake_client,
    ):
        from boxbot.memory.search import _search_lookup
        from unittest.mock import patch

        async def fake_hybrid_search(*args, **kwargs):
            from boxbot.memory.search import SearchCandidate
            return [
                SearchCandidate(
                    id="mem-1", source="memory", type="person",
                    person="Erik", content="Erik likes pizza.",
                    summary="Erik likes pizza", combined_score=0.9,
                    metadata={"status": "active"},
                ),
                SearchCandidate(
                    id="conv-1", source="conversation", type="conversation",
                    person=None, content="Erik liked dinner.",
                    summary="Erik liked dinner", combined_score=0.7,
                    metadata={
                        "started_at": "2026-04-01T18:00:00",
                        "participants": ["Jacob"],
                        "channel": "voice",
                    },
                ),
            ]

        fake_client.queue_response(
            "rerank_candidates",
            {
                "judgments": [
                    {
                        "candidate_id": "mem-1",
                        "relevant": True,
                        "title": "Erik food preference",
                        "summary": "Erik likes pizza",
                        "relevance_reason": "Direct food fact",
                    },
                    {
                        "candidate_id": "conv-1",
                        "relevant": True,
                        "title": "Dinner",
                        "summary": "Erik liked dinner",
                        "relevance_reason": "Mentions Erik eating",
                    },
                ]
            },
        )

        with patch("boxbot.memory.search.hybrid_search", fake_hybrid_search):
            result = await _search_lookup(
                fresh_store, "Erik food",
                types=None, person="Erik",
                include_conversations=True, include_archived=False,
                client=fake_client,
            )
        assert len(result["facts"]) == 1
        assert result["facts"][0]["title"] == "Erik food preference"
        assert result["facts"][0]["relevance"] == "Direct food fact"
        assert len(result["conversations"]) == 1
        assert result["conversations"][0]["id"] == "conv-1"
