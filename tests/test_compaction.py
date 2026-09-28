"""Tests for within-thread compaction (:mod:`boxbot.core.compaction`) and
its integration into the agent generate loop.

Two layers:
- Pure eviction planning (:func:`plan_compaction`) — the correctness-
  critical part: never split a ``tool_use`` from its ``tool_result``,
  keep valid role alternation.
- Async :func:`compact` — summarization success, and graceful fallback
  to deterministic truncation on failure / missing client.
- The context-overflow error path in the agent loop — detection helper
  and the compact-and-retry wiring (mirrors ``test_agent_image_scrub``).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from boxbot.core import compaction
from boxbot.core.compaction import (
    _OMITTED_NOTE,
    _SUMMARY_PREFIX,
    compact,
    estimate_tokens,
    plan_compaction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _tool_use(text: str = "", *, tid: str = "t1") -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tid, "name": "foo", "input": {}}],
    }


def _tool_result(text: str = "r", *, tid: str = "t1") -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": text}
        ],
    }


def _mixed_tool_result(text: str = "r", *, tid: str = "t1") -> dict:
    """A MIXED [tool_result, text] user turn.

    The inject-don't-interrupt path folds queued user speech into the
    same user turn that carries the tool_result. Its tool_use lives in
    the assistant turn immediately before it, so this is NOT a safe
    eviction boundary.
    """
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": "res"},
            {"type": "text", "text": text},
        ],
    }


def _big(text: str, tokens: int) -> str:
    # chars/4 heuristic → tokens*4 chars for ~`tokens` estimate.
    return (text + " ") + "x" * (tokens * 4)


def _assert_valid_sequence(messages: list[dict]) -> None:
    """First message user; no two adjacent same-role turns."""
    assert messages, "empty sequence"
    assert messages[0]["role"] == "user", "first message must be user"
    for prev, cur in zip(messages, messages[1:]):
        assert prev["role"] != cur["role"], (
            f"adjacent same-role turns: {prev['role']}"
        )


def _starts_with_orphan_tool_result(msg: dict) -> bool:
    """True if the message carries ANY tool_result block.

    A retained tail whose first turn contains a tool_result orphans it —
    the matching tool_use was evicted. Must be any(), not all(): a MIXED
    [tool_result, text] turn is just as unsafe as a tool_result-only one.
    """
    content = msg.get("content")
    return (
        isinstance(content, list)
        and any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
    )


def _assert_tool_pairs_intact(messages: list[dict]) -> None:
    """Every tool_result in the sequence has a matching earlier tool_use.

    True pair-integrity (not mere adjacency): walk the whole retained
    slice, collecting tool_use ids as we go, and assert each tool_result's
    tool_use_id was already seen.
    """
    seen_tool_use: set[str] = set()
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                seen_tool_use.add(block.get("id"))
            elif block.get("type") == "tool_result":
                tid = block.get("tool_use_id")
                assert tid in seen_tool_use, (
                    f"orphaned tool_result {tid!r} — no matching tool_use "
                    f"earlier in the retained sequence"
                )


# ---------------------------------------------------------------------------
# plan_compaction — pure eviction
# ---------------------------------------------------------------------------


def test_under_threshold_is_noop() -> None:
    msgs = [_user("hi"), _assistant("hello"), _user("bye")]
    evicted, retained = plan_compaction(
        msgs, threshold_tokens=10_000, keep_recent_tokens=5_000
    )
    assert evicted == []
    assert retained == msgs


def test_over_threshold_evicts_oldest_keeps_tail() -> None:
    msgs = [
        _user(_big("u0", 400)),
        _assistant(_big("a0", 400)),
        _user(_big("u1", 400)),
        _assistant(_big("a1", 50)),
        _user("fresh question"),
    ]
    evicted, retained = plan_compaction(
        msgs, threshold_tokens=500, keep_recent_tokens=200
    )
    assert evicted, "expected eviction"
    # Tail is a suffix of the original; the fresh input is always kept.
    assert retained[-1] == msgs[-1]
    assert evicted + retained == msgs
    # Retained tail starts with a real user turn (valid API sequence).
    assert retained[0]["role"] == "user"
    assert not _starts_with_orphan_tool_result(retained[0])


def test_never_splits_tool_use_from_tool_result() -> None:
    # A tool_result-only user turn is NOT a safe boundary — its tool_use
    # must stay with it. The only safe split is the fresh user turn.
    msgs = [
        _user(_big("u0", 400)),
        _tool_use(tid="t1"),
        _tool_result(tid="t1"),
        _assistant(_big("a0", 400)),
        _user("fresh"),
    ]
    evicted, retained = plan_compaction(
        msgs, threshold_tokens=300, keep_recent_tokens=50
    )
    # Boundary must land on the fresh user turn (index 4), keeping the
    # tool_use/tool_result pair together in the evicted head.
    assert retained == [msgs[4]]
    assert msgs[1] in evicted and msgs[2] in evicted
    assert not _starts_with_orphan_tool_result(retained[0])


def test_never_splits_across_mixed_tool_result_text_turn() -> None:
    # REGRESSION: a MIXED [tool_result, text] user turn is real-looking
    # (has a text block) but is NOT a safe boundary — its tool_use is in
    # the assistant turn before it. A naive all()-predicate would evict
    # across it and orphan the tool_result.
    msgs = [
        _user(_big("u0", 400)),
        _tool_use(tid="t1"),
        _mixed_tool_result("hey are you still working on it?", tid="t1"),
        _assistant(_big("a0", 400)),
        _user("fresh"),
    ]
    evicted, retained = plan_compaction(
        msgs, threshold_tokens=300, keep_recent_tokens=50
    )
    # The mixed turn (index 2) must NOT be the boundary; only the fresh
    # user turn (index 4) is safe.
    assert retained == [msgs[4]]
    assert msgs[2] in evicted
    assert not _starts_with_orphan_tool_result(retained[0])
    _assert_tool_pairs_intact(retained)


def test_keeps_more_when_budget_allows() -> None:
    # keep_recent large enough to retain from an earlier safe boundary.
    msgs = [
        _user(_big("u0", 400)),
        _assistant(_big("a0", 40)),
        _user(_big("u1", 40)),
        _assistant(_big("a1", 40)),
        _user("fresh"),
    ]
    evicted, retained = plan_compaction(
        msgs, threshold_tokens=300, keep_recent_tokens=200
    )
    # u1 onward fits the 200 budget → boundary at index 2, not the end.
    assert retained[0] is msgs[2]
    assert evicted == msgs[:2]


def test_no_safe_boundary_is_noop() -> None:
    # A single giant user turn: over threshold but nothing safe to split.
    msgs = [_user(_big("u0", 5_000))]
    evicted, retained = plan_compaction(
        msgs, threshold_tokens=100, keep_recent_tokens=50
    )
    assert evicted == []
    assert retained == msgs


# ---------------------------------------------------------------------------
# compact — async summarization + fallback
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, text: str = "SUMMARY", *, raises: bool = False) -> None:
        self._text = text
        self._raises = raises
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._text)]
        )


_OVER = dict(threshold_tokens=300, keep_recent_tokens=50)


def _long_thread() -> list[dict]:
    return [
        _user(_big("u0", 400)),
        _assistant(_big("a0", 400)),
        _user("fresh"),
    ]


@pytest.mark.asyncio
async def test_compact_merges_summary_into_retained_head() -> None:
    client = _FakeClient("Jacob asked about X; decided Y.")
    msgs = _long_thread()
    out = await compact(msgs, client=client, model="m", **_OVER)
    assert client.calls == 1
    _assert_valid_sequence(out)
    head = out[0]
    assert head["role"] == "user"
    # Summary note merged into the first retained user turn's content.
    assert _SUMMARY_PREFIX in head["content"]
    assert "Jacob asked about X" in head["content"]
    assert "fresh" in head["content"]


@pytest.mark.asyncio
async def test_compact_falls_back_on_summarization_error() -> None:
    client = _FakeClient(raises=True)
    msgs = _long_thread()
    out = await compact(msgs, client=client, model="m", **_OVER)
    # Degrades to deterministic truncation — never raises.
    _assert_valid_sequence(out)
    assert _OMITTED_NOTE in out[0]["content"]
    assert "fresh" in out[0]["content"]


@pytest.mark.asyncio
async def test_compact_none_client_truncates() -> None:
    msgs = _long_thread()
    out = await compact(msgs, client=None, model="m", **_OVER)
    _assert_valid_sequence(out)
    assert _OMITTED_NOTE in out[0]["content"]


@pytest.mark.asyncio
async def test_compact_noop_under_threshold() -> None:
    client = _FakeClient()
    msgs = [_user("hi"), _assistant("hello"), _user("bye")]
    out = await compact(
        msgs, client=client, model="m",
        threshold_tokens=100_000, keep_recent_tokens=50_000,
    )
    assert client.calls == 0
    assert out == msgs


@pytest.mark.asyncio
async def test_compact_preserves_tool_pair_and_alternation() -> None:
    client = _FakeClient()
    msgs = [
        _user(_big("u0", 400)),
        _tool_use(tid="t1"),
        _tool_result(tid="t1"),
        _assistant(_big("a0", 400)),
        _user("fresh"),
    ]
    out = await compact(msgs, client=client, model="m", **_OVER)
    _assert_valid_sequence(out)
    # No orphaned tool_result at the head of the compacted thread.
    assert not _starts_with_orphan_tool_result(out[0])
    _assert_tool_pairs_intact(out)


@pytest.mark.asyncio
async def test_compact_never_orphans_mixed_tool_result_turn() -> None:
    # End-to-end: a MIXED [tool_result, text] turn must not become the
    # compacted head with a dangling tool_result.
    client = _FakeClient()
    msgs = [
        _user(_big("u0", 400)),
        _tool_use(tid="t1"),
        _mixed_tool_result("still there?", tid="t1"),
        _assistant(_big("a0", 400)),
        _user("fresh"),
    ]
    out = await compact(msgs, client=client, model="m", **_OVER)
    _assert_valid_sequence(out)
    assert not _starts_with_orphan_tool_result(out[0])
    _assert_tool_pairs_intact(out)


def test_estimate_tokens_counts_blocks() -> None:
    # Text + tool_use + image all contribute; image is a flat estimate.
    msgs = [
        _user("x" * 400),  # ~100 tokens
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "y" * 400},
                {
                    "type": "image",
                    "source": {"type": "base64", "data": "z" * 10_000},
                },
            ],
        },
    ]
    est = estimate_tokens(msgs)
    # ~100 (text) + ~100 (text block) + flat image estimate.
    assert est >= 100 + 100 + compaction._IMAGE_TOKEN_EST
    # Image counted flat, NOT as its 10k base64 chars.
    assert est < 100 + 100 + compaction._IMAGE_TOKEN_EST + 500


# ---------------------------------------------------------------------------
# Context-overflow detection (mirrors test_agent_image_scrub pattern)
# ---------------------------------------------------------------------------


def test_detects_prompt_too_long() -> None:
    from boxbot.core.agent import _is_context_overflow_error

    err = (
        "Error code: 400 - {'type': 'error', 'error': {'type': "
        "'invalid_request_error', 'message': 'prompt is too long: "
        "215000 tokens > 200000 maximum'}}"
    )
    assert _is_context_overflow_error(err) is True


def test_detects_input_and_max_tokens_exceed() -> None:
    from boxbot.core.agent import _is_context_overflow_error

    err = (
        "input length and `max_tokens` exceed context limit: "
        "205000 + 8192 > 200000"
    )
    assert _is_context_overflow_error(err) is True


def test_ignores_unrelated_error() -> None:
    from boxbot.core.agent import _is_context_overflow_error

    assert _is_context_overflow_error("image exceeds 5 MB maximum") is False
    assert _is_context_overflow_error("some 500 server error") is False


# ---------------------------------------------------------------------------
# Context-overflow error path: compact + retry once (agent integration)
# ---------------------------------------------------------------------------


def _overflow_config():
    agent_ns = SimpleNamespace(
        compaction_enabled=True,
        compaction_threshold_tokens=150_000,
        compaction_keep_recent_tokens=30_000,
        max_turns=25,
        backend="raw_anthropic",
    )
    models_ns = SimpleNamespace(large="claude-sonnet-5", small="claude-haiku-4-5")
    return SimpleNamespace(agent=agent_ns, models=models_ns)


@pytest.mark.asyncio
async def test_overflow_compacts_and_retries(monkeypatch) -> None:
    from unittest.mock import AsyncMock

    import boxbot.core.agent as agent_module
    from boxbot.core.agent import BoxBotAgent, ContextOverflowError
    from boxbot.core.conversation import Conversation

    monkeypatch.setattr(agent_module, "get_config", _overflow_config)

    async def _noop_gen(_conv):
        return None

    agent = BoxBotAgent.__new__(BoxBotAgent)
    agent._client = object()  # only asserted non-None

    # Stub everything _generate_for_conversation needs except the retry.
    agent._compact_thread = AsyncMock(return_value=True)
    agent._build_system_prompt_blocks = AsyncMock(return_value=[])
    agent._prompt_dynamic_context = AsyncMock(return_value="")
    agent._get_most_recent_person = lambda: None
    agent._extract_summary = lambda messages: ""

    # First loop call overflows; the retry succeeds.
    agent._agent_loop = AsyncMock(side_effect=[
        ContextOverflowError("prompt is too long: 210000 tokens > 200000"),
        ([{"role": "user", "content": "hi"},
          {"role": "assistant", "content": "answer"}], 2),
    ])

    conv = Conversation(
        conversation_id="conv-of",
        channel="whatsapp",
        channel_key="whatsapp:+1",
        generate_fn=_noop_gen,
        rehydrated_thread=[{"role": "user", "content": "hi"}],
    )

    result = await agent._generate_for_conversation(conv)

    # Loop ran twice (overflow, then retry).
    assert agent._agent_loop.await_count == 2
    # Compaction ran twice: threshold pre-loop + aggressive on overflow.
    assert agent._compact_thread.await_count == 2
    aggressive = agent._compact_thread.await_args_list[1].kwargs
    assert aggressive["threshold_tokens"] == 30_000
    assert aggressive["keep_recent_tokens"] == 30_000
    # additions = produced turns beyond conv.thread (len 1).
    assert result.thread_additions == [
        {"role": "assistant", "content": "answer"}
    ]
