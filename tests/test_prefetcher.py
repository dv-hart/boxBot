"""Tests for the prefetch mini-agent (``boxbot.prefetch.runner``) and its
read-only tool whitelist. Uses a fake Anthropic client (canned tool_use
turns) and monkeypatched search backends — no network, no real store.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from boxbot.prefetch.request import PrefetchRequest
from boxbot.prefetch.runner import run_prefetch
from boxbot.prefetch.tools import build_tool_definitions


# --- fakes -----------------------------------------------------------------


def _text(t):
    return SimpleNamespace(type="text", text=t, id=None, name=None, input=None)


def _tool_use(tid, name, inp):
    return SimpleNamespace(type="tool_use", id=tid, name=name, input=inp, text=None)


def _response(blocks, *, stop_reason="tool_use"):
    return SimpleNamespace(content=blocks, usage=None, stop_reason=stop_reason)


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def _candidate(cid, summary, ctype="person"):
    return SimpleNamespace(
        id=cid, source="memory", type=ctype, person=None,
        content="", summary=summary, metadata={},
    )


_CFG = SimpleNamespace(model=None, max_iterations=6, token_budget=1500)


# --- tests -----------------------------------------------------------------


class TestWhitelist:
    def test_no_write_tools_exposed(self):
        names = {t["name"] for t in build_tool_definitions()}
        # Read-only + the finalize sink; nothing that can affect the world.
        forbidden = {
            "message", "execute_script", "manage_tasks", "identify_person",
            "switch_display", "save_memory", "bb", "create_event",
        }
        assert not (names & forbidden)
        assert "finalize" in names
        assert "search_memory" in names


class TestRunPrefetch:
    async def test_assembles_bundle_from_finalize(self, monkeypatch):
        async def fake_hybrid(store, query, **kw):
            return [_candidate("mem-11111111", "Jacob likes strong tea")]

        monkeypatch.setattr(
            "boxbot.memory.search.hybrid_search", fake_hybrid,
        )

        client = _FakeClient([
            _response([_tool_use("t1", "search_memory", {"query": "tea"})]),
            _response([
                _tool_use("t2", "finalize", {
                    "memory_ids": ["mem-11111111"],
                    "likely_next_note": "make tea",
                })
            ]),
        ])
        req = PrefetchRequest(
            key="conv-1", key_kind="conversation", channel="whatsapp",
            person="Jacob", text="put the kettle on",
        )
        result = await run_prefetch(
            req, store=object(), client=client, config=_CFG,
        )
        assert result.bundle.predicted_memory_ids() == ["mem-11111111"]
        assert result.bundle.likely_next_note == "make tea"
        assert not result.bundle.is_empty()
        # Only picked-up memory is included — precision, not everything seen.
        assert result.bundle.memories == [("mem-11111111", "Jacob likes strong tea")]

    async def test_empty_when_model_finalizes_nothing(self, monkeypatch):
        async def fake_hybrid(store, query, **kw):
            return []

        monkeypatch.setattr(
            "boxbot.memory.search.hybrid_search", fake_hybrid,
        )
        client = _FakeClient([
            _response([_tool_use("t1", "finalize", {})]),
        ])
        req = PrefetchRequest(
            key="conv-2", key_kind="conversation", channel="signal",
            person=None, text="hi",
        )
        result = await run_prefetch(
            req, store=object(), client=client, config=_CFG,
        )
        assert result.bundle.is_empty()

    async def test_stops_on_end_turn_without_finalize(self, monkeypatch):
        client = _FakeClient([
            _response([_text("nothing needed")], stop_reason="end_turn"),
        ])
        req = PrefetchRequest(
            key="conv-3", key_kind="conversation", channel="whatsapp",
            text="ok",
        )
        result = await run_prefetch(
            req, store=object(), client=client, config=_CFG,
        )
        assert result.bundle.is_empty()
        assert result.iterations == 1

    async def test_respects_iteration_cap(self, monkeypatch):
        # Model keeps searching, never finalizes → must stop at the cap.
        async def fake_hybrid(store, query, **kw):
            return []

        monkeypatch.setattr(
            "boxbot.memory.search.hybrid_search", fake_hybrid,
        )
        cfg = SimpleNamespace(model=None, max_iterations=3, token_budget=1500)
        client = _FakeClient([
            _response([_tool_use(f"t{i}", "search_memory", {"query": "x"})])
            for i in range(10)
        ])
        req = PrefetchRequest(
            key="conv-4", key_kind="conversation", channel="whatsapp", text="x",
        )
        result = await run_prefetch(
            req, store=object(), client=client, config=cfg,
        )
        assert result.iterations == 3
        assert result.bundle.is_empty()
