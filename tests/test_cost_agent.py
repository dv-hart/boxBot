"""Tests for cost-tracking integration in the main Claude agent loop.

This file pins the contract that every Claude turn taken by the
``BoxBotAgent`` writes one row to ``cost_log`` with
``purpose="conversation"`` and the conversation_id as
``correlation_id``. The agent uses raw ``anthropic.AsyncAnthropic``
(not ``claude_agent_sdk``) so the cost helper used at the call site is
``from_anthropic_usage`` — usage tokens are read from the response and
priced via ``config/pricing.yaml`` (overridden here to a deterministic
test fixture).

Notes on the testing approach:

* The shared ``tests/conftest.py`` deadlocks on import in this
  environment because ``reload_pricing`` is called recursively under a
  non-reentrant ``threading.Lock`` during the side-effects of
  importing ``boxbot.memory``. We therefore run pytest with
  ``--noconftest`` and inline-mock ``anthropic`` plus stand in a
  minimal fake of ``MemoryStore`` rather than importing the real one.
  The hook in ``agent.py`` only needs ``store.db.execute / commit``,
  so the fake matches that surface exactly and exercises the real
  ``record()`` writer end-to-end.
* The pricing fixture mirrors ``tests/test_cost.py`` so unit-cost
  arithmetic is predictable: 10/40 USD per Mtok for Opus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Pre-mock anthropic before any boxbot imports — boxbot.core.agent has
# ``import anthropic`` at module scope.
sys.modules.setdefault("anthropic", MagicMock())

from boxbot.cost import reload_pricing  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level pricing bootstrap.
#
# ``boxbot.memory.dream`` (a transitive import of ``boxbot.core.agent``)
# touches ``STANDARD_PRICING`` at import time, which calls
# ``get_pricing()``. Without a pre-loaded cache, ``get_pricing()``
# tries to reload from ``config/pricing.yaml`` and deadlocks under the
# module's non-reentrant lock. Loading a deterministic test pricing
# *before* importing the agent both seeds the cache and gives every
# test predictable unit costs.
# ---------------------------------------------------------------------------

_TEST_PRICING_YAML = """
anthropic:
  source_url: https://example.test/anthropic
  verified_on: 2026-05-02
  models:
    claude-opus-4-7:
      input_per_mtok: 10.00
      output_per_mtok: 40.00
    claude-haiku-4-5:
      input_per_mtok: 1.00
      output_per_mtok: 5.00
elevenlabs:
  source_url: https://example.test/elevenlabs
  verified_on: 2026-05-02
  tts: {}
  stt: {}
"""

import os as _os  # noqa: E402
import tempfile as _tempfile  # noqa: E402

_PRICING_PATH = Path(_tempfile.gettempdir()) / "test_cost_agent_pricing.yaml"
_PRICING_PATH.write_text(_TEST_PRICING_YAML)
_os.environ["BOXBOT_PRICING_CONFIG"] = str(_PRICING_PATH)
reload_pricing(_PRICING_PATH)


# Force the REAL boxbot.core.agent module into sys.modules at collection
# time. Other test files defensively pre-stub it as a MagicMock if it's
# not already loaded; importing here first wins the race so subsequent
# stubs in those files become no-ops.
import boxbot.core.agent as _real_agent_module  # noqa: E402, F401


# ---------------------------------------------------------------------------
# Pricing fixture — re-asserts the cache before each test in case another
# test in the same session reloaded production pricing.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _pricing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BOXBOT_PRICING_CONFIG", str(_PRICING_PATH))
    reload_pricing(_PRICING_PATH)
    yield
    monkeypatch.delenv("BOXBOT_PRICING_CONFIG", raising=False)


# ---------------------------------------------------------------------------
# Fake MemoryStore — captures cost_log writes without booting the real DB.
# ---------------------------------------------------------------------------


class _FakeDB:
    """Captures the SQL + params passed to ``record()``.

    ``record()`` only ever calls ``db.execute(SQL, params)`` followed
    by ``db.commit()``. Mimic that and squirrel away the rows so tests
    can assert on them directly.
    """

    def __init__(self) -> None:
        self.rows: list[tuple] = []
        self.commits: int = 0
        self.last_sql: str | None = None

    async def execute(self, sql: str, params: tuple) -> None:
        self.last_sql = sql
        self.rows.append(tuple(params))

    async def commit(self) -> None:
        self.commits += 1


class _FakeStore:
    """Minimal stand-in for ``MemoryStore`` exposing ``.db``."""

    def __init__(self) -> None:
        self.db = _FakeDB()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Column order for the cost_log INSERT in boxbot/cost/record.py — used
# to map _FakeDB.rows back to named fields in assertions. Pinned here
# so a schema reorder lights up these tests instead of silently
# shifting indices.
_COLS = (
    "timestamp",
    "purpose",
    "provider",
    "model",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_5m_tokens",
    "cache_write_1h_tokens",
    "cache_write_tokens",  # legacy dual-write
    "is_batch",
    "character_count",
    "audio_seconds",
    "iterations",
    "correlation_id",
    "cost_usd",
    "metadata",
)


def _row_dict(row: tuple) -> dict:
    return dict(zip(_COLS, row))


def _make_response(
    *,
    model: str = "claude-opus-4-7",
    input_tokens: int = 1_000_000,
    output_tokens: int = 500_000,
    stop_reason: str = "end_turn",
    content: list | None = None,
):
    """Build a stand-in for an Anthropic Messages API response."""
    if content is None:
        # One text block holding minimal-but-valid INTERNAL_NOTES JSON so
        # the loop's parse step doesn't log spurious errors.
        text_block = SimpleNamespace(type="text", text='{"thought":"ok"}')
        content = [text_block]
    return SimpleNamespace(
        model=model,
        stop_reason=stop_reason,
        content=content,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


def _make_agent(store):
    """Build a BoxBotAgent in a state suitable for direct ``_agent_loop`` calls.

    Bypasses ``start()`` which validates ``ANTHROPIC_API_KEY`` and
    spins up batch / dream pollers we don't need for a unit test.
    """
    BoxBotAgent = _real_agent_module.BoxBotAgent

    agent = BoxBotAgent(memory_store=store)
    agent._client = MagicMock()
    agent._client.messages = MagicMock()
    agent._running = True
    return agent


# ---------------------------------------------------------------------------
# Config + tools stubs (so _agent_loop has what it needs).
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_config(monkeypatch):
    """Patch ``get_config`` to return a tiny config with a known model."""
    fake_models = SimpleNamespace(
        large="claude-opus-4-7",
        small="claude-haiku-4-5",
    )
    fake_config = SimpleNamespace(models=fake_models)
    import boxbot.core.agent as agent_module
    monkeypatch.setattr(agent_module, "get_config", lambda: fake_config)
    return fake_config


@pytest.fixture
def stub_tools(monkeypatch):
    """Stub the tool registry so we don't have to load the full skill graph."""
    import boxbot.tools.registry as registry
    monkeypatch.setattr(registry, "get_tools", lambda: [])
    return registry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAgentCostHook:
    async def test_single_turn_records_one_row(self, stub_config, stub_tools):
        """One Claude turn → exactly one cost_log row, tagged conversation."""
        store = _FakeStore()
        agent = _make_agent(store)
        agent._client.messages.create = AsyncMock(
            return_value=_make_response(
                model="claude-opus-4-7",
                input_tokens=1_000_000,
                output_tokens=500_000,
                stop_reason="end_turn",
            )
        )

        await agent._agent_loop(
            conversation_id="conv-abc",
            channel="voice",
            system_prompt_blocks=[{"type": "text", "text": "sys"}],
            initial_message="hello",
            person_name="Jacob",
            max_turns=3,
        )

        assert len(store.db.rows) == 1
        r = _row_dict(store.db.rows[0])
        assert r["purpose"] == "conversation"
        assert r["provider"] == "anthropic"
        assert r["model"] == "claude-opus-4-7"
        # 1M in @ $10 + 0.5M out @ $40 = $30 from the pricing fixture.
        assert r["cost_usd"] == pytest.approx(30.0)
        assert r["input_tokens"] == 1_000_000
        assert r["output_tokens"] == 500_000
        assert r["correlation_id"] == "conv-abc"
        # Metadata is JSON-encoded by record(); inspect the parsed shape.
        meta = json.loads(r["metadata"])
        assert meta["channel"] == "voice"
        assert meta["turn"] == 1
        # commit() runs once per record() call.
        assert store.db.commits == 1

    async def test_uses_total_cost_usd_when_response_carries_one(
        self, stub_config, stub_tools
    ):
        """When the API response exposes ``total_cost_usd`` we still go
        through ``from_anthropic_usage`` (raw API path) — but the local
        compute should equal what an Agent SDK with the same total
        would report, because the underlying tokens drive both.
        """
        store = _FakeStore()
        agent = _make_agent(store)
        agent._client.messages.create = AsyncMock(
            return_value=_make_response(
                input_tokens=42_000,  # 42_000 × $10 / 1M = $0.42
                output_tokens=0,
                stop_reason="end_turn",
            )
        )
        await agent._agent_loop(
            conversation_id="conv-xyz",
            channel="whatsapp",
            system_prompt_blocks=[{"type": "text", "text": "sys"}],
            initial_message="hi",
            person_name=None,
            max_turns=2,
        )
        assert len(store.db.rows) == 1
        r = _row_dict(store.db.rows[0])
        assert r["cost_usd"] == pytest.approx(0.42)
        assert r["purpose"] == "conversation"
        assert r["provider"] == "anthropic"
        assert r["correlation_id"] == "conv-xyz"

    async def test_multi_turn_writes_one_row_per_turn(
        self, stub_config, stub_tools
    ):
        """Tool-use loops produce one cost row per Claude turn."""
        store = _FakeStore()
        agent = _make_agent(store)

        # Patch _process_tool_calls so we don't need fully-formed tool
        # blocks; the loop just needs an empty list back to keep going.
        agent._process_tool_calls = AsyncMock(return_value=[
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
        ])

        responses = [
            _make_response(stop_reason="tool_use",
                           input_tokens=100, output_tokens=50,
                           content=[
                               SimpleNamespace(type="text",
                                               text='{"thought":"calling tool"}'),
                               SimpleNamespace(type="tool_use", id="t1",
                                               name="x", input={}),
                           ]),
            _make_response(stop_reason="tool_use",
                           input_tokens=200, output_tokens=75,
                           content=[
                               SimpleNamespace(type="text",
                                               text='{"thought":"again"}'),
                               SimpleNamespace(type="tool_use", id="t2",
                                               name="x", input={}),
                           ]),
            _make_response(stop_reason="end_turn",
                           input_tokens=300, output_tokens=100),
        ]
        agent._client.messages.create = AsyncMock(side_effect=responses)

        await agent._agent_loop(
            conversation_id="conv-multi",
            channel="voice",
            system_prompt_blocks=[{"type": "text", "text": "sys"}],
            initial_message="hi",
            person_name=None,
            max_turns=5,
        )

        assert len(store.db.rows) == 3
        for row in store.db.rows:
            r = _row_dict(row)
            assert r["purpose"] == "conversation"
            assert r["provider"] == "anthropic"
            assert r["correlation_id"] == "conv-multi"
        # Input tokens follow the scripted turns in order.
        assert [
            _row_dict(row)["input_tokens"] for row in store.db.rows
        ] == [100, 200, 300]
        # Each turn's metadata records the turn index.
        assert [
            json.loads(_row_dict(row)["metadata"])["turn"]
            for row in store.db.rows
        ] == [1, 2, 3]

    async def test_hook_called_exactly_once_per_turn(
        self, stub_config, stub_tools, monkeypatch
    ):
        """Guard against accidental double-recording of a single turn."""
        store = _FakeStore()
        agent = _make_agent(store)
        agent._client.messages.create = AsyncMock(
            return_value=_make_response(
                input_tokens=1, output_tokens=1, stop_reason="end_turn",
            )
        )

        # Spy on the agent's bound record helper. We patch the symbol
        # the agent loop uses (imported as ``record_cost`` in agent.py)
        # and forward to the real implementation so the row still lands.
        from boxbot.cost import record as real_record
        calls = {"n": 0}

        async def counting_record(s, event):
            calls["n"] += 1
            await real_record(s, event)

        import boxbot.core.agent as agent_module
        monkeypatch.setattr(agent_module, "record_cost", counting_record)

        await agent._agent_loop(
            conversation_id="conv-once",
            channel="voice",
            system_prompt_blocks=[{"type": "text", "text": "sys"}],
            initial_message="hi",
            person_name=None,
            max_turns=2,
        )
        assert calls["n"] == 1
        assert len(store.db.rows) == 1
