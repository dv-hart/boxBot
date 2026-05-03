"""Tests for web_search cost recording.

The web_search tool runs a Haiku-powered sub-agent (a mini agent loop)
as a content firewall. Token usage is summed across iterations and a
single ``cost_log`` row is written per outer ``web_search`` invocation
— not one per inner Haiku turn. These tests cover that contract.

Pricing is overridden via the ``BOXBOT_PRICING_CONFIG`` env var so the
production ``config/pricing.yaml`` numbers cannot drift the assertions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ``anthropic`` is imported at the top of web_search.py via the lazy
# client builder, but other test runs may already have it mocked. Match
# the conftest pattern so this file works under ``--noconftest`` too.
if "anthropic" not in sys.modules:
    sys.modules["anthropic"] = MagicMock()

# ``boxbot.core.__init__`` eagerly imports the full agent stack
# (claude_agent_sdk, perception modules, etc.). For a tightly-scoped
# unit test that only needs ``web_search``, that import chain is
# overkill and can be unavailable in stripped-down envs. Stub the
# agent module before any ``boxbot.core.*`` import triggers it.
if "boxbot.core.agent" not in sys.modules:
    _stub_agent = MagicMock()
    _stub_agent.BoxBotAgent = MagicMock()
    sys.modules["boxbot.core.agent"] = _stub_agent

from boxbot.cost import reload_pricing
from boxbot.tools.builtins import web_search as ws


# ---------------------------------------------------------------------------
# Pricing fixture — small, deterministic YAML used by every test below.
# Mirrors tests/test_cost.py so we don't depend on production pricing.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _pricing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    yaml_text = """
anthropic:
  source_url: https://example.test/anthropic
  verified_on: 2026-05-02
  models:
    claude-haiku-4-5:
      input_per_mtok: 1.00
      output_per_mtok: 5.00
elevenlabs:
  source_url: https://example.test/elevenlabs
  verified_on: 2026-05-02
  tts: {}
  stt: {}
"""
    path = tmp_path / "pricing.yaml"
    path.write_text(yaml_text)
    monkeypatch.setenv("BOXBOT_PRICING_CONFIG", str(path))
    reload_pricing(path)
    yield
    monkeypatch.delenv("BOXBOT_PRICING_CONFIG", raising=False)


# ---------------------------------------------------------------------------
# Cost-store fixture — point the web_search singleton at a temp DB so we
# can read back the row it wrote without touching real boxBot data.
# ---------------------------------------------------------------------------


@pytest.fixture
def _isolated_cost_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Reset the web_search cost-store singleton between tests."""
    monkeypatch.setattr(ws, "_cost_store", None, raising=False)

    from boxbot.memory.store import MemoryStore

    db_path = tmp_path / "cost.db"
    store = MemoryStore(db_path=db_path)

    async def _factory():
        if store._db is None:
            await store.initialize()
        return store

    monkeypatch.setattr(ws, "_get_cost_store", _factory, raising=True)
    yield store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecordWebSearchCost:
    async def test_three_iteration_run_writes_one_row(
        self, _isolated_cost_store
    ):
        """Sub-agent ran 3 Haiku turns; the cost_log gets exactly one row."""
        store = _isolated_cost_store

        # Summed usage across 3 inner Haiku turns. The helper does not
        # care how many turns produced the totals — it sees the totals.
        summed_usage = {
            "input_tokens": 30_000,
            "output_tokens": 1_500,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

        await ws._record_web_search_cost(
            usage=summed_usage,
            iterations=3,
            model="claude-haiku-4-5",
            query="What is the weather in Berlin tomorrow?",
            url=None,
        )

        # One row, one row only.
        cur = await store.db.execute(
            """SELECT purpose, provider, model, iterations,
                      input_tokens, output_tokens, cost_usd
               FROM cost_log"""
        )
        rows = await cur.fetchall()
        assert len(rows) == 1
        purpose, provider, model, iterations, in_tok, out_tok, cost = rows[0]
        assert purpose == "web_search"
        assert provider == "anthropic"
        assert model == "claude-haiku-4-5"
        assert iterations == 3
        assert in_tok == 30_000
        assert out_tok == 1_500
        # 30k input @ $1/M = $0.03; 1.5k output @ $5/M = $0.0075; total $0.0375.
        assert cost == pytest.approx(0.0375)

    async def test_correlation_id_matches_parent_conversation(
        self, _isolated_cost_store, monkeypatch
    ):
        """The conversation_id from the active Conversation flows into the row."""
        store = _isolated_cost_store

        # Stand up a fake conversation in the ContextVar — the helper
        # reads ``current_conversation`` and treats whatever it finds
        # there as the parent turn.
        fake_conv = SimpleNamespace(conversation_id="conv-abc-123")
        from boxbot.tools._tool_context import current_conversation

        token = current_conversation.set(fake_conv)
        try:
            await ws._record_web_search_cost(
                usage={
                    "input_tokens": 1000,
                    "output_tokens": 100,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
                iterations=1,
                model="claude-haiku-4-5",
                query="ping",
                url=None,
            )
        finally:
            current_conversation.reset(token)

        cur = await store.db.execute(
            "SELECT correlation_id, metadata FROM cost_log"
        )
        rows = await cur.fetchall()
        assert len(rows) == 1
        correlation_id, metadata_json = rows[0]
        assert correlation_id == "conv-abc-123"
        # Query is captured for diagnostics in metadata, truncated to
        # 200 chars by the helper.
        meta = json.loads(metadata_json)
        assert meta["query"] == "ping"

    async def test_legacy_log_line_no_longer_emits_token_counts(self, caplog):
        """The legacy ``logger.info`` cost-summary line is gone.

        Old line shape:
            "web_search done elapsed=... tokens_in=N tokens_out=N
             cache_read=N cache_write=N"

        It was replaced by a structured cost_log row. The remaining
        ``web_search done`` log line must not carry token fields any
        more.
        """
        # Read the source rather than running the tool — token-count
        # phrasing in the legacy line is what matters here.
        source = Path(ws.__file__).read_text()
        # The legacy format string is a single concatenated literal in
        # the file. If any of its sub-strings remain, the log line
        # wasn't actually removed.
        assert "tokens_in=%d" not in source
        assert "tokens_out=%d" not in source
        assert "cache_read=%d" not in source
        assert "cache_write=%d" not in source

    async def test_empty_usage_writes_no_row(self, _isolated_cost_store):
        """No Haiku usage (e.g. early API-key failure) means no row.

        The cost_log is for billed events; when the sub-agent never made
        a call there is nothing to bill.
        """
        store = _isolated_cost_store
        # Pre-initialise so the table exists for the assertion below
        # even if the helper short-circuits.
        await store.initialize()
        await ws._record_web_search_cost(
            usage={},
            iterations=0,
            model="claude-haiku-4-5",
            query="something",
            url=None,
        )
        cur = await store.db.execute("SELECT COUNT(*) FROM cost_log")
        rows = await cur.fetchall()
        assert rows[0][0] == 0
