"""Tests for per-message dream-batch cost recording in DreamPoller.

The dream poller used to write a single synthetic-estimate row per
cycle (~2K input + 200 output tokens times the number of decisions).
That was wrong by up to 50%. It now reads the real ``usage`` returned
by every successful batch message and writes one ``cost_log`` row per
message via :func:`boxbot.cost.from_anthropic_usage` + ``record``.

These tests exercise that path against a fresh ``MemoryStore`` and a
hand-built fake batch result with three messages, each carrying a
distinct ``usage`` block.
"""

from __future__ import annotations

import inspect
import json
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

from boxbot.cost import reload_pricing


# ---------------------------------------------------------------------------
# Pricing fixture — same shape as tests/test_cost.py so this file can run
# standalone with --noconftest if conftest depends on absent ML deps.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _pricing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    yaml_text = """
anthropic:
  source_url: https://example.test/anthropic
  verified_on: 2026-05-02
  models:
    claude-opus-4-7:
      input_per_mtok: 15.00
      output_per_mtok: 75.00
    claude-haiku-4-5:
      input_per_mtok: 1.00
      output_per_mtok: 5.00
elevenlabs:
  source_url: https://example.test/elevenlabs
  verified_on: 2026-05-02
  tts:
    eleven_turbo_v2_5:
      dollars_per_char: 0.0000250
  stt:
    scribe_v2:
      dollars_per_minute: 0.003600
"""
    path = tmp_path / "pricing.yaml"
    path.write_text(yaml_text)
    monkeypatch.setenv("BOXBOT_PRICING_CONFIG", str(path))
    reload_pricing(path)
    yield
    # Restore canonical pricing so later test files don't read this
    # file's reduced model list out of the cached singleton.
    monkeypatch.delenv("BOXBOT_PRICING_CONFIG", raising=False)
    reload_pricing()


# ---------------------------------------------------------------------------
# Stubs — minimal Anthropic-batch shapes the poller reads.
# ---------------------------------------------------------------------------


class _StubBlock:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _StubMessage:
    def __init__(self, content, usage):
        self.content = content
        self.usage = usage


class _StubResult:
    def __init__(self, type_, message=None):
        self.type = type_
        self.message = message


class _StubResultEntry:
    def __init__(self, custom_id, result):
        self.custom_id = custom_id
        self.result = result


class _StubBatch:
    def __init__(self, batch_id, status="ended"):
        self.id = batch_id
        self.processing_status = status


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
    """Smallest possible stand-in for AsyncAnthropic for the poller."""

    def __init__(self, batch_id: str, entries: list[_StubResultEntry]):
        self._batch_id = batch_id
        self._entries = entries
        self.messages = types.SimpleNamespace(
            batches=types.SimpleNamespace(
                retrieve=self._retrieve,
                results=self._results,
            )
        )

    async def _retrieve(self, batch_id):
        return _StubBatch(batch_id, status="ended")

    async def _results(self, batch_id):
        return _AsyncIter(self._entries)


def _entry(custom_id: str, evidence_a: str, evidence_b: str, usage: dict) -> _StubResultEntry:
    msg = _StubMessage(
        content=[
            _StubBlock(
                type="tool_use",
                name="dedup_decision",
                input={
                    "decision": "merge_into_a",
                    "merged_content": "merged",
                    "merged_summary": "merged",
                    "evidence": [evidence_a, evidence_b],
                    "confidence": 0.95,
                    "notes": "",
                },
            )
        ],
        usage=usage,
    )
    return _StubResultEntry(custom_id, _StubResult("succeeded", message=msg))


# ---------------------------------------------------------------------------
# Store fixture (mirrors test_dream_phase.fresh_store but standalone).
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def fresh_store(tmp_path):
    import boxbot.core  # noqa: F401  (break import cycle)
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_cost_row_per_batch_message(fresh_store, tmp_path):
    """Three batch messages → exactly three dream rows in cost_log."""
    from boxbot.memory.dream_poller import DreamPoller

    a = await fresh_store.create_memory(
        type="person", content="aa", summary="aa",
    )
    b = await fresh_store.create_memory(
        type="person", content="bb", summary="bb",
    )
    await fresh_store.create_pending_dream(
        batch_id="msgbatch_three",
        candidate_ids=[a, b],
        request_types={"dedup": 3},
    )

    # Three messages with three distinct usage blocks.
    entries = [
        _entry("dream-x-0000", a, b, {
            "input_tokens": 1500,
            "output_tokens": 150,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }),
        _entry("dream-x-0001", a, b, {
            "input_tokens": 2500,
            "output_tokens": 250,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }),
        _entry("dream-x-0002", a, b, {
            "input_tokens": 3500,
            "output_tokens": 350,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }),
    ]
    fake_client = FakeAnthropicClient("msgbatch_three", entries)

    with patch(
        "boxbot.memory.dream.WORKSPACE_DIR", tmp_path / "workspace",
    ):
        poller = DreamPoller(
            fresh_store, fake_client,
            audit_only=True,
            model="claude-opus-4-7",
        )
        # Drive one sweep directly — no background loop, so we know the
        # batch is processed exactly once.
        await poller._sweep_once()

    cur = await fresh_store.db.execute(
        """SELECT purpose, model, is_batch, input_tokens, output_tokens,
                  cost_usd, correlation_id, metadata
           FROM cost_log
           WHERE purpose = 'dream'
           ORDER BY input_tokens"""
    )
    rows = await cur.fetchall()
    assert len(rows) == 3, f"expected 3 dream rows, got {len(rows)}"
    for row in rows:
        purpose, model, is_batch, in_tok, out_tok, cost, corr, meta = row
        assert purpose == "dream"
        assert model == "claude-opus-4-7"
        assert is_batch == 1
        assert corr == "msgbatch_three"
        assert json.loads(meta) == {"decisions": 1}
        assert cost > 0

    # Real per-message tokens captured (not a uniform synthetic value).
    in_tokens = [r[3] for r in rows]
    out_tokens = [r[4] for r in rows]
    assert in_tokens == [1500, 2500, 3500]
    assert out_tokens == [150, 250, 350]


@pytest.mark.asyncio
async def test_cost_uses_real_pricing_with_batch_discount(fresh_store, tmp_path):
    """1M input × $15/MTok × 0.5 batch discount = $7.50, exactly."""
    from boxbot.memory.dream_poller import DreamPoller

    a = await fresh_store.create_memory(
        type="person", content="aa", summary="aa",
    )
    b = await fresh_store.create_memory(
        type="person", content="bb", summary="bb",
    )
    await fresh_store.create_pending_dream(
        batch_id="msgbatch_one",
        candidate_ids=[a, b],
        request_types={"dedup": 1},
    )

    entries = [
        _entry("dream-y-0000", a, b, {
            "input_tokens": 1_000_000,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }),
    ]
    fake_client = FakeAnthropicClient("msgbatch_one", entries)

    with patch(
        "boxbot.memory.dream.WORKSPACE_DIR", tmp_path / "workspace",
    ):
        poller = DreamPoller(
            fresh_store, fake_client,
            audit_only=True,
            model="claude-opus-4-7",
        )
        await poller._sweep_once()

    cur = await fresh_store.db.execute(
        "SELECT cost_usd FROM cost_log WHERE purpose = 'dream'"
    )
    rows = await cur.fetchall()
    assert len(rows) == 1
    # $15/MTok × 1M input × 0.5 batch discount = $7.50.
    assert rows[0][0] == pytest.approx(7.50)


def test_synthetic_record_cycle_cost_path_is_gone():
    """The old synthetic estimator must not exist anywhere on DreamPoller."""
    from boxbot.memory import dream_poller
    from boxbot.memory.dream_poller import DreamPoller

    # Module-level + class-level: neither should expose it.
    assert not hasattr(dream_poller, "_record_cycle_cost")
    assert not hasattr(DreamPoller, "_record_cycle_cost")

    # Defence in depth: ensure no source line still references it.
    src = inspect.getsource(dream_poller)
    assert "_record_cycle_cost" not in src
    # And no leftover synthetic constants.
    assert "2000 * n" not in src
    assert "200 * n" not in src
