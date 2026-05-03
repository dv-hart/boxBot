"""Tests for the unified cost-tracking module (``boxbot.cost``).

Covers: pricing YAML load, per-provider helpers, the legacy
record_cost shim, and the new schema. Pricing is overridden via the
``BOXBOT_PRICING_CONFIG`` env var so tests do not depend on the
production ``config/pricing.yaml`` values.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from boxbot.cost import (
    CostEvent,
    from_agent_sdk_result,
    from_anthropic_usage,
    from_elevenlabs_stt,
    from_elevenlabs_tts,
    record,
    reload_pricing,
)


# ---------------------------------------------------------------------------
# Pricing fixture — small, deterministic YAML used by every test below.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _pricing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    yaml_text = """
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
    # Reset so other test files don't see the temp path; reload will
    # repopulate from the env var on next access if still set.
    monkeypatch.delenv("BOXBOT_PRICING_CONFIG", raising=False)


# ---------------------------------------------------------------------------
# Anthropic raw-usage helper
# ---------------------------------------------------------------------------


class TestFromAnthropicUsage:
    def test_basic_input_output(self):
        usage = SimpleNamespace(
            input_tokens=1_000_000,
            output_tokens=500_000,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        event = from_anthropic_usage(
            purpose="t", model="claude-opus-4-7", usage=usage
        )
        # 1M input @ $10 + 0.5M output @ $40 = $10 + $20 = $30
        assert event.cost_usd == pytest.approx(30.0)
        assert event.input_tokens == 1_000_000
        assert event.output_tokens == 500_000
        assert event.provider == "anthropic"

    def test_cache_read_is_ten_percent_of_input(self):
        usage = SimpleNamespace(
            input_tokens=0, output_tokens=0,
            cache_read_input_tokens=1_000_000,
            cache_creation_input_tokens=0,
        )
        event = from_anthropic_usage(
            purpose="t", model="claude-opus-4-7", usage=usage
        )
        # 1M cache_read tokens @ 0.10 × $10 = $1.00
        assert event.cost_usd == pytest.approx(1.0)
        assert event.cache_read_tokens == 1_000_000

    def test_flat_cache_creation_treated_as_1h(self):
        # Flat int is the older shape; preserve historical 1h-TTL assumption.
        usage = SimpleNamespace(
            input_tokens=0, output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=1_000_000,
        )
        event = from_anthropic_usage(
            purpose="t", model="claude-opus-4-7", usage=usage
        )
        # 1M @ 2.0 × $10 = $20
        assert event.cost_usd == pytest.approx(20.0)
        assert event.cache_write_5m_tokens == 0
        assert event.cache_write_1h_tokens == 1_000_000

    def test_structured_cache_creation_splits_5m_and_1h(self):
        usage = {
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 1_000_000,
                "ephemeral_1h_input_tokens": 1_000_000,
            },
        }
        event = from_anthropic_usage(
            purpose="t", model="claude-opus-4-7", usage=usage
        )
        # 1M @ 1.25 × $10 + 1M @ 2.0 × $10 = $12.50 + $20 = $32.50
        assert event.cost_usd == pytest.approx(32.50)
        assert event.cache_write_5m_tokens == 1_000_000
        assert event.cache_write_1h_tokens == 1_000_000

    def test_batch_applies_50pct_discount(self):
        usage = SimpleNamespace(
            input_tokens=1_000_000, output_tokens=0,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        event = from_anthropic_usage(
            purpose="t", model="claude-opus-4-7", usage=usage, is_batch=True,
        )
        # $10 × 0.5 = $5
        assert event.cost_usd == pytest.approx(5.0)
        assert event.is_batch is True

    def test_unknown_model_records_zero_cost(self, caplog):
        usage = SimpleNamespace(
            input_tokens=1000, output_tokens=1000,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        event = from_anthropic_usage(
            purpose="t", model="claude-fake-9000", usage=usage
        )
        assert event.cost_usd == 0.0
        # Tokens still captured for diagnostics.
        assert event.input_tokens == 1000

    def test_dict_usage_works_like_object(self):
        usage = {
            "input_tokens": 1000, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        }
        event = from_anthropic_usage(
            purpose="t", model="claude-haiku-4-5", usage=usage
        )
        # 1000 in × $1 / 1M = $0.001
        assert event.cost_usd == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Agent SDK ResultMessage helper
# ---------------------------------------------------------------------------


class TestFromAgentSdkResult:
    def test_uses_total_cost_usd_verbatim_for_single_model(self):
        result = SimpleNamespace(
            total_cost_usd=0.4242,
            usage={
                "input_tokens": 100, "output_tokens": 50,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            model="claude-opus-4-7",
            model_usage=None,
        )
        events = from_agent_sdk_result(purpose="conversation", result_message=result)
        assert len(events) == 1
        # SDK's dollar value flows through unchanged — not recomputed locally.
        assert events[0].cost_usd == pytest.approx(0.4242)
        assert events[0].input_tokens == 100
        assert events[0].purpose == "conversation"

    def test_per_model_breakdown_emits_one_event_per_model(self):
        result = SimpleNamespace(
            total_cost_usd=0.99,
            usage=None,
            model="claude-opus-4-7",
            model_usage={
                "claude-opus-4-7": {
                    "input_tokens": 1_000_000, "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
                "claude-haiku-4-5": {
                    "input_tokens": 1_000_000, "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        )
        events = from_agent_sdk_result(purpose="conversation", result_message=result)
        assert len(events) == 2
        models = {e.model for e in events}
        assert models == {"claude-opus-4-7", "claude-haiku-4-5"}
        # Per-model costs computed from pricing (no per-model SDK cost).
        opus = next(e for e in events if e.model == "claude-opus-4-7")
        haiku = next(e for e in events if e.model == "claude-haiku-4-5")
        assert opus.cost_usd == pytest.approx(10.0)
        assert haiku.cost_usd == pytest.approx(1.0)

    def test_per_model_cost_overrides_local_compute_when_present(self):
        result = SimpleNamespace(
            total_cost_usd=42.0,
            usage=None,
            model="claude-opus-4-7",
            model_usage={
                "claude-opus-4-7": {
                    "input_tokens": 1_000_000, "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cost_usd": 11.50,
                },
            },
        )
        events = from_agent_sdk_result(purpose="conversation", result_message=result)
        assert len(events) == 1
        assert events[0].cost_usd == pytest.approx(11.50)


# ---------------------------------------------------------------------------
# ElevenLabs helpers
# ---------------------------------------------------------------------------


class TestElevenLabs:
    def test_tts_uses_billed_chars(self):
        event = from_elevenlabs_tts(model="eleven_turbo_v2_5", billed_chars=4000)
        # 4000 × $0.0000250 = $0.10
        assert event.cost_usd == pytest.approx(0.10)
        assert event.character_count == 4000
        assert event.provider == "elevenlabs"
        assert event.purpose == "tts"

    def test_stt_uses_audio_seconds(self):
        event = from_elevenlabs_stt(model="scribe_v2", audio_seconds=60.0)
        # 1 minute × $0.003600 = $0.0036
        assert event.cost_usd == pytest.approx(0.0036)
        assert event.audio_seconds == 60.0
        assert event.purpose == "stt"

    def test_unknown_tts_model_zeroes_cost(self):
        event = from_elevenlabs_tts(model="fake_model", billed_chars=4000)
        assert event.cost_usd == 0.0
        # Units still captured.
        assert event.character_count == 4000


# ---------------------------------------------------------------------------
# Schema + record() integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSchemaAndRecord:
    async def test_fresh_db_has_new_columns(self, tmp_path):
        from boxbot.memory.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "m.db")
        await store.initialize()
        try:
            cur = await store.db.execute("PRAGMA table_info(cost_log)")
            cols = {row[1] for row in await cur.fetchall()}
            for expected in (
                "provider",
                "cache_write_5m_tokens",
                "cache_write_1h_tokens",
                "cache_write_tokens",
                "character_count",
                "audio_seconds",
                "iterations",
                "correlation_id",
            ):
                assert expected in cols, f"missing column: {expected}"
        finally:
            await store.close()

    async def test_record_writes_all_fields(self, tmp_path):
        from boxbot.memory.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "m.db")
        await store.initialize()
        try:
            event = CostEvent(
                purpose="conversation",
                provider="anthropic",
                model="claude-opus-4-7",
                cost_usd=0.0123,
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=2,
                cache_write_5m_tokens=3,
                cache_write_1h_tokens=4,
                is_batch=False,
                iterations=1,
                correlation_id="conv-xyz",
                metadata={"turn": 1},
            )
            await record(store, event)

            cur = await store.db.execute(
                """SELECT purpose, provider, model, cost_usd,
                          input_tokens, output_tokens,
                          cache_read_tokens, cache_write_5m_tokens,
                          cache_write_1h_tokens, cache_write_tokens,
                          iterations, correlation_id, metadata
                   FROM cost_log"""
            )
            rows = await cur.fetchall()
            assert len(rows) == 1
            r = rows[0]
            assert r[0] == "conversation"
            assert r[1] == "anthropic"
            assert r[2] == "claude-opus-4-7"
            assert r[3] == pytest.approx(0.0123)
            assert r[4] == 10
            assert r[5] == 5
            assert r[6] == 2
            assert r[7] == 3
            assert r[8] == 4
            # Legacy column dual-written = 5m + 1h.
            assert r[9] == 7
            assert r[10] == 1
            assert r[11] == "conv-xyz"
            assert r[12] == '{"turn": 1}'
        finally:
            await store.close()

    async def test_legacy_record_cost_writes_new_columns(self, tmp_path):
        from boxbot.memory.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "m.db")
        await store.initialize()
        try:
            await store.record_cost(
                purpose="extraction",
                model="claude-haiku-4-5",
                input_tokens=100,
                output_tokens=50,
                cache_read_tokens=10,
                cache_write_tokens=20,
                is_batch=True,
                cost_usd=0.001,
            )
            cur = await store.db.execute(
                """SELECT provider, cache_write_1h_tokens, cache_write_tokens,
                          is_batch
                   FROM cost_log"""
            )
            rows = await cur.fetchall()
            assert len(rows) == 1
            provider, cw_1h, cw_legacy, is_batch = rows[0]
            # Legacy shim assumes 1h TTL (matches old compute_cost behaviour).
            assert provider == "anthropic"
            assert cw_1h == 20
            assert cw_legacy == 20
            assert is_batch == 1
        finally:
            await store.close()
