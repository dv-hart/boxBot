"""Tests for PrefetchBundle rendering: nothing truncated, sizes logged."""

from __future__ import annotations

import logging

from boxbot.prefetch.bundle import PrefetchBundle


class TestRender:
    def test_empty_bundle_is_empty(self):
        assert PrefetchBundle().is_empty()
        assert PrefetchBundle().render(token_budget=20000) == ""

    def test_sections_render(self):
        b = PrefetchBundle(
            memories=[("abcdef12", "likes tea")],
            skill_bodies={"household-support": "the skill body"},
            sdk_modules={"panel": "the panel doc"},
            workspace_excerpts=[("notes/a.md", "hello")],
        )
        out = b.render(token_budget=20000)
        assert "#abcdef12" in out and "likes tea" in out
        assert "Skill `household-support`" in out and "the skill body" in out
        assert "bb module `panel`" in out and "the panel doc" in out
        assert "Workspace `notes/a.md`" in out
        assert b.token_estimate > 0

    def test_over_budget_logs_but_never_truncates(self, caplog):
        big = "x" * 40_000  # ~10k tokens
        b = PrefetchBundle(
            memories=[("m1", "keep me")],
            skill_bodies={"weather": big},
        )
        with caplog.at_level(logging.WARNING, logger="boxbot.prefetch.bundle"):
            out = b.render(token_budget=200)
        # Everything survives — the budget is a log threshold only.
        assert "keep me" in out
        assert "Skill `weather`" in out
        assert big in out
        assert b.token_estimate > 200
        assert any("exceeds budget" in r.message for r in caplog.records)

    def test_roundtrip_dict(self):
        b = PrefetchBundle(
            memories=[("m1", "s1")],
            sdk_modules={"panel": "doc"},
            workspace_excerpts=[("notes/a.md", "hello")],
            pulled_data=[{"source": "weather", "action": None,
                          "payload": {"t": 12}, "pulled_at": "now"}],
        )
        b2 = PrefetchBundle.from_dict(b.to_dict())
        assert b2.memories == [("m1", "s1")]
        assert b2.sdk_modules == {"panel": "doc"}
        assert b2.workspace_excerpts == [("notes/a.md", "hello")]
        assert b2.predicted_integration_calls()[0]["source"] == "weather"

    def test_from_dict_tolerates_legacy_cache_rows(self):
        # Cached bundles written by the pre-fanout code carried extra
        # keys (likely_next_note, history_highlights) and no sdk_modules.
        legacy = {
            "memories": [["m1", "s1"]],
            "skill_bodies": {},
            "workspace_excerpts": [],
            "history_highlights": ["old"],
            "pulled_data": [],
            "likely_next_note": "n",
            "token_estimate": 94,
        }
        b = PrefetchBundle.from_dict(legacy)
        assert b.memories == [("m1", "s1")]
        assert b.sdk_modules == {}
