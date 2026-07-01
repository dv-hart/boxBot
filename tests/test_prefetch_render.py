"""Tests for PrefetchBundle rendering + budget truncation."""

from __future__ import annotations

from boxbot.prefetch.bundle import PrefetchBundle


class TestRender:
    def test_empty_bundle_is_empty(self):
        assert PrefetchBundle().is_empty()

    def test_note_and_memories_render(self):
        b = PrefetchBundle(
            memories=[("abcdef12", "likes tea")],
            likely_next_note="make tea",
        )
        out = b.render(token_budget=1500)
        assert "make tea" in out
        assert "#abcdef12" in out
        assert "likes tea" in out
        assert b.token_estimate > 0

    def test_budget_drops_bulky_skill_body(self):
        # Big skill body is the lowest-priority section; a tiny budget must
        # keep the high-value note/memories and drop the skill body.
        big = "x" * 40_000  # ~10k tokens
        b = PrefetchBundle(
            memories=[("m1", "keep me")],
            likely_next_note="do the thing",
            skill_bodies={"weather": big},
        )
        out = b.render(token_budget=200)
        assert "do the thing" in out
        assert "Skill `weather`" not in out
        assert b.token_estimate <= 200

    def test_roundtrip_dict(self):
        b = PrefetchBundle(
            memories=[("m1", "s1")],
            workspace_excerpts=[("notes/a.md", "hello")],
            pulled_data=[{"source": "weather", "action": None,
                          "payload": {"t": 12}, "pulled_at": "now"}],
            likely_next_note="n",
        )
        b2 = PrefetchBundle.from_dict(b.to_dict())
        assert b2.memories == [("m1", "s1")]
        assert b2.workspace_excerpts == [("notes/a.md", "hello")]
        assert b2.predicted_integration_calls()[0]["source"] == "weather"
