"""Tests for the presence surface — [Present: ...] formatting + debouncer.

ws5: presence header + mid-conversation update machinery
(src/boxbot/perception/presence.py).
"""

from __future__ import annotations

import pytest

from boxbot.perception.presence import (
    PresenceDebouncer,
    format_presence_entries,
    format_presence_line,
)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestPresenceFormatting:
    def test_empty_returns_none(self):
        assert format_presence_line([]) is None

    def test_confirmed_high_tier(self):
        people = [{
            "ref": "Person A", "name": "Jacob",
            "confidence": 0.91, "tier": "high",
        }]
        assert format_presence_line(people) == "[Present: Jacob (confirmed)]"

    def test_likely_medium_tier(self):
        people = [{
            "ref": "Person A", "name": "Sarah",
            "confidence": 0.7, "tier": "medium",
        }]
        assert format_presence_line(people) == "[Present: Sarah (likely)]"

    def test_likely_low_tier(self):
        people = [{
            "ref": "Person A", "name": "Sarah",
            "confidence": 0.62, "tier": "low",
        }]
        assert format_presence_entries(people) == ["Sarah (likely)"]

    def test_unnamed_is_new_with_ref(self):
        people = [{"ref": "Person B", "name": None, "confidence": 0.0}]
        assert format_presence_line(people) == "[Present: Person B (new)]"

    def test_missing_ref_falls_back_to_unknown(self):
        assert format_presence_entries([{}]) == ["unknown (new)"]

    def test_docs_example_shape(self):
        """Matches the documented header exactly (docs/perception.md)."""
        people = [
            {"ref": "Person A", "name": "Jacob",
             "confidence": 0.9, "tier": "high"},
            {"ref": "Person B", "name": None, "confidence": 0.0},
        ]
        assert (
            format_presence_line(people)
            == "[Present: Jacob (confirmed), Person B (new)]"
        )

    def test_no_tier_falls_back_to_confidence(self):
        """Entries without a tier key (older callers) use the 0.8 cut."""
        assert format_presence_entries(
            [{"ref": "P", "name": "Jacob", "confidence": 0.85}]
        ) == ["Jacob (confirmed)"]
        assert format_presence_entries(
            [{"ref": "P", "name": "Jacob", "confidence": 0.7}]
        ) == ["Jacob (likely)"]


# ---------------------------------------------------------------------------
# Debouncer
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class TestPresenceDebouncer:
    def _make(self, **kw):
        clock = FakeClock()
        deb = PresenceDebouncer(
            stable_seconds=kw.pop("stable_seconds", 7.0),
            min_interval_seconds=kw.pop("min_interval_seconds", 30.0),
            clock=clock,
        )
        return deb, clock

    def test_not_ready_before_stable_window(self):
        deb, clock = self._make()
        deb.offer(("Jacob (confirmed)",))
        assert deb.ready() is None
        clock.advance(6.9)
        assert deb.ready() is None

    def test_ready_after_stable_window(self):
        deb, clock = self._make()
        snap = ("Jacob (confirmed)",)
        deb.offer(snap)
        clock.advance(7.1)
        assert deb.ready() == snap

    def test_flicker_resets_stability(self):
        """A change mid-window restarts the clock — flicker never fires."""
        deb, clock = self._make()
        deb.offer(("Jacob (confirmed)",))
        clock.advance(5.0)
        deb.offer(("Jacob (confirmed)", "Person B (new)"))  # flicker in
        clock.advance(5.0)
        deb.offer(("Jacob (confirmed)",))  # flicker out
        clock.advance(5.0)
        assert deb.ready() is None  # only 5s stable
        clock.advance(2.5)
        assert deb.ready() == ("Jacob (confirmed)",)

    def test_same_snapshot_not_reannounced(self):
        deb, clock = self._make()
        snap = ("Jacob (confirmed)",)
        deb.offer(snap)
        clock.advance(8.0)
        assert deb.ready() == snap
        deb.mark_announced(snap)
        clock.advance(100.0)
        deb.offer(snap)
        clock.advance(100.0)
        assert deb.ready() is None

    def test_min_interval_rate_limits(self):
        deb, clock = self._make()
        a = ("Jacob (confirmed)",)
        b = ("Jacob (confirmed)", "Sarah (likely)")
        deb.offer(a)
        clock.advance(8.0)
        deb.mark_announced(deb.ready())
        # New change immediately after — stable but inside min interval.
        deb.offer(b)
        clock.advance(8.0)
        assert deb.ready() is None
        clock.advance(30.0)
        assert deb.ready() == b

    def test_sync_baseline_blocks_identical_but_not_changes(self):
        """Baseline from the conversation-start header suppresses a
        re-announce of the same set without rate-limiting real changes."""
        deb, clock = self._make()
        a = ("Jacob (confirmed)",)
        deb.sync_baseline(a)
        deb.offer(a)
        clock.advance(10.0)
        assert deb.ready() is None  # agent already saw this set
        b = ("Jacob (confirmed)", "Person B (new)")
        deb.offer(b)
        clock.advance(7.5)
        assert deb.ready() == b  # not blocked by min-interval

    def test_offer_unchanged_does_not_reset_window(self):
        deb, clock = self._make()
        snap = ("Jacob (confirmed)",)
        deb.offer(snap)
        clock.advance(4.0)
        deb.offer(snap)  # heartbeat re-offer, same set
        clock.advance(3.5)
        assert deb.ready() == snap

    def test_no_candidate_never_ready(self):
        deb, clock = self._make()
        clock.advance(100.0)
        assert deb.ready() is None


# ---------------------------------------------------------------------------
# State machine exposes tier for presence formatting
# ---------------------------------------------------------------------------


class TestStateMachineTier:
    """Needs cv2 (state_machine -> person_detector); skipped on dev boxes."""

    def test_get_present_people_includes_tier(self):
        pytest.importorskip("cv2")
        from boxbot.perception.state_machine import PerceptionStateMachine
        from boxbot.perception.visual_reid import MatchResult

        sm = PerceptionStateMachine()
        sm.on_identification(
            "Person A",
            MatchResult(
                person_id="p1", person_name="Jacob",
                confidence=0.9, tier="high",
            ),
        )
        people = sm.get_present_people()
        assert people[0]["tier"] == "high"
        assert format_presence_line(people) == "[Present: Jacob (confirmed)]"

    def test_unmatched_person_has_none_tier(self):
        pytest.importorskip("cv2")
        from boxbot.perception.state_machine import PerceptionStateMachine
        from boxbot.perception.person_detector import Detection

        sm = PerceptionStateMachine()
        sm.on_motion(score=20.0, threshold=10.0)
        sm.on_person_detected([
            Detection(bbox=(0, 0, 100, 200), confidence=0.9, class_id=0)
        ])
        people = sm.get_present_people()
        assert people[0]["tier"] is None
        assert format_presence_entries(people) == ["Person A (new)"]
