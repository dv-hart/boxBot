"""Tests for the backfill cleanup script's regex matching.

The script invalidates routine-delivery log memories only. It must:
- Match real delivery-log summaries pulled from the Pi.
- Skip methodology / person / household memories with similar wording.
- Skip operational memories whose summary carries a lesson signal
  (bug observation, fix note, "still broken" assertion, etc.) even
  when the delivery-log shape matches.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


from backfill_memory_cleanup import KEEP_SIGNALS_RE, ROUTINE_TRIGGER_RE


# Real-world summaries pulled from data/memory/memory.db on the Pi
# (captured 2026-05-13). Pure delivery logs — these must be matched.
REAL_PURE_DELIVERY_SUMMARIES = [
    "Morning briefing delivered to Jacob on May 3 2026.",
    "Evening review May 6 delivered to Jacob via text — 4 open SDK todos.",
    "Morning briefing sent to Jacob & Carina on Fri May 8 2026.",
    "Morning briefing delivered to Jacob & Carina via text on 2026-05-11.",
    "Midday check 2026-05-11: 6 open todos, texted Jacob Dr. Fu appt reminder.",
    "Morning briefing sent 5/12 to Jacob & Carina; calendar feed down, reauth pending.",
    "Evening review sent to Jacob May 11; Dr. Fu Wed May 13 4:40 PM; calendar still down.",
    "Evening review delivered 2026-05-12; Dr. Fu appt May 13 flagged.",
    "Evening review delivered to Jacob Sun May 3 2026 via text.",
    "Midday check 5/12: texted Jacob re 2 open todos, calendar reauth is blocker.",
]


# Real-world summaries that match the delivery shape but carry a
# lesson signal — must be PRESERVED (the KEEP_SIGNALS_RE acts as a
# veto over a positive ROUTINE_TRIGGER_RE match).
REAL_LESSON_SUMMARIES = [
    # display name lesson, originally mis-typed operational
    "Display 'weather' is invalid; use 'morning_brief' or 'weather_simple'.",
    # timezone bug observation
    "Midday check trigger fires at 05:00 local due to UTC vs. Pacific timezone mismatch.",
    # display auto-revert bug
    "morning_brief display has no auto-revert; stuck until manually changed.",
    # bug observation
    "Morning briefing 2026-05-06 sent to Jacob & Carina; calendar SDK still broken.",
    # configuration / lesson record
    "Morning briefing trigger updated to include Carina (2026-05-04)",
    # bug discovery during midday check
    "Sandbox bootstrap script has permission denied error; execute_script broken as of 2026-05-01.",
    # success-state assertion that should NOT be invalidated
    "Calendar integration confirmed working 5/12; reauth todos closed",
    "Google Calendar integration confirmed working 5/13; stale todos closed.",
]


# Memories that don't even look like delivery logs — extra safety net.
KEEP_SUMMARIES = [
    "Calendar SDK: use bb.integrations.get('calendar', ...) not bb.calendar.*",
    "Jacob: Dr. Fu appt May 13 @ 4:40 PM",
    "bb.integrations has two pipes: weather (NOAA) and calendar (Google Calendar).",
    "Jacob lives in Hillsboro, OR; interested in splash pads at Hidden Creek Park.",
    "Jacob's camera is IR-only; using grayscale until IR filter acquired.",
    "Use photos.show_on_screen then switch_display(pin=True) for reliable picture display",
    "Duplicate calendar events: Memorial Day 5/25 and Cleaners 5/28 appear twice",
    "Never prompt Jacob about assigning due dates on todos.",
]


def _would_invalidate(summary: str) -> bool:
    """Mirror the script's filter: must match delivery shape AND not
    carry a lesson signal."""
    return bool(
        ROUTINE_TRIGGER_RE.search(summary)
    ) and not KEEP_SIGNALS_RE.search(summary)


def test_pure_delivery_logs_get_invalidated() -> None:
    for s in REAL_PURE_DELIVERY_SUMMARIES:
        assert _would_invalidate(s), f"should invalidate: {s!r}"


def test_delivery_shape_with_lesson_signal_is_preserved() -> None:
    """The veto rule keeps methodology-flavoured operational entries
    even if the delivery-shape regex matches."""
    for s in REAL_LESSON_SUMMARIES:
        assert not _would_invalidate(s), f"must preserve: {s!r}"


def test_keep_summaries_untouched() -> None:
    for s in KEEP_SUMMARIES:
        assert not _would_invalidate(s), f"must preserve: {s!r}"


def test_keep_signals_re_catches_resolution_words() -> None:
    """If a memory says calendar is 'working' / 'fixed' / 'resolved',
    we keep it — those are the *good* state-assertions that should
    supersede the stale ones."""
    assert KEEP_SIGNALS_RE.search("Calendar confirmed working")
    assert KEEP_SIGNALS_RE.search("Integration fixed today")
    assert KEEP_SIGNALS_RE.search("Issue resolved on 5/13")


def test_routine_re_requires_anchor_at_start() -> None:
    """A passing mention of 'morning briefing' inside another
    sentence must NOT trigger a match."""
    not_a_log = (
        "Jacob wants the family schedule shown during morning briefing."
    )
    assert not ROUTINE_TRIGGER_RE.search(not_a_log)


def test_routine_re_requires_delivery_verb() -> None:
    """Just the event name without a verb is not enough."""
    bare = "Morning briefing config: include both Jacob and Carina."
    assert not ROUTINE_TRIGGER_RE.search(bare)


def test_midday_check_with_colon_date_matches() -> None:
    """One of the real corpus shapes is 'Midday check 5/12: texted …'.
    The trigger event-name + date + texted/sent must match."""
    assert ROUTINE_TRIGGER_RE.search(
        "Midday check 5/12: texted Jacob re 2 open todos"
    )
