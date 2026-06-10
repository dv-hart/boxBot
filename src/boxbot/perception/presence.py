"""Presence surface — the ``[Present: …]`` header and its update debouncer.

The perception pipeline tracks who is physically present
(``PerceptionStateMachine.active_persons``). This module turns that
tracking into the agent-facing presence surface promised by
docs/perception.md:

- :func:`format_presence_line` renders the conversation-start header,
  e.g. ``[Present: Jacob (confirmed), Person B (new)]``. The agent core
  injects it into the dynamic context for voice and trigger
  conversations.
- :func:`get_presence_snapshot` reads the live pipeline and returns a
  stable, comparable tuple of formatted entries (or ``None`` when the
  pipeline isn't running).
- :class:`PresenceDebouncer` decides *when* a mid-conversation
  ``[Presence update: …]`` line should be injected: only after the
  presence set has been stable for a few seconds (tracking flicker is
  invisible) and never more often than a minimum interval.

Confidence tiers map to three labels:

- ``confirmed`` — high-tier identification (voice-confirmed or visual
  match above the high threshold).
- ``likely`` — named, but only a medium/low-tier match.
- ``new`` — no name; an unrecognized person (shown by session ref,
  e.g. ``Person B (new)``).

Note on timing: visual heartbeats are frozen while perception is in the
CONVERSATION state (Hailo is freed for the conversation), so presence
*events* mostly fire in DETECTED state — between utterances and during
trigger conversations. Departures don't publish a dedicated event; they
surface on the next person event or the next turn's header.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)

# A change must hold steady this long before it's announced. Aligned
# with the pipeline's DETECTED-state YOLO heartbeat (5s default) so one
# missed/extra heartbeat frame can't fire an update.
DEFAULT_STABLE_SECONDS = 7.0

# Floor between injected updates per conversation — debounce handles
# flicker, this guards against a slow oscillation (someone pacing in
# and out of frame) turning into a drumbeat of updates.
DEFAULT_MIN_INTERVAL_SECONDS = 30.0

Snapshot = tuple[str, ...]


def format_presence_entries(people: list[dict]) -> list[str]:
    """Format pipeline presence records into header entries.

    Args:
        people: Output of ``PerceptionPipeline.get_present_people()`` —
            dicts with ``ref``, ``name``, ``confidence``, and (when the
            ReID matcher has run) ``tier``.

    Returns:
        One formatted string per person, e.g. ``"Jacob (confirmed)"``,
        ``"Sarah (likely)"``, ``"Person B (new)"``.
    """
    entries: list[str] = []
    for p in people:
        name = p.get("name")
        if not name:
            entries.append(f"{p.get('ref', 'unknown')} (new)")
            continue
        tier = p.get("tier")
        if tier == "high" or (tier is None and p.get("confidence", 0.0) > 0.8):
            entries.append(f"{name} (confirmed)")
        else:
            entries.append(f"{name} (likely)")
    return entries


def format_presence_line(people: list[dict]) -> str | None:
    """Render the ``[Present: …]`` header, or ``None`` if nobody is tracked."""
    entries = format_presence_entries(people)
    if not entries:
        return None
    return f"[Present: {', '.join(entries)}]"


def get_presence_snapshot() -> Snapshot | None:
    """Read the live pipeline and return a comparable presence snapshot.

    Returns ``None`` when the perception pipeline is not running (the
    caller should skip presence handling entirely), and an empty tuple
    when it runs but tracks nobody.
    """
    try:
        from boxbot.perception.pipeline import get_pipeline

        people = get_pipeline().get_present_people()
    except Exception:
        return None
    return tuple(format_presence_entries(people))


class PresenceDebouncer:
    """Decide when a presence change is stable enough to announce.

    Pure bookkeeping — no asyncio, no event bus. The agent core feeds
    it snapshots as person events arrive (``offer``), polls ``ready``
    after a debounce sleep, and records announcements
    (``mark_announced``). Injectable clock keeps tests deterministic.

    Rules enforced:
    - a snapshot must be unchanged for ``stable_seconds`` before it can
      be announced (tracking flicker never surfaces);
    - the candidate must differ from the last-announced snapshot;
    - at least ``min_interval_seconds`` must elapse between
      announcements.
    """

    def __init__(
        self,
        *,
        stable_seconds: float = DEFAULT_STABLE_SECONDS,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stable_seconds = stable_seconds
        self.min_interval_seconds = min_interval_seconds
        self._clock = clock
        self._candidate: Snapshot | None = None
        self._candidate_since: float = 0.0
        self._last_announced: Snapshot | None = None
        self._last_announced_at: float | None = None

    @property
    def candidate(self) -> Snapshot | None:
        return self._candidate

    def offer(self, snapshot: Snapshot) -> None:
        """Feed the current presence snapshot; resets stability on change."""
        if snapshot != self._candidate:
            self._candidate = snapshot
            self._candidate_since = self._clock()

    def seconds_until_ready(self) -> float:
        """Remaining time before the current candidate could be announced."""
        if self._candidate is None:
            return self.stable_seconds
        now = self._clock()
        remaining = (self._candidate_since + self.stable_seconds) - now
        if self._last_announced_at is not None:
            remaining = max(
                remaining,
                (self._last_announced_at + self.min_interval_seconds) - now,
            )
        return max(0.0, remaining)

    def ready(self) -> Snapshot | None:
        """Return the candidate snapshot if it should be announced now.

        ``None`` means: no candidate, not yet stable, identical to the
        last announcement, or inside the rate-limit window.
        """
        if self._candidate is None:
            return None
        if self._candidate == self._last_announced:
            return None
        if self.seconds_until_ready() > 0:
            return None
        return self._candidate

    def mark_announced(self, snapshot: Snapshot) -> None:
        """Record that *snapshot* was delivered to the conversation."""
        self._last_announced = snapshot
        self._last_announced_at = self._clock()

    def sync_baseline(self, snapshot: Snapshot) -> None:
        """Set the baseline without rate-limiting future announcements.

        Used when the conversation-start header already showed this
        snapshot to the agent — an immediately following event should
        not re-announce it, but a *real* change soon after start should
        not be blocked by the min-interval either.
        """
        self._last_announced = snapshot
        self._last_announced_at = None
