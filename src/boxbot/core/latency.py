"""Per-turn voice round-trip latency tracking.

The voice pipeline is a chain of decoupled stages connected by an event
bus — STT → diarization → speaker resolution → agent generation → TTS →
playback — spread across several objects and files. No single stage can
see the end-to-end cost, so "why is it slow" has never had a single
number to point at.

This module is that number. It keeps a process-wide registry of
:class:`TurnLatency` trackers keyed by ``conversation_id``. A voice turn
opens a tracker at true speech-end (:func:`begin`), each stage stamps its
own cost as it flows through (:func:`mark`, :func:`add`, :func:`span`),
and when the first TTS audio reaches the speaker (:func:`first_audio`) we
emit one consolidated breakdown line and discard the tracker.

Design constraints:

- **Pure instrumentation.** No behaviour change. Every entry point is a
  no-op when no tracker exists for the given ``conversation_id`` (or when
  it is ``None``), so non-voice turns — WhatsApp, triggers — pay nothing
  but the dict lookup. Only :func:`begin` (called once per voice
  utterance) creates a tracker.
- **Tolerant of missing marks.** Barge-in, cancellation, or an error can
  end a turn before ``first_audio``. The next :func:`begin` for the same
  conversation discards the stale tracker (logged at debug as
  ``incomplete``), so memory is bounded to one tracker per live
  conversation.
- **Single clock.** All timestamps are :func:`time.monotonic`, the same
  clock the HAL stamps audio chunks with, so ``utterance.timestamp_end``
  composes directly with marks taken anywhere else in the pipeline.

All calls happen on the asyncio event-loop thread (diarization offloads
to an executor internally, but the timing wrappers await on the loop), so
the registry needs no locking.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger("boxbot.latency")


@dataclass
class TurnLatency:
    """Accumulated timing for one voice round-trip.

    ``t0`` is true speech-end (monotonic). ``marks`` hold elapsed-from-t0
    timestamps for stage boundaries (first write wins — a stage is
    reached once). ``spans`` accumulate durations for work that can
    repeat within a turn (the agent makes several API calls; tools run
    several times). ``counts`` tracks how many times each span fired.
    """

    conversation_id: str
    t0: float
    marks: dict[str, float] = field(default_factory=dict)
    spans: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    _finished: bool = False

    def _mark(self, label: str) -> None:
        # First write wins: a stage boundary is crossed once per turn.
        self.marks.setdefault(label, time.monotonic() - self.t0)

    def _add(self, label: str, seconds: float) -> None:
        self.spans[label] = self.spans.get(label, 0.0) + seconds
        self.counts[label] = self.counts.get(label, 0) + 1


# Process-wide registry. Keyed by conversation_id; at most one live
# tracker per conversation.
_trackers: dict[str, TurnLatency] = {}

# Alias map: agent-side conversation id -> tracker key. A voice turn is
# split across two identities — the voice *session* id (``voice_<uuid>``,
# which STT / diarize / TTS see and which keys the tracker) and the
# *Conversation* id (``voice:room``, which the agent loop runs under).
# The transcript handler knows both and registers the bridge so the
# agent loop's marks land on the live voice tracker. Non-voice
# conversations never register an alias, so this is a no-op for them.
_aliases: dict[str, str] = {}


def alias(agent_key: str | None, tracker_key: str | None) -> None:
    """Bridge an agent-side conversation id to a tracker key.

    Idempotent; re-registered every transcript so it tracks the current
    voice session. Stale entries for ended conversations are harmless
    (bounded by the number of distinct conversation ids).
    """
    if agent_key and tracker_key and agent_key != tracker_key:
        _aliases[agent_key] = tracker_key


def begin(conversation_id: str | None, t0: float | None = None) -> None:
    """Open a fresh round-trip tracker for ``conversation_id``.

    ``t0`` should be true speech-end (``utterance.timestamp_end``); if
    omitted, the current monotonic time is used. A pre-existing tracker
    for the same conversation (a prior turn that never reached
    ``first_audio`` — barge-in, cancel, error) is discarded and logged at
    debug, bounding the registry to one entry per live conversation.
    """
    if not conversation_id:
        return
    stale = _trackers.get(conversation_id)
    if stale is not None and not stale._finished:
        logger.debug(
            "latency: discarding incomplete turn conv=%s (reached: %s)",
            conversation_id,
            ", ".join(stale.marks) or "nothing",
        )
    _trackers[conversation_id] = TurnLatency(
        conversation_id=conversation_id,
        t0=t0 if t0 is not None else time.monotonic(),
    )


def get(conversation_id: str | None) -> TurnLatency | None:
    """Return the live tracker for ``conversation_id``, or ``None``.

    Resolves through the alias map first, so a mark made under the
    agent-side Conversation id finds the tracker opened under the voice
    session id.
    """
    if not conversation_id:
        return None
    key = _aliases.get(conversation_id, conversation_id)
    return _trackers.get(key)


def mark(conversation_id: str | None, label: str) -> None:
    """Stamp a stage-boundary timestamp (no-op without a live tracker)."""
    t = get(conversation_id)
    if t is not None:
        t._mark(label)


def add(conversation_id: str | None, label: str, seconds: float) -> None:
    """Accumulate a span duration (no-op without a live tracker)."""
    t = get(conversation_id)
    if t is not None:
        t._add(label, seconds)


@contextmanager
def span(conversation_id: str | None, label: str) -> Iterator[None]:
    """Time the enclosed block and accumulate it under ``label``.

    Records even on exception so a failing stage still shows its cost.
    """
    start = time.monotonic()
    try:
        yield
    finally:
        add(conversation_id, label, time.monotonic() - start)


def first_audio(conversation_id: str | None) -> None:
    """Mark first audible TTS output and emit the breakdown line.

    This is the end of the perceived round-trip. Safe to call more than
    once per turn (later outputs in the same turn) — only the first call
    finishes the tracker.
    """
    t = get(conversation_id)
    if t is None or t._finished:
        return
    t._mark("first_audio")
    _finish(t, reason="first_audio")


def discard(conversation_id: str | None, reason: str = "cancelled") -> None:
    """Drop a tracker without emitting the headline (e.g. barge-in).

    Logs the partial breakdown at debug so a cancelled turn still leaves
    a trace of where it got to.
    """
    t = get(conversation_id)
    if t is None:
        return
    logger.debug(
        "latency: turn %s conv=%s — %s",
        reason,
        t.conversation_id,
        _breakdown(t),
    )
    _trackers.pop(conversation_id, None)  # type: ignore[arg-type]


def _finish(t: TurnLatency, *, reason: str) -> None:
    t._finished = True
    _trackers.pop(t.conversation_id, None)
    # marks store elapsed-from-t0 in seconds; render the headline in ms.
    total = t.marks.get("first_audio")
    headline = f"{total * 1000:.0f}ms" if total is not None else "?ms"
    logger.info(
        "voice round-trip conv=%s: %s speech-end→first-audio | %s",
        t.conversation_id,
        headline,
        _breakdown(t),
    )


def _breakdown(t: TurnLatency) -> str:
    """Render the per-stage breakdown as a single compact line.

    Stages absent from the turn are omitted rather than printed as zero,
    so the line reflects what actually ran.
    """
    parts: list[str] = []

    tr = t.marks.get("transcript_ready")
    if tr is not None:
        # The pre-agent stages. stt runs concurrently with the voice
        # stage (diarize when enabled, else embed), so they overlap inside
        # the transcript window — shown individually, not summed.
        sub = []
        for label in ("stt", "diarize", "embed", "resolve"):
            v = t.spans.get(label)
            if v is not None:
                sub.append(f"{label}={v * 1000:.0f}")
        detail = f" ({' '.join(sub)})" if sub else ""
        parts.append(f"transcript@{tr * 1000:.0f}ms{detail}")

    gs = t.marks.get("gen_start")
    if gs is not None:
        sub = []
        api = t.spans.get("api")
        if api is not None:
            sub.append(f"api={api * 1000:.0f}/{t.counts.get('api', 0)}calls")
        tools = t.spans.get("tools")
        if tools is not None:
            sub.append(f"tools={tools * 1000:.0f}")
        detail = f" ({' '.join(sub)})" if sub else ""
        parts.append(f"gen@{gs * 1000:.0f}ms{detail}")

    ttfb = t.spans.get("tts_ttfb")
    if ttfb is not None:
        parts.append(f"tts_ttfb={ttfb * 1000:.0f}ms")

    return " | ".join(parts) if parts else "no stages recorded"
