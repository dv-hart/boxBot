"""Agent state tracker for display data binding.

Subscribes to agent lifecycle events on the event bus and maintains a
small read-only view of the agent's current state ("sleeping",
"listening", "thinking", "speaking"), the timestamp of the last
activity, and the next scheduled wake.

Used by displays.data_sources.AgentStatusSource to render a live status
indicator without coupling the renderer to the agent or scheduler.

Usage:
    from boxbot.core.agent_state import get_agent_state_tracker

    tracker = get_agent_state_tracker()
    await tracker.start()
    snapshot = await tracker.snapshot()
    # {"state": "listening", "last_active": "...", "next_wake": "..."}
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from boxbot.core.events import (
    AgentSpeaking,
    AgentSpeakingDone,
    ConversationEnded,
    ConversationStarted,
    TranscriptReady,
    TriggerFired,
    VoiceSessionEnded,
    WakeWordHeard,
    get_event_bus,
)

logger = logging.getLogger(__name__)


_STATE_SLEEPING = "sleeping"
_STATE_LISTENING = "listening"
_STATE_THINKING = "thinking"
_STATE_SPEAKING = "speaking"


class AgentStateTracker:
    """Maintains a thread-safe view of the agent's current activity state."""

    def __init__(self) -> None:
        self._state: str = _STATE_SLEEPING
        self._last_active: datetime | None = None
        self._active_conversations: int = 0
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        bus = get_event_bus()
        bus.subscribe(WakeWordHeard, self._on_wake_word)
        bus.subscribe(ConversationStarted, self._on_conversation_started)
        bus.subscribe(ConversationEnded, self._on_conversation_ended)
        bus.subscribe(TranscriptReady, self._on_transcript_ready)
        bus.subscribe(AgentSpeaking, self._on_agent_speaking)
        bus.subscribe(AgentSpeakingDone, self._on_agent_speaking_done)
        bus.subscribe(TriggerFired, self._on_trigger_fired)
        bus.subscribe(VoiceSessionEnded, self._on_voice_session_ended)
        self._running = True
        logger.debug("AgentStateTracker started")

    async def stop(self) -> None:
        if not self._running:
            return
        bus = get_event_bus()
        bus.unsubscribe(WakeWordHeard, self._on_wake_word)
        bus.unsubscribe(ConversationStarted, self._on_conversation_started)
        bus.unsubscribe(ConversationEnded, self._on_conversation_ended)
        bus.unsubscribe(TranscriptReady, self._on_transcript_ready)
        bus.unsubscribe(AgentSpeaking, self._on_agent_speaking)
        bus.unsubscribe(AgentSpeakingDone, self._on_agent_speaking_done)
        bus.unsubscribe(TriggerFired, self._on_trigger_fired)
        bus.unsubscribe(VoiceSessionEnded, self._on_voice_session_ended)
        self._running = False

    @property
    def state(self) -> str:
        return self._state

    @property
    def last_active(self) -> datetime | None:
        return self._last_active

    async def snapshot(self) -> dict[str, Any]:
        """Return a display-ready snapshot of agent state.

        Returns:
            Dict with `state`, `last_active`, and `next_wake` (all strings,
            human-friendly). Values may be None when unknown.
        """
        return {
            "state": self._state,
            "last_active": _format_relative(self._last_active),
            "next_wake": await _next_wake_str(),
        }

    # --- Event handlers ---

    async def _on_wake_word(self, event: WakeWordHeard) -> None:
        self._state = _STATE_LISTENING
        self._mark_active()

    async def _on_conversation_started(
        self, event: ConversationStarted
    ) -> None:
        self._active_conversations += 1
        self._state = _STATE_LISTENING
        self._mark_active()

    async def _on_conversation_ended(self, event: ConversationEnded) -> None:
        self._active_conversations = max(0, self._active_conversations - 1)
        if self._active_conversations == 0:
            self._state = _STATE_SLEEPING
        self._mark_active()

    async def _on_transcript_ready(self, event: TranscriptReady) -> None:
        # Transcript arrived → agent is now reasoning about a response
        self._state = _STATE_THINKING
        self._mark_active()

    async def _on_agent_speaking(self, event: AgentSpeaking) -> None:
        self._state = _STATE_SPEAKING
        self._mark_active()

    async def _on_agent_speaking_done(
        self, event: AgentSpeakingDone
    ) -> None:
        self._state = (
            _STATE_LISTENING if self._active_conversations > 0
            else _STATE_SLEEPING
        )
        self._mark_active()

    async def _on_trigger_fired(self, event: TriggerFired) -> None:
        self._state = _STATE_THINKING
        self._mark_active()

    async def _on_voice_session_ended(self, event: VoiceSessionEnded) -> None:
        # Voice adapter went idle (e.g. wake-word grace expired with no
        # speech). If no Conversation is active we'd otherwise stay stuck
        # in LISTENING from the WakeWordHeard transition — flip back to
        # sleeping. When a real conversation is in flight, leave state
        # alone; ConversationEnded is the authoritative reset.
        if self._active_conversations == 0:
            self._state = _STATE_SLEEPING
            self._mark_active()

    def _mark_active(self) -> None:
        self._last_active = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_relative(ts: datetime | None) -> str | None:
    """Format a UTC datetime as a short relative time string."""
    if ts is None:
        return None
    now = datetime.now(timezone.utc)
    delta = now - ts
    seconds = int(delta.total_seconds())
    if seconds < 5:
        return "just now"
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        mins = seconds // 60
        return f"{mins} min ago" if mins == 1 else f"{mins} min ago"
    if seconds < 86400:
        hours = seconds // 3600
        return f"{hours}h ago"
    days = seconds // 86400
    return f"{days}d ago"


async def _next_wake_str() -> str | None:
    """Return the next scheduled trigger fire time as a short string."""
    try:
        from boxbot.core.scheduler import list_triggers

        triggers = await list_triggers(status="active")
    except Exception:
        return None

    next_dt: datetime | None = None
    for t in triggers:
        fa = t.get("fire_at")
        if not fa:
            continue
        try:
            dt = datetime.fromisoformat(fa)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if next_dt is None or dt < next_dt:
            next_dt = dt

    if next_dt is None:
        return None

    # Display in local time
    local = next_dt.astimezone()
    now = datetime.now(local.tzinfo)
    if local.date() == now.date():
        return local.strftime("%-I:%M %p")
    if (local.date() - now.date()).days == 1:
        return f"tomorrow {local.strftime('%-I:%M %p')}"
    return local.strftime("%a %-I:%M %p")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_tracker: AgentStateTracker | None = None


def get_agent_state_tracker() -> AgentStateTracker:
    """Return the global AgentStateTracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = AgentStateTracker()
    return _tracker
