"""Conversation — the unified state machine for agent interactions.

A ``Conversation`` owns the full lifecycle of one logical conversation:
its thread of messages, which participants are part of it, which I/O
channels it communicates over, the current response in flight, and the
transitions between listening/thinking/speaking.

This replaces the split between the voice pipeline's session state and
the agent's per-channel task tracker that existed in the earlier
codebase. The voice and whatsapp adapters are now thin I/O layers;
conversation state lives here, keyed by conversation id.

Rules
-----
- One generation runs at a time per conversation. New user input while
  a generation is in flight cancels that generation and starts a new
  one whose thread contains the new input. Tool results and TTS audio
  already delivered are recorded so the model sees them on the next
  turn.
- Conversations across channels are independent. The room's voice
  conversation and a WhatsApp conversation with Carina run fully in
  parallel — each has its own generation task.
- State transitions publish events (``ConversationStarted`` when the
  conversation activates, ``ConversationEnded`` when it terminates).

Usage
-----
``Conversation`` is not typically constructed by tools or skills — the
agent core manages conversations and routes inbound events. A
``generate_fn`` must be supplied at construction time; the agent core
injects its agent-loop driver. Tests supply a fake generator.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from boxbot.core.events import (
    ConversationEnded,
    ConversationStarted,
    get_event_bus,
)

# Forward-declared for type hints only; the agent attaches one of these
# after constructing the Conversation. Keeping the import lazy avoids
# pulling the tools package from core during static analysis.
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from boxbot.tools.sandbox_runner import SandboxRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State + helpers
# ---------------------------------------------------------------------------


class ConversationState(Enum):
    """Lifecycle state for a single conversation."""

    IDLE = "idle"            # created but not yet active (before first input)
    LISTENING = "listening"  # waiting for user input
    THINKING = "thinking"    # generation_task is running; no audio yet
    SPEAKING = "speaking"    # TTS or other delivery in progress
    ENDED = "ended"          # terminated; will not accept further input


@dataclass
class SpokenSegment:
    """Record of one delivered output (full or partial).

    Used to reconstruct what the user actually heard/received before an
    interruption, so the model's next turn sees a faithful history.
    """

    channel: str          # "voice" | "text"
    to: str               # addressee name / phone / "room"
    content: str          # the text that was (at least partially) delivered
    interrupted: bool = False  # True if TTS/text was cut off mid-delivery


@dataclass
class GenerationResult:
    """Outcome of one agent-loop generation cycle.

    Returned by ``generate_fn``. The conversation uses this to update
    its thread and decide whether to advance state.
    """

    # Messages appended to the thread in Anthropic wire format. Typically
    # one assistant message per cycle, possibly followed by tool_result
    # user messages if tool calls happened; the generator is free to
    # decide the shape.
    thread_additions: list[dict[str, Any]] = field(default_factory=list)

    # Audio/text that was at least partially delivered during the cycle.
    # On cancel, the cancelled segment should be marked interrupted=True.
    spoken_segments: list[SpokenSegment] = field(default_factory=list)

    # Turn count used by memory extraction and logs.
    turn_count: int = 0

    # One-line summary extracted from the assistant's last text block.
    summary: str = ""

    # False if the cycle was cancelled mid-run (the conversation will
    # still apply thread_additions and spoken_segments on cancel so the
    # next cycle has an honest view).
    completed_cleanly: bool = True


# A generate_fn is injected by AgentCore. It takes the conversation and
# returns a GenerationResult. It MUST cooperate with cancellation — when
# the conversation cancels its generation task the generator should run
# any ``finally`` blocks needed to stop TTS, record what was spoken, and
# re-raise the CancelledError. The conversation's cancel-and-regenerate
# machinery catches the CancelledError and inspects generation_result
# (if the generator stored it on the conversation) for interrupt state.
GenerateFn = Callable[
    ["Conversation"],
    Awaitable[GenerationResult],
]


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


class Conversation:
    """One logical conversation — state + thread + in-flight generation.

    Typically not constructed directly. The agent core creates and
    indexes conversations keyed by channel identity (room for voice,
    sender phone for whatsapp, trigger id for triggers).
    """

    # A conversation that's silent for this long transitions to ENDED.
    DEFAULT_SILENCE_TIMEOUT = 180.0

    def __init__(
        self,
        *,
        conversation_id: str,
        channel: str,
        channel_key: str,
        generate_fn: GenerateFn,
        participants: Optional[set[str]] = None,
        silence_timeout: float | None = None,
    ) -> None:
        self.conversation_id = conversation_id
        self.channel = channel            # "voice" | "whatsapp" | "trigger"
        self.channel_key = channel_key    # "voice:room", "whatsapp:+1...", etc.
        self.participants: set[str] = set(participants or ())
        self.silence_timeout = (
            silence_timeout
            if silence_timeout is not None
            else self.DEFAULT_SILENCE_TIMEOUT
        )

        self._state = ConversationState.IDLE
        self._thread: list[dict[str, Any]] = []
        self._generate_fn = generate_fn

        # Current in-flight generation. None when idle or between turns.
        self._generation_task: asyncio.Task[GenerationResult] | None = None
        # Partial-delivery record populated by the generator while it
        # runs. On cancel, the conversation reads this so it can fold
        # interrupted output into the thread before regenerating.
        self._pending_segments: list[SpokenSegment] = []
        self._current_turn_started: float = 0.0

        # Per-conversation lock: serialises handle_input / end so input
        # bursts are always processed in order. Across conversations,
        # no lock contention — they run truly in parallel.
        self._lock = asyncio.Lock()

        # Silence timer. Reset on every input. Fires end().
        self._silence_timer_task: asyncio.Task[None] | None = None
        self._last_activity_monotonic: float = time.monotonic()

        self._closed = False

        # Per-conversation sandbox process. Attached by the agent after
        # construction (the agent has the config it needs to spawn one).
        # ``execute_script`` reads this via the current_conversation
        # ContextVar to route its scripts through. Stays alive across
        # turns so Python state (last_image, parsed CSVs) persists.
        self.sandbox_runner: "SandboxRunner | None" = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> ConversationState:
        return self._state

    @property
    def thread(self) -> list[dict[str, Any]]:
        """Live view of the message thread (do not mutate externally)."""
        return self._thread

    @property
    def pending_segments(self) -> list[SpokenSegment]:
        """Audio/text segments recorded during the current generation.

        The generator appends to this list as it delivers outputs. On
        cancel the conversation folds this into the thread so the next
        turn sees what was actually delivered.
        """
        return self._pending_segments

    @property
    def is_generating(self) -> bool:
        return (
            self._generation_task is not None
            and not self._generation_task.done()
        )

    @property
    def is_ended(self) -> bool:
        return self._state is ConversationState.ENDED

    # ------------------------------------------------------------------
    # Public API — input handling
    # ------------------------------------------------------------------

    async def handle_input(
        self,
        text: str,
        *,
        speaker_name: str | None = None,
        source: str = "user",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Accept new input and (re)drive the generation cycle.

        - Appends the user message to the thread.
        - If a generation is in flight, cancels it. Any partial output
          the generator recorded in ``pending_segments`` is folded into
          the thread as a synthetic assistant turn marked interrupted
          so the model sees what was actually delivered.
        - Starts a fresh generation task.

        Args:
            text: The incoming message text. Must be non-empty after
                stripping; empty inputs are dropped with a debug log.
            speaker_name: The speaker's display name for voice
                attribution. Appended to ``participants``.
            source: ``"user"`` (inbound from a person) or ``"trigger"``
                (scheduler wake — inserts as a system-style message).
            context: Per-turn context passed through to generate_fn via
                ``_current_context``. Examples: voice speaker identity
                block, whatsapp media url.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            logger.debug(
                "Conversation %s: dropped empty input",
                self.conversation_id,
            )
            return

        async with self._lock:
            if self._state is ConversationState.ENDED:
                logger.warning(
                    "Conversation %s: input after end ignored (%s)",
                    self.conversation_id, cleaned[:60],
                )
                return

            # Participants tracking (voice attribution).
            if speaker_name:
                self.participants.add(speaker_name)

            # Cancel any in-flight generation. We fold its partial
            # output into the thread before adding the new user input
            # so the regenerated turn sees BOTH: the interrupted
            # assistant output AND the fresh user message.
            if self.is_generating:
                await self._cancel_and_fold()

            # Append new user input to thread.
            user_message = self._format_user_message(
                cleaned, speaker_name=speaker_name, source=source,
            )
            self._thread.append(user_message)

            # First input: the conversation becomes active and publishes
            # ConversationStarted.
            was_idle = self._state is ConversationState.IDLE
            if was_idle:
                self._state = ConversationState.THINKING
                await self._publish_started()
            else:
                self._state = ConversationState.THINKING

            self._reset_silence_timer()
            self._last_activity_monotonic = time.monotonic()

            # Start a fresh generation task. It runs outside the lock
            # so concurrent tool calls / TTS / etc. don't re-enter
            # handle_input deadlocked.
            self._current_context = dict(context or {})
            self._pending_segments = []
            self._current_turn_started = time.monotonic()
            self._generation_task = asyncio.create_task(
                self._run_generation(),
                name=f"gen-{self.conversation_id}",
            )

    # ------------------------------------------------------------------
    # Public API — lifecycle
    # ------------------------------------------------------------------

    async def end(self, *, reason: str = "explicit") -> None:
        """Terminate this conversation, publish ConversationEnded.

        Cancels any in-flight generation. Idempotent — calling twice is
        a no-op. Safe to call from within the silence timer task itself:
        we avoid cancelling the current task so the remaining await
        points (publish_ended) run to completion.
        """
        current = asyncio.current_task()
        async with self._lock:
            if self._closed:
                return
            self._closed = True

            prev_state = self._state
            self._state = ConversationState.ENDED

            if self.is_generating:
                await self._cancel_and_fold()

            # Don't cancel the silence timer if we ARE the silence
            # timer — self-cancel would abort the publish_ended below.
            timer_task = self._silence_timer_task
            self._silence_timer_task = None
            if timer_task is not None and timer_task is not current:
                timer_task.cancel()

            logger.info(
                "Conversation %s ended (reason=%s, prev_state=%s, turns=%d)",
                self.conversation_id,
                reason,
                prev_state.value,
                self._turn_count(),
            )

        # Publish outside the lock so handlers can call back into the
        # conversation (e.g. agent core removing it from its index).
        await self._publish_ended(reason=reason)

    # ------------------------------------------------------------------
    # Generator-facing API
    # ------------------------------------------------------------------

    def record_segment(self, segment: SpokenSegment) -> None:
        """Called by the generator to record a delivered output.

        The conversation uses this on cancel to fold the partial
        delivery into the thread as an interrupted assistant message.
        """
        self._pending_segments.append(segment)

    def set_state(self, state: ConversationState) -> None:
        """Called by the generator to advance state during a cycle.

        Typical sequence: THINKING → SPEAKING → LISTENING. The
        conversation enforces transitions are monotonic and ignores
        illegal ones.
        """
        # Can't un-end.
        if self._state is ConversationState.ENDED:
            return
        # Can't go back to IDLE from an active state.
        if state is ConversationState.IDLE and self._state is not ConversationState.IDLE:
            return
        self._state = state

    @property
    def current_context(self) -> dict[str, Any]:
        """Per-turn context passed to the current generation."""
        return getattr(self, "_current_context", {})

    # ------------------------------------------------------------------
    # Internal — generation driver
    # ------------------------------------------------------------------

    async def _run_generation(self) -> GenerationResult:
        """Drive one generate_fn call to completion (or cancellation).

        Returns a GenerationResult summarising the cycle. On successful
        completion, applies thread_additions to the conversation thread
        and transitions back to LISTENING. On cancellation, the caller
        (_cancel_and_fold) handles partial state.
        """
        try:
            result = await self._generate_fn(self)
        except asyncio.CancelledError:
            logger.info(
                "Conversation %s: generation cancelled after %.2fs",
                self.conversation_id,
                time.monotonic() - self._current_turn_started,
            )
            raise
        except Exception:
            logger.exception(
                "Conversation %s: generation raised — ending turn",
                self.conversation_id,
            )
            # A broken cycle shouldn't wedge the conversation; flip
            # back to LISTENING so the next input can try again.
            if self._state is not ConversationState.ENDED:
                self._state = ConversationState.LISTENING
            self._pending_segments = []
            return GenerationResult(completed_cleanly=False)

        # Normal completion: extend the thread and mark ready for the
        # next input.
        self._thread.extend(result.thread_additions)
        self._pending_segments = []
        if self._state is not ConversationState.ENDED:
            self._state = ConversationState.LISTENING

        logger.info(
            "Conversation %s: turn complete (%d additions, %.2fs, state=%s)",
            self.conversation_id,
            len(result.thread_additions),
            time.monotonic() - self._current_turn_started,
            self._state.value,
        )
        return result

    async def _cancel_and_fold(self) -> None:
        """Cancel the in-flight generation and fold partials into the thread.

        Caller MUST hold ``self._lock``. Safe to call when no generation
        is in flight (no-op).
        """
        task = self._generation_task
        if task is None or task.done():
            self._generation_task = None
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                "Conversation %s: exception while awaiting cancelled task",
                self.conversation_id,
            )

        # Collect anything the generator managed to deliver before it
        # was cancelled, and record it as an interrupted assistant turn.
        partials = self._pending_segments
        if partials:
            combined = " ".join(
                seg.content for seg in partials if seg.content.strip()
            ).strip()
            if combined:
                self._thread.append({
                    "role": "assistant",
                    "content": (
                        f"{combined}\n\n(interrupted by new user input)"
                    ),
                })
                logger.info(
                    "Conversation %s: folded %d interrupted segment(s) "
                    "(%d chars) into thread",
                    self.conversation_id,
                    len(partials),
                    len(combined),
                )

        self._pending_segments = []
        self._generation_task = None

    # ------------------------------------------------------------------
    # Internal — silence timer
    # ------------------------------------------------------------------

    def _reset_silence_timer(self) -> None:
        """Restart the silence-timeout countdown."""
        if self._silence_timer_task is not None:
            self._silence_timer_task.cancel()
        if self.silence_timeout <= 0 or self._state is ConversationState.ENDED:
            self._silence_timer_task = None
            return
        self._silence_timer_task = asyncio.create_task(
            self._silence_timeout_loop(),
            name=f"silence-{self.conversation_id}",
        )

    async def _silence_timeout_loop(self) -> None:
        try:
            await asyncio.sleep(self.silence_timeout)
        except asyncio.CancelledError:
            return
        # Only auto-end if we've been in LISTENING. Don't end while
        # a generation is actively running — that's not silence.
        if self._state is ConversationState.LISTENING and not self.is_generating:
            logger.info(
                "Conversation %s: silence timeout (%.0fs), ending",
                self.conversation_id,
                self.silence_timeout,
            )
            await self.end(reason="silence_timeout")

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------

    def _format_user_message(
        self,
        text: str,
        *,
        speaker_name: str | None,
        source: str,
    ) -> dict[str, Any]:
        """Render an inbound message into the thread's message format."""
        if source == "trigger":
            # Triggers inject as a system/user-style prompt so the model
            # understands this is an automated wake, not a spoken line.
            return {"role": "user", "content": f"[trigger] {text}"}
        if speaker_name:
            # Voice attribution keeps the label format used elsewhere
            # so the model sees consistent speaker tags.
            return {"role": "user", "content": f"[{speaker_name}]: {text}"}
        return {"role": "user", "content": text}

    def _turn_count(self) -> int:
        """How many user/assistant messages have landed in the thread."""
        return sum(
            1 for m in self._thread
            if m.get("role") in ("user", "assistant")
        )

    async def _publish_started(self) -> None:
        bus = get_event_bus()
        primary_person: str | None = None
        for p in self.participants:
            primary_person = p
            break
        await bus.publish(
            ConversationStarted(
                conversation_id=self.conversation_id,
                channel=self.channel,
                person_name=primary_person,
                participants=sorted(self.participants),
            )
        )

    async def _publish_ended(self, *, reason: str) -> None:
        bus = get_event_bus()
        primary_person: str | None = None
        for p in self.participants:
            primary_person = p
            break
        # Build a short summary from the last assistant content we can
        # find. Memory extraction re-derives a richer transcript from
        # the thread, so this is only a human-readable log hint.
        summary = ""
        for msg in reversed(self._thread):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    summary = content[:120]
                break
        await bus.publish(
            ConversationEnded(
                conversation_id=self.conversation_id,
                channel=self.channel,
                person_name=primary_person,
                turn_count=self._turn_count(),
                summary=summary or f"(ended: {reason})",
            )
        )
