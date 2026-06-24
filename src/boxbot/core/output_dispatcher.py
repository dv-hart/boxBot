"""Output dispatcher — route deliveries to real channels.

The agent reaches humans by calling the ``message`` tool, NOT
by emitting text. Every text token the agent generates is constrained to
the private ``INTERNAL_NOTES_SCHEMA`` JSON shape — internal scratchpad
only, never broadcast.

This module owns:

- ``INTERNAL_NOTES_SCHEMA`` — the JSON shape pinned to every
  ``messages.create`` call via ``output_config.format``. It has no
  delivery channel: it is the agent's labelled scratchpad of private
  thoughts and observations. By construction, no field's content can
  reach a person.

- ``parse_internal_notes`` — parse one structured-output text block into
  a ``ParsedNotes`` (thought + observations) for logging and memory
  extraction.

- ``dispatch_outputs`` — invoked by the ``message`` tool (and
  by trigger-fired turns) to deliver one or more ``{to, channel, content}``
  entries through voice TTS or WhatsApp.

Routing:

- ``channel == "voice"`` → speak through the active voice session's TTS
  (the box speaker). ``to`` is semantic — the audience is whoever is in
  the room; we log the intended addressee for audit.
- ``channel == "text"`` → resolve ``to`` (a name, or ``"current_speaker"``)
  to a registered user's phone number via the AuthManager, then send via
  the WhatsApp client.

Invalid combinations (e.g. ``to: "room"`` with ``channel: "text"``, or
an unknown name with ``channel: "text"``) are logged and dropped — the
run does not crash. ``dispatch_outputs`` returns one ``DispatchResult``
per entry so the caller (the ``message`` tool) can tell the agent what
actually happened, including the list of valid recipients when a name
fails to resolve.

The dispatcher is stateless; all dependencies come from module
singletons that ``main.py`` sets up at boot (``get_voice_session``,
``get_auth_manager``) plus the outbound-channel registry from
``boxbot.communication.channels``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# Callback signature used by Conversation.record_segment. Imported
# lazily to avoid a circular dependency — output_dispatcher is imported
# by agent.py which imports conversation.
SegmentRecorder = Callable[[Any], None]


@dataclass
class DispatchResult:
    """Outcome of dispatching one ``{to, channel, content}`` entry.

    ``status`` is ``"delivered"`` or ``"dropped"``. On a drop, ``reason``
    is a human-readable explanation safe to hand back to the agent as a
    tool result. ``valid_recipients`` is populated only when the drop was
    caused by an unresolvable recipient name, so the agent can retry with
    a real name.
    """

    to: str
    channel: str
    status: str
    reason: str = ""
    valid_recipients: Optional[list[str]] = field(default=None)


# ---------------------------------------------------------------------------
# Schema — pinned via ``output_config.format`` on every messages.create call
#
# The schema has NO delivery channel by design. Its sole job is to occupy
# the model's text-output slot with a private scratchpad shape, removing
# the trained "respond as plain text to the user" affordance. To reach a
# human, the agent MUST call the ``message`` tool.
# ---------------------------------------------------------------------------

INTERNAL_NOTES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {
            "type": "string",
            "description": (
                "Private notes for yourself. NEVER reaches anyone. "
                "Use for your own reasoning and for post-conversation "
                "memory extraction. To actually speak or text someone, "
                "call the message tool — that is the only "
                "channel that reaches a person."
            ),
        },
        "observations": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Anything you noticed but chose not to act on right now "
                "(ambient facts, mood, who is in the room, what people "
                "are doing). PRIVATE — feeds memory extraction only. "
                "Never reaches anyone."
            ),
        },
    },
    "required": ["thought"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class ParsedNotes:
    """Result of parsing one agent text block as internal notes."""

    __slots__ = ("thought", "observations", "raw")

    def __init__(
        self,
        thought: str,
        observations: list[str],
        raw: str,
    ) -> None:
        self.thought = thought
        self.observations = observations
        self.raw = raw


def parse_internal_notes(raw_text: str) -> ParsedNotes | None:
    """Parse a JSON text block emitted by the agent into a ParsedNotes.

    Returns ``None`` on parse failure (should not occur under constrained
    decoding except for the refusal / truncation edge cases). Does not raise.
    """
    if not raw_text or not raw_text.strip():
        return None
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.warning(
            "Could not parse agent text block as JSON (%s). First 200 chars: %r",
            e,
            raw_text[:200],
        )
        return None
    if not isinstance(data, dict):
        logger.warning("Parsed agent output is not a dict: %r", type(data).__name__)
        return None
    thought = str(data.get("thought") or "")
    obs_raw = data.get("observations")
    observations: list[str] = []
    if isinstance(obs_raw, list):
        for entry in obs_raw:
            if isinstance(entry, str) and entry.strip():
                observations.append(entry)
    return ParsedNotes(thought=thought, observations=observations, raw=raw_text)


def parse_structured_notes(value: Any) -> ParsedNotes | None:
    """Build ``ParsedNotes`` from an SDK ``ResultMessage.structured_output``.

    The two backends surface the ``INTERNAL_NOTES_SCHEMA`` output in
    different places. On the raw-Anthropic path ``output_config.format``
    constrains every ``messages.create`` call, so each assistant text
    block *is* the JSON (use :func:`parse_internal_notes`). On the
    claude_agent_sdk path the schema is applied to the run's final
    structured result and returned **already parsed** as
    ``ResultMessage.structured_output`` (a dict) — the per-turn text
    blocks are free-form prose, not JSON.

    Accepts a dict directly; tolerates a JSON string for forward-compat;
    returns ``None`` for anything else (including ``None``).
    """
    if value is None:
        return None
    if isinstance(value, str):
        return parse_internal_notes(value)
    if not isinstance(value, dict):
        logger.warning(
            "structured_output is not a dict: %r", type(value).__name__
        )
        return None
    thought = str(value.get("thought") or "")
    obs_raw = value.get("observations")
    observations: list[str] = []
    if isinstance(obs_raw, list):
        for entry in obs_raw:
            if isinstance(entry, str) and entry.strip():
                observations.append(entry)
    return ParsedNotes(
        thought=thought, observations=observations, raw=json.dumps(value)
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


async def dispatch_outputs(
    outputs: list[dict[str, Any]],
    *,
    conversation_id: str,
    channel_context: str,
    current_speaker: str | None,
    segment_recorder: Optional[SegmentRecorder] = None,
) -> list[DispatchResult]:
    """Dispatch each output entry to its appropriate channel.

    Called by the ``message`` tool (one entry per call) and by
    trigger-fired turns (potentially multiple entries in a batch).

    Returns one ``DispatchResult`` per input entry, in order. The
    ``message`` tool uses this to report real delivery status back to
    the agent; trigger-fired callers may ignore it.

    Args:
        outputs: List of ``{to, channel, content}`` dicts.
        conversation_id: For audit logging.
        channel_context: The conversation's source channel — currently only
            used in log messages for provenance.
        current_speaker: The human the agent is addressing by default.
            Used to resolve ``to: "current_speaker"``.
        segment_recorder: If provided, called with a ``SpokenSegment``
            BEFORE each delivery so the conversation's interrupt-and-
            fold logic can see a partial record if the task is cancelled
            mid-delivery. Optional — non-conversation callers can skip.
    """
    # Import lazily to avoid circular import with agent -> conversation.
    from boxbot.core.conversation import SpokenSegment

    results: list[DispatchResult] = []

    for i, entry in enumerate(outputs):
        to = str(entry.get("to") or "").strip()
        channel = str(entry.get("channel") or "").strip()
        content = str(entry.get("content") or "").strip()

        if not to or not channel or not content:
            logger.warning(
                "Dropping malformed output entry (conv=%s idx=%d): %r",
                conversation_id, i, entry,
            )
            results.append(DispatchResult(
                to=to, channel=channel, status="dropped",
                reason=(
                    "malformed output: to, channel, and content must all "
                    "be non-empty"
                ),
            ))
            continue

        # Resolve "current_speaker" alias
        resolved_to = to
        if to == "current_speaker":
            resolved_to = current_speaker or "unknown"

        if channel == "voice":
            segment = SpokenSegment(
                channel="voice", to=resolved_to, content=content,
            )
            if segment_recorder is not None:
                # Record before dispatch so a cancel mid-TTS still
                # leaves a trace in conv.pending_segments.
                try:
                    segment_recorder(segment)
                except Exception:
                    logger.debug("segment_recorder raised", exc_info=True)
            try:
                result = await _dispatch_voice(
                    to=resolved_to,
                    content=content,
                    conversation_id=conversation_id,
                    channel_context=channel_context,
                )
            except BaseException:
                # Mark the segment interrupted if the delivery was
                # aborted (CancelledError, KeyboardInterrupt, etc.).
                segment.interrupted = True
                raise
            results.append(result)
        elif channel == "text":
            if resolved_to in ("room", "unknown"):
                logger.warning(
                    "Cannot dispatch text to '%s' (conv=%s); dropping. "
                    "Use channel=voice for the room, or name a registered "
                    "user for text.",
                    resolved_to, conversation_id,
                )
                results.append(DispatchResult(
                    to=resolved_to, channel="text", status="dropped",
                    reason=(
                        f"cannot send text to '{resolved_to}'. Text needs a "
                        "registered user's name; use channel='speak' to "
                        "reach whoever is in the room."
                    ),
                ))
                continue
            segment = SpokenSegment(
                channel="text", to=resolved_to, content=content,
            )
            if segment_recorder is not None:
                try:
                    segment_recorder(segment)
                except Exception:
                    logger.debug("segment_recorder raised", exc_info=True)
            try:
                result = await _dispatch_text(
                    to=resolved_to,
                    content=content,
                    conversation_id=conversation_id,
                    channel_context=channel_context,
                )
            except BaseException:
                segment.interrupted = True
                raise
            results.append(result)
        else:
            logger.warning(
                "Unknown channel %r in output entry (conv=%s); dropping: %r",
                channel, conversation_id, entry,
            )
            results.append(DispatchResult(
                to=resolved_to, channel=channel, status="dropped",
                reason=f"unknown channel '{channel}'; use 'speak' or 'text'",
            ))

    return results


async def _dispatch_voice(
    *,
    to: str,
    content: str,
    conversation_id: str,
    channel_context: str,
) -> DispatchResult:
    """Speak ``content`` through the active voice session's TTS.

    ``to`` is logged for audit but does not change routing — the speaker
    plays to the room regardless of who the intended addressee is.
    """
    from boxbot.communication.voice import get_voice_session

    session = get_voice_session()
    if session is None:
        logger.warning(
            "No voice session available; dropping voice output to=%s "
            "(conv=%s chan=%s)",
            to, conversation_id, channel_context,
        )
        return DispatchResult(
            to=to, channel="voice", status="dropped",
            reason=(
                "no active voice session — the box speaker is not "
                "available right now"
            ),
        )

    logger.info(
        "output: voice → %s (conv=%s chan=%s): %s",
        to, conversation_id, channel_context, content[:120],
    )
    try:
        await session.speak(content)
    except Exception:
        logger.exception(
            "Voice dispatch failed (conv=%s to=%s)", conversation_id, to
        )
        return DispatchResult(
            to=to, channel="voice", status="dropped",
            reason="voice playback failed",
        )
    return DispatchResult(to=to, channel="voice", status="delivered")


async def _dispatch_text(
    *,
    to: str,
    content: str,
    conversation_id: str,
    channel_context: str,
) -> DispatchResult:
    """Resolve ``to`` to a registered user and send via their outbound channel.

    The user's ``channel`` column picks which outbound client receives
    the send — WhatsApp for legacy users, Signal once migrated.
    """
    from boxbot.communication.auth import get_auth_manager
    from boxbot.communication.channels import Channel, get_outbound_channel

    auth = get_auth_manager()

    if auth is None:
        logger.warning(
            "Text dispatch unavailable (auth not configured); dropping output "
            "to=%s (conv=%s)",
            to, conversation_id,
        )
        return DispatchResult(
            to=to, channel="text", status="dropped",
            reason="text delivery is unavailable (auth not configured)",
        )

    # Resolve name → user. Exact case-insensitive name match on the user list.
    try:
        users = await auth.list_users()
    except Exception:
        logger.exception("Failed to list users for text dispatch")
        return DispatchResult(
            to=to, channel="text", status="dropped",
            reason="could not look up registered users",
        )

    matched = None
    for user in users:
        if user.name.strip().lower() == to.strip().lower():
            matched = user
            break
        if user.phone == to:  # allow direct phone if the agent provided one
            matched = user
            break

    if matched is None:
        names = [u.name for u in users]
        known = ", ".join(names) or "(no registered users)"
        logger.warning(
            "Cannot resolve '%s' to a registered user; dropping text output "
            "(conv=%s). Known: %s",
            to, conversation_id, known,
        )
        return DispatchResult(
            to=to, channel="text", status="dropped",
            reason=(
                f"unknown recipient '{to}'. Valid recipients: {known}. "
                "Use one of those names, or 'current_speaker'."
            ),
            valid_recipients=names,
        )

    # Pick the outbound channel from the user record.
    try:
        target_channel = Channel(matched.channel)
    except ValueError:
        logger.warning(
            "User %s has unknown channel '%s'; cannot dispatch (conv=%s)",
            matched.phone, matched.channel, conversation_id,
        )
        return DispatchResult(
            to=to, channel="text", status="dropped",
            reason=f"user channel '{matched.channel}' is not recognised",
        )

    out = get_outbound_channel(target_channel)
    if out is None:
        logger.warning(
            "No outbound client registered for channel %s; dropping text to %s "
            "(conv=%s)",
            target_channel.value, matched.phone, conversation_id,
        )
        return DispatchResult(
            to=to, channel="text", status="dropped",
            reason=(
                f"no outbound client registered for channel "
                f"'{target_channel.value}'"
            ),
        )

    logger.info(
        "output: text → %s (%s via %s) (conv=%s chan=%s): %s",
        to, matched.phone, out.name, conversation_id, channel_context,
        content[:120],
    )
    try:
        await out.send_text(matched.phone, content)
    except Exception:
        logger.exception(
            "Text dispatch failed (conv=%s to=%s phone=%s)",
            conversation_id, to, matched.phone,
        )
        return DispatchResult(
            to=to, channel="text", status="dropped",
            reason=f"{out.name} send failed",
        )
    return DispatchResult(to=to, channel="text", status="delivered")
