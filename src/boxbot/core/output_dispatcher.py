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
  entries through voice TTS or a registered outbound channel.

Routing:

- ``channel == "voice"`` → speak through the active voice session's TTS
  (the box speaker). ``to`` is semantic — the audience is whoever is in
  the room; we log the intended addressee for audit.
- ``channel == "text"`` → resolve ``to`` (a name, or ``"current_speaker"``)
  to a registered user via the AuthManager, then send via the outbound
  client for that user's ``channel`` column (WhatsApp or Signal).

Invalid combinations (e.g. ``to: "room"`` with ``channel: "text"``, or
an unknown name with ``channel: "text"``) are logged and dropped — the
run does not crash. So is content that is nothing but filler or
self-referential noise (see ``_is_degenerate_content``). ``dispatch_outputs`` returns one ``DispatchResult``
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
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# A malformed generation can leak the harness tool-call syntax into a
# ``message`` content (observed: an agent texting a user a bare
# ``</antml.parameter>``). Such a fragment is never legitimate outbound
# text, so we drop it rather than deliver garbage to a human.
_TOOL_SYNTAX_FRAGMENT_RE = re.compile(
    r"</?\s*(?:antml|function_calls?|invoke|parameter)\b", re.IGNORECASE
)

# Degenerate ``message`` content — the agent talking to itself out loud.
# Observed on-device: literal "placeholder" and "done" delivered to a
# user, "(no further action needed for now)", then texts apologising for
# those texts. All three classes below are matched EXACTLY, never by
# length or vagueness: "Yes." and "Done — the lights are off." are real
# answers and must survive.

# Closed sets, compared after strip + casefold + trailing ".!" trim.
# Never legitimate outbound text on any channel.
_FILLER_CONTENT = frozenset({"placeholder", "test"})

# Filler in a wake cycle nobody asked for, but a real answer when a human
# asked a yes/no question — "did you lock the door?" / "Done." So these
# are dropped ONLY on the trigger channel, where there is no question to
# be answering.
_TRIGGER_FILLER_CONTENT = frozenset({"done", "ok", "n/a", "none"})

# Content that is entirely a parenthetical aside ("(nothing to do)").
_PARENTHETICAL_ONLY_RE = re.compile(r"\([^()]*\)", re.DOTALL)

# The agent retracting or apologising for its own noise. Anchored to the
# WHOLE message: a genuine "Sorry for the noise above — the lights are
# off now" carries real content and is delivered.
_NOISE_APOLOGY_RE = re.compile(
    r"^(?:(?:i'm |i am )?sorry|apologies|my apologies|oops|whoops|please)?"
    r"[\s,—-]*"
    r"(?:"
    r"(?:please\s+)?(?:ignore|disregard)\s+(?:my|the|that)\s+"
    r"(?:last|previous|preceding|earlier|stray|blank|empty)\s*"
    r"(?:message|text|note|one)?"
    r"|(?:sorry|apologies)\s+(?:for|about)\s+(?:the|that)\s+"
    r"(?:noise|spam|stray|blank|empty|extra|duplicate)"
    r"(?:\s+(?:messages?|texts?))?"
    r")"
    r"[\s.!,]*$",
    re.IGNORECASE,
)

# Machine-readable ``reason_code`` markers for drops that RETRYING CANNOT
# FIX: the same call would be refused the same way. The agent loop's
# trigger backstop ends a turn on these (see
# ``agent._message_results_settled``) and grants a retry on every other
# failure. Anything genuinely recoverable — an unregistered recipient, a
# send error — must stay untagged.
DEGENERATE_CONTENT = "degenerate_content"  # filler; more filler follows
TOOL_SYNTAX = "tool_syntax"                # leaked harness syntax
BUDGET_SPENT = "budget_spent"              # wake-cycle message budget gone
UNRETRYABLE_DROPS = frozenset({DEGENERATE_CONTENT, TOOL_SYNTAX, BUDGET_SPENT})

# Handed back to the agent when a delivery is dropped for degeneracy.
# Points at the flag so it stops instead of retrying with more filler.
_DEGENERATE_CONTENT_REASON = (
    "filler and self-referential content is not deliverable. Say "
    "something the person actually needs, or — if you are finished — "
    "set final_turn=true in your notes instead of sending a message."
)


def _is_degenerate_content(content: str, channel_context: str = "") -> bool:
    """True when ``content`` is nothing but filler or self-referential noise.

    ``channel_context`` is the conversation's source channel. On
    ``"trigger"`` the bare-acknowledgement set is filler too; everywhere
    else a human may have asked the question it answers.
    """
    stripped = content.strip()
    if not stripped:
        return True
    if _PARENTHETICAL_ONLY_RE.fullmatch(stripped):
        return True
    normalised = stripped.casefold().rstrip(" .!")
    if normalised in _FILLER_CONTENT:
        return True
    if channel_context == "trigger" and normalised in _TRIGGER_FILLER_CONTENT:
        return True
    return _NOISE_APOLOGY_RE.match(normalised) is not None

# Conversation channels that carry a human who can be replied to. Speech
# dispatched from one of these is a relay — BB was asked to put something
# to the room and report the answer back, so the mic opens. Notably absent:
# "trigger" (a scheduled run announcing into an empty room has nobody
# waiting) and "voice" (already in the room, mic already managed).
_RELAY_ORIGIN_CHANNELS = frozenset({"signal", "whatsapp"})


# Callback signature used by Conversation.record_segment. Imported
# lazily to avoid a circular dependency — output_dispatcher is imported
# by agent.py which imports conversation.
SegmentRecorder = Callable[[Any], None]


@dataclass
class DispatchResult:
    """Outcome of dispatching one ``{to, channel, content}`` entry.

    ``status`` is ``"delivered"`` or ``"dropped"``. On a drop, ``reason``
    is a human-readable explanation safe to hand back to the agent as a
    tool result, and ``reason_code`` is the machine-readable class of
    drop for callers that must branch on it (currently only
    ``DEGENERATE_CONTENT``; see ``agent._message_results_settled``).
    ``valid_recipients`` is populated only when the drop was
    caused by an unresolvable recipient name, so the agent can retry with
    a real name.
    """

    to: str
    channel: str
    status: str
    reason: str = ""
    reason_code: str = ""
    valid_recipients: Optional[list[str]] = field(default=None)


# ---------------------------------------------------------------------------
# Schema — pinned via ``output_config.format`` on every messages.create call
#
# The schema has NO delivery channel by design. Its sole job is to occupy
# the model's text-output slot with a private scratchpad shape, removing
# the trained "respond as plain text to the user" affordance. To reach a
# human, the agent MUST call the ``message`` tool. ``final_turn`` is the
# one exception to "private": it is control, not prose — it ends the
# agent loop (see ``agent._agent_loop``).
#
# Mutating this schema invalidates the messages cache once at deploy.
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
        "final_turn": {
            "type": "boolean",
            "description": (
                "True on the response that finishes the job — typically "
                "alongside your final message call. False while more tool "
                "work remains. Never call message just to signal you are "
                "done; set this instead."
            ),
        },
    },
    # ``final_turn`` is required, not optional: the OpenAI path runs this
    # schema through strict mode (every property required) and a required
    # flag forces an explicit continue/stop decision on every response.
    "required": ["thought", "final_turn"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class ParsedNotes:
    """Result of parsing one agent text block as internal notes."""

    __slots__ = ("thought", "observations", "final_turn", "raw")

    def __init__(
        self,
        thought: str,
        observations: list[str],
        raw: str,
        final_turn: bool = False,
    ) -> None:
        self.thought = thought
        self.observations = observations
        self.final_turn = final_turn
        self.raw = raw


def _notes_from_dict(data: dict[str, Any], raw: str) -> ParsedNotes:
    """Build ``ParsedNotes`` from an already-decoded notes object."""
    obs_raw = data.get("observations")
    observations: list[str] = []
    if isinstance(obs_raw, list):
        for entry in obs_raw:
            if isinstance(entry, str) and entry.strip():
                observations.append(entry)
    return ParsedNotes(
        thought=str(data.get("thought") or ""),
        observations=observations,
        raw=raw,
        final_turn=bool(data.get("final_turn")),
    )


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
    return _notes_from_dict(data, raw_text)


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
    return _notes_from_dict(value, json.dumps(value))


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
        channel_context: The conversation's source channel. Logged for
            provenance, decides whether speech is a relay (see
            ``_dispatch_voice``), and widens the filler filter on
            ``"trigger"`` (see ``_is_degenerate_content``).
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
        attachments = [
            str(a) for a in (entry.get("attachments") or []) if a
        ]

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

        if _TOOL_SYNTAX_FRAGMENT_RE.search(content):
            logger.warning(
                "Dropping output entry with leaked tool-call syntax "
                "(conv=%s idx=%d): %r",
                conversation_id, i, content,
            )
            results.append(DispatchResult(
                to=to, channel=channel, status="dropped",
                reason="content contains tool-call syntax; not delivered",
                reason_code=TOOL_SYNTAX,
            ))
            continue

        if _is_degenerate_content(content, channel_context):
            logger.warning(
                "Dropping degenerate output entry (conv=%s idx=%d): %r",
                conversation_id, i, content,
            )
            results.append(DispatchResult(
                to=to, channel=channel, status="dropped",
                reason=_DEGENERATE_CONTENT_REASON,
                reason_code=DEGENERATE_CONTENT,
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
                    origin_person=current_speaker,
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
                    attachments=attachments,
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
    origin_person: str | None = None,
) -> DispatchResult:
    """Speak ``content`` through the active voice session's TTS.

    ``to`` is logged for audit but does not change routing — the speaker
    plays to the room regardless of who the intended addressee is.

    When the speech originates from a human's *text* conversation, BB is
    relaying: someone asked it to put a question to the room and report
    back. Speaking alone is not enough — the mic must be open to hear the
    answer, and the room conversation that hears it must know what was
    asked and who to tell. We hand the voice session a
    :class:`RelayContext` and let it open the room, rather than speaking
    into a box that is not listening.

    A trigger conversation is *not* a relay even though it is not voice:
    a timer announcement or a morning briefing has nobody on the other
    end waiting for an answer. It speaks and stops — the mic stays shut.
    """
    from boxbot.communication.voice import get_voice_session
    from boxbot.core.conversation import RelayContext

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

    # A relay needs a human to report back to. Both halves matter: the
    # channel must be one we can reply on, and we must know who asked.
    is_relay = channel_context in _RELAY_ORIGIN_CHANNELS and bool(origin_person)
    logger.info(
        "output: voice → %s (conv=%s chan=%s%s): %s",
        to, conversation_id, channel_context,
        " relay" if is_relay else "", content[:120],
    )
    try:
        if is_relay:
            await session.speak_and_listen(
                content,
                relay=RelayContext(
                    origin_conversation_id=conversation_id,
                    origin_channel=channel_context,
                    origin_person=origin_person or "",
                    addressee=to,
                    spoken_text=content,
                ),
            )
        else:
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
    attachments: list[str] | None = None,
) -> DispatchResult:
    """Resolve ``to`` to a registered user and send via their outbound channel.

    The user's ``channel`` column picks which outbound client receives
    the send — WhatsApp for legacy users, Signal once migrated.

    ``attachments`` are absolute, already-validated image paths (the
    message tool checks the allowlist). The first rides with ``content``
    as its caption; the rest follow bare. A transport without
    ``send_attachment`` gets the text alone and the result says so.
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

    # Resolve name → user, falling back to a direct phone if the agent
    # provided one. The name list is only needed to name valid
    # recipients when nothing matches.
    try:
        matched = await auth.get_user_by_name(to) or await auth.get_user(to)
        names = [] if matched else [u.name for u in await auth.list_users()]
    except Exception:
        logger.exception("Failed to resolve '%s' for text dispatch", to)
        return DispatchResult(
            to=to, channel="text", status="dropped",
            reason="could not look up registered users",
        )

    if matched is None:
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
        "output: text → %s (%s via %s) (conv=%s chan=%s%s): %s",
        to, matched.phone, out.name, conversation_id, channel_context,
        f" attachments={len(attachments)}" if attachments else "",
        content[:120],
    )
    try:
        if attachments:
            send_attachment = getattr(out, "send_attachment", None)
            if send_attachment is None:
                await out.send_text(matched.phone, content)
                return DispatchResult(
                    to=to, channel="text", status="delivered",
                    reason=(
                        f"{out.name} cannot carry attachments; the text "
                        "was sent without them"
                    ),
                )
            ok = await send_attachment(
                matched.phone, attachments[0], caption=content,
            )
            for extra in attachments[1:]:
                ok = await send_attachment(matched.phone, extra) and ok
            if not ok:
                return DispatchResult(
                    to=to, channel="text", status="dropped",
                    reason=f"{out.name} attachment send failed",
                )
        else:
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
