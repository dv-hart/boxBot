"""Output dispatcher — route the agent's structured outputs to real channels.

The agent emits structured JSON per turn matching ``OUTPUT_SCHEMA``:

    {
      "thought": "private reasoning",
      "outputs": [
        {"to": "<person|'current_speaker'|'room'>",
         "channel": "voice" | "text",
         "content": "..."}
      ]
    }

This module parses such JSON blocks and dispatches each output entry:

- ``channel == "voice"`` → speak through the active voice session's TTS
  (the box speaker). ``to`` is semantic — the audience is whoever is in the
  room; we log the intended addressee for audit.
- ``channel == "text"`` → resolve ``to`` (a name, or ``"current_speaker"``)
  to a registered user's phone number via the AuthManager, then send via
  the WhatsApp client.

Invalid combinations (e.g. ``to: "room"`` with ``channel: "text"``, or an
unknown name with ``channel: "text"``) are logged and dropped. The agent
learns from the prompt + dispatcher warnings; the run does not crash.

The dispatcher is stateless; all dependencies come from module singletons
that ``main.py`` sets up at boot (``get_voice_session``,
``get_whatsapp_client``, ``get_auth_manager``).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema — pinned via ``output_config.format`` on every messages.create call
# ---------------------------------------------------------------------------

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {
            "type": "string",
            "description": (
                "Short private reasoning for this turn. Never broadcast. "
                "Used for post-conversation memory extraction and logs."
            ),
        },
        "outputs": {
            "type": "array",
            "description": (
                "Zero or more messages to deliver. Empty array = silent turn "
                "(you noticed something but do not wish to respond)."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": (
                            "Recipient identifier. Use \"current_speaker\" "
                            "for whoever addressed you most recently in this "
                            "conversation. Use \"room\" to broadcast voice to "
                            "anyone physically present (voice channel only). "
                            "Otherwise use a registered user's name exactly "
                            "as it appears in the Registered users list."
                        ),
                    },
                    "channel": {
                        "enum": ["voice", "text"],
                        "description": (
                            "Delivery medium. \"voice\" plays TTS through "
                            "the box speaker (everyone in the room hears). "
                            "\"text\" sends a WhatsApp message to the "
                            "named user's registered phone."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "The exact words to deliver.",
                    },
                },
                "required": ["to", "channel", "content"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["thought", "outputs"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class ParsedBlock:
    """Result of parsing one agent text block as a structured output."""

    __slots__ = ("thought", "outputs", "raw")

    def __init__(self, thought: str, outputs: list[dict[str, Any]], raw: str):
        self.thought = thought
        self.outputs = outputs
        self.raw = raw


def parse_output_block(raw_text: str) -> ParsedBlock | None:
    """Parse a JSON text block emitted by the agent into a ParsedBlock.

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
    outputs_raw = data.get("outputs")
    outputs: list[dict[str, Any]] = []
    if isinstance(outputs_raw, list):
        for entry in outputs_raw:
            if isinstance(entry, dict):
                outputs.append(entry)
    return ParsedBlock(thought=thought, outputs=outputs, raw=raw_text)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


async def dispatch_outputs(
    outputs: list[dict[str, Any]],
    *,
    conversation_id: str,
    channel_context: str,
    current_speaker: str | None,
) -> None:
    """Dispatch each output entry to its appropriate channel.

    Args:
        outputs: List of ``{to, channel, content}`` dicts from the agent.
        conversation_id: For audit logging.
        channel_context: The conversation's source channel — currently only
            used in log messages for provenance.
        current_speaker: The human the agent is addressing by default.
            Used to resolve ``to: "current_speaker"``.
    """
    for i, entry in enumerate(outputs):
        to = str(entry.get("to") or "").strip()
        channel = str(entry.get("channel") or "").strip()
        content = str(entry.get("content") or "").strip()

        if not to or not channel or not content:
            logger.warning(
                "Dropping malformed output entry (conv=%s idx=%d): %r",
                conversation_id, i, entry,
            )
            continue

        # Resolve "current_speaker" alias
        resolved_to = to
        if to == "current_speaker":
            resolved_to = current_speaker or "unknown"

        if channel == "voice":
            await _dispatch_voice(
                to=resolved_to,
                content=content,
                conversation_id=conversation_id,
                channel_context=channel_context,
            )
        elif channel == "text":
            if resolved_to in ("room", "unknown"):
                logger.warning(
                    "Cannot dispatch text to '%s' (conv=%s); dropping. "
                    "Use channel=voice for the room, or name a registered "
                    "user for text.",
                    resolved_to, conversation_id,
                )
                continue
            await _dispatch_text(
                to=resolved_to,
                content=content,
                conversation_id=conversation_id,
                channel_context=channel_context,
            )
        else:
            logger.warning(
                "Unknown channel %r in output entry (conv=%s); dropping: %r",
                channel, conversation_id, entry,
            )


async def _dispatch_voice(
    *,
    to: str,
    content: str,
    conversation_id: str,
    channel_context: str,
) -> None:
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
        return

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


async def _dispatch_text(
    *,
    to: str,
    content: str,
    conversation_id: str,
    channel_context: str,
) -> None:
    """Resolve ``to`` to a registered user's phone and send via WhatsApp."""
    from boxbot.communication.auth import get_auth_manager
    from boxbot.communication.whatsapp import get_whatsapp_client

    auth = get_auth_manager()
    wa = get_whatsapp_client()

    if auth is None or wa is None:
        logger.warning(
            "Text dispatch unavailable (auth=%s wa=%s); dropping output "
            "to=%s (conv=%s)",
            auth is not None, wa is not None, to, conversation_id,
        )
        return

    # Resolve name → phone. Exact case-insensitive name match on the user list.
    try:
        users = await auth.list_users()
    except Exception:
        logger.exception("Failed to list users for text dispatch")
        return

    phone: str | None = None
    for user in users:
        if user.name.strip().lower() == to.strip().lower():
            phone = user.phone
            break
        if user.phone == to:  # allow direct phone if the agent provided one
            phone = user.phone
            break

    if phone is None:
        known = ", ".join(u.name for u in users) or "(no registered users)"
        logger.warning(
            "Cannot resolve '%s' to a registered user; dropping text output "
            "(conv=%s). Known: %s",
            to, conversation_id, known,
        )
        return

    logger.info(
        "output: text → %s (%s) (conv=%s chan=%s): %s",
        to, phone, conversation_id, channel_context, content[:120],
    )
    try:
        await wa.send_text(phone, content)
    except Exception:
        logger.exception(
            "Text dispatch failed (conv=%s to=%s phone=%s)",
            conversation_id, to, phone,
        )
