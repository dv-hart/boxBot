"""Claude Agent integration — the brain of boxBot.

Wraps the Anthropic Python SDK directly (``anthropic.AsyncAnthropic``) to
orchestrate conversations, dispatch tool calls, manage system prompt
construction with context injection, and trigger post-conversation memory
extraction.

**Do NOT migrate this to ``claude-agent-sdk`` / ``claude-code-sdk``.** Those
packages bundle Claude Code and its built-in filesystem/coding tools. boxBot
is a hardware-facing ambient assistant with bespoke tools (``switch_display``,
``identify_person``, etc.). See
``docs/plans/implementation-spec-2026-04-23.md`` §1 for full rationale.

Key design points for this file:

1. **Private-by-design text output** — every ``messages.create`` call
   passes ``output_config={"format": {"type": "json_schema", "schema":
   INTERNAL_NOTES_SCHEMA}}``. The model's text output is constrained to
   a private JSON shape: ``thought`` (private reasoning) + optional
   ``observations`` (ambient facts). By construction nothing in the
   text reaches a person — it is a labelled scratchpad consumed by
   logging and post-conversation memory extraction.

   To reach a person, the agent calls the ``message`` tool
   (``{to, channel, content}``). Multiple calls per turn are allowed
   and expected: filler-then-tool, voice-to-room plus text-to-spouse,
   etc. This avoids the "constrained-JSON ends the turn" trap, where a
   filler dispatched as JSON outputs would terminate the response
   before any tool ran.

2. **Prompt caching** — system prompt is a list of text blocks. The static
   block (persona + etiquette + capabilities + skills index) carries a 1h
   cache marker; the dynamic block (who's present, time, memories) does
   not. The last tool definition carries a 1h marker to cache the tools
   array. A top-level ``cache_control={"type": "ephemeral"}`` enables the
   5-minute rolling messages cache.

3. **No banned params on Opus 4.7** — we never pass ``temperature``,
   ``top_p``, ``top_k``, or ``thinking.budget_tokens``. ``max_tokens`` is
   bumped to 8192 for headroom under the new token accounting.

Usage:
    from boxbot.core.agent import BoxBotAgent

    agent = BoxBotAgent(memory_store)
    await agent.start()
    await agent.handle_conversation(
        channel="voice",
        initial_message="What's the weather like?",
        person_name="Jacob",
    )
    await agent.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anthropic

if TYPE_CHECKING:
    from boxbot.conversations.store import ConversationStore

from boxbot.core import latency
from boxbot.core.config import get_config
from boxbot.cost import (
    from_agent_sdk_result,
    from_anthropic_usage,
    record as record_cost,
)
from boxbot.core.conversation import (
    Conversation,
    ConversationState,
    GenerationResult,
    SpokenSegment,
)
from boxbot.core.events import (
    AgentSpeaking,
    AgentSpeakingDone,
    ConversationEnded,
    ConversationInterruptRequested,
    PersonIdentified,
    SignalMessage,
    SpeakerIdentified,
    TranscriptReady,
    TriggerFired,
    VoiceSessionEnded,
    WhatsAppMessage,
    get_event_bus,
)
from boxbot.core.output_dispatcher import (
    INTERNAL_NOTES_SCHEMA,
    parse_internal_notes,
)
from boxbot.core.scheduler import get_status_line
from boxbot.memory.batch_poller import BatchPoller
from boxbot.memory.dream_poller import DreamPoller
from boxbot.memory.retrieval import inject_memories
from boxbot.memory.store import MemoryStore
# Imported lazily inside _dispatch_tools / _process_tool_calls to avoid a
# circular import: registry → builtins.execute_script → core.config →
# core/__init__ → core.agent → (back here).

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of API round-trips (tool use loops) per conversation
_DEFAULT_MAX_TURNS = 25

# Opus 4.7 needs more headroom than Sonnet 4. Spec §3 mandates 8192.
_MAX_TOKENS = 8192

# The structured-output schema is defined in ``output_dispatcher`` so the
# dispatcher and the agent loop share a single source of truth. Schema
# mutation invalidates the messages cache — it is pinned at module scope
# there and imported here.


def _generate_conversation_id() -> str:
    """Generate a unique conversation ID."""
    return f"conv_{uuid.uuid4().hex[:12]}"


# Mime type → file extension for inbound images (WhatsApp + Signal).
# Restricted to the formats the multimodal attach pipeline accepts
# (build_image_block).
_INBOUND_IMAGE_EXTS: dict[str, str] = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
}


async def _stage_whatsapp_image(media_id: str, message_id: str) -> Path | None:
    """Download an inbound WhatsApp image to the sandbox tmp dir.

    Lands at ``{sandbox.tmp_dir}/inbound/whatsapp/{message_id}.{ext}``.
    The directory inherits group ``boxbot`` (setgid on tmp/), so the
    sandbox user can read the staged file. Bytes are owned by the
    main-process user.

    Returns the staged path on success, or None if the WhatsApp client
    is not configured, the download fails, or the mime type isn't
    supported by the multimodal attach pipeline.
    """
    from boxbot.communication.whatsapp import get_whatsapp_client

    client = get_whatsapp_client()
    if client is None:
        logger.warning("WhatsApp image %s: client not configured", media_id)
        return None

    result = await client.download_media(media_id)
    if result is None:
        return None
    data, mime_type = result

    # Trust the bytes, not the server-claimed MIME: sniff the real format
    # from magic bytes and reject anything that isn't a supported image.
    from boxbot.photos.imageutil import sniff_image_mime

    sniffed = sniff_image_mime(data)
    if sniffed is None:
        logger.warning(
            "WhatsApp image %s: not a recognised image (claimed %s)",
            media_id, mime_type,
        )
        return None
    ext = _INBOUND_IMAGE_EXTS[sniffed]

    try:
        from boxbot.core.config import get_config

        tmp_dir = Path(get_config().sandbox.tmp_dir)
    except Exception:
        tmp_dir = Path("/var/lib/boxbot-sandbox/tmp")

    inbound_dir = tmp_dir / "inbound" / "whatsapp"
    try:
        inbound_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("WhatsApp inbound dir create failed: %s", e)
        return None

    # message_id is a Meta-issued opaque token (e.g. ``wamid.HBg…``).
    # Strip path separators defensively before using it as a filename.
    safe_id = message_id.replace("/", "_").replace("\\", "_") or uuid.uuid4().hex
    dest = inbound_dir / f"{safe_id}.{ext}"
    try:
        dest.write_bytes(data)
    except OSError as e:
        logger.warning("WhatsApp image write failed for %s: %s", dest, e)
        return None

    logger.info("Staged WhatsApp image %s → %s (%d bytes)", media_id, dest, len(data))
    return dest


async def _stage_signal_image(
    attachment_id: str, message_id: str
) -> Path | None:
    """Read an inbound Signal image from the signal-cli cache and stage it.

    signal-cli auto-downloads attachments to its local data dir; we just
    copy into the sandbox-readable inbound staging path so the agent's
    multimodal attach pipeline can use it. Lands at
    ``{sandbox.tmp_dir}/inbound/signal/{message_id}.{ext}``.

    Returns the staged path on success, or None if the client is not
    configured, the file isn't on disk, or the mime type isn't accepted.
    """
    from boxbot.communication.signal_client import get_signal_client

    client = get_signal_client()
    if client is None:
        logger.warning("Signal image %s: client not configured", attachment_id)
        return None

    result = await client.download_media(attachment_id)
    if result is None:
        return None
    data, mime_type = result

    # Trust the bytes, not the claimed/extension MIME: sniff the real
    # format from magic bytes and reject anything that isn't an image.
    from boxbot.photos.imageutil import sniff_image_mime

    sniffed = sniff_image_mime(data)
    if sniffed is None:
        logger.warning(
            "Signal image %s: not a recognised image (claimed %s)",
            attachment_id, mime_type,
        )
        return None
    ext = _INBOUND_IMAGE_EXTS[sniffed]

    try:
        from boxbot.core.config import get_config

        tmp_dir = Path(get_config().sandbox.tmp_dir)
    except Exception:
        tmp_dir = Path("/var/lib/boxbot-sandbox/tmp")

    inbound_dir = tmp_dir / "inbound" / "signal"
    try:
        inbound_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("Signal inbound dir create failed: %s", e)
        return None

    safe_id = message_id.replace("/", "_").replace("\\", "_") or uuid.uuid4().hex
    dest = inbound_dir / f"{safe_id}.{ext}"
    try:
        dest.write_bytes(data)
    except OSError as e:
        logger.warning("Signal image write failed for %s: %s", dest, e)
        return None

    logger.info(
        "Staged Signal image %s → %s (%d bytes)", attachment_id, dest, len(data)
    )
    return dest


def _render_identity_section(
    identities: dict[str, dict[str, Any]] | None,
) -> str:
    """Render the per-session identity block for the dynamic context.

    Surfaces voice (and, when present, visual) ReID tiers + scores per
    speaker so the agent can pick the right behavior — address by name on
    high, verify on medium/low, defer to the onboarding skill on unknown.

    Returns an empty string if there's nothing to render.
    """
    if not identities:
        return ""

    lines: list[str] = []
    for display_label, info in identities.items():
        voice_tier = info.get("voice_tier", "unknown")
        voice_score = info.get("voice_score")
        visual_tier = info.get("visual_tier")
        visual_score = info.get("visual_score")
        person_name = info.get("person_name")
        source = info.get("source", "unknown")

        voice_bit = f"voice: {voice_tier}"
        if isinstance(voice_score, (int, float)) and voice_tier != "unknown":
            voice_bit += f" ({voice_score:.2f})"

        visual_bit = ""
        if visual_tier:
            visual_bit = f"   visual: {visual_tier}"
            if isinstance(visual_score, (int, float)) and visual_tier != "unknown":
                visual_bit += f" ({visual_score:.2f})"

        # Headline per speaker
        if person_name and source == "agent_identify":
            headline = f"- {display_label}  → confirmed as {person_name}"
        elif person_name and voice_tier == "high":
            headline = f"- {display_label}  → likely {person_name}"
        elif person_name and voice_tier in ("medium", "low"):
            headline = f"- {display_label}  → possibly {person_name}"
        else:
            headline = f"- {display_label}  → not recognized"

        lines.append(headline)
        lines.append(f"    {voice_bit}{visual_bit}")

    guidance = (
        "\n"
        "How to address each speaker, based on the tier:\n"
        "- high (voice ≥0.85 or confirmed): address by name, no hedging.\n"
        "- medium (voice 0.70-0.85): soft-verify before committing, e.g.\n"
        "  \"Hey Sarah — correct me if that's wrong?\" or similar. If they\n"
        "  confirm, call identify_person to pin it. If they correct, call\n"
        "  identify_person with the corrected name.\n"
        "- low (voice 0.60-0.70): weak match; don't assume. Ask who they\n"
        "  are, or load the onboarding skill if they're newly addressing you.\n"
        "- unknown: treat as a first meeting. Load the `onboarding` skill\n"
        "  for the procedure — do NOT guess a name."
    )

    return "## People in this session\n" + "\n".join(lines) + "\n" + guidance


# ---------------------------------------------------------------------------
# System-prompt helpers — each returns a string, composed into blocks below.
# Keep these FREE of timestamps / UUIDs / anything non-deterministic; the
# static block is cache-controlled with 1h TTL and will be invalidated by
# any per-call variability. Dynamic content lives in _prompt_dynamic_context.
# ---------------------------------------------------------------------------


def _prompt_persona(name: str, wake_word: str) -> str:
    """Return the persona / identity section of the static system prompt."""
    return (
        f"You are {name}, an ambient household assistant that lives in an "
        "elegant wooden box. You see through a camera, hear through a "
        "microphone array, speak through a speaker, and display information "
        "on a 7-inch screen. You communicate with your household via voice "
        "and text messages.\n\n"
        "You recognise the people around you and proactively help them — "
        "relaying messages, managing tasks, controlling displays, and "
        "remembering everything important about your household. You are warm, "
        "concise, and genuinely useful. You know when to speak up and when "
        "to stay quiet.\n\n"
        f"Your wake word is \"{wake_word}\"."
    )


def _agent_facing_channel(channel: str) -> str:
    """Normalise the internal channel id to what the agent should see.

    The agent reasons about *modality*, not the messaging vendor. Every
    text platform (WhatsApp, Signal, …) collapses to ``"text"`` so the
    prompt never names a vendor and the agent's channel choice stays
    platform-agnostic. Voice and trigger pass through unchanged.
    """
    if channel in ("whatsapp", "signal"):
        return "text"
    return channel


def _prompt_etiquette() -> str:
    """Return the multi-speaker + delivery-mechanics section.

    The big idea: the model's text output is PRIVATE (internal notes
    only). To reach a person, the model must call ``message``.
    Multiple calls per turn are allowed and expected.
    """
    return (
        "## Your text output never reaches a person\n"
        "\n"
        "Every time you respond, your text output is constrained to a private\n"
        "JSON note shape:\n"
        "\n"
        "    {\n"
        "      \"thought\": \"private reasoning for this turn\",\n"
        "      \"observations\": [\"things you noticed but didn't act on\", ...]\n"
        "    }\n"
        "\n"
        "- `thought` is your scratchpad for this turn. Nobody sees it.\n"
        "- `observations` collects ambient facts you noticed (mood, who's in\n"
        "  the room, what people are doing, things mentioned in passing).\n"
        "  Memory extraction reads this after the conversation. Optional.\n"
        "- Both fields are PRIVATE. The user hears NOTHING from your text\n"
        "  output. Saying \"Sure, here's the answer...\" in your text does not\n"
        "  reach anyone.\n"
        "\n"
        "## To reach a person, call message\n"
        "\n"
        "`message(to, channel, content)` is the ONLY way to actually\n"
        "speak through the box speaker or send a text message. Without a\n"
        "call to it, the user gets silence — no matter what your text said.\n"
        "\n"
        "Multiple calls per turn are normal and encouraged:\n"
        "- Interim acknowledgement before a tool call (filler), AND\n"
        "  the tool call itself, in the same response. Both run, you get\n"
        "  the tool result, and you call message again with the\n"
        "  final answer next turn.\n"
        "- Voice to the room AND text to an absent user, in the same response.\n"
        "\n"
        "Recipient (`to`):\n"
        "- `\"current_speaker\"` — whoever just addressed you. Resolves at\n"
        "  dispatch time.\n"
        "- `\"room\"` — broadcast spoken audio to anyone physically present.\n"
        "  Speak channel only.\n"
        "- A registered user's name exactly as listed in the Registered users\n"
        "  block of the dynamic context (e.g. `\"Sarah\"`).\n"
        "\n"
        "Channel:\n"
        "- `\"speak\"` — speak through the box speaker. Everyone in the room\n"
        "  hears. Does not reach absent people.\n"
        "- `\"text\"` — text message to the named user's phone. Requires\n"
        "  `to` be a registered user by name. Cannot text \"room\" or unknowns.\n"
        "\n"
        "Default to the channel you were contacted through — the current one\n"
        "is shown as `Channel:` in the dynamic context. In a `text`\n"
        "conversation reply with `text` (the person is not at the box and\n"
        "will not hear speech); in a `voice` conversation reply with `speak`.\n"
        "Switch only when it clearly makes sense (e.g. someone at the box\n"
        "asks you to text an absent person).\n"
        "\n"
        "## When to deliver vs stay silent\n"
        "\n"
        "You hear every utterance in the room. Diarization labels each speaker\n"
        "(e.g. \"[Jacob]: ...\" or \"[Unknown_1]: ...\"). Speakers are labeled\n"
        "for you; never expose internal labels like SPEAKER_00 to the user.\n"
        "\n"
        "Call message when:\n"
        "- You are directly addressed (\"BB ...\", \"Jarvis ...\", direct question).\n"
        "- You have an urgent, useful message to deliver (timer fired, a\n"
        "  correction, a notification someone asked to receive).\n"
        "- There is a clear action you should take from what was said\n"
        "  (\"add that to my list\"), in which case you confirm it.\n"
        "- A trigger fires and you need to deliver whatever it was set up to do.\n"
        "\n"
        "Stay silent (do NOT call message) when:\n"
        "- People are talking to each other, not to you.\n"
        "- They are thinking out loud or processing.\n"
        "- You already answered and they are confirming among themselves.\n"
        "\n"
        "When silent, the transcript still records what was said — memory\n"
        "extraction runs after the conversation and can pick up ambient facts.\n"
        "You do not need to explicitly note things you overheard; the system\n"
        "captures them. Use `thought` for private reasoning and `observations`\n"
        "for things worth flagging to memory extraction.\n"
        "\n"
        "When uncertain, prefer silence. When addressed by name, always deliver.\n"
        "\n"
        "## Noisy transcripts and overheard speech\n"
        "\n"
        "The transcript you see is best-effort. Things to expect:\n"
        "- Speech-to-text occasionally produces garbled or nonsensical\n"
        "  fragments when audio quality is low or someone speaks far from\n"
        "  the box. Treat these as noise.\n"
        "- More than one person may be in the room. You will see utterances\n"
        "  from people who are talking to each other, to a child, to a pet,\n"
        "  or to the TV — not to you.\n"
        "- Utterances that arrived while you were speaking are queued and\n"
        "  delivered to you on the next turn. They are particularly likely\n"
        "  to be overheard, not addressed to you.\n"
        "\n"
        "When you have a clear active task and additional input arrives\n"
        "that looks unrelated, garbled, or like background chatter, ignore\n"
        "it and proceed with the original task. When the additional input\n"
        "is a clear continuation or correction (\"oh, and also...\",\n"
        "\"wait, make that tomorrow instead\"), incorporate it. The signal\n"
        "is meaning, not volume — short fragments from kids' shows,\n"
        "song lyrics, or half-heard side conversations should not change\n"
        "your course of action. Continuations from the person who gave you\n"
        "the task usually should.\n"
        "\n"
        "## Muting the mic while you work — be DECISIVE\n"
        "\n"
        "You have `mute_mic(reason=\"...\")`. The wake word stays armed,\n"
        "so anyone in the room can re-engage you instantly. Muting does\n"
        "NOT interrupt your in-flight tools or the API call you're inside\n"
        "— it only stops FUTURE transcripts from yanking you off course.\n"
        "\n"
        "The rule: **if you have an active task AND the next transcript\n"
        "is unrelated to that task, mute on the SAME turn**. Do not wait\n"
        "for a second confirming turn. People follow conversational\n"
        "threads — once someone starts a side conversation with their\n"
        "spouse / kids / a guest, it almost always continues for several\n"
        "more turns. Burning an API call to stay silent on each fragment\n"
        "is exactly the failure mode mute_mic exists to prevent.\n"
        "\n"
        "Mute IMMEDIATELY when:\n"
        "- You were given a task and a transcript arrives that is not\n"
        "  the original speaker continuing or correcting their request.\n"
        "- A speaker is clearly addressing someone else in the room\n"
        "  (a child, spouse, guest, the TV).\n"
        "- The transcript is garbled, empty, fragmentary, or a Scribe\n"
        "  artefact.\n"
        "- The room switches to a language that has only ever been used\n"
        "  for side talk in this conversation.\n"
        "\n"
        "When mid-task and uncertain whether an utterance is for you,\n"
        "default to muting. The wake word brings the user right back if\n"
        "they actually need you. Wrong mute = one extra wake word. Wrong\n"
        "no-mute = a derailed task and a response the user never asked for.\n"
        "\n"
        "Important: do NOT call `mute_mic` in the same turn you call\n"
        "`message(channel=\"speak\")`. Speech automatically re-opens the\n"
        "mic at the end of TTS, so the two cancel out. If you want to\n"
        "say one last thing and let the room go quiet, just call\n"
        "`message(speak)` — if nothing more arrives the silence timer\n"
        "ends the conversation. If you want to answer AND stay muted,\n"
        "speak first; on your NEXT turn, mute.\n"
        "\n"
        "Calling `mute_mic` drops any utterances that were queued while\n"
        "you were thinking — that IS the point. Those queued lines are\n"
        "the same side-talk you are choosing to ignore.\n"
        "\n"
        "## Scenarios (memorise these patterns)\n"
        "\n"
        "- Someone at the box asks you a question you can answer immediately:\n"
        "    message(to=\"current_speaker\", channel=\"speak\",\n"
        "                     content=\"...\")\n"
        "\n"
        "- Someone texted you and you're replying:\n"
        "    message(to=\"current_speaker\", channel=\"text\",\n"
        "                     content=\"...\")\n"
        "\n"
        "- Someone at the box asks you to text their spouse — TWO calls in the\n"
        "  same response:\n"
        "    message(to=\"current_speaker\", channel=\"speak\",\n"
        "                     content=\"Okay, texting Sarah now.\")\n"
        "    message(to=\"Sarah\", channel=\"text\",\n"
        "                     content=\"Jacob says he'll be late.\")\n"
        "\n"
        "- You need a second to look something up — call message for\n"
        "  the filler AND the lookup tool in the same response. Send the\n"
        "  filler on the SAME channel you'll answer on — `text` in a text\n"
        "  conversation, `speak` in a voice one:\n"
        "    message(to=\"current_speaker\", channel=<conversation channel>,\n"
        "                     content=\"Sure thing, let me find that for you.\")\n"
        "    execute_script(...)   # or web_search, search_memory, etc.\n"
        "  Both fire. The tool result comes back next turn. You then call\n"
        "  message once more with the actual answer. Do not repeat\n"
        "  the filler in the final answer.\n"
        "\n"
        "- Someone asks to be notified later: call manage_tasks to set up a\n"
        "  person-trigger, then message with a confirmation. When the\n"
        "  trigger fires later, message delivers the message.\n"
        "\n"
        "- Ambient chat you're not part of: emit your private thought (e.g.\n"
        "  observations=[\"Jacob and Sarah are discussing weekend plans\"])\n"
        "  and DO NOT call message. Silence.\n"
        "\n"
        "## Within a responding turn\n"
    )


def _prompt_capabilities() -> str:
    """Return the capabilities / guidelines section of the static prompt.

    Tool list is narrated here; full schemas go via the ``tools=`` parameter
    of ``messages.create``. All speech and messaging to humans flows through
    the ``message`` tool described in ``_prompt_etiquette``.
    """
    return (
        "## Capabilities\n"
        "\n"
        "Always-available tools: message, execute_script, "
        "switch_display, identify_person, manage_tasks, mute_mic, "
        "search_memory, search_photos, web_search, and load_skill. For "
        "complex or infrequent operations, use execute_script to run "
        "Python in the sandbox with the boxbot_sdk.\n"
        "\n"
        "message is the ONLY tool that reaches a person. The other\n"
        "tools DO things — create a reminder, search memory, switch a\n"
        "display, run a sandbox script, look up the web. None of them speak.\n"
        "Without a message call, the user hears silence.\n"
        "\n"
        "## Injected context is not ground truth\n"
        "- `[Recent Conversations]` entries are RECEIPTS — pointers to\n"
        "  what was discussed, not statements of current fact. \"Discussed\n"
        "  the calendar\" does not mean the calendar is broken. If a\n"
        "  receipt looks relevant, go read the actual conversation.\n"
        "- For current state — weather, calendar, the to-do list, what's\n"
        "  on the display — query the live source every time. Never report\n"
        "  state from an injected summary or from your own past output.\n"
        "- Your own past words are never authoritative. A briefing you\n"
        "  sent yesterday, a summary you wrote, an observation you logged —\n"
        "  none of it is evidence. Evidence is: live data, the human's\n"
        "  words, and curated memories.\n"
        "\n"
        "## Guidelines\n"
        "- Be concise when speaking — no one wants a lecture from a box.\n"
        "- Prefer text for long, detailed information (easier to re-read);\n"
        "  prefer speak for quick, immediate replies when someone is at the box.\n"
        "- When you learn something important, memory extraction captures it\n"
        "  automatically after the conversation ends. Do not repeat facts in\n"
        "  `thought` — it adds noise.\n"
        "- You can search your memories at any time with search_memory.\n"
        "- Check your to-do list and triggers when waking up on a schedule.\n"
        "- For web lookups, use web_search — you never see raw web content directly.\n"
        "- For specialised how-tos (e.g. how to use the HAL from a sandbox\n"
        "  script), consult load_skill for on-demand guidance rather than\n"
        "  guessing.\n"
        "- Respect privacy: never share one person's information with another\n"
        "  unless it's clearly appropriate (e.g., relaying a message they\n"
        "  asked you to)."
    )


def _prompt_skills_index() -> str:
    """Return the skills index for the static prompt.

    Skills are loaded from a separate filesystem-based loader
    (``boxbot.skills.loader``) which is being built by a parallel
    subagent. We import lazily with a safe fallback so this file does not
    hard-depend on that module being present yet.
    """
    try:
        from boxbot.skills.loader import get_skill_index  # type: ignore
    except ImportError:
        return ""
    except Exception:
        # Defensive: any import-time surprise in the loader shouldn't
        # sink the whole agent.
        logger.debug("Skill loader import raised; falling back to empty index", exc_info=True)
        return ""

    try:
        return get_skill_index() or ""
    except Exception:
        logger.debug("get_skill_index() raised; falling back to empty", exc_info=True)
        return ""


def _render_static_system_prompt(name: str, wake_word: str) -> str:
    """Compose the static (cacheable) portion of the system prompt."""
    parts: list[str] = [
        _prompt_persona(name, wake_word),
        _prompt_etiquette(),
        _prompt_capabilities(),
    ]
    skills = _prompt_skills_index()
    if skills.strip():
        parts.append(skills.strip())
    return "\n\n".join(parts)


# Matches an Anthropic 400 saying a tool_result image exceeded the API
# cap. Captures the message index + tool_result-content index so we can
# scrub just that block.
_OVERSIZE_IMAGE_RE = re.compile(
    r"messages\.(\d+)\.content\.(\d+)\.tool_result\.content\.(\d+)\."
    r"image[^:]*:\s*image exceeds"
)


# Synthetic trigger messages land in the thread via
# `Conversation._format_user_message`, which prefixes them with
# ``"[trigger] "`` — NOT the raw ``"[Trigger fired:"`` text from
# `_on_trigger_fired`. ``_has_human_reply`` must match the wire
# format, not the pre-formatting string. (`_TRIGGER_DESC_RE` still
# works either way because it `.search`es for the inner fragment.)
_TRIGGER_WIRE_PREFIX = "[trigger]"
_TRIGGER_DESC_RE = re.compile(r"\[Trigger fired:\s*(.+?)\]")


def _has_human_reply(messages: list[dict[str, Any]]) -> bool:
    """True if any user turn in the thread is a real human reply
    (not a synthetic trigger fire, not a tool_result returning to
    the model).

    Trigger conversations don't usually accept follow-up — they're
    one-shot per firing per the channel-key contract — but if a
    human ever does land on a trigger thread, we let the full
    extraction path run rather than collapsing them to a stub.

    Synthetic trigger messages are recognised by the ``[trigger]``
    wire prefix that ``Conversation._format_user_message`` stamps
    on them — checked at any position, since a real human message
    never starts that way (human input is either raw text or
    ``[Name]:`` attributed).
    """
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.startswith(
            _TRIGGER_WIRE_PREFIX
        ):
            # synthetic trigger-initiation message
            continue
        if isinstance(content, list):
            # tool_result responses come back as user-role list content;
            # they don't count as human replies.
            if all(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content
            ):
                continue
        # Anything else (string body, list with non-tool_result blocks)
        # is a real inbound message.
        return True
    return False


def _workspace_artifacts_from_thread(
    messages: list[dict[str, Any]],
) -> list[str]:
    """Best-effort: collect workspace paths the run wrote to.

    Scans tool_result blocks for ``execute_script`` ``sdk_actions``
    entries whose action is ``workspace.write`` / ``workspace.append``
    / ``workspace.csv_write`` and pulls their ``path``. Defensive — a
    parse miss just means no pointer in the receipt, never a crash.
    """
    paths: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tr = block.get("content")
            # tool_result content is a JSON string, or a list whose
            # first text block holds the JSON.
            raw: str | None = None
            if isinstance(tr, str):
                raw = tr
            elif isinstance(tr, list):
                for inner in tr:
                    if isinstance(inner, dict) and inner.get("type") == "text":
                        raw = inner.get("text")
                        break
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                continue
            for action in parsed.get("sdk_actions", []) or []:
                if not isinstance(action, dict):
                    continue
                name = action.get("action", "")
                if not name.startswith("workspace."):
                    continue
                if name.split(".", 1)[1] not in (
                    "write", "append", "csv_write", "csv_append"
                ):
                    continue
                path = action.get("path")
                if path and path not in paths:
                    paths.append(path)
    return paths


def _trigger_description_from_thread(messages: list[dict[str, Any]]) -> str:
    """Pull the trigger's description from its opening ``[Trigger fired: …]``
    turn, for framing the bridged context. Falls back to a generic label."""
    if messages:
        first = messages[0]
        if first.get("role") == "user":
            content = first.get("content")
            if isinstance(content, str):
                m = _TRIGGER_DESC_RE.search(content)
                if m:
                    return m.group(1).strip()
    return "scheduled trigger"


def _delivered_text_messages_from_thread(
    messages: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Extract (recipient, content) pairs for every text-channel
    `message` tool call in the thread.

    Used by dispatch-as-bridge: after a trigger conversation finishes,
    each text it delivered is recorded into the recipient's real
    conversation. Voice deliveries are skipped — voice:room is
    transient and a spoken reply already lands there. Entries
    addressed to "current_speaker"/"room"/"unknown" are skipped too:
    a wake-cycle trigger has no current_speaker, so a real delivery
    always names a registered user explicitly.
    """
    from boxbot.core.agent_sdk_adapter import base_tool_name

    out: list[tuple[str, str]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            # The SDK backend records this as mcp__boxbot_tools__message.
            if base_tool_name(str(block.get("name") or "")) != "message":
                continue
            inp = block.get("input") or {}
            if inp.get("channel") != "text":
                continue
            to = str(inp.get("to") or "").strip()
            body = str(inp.get("content") or "").strip()
            if not to or not body:
                continue
            if to in ("current_speaker", "room", "unknown"):
                continue
            out.append((to, body))
    return out


def _summarize_trigger_thread(
    messages: list[dict[str, Any]], started_at: str = "",
) -> str:
    """Build a deterministic RECEIPT line for a routine trigger thread.

    A receipt — not a transcript. It records *that* the trigger ran,
    *what* it was, *when*, and *where its output went* — never the
    content (weather, calendar status, todo counts). The content of a
    trigger run is recoverable elsewhere: dispatched messages land in
    the recipient's real conversation thread (dispatch-as-bridge), and
    deliberately-saved work-products live in the workspace. The
    trigger's internal reasoning is intentionally not retained.

    This receipt goes in the conversations table as a queryable
    journal entry. It is NOT ambient-injected — `inject_memories`
    excludes trigger conversations — so it can't earworm; but
    `search_memory` can still surface it on a deliberate lookup.
    """
    from boxbot.core.agent_sdk_adapter import base_tool_name

    description = "trigger"
    recipients: list[str] = []
    if messages:
        first = messages[0]
        if first.get("role") == "user":
            content = first.get("content")
            if isinstance(content, str):
                m = _TRIGGER_DESC_RE.search(content)
                if m:
                    description = m.group(1).strip()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            if base_tool_name(str(block.get("name") or "")) != "message":
                continue
            to = (block.get("input") or {}).get("to")
            if to and to not in recipients:
                recipients.append(to)

    # Date stamp — "5/14" form, matching how a human refers to it.
    date_str = ""
    if started_at:
        try:
            dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            date_str = f" for {dt.month}/{dt.day}"
        except (ValueError, TypeError):
            date_str = ""

    artifacts = _workspace_artifacts_from_thread(messages)

    if recipients:
        receipt = f"Delivered {description}{date_str} → {', '.join(recipients)}"
    else:
        receipt = f"Ran {description}{date_str} — nothing delivered"
    if artifacts:
        receipt += f" (saved {', '.join(artifacts)})"
    return receipt


def _scrub_oversize_images(
    messages: list[dict[str, Any]], error_message: str
) -> int:
    """Drop oversize image blocks named in a 400 error from the history.

    The Anthropic API surfaces "image exceeds 5 MB maximum" with the
    exact path of the offending block, e.g.
    ``messages.50.content.0.tool_result.content.1.image.source.base64``.
    We pull those indices out and replace the image block with a text
    marker so the retry can succeed without the model losing context.

    Returns the number of blocks scrubbed (0 if the error didn't match
    or the indices were out of range).
    """
    scrubbed = 0
    for match in _OVERSIZE_IMAGE_RE.finditer(error_message):
        try:
            msg_i, content_i, tr_i = (int(g) for g in match.groups())
        except ValueError:
            continue
        if msg_i >= len(messages):
            continue
        msg_content = messages[msg_i].get("content")
        if not isinstance(msg_content, list) or content_i >= len(msg_content):
            continue
        tr_block = msg_content[content_i]
        tr_content = tr_block.get("content") if isinstance(tr_block, dict) else None
        if not isinstance(tr_content, list) or tr_i >= len(tr_content):
            continue
        inner = tr_content[tr_i]
        if not isinstance(inner, dict) or inner.get("type") != "image":
            continue
        tr_content[tr_i] = {
            "type": "text",
            "text": "[image dropped: exceeded 5 MB after encoding]",
        }
        scrubbed += 1
    return scrubbed


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class BoxBotAgent:
    """The central agent that orchestrates boxBot's behaviour.

    Wraps the Anthropic Python SDK, builds system prompts with context
    injection, registers tools, manages conversations, and handles the
    wake/sleep lifecycle.

    The agent is a long-lived object created once at startup. It maintains
    state about who is currently present (from perception events) and holds
    references to the memory store for injection and extraction.

    Attributes:
        memory_store: The shared MemoryStore instance for memory operations.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        *,
        conversation_store: "ConversationStore | None" = None,
    ) -> None:
        """Initialise the agent.

        Args:
            memory_store: The initialised MemoryStore for memory injection,
                extraction, and system memory reading.
            conversation_store: Optional persistent conversation store.
                When provided, WhatsApp conversations route through it
                (durable threads, sweep-based extraction). When None,
                WhatsApp falls back to the legacy in-memory + silence-
                timer behaviour. Voice/trigger never use it.
        """
        self._memory_store = memory_store
        self._conversation_store = conversation_store
        self._client: anthropic.AsyncAnthropic | None = None
        # Background poller for in-flight extraction batches. Started
        # alongside the agent and runs for the agent's lifetime.
        self._batch_poller: BatchPoller | None = None
        # Background poller for in-flight dream-phase batches (PR1:
        # nightly dedup consolidation; applies real merges by default —
        # set ``memory.dream_audit_only=True`` in config for an
        # audit-only soft-launch window).
        self._dream_poller: DreamPoller | None = None
        # Background sweep that closes WhatsApp threads whose rolling
        # window has expired and queues their extraction. Runs only
        # when ``conversation_store`` is wired.
        self._extraction_sweep_task: asyncio.Task[None] | None = None

        # People currently detected by the perception pipeline.
        # Updated via PersonIdentified event subscription.
        self._present_people: dict[str, datetime] = {}

        # Speaker identity mapping (SPEAKER_XX → person name) from
        # perception fusion. Updated via SpeakerIdentified events.
        self._speaker_identities: dict[str, str] = {}

        # Latest per-session identity block from TranscriptReady. Keyed by
        # display label (e.g. "Speaker A" or "Jacob"). Values include
        # voice/visual tier + score + source. Rendered into the dynamic
        # context prompt so the agent can reason about confidence.
        self._latest_speaker_identities: dict[str, dict[str, Any]] = {}

        # Active conversations, keyed by conversation_id. Each
        # Conversation is its own state machine with its own generation
        # task; cross-conversation concurrency is natural.
        self._conversations: dict[str, Conversation] = {}
        # Lookup by channel identity (e.g. "voice:room",
        # "whatsapp:+15551234567") so inbound events route to the
        # right existing conversation, or create one if none exists.
        self._conversation_by_key: dict[str, str] = {}
        # Snapshot of the latest voice_session_id so a TranscriptReady
        # from a fresh voice session ends the prior room conversation
        # (the voice pipeline owns session lifecycle; the agent owns
        # conversation lifecycle — they stay in sync via this field).
        self._current_voice_session_id: str | None = None
        # Index coordination lock. Held briefly during conversation
        # creation/lookup; generation itself runs without this lock.
        self._index_lock = asyncio.Lock()

        self._running = False

    @property
    def memory_store(self) -> MemoryStore:
        """Return the shared MemoryStore."""
        return self._memory_store

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise the agent: create the API client and subscribe to events.

        Must be called after configuration is loaded and the memory store
        is initialised.
        """
        if self._running:
            return

        config = get_config()

        # ANTHROPIC_API_KEY is always required: peripherals (memory
        # rerank Haiku, batch + dream pollers, web_search firewall,
        # photo tagging) bill via the API regardless of which backend
        # runs the main conversation loop. The OAuth-token Agent SDK
        # credit only covers the conversation turn itself.
        if not config.api_keys.anthropic:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. The agent cannot start "
                "without an API key — peripheral Haiku calls (memory "
                "rerank, batch pollers, web_search firewall, photo "
                "tagging) require it regardless of agent.backend."
            )
        if (
            config.agent.backend == "claude_agent_sdk"
            and not config.api_keys.claude_code_oauth_token
        ):
            raise RuntimeError(
                "agent.backend = 'claude_agent_sdk' but "
                "CLAUDE_CODE_OAUTH_TOKEN is not set. Run `claude "
                "setup-token` on a machine with a browser and set the "
                "result in .env, or switch backend back to "
                "'raw_anthropic'."
            )

        self._client = anthropic.AsyncAnthropic(
            api_key=config.api_keys.anthropic,
        )

        # Start the extraction batch poller. It will resume any
        # queued/submitted rows from the previous boot before returning.
        self._batch_poller = BatchPoller(
            self._memory_store, self._client,
        )
        await self._batch_poller.start()

        # Start the dream-phase poller. Apply-mode by default; set
        # ``memory.dream_audit_only=True`` in config to log decisions
        # without merging. Resumes any in-flight dream batches from the
        # previous boot.
        self._dream_poller = DreamPoller(
            self._memory_store,
            self._client,
            audit_only=config.memory.dream_audit_only,
        )
        await self._dream_poller.start()

        # Subscribe to events that initiate or inform conversations.
        # Note: WakeWordHeard is intentionally NOT handled here. The
        # voice pipeline activates the mic on wake word; the agent only
        # starts a conversation once a real transcript arrives. Handling
        # wake word here used to spawn a placeholder conversation that
        # raced with the first real utterance.
        bus = get_event_bus()
        bus.subscribe(WhatsAppMessage, self._on_whatsapp_message)
        bus.subscribe(SignalMessage, self._on_signal_message)
        bus.subscribe(TriggerFired, self._on_trigger_fired)
        bus.subscribe(PersonIdentified, self._on_person_identified)
        bus.subscribe(SpeakerIdentified, self._on_speaker_identified)
        bus.subscribe(TranscriptReady, self._on_transcript_ready)
        bus.subscribe(VoiceSessionEnded, self._on_voice_session_ended)
        bus.subscribe(ConversationEnded, self._on_conversation_ended)
        bus.subscribe(AgentSpeaking, self._on_agent_speaking)
        bus.subscribe(AgentSpeakingDone, self._on_agent_speaking_done)
        bus.subscribe(
            ConversationInterruptRequested,
            self._on_conversation_interrupt_requested,
        )

        # Warm-load any persistent WhatsApp threads still inside their
        # window. Each becomes a live Conversation in LISTENING state,
        # so the next inbound message resumes mid-thread instead of
        # opening a fresh conversation. Best-effort: failure here just
        # means warm-load doesn't happen, the next inbound will still
        # rehydrate via _get_or_create_conversation.
        if self._conversation_store is not None:
            try:
                await self._warm_load_persistent_conversations()
            except Exception:
                logger.exception("Persistent-conversation warm-load failed")

            # Start the extraction sweep. It runs for the agent's
            # lifetime and fires extraction on threads whose window
            # has expired.
            self._extraction_sweep_task = asyncio.create_task(
                self._extraction_sweep_loop(),
                name="whatsapp-extraction-sweep",
            )

        self._running = True
        logger.info("BoxBotAgent started (model: %s)", config.models.large)

    async def stop(self) -> None:
        """Graceful shutdown: unsubscribe from events and cancel active conversations."""
        if not self._running:
            return

        self._running = False

        # Unsubscribe from events
        bus = get_event_bus()
        bus.unsubscribe(WhatsAppMessage, self._on_whatsapp_message)
        bus.unsubscribe(SignalMessage, self._on_signal_message)
        bus.unsubscribe(TriggerFired, self._on_trigger_fired)
        bus.unsubscribe(PersonIdentified, self._on_person_identified)
        bus.unsubscribe(SpeakerIdentified, self._on_speaker_identified)
        bus.unsubscribe(TranscriptReady, self._on_transcript_ready)
        bus.unsubscribe(VoiceSessionEnded, self._on_voice_session_ended)
        bus.unsubscribe(ConversationEnded, self._on_conversation_ended)
        bus.unsubscribe(AgentSpeaking, self._on_agent_speaking)
        bus.unsubscribe(AgentSpeakingDone, self._on_agent_speaking_done)
        bus.unsubscribe(
            ConversationInterruptRequested,
            self._on_conversation_interrupt_requested,
        )

        # End any active conversations cleanly.
        for conv in list(self._conversations.values()):
            try:
                await conv.end(reason="agent_stop")
            except Exception:
                logger.exception(
                    "Error ending conversation %s during stop",
                    conv.conversation_id,
                )
        self._conversations.clear()
        self._conversation_by_key.clear()

        if self._batch_poller is not None:
            await self._batch_poller.stop()
            self._batch_poller = None

        if self._dream_poller is not None:
            await self._dream_poller.stop()
            self._dream_poller = None

        if self._extraction_sweep_task is not None:
            self._extraction_sweep_task.cancel()
            try:
                await self._extraction_sweep_task
            except (asyncio.CancelledError, Exception):
                pass
            self._extraction_sweep_task = None

        self._client = None
        logger.info("BoxBotAgent stopped")

    async def _warm_load_persistent_conversations(self) -> None:
        """Rehydrate persistent text threads (WhatsApp + Signal) still
        inside their window.

        Restores each as a live Conversation in LISTENING state — no
        ConversationStarted event is fired (the conversation already
        existed; the restart was transparent from the user's point of
        view).
        """
        store = self._conversation_store
        if store is None:
            return
        config = get_config()
        window = float(config.whatsapp.thread_window_seconds)
        # No channel filter: pick up any persistent-mode row regardless
        # of whether the user was on WhatsApp or Signal at the time.
        records = await store.list_active(max_inactive_seconds=window)
        if not records:
            return
        for rec in records:
            try:
                thread = await store.get_thread(rec.conversation_id)
                conv = Conversation(
                    conversation_id=rec.conversation_id,
                    channel=rec.channel,
                    channel_key=rec.channel_key,
                    generate_fn=self._generate_for_conversation,
                    participants=set(rec.participants),
                    lifecycle_mode="persistent",
                    store=store,
                    rehydrated_thread=thread,
                )
                self._conversations[rec.conversation_id] = conv
                self._conversation_by_key[rec.channel_key] = (
                    rec.conversation_id
                )
                self._attach_sandbox_runner(conv)
                logger.info(
                    "Warm-loaded %s conversation %s "
                    "(key=%s, turns=%d, last_activity=%s)",
                    rec.channel, rec.conversation_id, rec.channel_key,
                    len(thread), rec.last_activity_at_iso,
                )
            except Exception:
                logger.exception(
                    "Failed to warm-load conversation %s",
                    rec.conversation_id,
                )

    async def _extraction_sweep_loop(self) -> None:
        """Periodically extract WhatsApp threads whose window has expired.

        Runs every ``whatsapp.extraction_sweep_seconds`` (default 5
        min). For each expired row we mark_extracted (atomic), end the
        in-memory Conversation if it's still indexed, and queue
        post-conversation memory extraction the same way the
        synchronous voice/trigger path does.
        """
        store = self._conversation_store
        if store is None:
            return
        config = get_config()
        sweep_interval = max(30.0, float(config.whatsapp.extraction_sweep_seconds))
        window = float(config.whatsapp.thread_window_seconds)
        # Initial sweep right after startup catches anything that went
        # quiet while the agent was down.
        while self._running:
            try:
                await self._run_extraction_sweep(store, window)
            except Exception:
                logger.exception("Extraction sweep iteration failed")
            try:
                await asyncio.sleep(sweep_interval)
            except asyncio.CancelledError:
                return

    async def _run_extraction_sweep(
        self, store: "ConversationStore", window: float,
    ) -> None:
        """One pass of the sweep — separated for testability.

        Scans every persistent-text channel (WhatsApp + Signal). Voice
        and trigger conversations are transient; they extract
        synchronously on ConversationEnded and never appear here.
        """
        expired = await store.list_extractable(max_inactive_seconds=window)
        for rec in expired:
            flipped = await store.mark_extracted(rec.conversation_id)
            if not flipped:
                # Another sweeper raced us, or the row was already
                # extracted out-of-band.
                continue
            # Pull the canonical thread from the store (in-memory may
            # be missing if we restarted between window expiry and
            # warm-load).
            thread = await store.get_thread(rec.conversation_id)
            person_name = next(
                (p for p in rec.participants
                 if p != get_config().agent.name),
                None,
            )

            # End the in-memory Conversation if it's still indexed —
            # publishes ConversationEnded so anything else listening
            # (sandbox runner teardown, etc.) cleans up.
            async with self._index_lock:
                conv = self._conversations.get(rec.conversation_id)
            if conv is not None and not conv.is_ended:
                try:
                    await conv.end(reason="window_expired")
                except Exception:
                    logger.exception(
                        "Failed to end conversation %s on sweep",
                        rec.conversation_id,
                    )

            # Queue extraction. Counts every assistant turn — same
            # contract as the synchronous _on_conversation_ended path.
            turn_count = sum(
                1 for m in thread if m.get("role") == "assistant"
            )
            if thread and turn_count > 0:
                # Persistent (WhatsApp) conversations may be swept long
                # after the live Conversation object was discarded, so
                # we don't have an in-memory injection block to forward
                # here. The batch poller's legacy fallback handles
                # ID-only rendering for these.
                asyncio.create_task(
                    self._post_conversation(
                        conversation_id=rec.conversation_id,
                        channel=rec.channel,
                        person_name=person_name,
                        messages=list(thread),
                        accessed_memory_ids=[],
                        started_at=rec.started_at_iso,
                        injected_memories_block="",
                    ),
                    name=f"extraction-{rec.conversation_id}",
                )
            logger.info(
                "Sweep extracted conversation %s "
                "(key=%s, turns=%d, person=%s)",
                rec.conversation_id, rec.channel_key,
                len(thread), person_name,
            )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_voice_session_ended(self, event: VoiceSessionEnded) -> None:
        """End the room conversation when its voice session ends.

        The voice pipeline drives session lifecycle (ACTIVE → SUSPENDED
        → ENDED after silence). When the session fully ends we end the
        corresponding Conversation so its thread is archived, memory
        extraction fires, and the next wake word starts fresh.
        """
        async with self._index_lock:
            conv_id = self._conversation_by_key.get("voice:room")
            conv = self._conversations.get(conv_id) if conv_id else None
            self._current_voice_session_id = None
        if conv is not None:
            logger.info(
                "Voice session %s ended — ending room conversation %s",
                event.conversation_id, conv.conversation_id,
            )
            await conv.end(reason="voice_session_ended")

    async def _on_whatsapp_message(self, event: WhatsAppMessage) -> None:
        """Route an inbound WhatsApp message to its per-sender conversation."""
        logger.info(
            "WhatsApp message from %s", event.sender_name or event.sender_phone,
        )
        text = event.text or ""

        # For images, download to a sandbox-readable staging path so the
        # agent can view + ingest via bb.photos. Other media types are
        # surfaced as a marker only for now.
        attachment_path: Path | None = None
        if event.media_type == "image" and event.media_url:
            attachment_path = await _stage_whatsapp_image(
                media_id=event.media_url,
                message_id=event.message_id,
            )

        if attachment_path is not None:
            text = f"[image attached at {attachment_path}] {text}".strip()
        elif event.media_type == "image":
            text = f"[image attached, download failed] {text}".strip()
        elif event.media_type:
            text = f"[{event.media_type} attached] {text}".strip()

        channel_key = f"whatsapp:{event.sender_phone or 'unknown'}"
        # Persistent mode (durable thread + sweep extraction) only when
        # the conversation store is wired. Falls back to in-memory +
        # silence-timer behaviour if the store isn't available — keeps
        # tests and dev runs that don't init the store working.
        if self._conversation_store is not None:
            lifecycle_mode = "persistent"
            thread_window: float | None = float(
                get_config().whatsapp.thread_window_seconds
            )
        else:
            lifecycle_mode = "transient"
            thread_window = None
        conv = await self._get_or_create_conversation(
            channel="whatsapp",
            channel_key=channel_key,
            participants={event.sender_name} if event.sender_name else None,
            lifecycle_mode=lifecycle_mode,
            thread_window_seconds=thread_window,
        )
        await conv.handle_input(
            text,
            speaker_name=event.sender_name or None,
            source="user",
            context={
                "sender_phone": event.sender_phone,
                "media_url": event.media_url,
                "media_type": event.media_type,
                "attachment_path": str(attachment_path) if attachment_path else None,
            },
        )

    async def _on_signal_message(self, event: SignalMessage) -> None:
        """Route an inbound Signal message to its per-sender conversation.

        Mirrors :meth:`_on_whatsapp_message` — same persistent thread
        model, same window, just keyed on ``signal:`` and using the
        signal-cli attachment cache for inbound images.
        """
        logger.info(
            "Signal message from %s", event.sender_name or event.sender_phone,
        )
        text = event.text or ""

        attachment_path: Path | None = None
        if event.media_type == "image" and event.media_url:
            attachment_path = await _stage_signal_image(
                attachment_id=event.media_url,
                message_id=event.message_id,
            )

        if attachment_path is not None:
            text = f"[image attached at {attachment_path}] {text}".strip()
        elif event.media_type == "image":
            text = f"[image attached, download failed] {text}".strip()
        elif event.media_type:
            text = f"[{event.media_type} attached] {text}".strip()

        channel_key = f"signal:{event.sender_phone or 'unknown'}"
        if self._conversation_store is not None:
            lifecycle_mode = "persistent"
            # Signal text shares the same human-pacing window as
            # WhatsApp — both are async-by-default text channels.
            thread_window: float | None = float(
                get_config().whatsapp.thread_window_seconds
            )
        else:
            lifecycle_mode = "transient"
            thread_window = None
        conv = await self._get_or_create_conversation(
            channel="signal",
            channel_key=channel_key,
            participants={event.sender_name} if event.sender_name else None,
            lifecycle_mode=lifecycle_mode,
            thread_window_seconds=thread_window,
        )
        await conv.handle_input(
            text,
            speaker_name=event.sender_name or None,
            source="user",
            context={
                "sender_phone": event.sender_phone,
                "media_url": event.media_url,
                "media_type": event.media_type,
                "attachment_path": str(attachment_path) if attachment_path else None,
            },
        )

    async def _on_trigger_fired(self, event: TriggerFired) -> None:
        """Create a one-shot conversation from a scheduler trigger.

        Special-cased: dream-cycle triggers (description marked with
        ``[dream-cycle]``) run the nightly memory consolidation directly
        in the agent process rather than spawning a conversation. The
        dream phase is housekeeping; it has no user to talk to.
        """
        logger.info(
            "Trigger fired: %s (%s)",
            event.trigger_id, event.description,
        )

        # Dream-cycle triggers are intercepted here. We use the
        # description prefix as the marker (rather than adding a
        # ``source="dream-cycle"`` column) so the trigger schema stays
        # untouched. Config-seeded dream triggers always use this
        # exact prefix.
        if event.description.startswith("[dream-cycle]"):
            await self._run_dream_cycle_for_trigger(event)
            return

        initial_msg = (
            f"[Trigger fired: {event.description}]\n"
            f"Instructions: {event.instructions}"
        )
        if event.todo_id:
            initial_msg += f"\nLinked to-do: {event.todo_id}"

        # Triggers get their own unique channel_key so each firing is a
        # fresh conversation — they don't share context across firings.
        channel_key = f"trigger:{event.trigger_id}:{_generate_conversation_id()}"
        conv = await self._get_or_create_conversation(
            channel="trigger",
            channel_key=channel_key,
            participants={event.for_person} if event.for_person else None,
            # Triggers have no follow-up user input so a short timeout
            # keeps them from lingering after the agent's response.
            silence_timeout=10.0,
        )
        await conv.handle_input(
            initial_msg,
            speaker_name=event.for_person,
            source="trigger",
            context={
                "trigger_id": event.trigger_id,
                "trigger_description": event.description,
                "is_recurring": event.is_recurring,
                "person": event.person,
                "todo_id": event.todo_id,
            },
        )

    async def _run_dream_cycle_for_trigger(self, event: TriggerFired) -> None:
        """Execute the nightly dream-phase consolidation directly.

        Called from :meth:`_on_trigger_fired` when a dream-cycle
        trigger fires. Apply-mode by default (set config flag
        ``memory.dream_audit_only=True`` for audit-only). Result is
        written to
        ``data/workspace/notes/system/dream-log/<YYYY-MM-DD>.md``; the
        DreamPoller picks up the batch result later and applies any
        decisions.
        """
        from boxbot.core.config import get_config
        from boxbot.memory.dream import run_dream_cycle

        if self._client is None:
            logger.warning(
                "Dream cycle trigger fired but Anthropic client is not "
                "available; skipping cycle"
            )
            return
        config = get_config()
        try:
            summary = await run_dream_cycle(
                self._memory_store,
                self._client,
                audit_only=config.memory.dream_audit_only,
                max_dedup_pairs=config.memory.dream_max_dedup_pairs,
                near_dup_threshold=config.memory.dream_near_dup_threshold,
            )
            logger.info(
                "Dream cycle complete: %s candidates, %s pairs, batch=%s",
                summary.get("candidate_count"),
                summary.get("near_dup_pairs"),
                summary.get("batch_id"),
            )
        except Exception:
            logger.exception("Dream cycle failed")

        # Identity-cloud hygiene shares the nightly dream window. It's
        # independent of the memory dream (own data, own audit flag), so a
        # failure here must not abort anything above.
        if config.perception.id_reconcile_enabled:
            try:
                from boxbot.perception.reconcile import run_id_reconcile

                judge_on = config.perception.id_reconcile_judge_enabled
                id_report = await run_id_reconcile(
                    audit_only=config.perception.id_reconcile_audit_only,
                    client=self._client if judge_on else None,
                    model=config.models.large if judge_on else "",
                    auto_apply=config.perception.id_reconcile_auto_apply,
                )
                judge = id_report.get("judge")
                logger.info(
                    "ID reconcile complete: %d outlier(s), %d duplicate "
                    "candidate(s), %d mislabel(s), judge=%s (audit_only=%s)",
                    len(id_report.get("outliers", [])),
                    len(id_report.get("duplicate_persons", [])),
                    len(id_report.get("mislabels", [])),
                    "off" if judge is None else f"{judge['calls']} call(s)",
                    id_report.get("audit_only"),
                )
            except Exception:
                logger.exception("ID reconcile failed")

    async def _on_person_identified(self, event: PersonIdentified) -> None:
        """Update the set of currently-present people."""
        if event.person_name:
            self._present_people[event.person_name] = datetime.now()

    async def _on_speaker_identified(self, event: SpeakerIdentified) -> None:
        """Update speaker identity mapping from perception fusion."""
        if event.person_name and event.speaker_label:
            self._speaker_identities[event.speaker_label] = event.person_name
            self._present_people[event.person_name] = datetime.now()

            # Update voice session's identity mapping
            from boxbot.communication.voice import get_voice_session

            session = get_voice_session()
            if session is not None:
                session.update_speaker_identities(
                    {event.speaker_label: event.person_name}
                )

    async def _on_transcript_ready(self, event: TranscriptReady) -> None:
        """Route a voice transcript into the room's voice conversation.

        All voice in this room belongs to one Conversation keyed as
        ``voice:room``. A new voice_session_id signals the prior room
        conversation is done (the voice pipeline's session ended); we
        end it and start a fresh one so the new session doesn't inherit
        stale context.
        """
        transcript = event.transcript.strip()
        if not transcript:
            return

        # Apply speaker identities to transcript tags.
        if self._speaker_identities:
            for label, name in self._speaker_identities.items():
                transcript = transcript.replace(f"[{label}]:", f"[{name}]:")

        if event.speaker_identities:
            self._latest_speaker_identities = dict(event.speaker_identities)

        voice_session_id = event.conversation_id
        logger.info(
            "Transcript ready (voice_session=%s): %s",
            voice_session_id, transcript[:100],
        )

        # If the voice session id changed, end the previous room
        # conversation before creating a new one.
        if (
            self._current_voice_session_id is not None
            and voice_session_id != self._current_voice_session_id
        ):
            logger.info(
                "Voice session changed (%s → %s); ending previous room "
                "conversation",
                self._current_voice_session_id, voice_session_id,
            )
            async with self._index_lock:
                old_conv_id = self._conversation_by_key.pop("voice:room", None)
                old_conv = (
                    self._conversations.get(old_conv_id) if old_conv_id else None
                )
            if old_conv is not None:
                await old_conv.end(reason="voice_session_changed")
        self._current_voice_session_id = voice_session_id

        person_name = self._get_most_recent_person()
        participants: set[str] | None = {person_name} if person_name else None
        conv = await self._get_or_create_conversation(
            channel="voice",
            channel_key="voice:room",
            participants=participants,
        )
        # Bridge the latency tracker: voice stages (STT/diarize/TTS) keyed
        # it on the voice session id; the agent loop runs under the
        # Conversation id. Register the alias so gen_start/api/tools marks
        # land on the live tracker.
        latency.alias(conv.conversation_id, voice_session_id)
        await conv.handle_input(
            transcript,
            speaker_name=person_name,
            source="user",
            context={
                "voice_session_id": voice_session_id,
                "speaker_identities": dict(event.speaker_identities or {}),
            },
        )

    def _get_voice_room_conversation(self) -> Conversation | None:
        """Return the live voice:room conversation, if any."""
        conv_id = self._conversation_by_key.get("voice:room")
        if conv_id is None:
            return None
        conv = self._conversations.get(conv_id)
        if conv is None or conv.is_ended:
            return None
        return conv

    async def _on_agent_speaking(self, event: AgentSpeaking) -> None:
        """Mark the room conversation as SPEAKING when TTS begins.

        Voice ``speak()`` publishes this event before streaming TTS.
        Transitioning to SPEAKING flips ``handle_input`` from
        cancel-and-combine to queue-overheard-utterances mode, which is
        what we want while BB is talking.
        """
        conv = self._get_voice_room_conversation()
        if conv is None:
            return
        # Only THINKING → SPEAKING is a valid forward transition for
        # this signal. If the conversation has already been interrupted
        # or ended, a stale AgentSpeaking from a cancelled task must
        # not flip it back into SPEAKING.
        if conv.state is ConversationState.THINKING:
            conv.set_state(ConversationState.SPEAKING)

    async def _on_agent_speaking_done(self, event: AgentSpeakingDone) -> None:
        """If TTS was interrupted (wake word during SPEAKING), interrupt
        the room conversation: cancel the in-flight generation, fold
        the partial spoken segments into the thread, drop any queued
        utterances, and transition to LISTENING for the next turn.

        Non-interrupted completion is handled by ``_run_generation``'s
        normal drain-and-settle path; nothing to do here.
        """
        if not event.interrupted:
            return
        conv = self._get_voice_room_conversation()
        if conv is None:
            return
        try:
            await conv.interrupt()
        except Exception:
            logger.exception(
                "Conversation interrupt failed for voice:room (conv=%s)",
                conv.conversation_id,
            )

    async def _on_conversation_interrupt_requested(
        self, event: ConversationInterruptRequested,
    ) -> None:
        """Explicit user interrupt — currently fires when the wake word
        is heard during an active conversation.

        Re-saying the wake word means "drop what you're doing and
        listen to me now." Cancels-and-folds the in-flight generation,
        clears queued utterances, transitions to LISTENING. This is
        the explicit carve-out against the ambient inject-don't-
        interrupt model: every other transcript queues; only the wake
        word interrupts.

        ``Conversation.interrupt()`` is idempotent and a no-op when
        nothing is in flight, so it's safe to dispatch unconditionally.
        """
        conv = self._conversations.get(event.conversation_id)
        if conv is None:
            return
        try:
            await conv.interrupt()
        except Exception:
            logger.exception(
                "Conversation interrupt-on-wake-word failed (conv=%s)",
                conv.conversation_id,
            )

    # ------------------------------------------------------------------
    # Conversation index + generation
    # ------------------------------------------------------------------

    async def _get_or_create_conversation(
        self,
        *,
        channel: str,
        channel_key: str,
        participants: set[str] | None = None,
        silence_timeout: float | None = None,
        lifecycle_mode: str = "transient",
        thread_window_seconds: float | None = None,
    ) -> Conversation:
        """Look up an existing Conversation by channel key, or create one.

        This is the only place the conversation index is mutated from
        inbound events. The index itself is a dict of live, non-ended
        conversations — ended conversations are removed via
        ``_on_conversation_ended``.

        For ``lifecycle_mode="persistent"`` (WhatsApp), we additionally
        consult the persistent store: if there's an active row for this
        channel_key still inside ``thread_window_seconds`` we rehydrate
        the thread instead of creating a fresh one.
        """
        async with self._index_lock:
            existing_id = self._conversation_by_key.get(channel_key)
            if existing_id is not None:
                conv = self._conversations.get(existing_id)
                if conv is not None and not conv.is_ended:
                    if participants:
                        conv.participants.update(participants)
                        if conv._store is not None:
                            try:
                                await conv._store.update_participants(
                                    conv.conversation_id, conv.participants,
                                )
                            except Exception:
                                logger.exception(
                                    "Failed to update participants for %s",
                                    conv.conversation_id,
                                )
                    return conv
                # Stale entry — drop it and create fresh.
                self._conversation_by_key.pop(channel_key, None)
                self._conversations.pop(existing_id, None)

            # Persistent mode: try to resume an existing thread from
            # the store before minting a fresh conversation. This is
            # the path that survives restart.
            store = (
                self._conversation_store
                if lifecycle_mode == "persistent" else None
            )
            rehydrated_thread: list[dict[str, Any]] | None = None
            conv_id: str | None = None
            if store is not None and thread_window_seconds is not None:
                try:
                    record = await store.get_active(
                        channel_key,
                        max_inactive_seconds=thread_window_seconds,
                    )
                except Exception:
                    logger.exception(
                        "Failed to query conversation store for %s",
                        channel_key,
                    )
                    record = None
                if record is not None:
                    conv_id = record.conversation_id
                    rehydrated_thread = await store.get_thread(conv_id)
                    merged_participants = set(record.participants) | set(
                        participants or ()
                    )
                    participants = merged_participants
                    if set(record.participants) != merged_participants:
                        try:
                            await store.update_participants(
                                conv_id, merged_participants,
                            )
                        except Exception:
                            logger.exception(
                                "Failed to update store participants for %s",
                                conv_id,
                            )
                    logger.info(
                        "Conversation %s rehydrated (channel=%s, key=%s, "
                        "turns=%d, last_activity=%s)",
                        conv_id, channel, channel_key,
                        len(rehydrated_thread),
                        record.last_activity_at_iso,
                    )

            if conv_id is None:
                conv_id = _generate_conversation_id()
                if store is not None:
                    try:
                        await store.create(
                            channel=channel,
                            channel_key=channel_key,
                            participants=participants,
                            conversation_id=conv_id,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to create persistent conversation %s",
                            conv_id,
                        )
                        # Fall through — in-memory conversation still works,
                        # we just lose persistence for this one.
                        store = None

            conv = Conversation(
                conversation_id=conv_id,
                channel=channel,
                channel_key=channel_key,
                generate_fn=self._generate_for_conversation,
                participants=participants,
                silence_timeout=silence_timeout,
                lifecycle_mode=lifecycle_mode,
                store=store,
                rehydrated_thread=rehydrated_thread,
            )
            self._conversations[conv_id] = conv
            self._conversation_by_key[channel_key] = conv_id
            if rehydrated_thread is None:
                logger.info(
                    "Conversation %s created (channel=%s, key=%s)",
                    conv_id, channel, channel_key,
                )
            # Stub the memory.db conversations row under the live
            # conv_id so memories created mid-conversation can FK
            # against it. Extraction fills in summary/topics later
            # via update_conversation. INSERT OR IGNORE makes this
            # safe on every path: fresh create, rehydrate (the store
            # row may exist without a memory.db row — e.g. when
            # dispatch-as-bridge minted it without a live agent), and
            # repeat-revives. Best-effort — a stub failure just
            # degrades to "memories from this conversation won't
            # FK-resolve until extraction runs," which is the pre-stub
            # behavior.
            try:
                await self._memory_store.create_conversation_stub(
                    conversation_id=conv_id,
                    channel=channel,
                    participants=sorted(participants) if participants else [],
                )
            except Exception:
                logger.exception(
                    "Failed to stub memory conversations row for %s",
                    conv_id,
                )
            # Eager-start the per-conversation sandbox runner so the
            # boot cost (sudo + python + import bb) is hidden behind
            # wake-word activation rather than charged to the first
            # execute_script call. Best-effort: failure here just
            # leaves the runner null, and execute_script falls back to
            # its per-call subprocess path.
            self._attach_sandbox_runner(conv)
            return conv

    def _attach_sandbox_runner(self, conv: Conversation) -> None:
        """Construct + eager-start a SandboxRunner for this conversation."""
        from boxbot.tools.sandbox_runner import SandboxRunner

        try:
            cfg = get_config()
            venv_python = (
                Path(cfg.sandbox.venv_path) / "bin" / "python3"
            )
            timeout = cfg.sandbox.timeout
            sandbox_user = cfg.sandbox.user
        except Exception:
            logger.exception(
                "Sandbox config unavailable — runner not attached"
            )
            return
        enforce = os.environ.get("BOXBOT_SANDBOX_ENFORCE", "1") != "0"
        runner = SandboxRunner(
            venv_python=venv_python,
            sandbox_user=sandbox_user,
            enforce_sandbox=enforce,
            timeout=timeout,
            label=conv.conversation_id,
        )
        conv.sandbox_runner = runner
        # Kick start in the background so create() doesn't block on
        # subprocess spawn / sudo prompt resolution. start() handles
        # its own failures (logs + poisons the runner), but if it ever
        # raises anyway, surface it now instead of waiting for task GC
        # to mutter "exception was never retrieved".
        start_task = asyncio.create_task(
            runner.start(),
            name=f"sandbox-start-{conv.conversation_id}",
        )

        def _log_start_failure(task: "asyncio.Task[None]") -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logger.warning(
                    "Sandbox runner eager-start for %s raised: %r — "
                    "execute_script will fall back to per-call subprocess",
                    conv.conversation_id, exc,
                )

        start_task.add_done_callback(_log_start_failure)

    async def _on_conversation_ended(self, event: ConversationEnded) -> None:
        """Remove the conversation from the index and fire memory extraction.

        The Conversation publishes ``ConversationEnded`` from its own
        ``end()`` method (whether ended by silence timeout, explicit
        ``end()``, or voice-session-ended). We pop it from both indexes
        and kick off post-conversation memory extraction on its thread.
        """
        conv_id = event.conversation_id
        async with self._index_lock:
            conv = self._conversations.pop(conv_id, None)
            # Remove from channel-key index if this id still owned it.
            for key, cid in list(self._conversation_by_key.items()):
                if cid == conv_id:
                    self._conversation_by_key.pop(key, None)
                    break
        if conv is None:
            return

        # Tear down the per-conversation sandbox process. Fire-and-
        # forget; stop() handles its own timeouts and never raises.
        if conv.sandbox_runner is not None:
            asyncio.create_task(
                conv.sandbox_runner.stop(),
                name=f"sandbox-stop-{conv_id}",
            )
            conv.sandbox_runner = None

        # Tear down the per-conversation Claude Agent SDK client if
        # one was attached. disconnect() reaps the underlying ``claude``
        # CLI subprocess so we don't leak it between conversations.
        sdk_client = getattr(conv, "_sdk_client", None)
        if sdk_client is not None:
            async def _stop_sdk_client(client: Any) -> None:
                try:
                    await client.disconnect()
                except Exception:
                    logger.exception(
                        "SDK client disconnect failed (conv=%s)",
                        conv_id,
                    )

            asyncio.create_task(
                _stop_sdk_client(sdk_client),
                name=f"sdk-disconnect-{conv_id}",
            )
            conv._sdk_client = None

        # Fire-and-forget memory extraction on the ended thread.
        # Persistent conversations have extraction routed through the
        # sweep loop (see _run_extraction_sweep) so they don't need
        # the synchronous post-conversation kick here. Without this
        # guard a sweep-driven end() would double-extract.
        if (
            conv.thread
            and event.turn_count > 0
            and conv.lifecycle_mode != "persistent"
        ):
            asyncio.create_task(
                self._post_conversation(
                    conversation_id=conv_id,
                    channel=event.channel,
                    person_name=event.person_name,
                    messages=list(conv.thread),
                    accessed_memory_ids=list(conv.accessed_memory_ids),
                    started_at=conv.started_at_iso(),
                    injected_memories_block=conv.injected_memories_block,
                ),
                name=f"extraction-{conv_id}",
            )

    async def _generate_for_conversation(
        self, conv: Conversation,
    ) -> GenerationResult:
        """Run one agent-loop cycle for a Conversation.

        This is the generate_fn injected into every Conversation. It:
        1. Builds the two-block system prompt from live state.
        2. Runs the Claude agent loop using the Conversation's thread
           as the seed message history.
        3. Dispatches each output to its channel, recording a
           ``SpokenSegment`` on the conversation BEFORE the delivery
           await so interrupt-and-fold sees an accurate partial record.
        4. Returns the thread additions (everything beyond the thread's
           prior length) as a GenerationResult.
        """
        assert self._client is not None, "Agent not started"
        if not conv.thread:
            # Nothing to generate from — should not happen because
            # handle_input always appends before starting the task.
            return GenerationResult(completed_cleanly=False)

        # The last thread entry is the fresh user input that triggered
        # this cycle; everything earlier is prior context.
        last_user = conv.thread[-1]
        initial_message = str(last_user.get("content") or "")
        prior_history = list(conv.thread[:-1]) if len(conv.thread) > 1 else None

        context = conv.current_context
        person_name = self._get_most_recent_person()

        system_prompt_blocks = await self._build_system_prompt_blocks(
            person_name=person_name,
            channel=conv.channel,
            context=context,
            initial_message=initial_message,
            conv=conv,
        )

        # The agent loop returns the full message history — from
        # prior_history + initial user + assistant/tool turns produced
        # this cycle. We'll extract the additions beyond our thread.
        # Dispatch to the configured backend; both paths share the
        # same signature and return shape so callers don't branch.
        conv.set_state(ConversationState.THINKING)
        backend = get_config().agent.backend
        if backend == "claude_agent_sdk":
            messages, turn_count = await self._agent_loop_sdk(
                conv=conv,
                channel=conv.channel,
                system_prompt_blocks=system_prompt_blocks,
                initial_message=initial_message,
                person_name=person_name,
                max_turns=get_config().agent.max_turns,
                prior_history=prior_history,
            )
        else:
            messages, turn_count = await self._agent_loop(
                conversation_id=conv.conversation_id,
                channel=conv.channel,
                system_prompt_blocks=system_prompt_blocks,
                initial_message=initial_message,
                person_name=person_name,
                max_turns=get_config().agent.max_turns,
                prior_history=prior_history,
            )

        additions = messages[len(conv.thread):]
        summary = self._extract_summary(messages)

        # ``completed_cleanly`` is False when the loop ran out of turns
        # — even though we dispatched a graceful close-out, the
        # conversation did not end on its own terms. Memory extraction
        # and any future consumers can decide what to do with that.
        max_turns = get_config().agent.max_turns
        return GenerationResult(
            thread_additions=additions,
            turn_count=turn_count,
            summary=summary,
            completed_cleanly=(turn_count < max_turns),
        )

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    async def _build_system_prompt_blocks(
        self,
        person_name: str | None,
        channel: str,
        context: dict[str, Any] | None,
        initial_message: str,
        conv: Conversation | None = None,
    ) -> list[dict[str, Any]]:
        """Build the two-block system prompt for ``messages.create``.

        Block 1 — static content (persona, etiquette, capabilities, skills
        index) with a 1h ephemeral cache marker. Stable across turns.

        Block 2 — dynamic content (who is present, time, schedule status,
        injected memories, trigger context) WITHOUT a cache marker so it
        can vary without invalidating the static prefix.

        Returns:
            The list of content blocks ready to pass as ``system=`` to the
            Anthropic messages API.
        """
        config = get_config()

        static_text = _render_static_system_prompt(
            name=config.agent.name,
            wake_word=config.agent.wake_word,
        )

        dynamic_text = await self._prompt_dynamic_context(
            person_name=person_name,
            channel=channel,
            context=context,
            initial_message=initial_message,
            conv=conv,
        )

        blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": static_text,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
            {
                "type": "text",
                "text": dynamic_text,
                # No cache_control — this varies per conversation.
            },
        ]
        return blocks

    async def _prompt_dynamic_context(
        self,
        person_name: str | None,
        channel: str,
        context: dict[str, Any] | None,
        initial_message: str,
        conv: Conversation | None = None,
    ) -> str:
        """Build the dynamic (non-cached) portion of the system prompt.

        Includes:
        - System memory (always-loaded household facts) — strictly speaking
          slow-moving, but updated post-conversation, so lives here.
        - Current time / day / channel.
        - Who is present (from perception).
        - Scheduler status line (todo/trigger counts).
        - Trigger context (if this is a trigger-initiated conversation).
        - Injected fact memories for this speaker + initial utterance.
        """
        sections: list[str] = []

        # System memory
        system_memory = await self._read_system_memory()
        if system_memory.strip():
            sections.append(f"## System Memory\n{system_memory}")

        # Current context lines
        now = datetime.now()
        context_lines = [
            f"Current time: {now.strftime('%H:%M')}",
            f"Day: {now.strftime('%A, %B %d, %Y')}",
            f"Channel: {_agent_facing_channel(channel)}",
        ]
        if person_name:
            context_lines.append(f"Speaking with: {person_name}")
        present = self._get_present_people_with_status(exclude=person_name)
        if present:
            context_lines.append(f"Also present: {', '.join(present)}")
        display_line = self._format_active_display_line()
        if display_line:
            context_lines.append(display_line)
        sections.append(
            "## Current Context\n"
            + "\n".join(f"- {line}" for line in context_lines)
        )

        # Per-session identity block (voice + visual ReID tiers). Lets the
        # agent decide when to address by name (high), verify (medium/low),
        # or load the onboarding skill (unknown).
        identity_section = _render_identity_section(
            self._latest_speaker_identities
        )
        if identity_section:
            sections.append(identity_section)

        # Auto-load the onboarding skill body when any speaker in this
        # turn's identity block is unknown. Saves the round-trip the
        # agent would otherwise spend calling load_skill(onboarding) on
        # an unknown speaker's first utterance — that round-trip was
        # measured at ~8s on cold sandbox + 1st-turn cache (Carina,
        # 2026-06-05). Only fires when there's actually an unknown
        # speaker AND the channel is voice; whatsapp/trigger don't have
        # a notion of in-room identity to onboard.
        if (
            channel == "voice"
            and self._latest_speaker_identities
            and any(
                (info or {}).get("voice_tier") == "unknown"
                for info in self._latest_speaker_identities.values()
            )
        ):
            try:
                from boxbot.skills.loader import load_skill as _load_skill
                skill_body = _load_skill(name="onboarding")
            except Exception:
                logger.debug(
                    "auto-load onboarding skill failed; skipping",
                    exc_info=True,
                )
            else:
                sections.append(
                    "## Onboarding skill (auto-loaded — unknown speaker present)\n"
                    "An unknown speaker is in this conversation; the full\n"
                    "onboarding skill is included below so you don't need\n"
                    "to call `load_skill(\"onboarding\")` first. Apply §1\n"
                    "(voice first-meeting) if they address you directly.\n\n"
                    + skill_body
                )

        # Registered users the agent can address in `outputs[].to`. Names
        # (not phone numbers) are what the agent sees; the dispatcher
        # resolves names to numbers via AuthManager at delivery time.
        try:
            from boxbot.communication.auth import get_auth_manager
            auth = get_auth_manager()
            if auth is not None:
                users = await auth.list_users()
                if users:
                    user_lines = [
                        f"- {u.name}"
                        + (" (admin)" if u.role == "admin" else "")
                        for u in users
                    ]
                    sections.append(
                        "## Registered users\n"
                        "You can reach any of these people via `message` "
                        "(speak if they're at the box, text to reach them on "
                        "their phone).\n"
                        + "\n".join(user_lines)
                    )
                else:
                    sections.append(
                        "## Registered users\n"
                        "No users are registered yet. You cannot deliver "
                        "`channel: \"text\"` outputs until someone registers. "
                        "If `setup:` todos are present, run them by loading "
                        "the `onboarding` skill — it covers first-admin "
                        "bootstrap end-to-end."
                    )
        except Exception:
            logger.debug("Could not list registered users for prompt", exc_info=True)

        # Scheduler status
        try:
            status_line = await get_status_line()
            if status_line and status_line.strip():
                sections.append(status_line.strip())
        except Exception:
            logger.debug("Could not fetch scheduler status line")

        # Inbound image hint: when the inbound handler stages a photo, the
        # user message starts with "[image attached at <path>]". Tell the
        # agent how to act on it. Fires for any messaging channel that
        # stages images (WhatsApp + Signal), and only when this turn's
        # context actually carries a staged path so we don't waste prompt
        # bytes on plain text turns.
        if (
            channel in ("whatsapp", "signal")
            and context
            and context.get("attachment_path")
        ):
            sections.append(
                "## Inbound image\n"
                "The user's message starts with "
                "`[image attached at <path>]`. To see it, call "
                "`bb.photos.view_path(path)` from `execute_script` — the "
                "pixels attach to the tool result. If the photo is worth "
                "keeping (family moment, something the user asked you to "
                "remember, anything you'd want to surface later), save it "
                f"with `bb.photos.ingest(path, source=\"{channel}\", "
                "sender=<name>, caption=<their text>)`. Do NOT ingest "
                "memes, throwaway shares, or anything ephemeral — view, "
                "respond, and let the janitor reap it."
            )

        # Trigger details (trigger-initiated conversations)
        if context and channel == "trigger":
            trigger_lines = []
            if context.get("trigger_description"):
                trigger_lines.append(
                    f"Trigger: {context['trigger_description']}"
                )
            if context.get("is_recurring"):
                trigger_lines.append("(This is a recurring trigger)")
            if context.get("todo_id"):
                trigger_lines.append(
                    f"Linked to-do item: {context['todo_id']}"
                )
            if trigger_lines:
                sections.append(
                    "## Trigger Details\n"
                    + "\n".join(f"- {line}" for line in trigger_lines)
                )

        # Injected memories
        memory_block, surfaced_ids = await self._inject_memories(
            person_name=person_name,
            initial_message=initial_message,
        )
        if memory_block and memory_block.strip():
            sections.append(memory_block.strip())
        # Record surfaced memory IDs on the conversation so post-
        # conversation extraction knows which memories the model saw.
        # Dedupe across turns — the same memory can be re-surfaced.
        if conv is not None and surfaced_ids:
            seen = set(conv.accessed_memory_ids)
            for mid in surfaced_ids:
                if mid not in seen:
                    conv.accessed_memory_ids.append(mid)
                    seen.add(mid)
            # Stash the rendered block so post-conversation extraction
            # can apply invalidation rules against real summaries
            # (not just IDs). Multi-turn conversations overwrite each
            # other; we keep the latest because injection refreshes
            # the candidate set as the conversation evolves.
            if memory_block:
                conv.injected_memories_block = memory_block

        return "\n\n".join(sections)

    async def _read_system_memory(self) -> str:
        """Read the current system memory file.

        Returns the content of data/memory/system.md, which contains
        always-loaded household facts and standing instructions.

        Returns:
            The system memory text, or empty string if unavailable.
        """
        try:
            return await self._memory_store.read_system_memory()
        except Exception:
            logger.debug("Could not read system memory")
            return ""

    async def _inject_memories(
        self,
        person_name: str | None,
        initial_message: str,
    ) -> tuple[str, list[str]]:
        """Search for relevant memories and format them for prompt injection.

        Uses the shared search backend to find fact memories and recent
        conversations relevant to the current speaker and their first
        utterance.

        Returns:
            ``(block, memory_ids)``. The block is the formatted text
            for the system prompt; memory_ids identifies which records
            the model could see, so post-conversation extraction can
            consider them for invalidation.
        """
        try:
            return await inject_memories(
                self._memory_store,
                person=person_name,
                utterance=initial_message,
            )
        except Exception:
            logger.exception("Memory injection failed")
            return "", []

    # ------------------------------------------------------------------
    # Agent loop (Anthropic SDK)
    # ------------------------------------------------------------------

    async def _agent_loop(
        self,
        conversation_id: str,
        channel: str,
        system_prompt_blocks: list[dict[str, Any]],
        initial_message: str,
        person_name: str | None,
        max_turns: int = _DEFAULT_MAX_TURNS,
        prior_history: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Run the core agent conversation loop.

        Each ``messages.create`` call carries:
        - ``output_config.format`` pinning the ``INTERNAL_NOTES_SCHEMA``
          (defined in ``output_dispatcher``) — the agent's text output is a
          private scratchpad of ``thought`` + ``observations``. By design
          no field reaches a person; deliveries go through the
          ``message`` tool.
        - top-level ``cache_control`` for the 5-minute rolling messages cache
        - a ``tools`` list where the LAST tool holds a 1h cache breakpoint
        - a two-block ``system`` with a 1h breakpoint on the static block

        Text blocks are parsed as INTERNAL_NOTES_SCHEMA JSON for logging
        and memory extraction. They never trigger a delivery.
        ``message`` tool calls (run by ``_process_tool_calls``)
        are how the agent reaches a person.

        Turn cap (``max_turns``): if the model would otherwise loop
        past the cap, the **penultimate** iteration appends a
        ``[system]`` note to its tool-result user block telling the
        model the next response is final and only ``message`` is
        available. The **final** iteration is called with ``tools``
        filtered to just the ``message`` definition — the API
        forecloses every other tool call. After that turn, the loop
        exits; if no ``message`` was dispatched, a post-loop fallback
        sends a hardcoded close-out so the user is never left hanging.
        ``GenerationResult.completed_cleanly`` is set to ``False`` when
        the cap was hit.

        Args:
            conversation_id: Conversation ID for logging.
            channel: Active conversation channel (voice / whatsapp / trigger),
                used for log provenance only — the agent chooses its own
                delivery channel per ``message`` call.
            system_prompt_blocks: Two-block system prompt (static + dynamic).
            initial_message: The first user message.
            person_name: The speaker currently addressing the agent. The
                Conversation provides this to ``message`` via
                participants when the tool resolves ``current_speaker``.
            max_turns: Maximum number of API round-trips.

        Returns:
            Tuple of (message_history, turn_count).
        """
        assert self._client is not None, "Agent not started"

        config = get_config()
        model = config.models.large

        from boxbot.tools.registry import get_tools

        # Build tool definitions for the API (last tool carries 1h cache marker)
        tools = get_tools()
        tool_definitions = self._build_tool_definitions(tools)

        # Initialise the message history. For voice continuity we seed with
        # the accumulated history from prior utterances in this voice session
        # (passed in by ``_run_conversation`` from the voice-history slot).
        messages: list[dict[str, Any]] = []
        if prior_history:
            messages.extend(prior_history)
            logger.info(
                "Agent loop seeded with %d prior messages (conv=%s)",
                len(prior_history), conversation_id,
            )
        messages.append({"role": "user", "content": initial_message})

        turn_count = 0

        # Tracks whether the model emitted a ``message`` tool call on the
        # final allowed turn. If False after the loop ends because we hit
        # the cap, the post-loop fallback dispatches a hardcoded closing
        # line so the user is never left hanging.
        final_turn_message_dispatched = False

        while turn_count < max_turns:
            turn_count += 1

            # On the final allowed turn, restrict tools to ``message``
            # only. The agent gets one round to address the user; all
            # other tools are foreclosed at the API level so the model
            # cannot defy the cap. See _agent_loop docstring §"Turn cap".
            is_final_turn = turn_count == max_turns
            turn_tools = (
                [t for t in tool_definitions
                 if t.get("name") == "message"]
                if is_final_turn else tool_definitions
            )

            # IMPORTANT: the exact named kwargs below are the only ones
            # we pass. No temperature / top_p / top_k / thinking — those
            # will 400 on Opus 4.7. See spec §3.
            # Latency: stamp the agent-generation boundary once (voice
            # round-trip breakdown) and accumulate API wall time across
            # turns. No-op for non-voice conversations.
            if turn_count == 1:
                latency.mark(conversation_id, "gen_start")

            response = None
            for attempt in range(2):
                try:
                    with latency.span(conversation_id, "api"):
                        response = await self._client.messages.create(
                            model=model,
                            max_tokens=_MAX_TOKENS,
                            system=system_prompt_blocks,
                            messages=messages,
                            tools=turn_tools,
                            output_config={
                                "format": {
                                    "type": "json_schema",
                                    "schema": INTERNAL_NOTES_SCHEMA,
                                }
                            },
                            cache_control={"type": "ephemeral"},
                        )
                    break
                except anthropic.APIError as e:
                    logger.error(
                        "Anthropic API error on turn %d (attempt %d/2): %s",
                        turn_count, attempt + 1, e,
                    )
                    if attempt == 0:
                        # If the failure is "image too large", surgically
                        # drop the offending image block(s) from the
                        # message history before retrying so we don't
                        # just hit the same 400 again. The error message
                        # looks like:
                        #   messages.<i>.content.<j>.tool_result.content.<k>.image…: image exceeds 5 MB maximum
                        scrubbed = _scrub_oversize_images(messages, str(e))
                        if scrubbed:
                            logger.warning(
                                "Scrubbed %d oversize image block(s) "
                                "from turn %d history before retry",
                                scrubbed, turn_count,
                            )
                        await asyncio.sleep(3)
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": f"(API error: {e})",
                        })
            if response is None:
                break

            # Append the assistant's response to the history
            assistant_content = self._response_to_content_blocks(response)
            messages.append({
                "role": "assistant",
                "content": assistant_content,
            })

            # Cost tracking: one row per Claude turn (raw Anthropic API).
            # This is the single hook for conversation spend — keep it here
            # so retries / refusals / max_tokens all bill correctly.
            try:
                event = from_anthropic_usage(
                    purpose="conversation",
                    model=getattr(response, "model", model) or model,
                    usage=getattr(response, "usage", None),
                    correlation_id=conversation_id,
                    metadata={
                        "channel": channel,
                        "turn": turn_count,
                    },
                )
                await record_cost(self._memory_store, event)
            except Exception:
                logger.exception(
                    "Failed to record conversation cost (conv=%s turn=%d)",
                    conversation_id, turn_count,
                )

            stop_reason = getattr(response, "stop_reason", None)

            # Parse EVERY text block in the response as INTERNAL_NOTES_SCHEMA
            # JSON for logging and memory extraction. Text blocks are PRIVATE
            # by design — they never trigger deliveries. The agent reaches
            # people only via message tool calls (handled below).
            for block in getattr(response, "content", []) or []:
                if getattr(block, "type", None) != "text":
                    continue
                raw = getattr(block, "text", "") or ""
                parsed = parse_internal_notes(raw)
                if parsed is None:
                    # Parse failed under constrained decoding — log; the rest
                    # of the turn still progresses (tools still run if present).
                    if raw.strip():
                        logger.error(
                            "Could not parse internal notes JSON (conv=%s "
                            "turn=%d). First 200 chars: %r",
                            conversation_id, turn_count, raw[:200],
                        )
                    continue
                if parsed.thought:
                    logger.info(
                        "agent thought (conv=%s turn=%d): %s",
                        conversation_id, turn_count, parsed.thought,
                    )
                if parsed.observations:
                    logger.info(
                        "agent observations (conv=%s turn=%d): %s",
                        conversation_id, turn_count,
                        " | ".join(parsed.observations),
                    )

            # --- tool_use: outputs have already been dispatched (if any);
            # now run the tools and feed results back.
            if stop_reason == "tool_use":
                with latency.span(conversation_id, "tools"):
                    tool_results = await self._process_tool_calls(
                        response, tools, conversation_id=conversation_id,
                    )
                # Inject-don't-interrupt: drain any utterances that
                # arrived during this iteration's API call + tool
                # dispatch and fold them into the same role:"user" turn
                # as the tool_result blocks. The model sees both on the
                # next API call and can react in one round-trip. This
                # is the Claude Code / Agent SDK pattern — see
                # Conversation.handle_input THINKING branch.
                content_blocks: list[dict[str, Any]] = list(tool_results)
                conv = self._conversations.get(conversation_id) \
                    if conversation_id else None
                if conv is not None:
                    for item in conv.drain_pending_inputs():
                        text = str(item.get("content") or "").strip()
                        if not text:
                            continue
                        content_blocks.append({"type": "text", "text": text})

                # Final turn: the model just used its only remaining
                # tool (must be ``message`` — see is_final_turn filter
                # above). Record whether a ``message`` actually went out
                # so the post-loop fallback knows whether the user heard
                # anything, then exit without queuing another API call.
                if is_final_turn:
                    for block in response.content or []:
                        if (getattr(block, "type", None) == "tool_use"
                                and getattr(block, "name", None) == "message"):
                            final_turn_message_dispatched = True
                            break
                    break

                # Penultimate iteration: prime the model for its last
                # turn. The text rides the existing user-side content
                # block so it lands in the same place as drained inputs
                # — the inject-don't-interrupt seam already in place.
                if turn_count + 1 == max_turns:
                    content_blocks.append({
                        "type": "text",
                        "text": (
                            "[system] You have reached the conversation "
                            f"turn cap ({max_turns} turns). Your next "
                            "response is your last, and the only tool "
                            "available will be ``message`` — every "
                            "other tool is disabled. Send one closing "
                            "message to the user (via ``message``) "
                            "summarizing what you accomplished, what "
                            "you tried, and where you got stuck. Be "
                            "honest about uncertainty (e.g. \"I tried "
                            "X but couldn't verify it worked\"). Do "
                            "not attempt further work — anything other "
                            "than ``message`` will be blocked."
                        ),
                    })

                if content_blocks:
                    messages.append({
                        "role": "user",
                        "content": content_blocks,
                    })
                continue

            # --- refusal: log and stop. No canned voice output — the refusal
            # text may not be schema-conformant, so we don't speak it. Future
            # work can dispatch a canned apology via the outputs path if the
            # refusal rate becomes a UX problem.
            if stop_reason == "refusal":
                logger.warning(
                    "Model returned refusal on turn %d (conv=%s)",
                    turn_count, conversation_id,
                )
                break

            # --- max_tokens: truncated; outputs (if any) were dispatched above.
            if stop_reason == "max_tokens":
                logger.error(
                    "Model hit max_tokens on turn %d (conv=%s) — truncated",
                    turn_count, conversation_id,
                )
                break

            # --- end_turn (and any other terminal reason): we're done.
            break

        if turn_count >= max_turns:
            logger.warning(
                "Conversation reached max turns (%d, message_dispatched=%s)",
                max_turns, final_turn_message_dispatched,
            )
            if not final_turn_message_dispatched:
                await self._dispatch_max_turns_fallback(
                    conversation_id=conversation_id,
                    channel=channel,
                    person_name=person_name,
                    max_turns=max_turns,
                )

        return messages, turn_count

    async def _agent_loop_sdk(
        self,
        conv: Any,
        channel: str,
        system_prompt_blocks: list[dict[str, Any]],
        initial_message: str,
        person_name: str | None,
        max_turns: int = _DEFAULT_MAX_TURNS,
        prior_history: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Run the conversation through the Claude Agent SDK backend.

        Mirrors :meth:`_agent_loop` in signature and return shape so
        callers don't branch on the backend. The SDK takes ownership of
        the multi-turn tool loop; we observe the stream for logging,
        memory hooks, output dispatch tracking, and cost telemetry, then
        translate the SDK's view of the conversation back into the
        ``messages`` history shape ``_run_conversation`` expects.

        Lifecycle: one :class:`ClaudeSDKClient` per Conversation, cached
        on ``conv._sdk_client``. The first turn of a Conversation
        constructs the client and connects; subsequent turns reuse it,
        so the SDK's internal session state carries multi-turn voice
        continuity without us re-seeding ``prior_history``.

        Auth: relies on ``CLAUDE_CODE_OAUTH_TOKEN`` in the process
        environment (validated by :meth:`start`). The SDK reads the
        token from env via its own precedence rules.

        Cost: a single :class:`ResultMessage` arrives at the end of the
        loop; we run it through :func:`from_agent_sdk_result` and
        append one cost event (or one per model, in the multi-model
        case). ``num_turns`` from the ResultMessage becomes our
        ``turn_count`` return value.

        Cancellation / interruption: external callers invoke
        ``conv._sdk_client.interrupt()`` when new user input arrives
        mid-stream. The SDK halts generation cleanly; this method's
        ``receive_response()`` exits its loop on the resulting
        ResultMessage with ``stop_reason == "interrupted"``.

        Turn cap behavior: ``max_turns`` is passed straight through to
        ``ClaudeAgentOptions``. The SDK enforces the cap. If the agent
        finishes without invoking the ``message`` tool, the
        post-loop :meth:`_dispatch_max_turns_fallback` (shared with the
        raw backend) sends a hardcoded closing line.
        """
        from boxbot.core.agent_sdk_adapter import (
            build_options,
            flatten_system_prompt,
            mcp_tool_name,
        )
        from boxbot.tools.registry import get_tools

        config = get_config()
        model = config.models.large

        tools = get_tools()
        system_prompt = flatten_system_prompt(system_prompt_blocks)
        message_tool_full_name = mcp_tool_name("message")

        # Build the messages history the SDK path will return. The SDK
        # owns the actual session state; we track messages here only so
        # the surrounding code (memory extraction, summary) sees the
        # same return shape as the raw path produces.
        messages: list[dict[str, Any]] = list(prior_history or [])
        messages.append({"role": "user", "content": initial_message})

        # Bookkeeping for the post-loop fallback dispatch.
        message_dispatched = False
        turn_count = 0

        # Construct / reuse the SDK client. One per Conversation;
        # multi-turn voice continuity comes for free.
        sdk_client = getattr(conv, "_sdk_client", None)
        if sdk_client is None:
            options = build_options(
                model=model,
                max_turns=max_turns,
                system_prompt=system_prompt,
                tools=tools,
                output_format={
                    "type": "json_schema",
                    "schema": INTERNAL_NOTES_SCHEMA,
                },
                conv=conv,
            )
            from claude_agent_sdk import ClaudeSDKClient
            sdk_client = ClaudeSDKClient(options=options)
            await sdk_client.connect()
            conv._sdk_client = sdk_client

        latency.mark(conv.conversation_id, "gen_start")

        try:
            with latency.span(conv.conversation_id, "api"):
                await sdk_client.query(initial_message)

                from claude_agent_sdk import (
                    AssistantMessage,
                    ResultMessage,
                    TextBlock,
                    ToolUseBlock,
                )

                async for sdk_msg in sdk_client.receive_response():
                    if isinstance(sdk_msg, AssistantMessage):
                        assistant_content: list[dict[str, Any]] = []
                        for block in sdk_msg.content:
                            if isinstance(block, TextBlock):
                                assistant_content.append({
                                    "type": "text",
                                    "text": block.text,
                                })
                                parsed = parse_internal_notes(block.text)
                                if parsed is None:
                                    if block.text.strip():
                                        logger.error(
                                            "Could not parse internal "
                                            "notes JSON (conv=%s). "
                                            "First 200 chars: %r",
                                            conv.conversation_id,
                                            block.text[:200],
                                        )
                                    continue
                                if parsed.thought:
                                    logger.info(
                                        "agent thought (conv=%s): %s",
                                        conv.conversation_id,
                                        parsed.thought,
                                    )
                                if parsed.observations:
                                    logger.info(
                                        "agent observations (conv=%s): %s",
                                        conv.conversation_id,
                                        " | ".join(parsed.observations),
                                    )
                            elif isinstance(block, ToolUseBlock):
                                # Track whether the model dispatched a
                                # message tool at any point. The SDK
                                # runs the tool itself via our MCP
                                # server, so the actual delivery has
                                # already happened by the time we see
                                # this block.
                                assistant_content.append({
                                    "type": "tool_use",
                                    "id": block.id,
                                    "name": block.name,
                                    "input": block.input,
                                })
                                if block.name == message_tool_full_name:
                                    message_dispatched = True
                        if assistant_content:
                            messages.append({
                                "role": "assistant",
                                "content": assistant_content,
                            })
                    elif isinstance(sdk_msg, ResultMessage):
                        turn_count = sdk_msg.num_turns or 0
                        # The whole receive_response() loop runs inside
                        # one latency.span("api") above, which records
                        # one wall-clock duration with count=1. The SDK
                        # actually made `num_turns` API round-trips
                        # inside it — surface that in the headline so
                        # `api=<ms>/<N>calls` reflects reality and a
                        # multi-tool turn isn't disguised as a single
                        # slow call.
                        if turn_count > 0:
                            latency.set_count(
                                conv.conversation_id, "api", turn_count,
                            )
                        try:
                            events = from_agent_sdk_result(
                                purpose="conversation",
                                result_message=sdk_msg,
                                correlation_id=conv.conversation_id,
                                metadata={
                                    "channel": channel,
                                    "turn": turn_count,
                                    "backend": "claude_agent_sdk",
                                },
                            )
                            for event in events:
                                await record_cost(self._memory_store, event)
                        except Exception:
                            logger.exception(
                                "Failed to record SDK-loop cost "
                                "(conv=%s turns=%d)",
                                conv.conversation_id,
                                turn_count,
                            )
                        if sdk_msg.is_error:
                            logger.error(
                                "SDK loop ended with error "
                                "(conv=%s stop_reason=%s)",
                                conv.conversation_id,
                                sdk_msg.stop_reason,
                            )
                        break
        except asyncio.CancelledError:
            # New user input (wake word, barge-in) cancelled this
            # generation. Tell the SDK to halt cleanly so the next
            # query() starts from a settled state, then re-raise so the
            # caller's cancel-and-fold flow proceeds normally.
            try:
                await asyncio.shield(sdk_client.interrupt())
            except Exception:
                logger.exception(
                    "SDK client interrupt() failed during cancellation "
                    "(conv=%s)",
                    conv.conversation_id,
                )
            raise
        except Exception:
            logger.exception(
                "SDK-backed agent loop raised (conv=%s)",
                conv.conversation_id,
            )

        # Post-loop fallback: if the SDK hit the turn cap without
        # the agent ever dispatching a message tool, send the
        # hardcoded closing line so the user isn't left hanging.
        if turn_count >= max_turns and not message_dispatched:
            logger.warning(
                "SDK conversation reached max turns (%d) without "
                "dispatching a message — falling back",
                max_turns,
            )
            await self._dispatch_max_turns_fallback(
                conversation_id=conv.conversation_id,
                channel=channel,
                person_name=person_name,
                max_turns=max_turns,
            )

        return messages, turn_count

    async def _dispatch_max_turns_fallback(
        self,
        *,
        conversation_id: str,
        channel: str,
        person_name: str | None,
        max_turns: int,
    ) -> None:
        """Send a hardcoded closing line when the loop hit its cap silently.

        The agent loop tries to coax a graceful close-out via the
        penultimate-turn heads-up + final-turn ``message``-only filter.
        If that still fails to dispatch a ``message`` (e.g. the model
        produces text only, refuses, or errors), we owe the user some
        acknowledgement rather than radio silence. This bypasses the
        ``message`` tool and calls ``dispatch_outputs`` directly.
        """
        from boxbot.core.output_dispatcher import dispatch_outputs

        conv = self._conversations.get(conversation_id) \
            if conversation_id else None

        # Choose the dispatcher channel:
        # - voice/trigger conversations → "voice" (speak it in the room)
        # - text platforms (whatsapp/signal) → "text"
        # Trigger conversations may have no one in the room; speaking
        # there is still the right call because that's where any user
        # presence would be.
        out_channel = "voice" if channel in ("voice", "trigger") else "text"

        # Recipient: the person we're addressing, or "current_speaker"
        # so the dispatcher resolves it from the conversation's
        # participants. WhatsApp needs an explicit phone number.
        if out_channel == "text":
            to = person_name or "current_speaker"
        else:
            to = "current_speaker"

        content = (
            f"I hit my turn limit ({max_turns}) while working on this "
            "and couldn't get to a clean summary. Let me know if you "
            "want me to try again or take a different approach."
        )

        try:
            await dispatch_outputs(
                [{"to": to, "channel": out_channel, "content": content}],
                conversation_id=conversation_id,
                channel_context=channel,
                current_speaker=person_name,
                segment_recorder=conv.record_segment if conv else None,
            )
        except Exception:
            logger.exception(
                "Max-turns fallback dispatch failed (conv=%s)",
                conversation_id,
            )

    # ------------------------------------------------------------------
    # Tool handling
    # ------------------------------------------------------------------

    def _build_tool_definitions(
        self,
        tools: list[Any],
    ) -> list[dict[str, Any]]:
        """Convert boxBot Tool instances to Anthropic API tool definitions.

        Attaches a 1h ephemeral ``cache_control`` marker to the LAST tool
        definition so the entire tools array (+ anything earlier in the
        render order) caches together. See spec §5.

        Args:
            tools: List of Tool instances from the registry.

        Returns:
            List of tool definition dicts for the API.
        """
        definitions: list[dict[str, Any]] = []
        for tool in tools:
            definitions.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        if definitions:
            definitions[-1]["cache_control"] = {
                "type": "ephemeral",
                "ttl": "1h",
            }
        return definitions

    async def _process_tool_calls(
        self,
        response: Any,
        tools: list[Any],
        *,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Dispatch tool calls from a model response.

        Iterates through all tool_use content blocks in the response,
        looks up the corresponding Tool from the registry, calls its
        execute() method, and collects the results. Tool results support
        either ``str`` (legacy) or ``list[content-block]`` (future image
        attachments — see spec §10) as their content.

        Sets the ``current_conversation`` ContextVar around each tool's
        ``execute()`` call so tools that need conversation-scoped state
        (e.g. ``execute_script`` reaching the conversation's long-lived
        sandbox runner) can find it.

        Args:
            response: The Anthropic API response containing tool_use blocks.
            tools: The list of available Tool instances (for reference).
            conversation_id: ID of the conversation that triggered these
                tool calls; used to resolve the Conversation for the
                ContextVar. None disables conversation-scoped routing
                (tools fall back to per-call behavior).

        Returns:
            List of tool_result content blocks to send back to the model.
        """
        from boxbot.tools.registry import get_tool
        from boxbot.tools._tool_context import current_conversation

        conv = (
            self._conversations.get(conversation_id)
            if conversation_id else None
        )

        tool_results: list[dict[str, Any]] = []

        for content_block in response.content:
            if content_block.type != "tool_use":
                continue

            tool_name = content_block.name
            tool_input = content_block.input
            tool_use_id = content_block.id

            logger.debug(
                "Tool call: %s(%s)", tool_name, json.dumps(tool_input)[:200]
            )

            tool = get_tool(tool_name)
            if tool is None:
                result_content: Any = json.dumps({
                    "error": f"Unknown tool: {tool_name}",
                })
                logger.warning("Unknown tool requested: %s", tool_name)
            else:
                token = current_conversation.set(conv)
                try:
                    result_content = await tool.execute(**tool_input)
                except Exception as e:
                    logger.exception(
                        "Tool %s execution failed", tool_name
                    )
                    result_content = json.dumps({
                        "error": f"Tool execution failed: {e}",
                    })
                finally:
                    current_conversation.reset(token)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result_content,
            })

        return tool_results

    @staticmethod
    def _response_to_content_blocks(
        response: Any,
    ) -> list[dict[str, Any]]:
        """Convert an Anthropic API response to serialisable content blocks.

        The messages API returns typed content block objects. We convert
        them to plain dicts for storage in the message history.

        Args:
            response: The Anthropic API response.

        Returns:
            List of content block dicts.
        """
        blocks: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return blocks

    # ------------------------------------------------------------------
    # Post-conversation
    # ------------------------------------------------------------------

    async def _post_conversation(
        self,
        conversation_id: str,
        channel: str,
        person_name: str | None,
        messages: list[dict[str, Any]],
        accessed_memory_ids: list[str],
        started_at: str,
        injected_memories_block: str = "",
    ) -> None:
        """Persist transcript + queue extraction batch for this conversation.

        Runs after the conversation ends. The transcript is recorded in
        ``pending_extractions`` (durable queue, retained 14 days) and a
        1-request batch is submitted to Anthropic. The BatchPoller picks
        up the result when it lands (typically <30 min) and applies the
        memories.

        **Trigger-fired conversations are special-cased**: when the
        channel is ``trigger`` and no human reply landed on the thread,
        the conversation is a routine wake-up (morning brief, midday
        check, evening review). We write a deterministic conversation
        summary directly and skip the extraction batch entirely —
        otherwise every firing accumulates a near-duplicate operational
        "I sent the briefing today" memory that crowds out
        load-bearing methodology/person facts at injection time.

        On any failure, the row is left in queued status with no batch
        id, and the next boot's poller resume will retry submission.
        """
        if channel == "trigger" and not _has_human_reply(messages):
            await self._write_trigger_summary(
                conversation_id=conversation_id,
                person_name=person_name,
                messages=messages,
                started_at=started_at,
            )
            return

        try:
            transcript = self._build_transcript(messages, person_name)

            participants = [get_config().agent.name]
            if person_name:
                participants.append(person_name)

            # Persist first (durability), then submit. If submit fails,
            # the row stays in queued status for the next retry.
            await self._memory_store.create_pending_extraction(
                conversation_id=conversation_id,
                transcript=transcript,
                accessed_memory_ids=accessed_memory_ids,
                channel=channel,
                participants=participants,
                started_at=started_at,
                injected_memories_block=injected_memories_block,
            )

            poller = self._batch_poller
            if poller is None:
                # Agent stopped between conversation end and post-
                # processing. Row stays queued; next boot resumes.
                logger.warning(
                    "Batch poller unavailable; conversation %s queued for retry",
                    conversation_id,
                )
                return
            row = await self._memory_store.get_pending_extraction(conversation_id)
            if row is not None:
                await poller.submit(row)
            logger.info(
                "Conversation %s persisted and extraction batch queued",
                conversation_id,
            )
        except Exception:
            logger.exception(
                "Post-conversation processing failed for %s",
                conversation_id,
            )

    async def _write_trigger_summary(
        self,
        *,
        conversation_id: str,
        person_name: str | None,
        messages: list[dict[str, Any]],
        started_at: str,
    ) -> None:
        """Write a deterministic receipt for a routine trigger conversation.

        Avoids the extraction batch + memory creation entirely. The
        conversations table gets a queryable journal row (a receipt,
        not content). It is NOT ambient-injected — `inject_memories`
        excludes trigger conversations — so it can't earworm, but
        `search_memory` can still surface it on a deliberate lookup
        ("how did the morning briefing go?").
        """
        try:
            summary = _summarize_trigger_thread(messages, started_at)
            participants = [get_config().agent.name]
            if person_name:
                participants.append(person_name)
            existing = await self._memory_store.get_conversation(
                conversation_id
            )
            if existing is None:
                await self._memory_store.create_conversation(
                    channel="trigger",
                    participants=participants,
                    summary=summary,
                    topics=["trigger"],
                    accessed_memories=[],
                    conversation_id=conversation_id,
                    started_at=started_at,
                )
            else:
                await self._memory_store.update_conversation(
                    conversation_id,
                    summary=summary,
                    topics=["trigger"],
                    accessed_memories=[],
                )
            logger.info(
                "Trigger conversation %s summarised (no extraction): %s",
                conversation_id, summary[:80],
            )
        except Exception:
            logger.exception(
                "Failed to write trigger summary for %s", conversation_id,
            )

        # Dispatch-as-bridge: fold the trigger run's FULL reasoning into
        # each addressed recipient's real conversation, so a reply
        # continues a thread that contains not just the briefing text but
        # the run that produced it (calendar pulls with event ids, tool
        # results) — enough to fix the upstream source, not just a memory.
        # Runs after the receipt write and is independently fault-isolated
        # — a bridge failure must not lose the receipt.
        delivered = _delivered_text_messages_from_thread(messages)
        if delivered:
            description = _trigger_description_from_thread(messages)
            transcript = self._build_transcript(messages, person_name)
            # Distinct recipients, in first-delivered order.
            recipients = list(dict.fromkeys(r for r, _ in delivered))
            for recipient in recipients:
                recipient_texts = [
                    c for r, c in delivered if r == recipient
                ]
                turns = Conversation.build_trigger_context_turns(
                    description=description,
                    transcript=transcript,
                    recipient=recipient,
                    delivered_texts=recipient_texts,
                )
                try:
                    await self._bridge_trigger_delivery(recipient, turns)
                except Exception:
                    logger.exception(
                        "Failed to bridge trigger delivery to %s (conv=%s)",
                        recipient, conversation_id,
                    )

    async def _bridge_trigger_delivery(
        self, recipient: str, turns: list[dict[str, Any]],
    ) -> None:
        """Record a trigger run's context turns into the recipient's own
        conversation thread (dispatch-as-bridge).

        ``turns`` is the prebuilt block from
        ``Conversation.build_trigger_context_turns`` — the full run
        transcript plus the delivered message(s). If the recipient has a
        live in-memory conversation, it folds into that via
        ``Conversation.ingest_trigger_delivery`` (which handles the
        mid-generation race). Otherwise it goes straight to the store via
        ``get_or_create_active`` — resurrecting their thread if it's
        still inside the rolling window, or starting a fresh persistent
        one — so the next inbound rehydrates a thread that already
        contains the briefing and its reasoning.
        """
        store = self._conversation_store
        if store is None:
            logger.debug(
                "No conversation store; cannot bridge delivery to %s",
                recipient,
            )
            return

        # Resolve recipient name → phone (same path _dispatch_text uses).
        from boxbot.communication.auth import get_auth_manager
        auth = get_auth_manager()
        if auth is None:
            logger.warning(
                "No auth manager; cannot bridge delivery to %s", recipient,
            )
            return
        phone: str | None = None
        try:
            for user in await auth.list_users():
                if user.name.strip().lower() == recipient.strip().lower():
                    phone = user.phone
                    break
        except Exception:
            logger.exception("Failed to resolve %s for bridge", recipient)
            return
        if phone is None:
            logger.warning(
                "Cannot bridge delivery — '%s' is not a registered user",
                recipient,
            )
            return

        channel_key = f"whatsapp:{phone}"
        window = float(get_config().whatsapp.thread_window_seconds)
        agent_name = get_config().agent.name

        # Hold the index lock so a concurrent _get_or_create_conversation
        # can't rehydrate the same thread mid-bridge.
        async with self._index_lock:
            existing_id = self._conversation_by_key.get(channel_key)
            conv = (
                self._conversations.get(existing_id)
                if existing_id else None
            )
            if conv is not None and not conv.is_ended:
                recorded = await conv.ingest_trigger_delivery(
                    recipient=recipient, turns=turns,
                )
                if recorded:
                    logger.info(
                        "Bridged trigger delivery into live conversation "
                        "%s (%s)", conv.conversation_id, channel_key,
                    )
                    return
                # conv was ENDED between the check and the call — fall
                # through to the store path.

            # Store-only path: no live conversation. Resurrect the
            # recipient's thread within the window, or start a fresh
            # persistent one, and append the run's context turns.
            record, created = await store.get_or_create_active(
                channel="whatsapp",
                channel_key=channel_key,
                max_inactive_seconds=window,
                participants={recipient, agent_name},
            )
            await store.append_turns(record.conversation_id, turns)
            logger.info(
                "Bridged trigger delivery into %s conversation %s (%s)",
                "new" if created else "stored",
                record.conversation_id, channel_key,
            )

    @staticmethod
    def _build_transcript(
        messages: list[dict[str, Any]],
        person_name: str | None,
    ) -> str:
        """Build a human-readable transcript from message history.

        Renders:
        - User turns with speaker labels.
        - Assistant text blocks (private internal notes — thought + observations)
          as ``[boxBot thought]:`` / ``[boxBot observed]:`` lines so memory
          extraction sees them but downstream readers know they're private.
        - ``message`` tool calls as ``[boxBot → <to> via <channel>]:``
          — these are what the agent actually said.
        - Other tool calls as ``[boxBot used tool: <name>]``.

        Args:
            messages: The message history from the agent loop.
            person_name: The identified speaker name (used for user labels).

        Returns:
            Multi-line transcript string with speaker labels.
        """
        user_label = person_name or "User"
        lines: list[str] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                if isinstance(content, str):
                    lines.append(f"[{user_label}]: {content}")
                elif isinstance(content, list):
                    # Tool results — summarise
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "tool_result":
                                tool_content = block.get("content", "")
                                lines.append(
                                    f"[Tool Result]: {str(tool_content)[:200]}"
                                )
                            else:
                                text = block.get("text", "")
                                if text:
                                    lines.append(f"[{user_label}]: {text}")

            elif role == "assistant":
                if isinstance(content, str):
                    lines.append(f"[boxBot]: {content}")
                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        block_type = block.get("type")
                        if block_type == "text":
                            text = block.get("text", "")
                            if not text:
                                continue
                            try:
                                parsed = json.loads(text)
                            except (json.JSONDecodeError, TypeError):
                                lines.append(f"[boxBot thought]: {text}")
                                continue
                            if isinstance(parsed, dict):
                                thought = parsed.get("thought")
                                if thought:
                                    lines.append(f"[boxBot thought]: {thought}")
                                obs = parsed.get("observations")
                                if isinstance(obs, list):
                                    for entry in obs:
                                        if isinstance(entry, str) and entry:
                                            lines.append(
                                                f"[boxBot observed]: {entry}"
                                            )
                            else:
                                lines.append(f"[boxBot thought]: {text}")
                        elif block_type == "tool_use":
                            from boxbot.core.agent_sdk_adapter import (
                                base_tool_name,
                            )
                            name = base_tool_name(str(block.get("name") or ""))
                            tool_input = block.get("input") or {}
                            if name == "message":
                                to = str(tool_input.get("to", "")).strip() or "?"
                                channel = (
                                    str(tool_input.get("channel", "")).strip()
                                    or "?"
                                )
                                spoken = (
                                    str(tool_input.get("content", "")).strip()
                                )
                                if spoken:
                                    lines.append(
                                        f"[boxBot → {to} via {channel}]: "
                                        f"{spoken}"
                                    )
                            else:
                                lines.append(f"[boxBot used tool: {name}]")

        return "\n".join(lines)

    @staticmethod
    def _extract_summary(messages: list[dict[str, Any]]) -> str:
        """Extract a brief summary from the latest assistant turn.

        Prefers the most recent ``message`` tool call's ``content``
        (what the agent actually said to a person). Falls back to the
        ``thought`` field of the assistant's text JSON for silent turns.

        Args:
            messages: The message history.

        Returns:
            A brief summary string (truncated to 200 chars).
        """
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")

            if isinstance(content, list):
                # Prefer the latest message tool content as the summary.
                from boxbot.core.agent_sdk_adapter import base_tool_name
                for block in reversed(content):
                    if not isinstance(block, dict):
                        continue
                    if (
                        block.get("type") == "tool_use"
                        and base_tool_name(str(block.get("name") or ""))
                        == "message"
                    ):
                        tool_input = block.get("input") or {}
                        spoken = str(tool_input.get("content") or "").strip()
                        if spoken:
                            return spoken[:200]
                # Fall back to the thought field of the text block.
                for block in reversed(content):
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "text":
                        continue
                    text = block.get("text", "")
                    if not text:
                        continue
                    try:
                        parsed = json.loads(text)
                    except (json.JSONDecodeError, TypeError):
                        return text[:200]
                    if isinstance(parsed, dict):
                        thought = parsed.get("thought")
                        if thought:
                            return str(thought)[:200]
                    return text[:200]
            elif isinstance(content, str) and content:
                return content[:200]

        return "(no summary)"

    # ------------------------------------------------------------------
    # Presence helpers
    # ------------------------------------------------------------------

    def _get_present_people(
        self,
        exclude: str | None = None,
        window_minutes: int = 5,
    ) -> list[str]:
        """Return names of people currently present (seen recently).

        Args:
            exclude: A name to exclude from the list (typically the speaker).
            window_minutes: How many minutes since last detection to still
                consider someone "present".

        Returns:
            Sorted list of present person names.
        """
        now = datetime.now()
        present = []
        for name, last_seen in self._present_people.items():
            if exclude and name == exclude:
                continue
            elapsed = (now - last_seen).total_seconds()
            if elapsed <= window_minutes * 60:
                present.append(name)
        return sorted(present)

    def _get_present_people_with_status(
        self,
        exclude: str | None = None,
    ) -> list[str]:
        """Return formatted strings of present people with status.

        Queries the perception pipeline for richer presence data including
        confirmation status. Falls back to the basic list if the pipeline
        is not running.
        """
        try:
            from boxbot.perception.pipeline import get_pipeline

            people = get_pipeline().get_present_people()
            result = []
            for p in people:
                name = p.get("name")
                if name and name != exclude:
                    if p.get("confidence", 0) > 0.8:
                        result.append(f"{name} (confirmed)")
                    else:
                        result.append(f"{name} (visual)")
                elif not name:
                    result.append(f"{p.get('ref', 'unknown')} (new)")
            return result
        except (RuntimeError, Exception):
            # Pipeline not running
            return self._get_present_people(exclude=exclude)

    def _format_active_display_line(self) -> str | None:
        """Return a single-line summary of what is currently on the screen.

        ``None`` when the display manager is not running or nothing is
        active — keeps the prompt clean during early boot or tests.
        """
        try:
            from boxbot.displays.manager import get_display_manager

            mgr = get_display_manager()
            if mgr is None:
                return None
            name = mgr.get_active()
            if not name:
                return None
            theme = mgr.get_active_theme()
            theme_name = getattr(theme, "name", None) if theme else None
            args = mgr.get_active_args()
            line = f"Display: {name}"
            if theme_name:
                line += f" (theme={theme_name})"
            if args:
                line += f" args={args}"
            return line
        except Exception:
            logger.debug("Could not read active display for prompt", exc_info=True)
            return None

    def _get_most_recent_person(self) -> str | None:
        """Return the name of the most recently seen person.

        Used to infer the speaker when a wake word is heard without
        explicit speaker identification.

        Returns:
            The name of the most recently detected person, or None.
        """
        if not self._present_people:
            return None
        return max(self._present_people, key=self._present_people.get)
