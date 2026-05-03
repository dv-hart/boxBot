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
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

from boxbot.core.config import get_config
from boxbot.cost import from_anthropic_usage, record as record_cost
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
    PersonIdentified,
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
_DEFAULT_MAX_TURNS = 20

# Opus 4.7 needs more headroom than Sonnet 4. Spec §3 mandates 8192.
_MAX_TOKENS = 8192

# The structured-output schema is defined in ``output_dispatcher`` so the
# dispatcher and the agent loop share a single source of truth. Schema
# mutation invalidates the messages cache — it is pinned at module scope
# there and imported here.


def _generate_conversation_id() -> str:
    """Generate a unique conversation ID."""
    return f"conv_{uuid.uuid4().hex[:12]}"


# Mime type → file extension for inbound WhatsApp images. Restricted to
# the formats the multimodal attach pipeline accepts (build_image_block).
_WHATSAPP_IMAGE_EXTS: dict[str, str] = {
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

    ext = _WHATSAPP_IMAGE_EXTS.get(mime_type.lower())
    if ext is None:
        logger.warning(
            "WhatsApp image %s: unsupported mime %s", media_id, mime_type
        )
        return None

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
        "and WhatsApp.\n\n"
        "You recognise the people around you and proactively help them — "
        "relaying messages, managing tasks, controlling displays, and "
        "remembering everything important about your household. You are warm, "
        "concise, and genuinely useful. You know when to speak up and when "
        "to stay quiet.\n\n"
        f"Your wake word is \"{wake_word}\"."
    )


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
        "speak through the box speaker or send a WhatsApp message. Without a\n"
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
        "- `\"text\"` — WhatsApp message to the named user's phone. Requires\n"
        "  `to` be a registered user by name. Cannot text \"room\" or unknowns.\n"
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
        "  the filler AND the lookup tool in the same response:\n"
        "    message(to=\"current_speaker\", channel=\"speak\",\n"
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
        "switch_display, identify_person, manage_tasks, search_memory, "
        "search_photos, web_search, and load_skill. For complex or "
        "infrequent operations, use execute_script to run Python in the "
        "sandbox with the boxbot_sdk.\n"
        "\n"
        "message is the ONLY tool that reaches a person. The other\n"
        "tools DO things — create a reminder, search memory, switch a\n"
        "display, run a sandbox script, look up the web. None of them speak.\n"
        "Without a message call, the user hears silence.\n"
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

    def __init__(self, memory_store: MemoryStore) -> None:
        """Initialise the agent.

        Args:
            memory_store: The initialised MemoryStore for memory injection,
                extraction, and system memory reading.
        """
        self._memory_store = memory_store
        self._client: anthropic.AsyncAnthropic | None = None
        # Background poller for in-flight extraction batches. Started
        # alongside the agent and runs for the agent's lifetime.
        self._batch_poller: BatchPoller | None = None
        # Background poller for in-flight dream-phase batches (PR1:
        # nightly dedup consolidation; runs in audit-only mode by
        # default — flip ``memory.dream_audit_only=False`` in config to
        # enable real merges).
        self._dream_poller: DreamPoller | None = None

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

        # Validate that an Anthropic API key is available
        if not config.api_keys.anthropic:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. The agent cannot start "
                "without an API key."
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

        # Start the dream-phase poller. Audit-only by default; flip
        # ``memory.dream_audit_only=False`` in config to enable real
        # consolidation. Resumes any in-flight dream batches from the
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
        bus.subscribe(TriggerFired, self._on_trigger_fired)
        bus.subscribe(PersonIdentified, self._on_person_identified)
        bus.subscribe(SpeakerIdentified, self._on_speaker_identified)
        bus.subscribe(TranscriptReady, self._on_transcript_ready)
        bus.subscribe(VoiceSessionEnded, self._on_voice_session_ended)
        bus.subscribe(ConversationEnded, self._on_conversation_ended)
        bus.subscribe(AgentSpeaking, self._on_agent_speaking)
        bus.subscribe(AgentSpeakingDone, self._on_agent_speaking_done)

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
        bus.unsubscribe(TriggerFired, self._on_trigger_fired)
        bus.unsubscribe(PersonIdentified, self._on_person_identified)
        bus.unsubscribe(SpeakerIdentified, self._on_speaker_identified)
        bus.unsubscribe(TranscriptReady, self._on_transcript_ready)
        bus.unsubscribe(VoiceSessionEnded, self._on_voice_session_ended)
        bus.unsubscribe(ConversationEnded, self._on_conversation_ended)
        bus.unsubscribe(AgentSpeaking, self._on_agent_speaking)
        bus.unsubscribe(AgentSpeakingDone, self._on_agent_speaking_done)

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

        self._client = None
        logger.info("BoxBotAgent stopped")

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
        conv = await self._get_or_create_conversation(
            channel="whatsapp",
            channel_key=channel_key,
            participants={event.sender_name} if event.sender_name else None,
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
        trigger fires. Audit-only by default (config flag
        ``memory.dream_audit_only``). Result is written to
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
            )
            logger.info(
                "Dream cycle complete: %s candidates, %s pairs, batch=%s",
                summary.get("candidate_count"),
                summary.get("near_dup_pairs"),
                summary.get("batch_id"),
            )
        except Exception:
            logger.exception("Dream cycle failed")

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
    ) -> Conversation:
        """Look up an existing Conversation by channel key, or create one.

        This is the only place the conversation index is mutated from
        inbound events. The index itself is a dict of live, non-ended
        conversations — ended conversations are removed via
        ``_on_conversation_ended``.
        """
        async with self._index_lock:
            existing_id = self._conversation_by_key.get(channel_key)
            if existing_id is not None:
                conv = self._conversations.get(existing_id)
                if conv is not None and not conv.is_ended:
                    if participants:
                        conv.participants.update(participants)
                    return conv
                # Stale entry — drop it and create fresh.
                self._conversation_by_key.pop(channel_key, None)
                self._conversations.pop(existing_id, None)

            conv_id = _generate_conversation_id()
            conv = Conversation(
                conversation_id=conv_id,
                channel=channel,
                channel_key=channel_key,
                generate_fn=self._generate_for_conversation,
                participants=participants,
                silence_timeout=silence_timeout,
            )
            self._conversations[conv_id] = conv
            self._conversation_by_key[channel_key] = conv_id
            logger.info(
                "Conversation %s created (channel=%s, key=%s)",
                conv_id, channel, channel_key,
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
        # subprocess spawn / sudo prompt resolution.
        asyncio.create_task(
            runner.start(),
            name=f"sandbox-start-{conv.conversation_id}",
        )

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

        # Fire-and-forget memory extraction on the ended thread.
        if conv.thread and event.turn_count > 0:
            asyncio.create_task(
                self._post_conversation(
                    conversation_id=conv_id,
                    channel=event.channel,
                    person_name=event.person_name,
                    messages=list(conv.thread),
                    accessed_memory_ids=list(conv.accessed_memory_ids),
                    started_at=conv.started_at_iso(),
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
        conv.set_state(ConversationState.THINKING)
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

        return GenerationResult(
            thread_additions=additions,
            turn_count=turn_count,
            summary=summary,
            completed_cleanly=True,
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
            f"Channel: {channel}",
        ]
        if person_name:
            context_lines.append(f"Speaking with: {person_name}")
        present = self._get_present_people_with_status(exclude=person_name)
        if present:
            context_lines.append(f"Also present: {', '.join(present)}")
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
                        "You can reach any of these people via `outputs` "
                        "(voice if they're at the box, text for WhatsApp).\n"
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

        # WhatsApp inbound image hint: when the inbound handler stages a
        # photo, the user message starts with "[image attached at <path>]".
        # Tell the agent how to act on it. Only injected when this turn's
        # context actually carries a staged path so we don't waste prompt
        # bytes on plain text turns.
        if (
            channel == "whatsapp"
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
                "with `bb.photos.ingest(path, source=\"whatsapp\", "
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

        while turn_count < max_turns:
            turn_count += 1

            try:
                # IMPORTANT: the exact named kwargs below are the only ones
                # we pass. No temperature / top_p / top_k / thinking — those
                # will 400 on Opus 4.7. See spec §3.
                response = await self._client.messages.create(
                    model=model,
                    max_tokens=_MAX_TOKENS,
                    system=system_prompt_blocks,
                    messages=messages,
                    tools=tool_definitions,
                    output_config={
                        "format": {
                            "type": "json_schema",
                            "schema": INTERNAL_NOTES_SCHEMA,
                        }
                    },
                    cache_control={"type": "ephemeral"},
                )
            except anthropic.APIError as e:
                logger.error(
                    "Anthropic API error on turn %d: %s", turn_count, e
                )
                messages.append({
                    "role": "assistant",
                    "content": f"(API error: {e})",
                })
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
                tool_results = await self._process_tool_calls(
                    response, tools, conversation_id=conversation_id,
                )
                if tool_results:
                    messages.append({
                        "role": "user",
                        "content": tool_results,
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
                "Conversation reached max turns (%d)", max_turns
            )

        return messages, turn_count

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
    ) -> None:
        """Persist transcript + queue extraction batch for this conversation.

        Runs after the conversation ends. The transcript is recorded in
        ``pending_extractions`` (durable queue, retained 14 days) and a
        1-request batch is submitted to Anthropic. The BatchPoller picks
        up the result when it lands (typically <30 min) and applies the
        memories.

        On any failure, the row is left in queued status with no batch
        id, and the next boot's poller resume will retry submission.
        """
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
                            name = block.get("name", "")
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
                for block in reversed(content):
                    if not isinstance(block, dict):
                        continue
                    if (
                        block.get("type") == "tool_use"
                        and block.get("name") == "message"
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
