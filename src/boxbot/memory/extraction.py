"""Post-conversation memory extraction via Anthropic Message Batches.

After every conversation ends, the agent persists the transcript to
``pending_extractions`` and submits a 1-request batch to Anthropic. The
batch poller (``boxbot.memory.batch_poller``) wakes up, fetches the
result when ready (typically within ~30 minutes), parses the structured
``tool_use`` block, and applies the result to the memory store.

Public API:
    submit_extraction_batch(...)     # called on conversation end
    parse_extraction_result(...)     # called by poller from raw message
    process_extraction_result(...)   # apply parsed result to the store
    record_extraction_cost(...)      # cost log helper

The structured-output schema, system prompt, and pricing live here so
they're version-controlled together with the extraction logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anthropic

# MemoryStore is only used as a type annotation; deferring the import
# avoids a circular load when ``boxbot.core.__init__`` reaches agent.py
# (which imports batch_poller, which imports this module).
if TYPE_CHECKING:
    from boxbot.memory.store import MemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output types
# ---------------------------------------------------------------------------


@dataclass
class ConversationSummary:
    participants: list[str]
    channel: str
    topics: list[str]
    summary: str


@dataclass
class ExtractedMemory:
    type: str  # person, household, methodology, operational
    person: str | None
    content: str
    summary: str
    tags: list[str]
    action: str  # create | update | skip
    existing_memory_id: str | None = None
    reason: str | None = None


@dataclass
class Invalidation:
    memory_id: str
    reason: str
    replacement: ExtractedMemory | None = None


@dataclass
class SystemMemoryUpdate:
    section: str
    action: str  # set | add_entry | remove_entry
    content: str


@dataclass
class ExtractionResult:
    conversation_summary: ConversationSummary
    extracted_memories: list[ExtractedMemory] = field(default_factory=list)
    invalidations: list[Invalidation] = field(default_factory=list)
    system_memory_updates: list[SystemMemoryUpdate] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# Default extraction model. Sonnet 4.6 is plenty for structured fact
# extraction; Opus is overkill at the per-call price even with the 50%
# batch discount. Override via the ``model`` argument or by editing
# ``boxbot.core.config.models.large`` in YAML.
DEFAULT_EXTRACTION_MODEL = "claude-sonnet-4-6"

# Pricing lives in ``config/pricing.yaml`` and is read through
# :mod:`boxbot.cost.pricing`. The two symbols below are kept for
# backward compatibility with existing imports; values are sourced from
# the YAML at access time so there is no hardcoded duplicate.


def __getattr__(name: str):  # pragma: no cover - thin re-export
    """Lazily expose STANDARD_PRICING from the YAML config.

    Module-level ``__getattr__`` (PEP 562) lets ``from
    boxbot.memory.extraction import STANDARD_PRICING`` continue to
    work without baking prices into source.
    """
    if name == "STANDARD_PRICING":
        from boxbot.cost.pricing import get_pricing

        pricing = get_pricing()
        return {
            model: (entry["input_per_mtok"], entry["output_per_mtok"])
            for model, entry in pricing.anthropic_models.items()
        }
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def compute_cost(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens_1h: int = 0,
    is_batch: bool = False,
) -> float:
    """Compute USD cost for a single Anthropic call.

    Thin shim over :func:`boxbot.cost.from_anthropic_usage` — kept for
    callers in the memory subsystem that still pass keyword arguments.
    Cache-write at 1h TTL is 2x input, cache-read is 10% of input,
    batch applies a flat 50% discount; multipliers live in the cost
    module, prices in ``config/pricing.yaml``.
    """
    from boxbot.cost import from_anthropic_usage

    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read_tokens,
        "cache_creation_input_tokens": cache_write_tokens_1h,
    }
    event = from_anthropic_usage(
        purpose="_legacy_compute_cost",
        model=model,
        usage=usage,
        is_batch=is_batch,
    )
    return event.cost_usd


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


# Tool definition. Forces the model to emit a single, structured
# ``emit_extraction`` call matching the ExtractionResult dataclass. The
# JSON schema is duplicated here because ``input_schema`` lives in the
# API request, not the dataclass — keep them in sync.
EXTRACTION_TOOL: dict[str, Any] = {
    "name": "emit_extraction",
    "description": (
        "Emit the post-conversation extraction result. Call this exactly "
        "once with the conversation summary, any new facts to remember, "
        "any invalidations of injected memories, and any system memory "
        "updates the user explicitly authorised."
    ),
    "input_schema": {
        "type": "object",
        "required": ["conversation_summary"],
        "properties": {
            "conversation_summary": {
                "type": "object",
                "required": ["topics", "summary"],
                "properties": {
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "1-5 short topic tags.",
                    },
                    "summary": {
                        "type": "string",
                        "description": (
                            "Ultra-compact summary. Preserve who/what/"
                            "decisions/actions/outcomes. Strip filler. "
                            "Use shorthand. 1-3 sentences typical."
                        ),
                    },
                },
            },
            "extracted_memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["type", "content", "summary", "tags"],
                    "properties": {
                        "type": {"enum": [
                            "person", "household", "methodology", "operational",
                        ]},
                        "person": {"type": ["string", "null"]},
                        "content": {"type": "string"},
                        "summary": {"type": "string"},
                        "tags": {
                            "type": "array", "items": {"type": "string"},
                        },
                    },
                },
            },
            "invalidations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["memory_id", "reason"],
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": (
                                "ID of an INJECTED memory that the user "
                                "directly contradicted in this conversation."
                            ),
                        },
                        "reason": {"type": "string"},
                        "replacement": {
                            "type": ["object", "null"],
                            "properties": {
                                "type": {"type": "string"},
                                "person": {"type": ["string", "null"]},
                                "content": {"type": "string"},
                                "summary": {"type": "string"},
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
            "system_memory_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["section", "action", "content"],
                    "properties": {
                        "section": {"enum": [
                            "Household", "Standing Instructions",
                            "Operational Notes",
                        ]},
                        "action": {"enum": [
                            "set", "add_entry", "remove_entry",
                        ]},
                        "content": {"type": "string"},
                    },
                },
            },
        },
    },
}


# Cached system prompt. Encodes the extraction policy so the model
# emits high-quality, conservative results. Anything that varies per
# call (transcript, injected memories) goes in the user message — the
# system block stays identical across requests so prompt caching can
# hit on it.
EXTRACTION_SYSTEM_PROMPT = """\
You are the post-conversation memory extractor for boxBot, a household assistant.

Your job: read the transcript and decide what (if anything) is worth remembering for future conversations. Emit the result via the `emit_extraction` tool exactly once.

# Memory taxonomy

- **person**: a fact about a specific individual ("Jacob is allergic to shellfish", "Erik's school pickup is 3:15 PM weekdays"). Set `person` to the primary subject. Tag with relevant topics.
- **household**: a fact about the shared environment that doesn't belong to one person ("The WiFi password is on the fridge", "Family car is a blue 2022 Subaru").
- **methodology**: an approach BB itself learned and might re-use ("Pandas read_csv needs encoding='utf-8-sig' for Jacob's bank exports"). Use sparingly — only durable lessons.
- **operational**: something BB did, an activity-log entry ("Sent grocery list to Carina via WhatsApp on 4/29: pesto, mozzarella, chicken").

# Bias toward fewer, higher-quality memories

It is better to extract zero memories than to extract noise. Skip:
- Pleasantries, acknowledgements, small talk.
- Restatements of facts already in the injected memories.
- One-off questions that don't reveal a durable fact.
- Things BB itself "knows" generically (general world knowledge, weather, time).

Aim for 0-4 memories per conversation. Long technical work sessions can justify more, but be selective.

# Memory content style

- `content`: full prose, complete enough to make sense without context. 1-3 sentences.
- `summary`: one short line, suitable for injection into a future system prompt. <80 chars.
- `tags`: 1-4 lowercase topic words. Reuse tags consistently when possible.
- For `operational` entries about work-products (analyses, reports), reference the workspace path explicitly: "Saved Q1 expense report to workspace/notes/jacob/q1_summary.md".

# Invalidations

ONLY invalidate memories listed in the [Active Memories] block of the user message. Those are the memories that were injected into this conversation; the model has been told what they say.

Invalidate when the conversation directly contradicts an injected memory. Example: injected says "Jacob is vegetarian", user says "I'm not vegetarian anymore". Provide a `replacement` memory if the conversation gave the corrected fact.

Do NOT invalidate based on inference, omission, or memories you weren't shown.

# System memory updates

System memory is for facts BB must always know. Be VERY conservative. Update only when:
- The user said "always do X", "never do Y", or stated a standard operating procedure.
- The user stated a safety-critical fact (allergy, medication, accessibility need).
- A new household member's identity/role was established (NOT details about them — those are person memories).

Never propose system memory updates inferred from indirect signals. The bar is: the user explicitly authorised it, or it's a roster/safety/SOP fact stated unambiguously.

When in doubt, leave system memory alone. The dream phase will catch slow-burn patterns.

# Empty extractions are fine

Trigger-fired wake-ups, scheduler check-ins, very short exchanges, or pure "thank you" turns often have nothing worth extracting. Return `extracted_memories: []` with just the conversation_summary. That's a correct, expected outcome.

# Always emit the tool call

Even with zero extractions, call `emit_extraction` with at minimum the conversation_summary. Never respond with prose.
"""


def build_user_message(
    *,
    transcript: str,
    injected_memories_block: str,
    channel: str,
    participants: list[str],
    started_at: str,
) -> str:
    """Build the per-conversation user message for the batch.

    Everything dynamic goes here so the system prompt above stays cache-stable.
    """
    header = (
        f"Conversation metadata: channel={channel}, "
        f"participants={', '.join(participants) or '(unknown)'}, "
        f"started_at={started_at}.\n\n"
    )
    if injected_memories_block.strip():
        header += injected_memories_block.strip() + "\n\n"
    else:
        header += "[Active Memories]\n(none injected)\n\n"

    return header + "[Transcript]\n" + transcript.strip() + "\n"


# ---------------------------------------------------------------------------
# Batch submission
# ---------------------------------------------------------------------------


# Hard cap on transcript characters. Long technical sessions can blow
# past Sonnet's input limit; we trim aggressively at submission. The
# original is kept on disk in pending_extractions.transcript so the
# agent can still recover the full text via search_memory mode=transcript.
MAX_TRANSCRIPT_CHARS = 200_000


def _trim_transcript(transcript: str) -> str:
    """If transcript is too long, keep the head and tail with a marker."""
    if len(transcript) <= MAX_TRANSCRIPT_CHARS:
        return transcript
    half = MAX_TRANSCRIPT_CHARS // 2 - 100
    return (
        transcript[:half]
        + "\n\n... [transcript truncated for extraction] ...\n\n"
        + transcript[-half:]
    )


async def submit_extraction_batch(
    client: anthropic.AsyncAnthropic,
    *,
    transcript: str,
    injected_memories_block: str,
    conversation_id: str,
    channel: str,
    participants: list[str],
    started_at: str,
    model: str = DEFAULT_EXTRACTION_MODEL,
    max_tokens: int = 4096,
) -> str:
    """Submit a 1-request batch for this conversation. Returns batch_id.

    The system prompt is marked with a 1h cache TTL so subsequent
    extraction batches share the cached prefix (~30-50% input-token
    savings on cache hits).
    """
    user_msg = build_user_message(
        transcript=_trim_transcript(transcript),
        injected_memories_block=injected_memories_block,
        channel=channel,
        participants=participants,
        started_at=started_at,
    )

    request = {
        "custom_id": conversation_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": EXTRACTION_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
            ],
            "tools": [EXTRACTION_TOOL],
            "tool_choice": {"type": "tool", "name": "emit_extraction"},
            "messages": [{"role": "user", "content": user_msg}],
        },
    }

    batch = await client.messages.batches.create(requests=[request])
    logger.info(
        "Submitted extraction batch %s for conversation %s (model=%s)",
        batch.id, conversation_id, model,
    )
    return batch.id


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def parse_extraction_result(message: Any) -> ExtractionResult:
    """Parse a successful Anthropic Message into an ExtractionResult.

    The model is constrained to emit a single ``emit_extraction``
    tool_use block. We pull that block's ``input`` and convert it to
    our dataclass. Raises ValueError if the tool call is missing or
    malformed.
    """
    content = getattr(message, "content", None) or []
    tool_block = None
    for block in content:
        # Both anthropic SDK objects and dict-like access supported.
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type == "tool_use":
            block_name = getattr(block, "name", None) or block.get("name")
            if block_name == "emit_extraction":
                tool_block = block
                break

    if tool_block is None:
        raise ValueError("Extraction message had no emit_extraction tool_use block")

    payload = getattr(tool_block, "input", None)
    if payload is None and isinstance(tool_block, dict):
        payload = tool_block.get("input")
    if not isinstance(payload, dict):
        raise ValueError(f"emit_extraction input was not an object: {payload!r}")

    return _payload_to_result(payload)


def _payload_to_result(payload: dict) -> ExtractionResult:
    cs = payload.get("conversation_summary") or {}
    summary = ConversationSummary(
        participants=cs.get("participants", []),
        channel=cs.get("channel", ""),
        topics=cs.get("topics", []),
        summary=cs.get("summary", ""),
    )

    extracted = []
    for m in payload.get("extracted_memories", []) or []:
        extracted.append(ExtractedMemory(
            type=m.get("type", ""),
            person=m.get("person"),
            content=m.get("content", ""),
            summary=m.get("summary", ""),
            tags=m.get("tags", []) or [],
            action=m.get("action") or "create",
            existing_memory_id=m.get("existing_memory_id"),
            reason=m.get("reason"),
        ))

    invalidations = []
    for inv in payload.get("invalidations", []) or []:
        rep_payload = inv.get("replacement")
        replacement = None
        if rep_payload:
            replacement = ExtractedMemory(
                type=rep_payload.get("type", ""),
                person=rep_payload.get("person"),
                content=rep_payload.get("content", ""),
                summary=rep_payload.get("summary", ""),
                tags=rep_payload.get("tags", []) or [],
                action="create",
            )
        invalidations.append(Invalidation(
            memory_id=inv.get("memory_id", ""),
            reason=inv.get("reason", ""),
            replacement=replacement,
        ))

    sys_updates = []
    for u in payload.get("system_memory_updates", []) or []:
        sys_updates.append(SystemMemoryUpdate(
            section=u.get("section", ""),
            action=u.get("action", ""),
            content=u.get("content", ""),
        ))

    return ExtractionResult(
        conversation_summary=summary,
        extracted_memories=extracted,
        invalidations=invalidations,
        system_memory_updates=sys_updates,
    )


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


async def process_extraction_result(
    store: MemoryStore,
    result: ExtractionResult,
    conversation_id: str,
) -> str:
    """Apply the parsed extraction result to the memory store.

    Creates the conversation log entry, inserts/updates extracted
    memories, applies invalidations, and applies system memory updates.
    Returns the conversation log entry ID. Idempotency note: callers
    should mark the pending_extraction row applied AFTER this returns
    so a crash mid-apply doesn't double-write on resume.
    """
    summary = result.conversation_summary

    conv_id = await store.create_conversation(
        channel=summary.channel,
        participants=summary.participants,
        summary=summary.summary,
        topics=summary.topics,
        accessed_memories=[],
    )
    logger.info("Created conversation log entry %s", conv_id)

    created_count = 0
    updated_count = 0
    for mem in result.extracted_memories:
        if mem.action == "skip":
            continue
        if mem.action == "create" or mem.action not in {"create", "update"}:
            await store.create_memory(
                type=mem.type,
                content=mem.content,
                summary=mem.summary,
                person=mem.person,
                people=[mem.person] if mem.person else [],
                tags=mem.tags,
                source_conversation=conv_id,
            )
            created_count += 1
        elif mem.action == "update" and mem.existing_memory_id:
            await store.update_memory_content(
                mem.existing_memory_id,
                content=mem.content,
                summary=mem.summary,
                tags=mem.tags,
            )
            updated_count += 1

    logger.info(
        "Extraction: created %d, updated %d memories",
        created_count, updated_count,
    )

    for inv in result.invalidations:
        replacement_id = None
        if inv.replacement:
            replacement_id = await store.create_memory(
                type=inv.replacement.type,
                content=inv.replacement.content,
                summary=inv.replacement.summary,
                person=inv.replacement.person,
                people=[inv.replacement.person] if inv.replacement.person else [],
                tags=inv.replacement.tags,
                source_conversation=conv_id,
            )
        await store.invalidate_memory(
            inv.memory_id,
            invalidated_by=conversation_id,
            superseded_by=replacement_id,
        )
        logger.info(
            "Invalidated memory %s → %s: %s",
            inv.memory_id, replacement_id, inv.reason,
        )

    for update in result.system_memory_updates:
        try:
            await store.update_system_memory(
                section=update.section,
                action=update.action,
                content=update.content,
                updated_by=f"extraction:{conversation_id}",
            )
        except ValueError as e:
            logger.warning(
                "System memory update rejected: %s (section=%s, action=%s)",
                e, update.section, update.action,
            )

    return conv_id


# ---------------------------------------------------------------------------
# Cost recording
# ---------------------------------------------------------------------------


async def record_extraction_cost(
    store: MemoryStore,
    *,
    model: str,
    usage: Any,
    is_batch: bool,
    conversation_id: str,
    batch_id: str | None = None,
) -> float:
    """Compute USD cost from an Anthropic Usage object and append to cost_log.

    ``usage`` is the SDK's Usage type; we read input_tokens, output_tokens,
    and cache fields if present. Returns the recorded cost.
    """
    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_write_1h = int(
        getattr(usage, "cache_creation_input_tokens", 0) or 0
    )
    cost = compute_cost(
        model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_read_tokens=cache_read,
        cache_write_tokens_1h=cache_write_1h,
        is_batch=is_batch,
    )
    await store.record_cost(
        purpose="extraction",
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write_1h,
        is_batch=is_batch,
        cost_usd=cost,
        metadata={"conversation_id": conversation_id, "batch_id": batch_id},
    )
    return cost


# ---------------------------------------------------------------------------
# Backwards-compat shim
# ---------------------------------------------------------------------------
#
# The previous synchronous `extract_memories` function is removed. The
# agent now calls `submit_extraction_batch` on conversation end. The
# poller calls `parse_extraction_result` then `process_extraction_result`.
# Importers that still reference `extract_memories` will fail loudly so
# we catch any stale call site at boot, not silently no-op like before.
