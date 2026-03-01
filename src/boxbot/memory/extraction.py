"""Post-conversation memory extraction.

After every conversation, this module processes the transcript and produces:
- A conversation summary (for the conversation log)
- Extracted fact memories
- Contradiction invalidations with replacements
- System memory update proposals

The extraction agent uses the large model with structured output. The Claude
API call is stubbed but the full structured output format is defined.

Usage:
    from boxbot.memory.extraction import extract_memories, process_extraction_result

    result = await extract_memories(transcript, accessed_memory_ids, conversation_id)
    await process_extraction_result(store, result, conversation_id)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from boxbot.memory.store import MemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output types
# ---------------------------------------------------------------------------


@dataclass
class ConversationSummary:
    """Compact summary of a conversation."""

    participants: list[str]
    channel: str  # voice, whatsapp
    topics: list[str]
    summary: str


@dataclass
class ExtractedMemory:
    """A single fact extracted from a conversation."""

    type: str  # person, household, methodology, operational
    person: str | None
    content: str
    summary: str
    tags: list[str]
    action: str  # create, update, skip
    existing_memory_id: str | None = None
    reason: str | None = None


@dataclass
class Invalidation:
    """A contradiction detected between an injected memory and the conversation."""

    memory_id: str
    reason: str
    replacement: ExtractedMemory | None = None


@dataclass
class SystemMemoryUpdate:
    """A proposed update to system memory."""

    section: str
    action: str  # set, add_entry, remove_entry
    content: str


@dataclass
class ExtractionResult:
    """Complete extraction output from one conversation."""

    conversation_summary: ConversationSummary
    extracted_memories: list[ExtractedMemory] = field(default_factory=list)
    invalidations: list[Invalidation] = field(default_factory=list)
    system_memory_updates: list[SystemMemoryUpdate] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction (stubbed Claude API call)
# ---------------------------------------------------------------------------


async def extract_memories(
    transcript: str,
    accessed_memory_ids: list[str],
    conversation_id: str,
    *,
    channel: str = "voice",
    participants: list[str] | None = None,
) -> ExtractionResult:
    """Extract memories from a conversation transcript.

    In production, this calls the large Claude model with the full transcript
    and accessed memories to produce a structured ExtractionResult. The model
    receives:
    1. Full conversation transcript with speaker attribution
    2. Accessed memories (IDs + content) for contradiction detection
    3. Similar existing memories for deduplication (second pass)

    The structured output schema matches ExtractionResult exactly.

    Args:
        transcript: Full conversation transcript with speaker labels.
        accessed_memory_ids: IDs of memories injected into this conversation.
        conversation_id: The conversation's ID for provenance.
        channel: Conversation channel (voice, whatsapp).
        participants: People who participated.

    Returns:
        ExtractionResult with summary, extracted facts, invalidations,
        and system memory updates.
    """
    # TODO: Implement Claude API call with structured output
    # The model receives a system prompt defining the output schema,
    # plus the transcript and accessed memories as user content.
    #
    # System prompt structure:
    # - Define all output types (ConversationSummary, ExtractedMemory, etc.)
    # - Rules for when to create vs update vs skip
    # - Rules for invalidation (only when clear contradiction exists)
    # - Rules for system memory updates (only for always-needed info)
    # - Deduplication guidelines
    #
    # For now, return a minimal result with just the summary stub

    logger.info(
        "Extraction stub called for conversation %s (%d chars transcript)",
        conversation_id,
        len(transcript),
    )

    return ExtractionResult(
        conversation_summary=ConversationSummary(
            participants=participants or [],
            channel=channel,
            topics=[],
            summary=f"[Extraction stub] Conversation {conversation_id}",
        ),
        extracted_memories=[],
        invalidations=[],
        system_memory_updates=[],
    )


# ---------------------------------------------------------------------------
# Apply extraction results
# ---------------------------------------------------------------------------


async def process_extraction_result(
    store: MemoryStore,
    result: ExtractionResult,
    conversation_id: str,
) -> str:
    """Apply extraction results to the memory store.

    Processes all parts of the ExtractionResult:
    1. Creates the conversation log entry
    2. Creates/updates extracted memories
    3. Applies invalidations with replacements
    4. Applies system memory updates

    Args:
        store: The MemoryStore instance.
        result: The extraction result to process.
        conversation_id: ID for the conversation being processed.

    Returns:
        The conversation log entry ID.
    """
    summary = result.conversation_summary

    # 1. Create conversation log entry
    conv_id = await store.create_conversation(
        channel=summary.channel,
        participants=summary.participants,
        summary=summary.summary,
        topics=summary.topics,
        accessed_memories=[],  # Will be populated with injected memory IDs
    )
    logger.info("Created conversation log entry %s", conv_id)

    # 2. Process extracted memories
    created_count = 0
    updated_count = 0

    for mem in result.extracted_memories:
        if mem.action == "skip":
            continue

        if mem.action == "create":
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

    # 3. Process invalidations
    for inv in result.invalidations:
        replacement_id = None

        # Create the replacement memory first if provided
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

        # Invalidate the old memory
        await store.invalidate_memory(
            inv.memory_id,
            invalidated_by=conversation_id,
            superseded_by=replacement_id,
        )
        logger.info(
            "Invalidated memory %s → %s: %s",
            inv.memory_id, replacement_id, inv.reason,
        )

    # 4. Process system memory updates
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
