"""Claude Agent SDK integration — the brain of boxBot.

Wraps the Anthropic Python SDK to orchestrate conversations, dispatch tool
calls, manage system prompt construction with context injection, and trigger
post-conversation memory extraction.

The agent subscribes to events on the event bus (wake word, WhatsApp
messages, trigger fires, person identification) and manages the full
conversation lifecycle from prompt construction through to memory extraction.

Usage:
    from boxbot.core.agent import BoxBotAgent

    agent = BoxBotAgent()
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
import uuid
from datetime import datetime
from typing import Any

import anthropic

from boxbot.core.config import get_config
from boxbot.core.events import (
    ConversationEnded,
    ConversationStarted,
    PersonIdentified,
    TriggerFired,
    WakeWordHeard,
    WhatsAppMessage,
    get_event_bus,
)
from boxbot.core.scheduler import get_status_line
from boxbot.memory.extraction import extract_memories, process_extraction_result
from boxbot.memory.retrieval import inject_memories
from boxbot.memory.store import MemoryStore
from boxbot.tools.registry import get_tool, get_tools

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of API round-trips (tool use loops) per conversation
_DEFAULT_MAX_TURNS = 20

# Base system prompt — personality, identity, and capabilities overview.
# Context injections (memories, schedule status, who is present) are
# appended dynamically at conversation start.
_BASE_SYSTEM_PROMPT = """\
You are {name}, an ambient household assistant that lives in an elegant \
wooden box. You see through a camera, hear through a microphone array, \
speak through a speaker, and display information on a 7-inch screen. \
You communicate with your household via voice and WhatsApp.

You recognise the people around you and proactively help them — relaying \
messages, managing tasks, controlling displays, and remembering everything \
important about your household. You are warm, concise, and genuinely useful. \
You know when to speak up and when to stay quiet.

Your wake word is "{wake_word}".

## Capabilities
You have 9 always-available tools: execute_script, speak, switch_display, \
send_message, identify_person, manage_tasks, search_memory, search_photos, \
and web_search. For complex or infrequent operations, use execute_script to \
run Python in the sandbox with the boxbot_sdk.

## Guidelines
- Be concise in voice conversations — no one wants a lecture from a box.
- Use speak() for voice responses. For WhatsApp, just reply with text.
- When you learn something important, the memory system captures it \
  automatically after the conversation ends.
- You can search your memories at any time with search_memory.
- Check your to-do list and triggers when waking up on a schedule.
- For web lookups, use web_search — you never see raw web content directly.
- Respect privacy: never share one person's information with another \
  unless it's clearly appropriate (e.g., relaying a message they asked you to).
"""


def _generate_conversation_id() -> str:
    """Generate a unique conversation ID."""
    return f"conv_{uuid.uuid4().hex[:12]}"


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

        # People currently detected by the perception pipeline.
        # Updated via PersonIdentified event subscription.
        self._present_people: dict[str, datetime] = {}

        # Tracks active conversations to prevent overlapping sessions on
        # the same channel.
        self._active_conversations: dict[str, asyncio.Task[None]] = {}

        # Lock to serialise conversation starts (prevents race conditions
        # when multiple events arrive simultaneously).
        self._conversation_lock = asyncio.Lock()

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

        # Subscribe to events that initiate conversations
        bus = get_event_bus()
        bus.subscribe(WakeWordHeard, self._on_wake_word)
        bus.subscribe(WhatsAppMessage, self._on_whatsapp_message)
        bus.subscribe(TriggerFired, self._on_trigger_fired)
        bus.subscribe(PersonIdentified, self._on_person_identified)

        self._running = True
        logger.info("BoxBotAgent started (model: %s)", config.models.large)

    async def stop(self) -> None:
        """Graceful shutdown: unsubscribe from events and cancel active conversations."""
        if not self._running:
            return

        self._running = False

        # Unsubscribe from events
        bus = get_event_bus()
        bus.unsubscribe(WakeWordHeard, self._on_wake_word)
        bus.unsubscribe(WhatsAppMessage, self._on_whatsapp_message)
        bus.unsubscribe(TriggerFired, self._on_trigger_fired)
        bus.unsubscribe(PersonIdentified, self._on_person_identified)

        # Cancel any active conversations
        for conv_id, task in list(self._active_conversations.items()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info("Cancelled active conversation: %s", conv_id)
        self._active_conversations.clear()

        self._client = None
        logger.info("BoxBotAgent stopped")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_wake_word(self, event: WakeWordHeard) -> None:
        """Handle a wake word detection — start a voice conversation.

        The initial message is empty because the voice pipeline will
        provide the first utterance after the wake word is confirmed.
        The perception pipeline will identify who is speaking.
        """
        logger.info(
            "Wake word heard (confidence=%.2f), starting voice conversation",
            event.confidence,
        )
        # Determine who is present from the perception state
        person = self._get_most_recent_person()
        await self._start_conversation_task(
            channel="voice",
            initial_message="(wake word detected — listening)",
            person_name=person,
        )

    async def _on_whatsapp_message(self, event: WhatsAppMessage) -> None:
        """Handle an incoming WhatsApp message — start or continue a conversation."""
        logger.info(
            "WhatsApp message from %s, starting conversation",
            event.sender_name,
        )
        text = event.text or ""
        if event.media_type:
            text = f"[{event.media_type} attached] {text}".strip()

        await self._start_conversation_task(
            channel="whatsapp",
            initial_message=text,
            person_name=event.sender_name or None,
            context={
                "sender_phone": event.sender_phone,
                "media_url": event.media_url,
                "media_type": event.media_type,
            },
        )

    async def _on_trigger_fired(self, event: TriggerFired) -> None:
        """Handle a trigger fire — start a trigger-initiated conversation."""
        logger.info(
            "Trigger fired: %s (%s)",
            event.trigger_id,
            event.description,
        )
        # Build the initial message from the trigger's instructions
        initial_msg = (
            f"[Trigger fired: {event.description}]\n"
            f"Instructions: {event.instructions}"
        )
        if event.todo_id:
            initial_msg += f"\nLinked to-do: {event.todo_id}"

        await self._start_conversation_task(
            channel="trigger",
            initial_message=initial_msg,
            person_name=event.for_person,
            context={
                "trigger_id": event.trigger_id,
                "trigger_description": event.description,
                "is_recurring": event.is_recurring,
                "person": event.person,
                "todo_id": event.todo_id,
            },
        )

    async def _on_person_identified(self, event: PersonIdentified) -> None:
        """Update the set of currently-present people."""
        if event.person_name:
            self._present_people[event.person_name] = datetime.now()

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    async def _start_conversation_task(
        self,
        channel: str,
        initial_message: str,
        person_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Start a conversation as a background task.

        Uses a lock to prevent race conditions when multiple events arrive
        at the same time. If a conversation is already active on the same
        channel, the new request is logged but not started (for voice and
        trigger channels). WhatsApp conversations are per-sender, so
        concurrent conversations with different senders are allowed.
        """
        async with self._conversation_lock:
            # Build a unique key per logical conversation
            if channel == "whatsapp" and context:
                conv_key = f"whatsapp:{context.get('sender_phone', 'unknown')}"
            else:
                conv_key = channel

            # Check if there is already an active conversation on this channel
            if conv_key in self._active_conversations:
                existing = self._active_conversations[conv_key]
                if not existing.done():
                    logger.warning(
                        "Conversation already active on %s, skipping new request",
                        conv_key,
                    )
                    return

            conv_id = _generate_conversation_id()
            task = asyncio.create_task(
                self._run_conversation(
                    conversation_id=conv_id,
                    channel=channel,
                    initial_message=initial_message,
                    person_name=person_name,
                    context=context,
                ),
                name=f"conversation-{conv_id}",
            )
            self._active_conversations[conv_key] = task

            # Clean up when the task finishes
            def _cleanup(t: asyncio.Task[None]) -> None:
                self._active_conversations.pop(conv_key, None)

            task.add_done_callback(_cleanup)

    async def _run_conversation(
        self,
        conversation_id: str,
        channel: str,
        initial_message: str,
        person_name: str | None,
        context: dict[str, Any] | None,
    ) -> None:
        """Execute a full conversation: prompt build, agent loop, extraction.

        This is the core conversation lifecycle:
        1. Emit ConversationStarted event
        2. Build the system prompt with all context injections
        3. Run the agent loop (model calls + tool dispatch)
        4. Emit ConversationEnded event
        5. Trigger post-conversation memory extraction
        """
        config = get_config()
        bus = get_event_bus()

        participants = [config.agent.name]
        if person_name:
            participants.append(person_name)

        logger.info(
            "Conversation %s started: channel=%s, person=%s",
            conversation_id,
            channel,
            person_name,
        )

        # 1. Emit start event
        await bus.publish(
            ConversationStarted(
                conversation_id=conversation_id,
                channel=channel,
                person_name=person_name,
                participants=participants,
            )
        )

        messages: list[dict[str, Any]] = []
        turn_count = 0
        summary = ""

        try:
            # 2. Build the system prompt
            system_prompt = await self._build_system_prompt(
                person_name=person_name,
                channel=channel,
                context=context,
            )

            # 3. Inject relevant memories based on who is speaking and what they said
            memory_block = await self._inject_memories(
                person_name=person_name,
                initial_message=initial_message,
            )
            if memory_block:
                system_prompt += f"\n\n{memory_block}"

            # 4. Run the agent loop
            messages, turn_count = await self._agent_loop(
                system_prompt=system_prompt,
                initial_message=initial_message,
                max_turns=config.agent.max_turns,
            )

            # Extract a rough summary from the last assistant message
            summary = self._extract_summary(messages)

        except asyncio.CancelledError:
            logger.info("Conversation %s cancelled", conversation_id)
            summary = "(conversation cancelled)"
            raise
        except Exception:
            logger.exception(
                "Error in conversation %s", conversation_id
            )
            summary = "(conversation ended with error)"
        finally:
            # 5. Emit end event
            await bus.publish(
                ConversationEnded(
                    conversation_id=conversation_id,
                    channel=channel,
                    person_name=person_name,
                    turn_count=turn_count,
                    summary=summary,
                )
            )

            # 6. Trigger post-conversation memory extraction (fire-and-forget)
            if messages and turn_count > 0:
                asyncio.create_task(
                    self._post_conversation(
                        conversation_id=conversation_id,
                        channel=channel,
                        person_name=person_name,
                        messages=messages,
                    ),
                    name=f"extraction-{conversation_id}",
                )

    async def handle_conversation(
        self,
        channel: str,
        initial_message: str,
        person_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Run a full conversation with the agent.

        Public entry point for programmatic conversation initiation
        (e.g., from tests or future subsystems). For event-driven
        conversations, the agent subscribes to events directly.

        Builds the system prompt with context injection, runs the
        agent loop with tools, handles multiple turns, and triggers
        memory extraction at the end.

        Args:
            channel: Conversation channel — "voice", "whatsapp", or "trigger".
            initial_message: The first user message or trigger instructions.
            person_name: Identified speaker name, if known.
            context: Additional context dict (trigger details, sender info, etc.).
        """
        conv_id = _generate_conversation_id()
        await self._run_conversation(
            conversation_id=conv_id,
            channel=channel,
            initial_message=initial_message,
            person_name=person_name,
            context=context,
        )

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    async def _build_system_prompt(
        self,
        person_name: str | None,
        channel: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Build the full system prompt with all context injections.

        The system prompt is assembled from:
        1. Base personality and capability description
        2. System memory (always-loaded household facts / standing instructions)
        3. Current context block (time, day, who is present, channel)
        4. Scheduler status line (to-do count, active trigger count)
        5. Trigger context (if this is a trigger-initiated conversation)

        Memory injection (relevant fact memories) is handled separately
        by _inject_memories() and appended after this method returns.

        Args:
            person_name: The identified speaker, if known.
            channel: Conversation channel.
            context: Additional context (trigger details, etc.).

        Returns:
            The assembled system prompt string.
        """
        config = get_config()

        # 1. Base prompt with agent identity
        prompt = _BASE_SYSTEM_PROMPT.format(
            name=config.agent.name,
            wake_word=config.agent.wake_word,
        )

        # 2. System memory
        system_memory = await self._read_system_memory()
        if system_memory.strip():
            prompt += f"\n## System Memory\n{system_memory}\n"

        # 3. Current context
        now = datetime.now()
        context_lines = [
            f"Current time: {now.strftime('%H:%M')}",
            f"Day: {now.strftime('%A, %B %d, %Y')}",
            f"Channel: {channel}",
        ]

        if person_name:
            context_lines.append(f"Speaking with: {person_name}")

        # Who else is present (from perception, excluding the speaker)
        present = self._get_present_people(exclude=person_name)
        if present:
            context_lines.append(f"Also present: {', '.join(present)}")

        prompt += "\n## Current Context\n" + "\n".join(
            f"- {line}" for line in context_lines
        ) + "\n"

        # 4. Scheduler status
        try:
            status_line = await get_status_line()
            prompt += f"\n{status_line}\n"
        except Exception:
            logger.debug("Could not fetch scheduler status line")

        # 5. Trigger context (for trigger-initiated conversations)
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
                prompt += "\n## Trigger Details\n" + "\n".join(
                    f"- {line}" for line in trigger_lines
                ) + "\n"

        return prompt

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
    ) -> str:
        """Search for relevant memories and format them for prompt injection.

        Uses the shared search backend to find fact memories and recent
        conversations relevant to the current speaker and their first
        utterance.

        Args:
            person_name: The identified speaker, if known.
            initial_message: The first message text, used as search context.

        Returns:
            Formatted memory injection block, or empty string if no
            relevant memories found.
        """
        try:
            return await inject_memories(
                self._memory_store,
                person=person_name,
                utterance=initial_message,
            )
        except Exception:
            logger.exception("Memory injection failed")
            return ""

    # ------------------------------------------------------------------
    # Agent loop (Anthropic SDK)
    # ------------------------------------------------------------------

    async def _agent_loop(
        self,
        system_prompt: str,
        initial_message: str,
        max_turns: int = _DEFAULT_MAX_TURNS,
    ) -> tuple[list[dict[str, Any]], int]:
        """Run the core agent conversation loop.

        Sends the system prompt and initial message to the large model,
        then processes tool calls in a loop until the model produces a
        final text response or the turn limit is reached.

        Args:
            system_prompt: The assembled system prompt.
            initial_message: The first user message.
            max_turns: Maximum number of API round-trips.

        Returns:
            Tuple of (message_history, turn_count). message_history
            contains all messages exchanged including tool results.
        """
        assert self._client is not None, "Agent not started"

        config = get_config()
        model = config.models.large

        # Build tool definitions for the API
        tools = get_tools()
        tool_definitions = self._build_tool_definitions(tools)

        # Initialise the message history
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": initial_message},
        ]

        turn_count = 0

        while turn_count < max_turns:
            turn_count += 1

            try:
                response = await self._client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=messages,
                    tools=tool_definitions,
                )
            except anthropic.APIError as e:
                logger.error(
                    "Anthropic API error on turn %d: %s", turn_count, e
                )
                # Add an error message so extraction still has context
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

            # Check if the model wants to use tools
            if response.stop_reason == "tool_use":
                # Process all tool calls in this response
                tool_results = await self._process_tool_calls(
                    response, tools
                )
                if tool_results:
                    messages.append({
                        "role": "user",
                        "content": tool_results,
                    })
                # Continue the loop for the next model turn
                continue

            # Model produced a final response (stop_reason == "end_turn"
            # or "max_tokens") — conversation turn is complete
            break

        if turn_count >= max_turns:
            logger.warning(
                "Conversation reached max turns (%d)", max_turns
            )

        return messages, turn_count

    def _build_tool_definitions(
        self,
        tools: list[Any],
    ) -> list[dict[str, Any]]:
        """Convert boxBot Tool instances to Anthropic API tool definitions.

        Each tool's name, description, and JSON Schema parameters are
        formatted for the messages API ``tools`` parameter.

        Args:
            tools: List of Tool instances from the registry.

        Returns:
            List of tool definition dicts for the API.
        """
        definitions = []
        for tool in tools:
            definition: dict[str, Any] = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            definitions.append(definition)
        return definitions

    async def _process_tool_calls(
        self,
        response: Any,
        tools: list[Any],
    ) -> list[dict[str, Any]]:
        """Dispatch tool calls from a model response.

        Iterates through all tool_use content blocks in the response,
        looks up the corresponding Tool from the registry, calls its
        execute() method, and collects the results.

        Args:
            response: The Anthropic API response containing tool_use blocks.
            tools: The list of available Tool instances (for reference).

        Returns:
            List of tool_result content blocks to send back to the model.
        """
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
                result_text = json.dumps({
                    "error": f"Unknown tool: {tool_name}",
                })
                logger.warning("Unknown tool requested: %s", tool_name)
            else:
                try:
                    result_text = await tool.execute(**tool_input)
                except Exception as e:
                    logger.exception(
                        "Tool %s execution failed", tool_name
                    )
                    result_text = json.dumps({
                        "error": f"Tool execution failed: {e}",
                    })

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result_text,
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
    ) -> None:
        """Post-conversation cleanup: extract memories and log conversation.

        Runs asynchronously after the conversation ends. Builds a
        transcript from the message history and passes it to the
        extraction pipeline, which produces:
        - A conversation summary (for the conversation log)
        - Extracted fact memories
        - Contradiction invalidations
        - System memory update proposals

        Args:
            conversation_id: The conversation's unique ID.
            channel: The conversation channel.
            person_name: The primary speaker name.
            messages: The full message history from the agent loop.
        """
        try:
            # Build a transcript from the message history
            transcript = self._build_transcript(messages, person_name)

            # Collect IDs of memories that were injected into this conversation
            # (referenced in the system prompt injection block).
            # For now, pass an empty list — full tracking requires parsing
            # the injection block for memory IDs.
            accessed_memory_ids: list[str] = []

            participants = [get_config().agent.name]
            if person_name:
                participants.append(person_name)

            # Run extraction
            result = await extract_memories(
                transcript=transcript,
                accessed_memory_ids=accessed_memory_ids,
                conversation_id=conversation_id,
                channel=channel,
                participants=participants,
            )

            # Apply extraction results to the memory store
            conv_id = await process_extraction_result(
                self._memory_store,
                result,
                conversation_id,
            )

            logger.info(
                "Post-conversation extraction complete for %s (logged as %s)",
                conversation_id,
                conv_id,
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

        Converts the structured message list into a labelled transcript
        suitable for the extraction agent.

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
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text = block.get("text", "")
                                if text:
                                    lines.append(f"[boxBot]: {text}")
                            elif block.get("type") == "tool_use":
                                name = block.get("name", "")
                                lines.append(f"[boxBot used tool: {name}]")

        return "\n".join(lines)

    @staticmethod
    def _extract_summary(messages: list[dict[str, Any]]) -> str:
        """Extract a brief summary from the last assistant text in messages.

        Used for the ConversationEnded event summary field.

        Args:
            messages: The message history.

        Returns:
            A brief summary string (truncated to 200 chars).
        """
        # Walk backwards to find the last text from the assistant
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content[:200]
            if isinstance(content, list):
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            return text[:200]
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
