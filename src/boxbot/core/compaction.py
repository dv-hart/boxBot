"""Token-budgeted rolling compaction for long open threads.

Within one open conversation the whole thread is sent to the model every
generation (see ``agent.py``). Nothing bounds it: a long active thread
grows until the persistent-thread cutoff closes it, and can hit the
model's context limit ungracefully. This keeps an OPEN thread bounded —
when estimated tokens exceed a budget the oldest turns are summarized
into one synthetic note and the recent tail is kept verbatim.

Two-stage on purpose:
- :func:`plan_compaction` is pure — it decides WHICH leading turns to
  evict. This is where the correctness risk lives (never split a
  ``tool_use`` from its ``tool_result``; keep valid role alternation),
  so it is fully unit-testable without a model.
- :func:`compact` is async — it summarizes the evicted turns via the
  small model and returns the new message list. On any summarization
  failure it degrades to a deterministic truncation (drop the head,
  keep the valid tail) — it never crashes the conversation and never
  returns an over-budget thread.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from boxbot.prefetch.bundle import _est_tokens

logger = logging.getLogger(__name__)

Message = dict[str, Any]

# Flat token estimate for an image block. base64 payload is huge but the
# API bills images at roughly this; counting the raw chars would wildly
# overcount. Overcounting is only safe-ish (favours compaction) but this
# keeps the budget gate honest.
_IMAGE_TOKEN_EST = 1600

_SUMMARY_PREFIX = "[Earlier in this conversation — summary]\n"
_OMITTED_NOTE = (
    "[Earlier conversation history omitted to stay within context limits.]"
)

_SUMMARY_SYSTEM = (
    "You compress the earlier part of an ongoing conversation into a "
    "compact, faithful digest so it can be dropped from the live context "
    "without losing what matters. This is internal context for the "
    "assistant, never shown to a user. Preserve: names and who said what, "
    "decisions made, commitments/promises, open questions, unfinished "
    "threads, and any facts the later conversation depends on. Drop "
    "pleasantries and redundancy. Write terse notes, not prose. No "
    "preamble — output only the digest."
)
_SUMMARY_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def _block_tokens(block: Any) -> int:
    if isinstance(block, str):
        return _est_tokens(block)
    if not isinstance(block, dict):
        return 1
    btype = block.get("type")
    if btype == "image":
        return _IMAGE_TOKEN_EST
    if btype == "text":
        return _est_tokens(str(block.get("text") or ""))
    if btype == "tool_use":
        return _est_tokens(json.dumps(block.get("input") or {}, default=str))
    if btype == "tool_result":
        inner = block.get("content")
        if isinstance(inner, list):
            return sum(_block_tokens(b) for b in inner)
        return _est_tokens(str(inner or ""))
    # Unknown block: fall back to a serialized estimate.
    return _est_tokens(json.dumps(block, default=str))


def _message_tokens(msg: Message) -> int:
    # Prefetch bundles ride user turns as metadata and merge into
    # content only at wire-build time (agent._materialize_history) —
    # but they DO ship, every call, so the budget that decides when to
    # compact must count them or the thread silently outgrows the real
    # wire size by ~2-4k tokens per bundle-bearing turn.
    extra = _est_tokens(str(msg.get("prefetch_text") or ""))
    content = msg.get("content")
    if isinstance(content, str):
        return extra + _est_tokens(content)
    if isinstance(content, list):
        return extra + sum(_block_tokens(b) for b in content)
    return extra + 1


def estimate_tokens(messages: list[Message]) -> int:
    """Estimated token count of a message list (chars/4 heuristic)."""
    return sum(_message_tokens(m) for m in messages)


# ---------------------------------------------------------------------------
# Pure eviction planning
# ---------------------------------------------------------------------------


def _is_real_user_turn(msg: Message) -> bool:
    """True for a genuine user turn (not a ``tool_result``-only turn).

    A ``tool_result`` user turn must immediately follow the assistant
    ``tool_use`` turn that produced it, so it is never a safe boundary to
    start a retained tail on — its ``tool_use`` would be orphaned in the
    evicted head. A boundary at a real user turn is always safe: the API
    requires the first message to be ``user``, and a real user turn is
    never preceded by an unmatched ``tool_use`` (a ``tool_use`` turn is
    always followed by its ``tool_result`` turn, not a real user turn).
    """
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, list):
        # Reject ANY user turn carrying a tool_result block — including a
        # MIXED [tool_result, …, text] turn (the inject-don't-interrupt
        # path folds queued speech into the tool_result user turn). Its
        # tool_use lives in the assistant turn just before it, so a
        # boundary here would orphan the tool_result. any(), not all().
        if any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        ):
            return False
    return True


def plan_compaction(
    messages: list[Message],
    *,
    threshold_tokens: int,
    keep_recent_tokens: int,
) -> tuple[list[Message], list[Message]]:
    """Decide which leading turns to evict. Pure — no I/O.

    Returns ``(evicted, retained)``. When under threshold, or when no
    safe split point exists, returns ``([], messages)`` (a no-op).

    The split is chosen at a boundary that keeps every ``tool_use`` /
    ``tool_result`` pair intact and leaves the retained tail a valid
    Anthropic sequence (starts with a real user turn). Among safe
    boundaries the one is picked that keeps the most recent context
    within ``keep_recent_tokens``; if even the smallest safe tail exceeds
    that budget, the most that can be safely evicted is evicted.
    """
    if estimate_tokens(messages) <= threshold_tokens:
        return [], list(messages)

    # Safe boundaries: real user turns after index 0. messages[i:] then
    # starts with a real user turn (valid tail) and messages[:i] is
    # self-contained (every tool_use keeps its tool_result).
    boundaries = [
        i for i in range(1, len(messages)) if _is_real_user_turn(messages[i])
    ]
    if not boundaries:
        return [], list(messages)

    # Ascending i = smaller evicted head / larger retained tail. Pick the
    # smallest i (keep the most) whose tail fits keep_recent_tokens; if
    # none fits, evict the most we safely can (last boundary).
    chosen = boundaries[-1]
    for i in boundaries:
        if estimate_tokens(messages[i:]) <= keep_recent_tokens:
            chosen = i
            break

    return list(messages[:chosen]), list(messages[chosen:])


# ---------------------------------------------------------------------------
# Note injection (keeps role alternation valid)
# ---------------------------------------------------------------------------


def _prepend_note_to_user(msg: Message, note: str) -> Message:
    """Return a copy of a user turn with ``note`` merged into its front.

    ``msg`` MUST be a user turn (guaranteed by :func:`plan_compaction`,
    whose retained tail always starts with a real user turn). Merging —
    rather than inserting a separate user message — keeps role
    alternation valid (no two adjacent user turns). The original dict is
    not mutated.
    """
    content = msg.get("content")
    if isinstance(content, str):
        merged: Any = f"{note}\n\n{content}"
    elif isinstance(content, list):
        merged = [{"type": "text", "text": note}, *content]
    else:
        merged = note
    new_msg = dict(msg)
    new_msg["content"] = merged
    return new_msg


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def _render_transcript(messages: list[Message]) -> str:
    """Flatten evicted turns into a plain-text transcript for the summarizer."""
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role") or "?")
        content = msg.get("content")
        if isinstance(content, str):
            lines.append(f"{role}: {content}")
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                lines.append(f"{role}: {block}")
                continue
            btype = block.get("type")
            if btype == "text":
                lines.append(f"{role}: {block.get('text') or ''}")
            elif btype == "tool_use":
                lines.append(
                    f"{role} [tool_use {block.get('name')}]: "
                    f"{block.get('input')}"
                )
            elif btype == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    texts = [
                        b.get("text", "")
                        for b in inner
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    body = " ".join(t for t in texts if t) or "[non-text result]"
                else:
                    body = str(inner or "")
                lines.append(f"{role} [tool_result]: {body}")
            elif btype == "image":
                lines.append(f"{role} [image]")
    return "\n".join(lines)


async def _summarize(
    messages: list[Message], *, client: Any, model: str
) -> str:
    """Summarize evicted turns via the small model. Raises on failure."""
    transcript = _render_transcript(messages)
    response = await client.messages.create(
        model=model,
        max_tokens=_SUMMARY_MAX_TOKENS,
        system=_SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": transcript}],
    )
    parts: list[str] = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", "") or "")
    return "\n".join(p for p in parts if p).strip()


async def compact(
    messages: list[Message],
    *,
    threshold_tokens: int,
    keep_recent_tokens: int,
    client: Any,
    model: str,
) -> list[Message]:
    """Return a compacted message list: summary note + retained tail.

    No-op (returns a copy of ``messages``) when under threshold or when
    no safe split exists. If summarization errors, degrades to a
    deterministic truncation (drop the head, keep the valid tail behind a
    short "history omitted" note). Never raises, never returns an
    over-budget-by-eviction thread, always a valid Anthropic sequence.
    """
    evicted, retained = plan_compaction(
        messages,
        threshold_tokens=threshold_tokens,
        keep_recent_tokens=keep_recent_tokens,
    )
    if not evicted:
        # Over threshold but nothing safe to evict (e.g. a single giant
        # user turn, or the only safe boundary is the last message). Flag
        # it — otherwise this is a silent failed turn: the over-budget
        # thread 400s and, past the one overflow retry, the generation
        # just drops.
        if estimate_tokens(messages) > threshold_tokens:
            logger.warning(
                "Compaction found no safe split; thread ~%d tok > threshold "
                "%d stays intact (single oversize turn?)",
                estimate_tokens(messages), threshold_tokens,
            )
        return list(messages)

    # Compaction floor: even after eviction the retained tail is a single
    # turn still larger than the budget — nothing more can be split off.
    if len(retained) <= 1 and estimate_tokens(retained) > keep_recent_tokens:
        logger.warning(
            "Compaction hit its floor: retained a single ~%d tok turn > "
            "keep_recent %d — cannot reduce further",
            estimate_tokens(retained), keep_recent_tokens,
        )

    summary = ""
    if client is not None:
        try:
            summary = await _summarize(evicted, client=client, model=model)
        except Exception:
            logger.warning(
                "Compaction summarization failed; falling back to truncation",
                exc_info=True,
            )

    note = _SUMMARY_PREFIX + summary if summary else _OMITTED_NOTE
    merged_head = _prepend_note_to_user(retained[0], note)
    logger.info(
        "Compacted thread: evicted %d turn(s) (~%d tok) → note + %d tail turn(s)"
        " (summarized=%s)",
        len(evicted),
        estimate_tokens(evicted),
        len(retained),
        bool(summary),
    )
    return [merged_head, *retained[1:]]
