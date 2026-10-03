"""Input to one prefetch run.

Built at the seam (an inbound text message, or a scheduled trigger
about to fire) and handed to :func:`boxbot.prefetch.runner.run_prefetch`.

The briefing is deliberately tiny (~200 tokens hard-capped): it rides
in EVERY selector lane's prompt, so each extra char is multiplied by
the number of lanes. The thread tail keeps only what a human said and
what the assistant actually delivered via the ``message`` tool — the
assistant's private internal-notes JSON never enters a selector prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ~200 tokens at ~4 chars/token. The utterance is kept whole up to its
# own cap; the tail is trimmed oldest-first to fit.
_BRIEFING_MAX_CHARS = 850
_CONTENT_MAX_CHARS = 400
_TAIL_MAX_TURNS = 4
_TAIL_LINE_MAX_CHARS = 160
_ACTIVITY_MAX_LINES = 2
_ACTIVITY_LINE_MAX_CHARS = 140

# Conversation-id tags are for the main agent's transcript pulls;
# selectors can't use them — strip to save briefing chars × lanes.
_ACTIVITY_ID_RE = re.compile(r"\s*\[id [^\]]+\]")


@dataclass(slots=True)
class PrefetchRequest:
    """What the main agent is about to handle.

    ``key`` is the join key for telemetry: the conversation_id for text,
    or the trigger_id for scheduled triggers (the conversation_id is not
    minted until fire time). ``key_kind`` disambiguates the two.
    """

    key: str
    key_kind: str  # "conversation" | "trigger"
    channel: str
    person: str | None = None
    # The utterance (text) or the trigger description+instructions.
    text: str | None = None
    # Detailed notes for a linked to-do (scheduled path), if any.
    todo_notes: str | None = None
    # Tail of the persistent thread (text path), most-recent last.
    recent_thread_tail: list[dict[str, Any]] | None = field(default=None)
    # Names already in the main agent's context this conversation
    # (loaded skills, injected bb modules, injected memory ids) so
    # selectors don't re-pick them.
    already_loaded: list[str] | None = None
    # Recent-activity lines (prefetch/activity.py). Compressed into the
    # briefing so a contextless follow-up ("I did it, what's next?")
    # still gives every lane the prior conversation's topic. Present on
    # first turns only — later turns carry the thread tail instead.
    recent_activity: list[str] | None = None
    # Utterance embedding, filled by the hot-task matcher when it runs
    # first so the memory lane doesn't embed the same text twice.
    query_embedding: Any = field(default=None, repr=False)

    def briefing(self) -> str:
        """A compact natural-language brief shared by every selector."""
        lines: list[str] = []
        who = self.person or "an unknown person"
        if self.key_kind == "trigger":
            lines.append(
                f"A scheduled trigger is about to fire (for {who}, "
                f"channel={self.channel})."
            )
        else:
            lines.append(
                f"An inbound {self.channel} message just arrived from {who}."
            )
        if self.text:
            lines.append(f"Content:\n{self.text.strip()[:_CONTENT_MAX_CHARS]}")
        if self.todo_notes:
            lines.append(
                f"Linked to-do notes:\n{self.todo_notes.strip()[:_CONTENT_MAX_CHARS]}"
            )
        if self.recent_activity:
            acts = []
            for line in self.recent_activity[:_ACTIVITY_MAX_LINES]:
                # First physical line only (drops the "last reply"
                # continuation), id tag stripped.
                head = _ACTIVITY_ID_RE.sub("", line.splitlines()[0]).strip()
                if head:
                    acts.append(head[:_ACTIVITY_LINE_MAX_CHARS])
            if acts:
                lines.append("Recent prior conversations:\n" + "\n".join(acts))
        if self.already_loaded:
            lines.append(
                "Already in the assistant's context (do not re-select): "
                + ", ".join(sorted(set(self.already_loaded)))
            )
        tail_lines = _tail_lines(self.recent_thread_tail or [])
        base = "\n\n".join(lines)
        # Fit the tail into what's left of the cap, dropping oldest first.
        remaining = _BRIEFING_MAX_CHARS - len(base)
        kept: list[str] = []
        for line in reversed(tail_lines):
            cost = len(line) + 1
            if remaining - cost < 40:  # leave room for the section header
                break
            kept.insert(0, line)
            remaining -= cost
        if kept:
            base += "\n\nRecent exchange:\n" + "\n".join(kept)
        return base


def _tail_lines(turns: list[dict[str, Any]]) -> list[str]:
    """Render the conversation tail as ``role: text`` lines.

    Keeps only user utterances (plain strings) and assistant deliveries
    (``message``-tool calls). Assistant ``text`` blocks are the model's
    private internal notes and are excluded on purpose.
    """
    out: list[str] = []
    for turn in turns:
        role = str(turn.get("role") or "?")
        content = turn.get("content")
        if role == "user" and isinstance(content, str):
            text = content.strip()
            if text:
                out.append(f"user: {text[:_TAIL_LINE_MAX_CHARS]}")
        elif role == "assistant" and isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name") == "message"
                ):
                    delivered = str(
                        (block.get("input") or {}).get("content") or ""
                    ).strip()
                    if delivered:
                        out.append(
                            f"assistant: {delivered[:_TAIL_LINE_MAX_CHARS]}"
                        )
    return out[-_TAIL_MAX_TURNS:]
