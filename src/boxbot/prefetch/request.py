"""Input to one prefetch run.

Built at the seam (an inbound text message, or a scheduled trigger
about to fire) and handed to :func:`boxbot.prefetch.runner.run_prefetch`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    def briefing(self) -> str:
        """A compact natural-language brief for the mini-agent's first turn."""
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
            lines.append(f"Content:\n{self.text.strip()}")
        if self.todo_notes:
            lines.append(f"Linked to-do notes:\n{self.todo_notes.strip()}")
        if self.recent_thread_tail:
            tail = _format_thread_tail(self.recent_thread_tail)
            if tail:
                lines.append(f"Recent message history:\n{tail}")
        return "\n\n".join(lines)


def _format_thread_tail(turns: list[dict[str, Any]], *, max_turns: int = 6) -> str:
    """Render the last few turns as `role: text` lines (bounded)."""
    out: list[str] = []
    for turn in turns[-max_turns:]:
        role = str(turn.get("role") or "?")
        content = turn.get("content")
        text = _content_to_text(content)
        if text:
            out.append(f"{role}: {text[:300]}")
    return "\n".join(out)


def _content_to_text(content: Any) -> str:
    """Flatten Anthropic message content (str or block list) to plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return " ".join(p for p in parts if p).strip()
    return ""
