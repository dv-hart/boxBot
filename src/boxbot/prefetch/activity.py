"""Recent-activity log — deterministic cross-channel conversation recency.

Contextless follow-ups ("I did it, what's next?", a morning text after
last night's thread expired, "did Carina handle it?") carry no
retrieval signal: memory search finds nothing in them and no selector
lane can match candidates against them. Recency IS the ranking here, so
this leg of prefetch is deterministic — no selector, no model call,
three local reads deduped by conversation id, newest first:

1. conversation log (memory DB) — extraction applied; real receipt line
2. ``pending_extractions``     — extraction still in flight; first
                                 user utterance from the retained
                                 transcript stands in for the summary
3. ``ConversationStore``       — open text threads (no extraction row
                                 until their window closes, by design)

Lines attach to the bundle at CONSUME time and are never cached
(recency is perishable — same rule as ``live_context``). A compressed
copy rides the selector briefing (``PrefetchRequest.recent_activity``)
so contextless follow-ups still give the other lanes signal. The agent
pulls full text via ``search_memory(mode="transcript",
conversation_id=…)``; every id rendered here resolves there.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Snippet caps. Lines are injected verbatim on a conversation's first
# turn; keep the whole block around ~150 tokens at the default 5 items.
_ITEM_TEXT_CHARS = 110
_DELIVERY_CHARS = 110

# ``[Label]: text`` transcript lines (see agent._build_transcript and
# conversations.store.render_thread_text).
_LABEL_RE = re.compile(r"^\[([^\]]+)\]:\s*(.+)$")


@dataclass(slots=True)
class _Item:
    conversation_id: str
    channel: str
    who: str
    ts: datetime
    text: str
    source: str  # "summary" | "pending" | "open"
    last_delivery: str | None = None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_ts(value: Any) -> datetime | None:
    """ISO timestamp → aware datetime (naive values are UTC by convention)."""
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _rel_time(ts: datetime, now: datetime) -> str:
    seconds = max(0.0, (now - ts).total_seconds())
    if seconds < 90:
        return "just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = int(seconds // 3600)
    if hours < 24:
        return f"{hours}h ago"
    return f"{int(seconds // 86400)}d ago"


def _who(participants: Iterable[str], agent_name: str) -> str:
    names = [
        p for p in participants
        if p and p.strip() and p.strip().lower() != agent_name.lower()
    ]
    return ", ".join(names) if names else "unknown"


def _is_agent_label(label: str, agent_name: str) -> bool:
    return (
        label == "Tool Result"
        or label.lower().startswith(agent_name.lower())
    )


def _first_user_line(transcript: str, agent_name: str) -> str | None:
    """First human utterance from a ``[Label]: text`` transcript."""
    for raw in transcript.splitlines():
        m = _LABEL_RE.match(raw.strip())
        if not m:
            continue
        label, text = m.group(1), m.group(2).strip()
        if _is_agent_label(label, agent_name) or not text:
            continue
        return text
    return None


def _last_delivery_line(transcript: str, agent_name: str) -> str | None:
    """Last delivered message (``[boxBot → …]: text``) from a transcript."""
    prefix = f"[{agent_name} → "
    for raw in reversed(transcript.splitlines()):
        raw = raw.strip()
        if raw.startswith(prefix):
            _, _, rest = raw.partition("]: ")
            rest = rest.strip()
            if rest:
                return rest
    return None


def _strip_label(text: str, agent_name: str) -> str | None:
    """``[Name]: hi`` → ``hi`` (the line carries who already); agent-
    labelled lines yield None."""
    m = _LABEL_RE.match(text.strip())
    if m is None:
        return text.strip() or None
    if _is_agent_label(m.group(1), agent_name):
        return None
    return m.group(2).strip() or None


def _first_user_text(turns: list[dict[str, Any]], agent_name: str) -> str | None:
    """First human utterance from stored thread turns (wire-form dicts)."""
    for turn in turns:
        if turn.get("role") != "user":
            continue
        content = turn.get("content")
        if isinstance(content, str):
            got = _strip_label(content, agent_name)
            if got:
                return got
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    got = _strip_label(str(block.get("text") or ""), agent_name)
                    if got:
                        return got
    return None


def _last_delivery_text(turns: list[dict[str, Any]]) -> str | None:
    """Last ``message``-tool delivery from stored thread turns."""
    for turn in reversed(turns):
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name") == "message"
            ):
                delivered = str(
                    (block.get("input") or {}).get("content") or ""
                ).strip()
                if delivered:
                    return delivered
    return None


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------


def _render_line(item: _Item, *, now: datetime, include_delivery: bool) -> str:
    open_tag = ", still open" if item.source == "open" else ""
    line = (
        f"- {_rel_time(item.ts, now)} · {item.channel} with {item.who}"
        f"{open_tag}: {item.text} [id {item.conversation_id}]"
    )
    if include_delivery and item.last_delivery:
        line += f'\n  last reply: "{item.last_delivery[:_DELIVERY_CHARS]}"'
    return line


async def gather_recent_activity(
    *,
    memory_store: Any,
    conversation_store: Any = None,
    exclude_ids: Iterable[str] = (),
    limit: int = 5,
    window_hours: float = 48.0,
    agent_name: str = "boxBot",
    now: datetime | None = None,
) -> list[str]:
    """Rendered activity lines, newest first. Never raises.

    ``exclude_ids`` should carry the current conversation's id — its
    thread is already the context. The most recent item also gets the
    last delivered reply ("where we left off"), which answers the
    common "what's next?" resume without a transcript pull.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_hours)
    seen: set[str] = {str(x) for x in exclude_ids}
    items: list[_Item] = []

    # 1. Applied extraction receipts (conversation log).
    if memory_store is not None:
        try:
            for c in await memory_store.list_conversations(
                limit=max(25, limit * 5)
            ):
                ts = _parse_ts(c.started_at)
                summary = (c.summary or "").strip()
                if c.id in seen or ts is None or ts < cutoff or not summary:
                    continue
                seen.add(c.id)
                items.append(_Item(
                    conversation_id=c.id,
                    channel=c.channel,
                    who=_who(c.participants, agent_name),
                    ts=ts,
                    text=summary[:_ITEM_TEXT_CHARS],
                    source="summary",
                ))
        except Exception:
            logger.debug("activity: conversation-log read failed", exc_info=True)

        # 2. Extractions still in flight — snippet stands in for the
        #    summary until the receipt applies.
        try:
            for p in await memory_store.list_pending_extractions(limit=50):
                if p.conversation_id in seen or p.status == "applied":
                    continue
                ts = _parse_ts(p.started_at)
                if ts is None or ts < cutoff:
                    continue
                snippet = _first_user_line(p.transcript or "", agent_name)
                if not snippet:
                    continue
                seen.add(p.conversation_id)
                items.append(_Item(
                    conversation_id=p.conversation_id,
                    channel=p.channel,
                    who=_who(p.participants, agent_name),
                    ts=ts,
                    text=snippet[:_ITEM_TEXT_CHARS],
                    source="pending",
                    last_delivery=_last_delivery_line(
                        p.transcript or "", agent_name
                    ),
                ))
        except Exception:
            logger.debug("activity: pending-extraction read failed", exc_info=True)

    # 3. Open text threads — inside their window, so no extraction row
    #    exists yet by design.
    if conversation_store is not None:
        try:
            records = await conversation_store.list_active(
                max_inactive_seconds=window_hours * 3600,
            )
            for rec in records:
                if rec.conversation_id in seen:
                    continue
                ts = _parse_ts(rec.last_activity_at_iso)
                if ts is None or ts < cutoff:
                    continue
                turns = await conversation_store.get_thread(rec.conversation_id)
                snippet = _first_user_text(turns, agent_name)
                if not snippet:
                    continue
                seen.add(rec.conversation_id)
                items.append(_Item(
                    conversation_id=rec.conversation_id,
                    channel=rec.channel,
                    who=_who(rec.participants, agent_name),
                    ts=ts,
                    text=snippet[:_ITEM_TEXT_CHARS],
                    source="open",
                    last_delivery=_last_delivery_text(turns),
                ))
        except Exception:
            logger.debug("activity: open-thread read failed", exc_info=True)

    items.sort(key=lambda i: i.ts, reverse=True)
    items = items[:limit]

    # "Where we left off" for the newest item when its transcript is
    # still retained (summary-sourced items didn't carry a delivery).
    if (
        items
        and items[0].last_delivery is None
        and memory_store is not None
    ):
        try:
            text = await memory_store.get_transcript(items[0].conversation_id)
            if text:
                items[0].last_delivery = _last_delivery_line(text, agent_name)
        except Exception:
            logger.debug("activity: top-item transcript read failed", exc_info=True)

    return [
        _render_line(item, now=now, include_delivery=(idx == 0))
        for idx, item in enumerate(items)
    ]
