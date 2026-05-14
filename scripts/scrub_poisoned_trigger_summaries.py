#!/usr/bin/env python3
"""One-shot: scrub stale-state assertions from trigger conversation summaries.

Background. Until the ``_has_human_reply`` fix, every trigger
conversation (morning brief, midday check, evening review) wrongly
ran the full extraction model instead of getting a deterministic
summary. The extraction model, given a routine briefing transcript,
editorialised — writing conversation-log summaries like:

    "...calendar integration still down. BB flagged discrepancy:
     reauth todo may have been closed without resolving the
     underlying issue."

Those summaries then get injected into the NEXT briefing via the
``[Recent Conversations]`` block, so the agent reads "calendar still
down", repeats it, and the next extraction re-asserts it — a
self-perpetuating loop, even though the calendar integration is
actually working.

The code fix stops NEW poisoned summaries. This script cleans the
ones already in the DB so they stop being injected. It only touches
``channel='trigger'`` rows whose summary contains a stale-state
assertion, and rewrites the summary to a terse, factual form (which
also regenerates the embedding via ``update_conversation``).

Usage:
    python3 scripts/scrub_poisoned_trigger_summaries.py            # apply
    python3 scripts/scrub_poisoned_trigger_summaries.py --dry-run  # preview
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Stale-state assertions the extraction model injected into trigger
# summaries. Matches the editorialising patterns seen on the Pi.
STALE_STATE_RE = re.compile(
    r"\b("
    r"calendar (?:integration |feed |source )?(?:still )?"
    r"(?:down|offline|broken|unavailable)"
    r"|pending (?:google )?reauth"
    r"|reauth (?:pending|still outstanding|may have been closed)"
    r"|flagged (?:a )?discrepancy"
    r"|without resolving the underlying issue"
    r")\b",
    re.IGNORECASE,
)


def _terse_summary(channel_desc: str) -> str:
    """A factual, assertion-free replacement summary."""
    return (
        f"{channel_desc} trigger fired; routine briefing delivered. "
        f"(Summary scrubbed — original carried a stale calendar-state "
        f"assertion from the pre-fix extraction loop.)"
    )


def _describe(summary: str) -> str:
    """Best-effort label for the trigger kind, from the old summary."""
    low = summary.lower()
    if "morning brief" in low:
        return "Morning briefing"
    if "evening review" in low:
        return "Evening review"
    if "midday" in low:
        return "Midday check"
    return "Routine"


async def _run(dry_run: bool) -> int:
    import boxbot.core  # noqa: F401
    from boxbot.memory.store import MemoryStore

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("scrub")

    store = MemoryStore()
    await store.initialize()

    cur = await store.db.execute(
        "SELECT id, channel, summary, topics, accessed_memories "
        "FROM conversations WHERE channel = 'trigger'"
    )
    rows = await cur.fetchall()

    import json as _json

    # (id, old_summary, new_summary, topics_list, accessed_memories_list)
    hits: list[tuple[str, str, str, list, list]] = []
    for row in rows:
        summary = row["summary"] or ""
        if not summary.strip():
            continue
        if STALE_STATE_RE.search(summary):
            new_summary = _terse_summary(_describe(summary))
            try:
                topics = _json.loads(row["topics"]) if row["topics"] else []
            except (ValueError, TypeError):
                topics = []
            try:
                accessed = (
                    _json.loads(row["accessed_memories"])
                    if row["accessed_memories"] else []
                )
            except (ValueError, TypeError):
                accessed = []
            hits.append(
                (row["id"], summary, new_summary, topics, accessed)
            )

    log.info("=" * 72)
    log.info("POISONED TRIGGER SUMMARIES")
    log.info("=" * 72)
    for cid, old, new, _topics, _acc in hits:
        log.info(f"  [{cid}]")
        log.info(f"    old: {old}")
        log.info(f"    new: {new}")
        log.info("")
    log.info(f"Total: {len(hits)}")

    if dry_run:
        log.info("(dry-run; no changes written)")
        return 0

    for cid, _old, new, topics, accessed in hits:
        # update_conversation recomputes the embedding from the new
        # summary, so the scrubbed text also stops ranking on the
        # stale-state vector in hybrid search. topics + accessed
        # memories are preserved as-is (only the summary is poison).
        await store.update_conversation(
            cid, summary=new, topics=topics, accessed_memories=accessed,
        )
    log.info(f"Scrubbed {len(hits)} trigger conversation summaries.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing.",
    )
    args = parser.parse_args()
    return asyncio.run(_run(dry_run=args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
