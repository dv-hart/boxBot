#!/usr/bin/env python3
"""One-shot backfill: invalidate routine-delivery log memories.

The extraction agent has been writing "Morning briefing delivered to
Jacob & Carina on 5/12" / "Evening review sent to Jacob on May 10" as
operational memories after every routine trigger firing. The
conversation-log table (``conversations.summary``) already records
the same information, so these are pure noise in the ``memories``
store — they crowd out load-bearing methodology / person facts at
injection time.

**Scope is intentionally narrow.** Earlier drafts also tried to
catch "calendar feed down" / "reauth pending" state assertions, but
that proved unsafe: the working-state memories ("Calendar integration
confirmed working") share keywords with the stale ones, and several
operational memories that LOOK like delivery logs are actually
methodology lessons that got mis-typed. Cleaning those up requires
the persist-injection-block fix (lifecycle plan step 4) so the
extraction model can invalidate them properly with the working
memory as ``superseded_by``. This script only takes the obvious win.

Match criteria — ALL must hold:
1. ``type == 'operational'``
2. Summary matches a tight delivery-log shape: starts with a
   "Morning briefing" / "Evening review" / "Midday check" event
   name followed by a date or "delivered/sent/fired" verb.
3. Summary does NOT contain any methodology / bug / state words
   that signal the memory carries a lesson, not just a log.

Soft-deleted: ``status='invalidated'``, ``invalidated_by`` carries
:data:`BACKFILL_TAG` so the change is auditable / reversible.

Usage:
    python3 scripts/backfill_memory_cleanup.py            # apply
    python3 scripts/backfill_memory_cleanup.py --dry-run  # preview
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


# Marker stamped into invalidated_by so we can tell where the deletion
# came from later (and so the dream phase / future audits can recognise
# this row was operator-initiated, not model-driven).
BACKFILL_TAG = "backfill:2026-05-13:memory-cleanup"

# A pure delivery-log entry: trigger event name, optional date, then
# one of the delivery verbs. The summary must start with the event
# name (^) so we don't catch sentences that merely mention "morning
# briefing" in passing.
ROUTINE_TRIGGER_RE = re.compile(
    r"^(?:on\s+)?"
    r"(morning briefing|evening review|midday check(?:-in)?)"
    r"\b[^.]{0,60}\b"
    r"(delivered|sent|fired|completed|texted)\b",
    re.IGNORECASE,
)

# Words that suggest the memory is a LESSON or BUG OBSERVATION, not a
# pure delivery log. If any of these appear in the summary, we keep
# the memory regardless of what ROUTINE_TRIGGER_RE matched.
KEEP_SIGNALS_RE = re.compile(
    r"\b("
    r"bug|broken|invalid|fix(?:ed)?|fail|error|stuck|"
    r"working|confirmed|resolved|"
    r"lesson|use\s+(?:this|the|bb\.)|"
    r"do\s+not|don't|never|always|"
    r"updated|patched|adjusted|reconfigured|"
    r"recheck|re-check|"
    r"discovered|noticed|observed"
    r")\b",
    re.IGNORECASE,
)


async def _run(dry_run: bool) -> int:
    import boxbot.core  # noqa: F401  break the import cycle
    from boxbot.memory.store import MemoryStore

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("backfill")

    store = MemoryStore()
    await store.initialize()

    cur = await store.db.execute(
        "SELECT id, type, summary, content, status, source_conversation "
        "FROM memories WHERE status = 'active'"
    )
    rows = await cur.fetchall()

    invalidate: list[tuple[str, str]] = []  # (id, summary)
    skipped_lesson: list[tuple[str, str]] = []

    for row in rows:
        mid = row["id"]
        mtype = row["type"]
        summary = row["summary"] or ""

        # Only touch operational memories — methodology/person/household
        # are the load-bearing categories and we must not mistakenly
        # invalidate them with a regex hit.
        if mtype != "operational":
            continue

        # Match a tight delivery-log shape on the *summary only*.
        # Summaries are <80 chars by convention; ranging over content
        # risks catching methodology entries that mention "morning
        # briefing" while teaching a lesson.
        if not ROUTINE_TRIGGER_RE.search(summary):
            continue

        # If the summary also carries lesson/bug signals, keep it.
        if KEEP_SIGNALS_RE.search(summary):
            skipped_lesson.append((mid, summary))
            continue

        invalidate.append((mid, summary))

    log.info("=" * 72)
    log.info("INVALIDATE: routine-delivery log memories")
    log.info("=" * 72)
    for mid, summary in invalidate:
        log.info(f"  [{mid[:8]}] {summary}")
    log.info(f"  -> {len(invalidate)} memories")

    if skipped_lesson:
        log.info("")
        log.info("=" * 72)
        log.info("KEPT (matched delivery shape but carries a lesson signal):")
        log.info("=" * 72)
        for mid, summary in skipped_lesson:
            log.info(f"  [{mid[:8]}] {summary}")
        log.info(f"  -> {len(skipped_lesson)} memories preserved")

    log.info("")
    log.info(f"Total to invalidate: {len(invalidate)}")

    if dry_run:
        log.info("(dry-run; no changes written)")
        return 0

    for mid, _ in invalidate:
        await store.invalidate_memory(mid, invalidated_by=BACKFILL_TAG)
    log.info(f"Invalidated {len(invalidate)} memories with tag '{BACKFILL_TAG}'")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be invalidated without changing anything.",
    )
    args = parser.parse_args()
    return asyncio.run(_run(dry_run=args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
