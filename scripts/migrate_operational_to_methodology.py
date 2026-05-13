#!/usr/bin/env python3
"""Migrate remaining ``operational`` memories — lifecycle step 7.

The ``operational`` type was a catch-all that the extraction agent
filled with three different kinds of content:

1. **Workspace-artifact pointers** ("Weekly weight log at
   workspace/data/weight.csv"). These are durable methodology — the
   pattern *where to look* is the lesson.
2. **Durable lessons / configuration history** ("Cron triggers fixed
   to PDT offsets on 2026-04-30; DST re-check needed Nov 2026"). The
   recheck note is forward-looking; the OAuth-token-shape lesson
   generalizes. These are methodology.
3. **Activity logs and stale state assertions** ("Morning briefing
   delivered to Jacob on 5/12"). These are noise.

This script triages every active ``operational`` memory into one of
three buckets:

- **promote** → re-type to ``methodology``. Workspace pointers or
  rows whose summary carries a generalisable-lesson signal.
- **invalidate** → soft-delete with the migration tag. Rows whose
  summary matches the routine-delivery shape (these slipped past the
  backfill from step 2 because of the lesson-signal veto).
- **keep-as-is** → typed ``operational`` but neither promoted nor
  invalidated. The type-weighted injection (step 5) gives operational
  a budget of 0 so they don't surface anyway. Future iterations or a
  manual sweep can address them.

Run with ``--dry-run`` to inspect classifications without writing.
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


MIGRATION_TAG = "step7-migration:2026-05-13:drop-operational"

# A summary like "Weekly weight log at workspace/data/weight.csv" or
# "Saved Q1 report to workspace/notes/jacob/q1.md".
WORKSPACE_POINTER_RE = re.compile(
    r"workspace/[\w/.-]+",
    re.IGNORECASE,
)

# Signals that the operational row carries a durable lesson worth
# preserving as methodology. Tight by design — we'd rather leave
# something as type=operational (budget=0, invisible) than mis-promote
# a stale bug observation into a recurring injection.
#
# These patterns are the ones that proved themselves on the Pi corpus:
# - "use bb.X / use the Y": SDK-shape methodology
# - "DST", "recheck", "shift in november": calendar/cron forward-looking
# - "do not / don't / never / always": user-stated standing rules
# - explicit "no <thing>" SDK-shape lessons: "no vision API in SDK"
LESSON_SIGNAL_RE = re.compile(
    r"\b("
    r"use\s+bb\.|use\s+the\s+|"
    r"recheck|re-check|dst|shift\s+in\s+november|"
    r"do\s+not\b|don't|never\b|always\b|"
    r"no\s+(?:vision|onboarding|generator|api|module)\s+(?:in|for)\b"
    r")\b",
    re.IGNORECASE,
)

# Routine-delivery shape (same as step-2 backfill).
ROUTINE_TRIGGER_RE = re.compile(
    r"^(?:on\s+)?"
    r"(morning briefing|evening review|midday check(?:-in)?)"
    r"\b[^.]{0,60}\b"
    r"(delivered|sent|fired|completed|texted)\b",
    re.IGNORECASE,
)

# "X confirmed working" / "stale Y closed" — one-shot status updates
# the agent wrote after fixing something. The fact "X works" is the
# default state, not a memory-worthy fact; the conversation log
# captures when the fix happened.
RESOLUTION_UPDATE_RE = re.compile(
    r"\b(confirmed working|stale .* closed|todos closed|"
    r"reauth .* closed|integration is healthy|"
    r"marked complete)\b",
    re.IGNORECASE,
)


def _classify(summary: str, content: str) -> str:
    """Return one of 'promote', 'invalidate', 'keep'.

    Order matters:
    1. Workspace pointers always promote — strongest signal.
    2. Resolution-update entries ("confirmed working", "todos closed")
       invalidate — they're one-shot status updates, not durable facts.
       Checked BEFORE lesson signal because some resolution messages
       happen to contain methodology-shaped phrases.
    3. Routine-delivery logs invalidate.
    4. Lesson signals promote.
    5. Everything else keeps.
    """
    if WORKSPACE_POINTER_RE.search(summary) or WORKSPACE_POINTER_RE.search(content):
        return "promote"
    if RESOLUTION_UPDATE_RE.search(summary) or RESOLUTION_UPDATE_RE.search(content):
        return "invalidate"
    if ROUTINE_TRIGGER_RE.search(summary):
        return "invalidate"
    if LESSON_SIGNAL_RE.search(summary) or LESSON_SIGNAL_RE.search(content):
        return "promote"
    return "keep"


async def _run(dry_run: bool) -> int:
    import boxbot.core  # noqa: F401
    from boxbot.memory.store import MemoryStore

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("step7")

    store = MemoryStore()
    await store.initialize()

    cur = await store.db.execute(
        "SELECT id, type, summary, content FROM memories "
        "WHERE type = 'operational' AND status = 'active'"
    )
    rows = await cur.fetchall()

    promote: list[tuple[str, str]] = []
    invalidate: list[tuple[str, str]] = []
    keep: list[tuple[str, str]] = []

    for row in rows:
        mid = row["id"]
        summary = row["summary"] or ""
        content = row["content"] or ""
        verdict = _classify(summary, content)
        if verdict == "promote":
            promote.append((mid, summary))
        elif verdict == "invalidate":
            invalidate.append((mid, summary))
        else:
            keep.append((mid, summary))

    log.info("=" * 72)
    log.info("PROMOTE → methodology")
    log.info("=" * 72)
    for mid, summary in promote:
        log.info(f"  [{mid[:8]}] {summary}")
    log.info(f"  -> {len(promote)} memories")

    log.info("")
    log.info("=" * 72)
    log.info("INVALIDATE (routine-delivery logs that slipped past step-2 veto)")
    log.info("=" * 72)
    for mid, summary in invalidate:
        log.info(f"  [{mid[:8]}] {summary}")
    log.info(f"  -> {len(invalidate)} memories")

    log.info("")
    log.info("=" * 72)
    log.info("KEEP as-is (type=operational, budget=0 in step-5 injection)")
    log.info("=" * 72)
    for mid, summary in keep:
        log.info(f"  [{mid[:8]}] {summary}")
    log.info(f"  -> {len(keep)} memories")

    log.info("")
    log.info(
        f"Totals: promote={len(promote)}, "
        f"invalidate={len(invalidate)}, "
        f"keep={len(keep)}"
    )

    if dry_run:
        log.info("(dry-run; no changes written)")
        return 0

    # Promote: re-type operational → methodology in place.
    for mid, _ in promote:
        await store.db.execute(
            "UPDATE memories SET type = 'methodology' WHERE id = ?",
            (mid,),
        )

    # Invalidate: soft-delete with the migration tag.
    for mid, _ in invalidate:
        await store.invalidate_memory(mid, invalidated_by=MIGRATION_TAG)

    await store.db.commit()
    log.info(
        f"Applied: promoted {len(promote)} → methodology, "
        f"invalidated {len(invalidate)} with tag '{MIGRATION_TAG}'"
    )
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
