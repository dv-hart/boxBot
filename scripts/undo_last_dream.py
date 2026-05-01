#!/usr/bin/env python3
"""Revert the most recent applied dream-phase batch.

Walks ``pending_dreams`` for the most recent row with ``status='applied'``
and undoes every change keyed to that ``batch_id``:

- Memories with ``consolidated_by=<batch_id>`` and status='invalidated'
  are reactivated; their ``invalidated_by`` and ``superseded_by`` are
  cleared. Their ``consolidated_by`` audit field is also cleared.
- Memories with ``consolidated_by=<batch_id>`` and status='active' (the
  "keepers" of merges) get their ``consolidated_by`` cleared but no
  content rollback — the merged content becomes their new state. The
  spec says soft-delete only; we don't try to reconstruct pre-merge
  content from a snapshot we don't have.
- Memories with ``dream_created_by=<batch_id>`` are dropped (these are
  schema memories created by the cycle; PR1 doesn't create any but
  the column is supported now for forwards compatibility).

The pending_dreams row itself is marked status='failed' with a summary
that points at this script so the audit trail stays coherent.

Idempotent: running twice on the same batch is a no-op the second
time, because after the first run no memories carry the
``consolidated_by=<batch_id>`` stamp.

Usage:
    python3 scripts/undo_last_dream.py [--batch-id BATCH_ID] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


async def _undo(batch_id: str | None, dry_run: bool) -> int:
    # Imports inside the function so ``--help`` works without sqlite IO.
    import boxbot.core  # noqa: F401  break the import cycle
    from boxbot.memory.store import MemoryStore

    store = MemoryStore()
    await store.initialize()
    try:
        if batch_id is None:
            applied = await store.list_pending_dreams(status="applied")
            if not applied:
                print("No applied dream batches found; nothing to undo.")
                return 0
            target = applied[0]  # newest-first
            batch_id = target["batch_id"]
            print(f"Most recent applied dream batch: {batch_id}")
            print(f"  submitted_at: {target['submitted_at']}")
            print(f"  completed_at: {target['completed_at']}")
            print(f"  summary: {target.get('summary')}")
        else:
            target = await store.get_pending_dream(batch_id)
            if target is None:
                print(f"No pending_dreams row found for batch_id={batch_id}",
                      file=sys.stderr)
                return 1

        # Find affected memories.
        consolidated = await store.list_memories_by_dream(
            batch_id, field="consolidated_by",
        )
        dream_created = await store.list_memories_by_dream(
            batch_id, field="dream_created_by",
        )
        print(
            f"Affected memories: {len(consolidated)} consolidated, "
            f"{len(dream_created)} dream-created"
        )

        if not consolidated and not dream_created:
            print(
                "No memories carry this batch's audit stamps — already "
                "undone or never applied. No-op."
            )
            return 0

        if dry_run:
            print("[dry-run] Would reactivate invalidated memories, "
                  "clear consolidated_by, drop dream-created memories.")
            return 0

        # 1. Reactivate any memories that were invalidated by this batch.
        reactivated = await store.reactivate_invalidated_by_dream(batch_id)
        print(f"Reactivated {reactivated} invalidated memories")

        # 2. Clear the consolidated_by stamp on remaining touched rows
        #    (both keepers and just-reactivated losers).
        for mem in consolidated:
            await store.unset_dream_audit_fields(
                mem.id, clear_consolidated_by=True,
            )
        print(f"Cleared consolidated_by on {len(consolidated)} memories")

        # 3. Drop dream-created memories. PR1 creates none, but the
        #    column exists for forwards compat with PR2 (schemas).
        for mem in dream_created:
            await store.delete_memory(mem.id)
        if dream_created:
            print(f"Deleted {len(dream_created)} dream-created memories")

        # 4. Mark the pending_dreams row as failed so the same batch
        #    can't be undone twice with surprise effects.
        await store.mark_dream_failed(
            batch_id,
            f"undone by scripts/undo_last_dream.py",
        )
        print(f"Marked pending_dreams[{batch_id}] status=failed")
        return 0
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Undo the most recent applied dream-phase batch. Reverts "
            "soft-deletes (un-invalidate), clears audit fields, and "
            "drops any dream-created memories."
        ),
    )
    parser.add_argument(
        "--batch-id",
        help=(
            "Specific batch to undo. Defaults to the most recent "
            "applied row."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without mutating the DB.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rc = asyncio.run(_undo(args.batch_id, args.dry_run))
    sys.exit(rc)


if __name__ == "__main__":
    main()
