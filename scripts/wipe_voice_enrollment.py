#!/usr/bin/env python3
"""Wipe voice enrollment so a fresh re-enrollment can start cleanly.

Deletes every row in ``voice_embeddings`` and clears the
``voice_centroid`` column on every ``centroids`` row. ``persons``,
``visual_embeddings``, and ``visual_centroid`` are left intact.

Use this when the voice centroid has been contaminated — for example
by a NaN embedding poisoning the mean, or by BB's own TTS audio
leaking into the mic during the first ~1-2 s of replies (cold-start
AEC) before the wake-word-gated barge-in landed.

Usage:
    python3 scripts/wipe_voice_enrollment.py [--db PATH] [--yes]

By default targets ``data/perception/perception.db`` relative to the
project root. Pass ``--yes`` to skip the confirmation prompt.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def _summarize(cur: sqlite3.Cursor) -> dict[str, int]:
    """Return a snapshot of current voice-related row counts."""
    counts: dict[str, int] = {}
    counts["voice_embeddings"] = cur.execute(
        "SELECT COUNT(*) FROM voice_embeddings"
    ).fetchone()[0]
    counts["voice_centroids_set"] = cur.execute(
        "SELECT COUNT(*) FROM centroids WHERE voice_centroid IS NOT NULL"
    ).fetchone()[0]
    counts["centroid_rows"] = cur.execute(
        "SELECT COUNT(*) FROM centroids"
    ).fetchone()[0]
    counts["persons"] = cur.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    return counts


def _print_counts(label: str, counts: dict[str, int]) -> None:
    print(f"{label}:")
    for k in (
        "persons",
        "voice_embeddings",
        "voice_centroids_set",
        "centroid_rows",
    ):
        print(f"  {k:<22} = {counts[k]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--db",
        default="data/perception/perception.db",
        help="Path to the perception SQLite database",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"error: database not found: {db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        before = _summarize(cur)
        _print_counts("Before", before)

        if before["voice_embeddings"] == 0 and before["voice_centroids_set"] == 0:
            print("\nNothing to wipe — voice tables are already empty.")
            return 0

        if not args.yes:
            print()
            resp = input(
                "Delete all voice_embeddings and clear voice_centroid? "
                "[y/N] "
            ).strip().lower()
            if resp not in ("y", "yes"):
                print("Aborted.")
                return 1

        cur.execute("DELETE FROM voice_embeddings")
        deleted_embs = cur.rowcount
        cur.execute(
            "UPDATE centroids SET voice_centroid = NULL "
            "WHERE voice_centroid IS NOT NULL"
        )
        cleared_centroids = cur.rowcount
        # Drop centroid rows that have nothing left in them.
        cur.execute(
            "DELETE FROM centroids "
            "WHERE voice_centroid IS NULL AND visual_centroid IS NULL"
        )
        dropped_rows = cur.rowcount
        conn.commit()

        print()
        print(f"Deleted {deleted_embs} voice_embedding row(s).")
        print(f"Cleared voice_centroid on {cleared_centroids} centroid row(s).")
        print(f"Dropped {dropped_rows} now-empty centroid row(s).")
        print()
        _print_counts("After", _summarize(cur))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
