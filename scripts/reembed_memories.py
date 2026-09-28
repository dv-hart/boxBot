#!/usr/bin/env python3
"""Re-embed memory-store rows with the active embedding backend.

Ops action (deliberately not automatic at boot — see
``MemoryStore._check_embedding_model``). Targets rows whose ``embedding``
is NULL by default; ``--all`` re-embeds every row, which is what you want
after switching embedding backends (the provenance marker warns about
exactly that). Finishes by writing ``store_meta.embedding_model``.

Usage (dev):
    PYTHONPATH=src python3 scripts/reembed_memories.py
Usage (panel chroot):
    cd /opt/boxbot && python -m scripts.reembed_memories  # or direct path
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boxbot.core.config import load_config  # noqa: E402
from boxbot.memory.embeddings import active_model, embed_batch  # noqa: E402
from boxbot.memory.store import _embedding_to_blob  # noqa: E402

# Keep each embed_batch call modest so one API hiccup doesn't void a
# huge batch (a failed batch returns all-None and those rows stay NULL).
BATCH_SIZE = 100


def _reembed_table(
    db: sqlite3.Connection,
    table: str,
    text_for_row,
    only_null: bool,
    dry_run: bool,
) -> tuple[int, int]:
    """Re-embed one table; returns (updated, skipped)."""
    where = "WHERE embedding IS NULL" if only_null else ""
    rows = db.execute(f"SELECT * FROM {table} {where}").fetchall()

    updated = skipped = 0
    for start in range(0, len(rows), BATCH_SIZE):
        chunk = rows[start : start + BATCH_SIZE]
        texts, ids = [], []
        for row in chunk:
            text = text_for_row(row)
            if not text or not text.strip():
                skipped += 1
                continue
            texts.append(text)
            ids.append(row["id"])
        if not texts:
            continue
        vectors = embed_batch(texts)
        for row_id, vec in zip(ids, vectors):
            if vec is None:
                skipped += 1
                continue
            if not dry_run:
                db.execute(
                    f"UPDATE {table} SET embedding = ? WHERE id = ?",
                    (_embedding_to_blob(vec), row_id),
                )
            updated += 1
    return updated, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--db", default="data/memory/memory.db")
    parser.add_argument(
        "--all",
        action="store_true",
        help="re-embed every row, not just NULL ones (backend switch)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_config(args.config)
    model = active_model()
    if model is None:
        print(
            "No embedding backend available (no sentence-transformers, no "
            "BOXBOT_MODEL_EMBEDDING / models.embedding). Nothing to do."
        )
        return 1
    print(f"Embedding backend: {model}")

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    only_null = not args.all
    mem_updated, mem_skipped = _reembed_table(
        db,
        "memories",
        lambda r: f"{r['summary']} {r['content']}",
        only_null,
        args.dry_run,
    )
    conv_updated, conv_skipped = _reembed_table(
        db, "conversations", lambda r: r["summary"], only_null, args.dry_run
    )

    if not args.dry_run:
        db.execute(
            "INSERT INTO store_meta (key, value) VALUES ('embedding_model', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (model,),
        )
        db.commit()
    db.close()

    prefix = "[dry-run] would update" if args.dry_run else "updated"
    print(f"memories: {prefix} {mem_updated}, skipped {mem_skipped}")
    print(f"conversations: {prefix} {conv_updated}, skipped {conv_skipped}")
    if not args.dry_run:
        print(f"store_meta.embedding_model = {model}")
    # Skipped rows with a real text mean the embedder failed mid-run.
    return 0


if __name__ == "__main__":
    sys.exit(main())
