"""Backfill people labels on photos already in the library.

Re-runs the small-model people inference (the same tagger call as
ingest, see ``intake.py``) over existing photos, using the household
roster and each photo's stored description/tags as context.

Dry-run by default: prints the proposed labels per photo and writes
nothing. ``--apply`` writes them.

  - Cost: one small-model call per photo considered.
  - Idempotent: a re-run replaces earlier tagger labels, never stacks.
  - Human-confirmed photos (``bb.photos.set_people`` /
    ``confirm_people``) are skipped — no call, no write.
  - There is no caption at backfill time, so labels are at best
    ``likely``; review them with ``bb.photos.unconfirmed()``.
  - Descriptions and tags are not changed.

Usage (from the project root, so ``.env`` and ``config/`` resolve):
    python3 -m boxbot.photos.backfill                 # dry run, all photos
    python3 -m boxbot.photos.backfill --limit 5       # dry run, newest 5
    python3 -m boxbot.photos.backfill --photo photo_abc123def456
    python3 -m boxbot.photos.backfill --apply         # write labels
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, TextIO

from boxbot.photos.store import PhotoRecord, PhotoStore

logger = logging.getLogger(__name__)

# (image, photo record, roster) -> inferred people rows
# ({label, person_id, confidence, basis}), or None when the call failed.
PeopleInferrer = Callable[
    [Any, PhotoRecord, list[dict[str, Any]]],
    Awaitable[list[dict[str, Any]] | None],
]


@dataclass
class BackfillSummary:
    considered: int = 0
    proposed: int = 0      # photos with at least one proposed label
    written: int = 0       # photos written (--apply only)
    skipped_confirmed: int = 0
    failed: int = 0        # missing file / unreadable image / tagger failure


def pipeline_inferrer(pipeline: Any) -> PeopleInferrer:
    """The production inferrer: one tagger call via the intake pipeline."""

    async def infer(
        image: Any, record: PhotoRecord, roster: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        notes = record.description or ""
        if record.tags:
            notes += " [tags: " + ", ".join(record.tags) + "]"
        result = await pipeline._generate_tags(
            image, [],
            sender=record.sender,
            roster=roster,
            existing_notes=notes,
        )
        return result.people if result.ok else None

    return infer


def _load_image(path: Path) -> Any:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 50_000_000
    img = Image.open(path)
    img.load()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def _format_people(people: list[dict[str, Any]]) -> list[str]:
    if not people:
        return ["    (no household people)"]
    return [
        f"    {p['label']:<16} {p['confidence']:<8} {p.get('basis') or ''}"
        for p in people
    ]


async def run_backfill(
    store: PhotoStore,
    infer: PeopleInferrer,
    *,
    storage_path: Path,
    apply: bool = False,
    limit: int | None = None,
    photo_ids: list[str] | None = None,
    out: TextIO | None = None,
) -> BackfillSummary:
    """Run people inference over existing photos. See the module docstring."""
    out = out or sys.stdout
    summary = BackfillSummary()

    def emit(line: str = "") -> None:
        print(line, file=out)

    roster = await store.list_people_notes()
    if not roster:
        emit(
            "Household roster is empty — nothing to infer. Add people "
            "first: bb.photos.set_roster(name, note=...)."
        )
        return summary

    if photo_ids:
        records = []
        for pid in photo_ids:
            record = await store.get_photo(pid)
            if record is None or record.deleted_at:
                emit(f"{pid}  not found")
                summary.failed += 1
                continue
            records.append(record)
    else:
        records = await store.list_photos(limit=limit or 100_000)

    emit(
        f"{'APPLY' if apply else 'DRY RUN'}: {len(records)} photo(s), "
        f"roster: {', '.join(e['name'] for e in roster)}"
    )

    for record in records:
        header = f"{record.id}  {(record.created_at or '')[:10]}  " \
                 f"{(record.description or '')[:70]}"
        if record.people_confirmed_at:
            summary.skipped_confirmed += 1
            emit(f"{header}\n    skipped: human-confirmed")
            continue

        summary.considered += 1
        path = storage_path / record.filename
        try:
            image = await asyncio.to_thread(_load_image, path)
        except Exception as e:
            summary.failed += 1
            emit(f"{header}\n    FAILED: cannot read image ({e})")
            continue

        try:
            people = await infer(image, record, roster)
        except Exception as e:
            logger.exception("Backfill: inference failed for %s", record.id)
            people = None
            emit(f"{header}\n    FAILED: tagger error ({e})")
        else:
            if people is None:
                emit(f"{header}\n    FAILED: tagger returned nothing")
        if people is None:
            summary.failed += 1
            continue

        if people:
            summary.proposed += 1
        emit(header)
        for line in _format_people(people):
            emit(line)

        if apply and await store.apply_inferred_people(record.id, people):
            summary.written += 1

    emit()
    emit(
        f"{summary.considered} considered, {summary.proposed} with "
        f"proposed people, {summary.skipped_confirmed} skipped "
        f"(human-confirmed), {summary.failed} failed, "
        + (
            f"{summary.written} written."
            if apply
            else "nothing written (dry run; pass --apply to write)."
        )
    )
    return summary


async def _main_async(args: argparse.Namespace) -> int:
    from boxbot.core.config import get_config
    from boxbot.photos.intake import IntakePipeline

    config = get_config()
    store = PhotoStore(db_path=args.db) if args.db else PhotoStore()
    await store.initialize()
    try:
        pipeline = IntakePipeline(store, infer_people=True)
        if pipeline._get_client() is None:
            print(
                "No Anthropic API key configured (ANTHROPIC_API_KEY) — "
                "the tagger cannot run.",
                file=sys.stderr,
            )
            return 2
        summary = await run_backfill(
            store,
            pipeline_inferrer(pipeline),
            storage_path=Path(config.photos.storage_path),
            apply=args.apply,
            limit=args.limit,
            photo_ids=args.photo or None,
        )
    finally:
        await store.close()
    return 1 if summary.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m boxbot.photos.backfill",
        description="Infer household people on existing photos "
        "(dry run unless --apply).",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="write the proposed labels (default: print only)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only the newest N photos",
    )
    parser.add_argument(
        "--photo", action="append", metavar="PHOTO_ID",
        help="only this photo (repeatable)",
    )
    parser.add_argument("--config", default=None, help="config YAML path")
    parser.add_argument("--env", default=".env", help=".env path")
    parser.add_argument("--db", default=None, help="photos.db path override")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    from boxbot.core.config import load_config
    from boxbot.core.main import _load_dotenv

    _load_dotenv(args.env)
    load_config(args.config)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
