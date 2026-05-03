"""Background poller for in-flight dream-phase batches.

Mirrors :class:`boxbot.memory.batch_poller.BatchPoller` but for the
nightly dream cycle. A single asyncio task that:

1. On start, scans ``pending_dreams`` for rows in ``submitted`` status
   and seeds them for polling. (Unlike extractions, dream batches are
   never in ``queued`` — submission happens inline in
   :func:`run_dream_cycle`.)
2. Periodically calls ``client.messages.batches.retrieve(batch_id)``.
3. When ``processing_status`` transitions to ``ended``, streams the
   JSONL result, parses each ``dedup_decision`` tool call, applies the
   decisions via :func:`apply_dream_result` (respecting the
   ``audit_only`` flag), records cost, marks the row ``applied``, and
   appends the per-cycle decisions to the dream log.
4. Per-batch errors mark the row ``failed`` with the error message.

Wired into the agent lifecycle alongside ``BatchPoller`` so it starts
and stops with the agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anthropic

from boxbot.cost import from_anthropic_usage, record
from boxbot.memory.dream import (
    DEDUP_CONFIDENCE_THRESHOLD,
    DedupDecision,
    NearDupPair,
    _collect_entries,
    _decision_from_payload,
    _dream_log_path,
    _parse_dedup_message,
    apply_dream_result,
)
from boxbot.memory.extraction import DEFAULT_EXTRACTION_MODEL

if TYPE_CHECKING:
    from boxbot.memory.store import MemoryStore

logger = logging.getLogger(__name__)


# Polling cadence — same shape as BatchPoller but slightly slower since
# dream batches are not user-facing and a 5-minute lag on completion is
# fine.
POLL_INTERVAL_INITIAL = 120         # seconds
POLL_INTERVAL_MAX = 600             # 10 minutes
POLL_BACKOFF_FACTOR = 1.5

# After this many seconds in submitted state without ending, mark the
# row failed. Anthropic's hard 24h batch ceiling + headroom.
SUBMITTED_TIMEOUT_SECONDS = 25 * 3600


class DreamPoller:
    """Durable polling loop for dream-phase batches."""

    def __init__(
        self,
        store: MemoryStore,
        client: anthropic.AsyncAnthropic,
        *,
        audit_only: bool = True,
        model: str = DEFAULT_EXTRACTION_MODEL,
    ) -> None:
        self._store = store
        self._client = client
        self._audit_only = audit_only
        self._model = model
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        # Per-batch backoff state: batch_id -> next_check_at (monotonic seconds)
        self._next_check: dict[str, float] = {}
        self._current_interval: dict[str, float] = {}

    async def start(self) -> None:
        """Begin the polling loop. Resumes any in-flight batches first."""
        if self._task is not None:
            return
        self._stop_event.clear()
        await self._resume_pending_on_boot()
        self._task = asyncio.create_task(self._loop(), name="memory-dream-poller")
        logger.info("DreamPoller started (audit_only=%s)", self._audit_only)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=5)
        except asyncio.TimeoutError:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("DreamPoller stopped")

    # -------------------------------------------------------------------
    # Boot resume
    # -------------------------------------------------------------------

    async def _resume_pending_on_boot(self) -> None:
        """Seed initial backoff for submitted dream rows."""
        submitted = await self._store.list_pending_dreams(status="submitted")
        now = asyncio.get_event_loop().time()
        for row in submitted:
            self._next_check[row["batch_id"]] = now
            self._current_interval[row["batch_id"]] = POLL_INTERVAL_INITIAL
        if submitted:
            logger.info(
                "Resuming polling for %d in-flight dream batches",
                len(submitted),
            )

    # -------------------------------------------------------------------
    # Polling loop
    # -------------------------------------------------------------------

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._sweep_once()
            except Exception:
                logger.exception("DreamPoller sweep failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=POLL_INTERVAL_INITIAL,
                )
            except asyncio.TimeoutError:
                pass

    async def _sweep_once(self) -> None:
        rows = await self._store.list_pending_dreams(status="submitted")
        now = asyncio.get_event_loop().time()
        for row in rows:
            batch_id = row["batch_id"]
            if batch_id not in self._next_check:
                self._next_check[batch_id] = now
                self._current_interval[batch_id] = POLL_INTERVAL_INITIAL
            if now < self._next_check[batch_id]:
                continue
            await self._poll_one(row)

    async def _poll_one(self, row: dict) -> None:
        batch_id = row["batch_id"]
        try:
            batch = await self._client.messages.batches.retrieve(batch_id)
        except Exception as e:
            logger.warning(
                "Retrieve failed for dream batch %s: %s", batch_id, e,
            )
            self._reschedule(batch_id)
            return

        status = getattr(batch, "processing_status", None)
        if status == "ended":
            await self._fetch_and_apply(row, batch)
        elif status in {"in_progress", "canceling"}:
            if row.get("submitted_at"):
                submitted = datetime.fromisoformat(row["submitted_at"])
                age = (datetime.utcnow() - submitted).total_seconds()
                if age > SUBMITTED_TIMEOUT_SECONDS:
                    logger.error(
                        "Dream batch %s exceeded %ds without ending; "
                        "marking failed",
                        batch_id, SUBMITTED_TIMEOUT_SECONDS,
                    )
                    await self._store.mark_dream_failed(
                        batch_id,
                        f"timeout after {age:.0f}s in {status}",
                    )
                    self._next_check.pop(batch_id, None)
                    self._current_interval.pop(batch_id, None)
                    return
            self._reschedule(batch_id)
        else:
            logger.warning(
                "Unexpected dream batch status %r for %s", status, batch_id,
            )
            self._reschedule(batch_id)

    def _reschedule(self, batch_id: str) -> None:
        cur = self._current_interval.get(batch_id, POLL_INTERVAL_INITIAL)
        cur = min(cur * POLL_BACKOFF_FACTOR, POLL_INTERVAL_MAX)
        self._current_interval[batch_id] = cur
        self._next_check[batch_id] = asyncio.get_event_loop().time() + cur

    async def _fetch_and_apply(self, row: dict, batch: Any) -> None:
        """Pull JSONL results, apply, mark applied."""
        batch_id = row["batch_id"]
        try:
            results_iter = await self._client.messages.batches.results(batch_id)
        except Exception as e:
            logger.exception(
                "Failed to fetch results for dream batch %s: %s", batch_id, e,
            )
            self._reschedule(batch_id)
            return

        # Materialise the results once: ``apply_dream_result`` needs
        # them to drive merges, and we need them to record real
        # per-message usage for cost_log. The async iterator can only
        # be drained once.
        try:
            entries = await _collect_entries(results_iter)
        except Exception:
            logger.exception(
                "Failed to materialise results for dream batch %s",
                batch_id,
            )
            self._reschedule(batch_id)
            return

        try:
            apply_result = await apply_dream_result(
                self._store,
                entries,
                batch_id=batch_id,
                pairs_by_custom_id={},  # not retained across boots
                audit_only=self._audit_only,
            )
        except Exception:
            logger.exception(
                "Apply failed for dream batch %s; leaving row submitted "
                "for retry on next boot",
                batch_id,
            )
            return

        # Mark applied with a one-line summary.
        summary = (
            f"Dream batch {batch_id}: "
            f"{apply_result.applied_merges} merges (audit_only="
            f"{self._audit_only}), "
            f"{apply_result.skipped_low_confidence} skipped<{DEDUP_CONFIDENCE_THRESHOLD}, "
            f"{apply_result.skipped_unsure_or_distinct} distinct/unsure"
        )
        await self._store.mark_dream_applied(batch_id, summary=summary)
        logger.info(summary)

        # Per-message cost recording (mirrors record_extraction_cost in
        # the non-dream batch poller). One row per successful batch
        # message, using the real Anthropic ``usage`` returned by that
        # message — never a synthetic estimate.
        try:
            await self._record_per_message_costs(batch_id, entries)
        except Exception:
            logger.exception(
                "Cost recording failed for dream batch %s (apply OK)",
                batch_id,
            )

        # Append the decisions to the dream log for that day.
        try:
            self._append_decisions_to_log(batch_id, apply_result.decisions)
        except Exception:
            logger.exception(
                "Dream log append failed for batch %s (apply OK)", batch_id,
            )

        self._next_check.pop(batch_id, None)
        self._current_interval.pop(batch_id, None)

    # -------------------------------------------------------------------
    # Cost + log helpers
    # -------------------------------------------------------------------

    async def _record_per_message_costs(
        self,
        batch_id: str,
        entries: list[Any],
    ) -> None:
        """Record one ``cost_log`` row per successful batch message.

        Reads the real ``usage`` block from each succeeded message and
        delegates pricing to :func:`from_anthropic_usage`. Failed or
        errored messages are skipped (no synthetic fallback). Mirrors
        the per-message shape used by ``record_extraction_cost`` in
        the non-dream batch poller.
        """
        for entry in entries:
            result_obj = (
                getattr(entry, "result", None)
                or (entry.get("result") if isinstance(entry, dict) else None)
            )
            result_type = (
                getattr(result_obj, "type", None)
                or (result_obj.get("type") if isinstance(result_obj, dict) else None)
            )
            if result_type != "succeeded":
                continue
            message = (
                getattr(result_obj, "message", None)
                or (result_obj.get("message") if isinstance(result_obj, dict) else None)
            )
            if message is None:
                continue
            usage = getattr(message, "usage", None)
            if usage is None and isinstance(message, dict):
                usage = message.get("usage")
            if usage is None:
                continue
            decisions = 1 if _parse_dedup_message(message) is not None else 0
            event = from_anthropic_usage(
                purpose="dream",
                model=self._model,
                usage=usage,
                is_batch=True,
                correlation_id=batch_id,
                metadata={"decisions": decisions},
            )
            await record(self._store, event)

    def _append_decisions_to_log(
        self,
        batch_id: str,
        decisions: list[DedupDecision],
    ) -> None:
        """Append a Decisions section to the day's dream log file.

        :func:`run_dream_cycle` writes the candidate/cluster/pair view
        of the cycle at submission time; the poller appends decisions
        once the model returns. This keeps the per-day file as a single
        coherent audit trail.
        """
        path = _dream_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append("")
        lines.append(
            f"## Decisions (applied by DreamPoller, batch `{batch_id}`)"
        )
        if not decisions:
            lines.append("- (no decisions in batch results)")
        else:
            for d in decisions:
                pair_repr = (
                    f"(`{d.pair.memory_id_a[:8]}`, `{d.pair.memory_id_b[:8]}`)"
                    if d.pair else f"(evidence: {d.evidence})"
                )
                note = f' "{d.notes}"' if d.notes else ""
                lines.append(
                    f"- pair {pair_repr}: {d.decision}, "
                    f"confidence {d.confidence:.2f}{note}"
                )
        lines.append("")
        with path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines))
