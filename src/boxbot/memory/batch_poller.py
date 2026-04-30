"""Background poller for in-flight Anthropic extraction batches.

A single asyncio task that:

1. On start, scans ``pending_extractions`` for rows in ``queued`` or
   ``submitted`` status. Queued rows from a prior crash are re-submitted;
   submitted rows are picked back up for polling.
2. Periodically (every POLL_INTERVAL seconds, with backoff up to
   POLL_INTERVAL_MAX) calls ``client.messages.batches.retrieve(batch_id)``
   for each submitted row.
3. When a batch's ``processing_status`` transitions to ``ended``,
   streams the JSONL result, parses the structured tool call, applies
   the extraction to the memory store, records cost, and marks the
   pending row as ``applied``.
4. Handles per-request errors (errored / canceled / expired) by marking
   the pending row ``failed`` with the error message. Failed rows are
   not auto-retried — a separate manual or scheduled retry path can
   re-submit them.

Design notes:

- One task, all batches. Anthropic batch API has no webhook, so we poll.
  Single coroutine keeps the implementation simple and avoids fanning
  out N tasks for N pending batches.
- Idempotency: ``mark_pending_applied`` is the last write. If the
  process dies mid-apply, the row stays ``submitted`` and the poller
  re-fetches the result on next boot. ``process_extraction_result``
  creates a NEW conversation row each call, so re-applying would
  duplicate. To prevent that, we check whether the conversation log
  entry already exists for this conv_id before applying — but
  ``process_extraction_result`` generates a fresh conv_id internally
  rather than using the original. That's a known oddity carried over
  from the existing code; we mitigate by setting status to ``applied``
  immediately after the apply call, before any other awaits.
- Cost is recorded after each successful apply.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import anthropic

from boxbot.memory.extraction import (
    DEFAULT_EXTRACTION_MODEL,
    parse_extraction_result,
    process_extraction_result,
    record_extraction_cost,
    submit_extraction_batch,
)

# Deferred to break the agent → batch_poller → extraction → store →
# core.paths → core/__init__ → agent import cycle. MemoryStore /
# PendingExtraction are only used as type annotations here, and
# ``from __future__ import annotations`` postpones their evaluation.
if TYPE_CHECKING:
    from boxbot.memory.store import MemoryStore, PendingExtraction

logger = logging.getLogger(__name__)


# Polling cadence. Most batches finish in <1h; we start tight (60s) so a
# fast batch is picked up promptly, then back off to 5 minutes once the
# batch has been in flight for a while. We jitter slightly to avoid a
# stampede if many batches submitted at once.
POLL_INTERVAL_INITIAL = 60          # seconds
POLL_INTERVAL_MAX = 300             # 5 minutes
POLL_BACKOFF_FACTOR = 1.5

# After this many seconds in submitted state without ending, we mark
# the row failed. 25h covers Anthropic's hard 24h batch ceiling plus
# headroom for clock skew.
SUBMITTED_TIMEOUT_SECONDS = 25 * 3600


class BatchPoller:
    """Durable polling loop for extraction batches."""

    def __init__(
        self,
        store: MemoryStore,
        client: anthropic.AsyncAnthropic,
        *,
        model: str = DEFAULT_EXTRACTION_MODEL,
    ) -> None:
        self._store = store
        self._client = client
        self._model = model
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        # Per-batch backoff state: conv_id -> next_check_at (monotonic seconds)
        self._next_check: dict[str, float] = {}
        self._current_interval: dict[str, float] = {}

    async def start(self) -> None:
        """Begin the polling loop. Resumes any in-flight batches first."""
        if self._task is not None:
            return
        self._stop_event.clear()
        # Resume queued rows (re-submit any that died before submission)
        # and seed initial backoff for submitted rows.
        await self._resume_pending_on_boot()
        self._task = asyncio.create_task(self._loop(), name="memory-batch-poller")
        logger.info("BatchPoller started")

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
        logger.info("BatchPoller stopped")

    # -------------------------------------------------------------------
    # Boot resume
    # -------------------------------------------------------------------

    async def _resume_pending_on_boot(self) -> None:
        """Re-submit queued rows; mark submitted rows for immediate poll."""
        queued = await self._store.list_pending_extractions(status="queued")
        for row in queued:
            await self._try_submit(row)

        submitted = await self._store.list_pending_extractions(status="submitted")
        now = asyncio.get_event_loop().time()
        for row in submitted:
            self._next_check[row.conversation_id] = now
            self._current_interval[row.conversation_id] = POLL_INTERVAL_INITIAL
        if submitted:
            logger.info(
                "Resuming polling for %d in-flight batches", len(submitted),
            )

    # -------------------------------------------------------------------
    # Submission (called both on conversation end and on boot resume)
    # -------------------------------------------------------------------

    async def submit(self, row: PendingExtraction) -> None:
        """Public submission entry point used by the agent on conversation end."""
        await self._try_submit(row)

    async def _try_submit(self, row: PendingExtraction) -> None:
        """Submit (or re-submit) a queued extraction. On failure, leave it
        in queued status with an error annotation; the next boot or a
        future submission attempt can retry."""
        if not row.transcript:
            logger.warning(
                "Cannot submit conversation %s: transcript empty/purged",
                row.conversation_id,
            )
            await self._store.mark_pending_failed(
                row.conversation_id, "transcript missing at submit time"
            )
            return

        # Find the [Active Memories] block by reconstructing from accessed_ids.
        # The agent's injection block is dropped after the conversation
        # ends; we don't have it verbatim here. Build a minimal block
        # from the IDs alone — the model will treat missing summaries
        # as "I was told these were active but cannot see their content
        # in this prompt", which limits invalidation accuracy. The full
        # inject block could be persisted alongside the transcript in a
        # future refinement.
        block = _format_minimal_active_memories(row.accessed_memory_ids)

        try:
            batch_id = await submit_extraction_batch(
                self._client,
                transcript=row.transcript,
                injected_memories_block=block,
                conversation_id=row.conversation_id,
                channel=row.channel,
                participants=row.participants,
                started_at=row.started_at,
                model=self._model,
            )
        except Exception as e:
            logger.exception(
                "Batch submission failed for %s: %s", row.conversation_id, e
            )
            # Leave the row in queued state with an error; next boot will retry.
            return

        await self._store.mark_pending_submitted(row.conversation_id, batch_id)
        now = asyncio.get_event_loop().time()
        self._next_check[row.conversation_id] = now + POLL_INTERVAL_INITIAL
        self._current_interval[row.conversation_id] = POLL_INTERVAL_INITIAL

    # -------------------------------------------------------------------
    # Polling loop
    # -------------------------------------------------------------------

    async def _loop(self) -> None:
        """Main polling loop. Sleeps between sweeps; runs until stopped."""
        while not self._stop_event.is_set():
            try:
                await self._sweep_once()
            except Exception:
                logger.exception("BatchPoller sweep failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=POLL_INTERVAL_INITIAL,
                )
            except asyncio.TimeoutError:
                pass

    async def _sweep_once(self) -> None:
        """Check every submitted batch whose next_check time has passed."""
        rows = await self._store.list_pending_extractions(status="submitted")
        now = asyncio.get_event_loop().time()
        for row in rows:
            # Lazy seed for any row that wasn't tracked yet.
            if row.conversation_id not in self._next_check:
                self._next_check[row.conversation_id] = now
                self._current_interval[row.conversation_id] = POLL_INTERVAL_INITIAL
            if now < self._next_check[row.conversation_id]:
                continue
            await self._poll_one(row)

    async def _poll_one(self, row: PendingExtraction) -> None:
        """Retrieve one batch's status; apply if ended."""
        if not row.batch_id:
            return
        try:
            batch = await self._client.messages.batches.retrieve(row.batch_id)
        except Exception as e:
            logger.warning(
                "Retrieve failed for batch %s (conv %s): %s",
                row.batch_id, row.conversation_id, e,
            )
            self._reschedule(row.conversation_id)
            return

        status = getattr(batch, "processing_status", None)
        if status == "ended":
            await self._fetch_and_apply(row, batch)
        elif status in {"in_progress", "canceling"}:
            # Check submitted-too-long timeout
            if row.submitted_at:
                from datetime import datetime
                submitted = datetime.fromisoformat(row.submitted_at)
                age = (datetime.utcnow() - submitted).total_seconds()
                if age > SUBMITTED_TIMEOUT_SECONDS:
                    logger.error(
                        "Batch %s exceeded %ds without ending; marking failed",
                        row.batch_id, SUBMITTED_TIMEOUT_SECONDS,
                    )
                    await self._store.mark_pending_failed(
                        row.conversation_id,
                        f"timeout after {age:.0f}s in {status}",
                    )
                    self._next_check.pop(row.conversation_id, None)
                    self._current_interval.pop(row.conversation_id, None)
                    return
            self._reschedule(row.conversation_id)
        else:
            logger.warning(
                "Unexpected batch status %r for %s", status, row.batch_id,
            )
            self._reschedule(row.conversation_id)

    def _reschedule(self, conv_id: str) -> None:
        """Bump next_check using exponential backoff up to POLL_INTERVAL_MAX."""
        cur = self._current_interval.get(conv_id, POLL_INTERVAL_INITIAL)
        cur = min(cur * POLL_BACKOFF_FACTOR, POLL_INTERVAL_MAX)
        self._current_interval[conv_id] = cur
        self._next_check[conv_id] = asyncio.get_event_loop().time() + cur

    async def _fetch_and_apply(
        self,
        row: PendingExtraction,
        batch: Any,
    ) -> None:
        """Pull JSONL results, parse our request, apply, mark applied."""
        try:
            results_iter = await self._client.messages.batches.results(row.batch_id)
        except Exception as e:
            logger.exception(
                "Failed to fetch results for batch %s: %s", row.batch_id, e
            )
            self._reschedule(row.conversation_id)
            return

        # The batch contains exactly one request keyed by conv_id.
        target_entry = None
        async for entry in results_iter:
            entry_custom = (
                getattr(entry, "custom_id", None)
                or (entry.get("custom_id") if isinstance(entry, dict) else None)
            )
            if entry_custom == row.conversation_id:
                target_entry = entry
                break

        if target_entry is None:
            logger.error(
                "Batch %s ended but no result for custom_id=%s",
                row.batch_id, row.conversation_id,
            )
            await self._store.mark_pending_failed(
                row.conversation_id, "no result entry for custom_id"
            )
            self._next_check.pop(row.conversation_id, None)
            self._current_interval.pop(row.conversation_id, None)
            return

        result_obj = getattr(target_entry, "result", None)
        if result_obj is None and isinstance(target_entry, dict):
            result_obj = target_entry.get("result")
        result_type = (
            getattr(result_obj, "type", None)
            or (result_obj.get("type") if isinstance(result_obj, dict) else None)
        )

        if result_type != "succeeded":
            err = _extract_error_message(result_obj)
            logger.error(
                "Extraction batch %s for conv %s ended with %s: %s",
                row.batch_id, row.conversation_id, result_type, err,
            )
            await self._store.mark_pending_failed(
                row.conversation_id, f"{result_type}: {err}"
            )
            self._next_check.pop(row.conversation_id, None)
            self._current_interval.pop(row.conversation_id, None)
            return

        message = (
            getattr(result_obj, "message", None)
            or (result_obj.get("message") if isinstance(result_obj, dict) else None)
        )
        try:
            extraction = parse_extraction_result(message)
        except Exception as e:
            logger.exception(
                "Parse failed for conv %s batch %s: %s",
                row.conversation_id, row.batch_id, e,
            )
            await self._store.mark_pending_failed(
                row.conversation_id, f"parse error: {e}"
            )
            self._next_check.pop(row.conversation_id, None)
            self._current_interval.pop(row.conversation_id, None)
            return

        # Apply, then mark applied. The order matters for crash safety —
        # if we crash between apply and mark, we'll re-apply on next
        # boot, but apply only inserts new memories. Acceptable for v1.
        try:
            await process_extraction_result(
                self._store, extraction, row.conversation_id,
            )
        except Exception:
            logger.exception(
                "Apply failed for conv %s; leaving row in submitted "
                "state for retry on next boot",
                row.conversation_id,
            )
            return

        await self._store.mark_pending_applied(row.conversation_id)

        # Cost log (best-effort)
        try:
            usage = getattr(message, "usage", None)
            if usage is not None:
                cost = await record_extraction_cost(
                    self._store,
                    model=self._model,
                    usage=usage,
                    is_batch=True,
                    conversation_id=row.conversation_id,
                    batch_id=row.batch_id,
                )
                logger.info(
                    "Extraction applied for conv %s (cost=$%.5f)",
                    row.conversation_id, cost,
                )
            else:
                logger.info("Extraction applied for conv %s", row.conversation_id)
        except Exception:
            logger.exception(
                "Cost recording failed for conv %s (extraction applied OK)",
                row.conversation_id,
            )

        # Drop bookkeeping for this row.
        self._next_check.pop(row.conversation_id, None)
        self._current_interval.pop(row.conversation_id, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_minimal_active_memories(memory_ids: list[str]) -> str:
    """Render a minimal active-memories block for the extraction prompt.

    We only have IDs (not summaries) at submit time. Future refinement:
    persist the rendered injection block alongside the transcript so
    the model sees full memory text for accurate invalidation.
    """
    if not memory_ids:
        return ""
    lines = ["[Active Memories]"]
    for mid in memory_ids:
        lines.append(f"#{mid[:8]} (content not preserved at submit time)")
    return "\n".join(lines)


def _extract_error_message(result_obj: Any) -> str:
    if result_obj is None:
        return "(no result object)"
    err = getattr(result_obj, "error", None)
    if err is None and isinstance(result_obj, dict):
        err = result_obj.get("error")
    if err is None:
        return "(no error detail)"
    if isinstance(err, dict):
        return json.dumps(err)
    return str(err)
