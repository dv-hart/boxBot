"""Bounded graceful shutdown.

2026-07-18 and 2026-09-06: the RSS guardrail requested a graceful
shutdown, one stop() never returned, and the process sat as a zombie
that systemd reported ``active`` for days. ``_shutdown_bounded`` must
give up after the deadline so the caller can hard-exit.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from boxbot.core.main import _shutdown_bounded


class _Stoppable:
    def __init__(self, hang: bool = False) -> None:
        self.hang = hang
        self.stopped = False

    async def stop(self) -> None:
        if self.hang:
            await asyncio.Event().wait()  # never set
        self.stopped = True


@pytest.mark.asyncio
async def test_clean_shutdown_returns_true():
    sched = _Stoppable()
    ok = await _shutdown_bounded(
        {"scheduler": sched}, asyncio.get_running_loop(), deadline=1.0
    )
    assert ok is True
    assert sched.stopped


@pytest.mark.asyncio
async def test_hung_stop_is_abandoned_after_deadline():
    hung = _Stoppable(hang=True)
    start = time.monotonic()
    ok = await _shutdown_bounded(
        {"scheduler": hung}, asyncio.get_running_loop(), deadline=0.2
    )
    assert ok is False
    assert time.monotonic() - start < 2.0
