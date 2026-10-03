"""Live-context providers: fresh device state for bundles and triggers.

A provider is a named async callable returning ONE rendered line of
current state (or None when it has nothing to say). Hot-task bundles
declare provider names (``HotTaskConfig.providers``); the hot lookup
resolves them at consume time so the cached bundle stays static while
the injected context carries live data.

Two rules keep this off the critical path:

- **Main-process only.** Providers may read local in-process or
  loopback state (~100-200ms) but never the sandbox or the network —
  the voice reply path cannot wait on either.
- **Budgeted, omit on miss.** ``resolve_live`` bounds the whole batch;
  a slow or failing provider contributes nothing rather than latency.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Voice-path budget: generous for a local state read, far below
# anything a user would perceive.
DEFAULT_BUDGET_SECONDS = 0.3

Provider = Callable[[], Awaitable[str | None]]

_PROVIDERS: dict[str, Provider] = {}


def register_provider(name: str, fn: Provider) -> None:
    _PROVIDERS[name] = fn


async def resolve_live(
    names: list[str] | tuple[str, ...],
    *,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
) -> list[str]:
    """Resolve providers concurrently; return rendered lines.

    Unknown names, failures, Nones, and anything still pending at the
    budget are silently omitted — a bundle without live state is valid,
    a turn delayed by a wedged bridge is not.
    """
    fns = [(n, _PROVIDERS.get(n)) for n in names]
    known = [(n, fn) for n, fn in fns if fn is not None]
    for n, fn in fns:
        if fn is None:
            logger.debug("live provider %r not registered", n)
    if not known:
        return []

    async def _one(name: str, fn: Provider) -> str | None:
        try:
            return await fn()
        except Exception:
            logger.debug("live provider %r failed", name, exc_info=True)
            return None

    tasks = [asyncio.create_task(_one(n, fn)) for n, fn in known]
    done, pending = await asyncio.wait(tasks, timeout=budget_seconds)
    for t in pending:
        t.cancel()
    if pending:
        logger.info(
            "live providers over budget (%.0fms): %d of %d dropped",
            budget_seconds * 1000, len(pending), len(tasks),
        )
    # Preserve declaration order.
    lines: list[str] = []
    for t in tasks:
        if t in done:
            line = t.result()
            if line:
                lines.append(line)
    return lines
