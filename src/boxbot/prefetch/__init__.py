"""Prefetch layer — a read-only mini-agent that pre-assembles the context
the main agent will likely need into its first turn.

Ships gated behind ``config.prefetch`` and runs in ``shadow`` mode first
(log predictions, inject nothing) until the offline analysis harness
proves precision. See ``docs`` / the plan for the instrument→shadow→
analyze→activate rollout.
"""

from __future__ import annotations

import logging
from typing import Any

from boxbot.prefetch.bundle import PrefetchBundle
from boxbot.prefetch.request import PrefetchRequest
from boxbot.prefetch.runner import PrefetchResult, run_prefetch
from boxbot.prefetch.store import (
    cache_get,
    cache_has_fresh,
    cache_put,
    cache_stamp_conversation,
    record_prefetch_event,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PrefetchBundle",
    "PrefetchRequest",
    "PrefetchResult",
    "run_prefetch",
    "record_prefetch_event",
    "cache_get",
    "cache_put",
    "cache_has_fresh",
    "cache_stamp_conversation",
    "get_prefetch_config",
    "should_prefetch",
    "prefetch_mode",
    "is_active",
    "resolve_client",
]


def get_prefetch_config() -> Any:
    """Return the ``prefetch`` config section, or None if unavailable."""
    try:
        from boxbot.core.config import get_config

        return get_config().prefetch
    except Exception:
        return None


def should_prefetch(channel: str) -> bool:
    """True if prefetch is enabled for this channel (either mode)."""
    cfg = get_prefetch_config()
    if cfg is None or not getattr(cfg, "enabled", False):
        return False
    return channel in getattr(cfg, "channels", [])


def prefetch_mode() -> str:
    """'shadow' or 'active' (defaults to 'shadow')."""
    cfg = get_prefetch_config()
    return getattr(cfg, "mode", "shadow") if cfg else "shadow"


def is_active() -> bool:
    """True when bundles should actually be injected (not just logged)."""
    return prefetch_mode() == "active"


def resolve_client() -> Any | None:
    """Build an Anthropic client for the prefetch mini-agent.

    Peripheral small-model work bills against the API key (same as the
    web_search firewall and memory rerank), never the OAuth subscription
    credit. Returns None if no key is configured.
    """
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic SDK not installed; prefetch disabled")
        return None

    api_key: str | None = None
    try:
        from boxbot.core.config import get_config

        api_key = get_config().api_keys.anthropic
    except Exception:
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        return None
    return anthropic.AsyncAnthropic(api_key=api_key)
