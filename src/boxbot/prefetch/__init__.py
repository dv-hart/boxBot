"""Prefetch layer — parallel one-shot selector calls (one per context
source) that pre-assemble the context the main agent will likely need
into its first turn.

Two deterministic legs ride alongside the selector fan-out: live-context
providers (``providers.py``, fresh device state) and the recent-activity
log (``activity.py``, cross-channel conversation recency). Neither costs
a model call and neither is ever cached.

Ships gated behind ``config.prefetch`` and runs in ``shadow`` mode first
(log predictions, inject nothing) until the offline analysis harness
proves precision. See ``docs`` / the plan for the instrument→shadow→
analyze→activate rollout.
"""

from __future__ import annotations

import logging
from typing import Any

from boxbot.prefetch.activity import gather_recent_activity
from boxbot.prefetch.bundle import PrefetchBundle
from boxbot.prefetch.hot import HotTaskCache
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
    "HotTaskCache",
    "gather_recent_activity",
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
    """True when bundles should actually be injected (not just logged).

    Requires ``prefetch.enabled`` — ``mode: active`` with the layer
    switched off must inject nothing (and skip the activity/cache work).
    """
    cfg = get_prefetch_config()
    if cfg is None or not getattr(cfg, "enabled", False):
        return False
    return prefetch_mode() == "active"


# One handle per (provider, model, endpoint) for the process lifetime:
# an AsyncOpenAI/AsyncAnthropic client owns an httpx pool, and building
# a fresh one per fan-out paid a cold TLS handshake on every first turn.
_handle_cache: dict[tuple[str, str, str | None], Any] = {}


def resolve_client() -> Any | None:
    """Return the selector-model handle for the prefetch fan-out (cached).

    See :func:`_build_client` for resolution rules. The built handle is
    cached by (provider, model, api_base); a config reload that changes
    any of them yields a new handle.
    """
    from boxbot.core.models import provider_for_model
    from boxbot.prefetch.runner import _resolve_model

    model = _resolve_model(get_prefetch_config())
    provider = provider_for_model(model)
    api_base = None
    try:
        from boxbot.core.config import get_config

        api_base = get_config().openai.api_base if provider == "openai" else None
    except Exception:
        api_base = None
    key = (provider, model, api_base)
    handle = _handle_cache.get(key)
    if handle is None:
        handle = _build_client()
        if handle is not None:
            _handle_cache[key] = handle
    return handle


def _build_client() -> Any | None:
    """Build the selector-model handle for the prefetch fan-out.

    The model resolves ``prefetch.model`` → ``models.fast`` (the latency
    tier) → ``models.small``; the provider follows
    from the id (``boxbot.core.models``). Peripheral fast/small-model
    work bills against API keys (same as the web_search firewall and
    memory rerank), never the OAuth subscription credit. Returns None —
    prefetch disabled — when the resolved provider's key or SDK is
    missing.
    """
    import os

    from boxbot.core.models import provider_for_model
    from boxbot.prefetch.runner import LLMHandle, _resolve_model

    model = _resolve_model(get_prefetch_config())
    provider = provider_for_model(model)

    if provider == "openai":
        try:
            import openai
        except ImportError:
            logger.error("openai SDK not installed; prefetch disabled")
            return None
        api_key: str | None = None
        oa_cfg = None
        try:
            from boxbot.core.config import get_config

            api_key = get_config().api_keys.openai
            oa_cfg = get_config().openai
        except Exception:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "prefetch resolved to %s but OPENAI_API_KEY is not set; "
                "prefetch disabled", model,
            )
            return None
        if oa_cfg is not None and oa_cfg.is_azure:
            if not (oa_cfg.api_base and oa_cfg.api_version):
                logger.warning(
                    "prefetch resolved to Azure OpenAI but OPENAI_API_BASE/"
                    "OPENAI_API_VERSION is missing; prefetch disabled",
                )
                return None
            client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=oa_cfg.api_base,
                api_version=oa_cfg.api_version,
            )
        else:
            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=(oa_cfg.api_base or None) if oa_cfg else None,
            )
        return LLMHandle(provider="openai", model=model, client=client)

    try:
        import anthropic
    except ImportError:
        logger.error("anthropic SDK not installed; prefetch disabled")
        return None
    api_key = None
    try:
        from boxbot.core.config import get_config

        api_key = get_config().api_keys.anthropic
    except Exception:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return LLMHandle(
        provider="anthropic",
        model=model,
        client=anthropic.AsyncAnthropic(api_key=api_key),
    )
