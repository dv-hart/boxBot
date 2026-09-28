"""The prefetch selector fan-out.

One request → N parallel ONE-SHOT selector calls, one per context
source with non-empty candidates (see ``prefetch/sources.py``). Each
selector receives the shared briefing plus its own candidate corpus and
returns a schema-validated selection; there is no tool loop and no
multi-turn anything. Wall clock ≈ one fast-model round trip regardless
of how many sources are enabled.

The selector model resolves ``config.prefetch.model`` →
``models.fast`` (the latency tier, e.g. ``gpt-5.6-luna``) →
``models.small``. Provider routing follows from the model id
(``boxbot.core.models``): OpenAI ids go through Chat Completions with a
strict ``response_format``; Anthropic ids use a forced tool call. Both
are single round trips.

Runs in the main process, reuses the caller's ``MemoryStore``, and
spends only ``purpose="prefetch"`` cost rows.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from boxbot.cost import from_anthropic_usage, from_openai_usage, record as record_cost
from boxbot.prefetch.bundle import PrefetchBundle
from boxbot.prefetch.request import PrefetchRequest
from boxbot.prefetch.sources import SourceRun, gather_sources

logger = logging.getLogger(__name__)

_MAX_SELECTOR_TOKENS = 400

SYSTEM_PROMPT = """\
You are boxBot's context prefetcher ({source} lane). A larger assistant \
is about to handle the inbound message described by the briefing. From \
the candidates below, select ONLY what that assistant is very likely to \
need on its FIRST turn.

Rules:
- Precision over recall. Selecting NOTHING is the normal outcome; an \
included item the assistant doesn't use is a failure.
- Only pick names/ids that appear verbatim in the candidates. Never \
invent one.
- {instructions}"""


@dataclass(slots=True)
class LLMHandle:
    """A resolved selector endpoint: provider + model + async client."""

    provider: str  # "anthropic" | "openai"
    model: str
    client: Any


@dataclass(slots=True)
class PrefetchResult:
    bundle: PrefetchBundle
    iterations: int  # number of selector calls made
    cost_usd: float


def _resolve_model(config: Any) -> str:
    """``prefetch.model`` → ``models.fast`` → ``models.small``."""
    model = getattr(config, "model", None)
    if model:
        return model
    try:
        from boxbot.core.config import get_config

        models = get_config().models
        if models.fast:
            return models.fast
        if models.small:
            return models.small
    except Exception:
        pass
    return "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# One-shot selection (per provider)
# ---------------------------------------------------------------------------


async def _select_openai(
    handle: LLMHandle, source: SourceRun, briefing: str
) -> tuple[dict[str, Any] | None, Any]:
    """One Chat Completions call with a strict response_format."""
    from boxbot.core.agent_openai_adapter import build_response_format
    from boxbot.core.models import reasoning_effort_for_model

    kwargs: dict[str, Any] = {
        "model": handle.model,
        "max_completion_tokens": _MAX_SELECTOR_TOKENS,
        "response_format": build_response_format(
            source.schema, f"prefetch_{source.name}"
        ),
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    source=source.name, instructions=source.instructions
                ),
            },
            {
                "role": "user",
                "content": f"{briefing}\n\nCandidates:\n{source.candidates}",
            },
        ],
    }
    effort = reasoning_effort_for_model(handle.model)
    if effort is not None:
        kwargs["reasoning_effort"] = effort
    completion = await handle.client.chat.completions.create(**kwargs)
    usage = getattr(completion, "usage", None)
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return None, usage
    text = getattr(getattr(choices[0], "message", None), "content", None)
    if not text:
        return None, usage
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        logger.debug("prefetch %s: unparseable selection %.200r", source.name, text)
        return None, usage
    return (parsed if isinstance(parsed, dict) else None), usage


async def _select_anthropic(
    handle: LLMHandle, source: SourceRun, briefing: str
) -> tuple[dict[str, Any] | None, Any]:
    """One Messages call with a forced single tool."""
    tool_name = f"select_{source.name}"
    response = await handle.client.messages.create(
        model=handle.model,
        max_tokens=_MAX_SELECTOR_TOKENS,
        system=SYSTEM_PROMPT.format(
            source=source.name, instructions=source.instructions
        ),
        tools=[{
            "name": tool_name,
            "description": "Submit the selection.",
            "input_schema": source.schema,
        }],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{
            "role": "user",
            "content": f"{briefing}\n\nCandidates:\n{source.candidates}",
        }],
    )
    usage = getattr(response, "usage", None)
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "tool_use":
            inp = getattr(block, "input", None)
            return (inp if isinstance(inp, dict) else None), usage
    return None, usage


async def _run_selector(
    handle: LLMHandle,
    source: SourceRun,
    briefing: str,
    timeout: float,
) -> tuple[SourceRun, dict[str, Any] | None, Any]:
    """One lane: select with a per-call timeout. Never raises."""
    try:
        select = (
            _select_openai if handle.provider == "openai" else _select_anthropic
        )
        selection, usage = await asyncio.wait_for(
            select(handle, source, briefing), timeout=timeout
        )
        return source, selection, usage
    except asyncio.TimeoutError:
        logger.warning(
            "prefetch selector %s timed out after %.0fs", source.name, timeout
        )
        return source, None, None
    except Exception:
        logger.exception("prefetch selector %s failed", source.name)
        return source, None, None


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


async def run_prefetch(
    req: PrefetchRequest,
    *,
    store: Any,
    client: LLMHandle,
    config: Any,
) -> PrefetchResult:
    """Fan selectors out in parallel and assemble the bundle.

    Never raises for expected failures — a lane that errors or times
    out simply contributes nothing. The caller owns the overall timeout
    (``asyncio.wait_for``) and the prefetch_event log row.
    """
    per_call_timeout = float(getattr(config, "per_call_timeout_seconds", 6.0))
    token_budget = int(getattr(config, "token_budget", 20000))

    sources = await gather_sources(req, store=store, config=config)
    bundle = PrefetchBundle()
    if not sources:
        bundle.render(token_budget=token_budget)
        return PrefetchResult(bundle=bundle, iterations=0, cost_usd=0.0)

    briefing = req.briefing()
    results = await asyncio.gather(*[
        _run_selector(client, s, briefing, per_call_timeout) for s in sources
    ])

    usages = []
    for source, selection, usage in results:
        if usage is not None:
            usages.append(usage)
        if not selection:
            continue
        try:
            await source.materialize(selection, bundle)
        except Exception:
            logger.exception("prefetch materialize %s failed", source.name)

    bundle.render(token_budget=token_budget)
    cost_usd = await _record_cost(
        usages, len(sources), client, req, store,
    )
    logger.info(
        "prefetch done key=%s lanes=%d model=%s empty=%s tokens=%d",
        req.key, len(sources), client.model,
        bundle.is_empty(), bundle.token_estimate,
    )
    return PrefetchResult(
        bundle=bundle, iterations=len(sources), cost_usd=cost_usd,
    )


def _accumulate_usage(totals: dict[str, int], usage: Any, names: tuple[str, ...]) -> None:
    for name in names:
        val = getattr(usage, name, None)
        if val is None and isinstance(usage, dict):
            val = usage.get(name)
        if val is None:
            continue
        try:
            totals[name] = totals.get(name, 0) + int(val)
        except (TypeError, ValueError):
            continue


async def _record_cost(
    usages: list[Any],
    lanes: int,
    handle: LLMHandle,
    req: PrefetchRequest,
    store: Any,
) -> float:
    """Persist one collapsed cost row; return the cost in USD (0 on failure)."""
    if not usages:
        return 0.0
    totals: dict[str, int] = {}
    if handle.provider == "openai":
        names = ("prompt_tokens", "completion_tokens")
        builder = from_openai_usage
    else:
        names = (
            "input_tokens", "output_tokens",
            "cache_read_input_tokens", "cache_creation_input_tokens",
        )
        builder = from_anthropic_usage
    for usage in usages:
        _accumulate_usage(totals, usage, names)
    if not totals:
        return 0.0
    try:
        event = builder(
            purpose="prefetch",
            model=handle.model,
            usage=totals,
            iterations=lanes,
            correlation_id=req.key,
            metadata={"channel": req.channel, "key_kind": req.key_kind},
        )
        await record_cost(store, event)
        return float(getattr(event, "cost_usd", 0.0) or 0.0)
    except Exception:
        logger.debug("prefetch cost record failed", exc_info=True)
        return 0.0
