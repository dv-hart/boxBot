"""Provider-specific helpers that turn an API response into a CostEvent.

These are thin readers: pull the units the API returned (tokens,
billed characters, audio seconds), look up the per-unit price in
``pricing.yaml``, and apply the constant Anthropic cache/batch
multipliers. Where the SDK already computes a dollar cost
(``ResultMessage.total_cost_usd``), that value is used verbatim.

Multipliers and discounts live here, not in YAML, because they are
constant across models per Anthropic's docs.
"""

from __future__ import annotations

import logging
from typing import Any

from boxbot.cost.event import CostEvent
from boxbot.cost.pricing import get_pricing

logger = logging.getLogger(__name__)

# Constant Anthropic multipliers per
# https://platform.claude.com/docs/en/about-claude/pricing
_CACHE_READ_MULT = 0.10
_CACHE_WRITE_5M_MULT = 1.25
_CACHE_WRITE_1H_MULT = 2.00
_BATCH_DISCOUNT = 0.50


# ---------------------------------------------------------------------------
# Anthropic — raw messages.create / batch result usage
# ---------------------------------------------------------------------------


def _attr(obj: Any, name: str, default: int = 0) -> int:
    """Read int field from an SDK Usage object or a plain dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        val = obj.get(name, default)
    else:
        val = getattr(obj, name, default)
    return int(val or default)


def _split_cache_creation(usage: Any) -> tuple[int, int]:
    """Return (5m_tokens, 1h_tokens) from a usage object.

    The Anthropic API may return ``cache_creation_input_tokens`` as a
    flat int (older shape; we treat as 1h since cache writes default
    to 5m unless the request set 1h, but historical extraction code
    tagged the system prompt as 1h — mirror that) OR a structured
    ``cache_creation`` dict with TTL-broken-out fields. Handle both;
    prefer the structured form when present.
    """
    if usage is None:
        return 0, 0
    cc = (
        usage.get("cache_creation")
        if isinstance(usage, dict)
        else getattr(usage, "cache_creation", None)
    )
    if cc is not None:
        five_m = _attr(cc, "ephemeral_5m_input_tokens")
        one_h = _attr(cc, "ephemeral_1h_input_tokens")
        if five_m or one_h:
            return five_m, one_h

    flat = _attr(usage, "cache_creation_input_tokens")
    return 0, flat


def from_anthropic_usage(
    *,
    purpose: str,
    model: str,
    usage: Any,
    is_batch: bool = False,
    iterations: int = 0,
    correlation_id: str | None = None,
    metadata: dict | None = None,
) -> CostEvent:
    """Build a CostEvent from an Anthropic Usage object.

    ``usage`` may be the SDK's typed Usage, a plain dict, or any object
    with the same attribute names. Token counts come from the response;
    per-token prices come from ``pricing.yaml``.

    ``iterations`` is for agentic-loop callers (e.g. the web_search
    sub-agent) that sum usage across N inner turns and write one row
    per outer call. Defaults to 0 for the common single-call case.
    """
    in_tok = _attr(usage, "input_tokens")
    out_tok = _attr(usage, "output_tokens")
    cache_read = _attr(usage, "cache_read_input_tokens")
    cw_5m, cw_1h = _split_cache_creation(usage)

    pricing = get_pricing()
    in_per = pricing.anthropic_input_per_mtok(model)
    out_per = pricing.anthropic_output_per_mtok(model)

    if in_per is None or out_per is None:
        logger.warning(
            "No anthropic pricing for model %r; recording cost_usd=0.0", model
        )
        cost = 0.0
    else:
        cost = (
            in_tok * in_per
            + out_tok * out_per
            + cache_read * in_per * _CACHE_READ_MULT
            + cw_5m * in_per * _CACHE_WRITE_5M_MULT
            + cw_1h * in_per * _CACHE_WRITE_1H_MULT
        ) / 1_000_000
        if is_batch:
            cost *= _BATCH_DISCOUNT

    return CostEvent(
        purpose=purpose,
        provider="anthropic",
        model=model,
        cost_usd=cost,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_read_tokens=cache_read,
        cache_write_5m_tokens=cw_5m,
        cache_write_1h_tokens=cw_1h,
        is_batch=is_batch,
        iterations=iterations,
        correlation_id=correlation_id,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Anthropic — claude_agent_sdk ResultMessage
# ---------------------------------------------------------------------------


def from_agent_sdk_result(
    *,
    purpose: str,
    result_message: Any,
    correlation_id: str | None = None,
    metadata: dict | None = None,
) -> list[CostEvent]:
    """Build CostEvents from a claude_agent_sdk ResultMessage.

    Prefers the SDK's own dollar figures. Behaviour:

    * If ``model_usage`` is present (multi-model run, e.g. main +
      sub-agent), emit one event per model. Each event carries that
      model's per-call cost when the SDK exposes it; otherwise we fall
      back to computing from the model's tokens via pricing.yaml.
    * Otherwise emit a single event using ``total_cost_usd`` directly
      and the primary ``usage`` totals.

    Note: ``total_cost_usd`` is a client-side estimate maintained by the
    Agent SDK. It is used here as the source of truth for billing
    within boxBot — for authoritative monthly cost, reconcile against
    the Anthropic Usage and Cost API separately.
    """
    model_usage = _maybe_dict(getattr(result_message, "model_usage", None))
    total_cost = float(getattr(result_message, "total_cost_usd", 0.0) or 0.0)
    usage = getattr(result_message, "usage", None)

    if model_usage:
        events: list[CostEvent] = []
        for model, mu in model_usage.items():
            event = from_anthropic_usage(
                purpose=purpose,
                model=model,
                usage=mu,
                is_batch=False,
                correlation_id=correlation_id,
                metadata=metadata,
            )
            sdk_cost = _maybe_float(_pick(mu, "cost_usd", "costUSD"))
            if sdk_cost is not None:
                event = _replace_cost(event, sdk_cost)
            events.append(event)
        return events

    # Single-model fallback: trust total_cost_usd as authoritative.
    event = from_anthropic_usage(
        purpose=purpose,
        model=str(getattr(result_message, "model", "") or "unknown"),
        usage=usage,
        is_batch=False,
        correlation_id=correlation_id,
        metadata=metadata,
    )
    if total_cost > 0.0:
        event = _replace_cost(event, total_cost)
    return [event]


# ---------------------------------------------------------------------------
# ElevenLabs — TTS / STT
# ---------------------------------------------------------------------------


def from_elevenlabs_tts(
    *,
    model: str,
    billed_chars: int,
    correlation_id: str | None = None,
    metadata: dict | None = None,
) -> CostEvent:
    """Build a CostEvent from an ElevenLabs TTS call.

    ``billed_chars`` is the value of the ``x-character-count`` response
    header — the authoritative billed unit. Do not pass ``len(text)``;
    SSML and Unicode normalization can shift the billed count.
    """
    pricing = get_pricing()
    per_char = pricing.elevenlabs_tts_per_char(model)
    if per_char is None:
        logger.warning(
            "No elevenlabs TTS pricing for model %r; recording cost_usd=0.0",
            model,
        )
        cost = 0.0
    else:
        cost = billed_chars * per_char

    return CostEvent(
        purpose="tts",
        provider="elevenlabs",
        model=model,
        cost_usd=cost,
        character_count=billed_chars,
        correlation_id=correlation_id,
        metadata=metadata,
    )


def from_elevenlabs_stt(
    *,
    model: str,
    audio_seconds: float,
    correlation_id: str | None = None,
    metadata: dict | None = None,
) -> CostEvent:
    """Build a CostEvent from an ElevenLabs Scribe (STT) call.

    ``audio_seconds`` must be measured from the input audio (e.g.
    ``len(pcm) / (sample_rate * channels * sample_width)``). The API
    does not return billable duration in the response.
    """
    pricing = get_pricing()
    per_min = pricing.elevenlabs_stt_per_minute(model)
    if per_min is None:
        logger.warning(
            "No elevenlabs STT pricing for model %r; recording cost_usd=0.0",
            model,
        )
        cost = 0.0
    else:
        cost = (audio_seconds / 60.0) * per_min

    return CostEvent(
        purpose="stt",
        provider="elevenlabs",
        model=model,
        cost_usd=cost,
        audio_seconds=audio_seconds,
        correlation_id=correlation_id,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _maybe_dict(obj: Any) -> dict | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    # Some SDKs expose a model-shaped object; coerce via __dict__ if simple.
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict) and d:
        return d
    return None


def _pick(obj: Any, *names: str) -> Any:
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _replace_cost(event: CostEvent, cost: float) -> CostEvent:
    """Return a copy of event with cost_usd swapped (slots dataclass)."""
    return CostEvent(
        purpose=event.purpose,
        provider=event.provider,
        model=event.model,
        cost_usd=cost,
        input_tokens=event.input_tokens,
        output_tokens=event.output_tokens,
        cache_read_tokens=event.cache_read_tokens,
        cache_write_5m_tokens=event.cache_write_5m_tokens,
        cache_write_1h_tokens=event.cache_write_1h_tokens,
        is_batch=event.is_batch,
        character_count=event.character_count,
        audio_seconds=event.audio_seconds,
        iterations=event.iterations,
        correlation_id=event.correlation_id,
        metadata=event.metadata,
    )
