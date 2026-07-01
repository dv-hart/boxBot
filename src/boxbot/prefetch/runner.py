"""The prefetch mini-agent loop.

Structurally cloned from the web_search small-agent loop
(``tools/builtins/web_search.py``): a Haiku model given a whitelisted
set of READ-ONLY tools, an iteration cap, and a single collapsed cost
row. It reasons about what the main agent will need, gathers it, and
calls ``finalize`` to submit a tightly-curated bundle.

Runs in the main process and reuses the caller's ``MemoryStore`` and
Anthropic client — it never opens its own store or spends outside the
one ``purpose="prefetch"`` cost row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from boxbot.cost import from_anthropic_usage, record as record_cost
from boxbot.prefetch.bundle import PrefetchBundle
from boxbot.prefetch.request import PrefetchRequest
from boxbot.prefetch.tools import (
    PrefetchState,
    build_tool_definitions,
    dispatch_tool,
)

logger = logging.getLogger(__name__)

_MAX_OUTPUT_TOKENS = 1024
_DEFAULT_SMALL_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """\
You are boxBot's prefetch helper. A larger assistant is about to handle \
the request below. Your job: gather ONLY the context it will very likely \
need on its first turn, then call `finalize`.

You have read-only tools: search_memory, search_history, list_skills, \
load_skill, workspace_search, workspace_read, get_calendar, get_weather.

Principles:
- Precision over recall. You are measured on precision — an included item \
the assistant doesn't use is a failure. Prefer returning an EMPTY bundle \
over a marginally-relevant one.
- Be fast. A handful of targeted searches, then finalize. Don't explore \
exhaustively.
- Pre-load a skill body only when the task clearly needs that skill; skill \
bodies are large. At most one.
- Only pull calendar/weather when the task is actually about time or \
weather.
- When done, call `finalize` with the ids/names/paths of what you chose \
and a one-line note on what the assistant will likely do.

Always end by calling `finalize` exactly once."""


@dataclass(slots=True)
class PrefetchResult:
    bundle: PrefetchBundle
    iterations: int
    cost_usd: float


def _resolve_model(config: Any) -> str:
    model = getattr(config, "model", None)
    if model:
        return model
    try:
        from boxbot.core.config import get_config

        small = get_config().models.small
        if small:
            return small
    except Exception:
        pass
    return _DEFAULT_SMALL_MODEL


def _accumulate_usage(totals: dict[str, int], usage: Any) -> None:
    if usage is None:
        return
    for name in (
        "input_tokens", "output_tokens",
        "cache_read_input_tokens", "cache_creation_input_tokens",
    ):
        val = getattr(usage, name, None)
        if val is None and isinstance(usage, dict):
            val = usage.get(name)
        if val is None:
            continue
        try:
            totals[name] = totals.get(name, 0) + int(val)
        except (TypeError, ValueError):
            continue


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    val = getattr(obj, name, None)
    if val is None and isinstance(obj, dict):
        val = obj.get(name)
    return val if val is not None else default


async def run_prefetch(
    req: PrefetchRequest,
    *,
    store: Any,
    client: Any,
    config: Any,
) -> PrefetchResult:
    """Run the mini-agent and return a curated, budgeted bundle.

    Never raises for expected failures — on an empty/failed run it
    returns an empty bundle. The caller is responsible for the overall
    timeout (``asyncio.wait_for``) and for logging the prefetch_event.
    """
    model = _resolve_model(config)
    max_iter = int(getattr(config, "max_iterations", 6))
    token_budget = int(getattr(config, "token_budget", 1500))
    tools = build_tool_definitions()
    state = PrefetchState(store=store)
    usage_totals: dict[str, int] = {}

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": req.briefing()}
    ]
    iterations = 0
    while iterations < max_iter and state.selection is None:
        iterations += 1
        response = await client.messages.create(
            model=model,
            max_tokens=_MAX_OUTPUT_TOKENS,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            tools=tools,
            messages=messages,
        )
        _accumulate_usage(usage_totals, _attr(response, "usage"))

        content = _attr(response, "content", []) or []
        assistant_blocks: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in content:
            btype = _attr(block, "type")
            if btype == "text":
                assistant_blocks.append(
                    {"type": "text", "text": _attr(block, "text", "")}
                )
            elif btype == "tool_use":
                assistant_blocks.append({
                    "type": "tool_use",
                    "id": _attr(block, "id"),
                    "name": _attr(block, "name"),
                    "input": _attr(block, "input", {}) or {},
                })
                result = await dispatch_tool(
                    _attr(block, "name"),
                    _attr(block, "input", {}) or {},
                    state=state,
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": _attr(block, "id"),
                    "content": result,
                })

        if assistant_blocks:
            messages.append({"role": "assistant", "content": assistant_blocks})
        if _attr(response, "stop_reason") != "tool_use":
            # Model stopped without (or after) tool calls and didn't
            # finalize — we're done; use whatever selection exists.
            break
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    bundle = _assemble(state, token_budget=token_budget)
    cost_usd = await _record_cost(
        usage_totals, iterations, model, req, store,
    )
    logger.info(
        "prefetch done key=%s iter=%d empty=%s tokens=%d",
        req.key, iterations, bundle.is_empty(), bundle.token_estimate,
    )
    return PrefetchResult(bundle=bundle, iterations=iterations, cost_usd=cost_usd)


def _assemble(state: PrefetchState, *, token_budget: int) -> PrefetchBundle:
    """Deterministically build the bundle from the model's finalize picks.

    Re-fetches by id/name/path (never trusts model-copied text) and caps
    to one skill body. Rendering applies the token budget.
    """
    from boxbot.skills.loader import load_skill
    from boxbot.workspace.store import Workspace

    bundle = PrefetchBundle()
    sel = state.selection or {}

    for mid in sel.get("memory_ids", []):
        summ = state.seen.get(mid)
        if summ:
            bundle.memories.append((mid, summ))

    for name in list(sel.get("skills", []))[:1]:  # cap: at most one body
        try:
            bundle.skill_bodies[name] = load_skill(name)
        except Exception:
            continue

    ws = Workspace()
    for path in sel.get("workspace_paths", []):
        try:
            rec = ws.read(path)
        except Exception:
            continue
        content = rec.get("content")
        if isinstance(content, str) and content.strip():
            bundle.workspace_excerpts.append((path, content[:1200]))

    for src in sel.get("pulled_sources", []):
        pulled = state.pulled.get(src)
        if pulled:
            bundle.pulled_data.append({
                "source": src,
                "action": pulled.get("action"),
                "payload": pulled.get("payload"),
                "pulled_at": pulled.get("pulled_at"),
            })

    bundle.likely_next_note = str(sel.get("likely_next_note") or "")
    # Populate token_estimate + drop anything over budget now, so the
    # stored event and the injected section agree.
    bundle.render(token_budget=token_budget)
    return bundle


async def _record_cost(
    usage_totals: dict[str, int],
    iterations: int,
    model: str,
    req: PrefetchRequest,
    store: Any,
) -> float:
    """Persist one collapsed cost row; return the cost in USD (0 on failure)."""
    if not usage_totals:
        return 0.0
    try:
        event = from_anthropic_usage(
            purpose="prefetch",
            model=model,
            usage=usage_totals,
            iterations=iterations,
            correlation_id=req.key,
            metadata={"channel": req.channel, "key_kind": req.key_kind},
        )
        await record_cost(store, event)
        return float(getattr(event, "cost_usd", 0.0) or 0.0)
    except Exception:
        logger.debug("prefetch cost record failed", exc_info=True)
        return 0.0
