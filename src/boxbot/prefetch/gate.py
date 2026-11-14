"""Follow-up gate: ask the conversation model whether this turn needs
docs the thread does not hold yet.

First turns get the selector fan-out. Follow-up turns used to get
nothing (the fan-out blocked every reply for seconds), which left
"ok, order the first one" with no purchasing SOP in context. The gate
closes that hole with one call to the conversation's own model:

    cached thread  +  [the new utterance, the doc menus, "which?"]

The prefix is what the reply call is about to send (same system
prompt, tools, response format, bare history). The provider prompt
cache covers it up to the previous cycle's user turn — that turn went
out with a wire-only context prefix and is replayed bare, so the
previous cycle's tail is read uncached here. The reply would have paid
for that tail anyway; the gate pays it first and leaves the cache warm
for the reply. Output is a few tokens.

The model only *names* docs. Names are validated against the menus the
selector lanes build and the bodies are read from disk — never taken
from model output.

Best-effort throughout: any failure or timeout means "nothing needed",
and the agent can still ``load_skill`` for itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from boxbot.prefetch.bundle import PrefetchBundle
from boxbot.prefetch.request import PrefetchRequest

logger = logging.getLogger(__name__)

# The reply is one small JSON object inside the notes envelope.
_MAX_OUTPUT_TOKENS = 300

PURPOSE = "prefetch_gate"


@dataclass(slots=True)
class GateResult:
    """Outcome of one gate call."""

    bundle: PrefetchBundle          # may be empty: nothing needed
    usage: Any                      # provider usage object, or None
    latency_ms: int


def build_gate_message(
    text: str, *, skills_menu: str | None, sections_menu: str | None,
) -> str:
    """The user-role message appended after the cached thread."""
    parts = [
        "<context-check>",
        "Not a message from a person. Do not reply to anyone and do not "
        "call tools.",
        "",
        "The next thing said in this conversation (a speech transcript; "
        "words may be misheard), as a JSON string:",
        json.dumps(text, ensure_ascii=False),
        "",
        "Which reference docs would you need open to handle it well? "
        "Choose only from the lists below — they leave out what this "
        "conversation already holds. Most turns need none; pick only on "
        "a clear match.",
    ]
    if skills_menu:
        parts += ["", "Skills (at most 2):", skills_menu]
    if sections_menu:
        parts += ["", "SDK doc sections (at most 4):", sections_menu]
    parts += [
        "",
        'Put ONLY this JSON in `thought`: {"skills": [...], '
        '"sections": [...]} — names and keys exactly as listed, empty '
        "arrays when nothing is needed. `observations` empty, "
        "`final_turn` false.",
        "</context-check>",
    ]
    return "\n".join(parts)


def parse_gate_reply(content: str) -> dict[str, Any]:
    """Selection dict from the model's reply.

    Expected: the notes envelope with the selection as a JSON string in
    ``thought``. Also accepted: the selection as the top-level object,
    ``thought`` already an object, markdown fences. Raises ValueError
    when no selection can be recovered.
    """
    def _loads(raw: str) -> Any:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`\n")
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)

    try:
        outer = _loads(content or "")
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"gate reply was not JSON: {e}") from e
    if not isinstance(outer, dict):
        raise ValueError("gate reply was not an object")

    payload: Any = outer
    if "skills" not in outer and "sections" not in outer:
        payload = outer.get("thought")
        if isinstance(payload, str):
            try:
                payload = _loads(payload)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"gate `thought` was not JSON: {e}") from e
    if not isinstance(payload, dict):
        raise ValueError("gate reply carried no selection")
    return {
        "skills": payload.get("skills") or [],
        "sections": payload.get("sections") or [],
    }


async def run_gate(
    req: PrefetchRequest,
    *,
    client: Any,
    ctx: dict[str, Any],
    history: list[dict[str, Any]],
    timeout_seconds: float,
) -> GateResult | None:
    """Run the gate for ``req``. None = no call made or call failed.

    ``ctx`` is the conversation's last request shape (model,
    system_prompt, tools, response_format, effort_kwargs) and
    ``history`` its thread in wire form — together the prefix the reply
    call will send.
    """
    from boxbot.core.agent_openai_adapter import to_openai_messages
    from boxbot.prefetch.sources import _sdk_source, _skills_source
    from boxbot.telemetry import model_log

    text = (req.text or "").strip()
    if not text:
        return None
    skills = _skills_source(req)
    sdk = _sdk_source(req)
    if skills is None and sdk is None:
        return None  # everything is already loaded

    messages = to_openai_messages(ctx["system_prompt"], history)
    messages.append({
        "role": "user",
        "content": build_gate_message(
            text,
            skills_menu=skills.candidates if skills else None,
            sections_menu=sdk.candidates if sdk else None,
        ),
    })

    t0 = time.monotonic()
    try:
        with model_log.call_context(purpose=PURPOSE, conversation_id=req.key):
            completion = await asyncio.wait_for(
                client.chat.completions.create(
                    model=ctx["model"],
                    messages=messages,
                    # Same tools and response format as the reply call:
                    # both are part of the provider's cache key.
                    # tool_choice keeps the answer textual without
                    # touching it.
                    tools=ctx["tools"],
                    tool_choice="none",
                    response_format=ctx["response_format"],
                    max_completion_tokens=_MAX_OUTPUT_TOKENS,
                    **ctx["effort_kwargs"],
                ),
                timeout=timeout_seconds,
            )
    except asyncio.TimeoutError:
        logger.warning(
            "follow-up gate timed out after %.1fs (conv=%s)",
            timeout_seconds, req.key,
        )
        return None
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("follow-up gate call failed (conv=%s)", req.key,
                       exc_info=True)
        return None
    latency_ms = int((time.monotonic() - t0) * 1000)

    try:
        selection = parse_gate_reply(
            completion.choices[0].message.content or ""
        )
    except (ValueError, IndexError, AttributeError) as e:
        logger.warning("follow-up gate reply unusable (conv=%s): %s",
                       req.key, e)
        return None

    bundle = PrefetchBundle()
    if skills is not None:
        await skills.materialize(selection, bundle)
    if sdk is not None:
        await sdk.materialize(selection, bundle)
    return GateResult(
        bundle=bundle,
        usage=getattr(completion, "usage", None),
        latency_ms=latency_ms,
    )
