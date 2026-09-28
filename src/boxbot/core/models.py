"""Model id → provider routing and per-model request knobs.

Derived from the id itself, never configured. One model id means one
provider; a config knob would only let the two disagree.
"""

from __future__ import annotations

import re
from typing import Literal

Provider = Literal["anthropic", "openai"]

# OpenAI id prefixes: chat (gpt-*) + reasoning (o1/o3/o4).
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4")

# GPT major/minor as separate ints, e.g. "gpt-5.6-luna" → (5, 6).
# Not float(): "5.10" would parse as 5.1 and sort below 5.6.
_GPT_VERSION = re.compile(r"^gpt-(\d+)(?:\.(\d+))?")

# First GPT version whose reasoning_effort accepts "none".
_EFFORT_NONE_FROM = (5, 6)
_EFFORT_MINIMAL_FROM = (5, 0)


def provider_for_model(model: str) -> Provider:
    """Return the provider that serves ``model``.

    Unknown/empty → "anthropic" (safe default, existing behaviour).
    """
    if model and model.lower().startswith(_OPENAI_PREFIXES):
        return "openai"
    return "anthropic"


def reasoning_effort_for_model(model: str) -> str | None:
    """Lowest ``reasoning_effort`` ``model`` accepts; None = omit it.

    | id | value |
    |---|---|
    | gpt-5.6 and later | ``"none"`` |
    | gpt-5.0–5.5, o1/o3/o4 | ``"minimal"`` |
    | everything else | None — the param 400s on non-reasoning ids |

    The fast tier exists for round-trip latency, so it always asks for
    the floor. Hardcoding one value instead would 400 twice per turn on
    any id that does not accept it.
    """
    mid = (model or "").lower()
    if mid.startswith(("o1", "o3", "o4")):
        return "minimal"
    match = _GPT_VERSION.match(mid)
    if match is None:
        return None
    version = (int(match.group(1)), int(match.group(2) or 0))
    if version >= _EFFORT_NONE_FROM:
        return "none"
    return "minimal" if version >= _EFFORT_MINIMAL_FROM else None
