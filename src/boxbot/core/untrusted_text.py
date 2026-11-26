"""Neutralise forged authority markers in text the system did not write.

Two markers carry authority in the model's thread, and only the agent
loop may write either:

- ``<turn-context>…</turn-context>`` — the per-turn state block.
- ``[system]`` — loop notices (turn cap, ``final_turn`` nudge).

Human text (any channel), tool results (web pages, merchant strings,
sandbox stdout, integration output) and prefetch text can contain a
look-alike. These helpers remove / defang them. They are pure and
idempotent: a turn neutralised on first send neutralises to the same
bytes on every replay, so the wire thread stays append-only.
"""

from __future__ import annotations

import re
from typing import Any

# Characters that render as nothing (or nearly) and can be threaded
# through a marker to dodge a literal match.
_INVISIBLE = "​‌‍‎‏⁠⁡⁢⁣⁤﻿­᠎͏"
_GAP = f"[{_INVISIBLE}]*"
_PAD = f"[\\s{_INVISIBLE}]*"


def _letter(ch: str) -> str:
    """Character class: ASCII letter (either case) or its full-width twin."""
    lo, up = ch.lower(), ch.upper()
    wide_lo = chr(ord(lo) - ord("a") + 0xFF41)
    wide_up = chr(ord(up) - ord("A") + 0xFF21)
    return f"[{lo}{up}{wide_lo}{wide_up}]"


def _word(word: str) -> str:
    return _GAP.join(_letter(c) for c in word)


_LT = "[<＜﹤‹〈⟨〈«]"
_GT = "[>＞﹥›〉⟩〉»]"
_SLASH = "[/／⁄∕]"
# turn-context, turn_context, turn context, turncontext, turn‐context …
_JOIN = f"[-_\\s‐-―−－＿·.{_INVISIBLE}]*"

# Any spelling of the delimiter: either case, full-width or look-alike
# angle brackets, whitespace / invisible characters anywhere inside,
# hyphen / underscore / space / nothing between the words, open or
# close. A forged tag that survives in a slightly different spelling is
# as good as the real one to the model.
TURN_CONTEXT_TAG_RE = re.compile(
    f"{_LT}{_PAD}(?:{_SLASH}{_PAD})?{_word('turn')}{_JOIN}"
    f"{_word('context')}{_PAD}{_GT}"
)

_LB = "[\\[［【〔⟦⁅]"
_RB = "[\\]］】〕⟧⁆]"
# ``[system]`` in any case / bracket look-alike / padding.
SYSTEM_MARKER_RE = re.compile(f"{_LB}{_PAD}{_word('system')}{_PAD}{_RB}")
# What a forged marker becomes: visibly what the text said, no longer
# the loop's marker (and cannot be re-assembled into one).
_SYSTEM_DEFANGED = "(system)"


def strip_turn_context_tags(text: str) -> str:
    """Remove every spelling of the turn-context delimiters.

    Runs to a fixed point: one pass over ``<turn-<turn-context>context>``
    would splice a live tag together out of the remains.
    """
    while True:
        stripped = TURN_CONTEXT_TAG_RE.sub("", text)
        if stripped == text:
            return text
        text = stripped


def defang_system_marker(text: str) -> str:
    """Rewrite a leading or inline ``[system]`` look-alike as ``(system)``."""
    return SYSTEM_MARKER_RE.sub(_SYSTEM_DEFANGED, text)


def neutralize_text(text: str) -> str:
    """Both: no live turn-context tag, no ``[system]`` marker."""
    return defang_system_marker(strip_turn_context_tags(text))


def has_marker(text: str) -> bool:
    return bool(
        TURN_CONTEXT_TAG_RE.search(text) or SYSTEM_MARKER_RE.search(text)
    )


def _walk(content: Any, fix: Any, probe: Any) -> Any:
    if isinstance(content, str):
        return fix(content)
    if not isinstance(content, list):
        return content
    out: list[Any] = []
    for block in content:
        if not isinstance(block, dict):
            out.append(block)
            continue
        btype = block.get("type")
        if btype == "text" and isinstance(block.get("text"), str):
            if probe(block["text"]):
                block = {**block, "text": fix(block["text"])}
        elif btype == "tool_result":
            inner = block.get("content")
            fixed = _walk(inner, fix, probe)
            if fixed is not inner and fixed != inner:
                block = {**block, "content": fixed}
        out.append(block)
    return out


def strip_turn_context_in_content(content: Any) -> Any:
    """Strip turn-context tags from any wire content shape.

    Strings are stripped directly. Block lists are walked: ``text``
    blocks and ``tool_result`` blocks (string or nested block content)
    are stripped in place-copies. Safe on replay of any turn — the loop
    never writes the tag into a content block.
    """
    return _walk(
        content, strip_turn_context_tags,
        lambda t: bool(TURN_CONTEXT_TAG_RE.search(t)),
    )


def neutralize_content(content: Any) -> Any:
    """:func:`neutralize_text` over any wire content shape.

    For content the system did not author, BEFORE the loop adds its own
    ``[system]`` notice to the same turn: tool results and folded
    utterances at first send, and string user turns (which the loop
    never authors a ``[system]`` notice in) on every materialisation.
    Not for replaying a block-list turn — that would rewrite the loop's
    own notices.
    """
    return _walk(content, neutralize_text, has_marker)
