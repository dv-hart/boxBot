"""Translation seam between boxBot's Anthropic-shaped agent loop and the
OpenAI Chat Completions API.

Pure functions only — no network, no config reads, no main-process side
effects. Everything here is directly testable.

The thread stays canonical in **Anthropic** shape (memory extraction,
transcript rendering, and the persisted ``ConversationStore`` all read
it). This module translates at the two edges of one API call:

| direction | function |
|---|---|
| tool defs → OpenAI | :func:`to_openai_tools` |
| history → OpenAI | :func:`to_openai_messages` |
| completion → Anthropic-shaped response | :func:`to_anthropic_response` |
| JSON schema → OpenAI strict | :func:`to_strict_schema` |
| schema → ``response_format`` | :func:`build_response_format` |

:func:`to_anthropic_response` returns an :class:`AdaptedResponse` that
duck-types the Anthropic ``Message``: ``.content`` blocks expose
``.type`` / ``.text`` / ``.id`` / ``.name`` / ``.input``, plus
``.stop_reason`` / ``.usage`` / ``.model``. The rest of the loop —
``_response_to_content_blocks``, ``_process_tool_calls``, the
internal-notes scan — runs unchanged against it.

**Images.** ``execute_script`` and ``identify_person`` return image
blocks inside a ``tool_result``. OpenAI's ``role: "tool"`` message
accepts text only, so images ride a following ``role: "user"`` message
as ``image_url`` parts carrying a ``data:`` URI. Provenance is labelled
with the originating tool_use id so the model knows what it is looking
at.

**Structured output + tools.** Both are active on the same call.
``response_format`` pins ``INTERNAL_NOTES_SCHEMA`` for turns that emit
text; on tool-call turns OpenAI returns ``content: null`` and no schema
applies. OpenAI ``strict`` mode requires every property in ``required``
and ``additionalProperties: false``, so :func:`to_strict_schema`
rewrites optional properties as nullable rather than dropping strict —
``parse_internal_notes`` already tolerates ``null``.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MEDIA_TYPE = "image/jpeg"

# Placeholder text for a tool_result whose only payload is images.
# OpenAI rejects an empty tool message.
_IMAGE_ONLY_PLACEHOLDER = "(image attachment follows)"


# ---------------------------------------------------------------------------
# Anthropic-shaped response shim
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContentBlock:
    """One Anthropic-shaped content block (``text`` or ``tool_use``)."""

    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AdaptedResponse:
    """An OpenAI completion wearing the Anthropic ``Message`` interface.

    ``tool_argument_errors`` holds ready-made ``tool_result`` blocks for
    tool calls whose ``arguments`` string would not parse as JSON. The
    offending ``tool_use`` block stays in ``content`` (history must keep
    every id the model emitted), so the caller drops it from dispatch
    with :func:`drop_tool_calls` and feeds the error back instead.
    """

    content: list[ContentBlock]
    stop_reason: str | None
    usage: Any = None
    model: str = ""
    tool_argument_errors: list[dict[str, Any]] = field(default_factory=list)
    # Structured-Outputs refusal text (``message.refusal``). Model prose
    # — logged only, never delivered. Set ⇒ ``stop_reason`` is
    # ``"refusal"``.
    refusal: str | None = None


# finish_reason → Anthropic stop_reason. Anything unmapped falls through
# to "end_turn", which terminates the loop cleanly.
_STOP_REASONS = {
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "refusal",
}


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def to_openai_tools(
    definitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Anthropic tool defs → OpenAI ``tools``.

    ``{name, description, input_schema}`` →
    ``{type: "function", function: {name, description, parameters}}``.
    Anthropic-only keys (``cache_control``) are dropped; OpenAI 400s on
    unknown fields.

    Functions are **not** marked ``strict``: strict mode would require
    running every boxBot tool schema through :func:`to_strict_schema`
    (nullable-widening each optional parameter) for no gain — the loop
    already feeds malformed arguments back to the model as a
    ``tool_result`` error.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": d["name"],
                "description": d.get("description", ""),
                "parameters": d.get("input_schema")
                or {"type": "object", "properties": {}},
            },
        }
        for d in definitions
    ]


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


def to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """JSON schema → OpenAI ``strict`` mode equivalent.

    Strict mode requires every property listed in ``required`` and
    ``additionalProperties: false`` on every object. Optional properties
    are preserved by widening their type to include ``"null"`` — the
    documented substitute for omission — so no field is silently lost.
    Recurses through ``properties`` and array ``items``.
    """
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return schema

    props = schema.get("properties")
    if not isinstance(props, dict):
        return {**schema, "additionalProperties": False}

    required = set(schema.get("required") or [])
    rewritten: dict[str, Any] = {}
    for name, spec in props.items():
        if not isinstance(spec, dict):
            rewritten[name] = spec
            continue
        spec = _strict_property(spec)
        if name not in required:
            spec = _nullable(spec)
        rewritten[name] = spec

    return {
        **schema,
        "properties": rewritten,
        "required": list(props.keys()),
        "additionalProperties": False,
    }


def _strict_property(spec: dict[str, Any]) -> dict[str, Any]:
    """Apply strict-mode rules inside one property spec."""
    if spec.get("type") == "object":
        return to_strict_schema(spec)
    if spec.get("type") == "array" and isinstance(spec.get("items"), dict):
        return {**spec, "items": _strict_property(spec["items"])}
    return spec


def _nullable(spec: dict[str, Any]) -> dict[str, Any]:
    """Widen a property's ``type`` to admit ``null``."""
    kind = spec.get("type")
    if kind is None or kind == "null":
        return spec
    if isinstance(kind, list):
        return spec if "null" in kind else {**spec, "type": [*kind, "null"]}
    return {**spec, "type": [kind, "null"]}


def build_response_format(
    schema: dict[str, Any], name: str,
) -> dict[str, Any]:
    """Build the ``response_format`` payload pinning ``schema``."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": to_strict_schema(schema),
        },
    }


# ---------------------------------------------------------------------------
# History: Anthropic → OpenAI
# ---------------------------------------------------------------------------


def to_openai_messages(
    system_prompt: str,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Anthropic message history → OpenAI ``messages``.

    | Anthropic | OpenAI |
    |---|---|
    | ``system`` blocks | leading ``role: "system"`` |
    | assistant ``text`` | ``content`` (blocks joined by blank line) |
    | assistant ``tool_use`` | ``tool_calls[]`` with JSON-string args |
    | user ``tool_result`` | ``role: "tool"`` + ``tool_call_id`` |
    | image block | ``image_url`` part, ``data:`` URI |

    Images inside a ``tool_result`` cannot ride the tool message, so
    they are hoisted into a ``role: "user"`` message emitted right after
    the tool messages for that turn.
    """
    out: list[dict[str, Any]] = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            out.append(_assistant_message(content))
        elif role == "user":
            out.extend(_user_messages(content))
        elif role == "system":
            out.append({"role": "system", "content": _plain_text(content)})
    return out


def _assistant_message(content: Any) -> dict[str, Any]:
    """One Anthropic assistant turn → one OpenAI assistant message."""
    if not isinstance(content, list):
        return {"role": "assistant", "content": str(content or "")}

    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and block.get("text"):
            texts.append(str(block["text"]))
        elif kind == "tool_use":
            tool_calls.append({
                "id": str(block.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(block.get("name") or ""),
                    "arguments": json.dumps(block.get("input") or {}),
                },
            })

    # content may be null only alongside tool_calls; otherwise OpenAI
    # rejects the message.
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": "\n\n".join(texts) if texts else (
            None if tool_calls else ""
        ),
    }
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _user_messages(content: Any) -> list[dict[str, Any]]:
    """One Anthropic user turn → one or more OpenAI messages.

    A turn carrying tool results becomes N ``role: "tool"`` messages
    (one per result, order preserved) followed by at most one
    ``role: "user"`` message holding hoisted images plus any loose text
    (drained utterances, the turn-cap notice).
    """
    if not isinstance(content, list):
        text = str(content or "")
        return [{"role": "user", "content": text}] if text else []

    out: list[dict[str, Any]] = []
    trailing: list[dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "tool_result":
            text, images = _split_tool_result(block.get("content"))
            out.append({
                "role": "tool",
                "tool_call_id": str(block.get("tool_use_id") or ""),
                "content": text or _IMAGE_ONLY_PLACEHOLDER,
            })
            if images:
                trailing.append({
                    "type": "text",
                    "text": (
                        "Image attachment(s) from tool result "
                        f"{block.get('tool_use_id')}:"
                    ),
                })
                trailing.extend(images)
        elif kind == "image":
            part = _image_part(block)
            if part:
                trailing.append(part)
        elif kind == "text" and block.get("text"):
            trailing.append({"type": "text", "text": str(block["text"])})

    if trailing:
        out.append({"role": "user", "content": trailing})
    return out


def _split_tool_result(content: Any) -> tuple[str, list[dict[str, Any]]]:
    """Split tool_result content into ``(text, image_url parts)``."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return json.dumps(content), []

    texts: list[str] = []
    images: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            texts.append(str(block))
            continue
        if block.get("type") == "image":
            part = _image_part(block)
            if part:
                images.append(part)
        elif block.get("type") == "text":
            texts.append(str(block.get("text") or ""))
    return "\n".join(t for t in texts if t), images


def _image_part(block: dict[str, Any]) -> dict[str, Any] | None:
    """Anthropic image block → OpenAI ``image_url`` part.

    Base64 sources become a ``data:`` URI; ``url`` sources pass through.
    Anything unrecognised returns None (dropped rather than 400ing the
    whole turn).
    """
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    if source.get("type") == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": str(source["url"])}}
    data = source.get("data")
    if source.get("type") != "base64" or not data:
        logger.warning(
            "Dropping unsupported image source type %r", source.get("type"),
        )
        return None
    media_type = source.get("media_type") or _DEFAULT_MEDIA_TYPE
    if media_type == "image/gif" and _is_animated_gif(data):
        # OpenAI 400s on animated GIFs, and this path has no
        # ``_scrub_oversize_images`` analogue — one would poison every
        # later turn in the session. Drop it here instead.
        logger.warning("Dropping animated GIF: OpenAI rejects them")
        return None
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{data}"},
    }


def _is_animated_gif(data: str) -> bool:
    """True when base64 ``data`` is a multi-frame GIF.

    Two markers, either is enough: more than one Graphic Control
    Extension (``21 F9 04``, one per animated frame) or the NETSCAPE2.0
    loop extension. Undecodable input is treated as static — the API is
    the real validator.
    """
    try:
        raw = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError):
        return False
    if raw[:3] != b"GIF":
        return False
    return raw.count(b"\x21\xf9\x04") > 1 or b"NETSCAPE2.0" in raw


def _plain_text(content: Any) -> str:
    """Flatten any content shape to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(
            str(b.get("text") or "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content or "")


# ---------------------------------------------------------------------------
# Response: OpenAI → Anthropic
# ---------------------------------------------------------------------------


def to_anthropic_response(completion: Any) -> AdaptedResponse:
    """OpenAI completion → :class:`AdaptedResponse`.

    ``message.content`` becomes a ``text`` block (the
    INTERNAL_NOTES_SCHEMA JSON — private, never delivered);
    ``message.tool_calls`` become ``tool_use`` blocks with ``arguments``
    parsed back into a dict.

    A tool call whose ``arguments`` will not parse still yields its
    ``tool_use`` block — history must keep every id — and an entry in
    ``tool_argument_errors`` telling the model how to re-issue it.

    A Structured-Outputs refusal arrives as ``content: null`` +
    ``refusal: "…"`` + ``finish_reason: "stop"``. Mapping that through
    ``_STOP_REASONS`` alone would read as a clean ``end_turn`` and the
    box would just go quiet, so ``message.refusal`` overrides the
    finish_reason with ``"refusal"`` and the text rides
    ``AdaptedResponse.refusal``.
    """
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return AdaptedResponse(
            content=[],
            stop_reason="end_turn",
            usage=getattr(completion, "usage", None),
            model=getattr(completion, "model", "") or "",
        )

    choice = choices[0]
    message = getattr(choice, "message", None)
    blocks: list[ContentBlock] = []
    errors: list[dict[str, Any]] = []

    text = getattr(message, "content", None)
    if text:
        blocks.append(ContentBlock(type="text", text=str(text)))

    for call in getattr(message, "tool_calls", None) or []:
        fn = getattr(call, "function", None)
        name = str(getattr(fn, "name", "") or "")
        call_id = str(getattr(call, "id", "") or "")
        raw = getattr(fn, "arguments", "") or "{}"
        parsed, error = _parse_arguments(raw, name)
        blocks.append(
            ContentBlock(
                type="tool_use", id=call_id, name=name, input=parsed,
            )
        )
        if error:
            errors.append({
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": error,
            })

    # A refusal outranks finish_reason: OpenAI sends "stop" with it.
    refusal = getattr(message, "refusal", None)
    refusal = str(refusal) if refusal else None
    finish = getattr(choice, "finish_reason", None)
    return AdaptedResponse(
        content=blocks,
        stop_reason=(
            "refusal" if refusal
            else _STOP_REASONS.get(str(finish or ""), "end_turn")
        ),
        usage=getattr(completion, "usage", None),
        model=getattr(completion, "model", "") or "",
        tool_argument_errors=errors,
        refusal=refusal,
    )


def _parse_arguments(raw: Any, name: str) -> tuple[dict[str, Any], str | None]:
    """Parse a tool call's ``arguments`` string.

    Returns ``(input, error)``. The error string is a mini-prompt: what
    broke, then the recovery move.
    """
    if isinstance(raw, dict):
        return raw, None
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed, None
    logger.warning(
        "Tool call %r returned unparseable arguments: %.200r", name, raw,
    )
    return {}, json.dumps({
        "error": (
            f"Arguments for {name} were not a valid JSON object, so the "
            "tool did not run. Re-issue the call with well-formed JSON."
        )
    })


def drop_tool_calls(
    response: AdaptedResponse, ids: set[str],
) -> AdaptedResponse:
    """Return ``response`` without the ``tool_use`` blocks in ``ids``.

    Used to withhold malformed calls from dispatch while the full block
    list still goes into history.
    """
    if not ids:
        return response
    kept = [
        b for b in response.content
        if not (b.type == "tool_use" and b.id in ids)
    ]
    return replace(response, content=kept)


__all__ = [
    "AdaptedResponse",
    "ContentBlock",
    "build_response_format",
    "drop_tool_calls",
    "to_anthropic_response",
    "to_openai_messages",
    "to_openai_tools",
    "to_strict_schema",
]
