"""Tests for ``agent_openai_adapter`` — the Anthropic <-> OpenAI seam.

Everything here is pure translation: no client, no config, no network.
The invariants under test:

1. Tool defs keep their name/description/schema byte-for-byte and shed
   Anthropic-only keys.
2. History survives a round trip: text, tool_use/tool_calls,
   tool_result/role:"tool", and images.
3. Image blocks reach OpenAI as ``data:`` URIs — ``execute_script`` and
   ``identify_person`` attachments must actually be seen by the model.
4. ``INTERNAL_NOTES_SCHEMA`` survives the strict-mode rewrite with no
   field dropped.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from boxbot.core.agent_openai_adapter import (
    AdaptedResponse,
    build_response_format,
    drop_tool_calls,
    to_anthropic_response,
    to_openai_messages,
    to_openai_tools,
    to_strict_schema,
)
from boxbot.core.output_dispatcher import (
    INTERNAL_NOTES_SCHEMA,
    parse_internal_notes,
)

PNG_B64 = "iVBORw0KGgo="


def _image_block(media_type: str = "image/jpeg", data: str = PNG_B64) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64", "media_type": media_type, "data": data,
        },
    }


def _completion(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-5.6-luna",
    usage: Any = None,
    refusal: str | None = None,
) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=content, tool_calls=tool_calls, refusal=refusal,
            ),
            finish_reason=finish_reason,
        )],
        model=model,
        usage=usage,
    )


# Real Pillow-written 4x4 GIFs: one frame vs two. The animated one
# carries two Graphic Control Extensions + the NETSCAPE2.0 loop block.
_STATIC_GIF = (
    "R0lGODdhBAAEAIEAAP8AAAAAAAAAAAAAACwAAAAABAAEAAAICQABCBxIsCCAgAA7"
)
_ANIMATED_GIF = (
    "R0lGODlhBAAEAIEAAP8AAAAAAAAAAAAAACH/C05FVFNDQVBFMi4wAwEAAAAh+QQA"
    "CgAAACwAAAAABAAEAAAICQABCBxIsCCAgAAh+QQBCgABACwAAAAABAAEAIEAAP8A"
    "AAAAAAAAAAAICQABCBxIsCCAgAA7"
)


def _tool_call(call_id: str, name: str, arguments: str) -> Any:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def test_tool_defs_translate_and_drop_anthropic_keys():
    schema = {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    }
    defs = [
        {"name": "message", "description": "deliver", "input_schema": {}},
        {
            "name": "execute_script",
            "description": "run python",
            "input_schema": schema,
            "cache_control": {"type": "ephemeral"},
        },
    ]

    out = to_openai_tools(defs)

    assert [t["type"] for t in out] == ["function", "function"]
    assert out[0]["function"]["name"] == "message"
    assert out[1]["function"]["description"] == "run python"
    # Schema passes through untouched; cache_control is gone.
    assert out[1]["function"]["parameters"] is schema
    assert "cache_control" not in out[1]
    assert "cache_control" not in out[1]["function"]
    # Functions are not strict — boxBot tools use optional params.
    assert "strict" not in out[1]["function"]


def test_tool_def_without_schema_gets_empty_object():
    out = to_openai_tools([{"name": "mute_mic", "description": "d"}])
    assert out[0]["function"]["parameters"] == {
        "type": "object", "properties": {},
    }


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


def test_internal_notes_schema_survives_strict_rewrite():
    strict = to_strict_schema(INTERNAL_NOTES_SCHEMA)

    # No field dropped; all are required (strict-mode rule).
    assert set(strict["properties"]) == {
        "thought", "observations", "final_turn",
    }
    assert set(strict["required"]) == {
        "thought", "observations", "final_turn",
    }
    assert strict["additionalProperties"] is False

    # ``thought`` was required → type unchanged.
    assert strict["properties"]["thought"]["type"] == "string"
    # ``observations`` was optional → nullable, not removed.
    assert strict["properties"]["observations"]["type"] == ["array", "null"]
    assert strict["properties"]["observations"]["items"] == {"type": "string"}

    # Descriptions are the model's instructions — keep them byte-exact.
    assert (
        strict["properties"]["thought"]["description"]
        == INTERNAL_NOTES_SCHEMA["properties"]["thought"]["description"]
    )

    # Source schema is not mutated.
    assert INTERNAL_NOTES_SCHEMA["required"] == ["thought", "final_turn"]


def test_strict_null_observations_still_parse():
    """The nullable rewrite must not break ``parse_internal_notes``."""
    parsed = parse_internal_notes(
        json.dumps({"thought": "noted", "observations": None})
    )
    assert parsed is not None
    assert parsed.thought == "noted"
    assert parsed.observations == []


def test_response_format_payload():
    fmt = build_response_format(INTERNAL_NOTES_SCHEMA, "internal_notes")
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "internal_notes"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False


def test_strict_rewrite_recurses_into_nested_objects():
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": [],
            },
        },
        "required": ["outer"],
    }
    strict = to_strict_schema(schema)
    inner = strict["properties"]["outer"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["a"]
    assert inner["properties"]["a"]["type"] == ["string", "null"]


# ---------------------------------------------------------------------------
# History: Anthropic -> OpenAI
# ---------------------------------------------------------------------------


def test_system_prompt_leads_and_plain_user_passes_through():
    out = to_openai_messages("You are BB.", [
        {"role": "user", "content": "hey"},
    ])
    assert out == [
        {"role": "system", "content": "You are BB."},
        {"role": "user", "content": "hey"},
    ]


def test_assistant_tool_use_becomes_tool_calls():
    out = to_openai_messages("", [
        {"role": "assistant", "content": [
            {"type": "text", "text": '{"thought":"check"}'},
            {"type": "tool_use", "id": "t1", "name": "search_memory",
             "input": {"query": "erik"}},
        ]},
    ])
    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "assistant"
    assert msg["content"] == '{"thought":"check"}'
    call = msg["tool_calls"][0]
    assert call == {
        "id": "t1",
        "type": "function",
        "function": {
            "name": "search_memory",
            "arguments": '{"query": "erik"}',
        },
    }


def test_assistant_tool_use_only_has_null_content():
    out = to_openai_messages("", [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "mute_mic", "input": {}},
        ]},
    ])
    assert out[0]["content"] is None
    assert out[0]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_assistant_with_no_blocks_gets_empty_string_not_null():
    """OpenAI rejects null content when there are no tool_calls."""
    out = to_openai_messages("", [{"role": "assistant", "content": []}])
    assert out[0] == {"role": "assistant", "content": ""}


def test_tool_result_becomes_tool_role_message():
    out = to_openai_messages("", [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": '{"status":"delivered"}'},
        ]},
    ])
    assert out == [{
        "role": "tool",
        "tool_call_id": "t1",
        "content": '{"status":"delivered"}',
    }]


def test_multiple_tool_results_keep_order_and_ids():
    out = to_openai_messages("", [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "1"},
            {"type": "tool_result", "tool_use_id": "b", "content": "2"},
        ]},
    ])
    assert [m["tool_call_id"] for m in out] == ["a", "b"]
    assert all(m["role"] == "tool" for m in out)


def test_drained_utterance_rides_a_trailing_user_message():
    out = to_openai_messages("", [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            {"type": "text", "text": "actually never mind"},
        ]},
    ])
    assert out[0]["role"] == "tool"
    assert out[1]["role"] == "user"
    assert out[1]["content"] == [
        {"type": "text", "text": "actually never mind"},
    ]


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def test_tool_result_image_hoists_to_a_user_message_as_data_uri():
    out = to_openai_messages("", [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t9", "content": [
                {"type": "text", "text": "captured"},
                _image_block("image/png"),
            ]},
        ]},
    ])

    # The tool message keeps the text only — OpenAI tool messages are
    # text-only, so the pixels cannot ride here.
    assert out[0] == {
        "role": "tool", "tool_call_id": "t9", "content": "captured",
    }

    # The image follows as a user message, labelled with its provenance.
    assert out[1]["role"] == "user"
    label, image = out[1]["content"]
    assert label["type"] == "text"
    assert "t9" in label["text"]
    assert image == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
    }


def test_image_only_tool_result_gets_placeholder_text():
    out = to_openai_messages("", [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": [_image_block()]},
        ]},
    ])
    # Non-empty: OpenAI rejects an empty tool message.
    assert out[0]["content"]
    assert out[1]["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


def test_image_missing_media_type_defaults_to_jpeg():
    out = to_openai_messages("", [
        {"role": "user", "content": [
            {"type": "image",
             "source": {"type": "base64", "data": PNG_B64}},
        ]},
    ])
    assert out[0]["content"][0]["image_url"]["url"] == (
        f"data:image/jpeg;base64,{PNG_B64}"
    )


def test_url_image_source_passes_through():
    out = to_openai_messages("", [
        {"role": "user", "content": [
            {"type": "image",
             "source": {"type": "url", "url": "https://x/y.png"}},
        ]},
    ])
    assert out[0]["content"][0]["image_url"] == {"url": "https://x/y.png"}


def test_unrecognised_image_source_is_dropped_not_fatal():
    out = to_openai_messages("", [
        {"role": "user", "content": [
            {"type": "image", "source": {"type": "file", "file_id": "f1"}},
            {"type": "text", "text": "still here"},
        ]},
    ])
    assert out[0]["content"] == [{"type": "text", "text": "still here"}]


def test_animated_gif_is_dropped():
    """OpenAI 400s on animated GIFs, and this path never scrubs.

    One would poison every later turn in the session.
    """
    out = to_openai_messages("", [
        {"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/gif",
                "data": _ANIMATED_GIF,
            }},
            {"type": "text", "text": "still here"},
        ]},
    ])
    assert out[0]["content"] == [{"type": "text", "text": "still here"}]


def test_static_gif_passes_through():
    out = to_openai_messages("", [
        {"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/gif",
                "data": _STATIC_GIF,
            }},
        ]},
    ])
    assert out[0]["content"][0]["image_url"]["url"] == (
        f"data:image/gif;base64,{_STATIC_GIF}"
    )


def test_scrubbed_image_marker_survives_as_text():
    """``_scrub_oversize_images`` leaves a text marker in place."""
    out = to_openai_messages("", [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "text",
                 "text": "[image dropped: exceeded 5 MB after encoding]"},
            ]},
        ]},
    ])
    assert out[0]["content"] == (
        "[image dropped: exceeded 5 MB after encoding]"
    )


# ---------------------------------------------------------------------------
# Response: OpenAI -> Anthropic
# ---------------------------------------------------------------------------


def test_text_completion_becomes_a_text_block():
    notes = '{"thought":"thinking","observations":["quiet"]}'
    r = to_anthropic_response(_completion(content=notes))
    assert r.stop_reason == "end_turn"
    assert len(r.content) == 1
    assert r.content[0].type == "text"
    assert r.content[0].text == notes
    assert r.model == "gpt-5.6-luna"


def test_tool_calls_become_tool_use_blocks_with_parsed_input():
    r = to_anthropic_response(_completion(
        content=None,
        tool_calls=[_tool_call(
            "c1", "message",
            '{"to": "current_speaker", "channel": "voice", '
            '"content": "hi"}',
        )],
        finish_reason="tool_calls",
    ))
    assert r.stop_reason == "tool_use"
    block = r.content[0]
    assert block.type == "tool_use"
    assert block.id == "c1"
    assert block.name == "message"
    assert block.input == {
        "to": "current_speaker", "channel": "voice", "content": "hi",
    }
    assert r.tool_argument_errors == []


def test_finish_reason_mapping():
    for finish, expected in [
        ("stop", "end_turn"),
        ("tool_calls", "tool_use"),
        ("length", "max_tokens"),
        ("content_filter", "refusal"),
        ("something_new", "end_turn"),
    ]:
        r = to_anthropic_response(_completion(finish_reason=finish))
        assert r.stop_reason == expected, finish


def test_refusal_overrides_the_finish_reason():
    """Finding #2: OpenAI sends finish_reason "stop" with a refusal.

    Mapped naively that is an ``end_turn`` and the box goes quiet.
    """
    r = to_anthropic_response(_completion(
        content=None, finish_reason="stop", refusal="I can't help with that.",
    ))
    assert r.stop_reason == "refusal"
    assert r.refusal == "I can't help with that."
    assert r.content == []


def test_absent_refusal_leaves_the_finish_reason_alone():
    r = to_anthropic_response(_completion(content="{}"))
    assert r.stop_reason == "end_turn"
    assert r.refusal is None


def test_empty_choices_terminates_cleanly():
    r = to_anthropic_response(SimpleNamespace(
        choices=[], model="gpt-5.6-luna", usage=None,
    ))
    assert r.content == []
    assert r.stop_reason == "end_turn"


def test_usage_and_model_are_carried_for_cost():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=3)
    r = to_anthropic_response(_completion(content="{}", usage=usage))
    assert r.usage is usage
    assert r.model == "gpt-5.6-luna"


# ---------------------------------------------------------------------------
# Malformed tool arguments
# ---------------------------------------------------------------------------


def test_unparseable_arguments_yield_an_actionable_error():
    r = to_anthropic_response(_completion(
        tool_calls=[_tool_call("c1", "execute_script", '{"code": ')],
        finish_reason="tool_calls",
    ))

    # The block stays in history — the id must exist for the tool
    # result that answers it.
    assert r.content[0].type == "tool_use"
    assert r.content[0].id == "c1"
    assert r.content[0].input == {}

    err = r.tool_argument_errors[0]
    assert err["type"] == "tool_result"
    assert err["tool_use_id"] == "c1"
    text = json.loads(err["content"])["error"]
    assert "execute_script" in text
    assert "Re-issue the call" in text


def test_non_object_arguments_are_an_error_too():
    r = to_anthropic_response(_completion(
        tool_calls=[_tool_call("c1", "mute_mic", '"just a string"')],
        finish_reason="tool_calls",
    ))
    assert r.tool_argument_errors[0]["tool_use_id"] == "c1"


def test_drop_tool_calls_withholds_only_the_named_ids():
    r = to_anthropic_response(_completion(
        content="{}",
        tool_calls=[
            _tool_call("good", "mute_mic", "{}"),
            _tool_call("bad", "mute_mic", "{"),
        ],
        finish_reason="tool_calls",
    ))
    filtered = drop_tool_calls(r, {"bad"})
    kept = [b.id for b in filtered.content if b.type == "tool_use"]
    assert kept == ["good"]
    # Text block untouched; original response unchanged.
    assert any(b.type == "text" for b in filtered.content)
    assert len([b for b in r.content if b.type == "tool_use"]) == 2


def test_drop_tool_calls_preserves_every_other_field():
    r = to_anthropic_response(_completion(
        tool_calls=[
            _tool_call("good", "mute_mic", "{}"),
            _tool_call("bad", "mute_mic", "{"),
        ],
        finish_reason="tool_calls",
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    ))
    filtered = drop_tool_calls(r, {"bad"})
    assert filtered.tool_argument_errors == r.tool_argument_errors
    assert filtered.usage is r.usage
    assert filtered.model == r.model
    assert filtered.stop_reason == r.stop_reason


def test_drop_tool_calls_no_ids_is_identity():
    r = AdaptedResponse(content=[], stop_reason="end_turn")
    assert drop_tool_calls(r, set()) is r


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_full_turn_round_trips_through_both_directions():
    """assistant tool_use -> OpenAI -> tool result -> OpenAI, ids intact."""
    completion = _completion(
        content=None,
        tool_calls=[_tool_call("call_7", "camera_capture", "{}")],
        finish_reason="tool_calls",
    )
    response = to_anthropic_response(completion)

    # What the loop appends to the Anthropic-shaped thread.
    assistant = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
            for b in response.content if b.type == "tool_use"
        ],
    }
    tool_turn = {
        "role": "user",
        "content": [{
            "type": "tool_result", "tool_use_id": "call_7",
            "content": [
                {"type": "text", "text": "captured"}, _image_block(),
            ],
        }],
    }

    out = to_openai_messages("sys", [assistant, tool_turn])

    assert [m["role"] for m in out] == ["system", "assistant", "tool", "user"]
    assert out[1]["tool_calls"][0]["id"] == "call_7"
    assert out[2]["tool_call_id"] == "call_7"
    assert out[3]["content"][1]["type"] == "image_url"
