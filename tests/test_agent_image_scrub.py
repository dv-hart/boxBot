"""Tests for ``_scrub_oversize_images`` in :mod:`boxbot.core.agent`.

Covers the recovery path when Anthropic 400s with an "image exceeds
5 MB maximum" error. The retry loop calls the scrub helper to swap the
offending image block for a text marker so the next API call succeeds.
"""
from __future__ import annotations

from boxbot.core.agent import _scrub_oversize_images


def _tool_result_with_image(image_marker: str = "AAA") -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_x",
                "content": [
                    {"type": "text", "text": "ok"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_marker,
                        },
                    },
                ],
            }
        ],
    }


def test_scrubs_named_image_block() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        _tool_result_with_image("AAA"),
    ]
    err = (
        "Error code: 400 - {'type': 'error', 'error': {'type': "
        "'invalid_request_error', 'message': "
        "'messages.1.content.0.tool_result.content.1.image.source.base64: "
        "image exceeds 5 MB maximum: 5302368 bytes > 5242880 bytes'}}"
    )
    n = _scrub_oversize_images(messages, err)
    assert n == 1
    inner = messages[1]["content"][0]["content"][1]
    assert inner["type"] == "text"
    assert "image dropped" in inner["text"]


def test_returns_zero_when_error_doesnt_match() -> None:
    messages = [_tool_result_with_image("AAA")]
    n = _scrub_oversize_images(messages, "Some unrelated 500 error")
    assert n == 0
    # Message history must be untouched
    assert messages[0]["content"][0]["content"][1]["type"] == "image"


def test_scrubs_multiple_in_one_error() -> None:
    """A single retry may need to scrub several images at once if a
    user sent multiple oversize photos."""
    messages = [
        _tool_result_with_image("AAA"),
        _tool_result_with_image("BBB"),
    ]
    err = (
        "messages.0.content.0.tool_result.content.1.image.source.base64: "
        "image exceeds 5 MB maximum: 5400000 bytes > 5242880 bytes; "
        "messages.1.content.0.tool_result.content.1.image.source.base64: "
        "image exceeds 5 MB maximum: 5800000 bytes > 5242880 bytes"
    )
    n = _scrub_oversize_images(messages, err)
    assert n == 2
    assert messages[0]["content"][0]["content"][1]["type"] == "text"
    assert messages[1]["content"][0]["content"][1]["type"] == "text"


def test_handles_out_of_range_indices_safely() -> None:
    """If the error names indices we no longer have (e.g. history was
    trimmed mid-flight), the scrub must not raise."""
    messages = [_tool_result_with_image("AAA")]
    err = (
        "messages.99.content.0.tool_result.content.1.image.source.base64: "
        "image exceeds 5 MB maximum"
    )
    n = _scrub_oversize_images(messages, err)
    assert n == 0
    assert messages[0]["content"][0]["content"][1]["type"] == "image"


def test_only_scrubs_image_blocks() -> None:
    """If the targeted index isn't an image (defensive — shouldn't
    happen but the parser shouldn't mangle text blocks)."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "x",
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": "second"},
                    ],
                }
            ],
        }
    ]
    err = (
        "messages.0.content.0.tool_result.content.1.image.source.base64: "
        "image exceeds 5 MB maximum"
    )
    n = _scrub_oversize_images(messages, err)
    assert n == 0
    assert messages[0]["content"][0]["content"][1]["text"] == "second"
