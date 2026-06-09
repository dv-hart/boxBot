"""Regression tests for ``bb.photos.ingest`` argument handling.

The agent naturally calls ``ingest(path, source=..., sender=..., caption="")``
when an inbound image has no caption. An empty string is *not* ``None``, so
the old guard ran it through ``require_str`` — which rejects empty strings —
and the whole ingest raised ``ValueError: 'caption' must not be empty``.
Empty / whitespace-only sender and caption must be treated as "not provided".
"""

from __future__ import annotations

import pytest

from boxbot.sdk import _transport, photos


@pytest.fixture
def recorder(monkeypatch):
    """Stub ``_transport.request`` and capture the forwarded payload."""
    calls: list[tuple[str, dict]] = []

    def stub(action_type, payload, *, timeout=30):
        calls.append((action_type, payload))
        return {"status": "ok", "photo_id": "photo_test"}

    monkeypatch.setattr(_transport, "request", stub)
    return calls


def test_ingest_empty_caption_is_dropped_not_rejected(recorder):
    pid = photos.ingest("/tmp/x.jpg", source="signal", sender="Jacob", caption="")
    assert pid == "photo_test"
    _, payload = recorder[-1]
    assert "caption" not in payload  # empty caption omitted, not sent


def test_ingest_whitespace_sender_is_dropped(recorder):
    photos.ingest("/tmp/x.jpg", source="signal", sender="   ")
    _, payload = recorder[-1]
    assert "sender" not in payload


def test_ingest_real_caption_and_sender_forwarded(recorder):
    photos.ingest(
        "/tmp/x.jpg", source="signal", sender="Jacob", caption="beach day"
    )
    _, payload = recorder[-1]
    assert payload["sender"] == "Jacob"
    assert payload["caption"] == "beach day"
    assert payload["source"] == "signal"
