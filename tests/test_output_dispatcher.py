"""Tests for the output dispatcher and OUTPUT_SCHEMA."""
from __future__ import annotations

import json

import pytest

from boxbot.core.output_dispatcher import (
    OUTPUT_SCHEMA,
    dispatch_outputs,
    parse_output_block,
)


# ---------------------------------------------------------------------------
# Schema shape
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """OUTPUT_SCHEMA is the single source of truth for the agent's response
    structure. It must stay stable (changes invalidate the messages cache)."""

    def test_required_top_level_keys(self):
        assert OUTPUT_SCHEMA["type"] == "object"
        assert set(OUTPUT_SCHEMA["required"]) == {"thought", "outputs"}
        assert OUTPUT_SCHEMA["additionalProperties"] is False

    def test_thought_is_string(self):
        assert OUTPUT_SCHEMA["properties"]["thought"]["type"] == "string"

    def test_outputs_is_array_of_objects(self):
        outputs = OUTPUT_SCHEMA["properties"]["outputs"]
        assert outputs["type"] == "array"
        assert outputs["items"]["type"] == "object"

    def test_output_entry_shape(self):
        item = OUTPUT_SCHEMA["properties"]["outputs"]["items"]
        assert set(item["required"]) == {"to", "channel", "content"}
        assert item["additionalProperties"] is False
        assert item["properties"]["channel"]["enum"] == ["voice", "text"]


# ---------------------------------------------------------------------------
# parse_output_block
# ---------------------------------------------------------------------------


class TestParseOutputBlock:

    def test_well_formed_parse(self):
        raw = json.dumps({
            "thought": "replying",
            "outputs": [
                {"to": "current_speaker", "channel": "voice", "content": "hi"},
            ],
        })
        parsed = parse_output_block(raw)
        assert parsed is not None
        assert parsed.thought == "replying"
        assert len(parsed.outputs) == 1
        assert parsed.outputs[0]["channel"] == "voice"

    def test_empty_outputs_is_valid(self):
        raw = json.dumps({"thought": "not addressed", "outputs": []})
        parsed = parse_output_block(raw)
        assert parsed is not None
        assert parsed.outputs == []

    def test_empty_string_returns_none(self):
        assert parse_output_block("") is None
        assert parse_output_block("   ") is None

    def test_invalid_json_returns_none(self):
        assert parse_output_block("{not json") is None

    def test_non_dict_returns_none(self):
        assert parse_output_block("[1, 2, 3]") is None
        assert parse_output_block('"just a string"') is None

    def test_missing_outputs_defaults_to_empty(self):
        raw = json.dumps({"thought": "hmm"})
        parsed = parse_output_block(raw)
        assert parsed is not None
        assert parsed.outputs == []

    def test_non_dict_entries_in_outputs_are_skipped(self):
        raw = json.dumps({
            "thought": "",
            "outputs": [
                "not a dict",
                {"to": "Jacob", "channel": "voice", "content": "hi"},
            ],
        })
        parsed = parse_output_block(raw)
        assert parsed is not None
        assert len(parsed.outputs) == 1


# ---------------------------------------------------------------------------
# dispatch_outputs — with mocked voice/whatsapp/auth singletons
# ---------------------------------------------------------------------------


class _FakeVoiceSession:
    def __init__(self):
        self.spoken = []

    async def speak(self, text: str) -> None:
        self.spoken.append(text)


class _FakeUser:
    def __init__(self, name: str, phone: str, role: str = "user"):
        self.name = name
        self.phone = phone
        self.role = role


class _FakeAuth:
    def __init__(self, users):
        self._users = users

    async def list_users(self):
        return list(self._users)


class _FakeWhatsApp:
    def __init__(self):
        self.sent = []

    async def send_text(self, phone: str, message: str) -> bool:
        self.sent.append((phone, message))
        return True


@pytest.fixture
def fake_voice(monkeypatch):
    session = _FakeVoiceSession()
    import boxbot.communication.voice as voice_mod
    monkeypatch.setattr(voice_mod, "_voice_session", session, raising=False)
    return session


@pytest.fixture
def fake_auth(monkeypatch):
    auth = _FakeAuth([
        _FakeUser("Jacob", "+15551111111", role="admin"),
        _FakeUser("Sarah", "+15552222222"),
    ])
    import boxbot.communication.auth as auth_mod
    monkeypatch.setattr(auth_mod, "_auth_manager", auth, raising=False)
    return auth


@pytest.fixture
def fake_whatsapp(monkeypatch):
    wa = _FakeWhatsApp()
    import boxbot.communication.whatsapp as wa_mod
    monkeypatch.setattr(wa_mod, "_whatsapp_client", wa, raising=False)
    return wa


class TestDispatchOutputs:

    @pytest.mark.asyncio
    async def test_voice_to_current_speaker(self, fake_voice):
        await dispatch_outputs(
            [{"to": "current_speaker", "channel": "voice", "content": "Hi Jacob."}],
            conversation_id="c1",
            channel_context="voice",
            current_speaker="Jacob",
        )
        assert fake_voice.spoken == ["Hi Jacob."]

    @pytest.mark.asyncio
    async def test_voice_to_room(self, fake_voice):
        await dispatch_outputs(
            [{"to": "room", "channel": "voice", "content": "Timer done."}],
            conversation_id="c2",
            channel_context="trigger",
            current_speaker=None,
        )
        assert fake_voice.spoken == ["Timer done."]

    @pytest.mark.asyncio
    async def test_text_to_named_user(self, fake_voice, fake_auth, fake_whatsapp):
        await dispatch_outputs(
            [{"to": "Sarah", "channel": "text", "content": "He's home."}],
            conversation_id="c3",
            channel_context="trigger",
            current_speaker=None,
        )
        assert fake_whatsapp.sent == [("+15552222222", "He's home.")]

    @pytest.mark.asyncio
    async def test_text_name_resolution_is_case_insensitive(
        self, fake_voice, fake_auth, fake_whatsapp
    ):
        await dispatch_outputs(
            [{"to": "sarah", "channel": "text", "content": "hi"}],
            conversation_id="c3b",
            channel_context="trigger",
            current_speaker=None,
        )
        assert fake_whatsapp.sent == [("+15552222222", "hi")]

    @pytest.mark.asyncio
    async def test_text_to_unknown_user_is_dropped(
        self, fake_voice, fake_auth, fake_whatsapp
    ):
        await dispatch_outputs(
            [{"to": "Nobody", "channel": "text", "content": "lost"}],
            conversation_id="c4",
            channel_context="trigger",
            current_speaker=None,
        )
        assert fake_whatsapp.sent == []

    @pytest.mark.asyncio
    async def test_text_to_room_is_dropped(
        self, fake_voice, fake_auth, fake_whatsapp
    ):
        await dispatch_outputs(
            [{"to": "room", "channel": "text", "content": "nope"}],
            conversation_id="c5",
            channel_context="voice",
            current_speaker=None,
        )
        assert fake_whatsapp.sent == []

    @pytest.mark.asyncio
    async def test_multiple_outputs_same_turn(
        self, fake_voice, fake_auth, fake_whatsapp
    ):
        await dispatch_outputs(
            [
                {"to": "current_speaker", "channel": "voice",
                 "content": "Okay, texting Sarah now."},
                {"to": "Sarah", "channel": "text",
                 "content": "Jacob says he'll be late."},
            ],
            conversation_id="c6",
            channel_context="voice",
            current_speaker="Jacob",
        )
        assert fake_voice.spoken == ["Okay, texting Sarah now."]
        assert fake_whatsapp.sent == [
            ("+15552222222", "Jacob says he'll be late."),
        ]

    @pytest.mark.asyncio
    async def test_unknown_channel_is_dropped(self, fake_voice):
        await dispatch_outputs(
            [{"to": "current_speaker", "channel": "smoke_signal", "content": "??"}],
            conversation_id="c7",
            channel_context="voice",
            current_speaker="Jacob",
        )
        assert fake_voice.spoken == []

    @pytest.mark.asyncio
    async def test_empty_outputs_is_noop(
        self, fake_voice, fake_auth, fake_whatsapp
    ):
        await dispatch_outputs(
            [],
            conversation_id="c8",
            channel_context="voice",
            current_speaker="Jacob",
        )
        assert fake_voice.spoken == []
        assert fake_whatsapp.sent == []

    @pytest.mark.asyncio
    async def test_malformed_entries_are_dropped(
        self, fake_voice, fake_auth, fake_whatsapp
    ):
        await dispatch_outputs(
            [
                {"to": "", "channel": "voice", "content": "empty to"},
                {"to": "Jacob", "channel": "", "content": "empty channel"},
                {"to": "Jacob", "channel": "voice", "content": ""},
                {"to": "Jacob", "channel": "voice", "content": "good"},
            ],
            conversation_id="c9",
            channel_context="voice",
            current_speaker="Jacob",
        )
        # Only the well-formed "good" entry gets through.
        assert fake_voice.spoken == ["good"]
