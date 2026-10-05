"""Tests for the output dispatcher and INTERNAL_NOTES_SCHEMA."""
from __future__ import annotations

import json

import pytest
from conftest import FakeAuth

from boxbot.core.output_dispatcher import (
    BUDGET_SPENT,
    DEGENERATE_CONTENT,
    INTERNAL_NOTES_SCHEMA,
    TOOL_SYNTAX,
    UNRETRYABLE_DROPS,
    DispatchResult,
    _is_degenerate_content,
    dispatch_outputs,
    parse_internal_notes,
    parse_structured_notes,
)


# ---------------------------------------------------------------------------
# Schema shape
# ---------------------------------------------------------------------------


class TestInternalNotesSchema:
    """INTERNAL_NOTES_SCHEMA pins the agent's text output to a private
    scratchpad shape. It must stay stable (changes invalidate the messages
    cache) and must contain NO delivery channel — deliveries go through
    the message tool, not through this JSON.
    """

    def test_required_top_level_keys(self):
        assert INTERNAL_NOTES_SCHEMA["type"] == "object"
        assert set(INTERNAL_NOTES_SCHEMA["required"]) == {
            "thought", "final_turn",
        }
        assert INTERNAL_NOTES_SCHEMA["additionalProperties"] is False

    def test_thought_is_string(self):
        assert INTERNAL_NOTES_SCHEMA["properties"]["thought"]["type"] == "string"

    def test_observations_is_string_array(self):
        obs = INTERNAL_NOTES_SCHEMA["properties"]["observations"]
        assert obs["type"] == "array"
        assert obs["items"]["type"] == "string"

    def test_no_delivery_fields(self):
        # Hard guarantee: no `outputs`, `response_text`, `to`, `channel`,
        # `content`, etc. The schema must not provide a delivery path.
        forbidden = {
            "outputs", "response_text", "to", "channel", "content", "respond",
        }
        assert forbidden.isdisjoint(
            INTERNAL_NOTES_SCHEMA["properties"].keys()
        )


# ---------------------------------------------------------------------------
# parse_internal_notes
# ---------------------------------------------------------------------------


class TestParseInternalNotes:

    def test_well_formed_parse(self):
        raw = json.dumps({
            "thought": "replying",
            "observations": ["Jacob seems tired"],
        })
        parsed = parse_internal_notes(raw)
        assert parsed is not None
        assert parsed.thought == "replying"
        assert parsed.observations == ["Jacob seems tired"]

    def test_missing_observations_is_empty(self):
        raw = json.dumps({"thought": "not addressed"})
        parsed = parse_internal_notes(raw)
        assert parsed is not None
        assert parsed.thought == "not addressed"
        assert parsed.observations == []

    def test_empty_string_returns_none(self):
        assert parse_internal_notes("") is None
        assert parse_internal_notes("   ") is None

    def test_invalid_json_returns_none(self):
        assert parse_internal_notes("{not json") is None

    def test_non_dict_returns_none(self):
        assert parse_internal_notes("[1, 2, 3]") is None
        assert parse_internal_notes('"just a string"') is None

    def test_non_string_observations_are_skipped(self):
        raw = json.dumps({
            "thought": "",
            "observations": [42, "kept", None, "also kept"],
        })
        parsed = parse_internal_notes(raw)
        assert parsed is not None
        assert parsed.observations == ["kept", "also kept"]


# ---------------------------------------------------------------------------
# parse_structured_notes — the claude_agent_sdk ResultMessage.structured_output
# path, where the schema output arrives already parsed as a dict.
# ---------------------------------------------------------------------------


class TestParseStructuredNotes:

    def test_dict_is_parsed_directly(self):
        parsed = parse_structured_notes({
            "thought": "briefing delivered",
            "observations": ["calendar pulled cleanly"],
        })
        assert parsed is not None
        assert parsed.thought == "briefing delivered"
        assert parsed.observations == ["calendar pulled cleanly"]

    def test_none_returns_none(self):
        # The common case when a run produces no structured output.
        assert parse_structured_notes(None) is None

    def test_missing_observations_is_empty(self):
        parsed = parse_structured_notes({"thought": "ok"})
        assert parsed is not None
        assert parsed.observations == []

    def test_non_string_observations_are_skipped(self):
        parsed = parse_structured_notes(
            {"thought": "", "observations": [1, "kept", None, "also"]}
        )
        assert parsed is not None
        assert parsed.observations == ["kept", "also"]

    def test_json_string_is_tolerated(self):
        parsed = parse_structured_notes(
            json.dumps({"thought": "via string", "observations": []})
        )
        assert parsed is not None
        assert parsed.thought == "via string"

    def test_unexpected_type_returns_none(self):
        assert parse_structured_notes([1, 2, 3]) is None
        assert parse_structured_notes(42) is None


# ---------------------------------------------------------------------------
# dispatch_outputs — with mocked voice/whatsapp/auth singletons
#
# dispatch_outputs is now invoked from the message tool (one entry
# per call) and from trigger-fired turns (potentially batched). Its routing
# behaviour is unchanged from when the agent loop called it directly.
# ---------------------------------------------------------------------------


class _FakeVoiceSession:
    def __init__(self):
        self.spoken = []
        self.relays = []

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    async def speak_and_listen(self, text: str, *, relay=None) -> None:
        """Speak, then leave the mic open — used for relays only."""
        self.spoken.append(text)
        self.relays.append(relay)


class _FakeUser:
    def __init__(
        self,
        name: str,
        phone: str,
        role: str = "user",
        channel: str = "whatsapp",
    ):
        self.name = name
        self.phone = phone
        self.role = role
        self.channel = channel


class _FakeWhatsApp:
    name = "whatsapp"

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
    auth = FakeAuth([
        _FakeUser("Jacob", "+15551111111", role="admin"),
        _FakeUser("Sarah", "+15552222222"),
    ])
    import boxbot.communication.auth as auth_mod
    monkeypatch.setattr(auth_mod, "_auth_manager", auth, raising=False)
    return auth


@pytest.fixture
def fake_whatsapp():
    wa = _FakeWhatsApp()
    # Channel-agnostic dispatcher reaches the client through the registry,
    # so register via the canonical setter (it also handles teardown).
    import boxbot.communication.whatsapp as wa_mod
    wa_mod.set_whatsapp_client(wa)
    try:
        yield wa
    finally:
        wa_mod.set_whatsapp_client(None)


class TestDispatchOutputs:

    @pytest.mark.asyncio
    async def test_voice_to_current_speaker(self, fake_voice):
        results = await dispatch_outputs(
            [{"to": "current_speaker", "channel": "voice", "content": "Hi Jacob."}],
            conversation_id="c1",
            channel_context="voice",
            current_speaker="Jacob",
        )
        assert fake_voice.spoken == ["Hi Jacob."]
        assert [r.status for r in results] == ["delivered"]

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
    async def test_trigger_announcement_does_not_open_the_mic(self, fake_voice):
        """A timer or briefing has nobody waiting — it speaks and stops."""
        await dispatch_outputs(
            [{"to": "room", "channel": "voice", "content": "Timer done."}],
            conversation_id="c2",
            channel_context="trigger",
            current_speaker=None,
        )
        assert fake_voice.relays == []

    @pytest.mark.asyncio
    async def test_speak_from_text_channel_relays_and_opens_the_mic(
        self, fake_voice
    ):
        """The 2026-07-08 case: Jacob texts, BB asks the room, mic opens."""
        await dispatch_outputs(
            [{
                "to": "Carina",
                "channel": "voice",
                "content": "Carina — books tonight?",
            }],
            conversation_id="conv_934cf4780ab5",
            channel_context="signal",
            current_speaker="Jacob",
        )

        assert fake_voice.spoken == ["Carina — books tonight?"]
        [relay] = fake_voice.relays
        assert relay is not None, "speech on a human's behalf must relay"
        assert relay.origin_person == "Jacob"
        assert relay.addressee == "Carina"
        assert relay.origin_channel == "signal"
        assert relay.origin_conversation_id == "conv_934cf4780ab5"
        assert relay.spoken_text == "Carina — books tonight?"

    @pytest.mark.asyncio
    async def test_voice_channel_speech_is_not_a_relay(self, fake_voice):
        """Already in the room: the voice session owns the mic, not us."""
        await dispatch_outputs(
            [{"to": "current_speaker", "channel": "voice", "content": "Sure."}],
            conversation_id="c1",
            channel_context="voice",
            current_speaker="Jacob",
        )
        assert fake_voice.relays == []

    @pytest.mark.asyncio
    async def test_no_relay_without_a_known_asker(self, fake_voice):
        """An unattributed text turn has no one to report back to."""
        await dispatch_outputs(
            [{"to": "room", "channel": "voice", "content": "Anyone home?"}],
            conversation_id="c3",
            channel_context="signal",
            current_speaker=None,
        )
        assert fake_voice.relays == []
        assert fake_voice.spoken == ["Anyone home?"]

    @pytest.mark.asyncio
    async def test_text_to_named_user(self, fake_voice, fake_auth, fake_whatsapp):
        results = await dispatch_outputs(
            [{"to": "Sarah", "channel": "text", "content": "He's home."}],
            conversation_id="c3",
            channel_context="trigger",
            current_speaker=None,
        )
        assert fake_whatsapp.sent == [("+15552222222", "He's home.")]
        assert [r.status for r in results] == ["delivered"]

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
        results = await dispatch_outputs(
            [{"to": "Nobody", "channel": "text", "content": "lost"}],
            conversation_id="c4",
            channel_context="trigger",
            current_speaker=None,
        )
        assert fake_whatsapp.sent == []
        assert len(results) == 1
        result = results[0]
        assert result.status == "dropped"
        assert "unknown recipient 'Nobody'" in result.reason
        # The agent gets the valid names back so it can retry.
        assert result.valid_recipients == ["Jacob", "Sarah"]
        assert "Jacob" in result.reason and "Sarah" in result.reason

    @pytest.mark.asyncio
    async def test_text_to_room_is_dropped(
        self, fake_voice, fake_auth, fake_whatsapp
    ):
        results = await dispatch_outputs(
            [{"to": "room", "channel": "text", "content": "nope"}],
            conversation_id="c5",
            channel_context="voice",
            current_speaker=None,
        )
        assert fake_whatsapp.sent == []
        assert results[0].status == "dropped"
        assert "room" in results[0].reason
        # "room" is not an unresolved name — no recipient list to suggest.
        assert results[0].valid_recipients is None

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
        results = await dispatch_outputs(
            [{"to": "current_speaker", "channel": "smoke_signal", "content": "??"}],
            conversation_id="c7",
            channel_context="voice",
            current_speaker="Jacob",
        )
        assert fake_voice.spoken == []
        assert results[0].status == "dropped"
        assert "smoke_signal" in results[0].reason

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
        results = await dispatch_outputs(
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
        # One result per input entry, in order.
        assert [r.status for r in results] == [
            "dropped", "dropped", "dropped", "delivered",
        ]

    @pytest.mark.asyncio
    async def test_no_voice_session_is_reported_dropped(self, monkeypatch):
        import boxbot.communication.voice as voice_mod
        monkeypatch.setattr(voice_mod, "_voice_session", None, raising=False)
        results = await dispatch_outputs(
            [{"to": "room", "channel": "voice", "content": "anyone?"}],
            conversation_id="c10",
            channel_context="voice",
            current_speaker=None,
        )
        assert results[0].status == "dropped"
        assert "voice session" in results[0].reason


# ---------------------------------------------------------------------------
# Degenerate content — filler and self-referential noise the agent emits
# once it has run out of real work. Observed on-device: "placeholder" and
# "done" delivered to a user, then texts apologising for those texts.
# ---------------------------------------------------------------------------


class TestDegenerateContent:

    @pytest.mark.parametrize("content", [
        "placeholder",
        "Placeholder",
        "test",
        "(no further action needed for now)",
        "(nothing to report)",
        "ignore my last message",
        "Please ignore my previous message.",
        "Sorry for the noise.",
        "sorry about the spam",
        "Disregard the stray message.",
        "Oops — ignore that last text.",
        "   ",
    ])
    def test_degenerate(self, content):
        assert _is_degenerate_content(content)

    @pytest.mark.parametrize("content", [
        "Yes.",
        "No.",
        "Done — the lights are off.",
        "OK, I've added milk to the list.",
        "Test results are in: 14 passed.",
        "None of your triggers fired today.",
        # A genuine apology that carries real content: it opens like
        # noise but the answer is right there behind it.
        "Sorry for the noise above — ignore the placeholder text, the "
        "garage door is closed.",
        "Sorry about the spam earlier. Your 3pm moved to 4pm.",
        "I put a placeholder in the spec where the number goes.",
    ])
    def test_deliverable(self, content):
        assert not _is_degenerate_content(content)

    @pytest.mark.parametrize("content", ["done", "Done.", "ok", "OK!", "n/a", "none"])
    def test_bare_acknowledgement_is_filler_only_on_a_trigger(self, content):
        """"Did you lock the door?" — "Done." is the answer, not noise."""
        assert _is_degenerate_content(content, "trigger")
        for channel in ("voice", "signal", "whatsapp"):
            assert not _is_degenerate_content(content, channel), channel

    @pytest.mark.asyncio
    async def test_terse_answer_is_spoken(self, fake_voice, fake_auth):
        results = await dispatch_outputs(
            [{"to": "Jacob", "channel": "voice", "content": "Done."}],
            conversation_id="c12",
            channel_context="voice",
            current_speaker="Jacob",
        )
        assert results[0].status == "delivered"
        assert fake_voice.spoken == ["Done."]

    @pytest.mark.asyncio
    async def test_filler_is_dropped_before_delivery(
        self, fake_voice, fake_auth, fake_whatsapp
    ):
        results = await dispatch_outputs(
            [
                {"to": "Jacob", "channel": "text", "content": "placeholder"},
                {"to": "Jacob", "channel": "text", "content": "Bins go out."},
            ],
            conversation_id="c11",
            channel_context="trigger",
            current_speaker=None,
        )
        assert [r.status for r in results] == ["dropped", "delivered"]
        assert fake_whatsapp.sent == [("+15551111111", "Bins go out.")]
        # The drop reason points the agent at the flag, not at a retry.
        assert "final_turn=true" in results[0].reason
        assert results[0].reason_code == DEGENERATE_CONTENT

    @pytest.mark.asyncio
    async def test_leaked_tool_syntax_is_tagged_unretryable(
        self, fake_voice, fake_auth, fake_whatsapp
    ):
        """Regenerating the same broken generation is not a recovery, so
        this drop must not hold a trigger turn open either."""
        # Assembled, not written literally: a real fragment in this file
        # would be a tripwire for anything else scanning the tree.
        leaked = "Bins go out." + "</" + "invoke>"
        results = await dispatch_outputs(
            [{"to": "Jacob", "channel": "text", "content": leaked}],
            conversation_id="c13",
            channel_context="trigger",
            current_speaker=None,
        )
        assert results[0].status == "dropped"
        assert results[0].reason_code == TOOL_SYNTAX
        assert fake_whatsapp.sent == []

    def test_every_tagged_drop_is_declared_unretryable(self):
        """The tags exist for one consumer — agent._message_results_settled.
        A new one that is not in the set silently reopens the burn-the-cap
        bug, so the set is the definition, not a copy of it."""
        assert UNRETRYABLE_DROPS == {DEGENERATE_CONTENT, TOOL_SYNTAX, BUDGET_SPENT}


class _FakeSignalWithAttachments:
    name = "signal"

    def __init__(self):
        self.sent = []
        self.attachments = []

    async def send_text(self, phone: str, message: str) -> bool:
        self.sent.append((phone, message))
        return True

    async def send_attachment(self, phone, file_path, caption=None) -> bool:
        self.attachments.append((phone, file_path, caption))
        return True


class TestTextAttachments:
    @pytest.mark.asyncio
    async def test_first_attachment_carries_caption_rest_follow_bare(
        self, fake_voice, fake_auth, monkeypatch,
    ):
        import boxbot.communication.whatsapp as wa_mod

        out = _FakeSignalWithAttachments()
        out.name = "whatsapp"
        wa_mod.set_whatsapp_client(out)
        try:
            results = await dispatch_outputs(
                [{
                    "to": "Sarah", "channel": "text", "content": "Porch now.",
                    "attachments": ["/tmp/a.jpg", "/tmp/b.jpg"],
                }],
                conversation_id="c-att",
                channel_context="signal",
                current_speaker=None,
            )
        finally:
            wa_mod.set_whatsapp_client(None)
        assert [r.status for r in results] == ["delivered"]
        assert out.attachments == [
            ("+15552222222", "/tmp/a.jpg", "Porch now."),
            ("+15552222222", "/tmp/b.jpg", None),
        ]
        assert out.sent == []  # caption rode the attachment

    @pytest.mark.asyncio
    async def test_transport_without_attachments_sends_text_and_says_so(
        self, fake_voice, fake_auth, fake_whatsapp,
    ):
        results = await dispatch_outputs(
            [{
                "to": "Sarah", "channel": "text", "content": "Porch now.",
                "attachments": ["/tmp/a.jpg"],
            }],
            conversation_id="c-att2",
            channel_context="signal",
            current_speaker=None,
        )
        assert fake_whatsapp.sent == [("+15552222222", "Porch now.")]
        assert results[0].status == "delivered"
        assert "cannot carry attachments" in (results[0].reason or "")
