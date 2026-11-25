"""Agent-facing text that steers conversation flow.

These pin the *direction* of each rule, not its wording: every one of
them was once written the wrong way round and produced the behaviour it
describes in a real session (mic muted after a question; a request
answered with "What's your name?"; "tap the video on the screen").
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import boxbot.core.agent as agent_module
from boxbot.core.agent import BoxBotAgent, _render_static_system_prompt

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def prompt(mock_config) -> str:
    return _render_static_system_prompt("Jarvis", "hey jarvis")


def _section(prompt: str, heading: str) -> str:
    start = prompt.index(heading)
    end = prompt.find("\n## ", start + 1)
    return prompt[start:end if end != -1 else None]


# ---------------------------------------------------------------------------
# Muting
# ---------------------------------------------------------------------------


def test_mute_section_restricts_instead_of_urging(prompt):
    assert "be DECISIVE" not in prompt
    assert "SAME turn" not in prompt

    section = _section(prompt, "## Muting the mic")
    assert "only for sustained talk clearly not meant for you" in section
    assert "after you asked something" in section
    assert "when a reply to you is plausible" in section
    assert "single garbled" in section
    assert "Unsure: stay silent, do not mute." in section


def test_mute_section_states_the_configured_mic_sleep(mock_config):
    mock_config.voice.post_response_idle_seconds = 15.0
    section = _section(
        _render_static_system_prompt("Jarvis", "hey jarvis"),
        "## Muting the mic",
    )
    assert "The mic sleeps on its own after 15 s." in section

    mock_config.voice.post_response_idle_seconds = 0
    section = _section(
        _render_static_system_prompt("Jarvis", "hey jarvis"),
        "## Muting the mic",
    )
    assert "sleeps on its own" not in section


# ---------------------------------------------------------------------------
# Act, don't ask
# ---------------------------------------------------------------------------


def test_act_dont_ask_section(prompt):
    section = _section(prompt, "## Act, don't ask")
    assert "do it, then say what you did" in section
    for costly in ("buying", "sending for payment", "deleting",
                   "messaging someone"):
        assert costly in section
    assert "After speaking a question: `final_turn: true`" in section
    assert "no touch" in section


def test_final_turn_false_is_tied_to_a_tool_call(prompt):
    assert "`false` only beside a tool call" in prompt
    assert "Waiting for a person's answer is\n  `true`" in prompt


# ---------------------------------------------------------------------------
# No touch input: nothing the agent reads may suggest tapping the screen
# ---------------------------------------------------------------------------

_TOUCH = re.compile(
    r"\b(tap|taps|tapping|click|clicks|clicking|touchscreen|press play|"
    r"touch(?:es|ing)? the (?:screen|page|video|display))\b",
    re.IGNORECASE,
)
# A sentence that mentions touch only to forbid or deny it.
_NEGATED = re.compile(r"\b(never|no|not|cannot|can't)\b", re.IGNORECASE)


def _agent_facing_texts() -> dict[str, str]:
    from boxbot.tools.registry import get_tools

    texts: dict[str, str] = {
        str(p.relative_to(REPO)): p.read_text(encoding="utf-8")
        for p in sorted((REPO / "skills").rglob("*.md"))
    }
    for tool in get_tools():
        texts[f"tool:{tool.name}"] = (
            f"{tool.description}\n{tool.parameters}"
        )
    return texts


def test_no_agent_facing_text_suggests_touching_the_screen(prompt):
    texts = _agent_facing_texts()
    assert any(k.startswith("tool:") for k in texts)
    texts["system prompt"] = prompt

    offenders: list[str] = []
    for source, text in texts.items():
        # Sentence-ish units, so a prohibition covers its own clause only.
        for sentence in re.split(r"(?<=[.!?:;])\s+|\n\s*\n|\n\s*[-|*] ", text):
            if _TOUCH.search(sentence) and not _NEGATED.search(sentence):
                offenders.append(f"{source}: {' '.join(sentence.split())[:120]}")
    assert offenders == []


def test_web_display_doc_says_what_to_do_instead():
    doc = (REPO / "skills/bb/modules/display.md").read_text(encoding="utf-8")
    assert "the person taps the screen" not in doc
    assert "never tell anyone to tap or press play" in doc


# ---------------------------------------------------------------------------
# Onboarding must not swallow the request
# ---------------------------------------------------------------------------


def _onboarding_s1() -> str:
    from boxbot.skills.loader import load_skill

    body = load_skill(name="onboarding", root=REPO / "skills")
    return body[body.index("## §1"):body.index("## §2")]


def test_onboarding_serves_the_request_before_asking_a_name():
    s1 = _onboarding_s1()

    assert "End the turn. Wait." not in s1
    assert "Their request comes first" in s1
    assert "Never end a turn on the name question alone" in s1
    assert "By the\n     way, who am I talking to?" in s1
    assert "drop it" in s1
    # The request row comes before the greeting-only row.
    assert s1.index("A request or question") < s1.index("A greeting only")


def test_onboarding_keeps_identify_person_usage():
    s1 = _onboarding_s1()

    assert "**Pin it** with `identify_person`" in s1
    assert '`ref` = the session speaker ref' in s1
    assert 'identify_person(action="rename", name="Brian", new_name="Bri")' in s1
    for outcome in ("`create`", "`confirm`", "`correct`"):
        assert outcome in s1


def test_onboarding_no_longer_names_the_dead_low_tier():
    text = (REPO / "skills/onboarding/SKILL.md").read_text(encoding="utf-8")
    assert "low-confidence" not in text
    assert "`low`" not in text


@pytest.mark.asyncio
async def test_auto_injection_tells_the_agent_to_serve_first(
    mock_config, monkeypatch,
):
    a = BoxBotAgent.__new__(BoxBotAgent)
    a._inject_memories = AsyncMock(return_value=("", []))
    a._prefetch_injected = {}
    a._present_people = {}
    a._latest_speaker_identities = {
        "Speaker A": {
            "person_id": None, "person_name": None,
            "voice_tier": "unknown", "voice_score": 0.0, "source": "unknown",
        },
    }
    monkeypatch.setattr(agent_module, "get_presence_snapshot", lambda: None)
    monkeypatch.setattr(
        agent_module, "get_status_line", AsyncMock(return_value=""),
    )

    out = await a._prompt_dynamic_context(
        person_name=None, channel="voice", context={},
        initial_message="[Speaker A]: can you search for Titanic books?",
        conv=None,
    )

    head = out[out.index("## Onboarding skill (auto-loaded"):]
    preamble = head[:head.index("\n# Onboarding")]
    assert "do it first" in preamble
    assert "never a reply\nthat is only the name question" in preamble
    assert "if they address you directly" not in preamble
    assert "Speaking with" not in out
