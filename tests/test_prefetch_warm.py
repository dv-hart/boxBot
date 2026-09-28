"""Tests for the draft-warmed prefetch path and the connection pre-warm.

TranscriptDraft (bare STT text, published before speaker resolution)
lets the agent start the prefetch lookup ~0.5s early;
``_prefetch_context_for_text`` must consume the warm result only when
session/text/person still match, and fall back to a fresh lookup on any
drift or failure.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import boxbot.core.agent as agent_module
from boxbot.core.agent import BoxBotAgent
from boxbot.core.events import ButtonPressed, TranscriptDraft
from boxbot.prefetch.bundle import PrefetchBundle


class _Conv:
    conversation_id = "c1"
    thread: list = []

    def __init__(self):
        self.accessed_memory_ids: list = []
        self.injected_memories_block = ""


def _agent(monkeypatch, *, active=True):
    a = BoxBotAgent.__new__(BoxBotAgent)
    a._prefetch_warm = None
    a._prefetch_injected = {}
    a._memory_store = None
    a._conn_warm_at = 0.0
    a._conn_warm_task = None
    a._get_voice_room_conversation = MagicMock(return_value=None)
    a._get_most_recent_person = MagicMock(return_value="Jacob")
    a._kick_openai_conn_warm = MagicMock()
    a._prefetch_already_loaded = MagicMock(return_value=[])
    monkeypatch.setattr(
        agent_module.prefetch_layer, "is_active", lambda: active
    )
    monkeypatch.setattr(
        agent_module.prefetch_layer, "should_prefetch", lambda _ch: True
    )
    return a


BUNDLE = PrefetchBundle(skill_bodies={"home-control": "body"})


@pytest.mark.asyncio
async def test_matching_warm_result_is_consumed(monkeypatch):
    a = _agent(monkeypatch)
    a._prefetch_lookup = AsyncMock(side_effect=AssertionError("fresh lookup ran"))
    task = asyncio.create_task(_return((BUNDLE, True)))
    a._prefetch_warm = ("vs1", "lock the door", "Jacob", None, task)

    ctx = await a._prefetch_context_for_text(
        _Conv(), "voice", "Jacob", "lock the door", warm_key="vs1"
    )
    assert "home-control" in ctx["prefetch_text"]
    assert a._prefetch_warm is None  # consumed


async def _return(value):
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stash", [
        ("vs1", "different text", "Jacob"),   # text drifted
        ("other-session", "lock the door", "Jacob"),  # stale session
    ],
)
async def test_mismatched_warm_is_discarded_and_fresh_runs(monkeypatch, stash):
    a = _agent(monkeypatch)
    a._prefetch_lookup = AsyncMock(return_value=(BUNDLE, True))
    task = asyncio.create_task(_return((BUNDLE, True)))
    a._prefetch_warm = (*stash, None, task)

    ctx = await a._prefetch_context_for_text(
        _Conv(), "voice", "Jacob", "lock the door", warm_key="vs1"
    )
    assert a._prefetch_lookup.await_count == 1
    assert "home-control" in ctx["prefetch_text"]
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_failed_warm_task_falls_back_to_fresh(monkeypatch):
    a = _agent(monkeypatch)
    a._prefetch_lookup = AsyncMock(return_value=(BUNDLE, True))

    async def _boom():
        raise RuntimeError("lane failed")

    task = asyncio.create_task(_boom())
    a._prefetch_warm = ("vs1", "lock the door", "Jacob", None, task)
    ctx = await a._prefetch_context_for_text(
        _Conv(), "voice", "Jacob", "lock the door", warm_key="vs1"
    )
    assert a._prefetch_lookup.await_count == 1
    assert "home-control" in ctx["prefetch_text"]


@pytest.mark.asyncio
async def test_text_channels_ignore_the_warm_stash(monkeypatch):
    a = _agent(monkeypatch)
    a._prefetch_lookup = AsyncMock(return_value=(None, False))
    task = asyncio.create_task(_return((BUNDLE, True)))
    a._prefetch_warm = ("vs1", "hi", "Jacob", None, task)

    ctx = await a._prefetch_context_for_text(_Conv(), "whatsapp", "Jacob", "hi")
    assert ctx == {}
    assert a._prefetch_warm is not None  # untouched: no warm_key given
    task.cancel()


@pytest.mark.asyncio
async def test_empty_hot_hit_injects_nothing(monkeypatch):
    a = _agent(monkeypatch)
    a._prefetch_lookup = AsyncMock(return_value=(PrefetchBundle(), True))
    ctx = await a._prefetch_context_for_text(_Conv(), "voice", "Jacob", "hi")
    assert ctx == {}


@pytest.mark.asyncio
async def test_draft_starts_warm_task_and_replaces_previous(monkeypatch):
    a = _agent(monkeypatch)
    a._prefetch_lookup = AsyncMock(return_value=(BUNDLE, True))

    await a._on_transcript_draft(TranscriptDraft(conversation_id="vs1", text="lock up"))
    assert a._prefetch_warm is not None
    key, text, person, _req, first_task = a._prefetch_warm
    assert (key, text, person) == ("vs1", "lock up", "Jacob")
    assert a._kick_openai_conn_warm.call_count == 1

    await a._on_transcript_draft(
        TranscriptDraft(conversation_id="vs1", text="never mind")
    )
    _, text2, _, _, second_task = a._prefetch_warm
    assert text2 == "never mind"
    assert second_task is not first_task
    await asyncio.gather(first_task, second_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_draft_is_a_noop_when_prefetch_inactive(monkeypatch):
    a = _agent(monkeypatch, active=False)
    await a._on_transcript_draft(TranscriptDraft(conversation_id="vs1", text="hi"))
    assert a._prefetch_warm is None


@pytest.mark.asyncio
async def test_draft_ignores_empty_text(monkeypatch):
    a = _agent(monkeypatch)
    await a._on_transcript_draft(TranscriptDraft(conversation_id="vs1", text="  "))
    assert a._prefetch_warm is None


@pytest.mark.asyncio
async def test_person_drift_on_hot_bundle_salvages_with_fresh_memories(
    monkeypatch,
):
    """First-turn case: draft saw no person, resolve identified one."""
    a = _agent(monkeypatch)
    a._prefetch_lookup = AsyncMock(side_effect=AssertionError("fresh lookup ran"))
    a._hot_prefetch = MagicMock()
    a._hot_prefetch.refresh_memories = AsyncMock(
        return_value=[("m1", "Jacob is vegetarian")]
    )
    bundle = PrefetchBundle(
        skill_bodies={"home-control": "body"},
        memories=[("stale", "wrong-person memory")],
    )
    req = object()
    task = asyncio.create_task(_return((bundle, True)))
    a._prefetch_warm = ("vs1", "lock the door", None, req, task)

    ctx = await a._prefetch_context_for_text(
        _Conv(), "voice", "Jacob", "lock the door", warm_key="vs1"
    )
    assert "prefetch_text" in ctx
    assert bundle.memories == [("m1", "Jacob is vegetarian")]
    a._hot_prefetch.refresh_memories.assert_awaited_once_with(
        req, store=a._memory_store, person="Jacob"
    )


@pytest.mark.asyncio
async def test_person_drift_on_fanout_bundle_falls_back_to_fresh(monkeypatch):
    """Fan-out lanes baked the wrong person in — discard, run fresh."""
    a = _agent(monkeypatch)
    a._prefetch_lookup = AsyncMock(return_value=(BUNDLE, False))
    task = asyncio.create_task(_return((BUNDLE, False)))
    a._prefetch_warm = ("vs1", "lock the door", None, None, task)

    ctx = await a._prefetch_context_for_text(
        _Conv(), "voice", "Jacob", "lock the door", warm_key="vs1"
    )
    assert a._prefetch_lookup.await_count == 1
    assert "home-control" in ctx["prefetch_text"]


# ---------------------------------------------------------------------------
# Connection pre-warm
# ---------------------------------------------------------------------------


def _warm_agent(monkeypatch, mock_config, *, provider="openai"):
    a = BoxBotAgent.__new__(BoxBotAgent)
    a._conn_warm_at = 0.0
    a._conn_warm_task = None
    a._warm_openai_connection = AsyncMock()
    monkeypatch.setattr(
        agent_module, "provider_for_model", lambda _m: provider
    )
    return a


@pytest.mark.asyncio
async def test_press_warms_connection_once_within_interval(
    monkeypatch, mock_config
):
    a = _warm_agent(monkeypatch, mock_config)
    await a._on_button_pressed(ButtonPressed(button_id="screen", action="press"))
    task = a._conn_warm_task
    assert task is not None
    await task
    # Second press inside the debounce window: no new task.
    await a._on_button_pressed(ButtonPressed(button_id="screen", action="press"))
    assert a._conn_warm_task is task
    assert a._warm_openai_connection.await_count == 1


@pytest.mark.asyncio
async def test_release_does_not_warm(monkeypatch, mock_config):
    a = _warm_agent(monkeypatch, mock_config)
    await a._on_button_pressed(ButtonPressed(button_id="screen", action="release"))
    assert a._conn_warm_task is None


@pytest.mark.asyncio
async def test_no_warm_for_anthropic_only_models(monkeypatch, mock_config):
    a = _warm_agent(monkeypatch, mock_config, provider="anthropic")
    await a._on_button_pressed(ButtonPressed(button_id="screen", action="press"))
    assert a._conn_warm_task is None


# ---------------------------------------------------------------------------
# Stage A (context-aware prefetch): follow-up hot-only, injection dedup,
# honest tracking
# ---------------------------------------------------------------------------


class _ConvWithReply:
    conversation_id = "c1"
    thread = [
        {"role": "user", "content": "lock the door"},
        {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
    ]


def test_followup_detection():
    assert BoxBotAgent._is_followup_turn(None) is False
    assert BoxBotAgent._is_followup_turn(_Conv()) is False
    assert BoxBotAgent._is_followup_turn(_ConvWithReply()) is True


@pytest.mark.asyncio
async def test_hot_only_skips_the_fanout_on_a_miss(monkeypatch):
    """Voice follow-up + hot miss = inject nothing; the selector fan-out
    must not run mid-conversation (it blocked replies +2-3s to re-pick
    in-thread content)."""
    a = _agent(monkeypatch)
    a._hot_prefetch = MagicMock()
    a._hot_prefetch.lookup = AsyncMock(return_value=None)
    a._run_prefetch = AsyncMock(side_effect=AssertionError("fan-out ran"))

    from boxbot.prefetch.request import PrefetchRequest

    req = PrefetchRequest(
        key="c1", key_kind="conversation", channel="voice", text="stay",
    )
    bundle, was_hot = await a._prefetch_lookup(req, hot_only=True)
    assert bundle is None and was_hot is False


@pytest.mark.asyncio
async def test_hot_only_still_serves_hot_hits(monkeypatch):
    a = _agent(monkeypatch)
    a._hot_prefetch = MagicMock()
    a._hot_prefetch.lookup = AsyncMock(return_value=BUNDLE)
    a._run_prefetch = AsyncMock(side_effect=AssertionError("fan-out ran"))

    from boxbot.prefetch.request import PrefetchRequest

    req = PrefetchRequest(
        key="c1", key_kind="conversation", channel="voice",
        text="lock the door",
    )
    bundle, was_hot = await a._prefetch_lookup(req, hot_only=True)
    assert bundle is BUNDLE and was_hot is True


@pytest.mark.asyncio
async def test_fanout_bundle_is_deduped_at_injection(monkeypatch):
    """The 4x-onboarding regression: a fresh fan-out bundle carrying an
    already-in-thread skill body must inject nothing."""
    a = _agent(monkeypatch)
    a._prefetch_already_loaded = MagicMock(return_value=["onboarding"])
    fresh = PrefetchBundle(skill_bodies={"onboarding": "big body"})
    a._prefetch_lookup = AsyncMock(return_value=(fresh, False))

    ctx = await a._prefetch_context_for_text(
        _ConvWithReply(), "signal", "Jacob", "hello again"
    )
    assert ctx == {}, "fully-deduped bundle must not inject"
    assert fresh.skill_bodies == {}


def test_tracking_never_overclaims_legacy_splices():
    """A splice with no section record (legacy cache format) records
    nothing — the whole-module fallback once claimed all of panel.md
    off a 588-token splice and dedup then gutted the next bundle."""
    a = BoxBotAgent.__new__(BoxBotAgent)
    a._prefetch_injected = {}
    legacy = PrefetchBundle(sdk_modules={"panel": "some splice"})
    a._track_prefetch_injected("c1", legacy)
    assert "bb/modules/panel.md" not in a._prefetch_injected["c1"]

    whole = PrefetchBundle(
        sdk_modules={"panel": "whole doc"},
        sdk_sections={"panel": ["panel"]},
    )
    a._track_prefetch_injected("c1", whole)
    assert "bb/modules/panel.md" in a._prefetch_injected["c1"]

    spliced = PrefetchBundle(
        sdk_modules={"camera": "splice"},
        sdk_sections={"camera": ["camera.0", "camera.2"]},
    )
    a._track_prefetch_injected("c1", spliced)
    assert {"camera.0", "camera.2"} <= a._prefetch_injected["c1"]
    assert "bb/modules/camera.md" not in a._prefetch_injected["c1"]


def test_drop_already_loaded_filters_memories():
    from boxbot.prefetch.hot import _drop_already_loaded

    b = PrefetchBundle(memories=[("m1", "a"), ("m2", "b")])
    _drop_already_loaded(b, {"m1"})
    assert b.memories == [("m2", "b")]
