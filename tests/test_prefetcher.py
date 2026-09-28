"""Tests for the prefetch selector fan-out (``boxbot.prefetch.runner``).

Fake OpenAI- and Anthropic-shaped clients, monkeypatched source
gathering — no network, no real store.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from boxbot.prefetch.bundle import PrefetchBundle
from boxbot.prefetch.request import PrefetchRequest
from boxbot.prefetch.runner import LLMHandle, run_prefetch
from boxbot.prefetch.sources import SourceRun


# --- fakes -----------------------------------------------------------------


def _openai_completion(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(payload)),
        )],
        usage={"prompt_tokens": 100, "completion_tokens": 10},
    )


class _FakeChatCompletions:
    def __init__(self, payloads):
        self._payloads = dict(payloads)  # response_format name -> payload
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        name = kwargs["response_format"]["json_schema"]["name"]
        return _openai_completion(self._payloads.get(name, {}))


class _FakeOpenAIClient:
    def __init__(self, payloads):
        self.chat = SimpleNamespace(
            completions=_FakeChatCompletions(payloads)
        )


class _FakeAnthropicMessages:
    def __init__(self, payloads):
        self._payloads = dict(payloads)  # tool name -> selection
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        name = kwargs["tool_choice"]["name"]
        return SimpleNamespace(
            content=[SimpleNamespace(
                type="tool_use", input=self._payloads.get(name, {}),
            )],
            usage={"input_tokens": 100, "output_tokens": 10},
        )


class _FakeAnthropicClient:
    def __init__(self, payloads):
        self.messages = _FakeAnthropicMessages(payloads)


def _memory_source(seen: dict[str, str]) -> SourceRun:
    async def materialize(sel, bundle: PrefetchBundle):
        for mid in sel.get("memory_ids", []):
            if mid in seen:
                bundle.memories.append((mid, seen[mid]))

    return SourceRun(
        name="memory",
        instructions="pick memories",
        schema={
            "type": "object",
            "properties": {
                "memory_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["memory_ids"],
        },
        candidates="\n".join(f"- {k}: {v}" for k, v in seen.items()),
        materialize=materialize,
    )


def _skills_source() -> SourceRun:
    async def materialize(sel, bundle: PrefetchBundle):
        for name in sel.get("skills", [])[:1]:
            bundle.skill_bodies[name] = f"body of {name}"

    return SourceRun(
        name="skills",
        instructions="pick a skill",
        schema={
            "type": "object",
            "properties": {
                "skills": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["skills"],
        },
        candidates="- household-support: household support",
        materialize=materialize,
    )


_CFG = SimpleNamespace(
    model=None, token_budget=20000, per_call_timeout_seconds=5.0,
)


def _req(**kw):
    defaults = dict(
        key="conv-1", key_kind="conversation", channel="signal",
        person="Jacob", text="how do I pair a bulb",
    )
    defaults.update(kw)
    return PrefetchRequest(**defaults)


def _patch_sources(monkeypatch, sources):
    async def fake_gather(req, *, store, config):
        return list(sources)

    monkeypatch.setattr("boxbot.prefetch.runner.gather_sources", fake_gather)


# --- tests -----------------------------------------------------------------


class TestFanOut:
    async def test_parallel_lanes_assemble_bundle(self, monkeypatch):
        seen = {"mem-1": "Jacob likes strong tea"}
        _patch_sources(monkeypatch, [_memory_source(seen), _skills_source()])
        client = _FakeOpenAIClient({
            "prefetch_memory": {"memory_ids": ["mem-1"]},
            "prefetch_skills": {"skills": ["household-support"]},
        })
        handle = LLMHandle(provider="openai", model="gpt-5.6-luna", client=client)

        result = await run_prefetch(_req(), store=object(), client=handle, config=_CFG)

        assert result.iterations == 2
        assert result.bundle.memories == [("mem-1", "Jacob likes strong tea")]
        assert result.bundle.predicted_skills() == ["household-support"]
        # One call per lane, no loop.
        assert len(client.chat.completions.calls) == 2

    async def test_luna_gets_reasoning_effort_floor(self, monkeypatch):
        _patch_sources(monkeypatch, [_skills_source()])
        client = _FakeOpenAIClient({"prefetch_skills": {"skills": []}})
        handle = LLMHandle(provider="openai", model="gpt-5.6-luna", client=client)

        await run_prefetch(_req(), store=object(), client=handle, config=_CFG)

        call = client.chat.completions.calls[0]
        assert call["reasoning_effort"] == "none"
        assert call["response_format"]["json_schema"]["strict"] is True

    async def test_empty_selections_yield_empty_bundle(self, monkeypatch):
        _patch_sources(monkeypatch, [_memory_source({}), _skills_source()])
        client = _FakeOpenAIClient({
            "prefetch_memory": {"memory_ids": []},
            "prefetch_skills": {"skills": []},
        })
        handle = LLMHandle(provider="openai", model="gpt-5.6-luna", client=client)

        result = await run_prefetch(_req(), store=object(), client=handle, config=_CFG)

        assert result.bundle.is_empty()

    async def test_no_sources_makes_no_calls(self, monkeypatch):
        _patch_sources(monkeypatch, [])
        client = _FakeOpenAIClient({})
        handle = LLMHandle(provider="openai", model="gpt-5.6-luna", client=client)

        result = await run_prefetch(_req(), store=object(), client=handle, config=_CFG)

        assert result.iterations == 0
        assert result.bundle.is_empty()
        assert not client.chat.completions.calls

    async def test_anthropic_fallback_uses_forced_tool(self, monkeypatch):
        _patch_sources(monkeypatch, [_skills_source()])
        client = _FakeAnthropicClient({
            "select_skills": {"skills": ["household-support"]},
        })
        handle = LLMHandle(
            provider="anthropic", model="claude-haiku-4-5-20251001",
            client=client,
        )

        result = await run_prefetch(_req(), store=object(), client=handle, config=_CFG)

        assert result.bundle.predicted_skills() == ["household-support"]
        call = client.messages.calls[0]
        assert call["tool_choice"] == {"type": "tool", "name": "select_skills"}

    async def test_one_lane_failing_does_not_sink_the_rest(self, monkeypatch):
        seen = {"mem-1": "s1"}

        class _Boom:
            async def create(self, **kwargs):
                name = kwargs["response_format"]["json_schema"]["name"]
                if name == "prefetch_skills":
                    raise RuntimeError("lane down")
                return _openai_completion({"memory_ids": ["mem-1"]})

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=_Boom())
        )
        _patch_sources(monkeypatch, [_memory_source(seen), _skills_source()])
        handle = LLMHandle(provider="openai", model="gpt-5.6-luna", client=client)

        result = await run_prefetch(_req(), store=object(), client=handle, config=_CFG)

        assert result.bundle.memories == [("mem-1", "s1")]
        assert not result.bundle.skill_bodies

    async def test_slow_lane_times_out_alone(self, monkeypatch):
        class _Slow:
            async def create(self, **kwargs):
                name = kwargs["response_format"]["json_schema"]["name"]
                if name == "prefetch_skills":
                    await asyncio.sleep(1.0)
                return _openai_completion({"memory_ids": ["mem-1"]})

        client = SimpleNamespace(chat=SimpleNamespace(completions=_Slow()))
        _patch_sources(
            monkeypatch, [_memory_source({"mem-1": "s1"}), _skills_source()],
        )
        cfg = SimpleNamespace(
            model=None, token_budget=20000, per_call_timeout_seconds=0.05,
        )
        handle = LLMHandle(provider="openai", model="gpt-5.6-luna", client=client)

        result = await run_prefetch(_req(), store=object(), client=handle, config=cfg)

        assert result.bundle.memories == [("mem-1", "s1")]
        assert not result.bundle.skill_bodies

    async def test_briefing_excludes_private_notes(self):
        req = _req(recent_thread_tail=[
            {"role": "user", "content": "[Jacob]: hello"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "name": "message",
                 "input": {"content": "delivered reply"}},
                {"type": "text", "text": '{"thought":"private"}'},
            ]},
        ])
        b = req.briefing()
        assert "delivered reply" in b
        assert "private" not in b

    async def test_briefing_is_capped(self):
        req = _req(
            text="x" * 5000,
            recent_thread_tail=[
                {"role": "user", "content": "y" * 5000} for _ in range(10)
            ],
        )
        assert len(req.briefing()) <= 900  # ~200 tokens + slack

    async def test_briefing_lists_already_loaded(self):
        req = _req(already_loaded=["bb", "bb/modules/panel.md"])
        assert "bb/modules/panel.md" in req.briefing()
