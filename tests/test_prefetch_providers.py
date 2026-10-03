"""Live-context providers + their ride in bundles and hot lookups.

Contract under test: providers are main-process, budgeted, and
omit-on-failure; live lines render first in the bundle and are NEVER
persisted (a cached bundle must not serve stale device state).
"""

from __future__ import annotations

import asyncio

import pytest

from boxbot.prefetch.bundle import PrefetchBundle
from boxbot.prefetch.providers import (
    register_provider,
    resolve_live,
)




class TestResolveLive:
    @pytest.mark.asyncio
    async def test_lines_come_back_in_declaration_order(self):
        async def a():
            return "line-a"

        async def b():
            return "line-b"

        register_provider("_test_a", a)
        register_provider("_test_b", b)
        assert await resolve_live(["_test_b", "_test_a"]) == [
            "line-b", "line-a",
        ]

    @pytest.mark.asyncio
    async def test_unknown_and_failing_and_none_are_omitted(self):
        async def ok():
            return "fine"

        async def boom():
            raise RuntimeError("bridge exploded")

        async def nothing():
            return None

        register_provider("_test_ok", ok)
        register_provider("_test_boom", boom)
        register_provider("_test_none", nothing)
        lines = await resolve_live(
            ["_test_boom", "_test_ok", "_test_none", "_test_unregistered"]
        )
        assert lines == ["fine"]

    @pytest.mark.asyncio
    async def test_over_budget_provider_is_dropped(self):
        async def slow():
            await asyncio.sleep(5)
            return "too late"

        async def fast():
            return "on time"

        register_provider("_test_slow", slow)
        register_provider("_test_fast", fast)
        lines = await resolve_live(
            ["_test_slow", "_test_fast"], budget_seconds=0.05
        )
        assert lines == ["on time"]


class TestBundleLiveContext:
    def test_live_lines_render_first_and_size_logged(self):
        bundle = PrefetchBundle(
            live_context=["[locks] Front Door: Unlocked"],
            memories=[("m1", "the front door sticks")],
        )
        rendered = bundle.render(token_budget=1000)
        assert "Live device state" in rendered
        assert rendered.index("[locks]") < rendered.index("Relevant memories")
        assert "trust it, don't re-read" in rendered

    def test_live_context_is_never_persisted(self):
        bundle = PrefetchBundle(live_context=["[locks] stale"])
        d = bundle.to_dict()
        assert "live_context" not in d
        assert PrefetchBundle.from_dict(d).live_context == []

    def test_live_context_alone_makes_the_bundle_non_empty(self):
        assert PrefetchBundle(live_context=["x"]).is_empty() is False
        assert PrefetchBundle().is_empty() is True


class TestHotLookupLiveContext:
    @pytest.mark.asyncio
    async def test_hot_task_providers_resolve_into_the_bundle(
        self, monkeypatch
    ):
        from boxbot.prefetch.hot import HotTaskCache

        class _Task:
            name = "locks"
            exemplars = ["lock the door"]
            providers = ["_test_hot_line"]

        class _Cfg:
            hot_tasks = [_Task()]

        async def line():
            return "[locks] Front Door: Unlocked"

        register_provider("_test_hot_line", line)
        cache = HotTaskCache()
        lines = await cache._live_context(_Cfg(), "locks")
        assert lines == ["[locks] Front Door: Unlocked"]

    @pytest.mark.asyncio
    async def test_task_without_providers_resolves_nothing(self):
        from boxbot.prefetch.hot import HotTaskCache

        class _Task:
            name = "time_date"
            exemplars = ["what time is it"]
            providers: list[str] = []

        class _Cfg:
            hot_tasks = [_Task()]

        assert await HotTaskCache()._live_context(_Cfg(), "time_date") == []
