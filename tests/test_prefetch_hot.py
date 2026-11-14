"""Tests for the hot-task prefetch cache (``boxbot.prefetch.hot``)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from boxbot.prefetch import hot as hot_mod
from boxbot.prefetch.bundle import PrefetchBundle
from boxbot.prefetch.hot import (
    HotTaskCache,
    _drop_already_loaded,
    _exemplar_fingerprint,
    content_fingerprint,
)
from boxbot.prefetch.request import PrefetchRequest
from boxbot.prefetch.store import kv_get, kv_put


async def _store(tmp_path):
    from boxbot.memory.store import MemoryStore

    store = MemoryStore(db_path=tmp_path / "m.db")
    await store.initialize()
    return store


def _task(name="locks", exemplars=("lock the door", "unlock the door")):
    return SimpleNamespace(name=name, exemplars=list(exemplars))


def _cfg(tasks=None, threshold=0.6, enabled=True):
    return SimpleNamespace(
        enabled=enabled,
        hot_tasks_enabled=True,
        hot_match_threshold=threshold,
        hot_tasks=tasks if tasks is not None else [_task()],
    )


def _req(**kw):
    defaults = dict(
        key="conv-1", key_kind="conversation", channel="voice",
        person="Jacob", text="lock the front door",
    )
    defaults.update(kw)
    return PrefetchRequest(**defaults)


_VEC = np.ones(4, dtype=np.float32) / 2.0  # unit vector
_OTHER = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32) / 2.0  # ⟂ to _VEC


@pytest.fixture
def hot_env(monkeypatch):
    """Deterministic embeddings + config + content fingerprint."""
    monkeypatch.setattr(
        "boxbot.prefetch.hot.content_fingerprint", lambda: "fp-content",
    )
    monkeypatch.setattr(
        "boxbot.memory.embeddings.embed", lambda text: _VEC.copy(),
    )
    monkeypatch.setattr(
        "boxbot.memory.embeddings.active_model", lambda: "test-embedder",
    )
    cfg = _cfg()
    monkeypatch.setattr(
        "boxbot.prefetch.get_prefetch_config", lambda: cfg,
    )

    async def _no_memories(store, *, text, person, query_embedding=None):
        return []

    monkeypatch.setattr(
        "boxbot.prefetch.sources.gather_memory_candidates", _no_memories,
    )
    return cfg


async def _seed(store, cfg, *, bundle_dict, centroid=_VEC):
    fp = _exemplar_fingerprint(list(cfg.hot_tasks), "test-embedder")
    await kv_put(store, "hot:__centroids__", {
        "fingerprint": fp,
        "vectors": {cfg.hot_tasks[0].name: [float(x) for x in centroid]},
    })
    await kv_put(store, "hot:locks", {
        "fingerprint": "fp-content", "bundle": bundle_dict,
    })


class TestKvRows:
    async def test_roundtrip_and_overwrite(self, tmp_path):
        store = await _store(tmp_path)
        try:
            await kv_put(store, "hot:x", {"a": 1})
            await kv_put(store, "hot:x", {"a": 2})
            assert await kv_get(store, "hot:x") == {"a": 2}
            assert await kv_get(store, "hot:missing") is None
        finally:
            await store.close()


class TestLookup:
    async def test_hit_returns_canned_bundle_with_fresh_memories(
        self, tmp_path, hot_env, monkeypatch,
    ):
        store = await _store(tmp_path)
        try:
            canned = PrefetchBundle(
                skill_bodies={"home-control": "skill body"},
                sdk_modules={"panel": "panel sections"},
                sdk_sections={"panel": ["panel.1"]},
            )
            await _seed(store, hot_env, bundle_dict=canned.to_dict())

            async def _memories(store_, *, text, person, query_embedding=None):
                assert query_embedding is not None  # embedding is reused
                return [
                    SimpleNamespace(id=f"m{i}", summary=f"s{i}")
                    for i in range(8)
                ]

            monkeypatch.setattr(
                "boxbot.prefetch.sources.gather_memory_candidates",
                _memories,
            )

            bundle = await HotTaskCache().lookup(_req(), store=store)

            assert bundle is not None
            assert bundle.skill_bodies == {"home-control": "skill body"}
            assert len(bundle.memories) == 5  # top-K cap
            assert bundle.memories[0] == ("m0", "s0")
        finally:
            await store.close()

    async def test_below_threshold_is_a_miss(self, tmp_path, hot_env):
        store = await _store(tmp_path)
        try:
            await _seed(
                store, hot_env,
                bundle_dict=PrefetchBundle().to_dict(),
                centroid=_OTHER,  # orthogonal to the utterance embedding
            )
            assert await HotTaskCache().lookup(_req(), store=store) is None
        finally:
            await store.close()

    async def test_stale_fingerprint_misses_and_kicks_rebuild(
        self, tmp_path, hot_env, monkeypatch,
    ):
        store = await _store(tmp_path)
        try:
            await _seed(store, hot_env, bundle_dict=PrefetchBundle().to_dict())
            # Content changed since the bundle was built.
            monkeypatch.setattr(
                "boxbot.prefetch.hot.content_fingerprint", lambda: "fp-NEW",
            )
            cache = HotTaskCache()
            built = asyncio.Event()

            async def _fake_build(store_):
                built.set()

            monkeypatch.setattr(cache, "_build_safe", _fake_build)

            assert await cache.lookup(_req(), store=store) is None
            await asyncio.wait_for(built.wait(), 1.0)
        finally:
            await store.close()

    async def test_no_embedding_backend_is_a_miss(
        self, tmp_path, hot_env, monkeypatch,
    ):
        store = await _store(tmp_path)
        try:
            monkeypatch.setattr(
                "boxbot.memory.embeddings.embed", lambda text: None,
            )
            assert await HotTaskCache().lookup(_req(), store=store) is None
        finally:
            await store.close()

    async def test_empty_canned_bundle_is_still_a_hit(
        self, tmp_path, hot_env,
    ):
        store = await _store(tmp_path)
        try:
            await _seed(store, hot_env, bundle_dict=PrefetchBundle().to_dict())
            bundle = await HotTaskCache().lookup(_req(), store=store)
            assert bundle is not None and bundle.is_empty()
        finally:
            await store.close()


class TestMasterSwitch:
    def test_prefetch_disabled_kills_the_hot_path(self, monkeypatch):
        """prefetch.enabled=false must gate lookup AND the warm build
        (the operator kill switch covers the whole layer)."""
        cfg = _cfg(enabled=False)
        monkeypatch.setattr(
            "boxbot.prefetch.get_prefetch_config", lambda: cfg,
        )
        assert HotTaskCache._config() is None

    def test_reserved_centroid_name_is_excluded(self):
        cfg = _cfg(tasks=[_task(), _task(name="__centroids__")])
        names = [t.name for t in HotTaskCache._tasks(cfg)]
        assert names == ["locks"]

    async def test_empty_content_fingerprint_is_a_miss(
        self, tmp_path, hot_env, monkeypatch,
    ):
        """A box that lost its skills tree must miss, not serve stale
        bundles behind a stable empty-walk digest."""
        store = await _store(tmp_path)
        try:
            await _seed(store, hot_env, bundle_dict=PrefetchBundle().to_dict())
            monkeypatch.setattr(
                "boxbot.prefetch.hot.content_fingerprint", lambda: "",
            )
            cache = HotTaskCache()
            monkeypatch.setattr(cache, "_kick_build", lambda store_: None)
            assert await cache.lookup(_req(), store=store) is None
        finally:
            await store.close()


class TestRelevanceBump:
    async def test_fresh_memories_bump_last_relevant_at(
        self, tmp_path, hot_env, monkeypatch,
    ):
        store = await _store(tmp_path)
        try:
            await _seed(store, hot_env, bundle_dict=PrefetchBundle().to_dict())

            async def _memories(store_, *, text, person, query_embedding=None):
                return [SimpleNamespace(id="m0", summary="s0")]

            monkeypatch.setattr(
                "boxbot.prefetch.sources.gather_memory_candidates",
                _memories,
            )
            bumped: list[str] = []

            async def _bump(mid):
                bumped.append(mid)

            monkeypatch.setattr(store, "update_memory_relevance", _bump)

            bundle = await HotTaskCache().lookup(_req(), store=store)

            assert bundle is not None
            assert bumped == ["m0"]
        finally:
            await store.close()


class TestBuild:
    async def test_build_creates_centroids_and_bundles(
        self, tmp_path, hot_env, monkeypatch,
    ):
        store = await _store(tmp_path)
        try:
            monkeypatch.setattr(
                "boxbot.memory.embeddings.embed_batch",
                lambda texts: [_VEC.copy() for _ in texts],
            )
            monkeypatch.setattr(
                "boxbot.prefetch.resolve_client",
                lambda: SimpleNamespace(provider="openai", model="m", client=None),
            )
            captured: dict = {}

            async def _fake_run(req, *, store, client, config):
                captured["sources"] = list(config.sources)
                captured["pull_sources"] = list(config.pull_sources)
                bundle = PrefetchBundle(skill_bodies={"home-control": "b"})
                return SimpleNamespace(bundle=bundle)

            monkeypatch.setattr(
                "boxbot.prefetch.runner.run_prefetch", _fake_run,
            )

            cache = HotTaskCache()
            await cache._build(store)

            # Only the cacheable lanes run; pulls disabled.
            assert captured["sources"] == ["skills", "sdk"]
            assert captured["pull_sources"] == []
            row = await kv_get(store, "hot:locks")
            assert row["fingerprint"] == "fp-content"
            assert row["bundle"]["skill_bodies"] == {"home-control": "b"}
            cent = await kv_get(store, "hot:__centroids__")
            assert "locks" in cent["vectors"]

            # A fresh cache instance now matches end-to-end.
            bundle = await cache.lookup(_req(), store=store)
            assert bundle is not None
            assert bundle.skill_bodies == {"home-control": "b"}
        finally:
            await store.close()


class TestDropAlreadyLoaded:
    def _bundle(self):
        return PrefetchBundle(
            skill_bodies={"home-control": "b"},
            sdk_modules={"panel": "spliced", "display": "spliced"},
            sdk_sections={
                "panel": ["panel.1", "panel.2"],
                "display": ["display.0"],
            },
        )

    def test_drops_loaded_skill_and_fully_covered_sections(self):
        b = self._bundle()
        _drop_already_loaded(
            b, {"home-control", "panel.1", "panel.2"},
        )
        assert b.skill_bodies == {}
        assert set(b.sdk_modules) == {"display"}

    def test_whole_module_load_drops_its_sections(self):
        b = self._bundle()
        _drop_already_loaded(b, {"bb/modules/display.md"})
        assert set(b.sdk_modules) == {"panel"}

    def test_partial_overlap_keeps_the_module(self):
        b = self._bundle()
        _drop_already_loaded(b, {"panel.1"})
        assert set(b.sdk_modules) == {"panel", "display"}


class TestContentFingerprint:
    def test_changes_when_a_doc_changes(self, tmp_path, monkeypatch):
        root = tmp_path / "skill"
        (root / "modules").mkdir(parents=True)
        (root / "SKILL.md").write_text("skill")
        (root / "modules" / "a.md").write_text("doc a")

        monkeypatch.setattr(
            "boxbot.skills.loader.discover_skills",
            lambda: [SimpleNamespace(root_path=root)],
        )
        fp1 = content_fingerprint()
        (root / "modules" / "a.md").write_text("doc a CHANGED — longer")
        assert content_fingerprint() != fp1
