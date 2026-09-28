"""Hot-task prefetch: canned bundles for the household's common requests.

A hot task is a recurring, formulaic request ("show me the garage",
"lock the door", "arm the alarm"). Instead of paying the selector
fan-out on the reply path, each configured task's skills+sdk selection
is precomputed once — through the normal selector code path — and
cached in ``prefetch_cache``. At message time the utterance embedding
(reused for memory retrieval, so it costs nothing extra) is matched
against per-task exemplar centroids; a hit injects the canned bundle
plus fresh deterministic memory retrieval, and no selector call runs.

Validity is content-addressed, not time-based: every cached bundle
carries a fingerprint over the skill/SDK doc files, and the centroid
row carries one over the exemplar list + embedding backend. A mismatch
(deploy, runtime skill edit, embedding-model swap) reads as a miss and
kicks ONE debounced background rebuild — self-healing, no hook wiring.

Everything here is best-effort: any failure is a miss and the normal
selector fan-out runs instead.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

from boxbot.prefetch.bundle import PrefetchBundle
from boxbot.prefetch.request import PrefetchRequest
from boxbot.prefetch.store import kv_get, kv_put, record_prefetch_event

logger = logging.getLogger(__name__)

_CENTROIDS_KEY = "hot:__centroids__"
_MAX_MEMORIES = 5


def _bundle_key(task: str) -> str:
    return f"hot:{task}"


def content_fingerprint() -> str:
    """Fingerprint of every skill body + bb module doc.

    (path, mtime_ns, size) tuples — cheap enough to check on every
    lookup, and skill/SDK docs only change on deploy or a runtime skill
    edit, both of which touch mtimes.
    """
    entries: list[tuple[str, int, int]] = []

    def _stat(path: Any) -> None:
        try:
            st = path.stat()
            entries.append((str(path), st.st_mtime_ns, st.st_size))
        except OSError:
            pass

    try:
        from boxbot.skills.loader import discover_skills

        for meta in discover_skills():
            root = getattr(meta, "root_path", None)
            if root is None:
                continue
            _stat(root / "SKILL.md")
            modules_dir = root / "modules"
            if modules_dir.is_dir():
                for f in sorted(modules_dir.glob("*.md")):
                    _stat(f)
    except Exception:
        logger.debug("content fingerprint walk failed", exc_info=True)
    if not entries:
        # A box that lost its skills tree must MISS (and rebuild), not
        # serve stale bundles behind a stable empty-walk digest.
        return ""
    return hashlib.sha256(json.dumps(sorted(entries)).encode()).hexdigest()


def _exemplar_fingerprint(tasks: list[Any], backend: str) -> str:
    payload = json.dumps(
        [[t.name, list(t.exemplars)] for t in tasks], sort_keys=True
    )
    return hashlib.sha256(f"{backend}|{payload}".encode()).hexdigest()


class HotTaskCache:
    """Matcher + cache for hot-task bundles. One instance per agent."""

    def __init__(self) -> None:
        # task name -> centroid vector, valid for self._centroid_fp.
        self._centroids: dict[str, Any] = {}
        self._centroid_fp: str | None = None
        self._build_task: asyncio.Task[None] | None = None

    # -- config plumbing ------------------------------------------------

    @staticmethod
    def _config() -> Any:
        from boxbot.prefetch import get_prefetch_config

        cfg = get_prefetch_config()
        if (
            cfg is None
            # Master switch: prefetch.enabled=false must kill the hot
            # path too, not just the selector fan-out.
            or not getattr(cfg, "enabled", False)
            or not getattr(cfg, "hot_tasks_enabled", False)
            or not getattr(cfg, "hot_tasks", None)
        ):
            return None
        return cfg

    @staticmethod
    def _tasks(cfg: Any) -> list[Any]:
        """Configured tasks minus the reserved centroid-row name."""
        return [t for t in cfg.hot_tasks if t.name != "__centroids__"]

    # -- lookup (reply path) --------------------------------------------

    async def lookup(
        self, req: PrefetchRequest, *, store: Any
    ) -> PrefetchBundle | None:
        """Canned bundle for a recognized hot task, or None (= miss).

        A returned EMPTY bundle is still a hit: the task predictably
        needs no extra context, so the caller should skip the selector
        fan-out and inject nothing.
        """
        cfg = self._config()
        text = (req.text or "").strip()
        if cfg is None or not text or store is None:
            return None
        t0 = time.monotonic()

        from boxbot.memory.embeddings import cosine_similarity, embed

        embedding = await asyncio.to_thread(embed, text)
        if embedding is None:
            return None
        # Hand the embedding to whoever runs next (the memory lane on a
        # miss, _fresh_memories on a hit) — never embed the text twice.
        req.query_embedding = embedding

        if not await self._centroids_ready(store, cfg):
            self._kick_build(store)
            return None

        threshold = float(getattr(cfg, "hot_match_threshold", 0.60))
        task, score = None, threshold
        for name, centroid in self._centroids.items():
            sim = cosine_similarity(embedding, centroid)
            if sim >= score:
                task, score = name, sim
        if task is None:
            return None

        row = await kv_get(store, _bundle_key(task))
        fp = content_fingerprint()
        if row is None or not fp or row.get("fingerprint") != fp:
            self._kick_build(store)
            return None
        try:
            bundle = PrefetchBundle.from_dict(row.get("bundle") or {})
        except Exception:
            return None

        already = set(req.already_loaded or ())
        _drop_already_loaded(bundle, already)
        bundle.memories = await self._fresh_memories(
            store, req, embedding, already
        )
        bundle.live_context = await self._live_context(cfg, task)
        logger.info(
            "hot prefetch hit task=%s score=%.2f key=%s empty=%s (%.0fms)",
            task, score, req.key, bundle.is_empty(),
            (time.monotonic() - t0) * 1000,
        )
        try:
            await record_prefetch_event(
                store,
                key=req.key,
                key_kind=req.key_kind,
                channel=req.channel,
                mode="hot",
                bundle=bundle,
                latency_ms=int((time.monotonic() - t0) * 1000),
                cost_usd=0.0,
            )
        except Exception:
            logger.debug("hot prefetch event write failed", exc_info=True)
        return bundle

    async def refresh_memories(
        self, req: PrefetchRequest, *, store: Any, person: str | None,
    ) -> list[tuple[str, str]]:
        """Re-run the memory pass of a completed lookup for ``person``.

        Salvage path for a draft-warmed hot bundle whose person resolved
        differently after the fact: skills/sdk content is
        person-independent, only the memory pick needs redoing. Reuses
        ``req.query_embedding`` (set by :meth:`lookup`); the caller
        keeps the rest of the bundle.
        """
        req.person = person
        return await self._fresh_memories(
            store, req, req.query_embedding, set(req.already_loaded or ()),
        )

    async def _live_context(self, cfg: Any, task: str) -> list[str]:
        """Resolve the task's live-state providers (fresh every hit).

        Main-process reads only, hard-budgeted, omit-on-failure — see
        ``prefetch/providers.py``. Salvage after person drift keeps
        these lines untouched: device state is person-independent.
        """
        task_cfg = next(
            (t for t in self._tasks(cfg) if t.name == task), None
        )
        names = list(getattr(task_cfg, "providers", None) or ())
        if not names:
            return []
        from boxbot.prefetch.providers import resolve_live

        return await resolve_live(names)

    async def _fresh_memories(
        self,
        store: Any,
        req: PrefetchRequest,
        embedding: Any,
        already: set[str],
    ) -> list[tuple[str, str]]:
        """Top-K deterministic retrieval — no selector filtering."""
        from boxbot.prefetch.sources import _bump_relevance, gather_memory_candidates

        cands = await gather_memory_candidates(
            store, text=req.text, person=req.person, query_embedding=embedding,
        )
        picked = [
            (c.id, c.summary) for c in cands if c.id not in already
        ][:_MAX_MEMORIES]
        for mid, _ in picked:
            await _bump_relevance(store, mid)
        return picked

    async def _centroids_ready(self, store: Any, cfg: Any) -> bool:
        from boxbot.memory.embeddings import active_model

        backend = active_model()
        if backend is None:
            return False
        fp = _exemplar_fingerprint(self._tasks(cfg), backend)
        if self._centroid_fp == fp and self._centroids:
            return True
        row = await kv_get(store, _CENTROIDS_KEY)
        if row is None or row.get("fingerprint") != fp:
            return False
        import numpy as np

        vectors = row.get("vectors") or {}
        self._centroids = {
            name: np.asarray(vec, dtype=np.float32)
            for name, vec in vectors.items()
        }
        self._centroid_fp = fp
        return bool(self._centroids)

    # -- build (background) ----------------------------------------------

    def _kick_build(self, store: Any) -> None:
        """Debounced background (re)build of centroids + bundles."""
        if self._build_task is not None and not self._build_task.done():
            return
        self._build_task = asyncio.create_task(self._build_safe(store))

    async def ensure_built(self, store: Any) -> None:
        """Warm build at agent start (no-op when everything is fresh)."""
        if self._config() is not None and store is not None:
            self._kick_build(store)

    async def _build_safe(self, store: Any) -> None:
        try:
            await self._build(store)
        except Exception:
            logger.exception("hot prefetch build failed")

    async def _build(self, store: Any) -> None:
        cfg = self._config()
        if cfg is None:
            return
        tasks = self._tasks(cfg)

        from boxbot.memory.embeddings import active_model, embed_batch

        backend = active_model()
        if backend is not None:
            cent_fp = _exemplar_fingerprint(tasks, backend)
            row = await kv_get(store, _CENTROIDS_KEY)
            if row is None or row.get("fingerprint") != cent_fp:
                import numpy as np

                vectors: dict[str, list[float]] = {}
                for t in tasks:
                    embs = await asyncio.to_thread(
                        embed_batch, list(t.exemplars)
                    )
                    vecs = [e for e in embs if e is not None]
                    if not vecs:
                        continue
                    centroid = np.mean(np.stack(vecs), axis=0)
                    norm = float(np.linalg.norm(centroid))
                    if norm > 0:
                        centroid = centroid / norm
                    vectors[t.name] = [float(x) for x in centroid]
                if vectors:
                    await kv_put(store, _CENTROIDS_KEY, {
                        "fingerprint": cent_fp, "vectors": vectors,
                    })
                    logger.info(
                        "hot prefetch centroids built: %d tasks (%s)",
                        len(vectors), backend,
                    )
            await self._centroids_ready(store, cfg)

        from boxbot.prefetch import resolve_client
        from boxbot.prefetch.runner import run_prefetch

        client = resolve_client()
        if client is None:
            return
        content_fp = content_fingerprint()
        if not content_fp:
            logger.warning("hot prefetch: no skill/SDK docs found; skipping bundle build")
            return
        # Selector fan-out limited to the cacheable lanes: memory /
        # workspace / pulled are utterance- or time-specific.
        shim = _ShimConfig(cfg, sources=["skills", "sdk"])
        built = 0
        for t in tasks:
            row = await kv_get(store, _bundle_key(t.name))
            if row is not None and row.get("fingerprint") == content_fp:
                continue
            req = PrefetchRequest(
                key=_bundle_key(t.name),
                key_kind="conversation",
                channel="voice",
                text=" / ".join(t.exemplars),
            )
            try:
                result = await run_prefetch(
                    req, store=store, client=client, config=shim,
                )
            except Exception:
                logger.debug(
                    "hot bundle build failed for %s", t.name, exc_info=True
                )
                continue
            await kv_put(store, _bundle_key(t.name), {
                "fingerprint": content_fp,
                "bundle": result.bundle.to_dict(),
            })
            built += 1
        if built:
            logger.info("hot prefetch bundles built: %d task(s)", built)


class _ShimConfig:
    """PrefetchConfig view with an overridden source list."""

    def __init__(self, cfg: Any, *, sources: list[str]) -> None:
        self._cfg = cfg
        self.sources = sources
        self.pull_sources: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg, name)


def _drop_already_loaded(bundle: PrefetchBundle, already: set[str]) -> None:
    """Strip bundle items this conversation's context already holds.

    Called on the hot path at lookup time AND on every bundle at
    injection time (agent._prefetch_context_for_text) — the fan-out
    lanes filter their menus too, but the selector is a model and this
    is the structural guarantee. Idempotent.
    """
    if not already:
        return
    for name in list(bundle.skill_bodies):
        if name in already:
            del bundle.skill_bodies[name]
    for module in list(bundle.sdk_modules):
        keys = set(bundle.sdk_sections.get(module) or ())
        # Whole-module includes (keys == {module}) match ONLY the
        # unambiguous path form: the already set carries bare skill
        # names too, and a skill named like a module must not shadow
        # it. Section-key subsets ("display.0" style) are collision-free.
        # Partial overlap re-injects the whole splice by design —
        # splices are one blob; only full coverage drops.
        whole = keys == {module}
        covered = (
            f"bb/modules/{module}.md" in already
            or (not whole and keys and keys <= already)
        )
        if covered:
            del bundle.sdk_modules[module]
            bundle.sdk_sections.pop(module, None)
    if bundle.memories:
        bundle.memories = [
            (mid, summ) for mid, summ in bundle.memories if mid not in already
        ]
