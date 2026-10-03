"""Context sources for the prefetch layer — one selector lane per source.

Each source contributes three things:

- ``gather``      deterministic, local, fast (<100ms): the candidate
                  corpus for this request, or None to skip the lane
                  entirely (no model call).
- a selection     what the one-shot selector model returns, validated
  schema          against a plain JSON schema.
- ``materialize`` re-fetch the picked items by id/name — never trust
                  model-copied text — and write them into the bundle.

SECURITY: sources are deliberately read-only. No ``message``, no
``execute_script``, no memory/workspace writes, no integration
create/update/delete. Integration pulls are limited to
``prefetch.pull_sources`` (config-side allowlist) crossed with a
hard-coded read-only action map — the model can only pick from names
the operator listed; it never supplies a source or action string.

Adding a context source = one more ``_SourceDef`` entry here. The
runner fans selectors out in parallel, so a new source adds cost but
never wall-clock latency.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from boxbot.prefetch.bundle import PrefetchBundle
from boxbot.prefetch.request import PrefetchRequest

logger = logging.getLogger(__name__)

_MEMORY_LIMIT = 8
_CONVERSATION_LIMIT = 3
_WORKSPACE_HITS = 8
_EXCERPT_CHARS = 1200

# Caps applied at materialize time, regardless of what the model returns.
# Skills cap is 2: with injection-time dedup making repeat picks free,
# the old cap of 1 only bought crowding-out (a session where both
# onboarding and a support skill matched loaded onboarding 4x and
# the support skill never).
_MAX_SKILLS = 2
_MAX_SDK_SECTIONS = 4
_MAX_MEMORIES = 5
_MAX_WORKSPACE = 2
_MAX_PULLS = 2
# Extra person-keyed retrieval pass (see gather_memory_candidates).
_PERSON_MEMORY_LIMIT = 4

# Read-only action map for integration pulls. ``prefetch.pull_sources``
# (config) selects WHICH of these are offered; this map fixes HOW each
# is called. The model never supplies an action or input.
_PULL_ACTIONS: dict[str, dict[str, Any]] = {
    "calendar": {"action": "list_upcoming_events"},
    "weather": {},
}


@dataclass(slots=True)
class SourceRun:
    """One prepared selector lane: corpus in, validated selection out."""

    name: str
    instructions: str
    schema: dict[str, Any]
    candidates: str
    materialize: Callable[[dict[str, Any], PrefetchBundle], Awaitable[None]]


def _string_array(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [str(x) for x in items if x]


# ---------------------------------------------------------------------------
# skills — pre-load at most one skill body
# ---------------------------------------------------------------------------


def _skills_source(req: PrefetchRequest) -> SourceRun | None:
    from boxbot.skills.loader import get_skill_index

    # Already-loaded skills leave the menu entirely — the instruction
    # "never re-select" only works when the selector knows what is
    # loaded, and an absent candidate can't be picked at all.
    index = get_skill_index(exclude=set(req.already_loaded or ()))
    if not index or not index.strip():
        return None
    # Names actually offered — a pick outside the menu is model-copied
    # text, never trusted as a load target.
    offered = {
        line[2:].split(":", 1)[0].strip()
        for line in index.splitlines()
        if line.startswith("- ")
    }

    async def materialize(sel: dict[str, Any], bundle: PrefetchBundle) -> None:
        from boxbot.skills.loader import load_skill

        for name in _string_array(sel.get("skills"))[:_MAX_SKILLS]:
            if name not in offered:
                logger.debug("skill pick %r not in menu; ignored", name)
                continue
            try:
                bundle.skill_bodies[name] = load_skill(name)
            except Exception:
                logger.debug("skill %r failed to load", name, exc_info=True)

    return SourceRun(
        name="skills",
        instructions=(
            "Pick up to TWO skills whose full bodies the assistant will "
            "need to act on this message, or none. Skill bodies are large "
            "— select only on a clear match with a skill's stated "
            "conditions; one is the norm, two only when both clearly apply."
        ),
        schema={
            "type": "object",
            "properties": {
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "0-2 skill names from the index.",
                },
            },
            "required": ["skills"],
        },
        candidates=index.strip(),
        materialize=materialize,
    )


# ---------------------------------------------------------------------------
# sdk — pre-load bb module doc SECTIONS (Level 3, H2-granular)
# ---------------------------------------------------------------------------


def _split_module_doc(text: str) -> tuple[str, list[tuple[str, str]]]:
    """``(preamble, [(heading, section_text)])`` split on H2 headings.

    The preamble (H1 title + intro, everything before the first ``## ``)
    always rides along with any selected section so the agent keeps the
    module's import/usage framing.
    """
    preamble: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    in_fence = False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if line.startswith("## ") and not in_fence:
            sections.append((line[3:].strip(), [line]))
            current = sections[-1][1]
        elif current is None:
            preamble.append(line)
        else:
            current.append(line)
    return "".join(preamble).strip(), [
        (heading, "".join(body).strip()) for heading, body in sections
    ]


def _sdk_section_corpus() -> dict[str, tuple[str, list[tuple[str, str]]]]:
    """``module -> (preamble, sections)`` for skills/bb/modules/*.md.

    Read once per gather; materialize reuses this snapshot so the picked
    keys can never drift against a file edited mid-request.
    """
    from boxbot.skills.loader import _find_skill  # same root resolution

    try:
        meta = _find_skill("bb", None)
        modules_dir = meta.root_path / "modules"
    except Exception:
        return {}
    if not modules_dir.is_dir():
        return {}
    corpus: dict[str, tuple[str, list[tuple[str, str]]]] = {}
    for f in sorted(modules_dir.glob("*.md")):
        try:
            corpus[f.stem] = _split_module_doc(f.read_text(encoding="utf-8"))
        except OSError:
            continue
    return corpus


def _splice_module(
    module: str,
    preamble: str,
    sections: list[tuple[str, str]],
    chosen: list[int],
) -> str:
    """Preamble + chosen sections + a pointer at the rest."""
    parts = [preamble] if preamble else []
    parts.extend(sections[i][1] for i in chosen)
    if len(chosen) < len(sections):
        parts.append(
            f"_(partial doc — {len(chosen)} of {len(sections)} sections; "
            f'load_skill("bb", "modules/{module}.md") for the rest)_'
        )
    return "\n\n".join(parts)


def _sdk_source(req: PrefetchRequest) -> SourceRun | None:
    corpus = _sdk_section_corpus()
    if not corpus:
        return None

    # In-context content leaves the menu: a section already injected (or
    # a module fully loaded, incl. via load_skill) can't be re-picked.
    # Whole-module state is matched ONLY on the unambiguous
    # "bb/modules/<m>.md" form — the already set also carries bare
    # SKILL names, and agent-created skills can take any name, so a
    # skill called "display" must not blank the display module.
    already = set(req.already_loaded or ())
    lines: list[str] = []
    valid: set[str] = set()
    for module, (preamble, sections) in corpus.items():
        if f"bb/modules/{module}.md" in already:
            continue
        if sections:
            for i, (heading, _) in enumerate(sections):
                key = f"{module}.{i}"
                if key in already:
                    continue
                lines.append(f"- {key}: {heading}")
                valid.add(key)
        else:
            # Sectionless (short) doc — offered whole.
            title = preamble.splitlines()[0].lstrip("# ") if preamble else module
            lines.append(f"- {module}: {title}")
            valid.add(module)
    if not valid:
        return None

    async def materialize(sel: dict[str, Any], bundle: PrefetchBundle) -> None:
        picks = [
            k for k in _string_array(sel.get("sections")) if k in valid
        ][:_MAX_SDK_SECTIONS]
        by_module: dict[str, list[int]] = {}
        whole: set[str] = set()
        for key in picks:
            # Whole-module keys are exactly the module name; section
            # keys append ".N". rpartition keeps module filenames that
            # themselves contain dots intact.
            if key in corpus:
                whole.add(key)
                continue
            module, _, idx = key.rpartition(".")
            if module in corpus and idx.isdigit():
                by_module.setdefault(module, []).append(int(idx))
        for module in whole:
            preamble, sections = corpus[module]
            bundle.sdk_modules[module] = _splice_module(
                module, preamble, sections, list(range(len(sections)))
            )
            bundle.sdk_sections[module] = [module]
        for module, idxs in by_module.items():
            if module in whole:
                continue
            preamble, sections = corpus[module]
            chosen = sorted({i for i in idxs if 0 <= i < len(sections)})
            if not chosen:
                continue
            bundle.sdk_modules[module] = _splice_module(
                module, preamble, sections, chosen
            )
            bundle.sdk_sections[module] = [f"{module}.{i}" for i in chosen]

    return SourceRun(
        name="sdk",
        instructions=(
            "These are section headings from the boxBot SDK (`bb`) module "
            "docs. Pick the sections (at most 4) whose API the assistant "
            "will need to write a script for this message, or none. Skip "
            "sections already in context, and every section of a module "
            "already loaded in full."
        ),
        schema={
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "0-4 section keys from the list.",
                },
            },
            "required": ["sections"],
        },
        candidates="\n".join(lines),
        materialize=materialize,
    )


# ---------------------------------------------------------------------------
# memory — inject fact/conversation summaries (deterministic retrieval
# first; the selector only filters)
# ---------------------------------------------------------------------------


async def _bump_relevance(store: Any, memory_id: str) -> None:
    """Refresh ``last_relevant_at`` for an injected memory.

    Access-based retention (maintenance archiving, dream usage signal)
    keys on this timestamp; the legacy inject path bumped it and this
    lane supersedes that path. No-ops for conversation ids.
    """
    try:
        await store.update_memory_relevance(memory_id)
    except Exception:
        logger.debug("relevance bump failed for %s", memory_id, exc_info=True)


async def gather_memory_candidates(
    store: Any,
    *,
    text: str | None,
    person: str | None,
    query_embedding: Any = None,
) -> list[Any]:
    """Deterministic memory retrieval: utterance pass + person pass.

    The person pass keeps person-keyed recall alive now that the
    prefetch memory lane supersedes the legacy ``_inject_memories``
    block (which searched by speaker as well as utterance). Shared by
    the selector lane and the hot-task path.
    """
    from boxbot.memory.search import hybrid_search

    query = (text or "").strip()
    if store is None or not (query or person):
        return []
    cands: list[Any] = []
    if query:
        try:
            cands = await hybrid_search(
                store,
                query,
                include_conversations=True,
                memory_limit=_MEMORY_LIMIT,
                conversation_limit=_CONVERSATION_LIMIT,
                query_embedding=query_embedding,
            )
        except Exception:
            logger.debug("memory candidate retrieval failed", exc_info=True)
    if person:
        try:
            # BM25-only: a supplemental name-recall pass isn't worth a
            # second embed round trip on the reply path.
            extra = await hybrid_search(
                store,
                person,
                include_conversations=False,
                memory_limit=_PERSON_MEMORY_LIMIT,
                allow_vector=False,
            )
        except Exception:
            extra = []
        known = {c.id for c in cands}
        cands.extend(c for c in extra if c.id not in known)
    return cands


async def _memory_source(req: PrefetchRequest, store: Any) -> SourceRun | None:
    cands = await gather_memory_candidates(
        store, text=req.text, person=req.person,
        query_embedding=req.query_embedding,
    )
    if not cands:
        return None

    seen = {c.id: c.summary for c in cands}
    lines = [f"- {c.id}: [{c.type}] {c.summary}" for c in cands]

    async def materialize(sel: dict[str, Any], bundle: PrefetchBundle) -> None:
        for mid in _string_array(sel.get("memory_ids"))[:_MAX_MEMORIES]:
            summary = seen.get(mid)
            if summary:
                bundle.memories.append((mid, summary))
                await _bump_relevance(store, mid)

    return SourceRun(
        name="memory",
        instructions=(
            "These memories matched the message by retrieval. Keep only "
            "the ones the assistant genuinely needs for THIS message — "
            "retrieval match alone is not enough. Skip anything already "
            "in context."
        ),
        schema={
            "type": "object",
            "properties": {
                "memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "ids worth injecting (often none).",
                },
            },
            "required": ["memory_ids"],
        },
        candidates="\n".join(lines),
        materialize=materialize,
    )


# ---------------------------------------------------------------------------
# workspace — excerpt files from the agent's notebook
# ---------------------------------------------------------------------------


def _workspace_source(req: PrefetchRequest) -> SourceRun | None:
    from boxbot.workspace.store import Workspace

    query = (req.text or "").strip()
    if not query:
        return None
    try:
        ws = Workspace()
        hits = ws.search(query, limit=_WORKSPACE_HITS)
    except Exception:
        logger.debug("workspace candidate search failed", exc_info=True)
        return None
    if not hits:
        return None

    lines = []
    offered_paths: set[str] = set()
    for h in hits:
        path = h.get("path") if isinstance(h, dict) else None
        if not path:
            continue
        offered_paths.add(path)
        snippet = str(h.get("text") or h.get("line") or "").strip()
        lines.append(f"- {path}: {snippet[:120]}")
    if not lines:
        return None

    async def materialize(sel: dict[str, Any], bundle: PrefetchBundle) -> None:
        for path in _string_array(sel.get("paths"))[:_MAX_WORKSPACE]:
            if path not in offered_paths:
                logger.debug("workspace pick %r not in menu; ignored", path)
                continue
            try:
                rec = Workspace().read(path)
            except Exception:
                continue
            content = rec.get("content")
            if isinstance(content, str) and content.strip():
                bundle.workspace_excerpts.append(
                    (path, content[:_EXCERPT_CHARS])
                )

    return SourceRun(
        name="workspace",
        instructions=(
            "These workspace files (the assistant's own notebook) "
            "matched the message. Pick the files (at most 2) whose "
            "content the assistant will need to read, or none."
        ),
        schema={
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "0-2 workspace paths from the hits.",
                },
            },
            "required": ["paths"],
        },
        candidates="\n".join(lines),
        materialize=materialize,
    )


# ---------------------------------------------------------------------------
# pulled — run allowlisted read-only integrations when the turn needs
# their data (calendar/weather). The pull happens at materialize time,
# only when actually selected.
# ---------------------------------------------------------------------------


def _pulled_source(
    req: PrefetchRequest, pull_sources: list[str]
) -> SourceRun | None:
    from boxbot.integrations.loader import get_integration

    offered: list[tuple[str, str]] = []
    for name in pull_sources:
        if name not in _PULL_ACTIONS:
            continue
        meta = get_integration(name)
        if meta is None:
            continue
        offered.append((name, meta.description.split(". ")[0]))
    if not offered:
        return None

    listing = "\n".join(f"- {name}: {desc}" for name, desc in offered)
    valid = {name for name, _ in offered}

    async def materialize(sel: dict[str, Any], bundle: PrefetchBundle) -> None:
        from boxbot.integrations.runner import run

        picks = [
            s for s in _string_array(sel.get("pull")) if s in valid
        ][:_MAX_PULLS]
        for source in picks:
            inputs = dict(_PULL_ACTIONS[source])
            try:
                result = await run(source, inputs)
            except Exception:
                logger.debug("prefetch pull %r failed", source, exc_info=True)
                continue
            if result.get("status") != "ok":
                continue
            bundle.pulled_data.append({
                "source": source,
                "action": inputs.get("action"),
                "payload": result.get("output"),
                "pulled_at": datetime.now(timezone.utc).isoformat(),
            })

    return SourceRun(
        name="pulled",
        instructions=(
            "Pull a data source ONLY when the message is actually about "
            "its data (time/schedule → calendar, weather → weather). "
            "Selecting nothing is the normal outcome."
        ),
        schema={
            "type": "object",
            "properties": {
                "pull": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sources to pull now (usually none).",
                },
            },
            "required": ["pull"],
        },
        candidates=listing,
        materialize=materialize,
    )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


async def gather_sources(
    req: PrefetchRequest, *, store: Any, config: Any
) -> list[SourceRun]:
    """Build the selector lanes for one request.

    Sources not in ``config.sources``, or whose candidate gather comes
    back empty, are skipped — no model call is spent on them.
    """
    enabled = list(
        getattr(config, "sources", None)
        or ["skills", "sdk", "memory", "workspace", "pulled"]
    )
    pull_sources = list(getattr(config, "pull_sources", None) or [])

    runs: list[SourceRun] = []
    for name in enabled:
        try:
            if name == "skills":
                run = _skills_source(req)
            elif name == "sdk":
                run = _sdk_source(req)
            elif name == "memory":
                run = await _memory_source(req, store)
            elif name == "workspace":
                run = _workspace_source(req)
            elif name == "pulled":
                run = _pulled_source(req, pull_sources)
            else:
                logger.warning("unknown prefetch source %r; skipping", name)
                run = None
        except Exception:
            logger.debug("prefetch source %s gather failed", name, exc_info=True)
            run = None
        if run is not None:
            runs.append(run)
    return runs
