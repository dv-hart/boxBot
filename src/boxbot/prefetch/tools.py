"""Read-only capabilities the prefetch mini-agent may call.

SECURITY: this module is deliberately read-only. It never imports a
write path — no ``message``, no ``execute_script``, no memory/workspace
writes, no ``manage_tasks`` mutations, no integration create/update/
delete. Integration access is limited to a hard-coded allowlist of
read-only (source, action) pairs; the model cannot name an arbitrary
integration or action. This mirrors the web_search content-firewall
posture: a small model searching broadly, fenced off from anything that
can affect the world.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_WORKSPACE_READ_CHARS = 2000
_MEMORY_LIMIT = 8

# Hard-coded read-only integration allowlist: tool_name -> (source, inputs).
# The model calls the tool by name; it never supplies source/action.
_INTEGRATION_ALLOWLIST: dict[str, tuple[str, dict[str, Any]]] = {
    "get_calendar": ("calendar", {"action": "list_upcoming_events"}),
    "get_weather": ("weather", {}),
}


@dataclass(slots=True)
class PrefetchState:
    """Mutable scratch accumulated across the mini-agent's tool calls."""

    store: Any  # MemoryStore (main-process instance, read-only use)
    # memory/conversation id -> summary, for resolving finalize() picks.
    seen: dict[str, str] = field(default_factory=dict)
    # source -> {payload, action, pulled_at}
    pulled: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Selection captured by finalize(); None until the model finalizes.
    selection: dict[str, Any] | None = None


def build_tool_definitions() -> list[dict[str, Any]]:
    """Client-side tool schemas the mini-agent may call."""
    tools: list[dict[str, Any]] = [
        {
            "name": "search_memory",
            "description": (
                "Search boxBot's long-term memory (facts about people, the "
                "household, and how things are done). Returns candidate "
                "records with ids and one-line summaries. Search broadly — "
                "try a few queries if useful."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "search_history",
            "description": (
                "Search summaries of prior conversations. Use to recall what "
                "was already discussed with this person."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "list_skills",
            "description": (
                "List available skills with their one-line 'when to use' "
                "blurbs. Use to decide whether a skill body is worth "
                "pre-loading for the agent."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "load_skill",
            "description": (
                "Read the full body of a skill by name. Only pre-load a "
                "skill the agent is very likely to need — skill bodies are "
                "large."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
        {
            "name": "workspace_search",
            "description": (
                "Grep boxBot's workspace notebook (notes, CSVs the agent "
                "keeps). Returns path/line/text hits."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {
                        "type": "string",
                        "description": "Optional subtree to limit the search.",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "workspace_read",
            "description": "Read a workspace file's text content (truncated).",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "get_calendar",
            "description": (
                "Pull upcoming calendar events (read-only). Use when the "
                "task is time/schedule related."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "get_weather",
            "description": (
                "Pull current weather conditions (read-only). Use when the "
                "task is weather related."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "finalize",
            "description": (
                "Submit the curated bundle and finish. Include ONLY items the "
                "agent is very likely to need on its first turn — precision "
                "matters more than recall. Prefer an empty bundle over a "
                "marginally-relevant one."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs from search_memory/search_history.",
                    },
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Skill names to inline (at most one).",
                    },
                    "workspace_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "pulled_sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Subset of calendar/weather to include.",
                    },
                    "likely_next_note": {
                        "type": "string",
                        "description": "One line: what the agent will likely do.",
                    },
                },
            },
        },
    ]
    return tools


async def dispatch_tool(
    name: str, tool_input: dict[str, Any], *, state: PrefetchState
) -> str:
    """Run one read-only tool call and return a string tool_result."""
    try:
        if name == "search_memory":
            return await _search_memory(tool_input, state)
        if name == "search_history":
            return await _search_history(tool_input, state)
        if name == "list_skills":
            return _list_skills()
        if name == "load_skill":
            return _load_skill(tool_input)
        if name == "workspace_search":
            return _workspace_search(tool_input)
        if name == "workspace_read":
            return _workspace_read(tool_input)
        if name in _INTEGRATION_ALLOWLIST:
            return await _integration(name, state)
        if name == "finalize":
            state.selection = _clean_selection(tool_input)
            return "ok"
        return json.dumps({"error": f"unknown tool: {name}"})
    except Exception as exc:  # never let a tool failure crash the loop
        logger.debug("prefetch tool %s failed", name, exc_info=True)
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# --------------------------------------------------------------------------
# Individual tools
# --------------------------------------------------------------------------


async def _search_memory(tool_input: dict[str, Any], state: PrefetchState) -> str:
    from boxbot.memory.search import hybrid_search

    query = str(tool_input.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "query required"})
    cands = await hybrid_search(
        state.store, query, include_conversations=False,
        memory_limit=_MEMORY_LIMIT,
    )
    out = []
    for c in cands:
        state.seen[c.id] = c.summary
        out.append({"id": c.id, "type": c.type, "summary": c.summary})
    return json.dumps({"results": out})


async def _search_history(tool_input: dict[str, Any], state: PrefetchState) -> str:
    from boxbot.memory.search import hybrid_search

    query = str(tool_input.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "query required"})
    cands = await hybrid_search(
        state.store, query, include_conversations=True, memory_limit=0,
        conversation_limit=6,
    )
    out = []
    for c in cands:
        if c.source != "conversation":
            continue
        state.seen[c.id] = c.summary
        meta = c.metadata or {}
        out.append({
            "id": c.id,
            "summary": c.summary,
            "when": meta.get("started_at"),
            "participants": meta.get("participants"),
        })
    return json.dumps({"results": out})


def _list_skills() -> str:
    from boxbot.skills.loader import get_skill_index

    return get_skill_index() or "(no skills available)"


def _load_skill(tool_input: dict[str, Any]) -> str:
    from boxbot.skills.loader import load_skill

    name = str(tool_input.get("name") or "").strip()
    if not name:
        return json.dumps({"error": "name required"})
    try:
        return load_skill(name)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})


def _workspace() -> Any:
    from boxbot.workspace.store import Workspace

    return Workspace()


def _workspace_search(tool_input: dict[str, Any]) -> str:
    query = str(tool_input.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "query required"})
    path = tool_input.get("path")
    hits = _workspace().search(query, path=path, limit=20)
    return json.dumps({"hits": hits})


def _workspace_read(tool_input: dict[str, Any]) -> str:
    path = str(tool_input.get("path") or "").strip()
    if not path:
        return json.dumps({"error": "path required"})
    try:
        rec = _workspace().read(path)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    content = rec.get("content")
    if isinstance(content, str) and len(content) > _WORKSPACE_READ_CHARS:
        content = content[:_WORKSPACE_READ_CHARS] + "…(truncated)"
        rec = {**rec, "content": content}
    return json.dumps(rec, default=str)


async def _integration(tool_name: str, state: PrefetchState) -> str:
    from boxbot.integrations.loader import get_integration
    from boxbot.integrations.runner import run

    source, inputs = _INTEGRATION_ALLOWLIST[tool_name]
    if get_integration(source) is None:
        return json.dumps({"error": f"{source} integration not configured"})
    result = await run(source, dict(inputs))
    if result.get("status") != "ok":
        return json.dumps({
            "error": result.get("error") or "integration error",
            "status": result.get("status"),
        })
    payload = result.get("output")
    state.pulled[source] = {
        "payload": payload,
        "action": inputs.get("action"),
        "pulled_at": datetime.now(timezone.utc).isoformat(),
    }
    return json.dumps({"output": payload})


def _clean_selection(tool_input: dict[str, Any]) -> dict[str, Any]:
    def _list(key: str) -> list[str]:
        v = tool_input.get(key)
        return [str(x) for x in v] if isinstance(v, list) else []

    return {
        "memory_ids": _list("memory_ids"),
        "skills": _list("skills"),
        "workspace_paths": _list("workspace_paths"),
        "pulled_sources": _list("pulled_sources"),
        "likely_next_note": str(tool_input.get("likely_next_note") or ""),
    }
