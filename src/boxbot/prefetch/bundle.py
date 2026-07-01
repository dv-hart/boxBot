"""The assembled prefetch bundle and its budgeted rendering.

A bundle is intentionally small. The whole point of the prefetch layer
is to REDUCE bloat and repeat tool calls — a bundle that injects context
the agent didn't need is a failure (low precision, measured in shadow
mode). :meth:`PrefetchBundle.render` therefore emits sections in
priority order and hard-stops at the configured token budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Rough tokens ≈ chars / 4. Good enough for a budget gate; we never bill
# on this number (the real cost row uses the API usage totals).
_CHARS_PER_TOKEN = 4


def _est_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


@dataclass(slots=True)
class PrefetchBundle:
    """Curated context for the main agent's first turn.

    Every field is what the prefetcher decided is very likely needed —
    not everything it looked at. Selection happens in the mini-agent;
    this object is the deterministic, budget-truncated result.
    """

    # (memory_id, summary) pairs, highest-relevance first.
    memories: list[tuple[str, str]] = field(default_factory=list)
    # skill_name -> full SKILL.md body (inlined so the agent skips a
    # load_skill round-trip). Capped to 1 by the runner.
    skill_bodies: dict[str, str] = field(default_factory=dict)
    # (workspace_path, excerpt) pairs.
    workspace_excerpts: list[tuple[str, str]] = field(default_factory=list)
    # Free-text highlights pulled from prior conversations.
    history_highlights: list[str] = field(default_factory=list)
    # Pulled/reviewed data: [{source, action, payload, pulled_at}].
    pulled_data: list[dict[str, Any]] = field(default_factory=list)
    # One-line "what you'll likely need to do".
    likely_next_note: str = ""
    # Filled by render(); the estimated size of the rendered block.
    token_estimate: int = 0

    # -- predicted-set accessors (for prefetch_events / offline join) --

    def predicted_memory_ids(self) -> list[str]:
        return [mid for mid, _ in self.memories]

    def predicted_skills(self) -> list[str]:
        return list(self.skill_bodies.keys())

    def predicted_workspace_paths(self) -> list[str]:
        return [p for p, _ in self.workspace_excerpts]

    def predicted_integration_calls(self) -> list[dict[str, Any]]:
        return [
            {
                "source": d.get("source"),
                "action": d.get("action"),
                "pulled_at": d.get("pulled_at"),
            }
            for d in self.pulled_data
        ]

    def is_empty(self) -> bool:
        return not (
            self.memories
            or self.skill_bodies
            or self.workspace_excerpts
            or self.history_highlights
            or self.pulled_data
            or self.likely_next_note.strip()
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for the scheduled prefetch_cache."""
        return {
            "memories": [list(m) for m in self.memories],
            "skill_bodies": self.skill_bodies,
            "workspace_excerpts": [list(w) for w in self.workspace_excerpts],
            "history_highlights": self.history_highlights,
            "pulled_data": self.pulled_data,
            "likely_next_note": self.likely_next_note,
            "token_estimate": self.token_estimate,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PrefetchBundle":
        return cls(
            memories=[tuple(m) for m in d.get("memories", [])],
            skill_bodies=dict(d.get("skill_bodies", {})),
            workspace_excerpts=[
                tuple(w) for w in d.get("workspace_excerpts", [])
            ],
            history_highlights=list(d.get("history_highlights", [])),
            pulled_data=list(d.get("pulled_data", [])),
            likely_next_note=str(d.get("likely_next_note", "")),
            token_estimate=int(d.get("token_estimate", 0) or 0),
        )

    def render(self, *, token_budget: int) -> str:
        """Render the injected markdown section, truncated to budget.

        Sections are appended in priority order; once the running token
        estimate would exceed ``token_budget`` the remaining lower-
        priority sections are dropped. Sets ``self.token_estimate``.
        """
        header = (
            "## Prefetched context (assembled for this turn)\n"
            "_A helper pre-gathered what you'll likely need. Treat it as a "
            "head start, not ground truth — verify before acting._"
        )
        blocks: list[str] = [header]
        used = _est_tokens(header)

        def _try_add(text: str) -> bool:
            nonlocal used
            cost = _est_tokens(text)
            if used + cost > token_budget:
                return False
            blocks.append(text)
            used += cost
            return True

        # Priority 1 — the note (cheap, high value).
        if self.likely_next_note.strip():
            _try_add(f"**Likely next:** {self.likely_next_note.strip()}")

        # Priority 2 — memories.
        if self.memories:
            lines = [f"- #{mid[:8]}: {summ}" for mid, summ in self.memories]
            _try_add("**Relevant memories:**\n" + "\n".join(lines))

        # Priority 3 — pulled data (calendar/weather).
        for d in self.pulled_data:
            payload = d.get("payload")
            src = d.get("source")
            at = d.get("pulled_at")
            _try_add(
                f"**{src}** (pulled {at}):\n"
                + _stringify_payload(payload)
            )

        # Priority 4 — workspace excerpts.
        for path, excerpt in self.workspace_excerpts:
            if not _try_add(f"**Workspace `{path}`:**\n{excerpt}"):
                break

        # Priority 5 — history highlights.
        if self.history_highlights:
            lines = [f"- {h}" for h in self.history_highlights]
            _try_add("**From prior conversations:**\n" + "\n".join(lines))

        # Priority 6 (bulkiest, lowest) — inlined skill bodies.
        for name, body in self.skill_bodies.items():
            if not _try_add(f"**Skill `{name}` (pre-loaded):**\n{body}"):
                break

        self.token_estimate = used
        return "\n\n".join(blocks)


def _stringify_payload(payload: Any) -> str:
    """Compactly render a pulled integration payload for the prompt."""
    import json

    if isinstance(payload, str):
        return payload.strip()
    try:
        return json.dumps(payload, default=str, ensure_ascii=False, indent=None)
    except Exception:
        return str(payload)
