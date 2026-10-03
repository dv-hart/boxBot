"""The assembled prefetch bundle and its rendering.

A bundle is small BY CONSTRUCTION: every source lane caps its own picks
(1 skill body, 2 sdk modules, 5 memories, 2 workspace excerpts, 2
pulls — see ``prefetch/sources.py``). ``render`` therefore never
truncates; it emits everything and LOGS the per-section token estimates
so the offline harness can analyse real bundle sizes. ``token_budget``
is a log-threshold only — a bundle over it draws a warning, not a cut.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Rough tokens ≈ chars / 4. Good enough for size telemetry; we never
# bill on this number (the real cost row uses the API usage totals).
_CHARS_PER_TOKEN = 4


def _est_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


@dataclass(slots=True)
class PrefetchBundle:
    """Curated context for the main agent's first turn.

    Every field is what a selector lane decided is very likely needed —
    materialized by re-fetching ids/names, never from model-copied text.
    """

    # (memory_id, summary) pairs, highest-relevance first.
    memories: list[tuple[str, str]] = field(default_factory=list)
    # skill_name -> full SKILL.md body (inlined so the agent skips a
    # load_skill round-trip). Capped to 1 by the skills lane.
    skill_bodies: dict[str, str] = field(default_factory=dict)
    # bb module name -> spliced doc text (preamble + selected H2
    # sections; see sources._sdk_source). Capped to 4 sections total.
    sdk_modules: dict[str, str] = field(default_factory=dict)
    # bb module name -> selected section keys ("display.3", or the bare
    # module name when the whole doc was included). Feeds already_loaded
    # so later turns dedup at section granularity.
    sdk_sections: dict[str, list[str]] = field(default_factory=dict)
    # (workspace_path, excerpt) pairs.
    workspace_excerpts: list[tuple[str, str]] = field(default_factory=list)
    # Pulled/reviewed data: [{source, action, payload, pulled_at}].
    pulled_data: list[dict[str, Any]] = field(default_factory=list)
    # Live device-state lines resolved at consume time (see
    # prefetch/providers.py). NEVER persisted — to_dict drops them so a
    # cached bundle can't serve stale state; each hit resolves fresh.
    live_context: list[str] = field(default_factory=list)
    # Recent-activity lines (see prefetch/activity.py): deterministic
    # cross-channel conversation recency, attached at consume time on a
    # conversation's first turn. Same never-persisted rule as
    # live_context — recency is perishable.
    recent_activity: list[str] = field(default_factory=list)
    # Filled by render(); the estimated size of the rendered block.
    token_estimate: int = 0

    # -- predicted-set accessors (for prefetch_events / offline join) --

    def predicted_memory_ids(self) -> list[str]:
        return [mid for mid, _ in self.memories]

    def predicted_skills(self) -> list[str]:
        return list(self.skill_bodies.keys())

    def predicted_sdk_modules(self) -> list[str]:
        return list(self.sdk_modules.keys())

    def predicted_sdk_sections(self) -> list[str]:
        """Section-granular keys; falls back to whole-module form for
        bundles cached before section selection existed."""
        if self.sdk_sections:
            return [k for keys in self.sdk_sections.values() for k in keys]
        return [f"bb/modules/{m}.md" for m in self.sdk_modules]

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
            or self.sdk_modules
            or self.workspace_excerpts
            or self.pulled_data
            or self.live_context
            or self.recent_activity
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for the scheduled prefetch_cache."""
        return {
            "memories": [list(m) for m in self.memories],
            "skill_bodies": self.skill_bodies,
            "sdk_modules": self.sdk_modules,
            "sdk_sections": self.sdk_sections,
            "workspace_excerpts": [list(w) for w in self.workspace_excerpts],
            "pulled_data": self.pulled_data,
            "token_estimate": self.token_estimate,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PrefetchBundle":
        return cls(
            memories=[tuple(m) for m in d.get("memories", [])],
            skill_bodies=dict(d.get("skill_bodies", {})),
            sdk_modules=dict(d.get("sdk_modules", {})),
            sdk_sections={
                k: list(v) for k, v in d.get("sdk_sections", {}).items()
            },
            workspace_excerpts=[
                tuple(w) for w in d.get("workspace_excerpts", [])
            ],
            pulled_data=list(d.get("pulled_data", [])),
            token_estimate=int(d.get("token_estimate", 0)),
        )

    def render(self, *, token_budget: int) -> str:
        """Render the injected markdown section and log section sizes.

        Nothing is truncated: the per-lane caps bound the bundle at
        assembly time. ``token_budget`` only sets the warning threshold
        for the size log. Sets ``self.token_estimate``.
        """
        header = (
            "## Prefetched context (assembled for this turn)\n"
            "_A helper pre-gathered what you'll likely need. Treat it as a "
            "head start, not ground truth — verify before acting._"
        )
        blocks: list[str] = [header]
        sizes: dict[str, int] = {}

        def _add(section: str, text: str) -> None:
            blocks.append(text)
            sizes[section] = sizes.get(section, 0) + _est_tokens(text)

        # Perishable data first, then capability docs, then recall.
        if self.live_context:
            _add(
                "live",
                "**Live device state (read just now — trust it, don't "
                "re-read):**\n" + "\n".join(f"- {l}" for l in self.live_context),
            )

        if self.recent_activity:
            _add(
                "activity",
                "**Recent conversations** (newest first; full text: "
                'search_memory mode="transcript" with the id):\n'
                + "\n".join(self.recent_activity),
            )

        for d in self.pulled_data:
            _add(
                "pulled",
                f"**{d.get('source')}** (pulled {d.get('pulled_at')}):\n"
                + _stringify_payload(d.get("payload")),
            )

        for name, body in self.skill_bodies.items():
            _add(
                "skills",
                f"**Skill `{name}` (pre-loaded — treat as if you called "
                f"load_skill):**\n{body}",
            )

        for name, body in self.sdk_modules.items():
            _add("sdk", f"**bb module `{name}` (pre-loaded):**\n{body}")

        if self.memories:
            lines = [f"- #{mid[:8]}: {summ}" for mid, summ in self.memories]
            _add("memory", "**Relevant memories:**\n" + "\n".join(lines))

        for path, excerpt in self.workspace_excerpts:
            _add("workspace", f"**Workspace `{path}`:**\n{excerpt}")

        rendered = "\n\n".join(blocks)
        self.token_estimate = _est_tokens(rendered)
        if sizes:
            logger.info(
                "prefetch bundle rendered: total≈%d tokens, sections=%s",
                self.token_estimate,
                {k: f"≈{v}t" for k, v in sizes.items()},
            )
        if self.token_estimate > token_budget:
            logger.warning(
                "prefetch bundle ≈%d tokens exceeds budget %d — injected "
                "anyway; tighten per-lane caps if this recurs",
                self.token_estimate, token_budget,
            )
        return rendered if not self.is_empty() else ""


def _stringify_payload(payload: Any) -> str:
    """Compact, deterministic text form of a pulled payload."""
    import json

    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, indent=None, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(payload)
