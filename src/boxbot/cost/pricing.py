"""Load and access pricing from ``config/pricing.yaml``.

Pricing is data, not code. Updating a model's per-MTok price is a
YAML edit and a ``verified_on`` bump — no Python change required.

Lookup model:
    pricing = get_pricing()
    in_per = pricing.anthropic_input_per_mtok("claude-opus-4-7")

Unknown model returns ``None``. Callers decide whether to treat that
as a hard error (cost computation impossible) or log-and-zero. The
helpers in :mod:`boxbot.cost.compute` log a warning and write a row
with ``cost_usd=0.0`` so the call is at least visible in the log.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)
_ENV_VAR = "BOXBOT_PRICING_CONFIG"

# RLock so ``get_pricing()`` (which holds the lock) can call
# ``reload_pricing()`` (which also acquires it) on the first-access
# code path without deadlocking.
_lock = threading.RLock()
_cached: Pricing | None = None


@dataclass(frozen=True, slots=True)
class Pricing:
    """In-memory view of pricing.yaml. Immutable; reload via reload_pricing()."""

    anthropic_models: dict[str, dict[str, float]]
    openai_models: dict[str, dict[str, float]]
    elevenlabs_tts: dict[str, dict[str, float]]
    elevenlabs_stt: dict[str, dict[str, float]]
    source_path: Path
    anthropic_verified_on: str | None = None
    openai_verified_on: str | None = None
    elevenlabs_verified_on: str | None = None

    def anthropic_input_per_mtok(self, model: str) -> float | None:
        entry = self.anthropic_models.get(model)
        return entry["input_per_mtok"] if entry else None

    def anthropic_output_per_mtok(self, model: str) -> float | None:
        entry = self.anthropic_models.get(model)
        return entry["output_per_mtok"] if entry else None

    def openai_input_per_mtok(self, model: str) -> float | None:
        entry = self.openai_models.get(model)
        return entry["input_per_mtok"] if entry else None

    def openai_output_per_mtok(self, model: str) -> float | None:
        entry = self.openai_models.get(model)
        return entry["output_per_mtok"] if entry else None

    def openai_cached_input_per_mtok(self, model: str) -> float | None:
        """Optional per-model cached-input rate; None = no discount."""
        entry = self.openai_models.get(model)
        return entry.get("cached_input_per_mtok") if entry else None

    def elevenlabs_tts_per_char(self, model: str) -> float | None:
        entry = self.elevenlabs_tts.get(model)
        return entry["dollars_per_char"] if entry else None

    def elevenlabs_stt_per_minute(self, model: str) -> float | None:
        entry = self.elevenlabs_stt.get(model)
        return entry["dollars_per_minute"] if entry else None


def _default_path() -> Path:
    """``config/pricing.yaml`` under the project root, not the cwd.

    boxbot is launched from systemd, cron, and bring-up scripts,
    none of which promise to start in the repo directory.
    PROJECT_ROOT is imported lazily: ``boxbot.core`` pulls in the agent,
    which imports this module.
    """
    from boxbot.core.paths import PROJECT_ROOT

    return PROJECT_ROOT / "config" / "pricing.yaml"


def _resolve_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env)
    return _default_path()


def _parse(data: dict[str, Any], source_path: Path) -> Pricing:
    anthropic = data.get("anthropic") or {}
    openai = data.get("openai") or {}
    elevenlabs = data.get("elevenlabs") or {}
    return Pricing(
        anthropic_models=anthropic.get("models") or {},
        openai_models=openai.get("models") or {},
        elevenlabs_tts=(elevenlabs.get("tts") or {}),
        elevenlabs_stt=(elevenlabs.get("stt") or {}),
        anthropic_verified_on=anthropic.get("verified_on"),
        openai_verified_on=openai.get("verified_on"),
        elevenlabs_verified_on=elevenlabs.get("verified_on"),
        source_path=source_path,
    )


def reload_pricing(path: str | Path | None = None) -> Pricing:
    """Load (or reload) pricing.yaml. Replaces the cached singleton."""
    global _cached
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"pricing config not found at {resolved}. "
            f"Set {_ENV_VAR} or place a pricing.yaml in config/."
        )
    with open(resolved) as f:
        data = yaml.safe_load(f) or {}
    pricing = _parse(data, resolved)
    with _lock:
        _cached = pricing
    logger.info(
        "Loaded pricing from %s (anthropic verified %s, openai verified %s, "
        "elevenlabs verified %s)",
        resolved,
        pricing.anthropic_verified_on,
        pricing.openai_verified_on,
        pricing.elevenlabs_verified_on,
    )
    return pricing


def get_pricing() -> Pricing:
    """Return the cached Pricing, loading on first access.

    A missing or unreadable file degrades to empty pricing — every
    lookup then returns ``None``, which callers already handle as
    "unknown model" (warn, log the row at zero). Cost accounting is not
    worth failing a conversation over. ``reload_pricing()`` still
    raises: an explicit reload must report its own failure.
    """
    global _cached
    if _cached is None:
        with _lock:
            if _cached is None:
                try:
                    return reload_pricing()
                except (OSError, yaml.YAMLError) as e:
                    logger.warning(
                        "Pricing unavailable (%s); costs will be logged as "
                        "0.0 until it loads.", e,
                    )
                    _cached = _parse({}, _resolve_path(None))
    return _cached
