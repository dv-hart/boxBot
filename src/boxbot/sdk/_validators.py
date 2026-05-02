"""Input validation for SDK operations.

All validation uses simple isinstance checks and value range checks.
Raises ValueError with clear messages on invalid input.
"""

from __future__ import annotations

from typing import Any, Sequence

# --- Display constants ---

VALID_THEMES = {"boxbot", "midnight", "daylight", "classic"}

VALID_TEXT_SIZES = {"title", "heading", "subtitle", "body", "caption", "small"}
VALID_TEXT_COLORS = {
    "default", "muted", "dim", "accent", "success", "warning", "error",
}
VALID_TEXT_WEIGHTS = {"normal", "medium", "semibold", "bold"}
VALID_TEXT_ALIGNS = {"left", "center", "right"}
VALID_TEXT_ANIMATIONS = {"none", "fade", "typewriter", "count_up", "slide_up"}

VALID_CONTAINER_ALIGNS = {"start", "center", "end", "spread"}

VALID_ICON_SIZES = {"sm", "md", "lg", "xl"}
VALID_EMOJI_SIZES = {"md", "lg", "xl"}

VALID_CLOCK_FORMATS = {"12h", "24h"}
VALID_CLOCK_SIZES = {"md", "lg", "xl"}

VALID_CHART_TYPES = {"line", "bar", "area"}

VALID_LIST_STYLES = {"bullet", "number", "check", "none"}

VALID_IMAGE_FITS = {"cover", "contain", "fill"}

VALID_METRIC_ANIMATIONS = {"none", "fade", "count_up"}

VALID_TRANSITIONS = {"crossfade", "slide_left", "slide_right", "none"}

VALID_DIVIDER_ORIENTATIONS = {"horizontal", "vertical"}

VALID_DATA_SOURCE_TYPES = {"http_json", "http_text", "static", "memory_query"}
VALID_BUILTIN_SOURCES = {"clock", "weather", "calendar", "tasks", "people", "agent_status"}

VALID_MEMORY_TYPES = {"person", "household", "methodology", "operational"}

VALID_TRIGGER_STATUSES = {"active", "expired", "cancelled"}
VALID_TODO_STATUSES = {"pending", "completed", "cancelled"}

VALID_SKILL_PARAM_TYPES = {"string", "integer", "float", "boolean"}

# Reserved skill names per Anthropic's Agent Skills spec — must not be used.
RESERVED_SKILL_NAMES = {"anthropic", "claude"}

# Hard caps from Anthropic's Agent Skills spec.
SKILL_NAME_MAX_LEN = 64
SKILL_DESCRIPTION_MAX_LEN = 1024
SKILL_BODY_SOFT_MAX = 5 * 1024  # body should be ≤5KB; longer split to Level 3 sub-docs


# --- Validation helpers ---

def require(value: Any, name: str) -> None:
    """Raise if value is None."""
    if value is None:
        raise ValueError(f"'{name}' is required")


def require_str(value: Any, name: str, *, allow_empty: bool = False) -> str:
    """Validate string parameter."""
    if not isinstance(value, str):
        raise ValueError(f"'{name}' must be a string, got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise ValueError(f"'{name}' must not be empty")
    return value


def require_int(value: Any, name: str, *, min_val: int | None = None,
                max_val: int | None = None) -> int:
    """Validate integer parameter."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"'{name}' must be an integer, got {type(value).__name__}")
    if min_val is not None and value < min_val:
        raise ValueError(f"'{name}' must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValueError(f"'{name}' must be <= {max_val}, got {value}")
    return value


def require_float(value: Any, name: str, *, min_val: float | None = None,
                  max_val: float | None = None) -> float:
    """Validate float parameter."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"'{name}' must be a number, got {type(value).__name__}")
    val = float(value)
    if min_val is not None and val < min_val:
        raise ValueError(f"'{name}' must be >= {min_val}, got {val}")
    if max_val is not None and val > max_val:
        raise ValueError(f"'{name}' must be <= {max_val}, got {val}")
    return val


def require_bool(value: Any, name: str) -> bool:
    """Validate boolean parameter."""
    if not isinstance(value, bool):
        raise ValueError(f"'{name}' must be a boolean, got {type(value).__name__}")
    return value


def require_list(value: Any, name: str) -> list:
    """Validate list parameter."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"'{name}' must be a list, got {type(value).__name__}")
    return list(value)


def require_dict(value: Any, name: str) -> dict:
    """Validate dict parameter."""
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a dict, got {type(value).__name__}")
    return value


def validate_one_of(value: Any, name: str, valid: set[str]) -> str:
    """Validate value is one of the allowed options."""
    if value not in valid:
        raise ValueError(
            f"'{name}' must be one of {sorted(valid)}, got '{value}'"
        )
    return value


def validate_padding(value: Any) -> int | list[int]:
    """Validate padding parameter.

    Accepts: int, or list of 2 or 4 ints.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ValueError(f"padding must be >= 0, got {value}")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) not in (2, 4):
            raise ValueError(
                f"padding list must have 2 or 4 values, got {len(value)}"
            )
        for i, v in enumerate(value):
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(
                    f"padding[{i}] must be a non-negative integer, got {v}"
                )
        return list(value)
    raise ValueError(
        f"padding must be an int or list of 2/4 ints, got {type(value).__name__}"
    )


def validate_ratios(ratios: Any) -> list[int]:
    """Validate column ratios."""
    if not isinstance(ratios, (list, tuple)):
        raise ValueError(f"ratios must be a list, got {type(ratios).__name__}")
    if len(ratios) < 2:
        raise ValueError("ratios must have at least 2 values")
    for i, r in enumerate(ratios):
        if not isinstance(r, int) or isinstance(r, bool) or r < 1:
            raise ValueError(
                f"ratios[{i}] must be a positive integer, got {r}"
            )
    return list(ratios)


def validate_name(value: str, kind: str = "name") -> str:
    """Validate an identifier name (display, skill, etc)."""
    require_str(value, kind)
    if not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            f"'{kind}' must contain only alphanumeric characters, "
            f"underscores, and hyphens, got '{value}'"
        )
    return value


def validate_skill_name(value: str) -> str:
    """Validate a skill name per Anthropic's Agent Skills spec.

    Stricter than ``validate_name``: enforces lowercase, ≤64-char cap,
    and rejects the reserved words ``anthropic`` / ``claude``.
    Underscores are kept (project convention; existing skills use them).
    """
    require_str(value, "skill name")
    if value != value.lower():
        raise ValueError(f"skill name must be lowercase, got '{value}'")
    if len(value) > SKILL_NAME_MAX_LEN:
        raise ValueError(
            f"skill name must be ≤{SKILL_NAME_MAX_LEN} chars, got {len(value)}"
        )
    if value.lower() in RESERVED_SKILL_NAMES:
        raise ValueError(
            f"'{value}' is a reserved skill name and cannot be used"
        )
    if "<" in value or ">" in value:
        raise ValueError(f"skill name must not contain XML brackets, got '{value}'")
    if not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            f"skill name must contain only lowercase letters, digits, "
            f"underscores, and hyphens, got '{value}'"
        )
    return value


def validate_skill_description(value: str) -> str:
    """Validate a skill description per Anthropic's Agent Skills spec.

    ≤1024 chars, non-empty, no XML brackets.
    """
    require_str(value, "description")
    if len(value) > SKILL_DESCRIPTION_MAX_LEN:
        raise ValueError(
            f"description must be ≤{SKILL_DESCRIPTION_MAX_LEN} chars, "
            f"got {len(value)}"
        )
    if "<" in value or ">" in value:
        raise ValueError("description must not contain XML brackets")
    return value


def validate_data_source_config(name: str, source_type: str | None = None,
                                **kwargs: Any) -> dict:
    """Validate a data source declaration."""
    require_str(name, "data source name")

    # Built-in sources need no config
    if source_type is None:
        if name not in VALID_BUILTIN_SOURCES:
            raise ValueError(
                f"Unknown built-in data source '{name}'. "
                f"Valid built-in sources: {sorted(VALID_BUILTIN_SOURCES)}. "
                f"For custom sources, provide a 'type' parameter."
            )
        return {"name": name}

    validate_one_of(source_type, "type", VALID_DATA_SOURCE_TYPES)

    config: dict[str, Any] = {"name": name, "type": source_type}

    if source_type in ("http_json", "http_text"):
        if "url" not in kwargs:
            raise ValueError(f"'{source_type}' data source requires 'url'")
        config["url"] = require_str(kwargs["url"], "url")
        if "params" in kwargs:
            config["params"] = require_dict(kwargs["params"], "params")
        if "secret" in kwargs:
            config["secret"] = require_str(kwargs["secret"], "secret")
        if "refresh" in kwargs:
            config["refresh"] = require_int(kwargs["refresh"], "refresh", min_val=1)
        if "fields" in kwargs:
            config["fields"] = require_dict(kwargs["fields"], "fields")

    elif source_type == "static":
        if "value" not in kwargs:
            raise ValueError("'static' data source requires 'value'")
        config["value"] = kwargs["value"]

    elif source_type == "memory_query":
        if "query" not in kwargs:
            raise ValueError("'memory_query' data source requires 'query'")
        config["query"] = require_str(kwargs["query"], "query")
        if "refresh" in kwargs:
            config["refresh"] = require_int(kwargs["refresh"], "refresh", min_val=1)

    return config
