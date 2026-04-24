"""Display spec parsing, validation, and data binding resolution.

A display spec is a JSON document that declares a display's layout (as a
block tree), data sources, theme, and transition. This module handles
parsing specs into structured objects, resolving data bindings, and
validating specs against the block schema.

Binding syntax:
    {source.field}       — scalar or array from a named data source
    {source.field[0].x}  — nested field access
    {.field}             — current item in a repeat block
    {current.field}      — current item in a rotate block

Usage:
    from boxbot.displays.spec import parse_spec, resolve_bindings, validate_spec

    spec = parse_spec(json_dict)
    errors = validate_spec(spec)
    resolved = resolve_bindings(spec, data)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from boxbot.displays.blocks import Block, parse_block

logger = logging.getLogger(__name__)

# Pattern matching {source.field}, {.field}, {current.field}
_BINDING_PATTERN = re.compile(r"\{([^}]+)\}")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DataSourceSpec:
    """Declaration of a data source within a display spec."""

    name: str
    source_type: str = "builtin"  # builtin, http_json, http_text, static, memory_query
    url: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    secret: str | None = None
    refresh: int | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    value: Any = None  # for static sources
    query: str | None = None  # for memory_query


@dataclass
class DisplaySpec:
    """Complete parsed display specification."""

    name: str
    theme: str = "boxbot"
    data_sources: list[DataSourceSpec] = field(default_factory=list)
    root_block: Block | None = None
    transition: str = "crossfade"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_spec(data: dict[str, Any]) -> DisplaySpec:
    """Parse a JSON display spec dict into a DisplaySpec.

    Args:
        data: The raw JSON dict (from display.json or SDK output).

    Returns:
        A validated DisplaySpec instance.
    """
    name = data.get("name", "unnamed")
    theme = data.get("theme", "boxbot")
    transition = data.get("transition", "crossfade")

    # Parse data sources
    sources: list[DataSourceSpec] = []
    for src_data in data.get("data_sources", []):
        sources.append(_parse_data_source(src_data))

    # Parse the root block tree
    root_block = None
    layout_data = data.get("layout")
    if layout_data:
        root_block = parse_block(layout_data)

    return DisplaySpec(
        name=name,
        theme=theme,
        data_sources=sources,
        root_block=root_block,
        transition=transition,
    )


def _parse_data_source(data: dict[str, Any]) -> DataSourceSpec:
    """Parse a single data source declaration."""
    return DataSourceSpec(
        name=data.get("name", ""),
        source_type=data.get("type", "builtin"),
        url=data.get("url"),
        params=data.get("params", {}),
        secret=data.get("secret"),
        refresh=data.get("refresh"),
        fields=data.get("fields", {}),
        value=data.get("value"),
        query=data.get("query"),
    )


# ---------------------------------------------------------------------------
# Data binding resolution
# ---------------------------------------------------------------------------


def resolve_bindings(block: Block, data: dict[str, Any],
                     repeat_item: dict[str, Any] | None = None,
                     current_item: dict[str, Any] | None = None) -> Block:
    """Resolve data bindings in a block tree, replacing {source.field}
    placeholders with actual values from the data dict.

    Args:
        block: The root block to resolve.
        data: Dict mapping source names to their fetched data dicts.
        repeat_item: The current item when inside a repeat block.
        current_item: The current item when inside a rotate block.

    Returns:
        A new block tree with bindings resolved to concrete values.
    """
    import copy
    resolved = copy.deepcopy(block)
    _resolve_block(resolved, data, repeat_item, current_item)
    return resolved


def _resolve_block(block: Block, data: dict[str, Any],
                   repeat_item: dict[str, Any] | None,
                   current_item: dict[str, Any] | None) -> None:
    """Recursively resolve bindings in-place on a block and its children."""
    # Resolve params
    for key, value in block.params.items():
        block.params[key] = _resolve_value(value, data, repeat_item, current_item)

    # Handle repeat blocks: iterate over bound data
    if block.block_type == "repeat" and block.children:
        source_binding = block.params.get("source", "")
        items = _resolve_value(source_binding, data, repeat_item, current_item)
        # Accept bare paths (e.g. "tasks.items") in addition to brace form
        # ("{tasks.items}"). The brace form would have been resolved above;
        # fall back to a direct lookup if the value is still a string path.
        if not isinstance(items, list) and isinstance(source_binding, str):
            path = source_binding.strip().strip("{}").strip()
            if path:
                items = _lookup_binding(path, data, repeat_item, current_item)
        if isinstance(items, list):
            max_items = block.params.get("max")
            if max_items and len(items) > max_items:
                items = items[:max_items]
            # The template is the first child. Expand it for each item.
            template = block.children[0] if block.children else None
            if template:
                import copy
                expanded: list[Block] = []
                for item in items:
                    child = copy.deepcopy(template)
                    _resolve_block(child, data, repeat_item=item, current_item=current_item)
                    expanded.append(child)
                block.children = expanded
        return

    # Recurse into children
    for child in block.children:
        _resolve_block(child, data, repeat_item, current_item)


def _resolve_value(value: Any, data: dict[str, Any],
                   repeat_item: dict[str, Any] | None,
                   current_item: dict[str, Any] | None) -> Any:
    """Resolve a single value, which may contain binding expressions."""
    if isinstance(value, str):
        return _resolve_string(value, data, repeat_item, current_item)
    if isinstance(value, list):
        return [_resolve_value(v, data, repeat_item, current_item) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_value(v, data, repeat_item, current_item)
                for k, v in value.items()}
    return value


def _resolve_string(text: str, data: dict[str, Any],
                    repeat_item: dict[str, Any] | None,
                    current_item: dict[str, Any] | None) -> Any:
    """Resolve binding expressions in a string.

    If the entire string is a single binding (e.g. "{weather.forecast}"),
    returns the raw value (could be a list, dict, etc.).
    If the string contains mixed text and bindings (e.g. "{weather.temp}F"),
    returns a string with bindings replaced by their string representations.
    """
    # Check if the entire string is exactly one binding
    match = _BINDING_PATTERN.fullmatch(text)
    if match:
        return _lookup_binding(match.group(1), data, repeat_item, current_item)

    # Mixed: replace each binding with its string form
    def replacer(m: re.Match) -> str:
        val = _lookup_binding(m.group(1), data, repeat_item, current_item)
        return str(val) if val is not None else ""

    return _BINDING_PATTERN.sub(replacer, text)


def _lookup_binding(path: str, data: dict[str, Any],
                    repeat_item: dict[str, Any] | None,
                    current_item: dict[str, Any] | None) -> Any:
    """Look up a binding path in the data context.

    Paths:
        .field           → repeat_item[field]
        current.field    → current_item[field]
        source.field     → data[source][field]
        source.field[0]  → data[source][field][0]
    """
    if path.startswith("."):
        # Repeat item binding: {.field}
        field_path = path[1:]
        if repeat_item is not None:
            return _navigate(repeat_item, field_path)
        return None

    if path.startswith("current."):
        # Rotate current item binding: {current.field}
        field_path = path[len("current."):]
        if current_item is not None:
            return _navigate(current_item, field_path)
        return None

    # Source.field binding: {weather.temp}, {weather.forecast[0].high}
    parts = path.split(".", 1)
    source_name = parts[0]
    field_path = parts[1] if len(parts) > 1 else ""

    source_data = data.get(source_name)
    if source_data is None:
        return None

    if not field_path:
        return source_data

    return _navigate(source_data, field_path)


def _navigate(obj: Any, path: str) -> Any:
    """Navigate a dotted path with optional array indices.

    Examples: "temp", "forecast[0].high", "items[2]"
    """
    if not path:
        return obj

    # Split on dots, but handle [N] indices
    segments = re.split(r"\.(?![^[]*\])", path)

    current = obj
    for segment in segments:
        if current is None:
            return None

        # Check for array index: "field[0]"
        idx_match = re.match(r"^(\w+)\[(\d+)\]$", segment)
        if idx_match:
            field_name = idx_match.group(1)
            index = int(idx_match.group(2))
            if isinstance(current, dict):
                current = current.get(field_name)
            else:
                current = getattr(current, field_name, None)
            if isinstance(current, (list, tuple)) and index < len(current):
                current = current[index]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(segment)
        else:
            current = getattr(current, segment, None)

    return current


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_spec(spec: DisplaySpec) -> list[str]:
    """Validate a display spec and return a list of error messages.

    An empty list means the spec is valid.

    Args:
        spec: The DisplaySpec to validate.

    Returns:
        List of human-readable error strings.
    """
    errors: list[str] = []

    if not spec.name:
        errors.append("Display spec must have a name")

    if spec.transition not in ("crossfade", "slide_left", "slide_right", "none"):
        errors.append(
            f"Invalid transition '{spec.transition}'. "
            "Must be: crossfade, slide_left, slide_right, none"
        )

    # Validate data sources
    source_names: set[str] = set()
    for src in spec.data_sources:
        if not src.name:
            errors.append("Data source must have a name")
        elif src.name in source_names:
            errors.append(f"Duplicate data source name: '{src.name}'")
        source_names.add(src.name)

        if src.source_type == "http_json" and not src.url:
            errors.append(f"http_json source '{src.name}' requires a URL")
        if src.source_type == "http_text" and not src.url:
            errors.append(f"http_text source '{src.name}' requires a URL")
        if src.source_type == "memory_query" and not src.query:
            errors.append(f"memory_query source '{src.name}' requires a query")

    # Validate block tree
    if spec.root_block:
        _validate_block(spec.root_block, errors, source_names)

    return errors


def _validate_block(block: Block, errors: list[str],
                    source_names: set[str]) -> None:
    """Recursively validate a block and its children."""
    from boxbot.displays.blocks import BLOCK_REGISTRY

    if block.block_type not in BLOCK_REGISTRY:
        errors.append(f"Unknown block type: '{block.block_type}'")

    # Validate specific block types
    if block.block_type == "text" and not block.params.get("content"):
        errors.append("Text block requires 'content'")

    if block.block_type == "metric" and not block.params.get("value"):
        errors.append("Metric block requires 'value'")

    if block.block_type == "chart":
        if not block.params.get("data") and not block.params.get("series"):
            errors.append("Chart block requires 'data' or 'series'")

    # Recurse
    for child in block.children:
        _validate_block(child, errors, source_names)
