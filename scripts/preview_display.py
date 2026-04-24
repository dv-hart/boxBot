#!/usr/bin/env python3
"""Render a boxBot display spec to a PNG for visual review.

Runs headless on any machine (no HDMI or pygame needed). Uses the same
DisplayRenderer the live system uses, so previews are pixel-accurate.

Usage:
    scripts/preview_display.py <spec.json|spec.yaml> [-o out.png] [-t theme] [--builtin clock]
    scripts/preview_display.py --builtin weather_simple -t midnight -o weather.png
    scripts/preview_display.py --list

Specs can be JSON or YAML. When --theme is passed, it overrides the theme
declared in the spec. Missing data sources are filled with placeholder data
(weather, calendar, tasks, etc.) so layouts render realistically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from boxbot.displays.builtins import get_builtin_specs
from boxbot.displays.data_sources import get_placeholder_data
from boxbot.displays.renderer import DisplayRenderer
from boxbot.displays.spec import DisplaySpec, parse_spec, validate_spec
from boxbot.displays.themes import get_theme, list_themes


def load_spec(path: Path) -> DisplaySpec:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    spec = parse_spec(data)
    errors = validate_spec(spec)
    if errors:
        print(f"warning: spec has validation errors: {errors}", file=sys.stderr)
    return spec


def find_builtin(name: str) -> DisplaySpec:
    for spec in get_builtin_specs():
        if spec.name == name:
            return spec
    names = [s.name for s in get_builtin_specs()]
    raise SystemExit(f"no built-in display named '{name}'. available: {names}")


def build_preview_data(spec: DisplaySpec) -> dict:
    data: dict = {}
    for src in spec.data_sources:
        placeholder = get_placeholder_data(src.name)
        if placeholder:
            data[src.name] = placeholder
    return data


def build_live_data(spec: DisplaySpec) -> dict:
    """Fetch each declared source via its real backend.

    Falls back to the placeholder if the source has no live data
    available (e.g. perception not running, calendar not authenticated).
    """
    import asyncio

    from boxbot.displays.data_sources import create_source

    async def _gather() -> dict:
        out: dict = {}
        for src_def in spec.data_sources:
            try:
                config: dict = {}
                if src_def.refresh is not None:
                    config["refresh"] = src_def.refresh
                if src_def.url:
                    config["url"] = src_def.url
                if src_def.params:
                    config["params"] = src_def.params
                if src_def.value is not None:
                    config["value"] = src_def.value
                if src_def.query:
                    config["query"] = src_def.query
                src = create_source(
                    src_def.name, src_def.source_type, config
                )
                fetched = await src.do_fetch()
                out[src_def.name] = fetched or {}
            except Exception as e:
                print(
                    f"warning: live fetch failed for '{src_def.name}': {e}",
                    file=sys.stderr,
                )
                out[src_def.name] = {}
            if not out[src_def.name]:
                placeholder = get_placeholder_data(src_def.name)
                if placeholder:
                    out[src_def.name] = placeholder
        return out

    return asyncio.run(_gather())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src_group = p.add_mutually_exclusive_group()
    src_group.add_argument("spec", nargs="?", type=Path, help="Path to display spec (JSON or YAML)")
    src_group.add_argument("--builtin", help="Name of built-in spec (e.g. clock, weather_simple)")
    p.add_argument("-o", "--out", type=Path, default=Path("preview.png"), help="Output PNG path (default: preview.png)")
    p.add_argument("-t", "--theme", help=f"Override spec theme. Options: {', '.join(list_themes())}")
    p.add_argument("-W", "--width", type=int, default=1024)
    p.add_argument("-H", "--height", type=int, default=600)
    p.add_argument("--list", action="store_true", help="List built-in specs and themes, then exit")
    p.add_argument("--live", action="store_true",
                   help="Fetch real data from backends (scheduler, calendar, perception, ...). "
                        "Falls back to placeholder if a source returns nothing.")
    args = p.parse_args()

    if args.list:
        print("Built-in specs:")
        for s in get_builtin_specs():
            print(f"  {s.name} (theme: {s.theme})")
        print("\nThemes:")
        for t in list_themes():
            print(f"  {t}")
        return 0

    if args.builtin:
        spec = find_builtin(args.builtin)
    elif args.spec:
        spec = load_spec(args.spec)
    else:
        p.error("must provide a spec path or --builtin name (or --list)")
        return 2

    theme = get_theme(args.theme) if args.theme else get_theme(spec.theme)

    renderer = DisplayRenderer(width=args.width, height=args.height)
    data = build_live_data(spec) if args.live else build_preview_data(spec)
    img = renderer.render(spec, theme=theme, data=data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"wrote {args.out} ({img.size[0]}x{img.size[1]}, theme: {theme.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
