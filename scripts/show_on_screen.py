#!/usr/bin/env python3
"""Display a boxBot preview PNG (or live-rendered spec) fullscreen on the 7" HDMI screen.

Designed to run on the Pi (or any machine with a display) so you can eyeball
real renders during iteration. It either takes a pre-rendered PNG (fast, good
for SCP'ing previews from your dev machine) or renders a spec in-place on the
Pi so fonts and any platform-specific paths match the live system.

Usage (on the Pi):
    scripts/show_on_screen.py <image.png>
    scripts/show_on_screen.py --builtin clock
    scripts/show_on_screen.py --spec displays/my_dashboard/display.json -t midnight
    scripts/show_on_screen.py --image preview.png --rotate 180

Controls:
    q / Esc — quit
    r       — re-render (if --spec or --builtin, picks up edits on disk)
    1/2/3/4 — swap theme on the fly (boxbot/midnight/daylight/classic)
    left/right — previous/next image (if --image is a directory or --glob used)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Fallback SDL drivers so this works over SSH against a framebuffer console
# as well as under X / Wayland. Each driver is tried in order.
_SDL_DRIVERS_DEFAULT = "kmsdrm,fbcon,x11,wayland"


def _render_spec(spec_path: Path | None, builtin: str | None, theme_name: str | None):
    from boxbot.displays.builtins import get_builtin_specs
    from boxbot.displays.data_sources import get_placeholder_data
    from boxbot.displays.renderer import DisplayRenderer
    from boxbot.displays.spec import parse_spec, validate_spec
    from boxbot.displays.themes import get_theme

    if builtin:
        specs = {s.name: s for s in get_builtin_specs()}
        if builtin not in specs:
            raise SystemExit(f"no built-in '{builtin}'. available: {list(specs)}")
        spec = specs[builtin]
    elif spec_path:
        import json
        text = spec_path.read_text()
        if spec_path.suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        spec = parse_spec(data)
        errs = validate_spec(spec)
        if errs:
            print(f"warning: validation: {errs}", file=sys.stderr)
    else:
        raise SystemExit("need --spec or --builtin")

    theme = get_theme(theme_name) if theme_name else get_theme(spec.theme)
    data = {}
    for src in spec.data_sources:
        ph = get_placeholder_data(src.name)
        if ph:
            data[src.name] = ph

    renderer = DisplayRenderer()
    return renderer.render(spec, theme=theme, data=data)


def _pil_to_surface(pygame, img, rotate: int):
    if rotate:
        img = img.rotate(rotate, expand=False)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return pygame.image.fromstring(img.tobytes(), img.size, "RGB")


def _try_init_display(pygame, width: int, height: int, fullscreen: bool, driver_list: str):
    last_err = None
    for drv in driver_list.split(","):
        drv = drv.strip()
        if not drv:
            continue
        os.environ["SDL_VIDEODRIVER"] = drv
        try:
            pygame.display.quit()
            pygame.display.init()
            flags = pygame.FULLSCREEN | pygame.NOFRAME if fullscreen else 0
            surf = pygame.display.set_mode((width, height), flags)
            print(f"pygame display driver: {drv}, size: {surf.get_size()}")
            return surf
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise SystemExit(f"could not initialize pygame display with any driver: {last_err}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("image", nargs="?", type=Path, help="Path to a PNG to display")
    grp.add_argument("--image", dest="image_flag", type=Path, help="Path to a PNG to display")
    grp.add_argument("--spec", type=Path, help="Render a display spec in-place (JSON or YAML)")
    grp.add_argument("--builtin", help="Render a built-in spec (e.g. clock)")
    p.add_argument("-t", "--theme", help="Override theme (boxbot, midnight, daylight, classic)")
    p.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270))
    p.add_argument("--no-fullscreen", action="store_true")
    p.add_argument("--driver", default=_SDL_DRIVERS_DEFAULT, help="SDL driver priority list")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=600)
    args = p.parse_args()

    try:
        import pygame
    except ImportError:
        raise SystemExit("pygame is required. install with: pip install pygame")

    pygame.init()
    screen = _try_init_display(pygame, args.width, args.height, not args.no_fullscreen, args.driver)
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    def load_frame(theme_override: str | None):
        img_path = args.image or args.image_flag
        if img_path:
            from PIL import Image
            return Image.open(img_path).convert("RGB"), f"png: {img_path.name}"
        img = _render_spec(args.spec, args.builtin, theme_override or args.theme)
        label = args.builtin or (args.spec.name if args.spec else "?")
        return img, f"spec: {label} theme: {theme_override or args.theme or '(spec default)'}"

    theme_override = args.theme
    img, label = load_frame(theme_override)
    surface = _pil_to_surface(pygame, img, args.rotate)
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    print(f"showing {label}. press q/Esc to quit.")

    theme_map = {pygame.K_1: "boxbot", pygame.K_2: "midnight", pygame.K_3: "daylight", pygame.K_4: "classic"}

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    img, label = load_frame(theme_override)
                    surface = _pil_to_surface(pygame, img, args.rotate)
                    screen.blit(surface, (0, 0))
                    pygame.display.flip()
                    print(f"reloaded — {label}")
                elif event.key in theme_map and not (args.image or args.image_flag):
                    theme_override = theme_map[event.key]
                    img, label = load_frame(theme_override)
                    surface = _pil_to_surface(pygame, img, args.rotate)
                    screen.blit(surface, (0, 0))
                    pygame.display.flip()
                    print(f"theme → {theme_override}")
        clock.tick(30)
        time.sleep(0)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
