"""The Living Sky page in a real Chromium.

``test_sky_renderer.py`` covers the Python side against a fake browser.
This file loads the actual page (``src/boxbot/displays/sky``) and checks
what only a browser can: every fixture and every shipped spec draws
with no unknown or failed block and no leaked ``undefined``; hostile
text stays text; the page cannot reach the network.

Skipped when no Chromium is installed. One browser serves each test —
a fixture per browser launch would be slow and, run in parallel, heavy.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from boxbot.core.config import SkySurfaceConfig
from boxbot.displays.manager import DisplayManager
from boxbot.displays.sky_surface import PAGE_DIR, SkySurface
from boxbot.displays.spec import parse_spec, validate_spec

BROWSER = next(
    (b for b in ("chromium", "chromium-browser", "google-chrome") if shutil.which(b)),
    None,
)
pytestmark = pytest.mark.skipif(BROWSER is None, reason="no Chromium installed")

FIXTURES = sorted(p.stem for p in (PAGE_DIR / "fixtures").glob("*.js"))
SPECS = sorted((PAGE_DIR / "specs").glob("*.json"))

HOSTILE = "<img src=x onerror=\"window.__pwned=1\"><script>window.__pwned=1</script>'\"&"

# What the page reports about itself after a state is applied.
PROBE_JS = """(() => {
  const view = document.getElementById('view'), stage = document.getElementById('stage');
  const text = stage.innerText;
  return {
    unknown: [...stage.querySelectorAll('.bb-unknown')].map(e => e.textContent),
    leaks: (text.match(/undefined|NaN|\\[object Object\\]|Invalid Date/g) || []),
    chars: view.innerText.trim().length,
    pwned: window.__pwned === 1,
    scripts: stage.querySelectorAll('script').length,
    handlers: [...stage.querySelectorAll('*')].filter(e =>
      [...e.attributes].some(a => a.name.startsWith('on'))).length,
    badUrls: [...stage.querySelectorAll('[src],[href]')].map(e =>
      e.getAttribute('src') || e.getAttribute('href')).filter(u => /^\\s*javascript:/i.test(u)),
  };
})()"""


def config(tmp_path: Path) -> SkySurfaceConfig:
    return SkySurfaceConfig(
        browser=BROWSER, headless=True, state_dir=str(tmp_path / "display"),
        people=["Jacob", "Carina", "Erik", "Mia"],
        extra_args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
    )


def problems(name: str, probe: dict[str, Any], *, allow_unknown: bool = False) -> list[str]:
    out = []
    if probe["unknown"] and not allow_unknown:
        out.append(f"{name}: unknown/failed blocks {probe['unknown']}")
    if probe["leaks"]:
        out.append(f"{name}: leaked {sorted(set(probe['leaks']))}")
    if probe["pwned"] or probe["scripts"] or probe["handlers"] or probe["badUrls"]:
        out.append(f"{name}: data became markup or code {probe}")
    return out


def hostile(value: Any) -> Any:
    """``value`` with every string that a block prints replaced by markup.

    Structural strings — block types, enum-like single words, paths,
    times — are left alone so the block still takes its normal path.
    """
    if isinstance(value, dict):
        return {k: (v if k in ("type", "src", "thumb", "source") else hostile(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [hostile(v) for v in value]
    if isinstance(value, str) and (" " in value or len(value) > 24):
        return HOSTILE
    return value


async def test_every_fixture_draws_cleanly(tmp_path) -> None:
    assert FIXTURES, "no fixtures found"
    surface = SkySurface(config(tmp_path))
    await surface.start()
    failures: list[str] = []
    try:
        for name in FIXTURES:
            source = (PAGE_DIR / "fixtures" / f"{name}.js").read_text()
            await surface.evaluate(f"BB.fixture=null;{source}\n;BB.apply(BB.fixture)")
            probe = await surface.evaluate(PROBE_JS)
            # ``smoke`` contains an unknown block on purpose.
            failures += problems(name, probe, allow_unknown=name == "smoke")
            if not probe["chars"] and "empty" not in name and "picture" not in name:
                failures.append(f"{name}: drew no text at all")
    finally:
        await surface.stop()
    assert not failures, "\n".join(failures)


async def test_hostile_text_stays_text(tmp_path) -> None:
    """Every fixture again, with its prose swapped for markup that
    would run if any block forgot to escape."""
    surface = SkySurface(config(tmp_path))
    await surface.start()
    failures: list[str] = []
    try:
        for name in FIXTURES:
            source = (PAGE_DIR / "fixtures" / f"{name}.js").read_text()
            state = await surface.evaluate(f"BB.fixture=null;{source}\n;BB.fixture")
            for key in ("layout", "data", "args", "presence", "status", "people"):
                if key in state:
                    state[key] = hostile(state[key])
            state["status"] = HOSTILE
            state["presence"] = {"state": "speaking", "who": HOSTILE, "text": HOSTILE}
            await surface.apply(state)
            probe = await surface.evaluate(PROBE_JS)
            if probe["pwned"] or probe["scripts"] or probe["handlers"] or probe["badUrls"]:
                failures.append(f"{name}: {probe}")
    finally:
        await surface.stop()
    assert not failures, "\n".join(failures)


async def test_page_is_offline_and_refuses_inline_script(tmp_path) -> None:
    surface = SkySurface(config(tmp_path))
    await surface.start()
    try:
        reached = await surface.evaluate(
            "fetch('https://example.com/').then(() => 'reached', e => 'blocked')"
        )
        assert reached == "blocked"
        by_ip = await surface.evaluate(
            "fetch('http://1.1.1.1/').then(() => 'reached', e => 'blocked')"
        )
        assert by_ip == "blocked"
        # Markup injected past the blocks (as if one forgot to escape):
        # the CSP must still keep it from running.
        await surface.evaluate(
            "document.getElementById('view').innerHTML = "
            + json.dumps(HOSTILE) + "; new Promise(r => setTimeout(r, 300))"
        )
        assert await surface.evaluate("window.__pwned === 1") is False
    finally:
        await surface.stop()


async def test_every_shipped_spec_validates_and_previews(tmp_path) -> None:
    """Spec → bindings → state → page, on placeholder data, the way
    ``bb.display.preview`` does it."""
    assert SPECS, "no sky specs found"
    for path in SPECS:
        spec = parse_spec(json.loads(path.read_text()))
        assert validate_spec(spec) == [], path.name
        assert spec.theme == "living_sky", path.name

    manager = DisplayManager(renderer="html", sky_config=config(tmp_path))
    await manager.start(user_dir=tmp_path / "none", agent_dir=tmp_path / "none")
    failures: list[str] = []
    try:
        assert manager.renderer == "html"
        for name in sorted(manager.list_available()):
            image = await manager.render_preview_image(name)
            if image is None or image.size != (1024, 600):
                failures.append(f"{name}: no preview")
                continue
            probe = await manager._sky_preview.evaluate(PROBE_JS)
            failures += problems(name, probe)
    finally:
        await manager.stop()
    assert not failures, "\n".join(failures)


async def test_page_and_python_agree_on_the_block_vocabulary(tmp_path) -> None:
    """A block the page draws but Python does not know cannot be saved
    by the agent; one Python knows but the page does not draw shows a
    placeholder. The two lists must be the same."""
    from boxbot.displays.blocks import BLOCK_REGISTRY
    from boxbot.displays.sky_blocks import SKY_BLOCKS

    surface = SkySurface(config(tmp_path))
    await surface.start()
    try:
        drawn = set(await surface.evaluate("Object.keys(BB.blocks)"))
    finally:
        await surface.stop()
    overlays = {"status_pill", "presence"}  # drawn from state, not from a layout
    # A chart's own ``type`` param (line/bar/area) overwrites the block
    # type when the tree is serialised, so the page answers to those too.
    chart_kinds = {"line", "bar", "area"}
    assert drawn - overlays - chart_kinds == set(BLOCK_REGISTRY) | set(SKY_BLOCKS)


async def test_page_loads_only_images_python_listed(tmp_path) -> None:
    """A path is an ``<img>`` source only if ``state.images`` lists it."""
    from PIL import Image

    listed, unlisted = tmp_path / "listed.png", tmp_path / "unlisted.png"
    for path in (listed, unlisted):
        Image.new("RGB", (4, 4), (255, 0, 0)).save(path)
    surface = SkySurface(config(tmp_path))
    await surface.start()
    try:
        await surface.apply({
            "v": 1, "display": "t", "transition": "none", "people": [],
            "atmosphere": {"phase": "noon", "weather": "clear", "time": "12:00"},
            "images": [str(listed)],
            "layout": {"type": "column", "children": [
                {"type": "image", "source": str(listed)},
                {"type": "image", "source": str(unlisted)},
                {"type": "image", "source": "file://" + str(unlisted)},
                {"type": "image", "source": "file:///etc/hostname"},
                {"type": "picture", "photos": [str(unlisted), {"src": "../../../../../etc/x.png"}]},
                {"type": "capture", "image": str(unlisted)},
                {"type": "picture", "photos": [{"id": "ok", "src": str(listed)}]},
            ]},
        })
        sources = await surface.evaluate(
            "[...document.querySelectorAll('#view img')].map(i => i.getAttribute('src'))"
        )
        assert sources and all(s.endswith("/listed.png") for s in sources), sources
        assert await surface.evaluate(
            "[BB.img('/etc/passwd'), BB.img('file:///etc/passwd'), BB.img('../x.png'),"
            " BB.img('a/../../x.png'), BB.img('https://example.com/x.png'),"
            " BB.img('data:text/html;base64,AAAA')]"
        ) == ["", "", "", "", "", ""]
    finally:
        await surface.stop()


async def test_a_new_minute_repaints_the_scene_not_the_view(tmp_path) -> None:
    """The time moves the scene (ferry, windows). The view pane must
    survive it, or captions, zooms and countdowns restart every minute."""
    source = (PAGE_DIR / "fixtures" / "inbox-sent.js").read_text()
    surface = SkySurface(config(tmp_path))
    await surface.start()
    try:
        state = await surface.evaluate(f"BB.fixture=null;{source}\n;BB.fixture")
        state["atmosphere"] = {**state.get("atmosphere", {}), "time": "16:02"}
        await surface.apply(state)
        probe = (
            "(() => { const p = document.querySelector('#view > *'), u = document.querySelector('.ib-undo');"
            " return {mark: (p && p.dataset.mark) || null, until: u ? u.dataset.until : null,"
            " scene: document.getElementById('scene').innerHTML.length}; })()"
        )
        await surface.evaluate("document.querySelector('#view > *').dataset.mark = 'first'")
        before = await surface.evaluate(probe)
        assert before["mark"] == "first" and before["until"]
        state["atmosphere"] = {**state["atmosphere"], "time": "16:03"}
        await surface.apply(state)
        after = await surface.evaluate(probe)
        assert after["mark"] == "first", "the view pane was rebuilt by a time change"
        assert after["until"] == before["until"]
        # A real change rebuilds the pane, and the countdown keeps its deadline.
        state["status"] = None
        state["people"] = list(state.get("people") or []) + ["Zed"]
        await surface.apply(state)
        rebuilt = await surface.evaluate(probe)
        assert rebuilt["mark"] is None and rebuilt["until"] == before["until"]
        assert await surface.evaluate("document.getElementById('frame').getAttribute('src')") is None
    finally:
        await surface.stop()


async def test_frame_is_released_when_no_longer_shown(tmp_path) -> None:
    from tests.fake_sky_browser import PNG

    surface = SkySurface(config(tmp_path))
    await surface.start()
    try:
        base = {"v": 1, "display": "t", "transition": "none", "people": [],
                "atmosphere": {"phase": "noon", "weather": "clear", "time": "12:00"},
                "layout": {"type": "column"}}
        await surface.apply({**base, "frame": "data:image/png;base64," + PNG})
        assert await surface.evaluate("document.getElementById('stage').classList.contains('raw')")
        await surface.apply({**base, "frame": "javascript:alert(1)"})
        assert not await surface.evaluate("document.getElementById('stage').classList.contains('raw')")
        assert await surface.evaluate("document.getElementById('frame').getAttribute('src')") is None
    finally:
        await surface.stop()


async def test_cross_fade_keeps_glass_blurring_the_scene(tmp_path) -> None:
    """A pane must never fade as a whole: opacity on an ancestor cuts a
    glass surface off from the scene, and its blur drops out for the
    length of the fade. Pieces fade instead."""
    surface = SkySurface(config(tmp_path))
    await surface.start()
    try:
        load = "BB.fixture=null;{src}\n;BB.still=false;BB.apply(BB.fixture)"
        for name in ("home", "brief"):
            source = (PAGE_DIR / "fixtures" / f"{name}.js").read_text()
            await surface.evaluate(load.format(src=source))
        mid = await surface.evaluate("""new Promise(done => setTimeout(() => {
          const panes = [...document.querySelectorAll('#view > .pane')];
          const glass = document.querySelector('#view > .pane:not(.out) .slab');
          const op = e => +getComputedStyle(e).opacity;
          let cut = false;
          for (let e = glass.parentElement; e && e.id !== 'stage'; e = e.parentElement) {
            const s = getComputedStyle(e);
            if (+s.opacity < 1 || s.filter !== 'none' || s.maskImage !== 'none') cut = true;
          }
          done({panes: panes.length, cut, glass: op(glass),
                blur: getComputedStyle(glass).backdropFilter,
                leaving: Math.max(...[...panes[0].querySelectorAll('.fx')].map(op))});
        }, 250))""")
        assert mid["panes"] == 2
        assert not mid["cut"], "an ancestor of the glass surface is fading"
        assert "blur" in mid["blur"]
        assert 0 < mid["glass"] < 1 and 0 < mid["leaving"] < 1, mid
        left = await surface.evaluate("""new Promise(done => setTimeout(() => done(
          [...document.querySelectorAll('#view > .pane')].map(p =>
            [p.className, getComputedStyle(p.querySelector('.slab')).opacity])), 900))""")
        assert len(left) == 1 and "d-brief" in left[0][0] and left[0][1] == "1"
    finally:
        await surface.stop()
