"""Built-in display specs shipped with boxBot.

These are registered programmatically with the DisplayManager during
startup. They use the block system (not raw pygame) so they work with
both the preview renderer and the live pygame renderer.

Built-in displays:
- clock: Simple clock with date
- weather_simple: Current weather with temperature and condition
- picture: Show a single photo, parameterized by args.image_ids
- notice: Centered text card; agent's quick "show this short message"
  channel — used during onboarding to surface a bootstrap code, etc.
- web: Placeholder card under the kiosk browser (args.url). The display
  manager opens the browser window on top; this is what shows while it
  loads.
- product: One product from a purchase search — image, title, price.
"""

from boxbot.displays.blocks import (
    CardBlock,
    ClockBlock,
    ColumnBlock,
    ImageBlock,
    RowBlock,
    TextBlock,
)
from boxbot.displays.spec import DataSourceSpec, DisplaySpec


def get_builtin_specs() -> list[DisplaySpec]:
    """Return all built-in display specs."""
    return [
        _clock_display(),
        _weather_simple_display(),
        _picture_display(),
        _notice_display(),
        _web_display(),
        _product_display(),
    ]


def _clock_display() -> DisplaySpec:
    """Full-screen clock, time + date centered as a group."""
    root = ColumnBlock(gap=0, align="center", padding=[0, 0, 0, 0])
    root.children = [
        ClockBlock(format="12h", show_date=True, show_seconds=False, size="xl"),
    ]

    return DisplaySpec(
        name="clock",
        theme="boxbot",
        data_sources=[],
        root_block=root,
        transition="crossfade",
    )


def _weather_simple_display() -> DisplaySpec:
    """Current conditions as a centered hero: icon + temp, condition,
    humidity/wind footnote."""
    from boxbot.displays.blocks import IconBlock, SpacerBlock

    root = ColumnBlock(
        gap=10, align="center", item_align="center", padding=[40, 40, 40, 40],
    )

    hero = RowBlock(gap=20, align="center")
    hero.children = [
        IconBlock(name="{weather.icon}", size="xl", color="accent"),
        TextBlock(content="{weather.temp}\u00b0", size="title", weight="bold"),
    ]

    details = RowBlock(gap=12, align="center")
    details.children = [
        TextBlock(content="Humidity {weather.humidity}%", size="caption", color="muted"),
        TextBlock(content="\u00b7", size="caption", color="dim"),
        TextBlock(content="Wind {weather.wind}", size="caption", color="muted"),
    ]

    root.children = [
        hero,
        TextBlock(
            content="{weather.condition}", size="subtitle", color="muted",
            align="center",
        ),
        SpacerBlock(size=14),
        details,
    ]

    return DisplaySpec(
        name="weather_simple",
        theme="boxbot",
        data_sources=[
            DataSourceSpec(name="weather", source_type="builtin", refresh=3600),
        ],
        root_block=root,
        transition="crossfade",
    )


def _notice_display() -> DisplaySpec:
    """Short centered text card.

    Called via:
        mgr.switch("notice", args={
            "title": "Welcome to boxBot!",
            "lines": ["Text this code to +1 555 635 8748:",
                      "Code: 123456",
                      "Expires in 10 minutes"],
        })

    Use sparingly — this is the agent's interrupt channel for important
    short strings (bootstrap codes, setup prompts) where the regular
    rotating displays would bury the message.
    """
    from boxbot.displays.blocks import SpacerBlock

    root = ColumnBlock(
        gap=0, align="center", item_align="center", padding=[0, 0, 0, 0],
    )
    card = CardBlock(padding=[44, 64, 44, 64])
    body = ColumnBlock(gap=10, item_align="center")
    body.children = [
        TextBlock(
            content="{args.title}", size="heading", weight="bold",
            align="center",
        ),
        SpacerBlock(size=8),
        TextBlock(content="{args.lines[0]}", size="body", align="center"),
        TextBlock(content="{args.lines[1]}", size="subtitle", align="center"),
        TextBlock(
            content="{args.lines[2]}",
            size="caption",
            color="muted",
            align="center",
        ),
    ]
    card.children = [body]
    root.children = [card]

    return DisplaySpec(
        name="notice",
        theme="boxbot",
        data_sources=[],
        root_block=root,
        transition="crossfade",
    )


def _picture_display() -> DisplaySpec:
    """Full-screen photo viewer, parameterized by ``args.image_ids``.

    Called via:
        mgr.switch("picture", args={"image_ids": ["abc123..."]})
        mgr.switch("picture", args={"image_ids": ["a", "b", "c"],
                                     "interval": 8})  # slideshow

    Shows ``args.image_ids[0]`` full-screen with "contain" fit. The
    image block is the root so it receives the full 1024x600 canvas —
    wrapping it in a column made the image-height estimator cap it at
    200 px. ``_render_image`` then scales the photo to fit and centers
    it inside the rect. The ``photo:`` source is pre-resolved to an
    absolute file path by the display manager (``_resolve_photo_sources``)
    before rendering.

    Slideshow: when ``image_ids`` has more than one entry, the display
    manager starts a per-photo tick that rotates the id list every
    ``args.interval`` seconds (default 8). The spec stays unchanged;
    only the list rotates and the manager re-renders. Switching to
    another display (or to picture with a single id) stops the tick.
    """
    return DisplaySpec(
        name="picture",
        theme="boxbot",
        data_sources=[],
        root_block=ImageBlock(
            source="photo:{args.image_ids[0]}",
            fit="contain",
        ),
        transition="crossfade",
    )


def _web_display() -> DisplaySpec:
    """Card shown underneath the kiosk browser while it starts.

    Called via:
        mgr.switch("web", args={"url": "https://youtu.be/..."})

    The page itself is not drawn by the block renderer. The display
    manager validates ``args.url``, renders this card, then opens a
    browser window over it (see ``boxbot.displays.web_view``).
    ``args.host`` is filled in by the manager.
    """
    from boxbot.displays.blocks import IconBlock

    root = ColumnBlock(
        gap=14, align="center", item_align="center", padding=[40, 40, 40, 40],
    )
    root.children = [
        IconBlock(name="tv", size="xl", color="muted"),
        TextBlock(content="Opening…", size="heading", weight="bold", align="center"),
        TextBlock(
            content="{args.host}", size="body", color="muted", align="center",
            max_lines=1,
        ),
    ]
    return DisplaySpec(
        name="web",
        theme="boxbot",
        data_sources=[],
        root_block=root,
        transition="none",
    )


def _product_display() -> DisplaySpec:
    """One product, as BB found it: image left, details right.

    Called via:
        mgr.switch("product", args={
            "title": "Goldilocks Board Book",
            "price": "$11.99",
            "seller": "Cottage Door Press",
            "detail": "4.8 stars (212) · arrives Thursday",
            "image": "tmp/product.jpg",   # workspace path, optional
        })

    Rendered from the search result's own fields rather than the
    merchant's web page: nothing merchant-controlled runs on the box.
    """
    from boxbot.displays.blocks import ColumnsBlock, SpacerBlock

    details = ColumnBlock(gap=12, align="center")
    details.children = [
        TextBlock(content="{args.seller}", size="caption", color="muted", max_lines=1),
        TextBlock(content="{args.title}", size="heading", weight="bold", max_lines=4),
        SpacerBlock(size=4),
        TextBlock(content="{args.price}", size=64, weight="bold", color="accent",
                  max_lines=1),
        TextBlock(content="{args.detail}", size="body", color="muted", max_lines=3),
    ]
    root = ColumnsBlock(ratios=[5, 6], gap=36, padding=[36, 40, 36, 40])
    root.children = [
        ImageBlock(source="photo:{args.image}", fit="contain", radius=18),
        details,
    ]
    return DisplaySpec(
        name="product",
        theme="boxbot",
        data_sources=[],
        root_block=root,
        transition="crossfade",
    )
