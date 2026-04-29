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
    ]


def _clock_display() -> DisplaySpec:
    """Simple full-screen clock with date."""
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
    """Simple weather display with current conditions."""
    from boxbot.displays.blocks import IconBlock, MetricBlock, SpacerBlock

    root = ColumnBlock(gap=16, align="center", padding=[40, 40, 40, 40])

    header = RowBlock(gap=16, align="center")
    header.children = [
        IconBlock(name="{weather.icon}", size="xl"),
        MetricBlock(
            value="{weather.temp}\u00b0F",
            label="{weather.condition}",
            animation="count_up",
        ),
    ]

    details = RowBlock(gap=24, align="center")
    details.children = [
        TextBlock(content="Humidity: {weather.humidity}%", size="body", color="muted"),
        TextBlock(content="Wind: {weather.wind}", size="body", color="muted"),
    ]

    root.children = [
        SpacerBlock(size=40),
        header,
        SpacerBlock(size=16),
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

    root = ColumnBlock(gap=0, align="center", padding=[0, 0, 0, 0])
    card = CardBlock(padding=[40, 56, 40, 56])
    body = ColumnBlock(gap=20, align="center")
    body.children = [
        TextBlock(content="{args.title}", size="xl", weight="bold", align="center"),
        SpacerBlock(size=8),
        TextBlock(content="{args.lines[0]}", size="body", align="center"),
        TextBlock(content="{args.lines[1]}", size="body", align="center"),
        TextBlock(
            content="{args.lines[2]}",
            size="body",
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
        mgr.switch("picture", args={"image_ids": ["abc123...", ...]})

    For v1, shows the first id full-screen with "contain" fit. The
    photo: source is pre-resolved to an absolute file path by the
    display manager (``_resolve_photo_sources``) before rendering.
    """
    root = ColumnBlock(gap=0, align="center", padding=[0, 0, 0, 0])
    root.children = [
        ImageBlock(
            source="photo:{args.image_ids[0]}",
            fit="contain",
        ),
    ]

    return DisplaySpec(
        name="picture",
        theme="boxbot",
        data_sources=[],
        root_block=root,
        transition="crossfade",
    )
