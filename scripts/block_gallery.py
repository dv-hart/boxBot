#!/usr/bin/env python3
"""Render every block type across every theme into labeled PNGs.

Produces two artifact sets under data/previews/gallery/:

  per-block/<theme>/<block>.png     1024x600, one block in context
  contact_sheet_<theme>.png         composite grid of all 24 blocks

Use the contact sheets for design review at a glance; use the per-block
PNGs when you need to inspect a specific block at full resolution.

This script is also a visual regression baseline — re-run it after changes
to themes, blocks, or the renderer, and diff the PNGs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont

from boxbot.displays import blocks as B
from boxbot.displays.data_sources import get_placeholder_data
from boxbot.displays.renderer import DisplayRenderer, _get_font
from boxbot.displays.spec import DataSourceSpec, DisplaySpec
from boxbot.displays.themes import get_theme, hex_to_rgb, list_themes

OUT_DIR = _REPO_ROOT / "data" / "previews" / "gallery"


@dataclass
class BlockDemo:
    """A single block in a self-contained demo spec."""

    block_id: str          # slug used in filenames
    label: str             # human-readable label
    category: str          # Layout / Content / Widget / Meta
    spec: DisplaySpec


def _wrap(children: list[B.Block], pad: int = 40, gap: int = 16) -> B.Block:
    col = B.ColumnBlock(gap=gap, align="start", padding=[pad, pad, pad, pad])
    col.children = children
    return col


def _heading(text: str) -> B.Block:
    return B.TextBlock(content=text, size="subtitle", color="muted")


def _make_demos() -> list[BlockDemo]:
    demos: list[BlockDemo] = []

    # --- Layout containers -------------------------------------------------
    row = B.RowBlock(gap=16, align="center")
    row.children = [
        B.BadgeBlock(text="one", color="accent"),
        B.BadgeBlock(text="two", color="secondary"),
        B.BadgeBlock(text="three", color="success"),
    ]
    demos.append(BlockDemo("row", "row — horizontal flow", "Layout",
        DisplaySpec(name="demo_row", theme="boxbot", root_block=_wrap([_heading("row"), row]))))

    col = B.ColumnBlock(gap=8, align="start")
    col.children = [
        B.TextBlock(content="first", size="heading"),
        B.TextBlock(content="second", size="body", color="muted"),
        B.TextBlock(content="third", size="body", color="muted"),
    ]
    demos.append(BlockDemo("column", "column — vertical stack", "Layout",
        DisplaySpec(name="demo_col", theme="boxbot", root_block=_wrap([_heading("column"), col]))))

    cols = B.ColumnsBlock(ratios=[2, 1, 1], gap=16)
    cols.children = [
        _card_with(B.TextBlock(content="2/4 width", size="heading")),
        _card_with(B.TextBlock(content="1/4", size="body")),
        _card_with(B.TextBlock(content="1/4", size="body")),
    ]
    demos.append(BlockDemo("columns", "columns — weighted split", "Layout",
        DisplaySpec(name="demo_cols", theme="boxbot", root_block=_wrap([_heading("columns ratios=[2,1,1]"), cols]))))

    card = B.CardBlock(padding=24)
    card_col = B.ColumnBlock(gap=8)
    card_col.children = [
        B.TextBlock(content="Card surface", size="heading"),
        B.TextBlock(content="rounded corners, themed fill, optional shadow", size="caption", color="muted"),
    ]
    card.children = [card_col]
    demos.append(BlockDemo("card", "card — surface container", "Layout",
        DisplaySpec(name="demo_card", theme="boxbot", root_block=_wrap([_heading("card"), card]))))

    spacer_row = B.RowBlock(gap=0, align="center")
    spacer_row.children = [
        B.BadgeBlock(text="left", color="accent"),
        B.SpacerBlock(size=120),
        B.BadgeBlock(text="gap of 120", color="secondary"),
        B.SpacerBlock(),
        B.BadgeBlock(text="right", color="success"),
    ]
    demos.append(BlockDemo("spacer", "spacer — fixed or flexible space", "Layout",
        DisplaySpec(name="demo_spacer", theme="boxbot", root_block=_wrap([_heading("spacer"), spacer_row]))))

    div_col = B.ColumnBlock(gap=12)
    div_col.children = [
        B.TextBlock(content="Above divider", size="body"),
        B.DividerBlock(),
        B.TextBlock(content="Below divider", size="body"),
        B.SpacerBlock(size=12),
        B.DividerBlock(color="accent", thickness=2),
        B.TextBlock(content="After accent divider", size="body", color="muted"),
    ]
    demos.append(BlockDemo("divider", "divider — horizontal separator", "Layout",
        DisplaySpec(name="demo_div", theme="boxbot", root_block=_wrap([_heading("divider"), div_col]))))

    # repeat — iterates a list template
    repeat = B.RepeatBlock(source="{tasks.items}", max=3)
    template = B.RowBlock(gap=12, align="center")
    template.children = [
        B.IconBlock(name="check-circle", size="sm", color="accent"),
        B.TextBlock(content="{.description}", size="body"),
        B.SpacerBlock(),
        B.TextBlock(content="{.due_date}", size="caption", color="muted"),
    ]
    repeat.children = [template]
    demos.append(BlockDemo("repeat", "repeat — data-driven list", "Layout",
        DisplaySpec(name="demo_repeat", theme="boxbot",
            data_sources=[DataSourceSpec(name="tasks", source_type="builtin")],
            root_block=_wrap([_heading("repeat {.field}"), repeat]))))

    # --- Content blocks ----------------------------------------------------
    text_col = B.ColumnBlock(gap=10, align="start")
    text_col.children = [
        B.TextBlock(content="Title size", size="title"),
        B.TextBlock(content="Heading size", size="heading"),
        B.TextBlock(content="Subtitle size", size="subtitle", color="muted"),
        B.TextBlock(content="Body size — default for most content", size="body"),
        B.TextBlock(content="Caption size — fine print and metadata", size="caption", color="muted"),
        B.TextBlock(content="small size", size="small", color="dim"),
    ]
    demos.append(BlockDemo("text", "text — size and color variants", "Content",
        DisplaySpec(name="demo_text", theme="boxbot", root_block=_wrap([_heading("text"), text_col]))))

    metric_row = B.RowBlock(gap=40, align="center")
    metric_row.children = [
        B.MetricBlock(value="72°F", label="Partly Cloudy", icon="cloud-sun"),
        B.MetricBlock(value="$432.10", label="AAPL", change="+2.4%", change_color="success"),
        B.MetricBlock(value="87%", label="Battery"),
    ]
    demos.append(BlockDemo("metric", "metric — large number with context", "Content",
        DisplaySpec(name="demo_metric", theme="boxbot", root_block=_wrap([_heading("metric"), metric_row]))))

    badges = B.RowBlock(gap=10, align="center")
    badges.children = [
        B.BadgeBlock(text="Live", color="accent"),
        B.BadgeBlock(text="Active", color="success"),
        B.BadgeBlock(text="3 new", color="secondary"),
        B.BadgeBlock(text="Warn", color="warning"),
        B.BadgeBlock(text="Error", color="error"),
        B.BadgeBlock(text="Muted", color="muted"),
    ]
    demos.append(BlockDemo("badge", "badge — colored pill", "Content",
        DisplaySpec(name="demo_badge", theme="boxbot", root_block=_wrap([_heading("badge"), badges]))))

    list_block = B.ListBlock(items=["Read the display docs", "Write a demo spec", "Render it", "Review", "Iterate"], style="check")
    list_block2 = B.ListBlock(items=["one", "two", "three"], style="number")
    list_block3 = B.ListBlock(items=["alpha", "beta", "gamma"], style="bullet")
    list_row = B.RowBlock(gap=40, align="start")
    for lb in (list_block, list_block2, list_block3):
        holder = B.ColumnBlock(gap=8)
        holder.children = [B.TextBlock(content=f"style={lb.params.get('style','bullet')}", size="caption", color="muted"), lb]
        list_row.children.append(holder)
    demos.append(BlockDemo("list", "list — bullet/number/check styles", "Content",
        DisplaySpec(name="demo_list", theme="boxbot", root_block=_wrap([_heading("list"), list_row]))))

    table = B.TableBlock(
        headers=["Ticker", "Price", "Change"],
        rows=[["AAPL", "$432.10", "+2.4%"], ["MSFT", "$410.22", "-0.8%"], ["ANTH", "$—", "pvt"]],
        striped=True,
    )
    demos.append(BlockDemo("table", "table — headers + rows", "Content",
        DisplaySpec(name="demo_table", theme="boxbot", root_block=_wrap([_heading("table (striped)"), table]))))

    kv = B.KeyValueBlock(data={"Status": "listening", "Last heard": "2 min ago", "Next wake": "7:00 AM", "Uptime": "3d 14h"})
    demos.append(BlockDemo("key_value", "key_value — two-column pairs", "Content",
        DisplaySpec(name="demo_kv", theme="boxbot", root_block=_wrap([_heading("key_value"), kv]))))

    icon_row = B.RowBlock(gap=20, align="center")
    for n in ("home", "sun", "cloud", "cloud-rain", "check-circle", "alert-triangle", "bell", "zap"):
        holder = B.ColumnBlock(gap=6, align="center")
        holder.children = [B.IconBlock(name=n, size="lg"), B.TextBlock(content=n, size="small", color="muted")]
        icon_row.children.append(holder)
    demos.append(BlockDemo("icon", "icon — Lucide glyphs", "Content",
        DisplaySpec(name="demo_icon", theme="boxbot", root_block=_wrap([_heading("icon"), icon_row]))))

    emoji_row = B.RowBlock(gap=18, align="center")
    for n in ("sunny", "smile", "heart", "fire", "pizza", "rocket"):
        holder = B.ColumnBlock(gap=6, align="center")
        holder.children = [B.EmojiBlock(name=n, size="xl"), B.TextBlock(content=n, size="small", color="muted")]
        emoji_row.children.append(holder)
    demos.append(BlockDemo("emoji", "emoji — twemoji glyphs", "Content",
        DisplaySpec(name="demo_emoji", theme="boxbot", root_block=_wrap([_heading("emoji"), emoji_row]))))

    image_note = B.TextBlock(content="(image requires url/asset/photo — shown empty in gallery)", size="caption", color="muted")
    demos.append(BlockDemo("image", "image — url/asset/photo", "Content",
        DisplaySpec(name="demo_image", theme="boxbot", root_block=_wrap([_heading("image"), image_note]))))

    chart = B.ChartBlock(
        data=[12, 18, 15, 22, 26, 21, 27, 30, 28, 32, 34, 31],
        chart_type="area",
        height=260,
        x_labels=["J","F","M","A","M","J","J","A","S","O","N","D"],
        show_dots=True,
    )
    chart2 = B.ChartBlock(
        chart_type="bar",
        series=[{"label": "A", "data": [3, 6, 2, 8, 5]}, {"label": "B", "data": [4, 2, 7, 3, 6]}],
        x_labels=["Mon","Tue","Wed","Thu","Fri"],
        show_legend=True,
        height=200,
    )
    chart_col = B.ColumnBlock(gap=16)
    chart_col.children = [chart, chart2]
    demos.append(BlockDemo("chart", "chart — area + bar", "Content",
        DisplaySpec(name="demo_chart", theme="boxbot", root_block=_wrap([_heading("chart"), chart_col]))))

    prog_col = B.ColumnBlock(gap=14)
    prog_col.children = [
        B.ProgressBlock(value=0.25, label="25%"),
        B.ProgressBlock(value=0.6, label="60%"),
        B.ProgressBlock(value=0.9, label="90%", color="success"),
        B.ProgressBlock(value=1.0, label="Full", color="accent"),
    ]
    demos.append(BlockDemo("progress", "progress — capacity bar", "Content",
        DisplaySpec(name="demo_progress", theme="boxbot", root_block=_wrap([_heading("progress"), prog_col]))))

    clock_row = B.RowBlock(gap=40, align="center")
    clock_row.children = [
        B.ClockBlock(size="md", show_date=True, show_seconds=False),
        B.ClockBlock(size="lg", format="24h", show_date=False),
        B.ClockBlock(size="xl", show_seconds=True),
    ]
    demos.append(BlockDemo("clock", "clock — live, multiple sizes", "Content",
        DisplaySpec(name="demo_clock", theme="boxbot", root_block=_wrap([_heading("clock"), clock_row]))))

    # Set countdown target to a plausible future time (use static string)
    countdown = B.CountdownBlock(target="2026-12-31T23:59:59", label="Until New Year")
    demos.append(BlockDemo("countdown", "countdown — live ticker", "Content",
        DisplaySpec(name="demo_countdown", theme="boxbot", root_block=_wrap([_heading("countdown"), countdown]))))

    # --- Composite widgets -------------------------------------------------
    weather = B.WeatherWidget(data_source="weather")
    demos.append(BlockDemo("weather_widget", "weather_widget — prebuilt", "Widget",
        DisplaySpec(name="demo_weather_widget", theme="boxbot",
            data_sources=[DataSourceSpec(name="weather", source_type="builtin")],
            root_block=_wrap([_heading("weather_widget"), weather]))))

    calendar = B.CalendarWidget(data_source="calendar")
    demos.append(BlockDemo("calendar_widget", "calendar_widget — agenda", "Widget",
        DisplaySpec(name="demo_calendar_widget", theme="boxbot",
            data_sources=[DataSourceSpec(name="calendar", source_type="builtin")],
            root_block=_wrap([_heading("calendar_widget"), calendar]))))

    # --- Meta blocks -------------------------------------------------------
    rotate = B.RotateBlock(source="weather", key="forecast", interval=5)
    rot_template = B.CardBlock(padding=24)
    rot_col = B.ColumnBlock(gap=8, align="center")
    rot_col.children = [
        B.TextBlock(content="{current.day}", size="heading"),
        B.IconBlock(name="{current.icon}", size="xl"),
        B.TextBlock(content="H {current.high}° / L {current.low}°", size="body", color="muted"),
    ]
    rot_template.children = [rot_col]
    rotate.children = [rot_template]
    demos.append(BlockDemo("rotate", "rotate — timed carousel", "Meta",
        DisplaySpec(name="demo_rotate", theme="boxbot",
            data_sources=[DataSourceSpec(name="weather", source_type="builtin")],
            root_block=_wrap([_heading("rotate (first frame shown)"), rotate]))))

    dots = B.PageDotsBlock()
    dots_row = B.RowBlock(gap=24, align="center")
    dots_row.children = [dots]
    demos.append(BlockDemo("page_dots", "page_dots — carousel indicator", "Meta",
        DisplaySpec(name="demo_dots", theme="boxbot",
            data_sources=[DataSourceSpec(name="weather", source_type="builtin")],
            root_block=_wrap([_heading("page_dots"), dots_row]))))

    return demos


def _card_with(child: B.Block) -> B.Block:
    c = B.CardBlock(padding=20)
    c.children = [child]
    return c


def _render_demo(demo: BlockDemo, theme_name: str, renderer: DisplayRenderer) -> Image.Image:
    theme = get_theme(theme_name)
    data: dict = {}
    for src in demo.spec.data_sources:
        ph = get_placeholder_data(src.name)
        if ph:
            data[src.name] = ph
    return renderer.render(demo.spec, theme=theme, data=data)


def _compose_contact_sheet(theme_name: str, images: list[tuple[BlockDemo, Image.Image]]) -> Image.Image:
    theme = get_theme(theme_name)
    bg = hex_to_rgb(theme.colors.background)
    text_color = hex_to_rgb(theme.colors.text)
    muted = hex_to_rgb(theme.colors.muted)
    accent = hex_to_rgb(theme.colors.accent)

    cols = 4
    cell_w, cell_h = 512, 300
    label_h = 44
    margin = 28
    header_h = 80
    rows = (len(images) + cols - 1) // cols
    W = margin * 2 + cols * cell_w + (cols - 1) * margin
    H = margin + header_h + rows * (cell_h + label_h + margin) + margin

    sheet = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(sheet)

    title_font = _get_font("Inter", 32, 700)
    sub_font = _get_font("Inter", 18, 400)
    label_font = _get_font("Inter", 18, 600)
    caption_font = _get_font("Inter", 14, 400)

    draw.text((margin, margin), f"boxBot block gallery — theme: {theme_name}", fill=text_color, font=title_font)
    draw.text((margin, margin + 38), f"{len(images)} blocks · 1024x600 → 512x300 thumbnails", fill=muted, font=sub_font)

    for idx, (demo, full_img) in enumerate(images):
        r, c = divmod(idx, cols)
        x = margin + c * (cell_w + margin)
        y = margin + header_h + r * (cell_h + label_h + margin)

        thumb = full_img.resize((cell_w, cell_h), Image.LANCZOS)
        sheet.paste(thumb, (x, y))
        # border in accent color
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=accent, width=1)
        # label
        draw.text((x, y + cell_h + 6), demo.label, fill=text_color, font=label_font)
        draw.text((x, y + cell_h + 26), demo.category, fill=muted, font=caption_font)

    return sheet


def main() -> int:
    demos = _make_demos()
    print(f"prepared {len(demos)} block demos")

    renderer = DisplayRenderer(width=1024, height=600)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for theme_name in list_themes():
        theme_dir = OUT_DIR / "per-block" / theme_name
        theme_dir.mkdir(parents=True, exist_ok=True)

        rendered: list[tuple[BlockDemo, Image.Image]] = []
        for demo in demos:
            img = _render_demo(demo, theme_name, renderer)
            img.save(theme_dir / f"{demo.block_id}.png")
            rendered.append((demo, img))

        sheet = _compose_contact_sheet(theme_name, rendered)
        sheet_path = OUT_DIR / f"contact_sheet_{theme_name}.png"
        sheet.save(sheet_path)
        print(f"theme {theme_name}: per-block → {theme_dir} · sheet → {sheet_path} ({sheet.size[0]}x{sheet.size[1]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
