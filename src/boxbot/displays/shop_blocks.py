"""Shopping screens: numbered product cards and the cart.

Two renderer-only blocks behind the ``products`` and ``cart`` built-in
displays. They are deliberately **not** in ``BLOCK_REGISTRY``: an
agent-authored display cannot use them, and what they draw never comes
from display args. ``args.view`` is only a token; the content is a
payload the main process published with :func:`publish` (from the
purchase result cache and cart — see ``boxbot.purchases.shopping``).
So a product card's title, price, store and picture are always the
catalogue's, never typed by the model.

Everything is drawn as text or a re-encoded bitmap. Merchant strings
are sanitised before they get here and are never interpreted.

Sizes are for reading from across a room on 1024x600: no text under
28 px, the card number in an 80 px+ filled circle, price 56 px+.

The variant label ("Hardcover · 2019", "ISBN ending 6945") is never
cut to an ellipsis while there is room to wrap it: it is the only thing
that tells two editions of one book apart. Cards that share a title
with another card on the screen give the title one line and the label
up to five; the picture gives up the height.
"""

from __future__ import annotations

import secrets
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from boxbot.displays.blocks import Block
from boxbot.displays.renderer import (
    _BLOCK_RENDERERS,
    Rect,
    RenderContext,
    _fit_image,
    _get_font,
    _measure_text,
    _wrap_text,
)

MIN_TEXT_PX = 28
NUMBER_CIRCLE_PX = 84
PRICE_PX = 56
TITLE_PX = 30
MAX_CARDS = 4
MAX_CART_LINES = 5
_TITLE_LINE = 38
_BADGE_LINE = 34            # line pitch inside the variant label
_BADGE_PAD = 5              # above and below its text
_BADGE_LINES = 3            # most lines for a card whose title is unique
_BADGE_LINES_SAME_TITLE = 5

_MAX_VIEWS = 8
_views: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


def publish(payload: dict[str, Any]) -> str:
    """Hold ``payload`` for rendering; return the token for ``args.view``."""
    token = secrets.token_hex(6)
    _views[token] = payload
    while len(_views) > _MAX_VIEWS:
        _views.popitem(last=False)
    return token


def payload_for(token: Any) -> dict[str, Any] | None:
    return _views.get(token) if isinstance(token, str) else None


@dataclass
class ProductCardsBlock(Block):
    """1–4 numbered product cards: one large card, or equal columns."""

    block_type: str = field(default="product_cards", init=False)
    view: str = ""

    def __post_init__(self) -> None:
        self.params = {"view": self.view}


@dataclass
class CartBlock(Block):
    """Cart lines on the left, cost breakdown and total on the right."""

    block_type: str = field(default="cart_summary", init=False)
    view: str = ""

    def __post_init__(self) -> None:
        self.params = {"view": self.view}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _font(ctx: RenderContext, size: int, weight: int = 400):
    return _get_font(ctx.theme.fonts.family, max(MIN_TEXT_PX, size), weight)


def _clip(text: str, font: Any, max_w: int) -> str:
    """``text`` on one line, cut with an ellipsis to fit ``max_w``."""
    text = " ".join(str(text or "").split())
    if _measure_text(text, font)[0] <= max_w:
        return text
    while text and _measure_text(text + "...", font)[0] > max_w:
        text = text[:-1]
    return (text.rstrip() + "...") if text else ""


def _text(ctx: RenderContext, xy: tuple[int, int], text: str, font: Any,
          color: str = "text") -> None:
    ctx.draw.text(xy, text, fill=ctx.theme.color_rgb(color), font=font)


def _text_right(ctx: RenderContext, right: int, y: int, text: str, font: Any,
                color: str = "text") -> None:
    _text(ctx, (right - _measure_text(text, font)[0], y), text, font, color)


def _fit_font(ctx: RenderContext, text: str, size: int, floor: int,
              max_w: int, weight: int = 700):
    """Largest font from ``size`` down to ``floor`` that fits ``max_w``."""
    while size > floor:
        font = _font(ctx, size, weight)
        if _measure_text(text, font)[0] <= max_w:
            return font
        size -= 4
    return _font(ctx, floor, weight)


def _number_circle(ctx: RenderContext, x: int, y: int, number: Any,
                   size: int = NUMBER_CIRCLE_PX) -> None:
    ctx.draw.ellipse([x, y, x + size, y + size], fill=ctx.theme.color_rgb("accent"))
    label = str(number)[:2]
    font = _font(ctx, int(size * 0.6), 700)
    box = font.getbbox(label)
    ctx.draw.text(
        (x + (size - (box[2] - box[0])) // 2 - box[0],
         y + (size - (box[3] - box[1])) // 2 - box[1]),
        label, fill=ctx.theme.color_rgb("background"), font=font,
    )


def _picture(ctx: RenderContext, rect: Rect, path: Any, radius: int) -> None:
    """The product photo, or a quiet placeholder while there is none."""
    from boxbot.purchases.images import open_cached

    # Cache files only, JPEG only, pixel count capped before decoding.
    image = open_cached(path)
    box = [rect.x, rect.y, rect.x + rect.w, rect.y + rect.h]
    if image is None:
        ctx.draw.rounded_rectangle(
            box, radius=radius, fill=ctx.theme.color_rgb("surface_alt"),
        )
        return
    ctx.draw.rounded_rectangle(box, radius=radius, fill=(255, 255, 255))
    pad = 8
    fitted = _fit_image(image, max(1, rect.w - 2 * pad), max(1, rect.h - 2 * pad),
                        "contain")
    ctx.image.paste(
        fitted,
        (rect.x + (rect.w - fitted.width) // 2,
         rect.y + (rect.h - fitted.height) // 2),
    )


def _badge_lines(ctx: RenderContext, text: str, max_w: int,
                 max_lines: int) -> list[str]:
    """``text`` laid out for the variant label, ≤ ``max_lines`` lines.

    One line when it fits. Otherwise it breaks at the " · " between
    parts first ("Hardcover" / "2019" — no dangling separator), and
    inside a part only when the part alone is too wide. Only a label
    too long for every allowed line ends in an ellipsis.
    """
    text = " ".join(str(text or "").split())
    if not text:
        return []
    font = _font(ctx, MIN_TEXT_PX, 600)
    inner = max(1, max_w - 28)

    def fits(candidate: str) -> bool:
        return _measure_text(candidate, font)[0] <= inner

    lines: list[str] = []
    for part in [p.strip() for p in text.split(" · ") if p.strip()]:
        if lines and fits(f"{lines[-1]} · {part}"):
            lines[-1] = f"{lines[-1]} · {part}"
        elif fits(part):
            lines.append(part)
        else:
            lines.extend(_wrap_text(part, font, inner))
    max_lines = max(1, max_lines)
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + [" ".join(lines[max_lines - 1:])]
    return [_clip(line, font, inner) for line in lines]


def _badge_height(lines: list[str]) -> int:
    return len(lines) * _BADGE_LINE + 2 * _BADGE_PAD if lines else 0


def _badge(ctx: RenderContext, x: int, y: int, lines: list[str]) -> int:
    """The variant label, one or more lines on a rounded panel.

    Returns its height (0 when there is nothing to show).
    """
    if not lines:
        return 0
    font = _font(ctx, MIN_TEXT_PX, 600)
    width = max(_measure_text(line, font)[0] for line in lines) + 28
    height = _badge_height(lines)
    ctx.draw.rounded_rectangle(
        [x, y, x + width, y + height],
        radius=min(height // 2, (_BADGE_LINE + 2 * _BADGE_PAD) // 2),
        fill=ctx.theme.color_rgb("surface_alt"),
    )
    # Centre each line on its pitch by the font's own metrics, so
    # descenders do not push one line off the others.
    ascent, descent = font.getmetrics()
    top = y + _BADGE_PAD + (_BADGE_LINE - (ascent + descent)) // 2
    for index, line in enumerate(lines):
        ctx.draw.text(
            (x + 14, top + index * _BADGE_LINE), line,
            fill=ctx.theme.color_rgb("text"), font=font,
        )
    return height


def _store_line(ctx: RenderContext, x: int, y: int, item: dict[str, Any],
                max_w: int) -> None:
    """Store name; an unapproved store gets a filled "!" marker."""
    font = _font(ctx, MIN_TEXT_PX, 500)
    name = str(item.get("store") or "")
    if item.get("approved") is False:
        mark = 30
        warn = ctx.theme.color_rgb("warning")
        ctx.draw.ellipse([x, y + 2, x + mark, y + 2 + mark], fill=warn)
        bang = _font(ctx, MIN_TEXT_PX, 700)
        box = bang.getbbox("!")
        ctx.draw.text(
            (x + (mark - (box[2] - box[0])) // 2 - box[0],
             y + 2 + (mark - (box[3] - box[1])) // 2 - box[1]),
            "!", fill=ctx.theme.color_rgb("background"), font=bang,
        )
        _text(ctx, (x + mark + 8, y), _clip(name, font, max_w - mark - 8), font,
              "warning")
    else:
        _text(ctx, (x, y), _clip(name, font, max_w), font, "muted")


def _footer(ctx: RenderContext, rect: Rect, hint: str) -> None:
    font = _font(ctx, MIN_TEXT_PX, 500)
    hint = _clip(hint, font, rect.w - 48)
    width = _measure_text(hint, font)[0]
    _text(ctx, (rect.x + (rect.w - width) // 2, rect.y + rect.h - 46), hint,
          font, "muted")


def _empty(ctx: RenderContext, rect: Rect, text: str) -> None:
    font = _font(ctx, 34, 600)
    width = _measure_text(text, font)[0]
    _text(ctx, (rect.x + (rect.w - width) // 2, rect.y + rect.h // 2 - 20), text,
          font, "muted")


def _items(payload: dict[str, Any] | None, key: str, cap: int) -> list[dict[str, Any]]:
    raw = (payload or {}).get(key)
    if not isinstance(raw, list):
        return []
    return [i for i in raw if isinstance(i, dict)][:cap]


# ---------------------------------------------------------------------------
# Product cards
# ---------------------------------------------------------------------------


def _render_product_cards(ctx: RenderContext, block: Block, rect: Rect) -> None:
    payload = payload_for(block.params.get("view"))
    items = _items(payload, "items", MAX_CARDS)
    if not items:
        _empty(ctx, rect, "Nothing to show")
        return
    hint = str((payload or {}).get("hint") or "Say a number")
    if any(i.get("approved") is False for i in items):
        hint += "   ·   ! = store needs an admin's OK"
    _footer(ctx, rect, hint)
    body = Rect(rect.x + 24, rect.y + 16, rect.w - 48, rect.h - 16 - 60)
    if len(items) == 1:
        _single_card(ctx, body, items[0])
        return
    gap = 16
    # One row of equal columns, always — never a grid.
    width = (body.w - gap * (len(items) - 1)) // len(items)
    plans = _plan_columns(ctx, items, width - 2 * _CARD_PAD)
    # One text-block height for the row, so prices and stores line up.
    head_h = max(plan["head_h"] for plan in plans)
    for index, (item, plan) in enumerate(zip(items, plans)):
        _column_card(
            ctx, Rect(body.x + index * (width + gap), body.y, width, body.h),
            item, plan, head_h,
        )


_CARD_PAD = 12


def _plan_columns(ctx: RenderContext, items: list[dict[str, Any]],
                  inner_w: int) -> list[dict[str, Any]]:
    """Title and variant lines per card, before anything is drawn.

    A card that shares its title with another card on the screen is
    told apart by its variant alone: one title line (a second would
    repeat its neighbour's), and the variant in full on up to
    :data:`_BADGE_LINES_SAME_TITLE` lines. Other cards keep two title
    lines and wrap the variant to :data:`_BADGE_LINES`.
    """
    def key(item: dict[str, Any]) -> str:
        return " ".join(str(item.get("title") or "").split()).casefold()

    counts: dict[str, int] = {}
    for item in items:
        counts[key(item)] = counts.get(key(item), 0) + 1
    title_font = _font(ctx, TITLE_PX, 700)
    plans = []
    for item in items:
        shared = counts[key(item)] > 1 and bool(item.get("variant"))
        title = [
            _clip(line, title_font, inner_w) for line in _wrap_text(
                str(item.get("title") or ""), title_font, inner_w,
                1 if shared else 2,
            )
        ]
        badge = _badge_lines(
            ctx, str(item.get("variant") or ""), inner_w,
            _BADGE_LINES_SAME_TITLE if shared else _BADGE_LINES,
        )
        plans.append({
            "title": title, "badge": badge,
            "head_h": len(title) * _TITLE_LINE + 8 + _badge_height(badge),
        })
    return plans


def _single_card(ctx: RenderContext, rect: Rect, item: dict[str, Any]) -> None:
    radius = ctx.theme.radius
    picture = Rect(rect.x, rect.y, int(rect.w * 0.42), rect.h)
    _picture(ctx, picture, item.get("image"), radius)
    x = picture.x + picture.w + 36
    width = rect.x + rect.w - x
    y = rect.y + 8
    _number_circle(ctx, x, y, item.get("number", 1), NUMBER_CIRCLE_PX + 8)
    _store_line(
        ctx, x + NUMBER_CIRCLE_PX + 28, y + 28, item,
        width - NUMBER_CIRCLE_PX - 28,
    )
    y += NUMBER_CIRCLE_PX + 8 + 24
    title_font = _font(ctx, 36, 700)
    for line in _wrap_text(str(item.get("title") or ""), title_font, width, 3):
        _text(ctx, (x, y), line, title_font)
        y += 46
    y += 10
    used = _badge(
        ctx, x, y,
        _badge_lines(ctx, str(item.get("variant") or ""), width, _BADGE_LINES),
    )
    y += used + (18 if used else 0)
    price = str(item.get("price") or "")
    _text(ctx, (x, y), price, _fit_font(ctx, price, 72, PRICE_PX, width), "accent")


def _column_card(ctx: RenderContext, rect: Rect, item: dict[str, Any],
                 plan: dict[str, Any], head_h: int) -> None:
    radius = ctx.theme.radius
    ctx.draw.rounded_rectangle(
        [rect.x, rect.y, rect.x + rect.w, rect.y + rect.h],
        radius=radius, fill=ctx.theme.color_rgb("surface"),
    )
    pad = _CARD_PAD
    inner_w = rect.w - 2 * pad
    x = rect.x + pad
    # Text block below the picture: title + variant (``head_h``, the
    # tallest in the row), then price and store on shared baselines.
    # The picture takes what is left.
    text_h = head_h + 8 + 66 + 6 + 36
    picture = Rect(x, rect.y + pad, inner_w, rect.h - 2 * pad - text_h - 10)
    _picture(ctx, picture, item.get("image"), radius)
    _number_circle(ctx, x + 8, picture.y + 8, item.get("number", ""))
    top = picture.y + picture.h + 10
    y = top
    title_font = _font(ctx, TITLE_PX, 700)
    for line in plan["title"]:
        _text(ctx, (x, y), line, title_font)
        y += _TITLE_LINE
    _badge(ctx, x, y + 8, plan["badge"])
    y = top + head_h + 8
    price = str(item.get("price") or "")
    _text(ctx, (x, y), price, _fit_font(ctx, price, PRICE_PX, 40, inner_w), "accent")
    y += 66 + 6
    _store_line(ctx, x, y, item, inner_w)


# ---------------------------------------------------------------------------
# Cart
# ---------------------------------------------------------------------------


def _render_cart(ctx: RenderContext, block: Block, rect: Rect) -> None:
    payload = payload_for(block.params.get("view")) or {}
    lines = _items(payload, "lines", MAX_CART_LINES)
    if not lines:
        _empty(ctx, rect, "Cart is empty")
        return
    _footer(ctx, rect, str(payload.get("hint") or ""))
    top = rect.y + 16
    height = rect.h - 16 - 60
    split = rect.x + int(rect.w * 0.60)

    # Left: the lines.
    x = rect.x + 28
    width = split - x - 20
    y = top + 4
    _text(ctx, (x, y), "Cart", _font(ctx, 34, 700))
    y += 58
    title_font = _font(ctx, TITLE_PX, 600)
    small = _font(ctx, MIN_TEXT_PX, 400)
    for line in lines:
        price = str(line.get("price") or "")
        price_w = _measure_text(price, title_font)[0]
        qty = line.get("qty")
        label = (f"{qty} × " if isinstance(qty, int) and qty != 1 else "") + str(
            line.get("title") or ""
        )
        _text(ctx, (x, y), _clip(label, title_font, width - price_w - 16), title_font)
        _text_right(ctx, x + width, y, price, title_font)
        sub = " · ".join(
            str(part) for part in (line.get("variant"), line.get("store")) if part
        )
        _store_line(
            ctx, x, y + 38,
            {"store": sub, "approved": line.get("approved")}, width,
        )
        y += 84
    more = payload.get("more")
    if isinstance(more, int) and more > 0:
        _text(ctx, (x, y), f"+ {more} more", small, "muted")

    # Right: the figures, total largest.
    panel = Rect(split, top, rect.x + rect.w - split - 24, height)
    ctx.draw.rounded_rectangle(
        [panel.x, panel.y, panel.x + panel.w, panel.y + panel.h],
        radius=ctx.theme.radius, fill=ctx.theme.color_rgb("surface"),
    )
    px = panel.x + 24
    right = panel.x + panel.w - 24
    y = panel.y + 24
    figure_font = _font(ctx, TITLE_PX, 500)
    for row in _figures(payload):
        _text(ctx, (px, y), row[0], figure_font, "muted")
        _text_right(ctx, right, y, row[1], figure_font)
        y += 48
    y += 8
    ctx.draw.line([(px, y), (right, y)], fill=ctx.theme.color_rgb("dim"), width=2)
    y += 20
    _text(ctx, (px, y), str(payload.get("total_label") or "Total"), figure_font, "muted")
    y += 44
    total = str(payload.get("total") or "")
    total_font = _fit_font(ctx, total, 76, PRICE_PX, right - px)
    _text_right(ctx, right, y, total, total_font, "accent")
    note = str(payload.get("note") or "")
    if note:
        ny = panel.y + panel.h - 24 - 2 * 36
        for line in _wrap_text(note, small, right - px, 2):
            _text(ctx, (px, ny), line, small, "warning")
            ny += 36


def _figures(payload: dict[str, Any]) -> list[tuple[str, str]]:
    rows = []
    for row in payload.get("figures") or []:
        if isinstance(row, (list, tuple)) and len(row) == 2:
            rows.append((str(row[0])[:20], str(row[1])[:20]))
    return rows[:4]


_BLOCK_RENDERERS.setdefault("product_cards", _render_product_cards)
_BLOCK_RENDERERS.setdefault("cart_summary", _render_cart)
