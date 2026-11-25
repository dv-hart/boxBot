"""find → cart → send: the agent-facing shopping flow, by short id.

The conversation agent never handles a variant id, a store domain, a
price or a product title as something it types. It gets slim rows with
ids (``a1`` … ``a8``, then ``b1`` …) and passes ids back. Everything
behind an id lives here, in the main process:

- **Result cache.** The full catalogue row (variant id, store domain,
  price, image URL, raw title) per conversation, looked up only through
  the conversation the tool call is running in — an id from one
  conversation means nothing in another. Idle sessions expire after
  :data:`SESSION_TTL` seconds; sessions and rows are capped; a session
  is dropped when its conversation ends.
- **Cart.** id → quantity for the conversation. Never persisted. Adding
  to it or pricing it creates no purchase and texts nobody.
- **Quote.** ``cart(quote=True)`` builds a store checkout to read real
  shipping and tax — only at approved stores, because a checkout hands
  the store the household's contact and delivery details. Cached per
  (store, lines) for :data:`QUOTE_TTL` seconds and adopted by ``send``.
- **Send.** One purchase per store through :func:`service.propose`
  (rules, checker, review link or veto window — all unchanged). A
  changed cart sent again supersedes the earlier purchase: one text,
  marked as replacing it.

Screens (``products`` / ``cart`` displays) are drawn from the same
cached rows, so what is on the screen is what is in the cart.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from boxbot.purchases import images, service
from boxbot.purchases.service import PurchaseError
from boxbot.purchases.shop_client import (
    ShopError,
    ShopNotConnected,
    plain_text,
    summarize_checkout,
)
from boxbot.purchases.store import get_purchase_store

if TYPE_CHECKING:
    from boxbot.core.conversation import Conversation

logger = logging.getLogger(__name__)

SESSION_TTL = 1800          # idle seconds before a conversation's ids/cart expire
QUOTE_TTL = 600
MAX_RESULTS = 8
# Unapproved stores are listed only when fewer than this many offers
# from approved stores match.
MIN_APPROVED = 3
MAX_SHOW = 4
_MAX_SESSIONS = 24
_MAX_ROWS = 160             # per conversation; oldest searches dropped first
_MAX_IDS = 10               # ids per list argument
_MAX_LINES = 10
_MAX_QUANTITY = 10
_CATALOGUE_FETCH = 20       # offers asked of the catalogue per find
_RESEND_SECONDS = 600
_IMAGE_BUDGET = 12.0
_ID_RE = re.compile(r"^[a-z]{1,2}[1-8]$")
_LIVE = ("review", "scheduled", "verifying")

_now = time.monotonic       # indirection so tests can move the clock


@dataclass
class _Quote:
    key: str
    checkout: dict[str, Any]     # raw; never leaves the main process
    facts: dict[str, Any]
    at: float


@dataclass
class _Sent:
    purchase_id: str
    key: str
    at: float


@dataclass
class _Session:
    touched: float
    searches: int = 0
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    cart: dict[str, int] = field(default_factory=dict)
    quotes: dict[str, _Quote] = field(default_factory=dict)      # store host →
    sent: dict[str, _Sent] = field(default_factory=dict)         # store host →
    batches: list[list[str]] = field(default_factory=list)       # purchase ids per send


_sessions: "OrderedDict[str, _Session]" = OrderedDict()
_store_names: dict[str, str] = {}            # store host → seller name last seen
_quote_rate = service._RateWindow(12, 3600)
_send_rate = service._RateWindow(12, 3600)
_tasks: set[asyncio.Task] = set()
_subscribed = False


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def _conversation_key(conv: "Conversation | None") -> str:
    """The session key — taken from the running conversation, never a payload."""
    if conv is None:
        raise PurchaseError("shopping needs a conversation; none is running")
    key = getattr(conv, "conversation_id", None) or getattr(conv, "channel_key", None)
    if not isinstance(key, str) or not key:
        raise PurchaseError("shopping needs a conversation; none is running")
    return key


def _expire() -> None:
    cutoff = _now() - SESSION_TTL
    for key in [k for k, s in _sessions.items() if s.touched < cutoff]:
        _sessions.pop(key, None)


def _session(conv: "Conversation | None", *, create: bool = True) -> _Session:
    key = _conversation_key(conv)
    _expire()
    session = _sessions.get(key)
    if session is None:
        if not create:
            raise PurchaseError(
                "nothing found in this conversation yet (or it expired); "
                "find() first"
            )
        session = _sessions[key] = _Session(touched=_now())
        while len(_sessions) > _MAX_SESSIONS:
            _sessions.popitem(last=False)
    session.touched = _now()
    _sessions.move_to_end(key)
    return session


def forget(conversation_key: str) -> None:
    """Drop a conversation's results, cart and quotes."""
    _sessions.pop(conversation_key, None)


def reset() -> None:
    """Drop every session and counter (tests)."""
    _sessions.clear()
    _store_names.clear()
    _quote_rate._events.clear()
    _send_rate._events.clear()


def start() -> None:
    """Drop a session when its conversation ends. Call once at startup."""
    global _subscribed
    if _subscribed:
        return
    try:
        from boxbot.core.events import ConversationEnded, get_event_bus

        async def _ended(event: Any) -> None:
            forget(getattr(event, "conversation_id", "") or "")

        get_event_bus().subscribe(ConversationEnded, _ended)
        _subscribed = True
    except Exception:  # noqa: BLE001 — the TTL still bounds every session
        logger.exception("shopping: could not subscribe to conversation end")


def _on_screen(conv: "Conversation | None") -> bool:
    """Whether this conversation is one the box's screen belongs to."""
    return getattr(conv, "channel", None) not in service._TEXT_CHANNELS


# ---------------------------------------------------------------------------
# Argument validation — everything below arrives from the sandbox
# ---------------------------------------------------------------------------


def _flag(value: Any, name: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise PurchaseError(f"{name} must be true or false")
    return value


def _id(value: Any) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise PurchaseError(
            f"{str(value)[:20]!r} is not a result id; ids look like 'a2' "
            "and come from find()"
        )
    return value


def _id_list(value: Any, name: str, *, least: int = 0, most: int = _MAX_IDS) -> list[str]:
    if value is None:
        value = []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not least <= len(value) <= most:
        raise PurchaseError(f"{name} takes {max(least, 1)}–{most} result ids")
    return [_id(v) for v in value]


def _row(session: _Session, result_id: str) -> dict[str, Any]:
    row = session.rows.get(result_id)
    if row is None:
        raise PurchaseError(
            f"no result '{result_id}' in this conversation (ids expire "
            f"after {SESSION_TTL // 60} idle minutes); find() again"
        )
    return row


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _letter(index: int) -> str:
    """a … z, then aa … zz, wrapping."""
    index %= 26 + 26 * 26
    if index < 26:
        return chr(97 + index)
    index -= 26
    return chr(97 + index // 26) + chr(97 + index % 26)


def _handle_tag(handle: Any, title: str) -> str | None:
    """A short tag from a product handle: an ISBN tail, or the handle."""
    if not isinstance(handle, str) or not handle:
        return None
    digits = handle.replace("-", "")
    if digits.isdigit() and len(digits) in (10, 13):
        return f"ISBN ending {digits[-4:]}"
    words = re.sub(r"[-_]+", " ", handle).strip()
    if not words or words == re.sub(r"[^a-z0-9]+", " ", title.lower()).strip():
        return None
    return plain_text(words, 24)


def _prepare(raw: dict[str, Any]) -> dict[str, Any]:
    """The cached form of one catalogue offer (main process only)."""
    options = [o for o in raw.get("options") or [] if isinstance(o, str)]
    product_title = raw.get("product_title") or ""
    title = product_title if (product_title and options) else (raw.get("title") or "")
    host = raw["shop"]
    seller = plain_text(raw.get("seller"), 40)
    if seller:
        _store_names[host] = seller
        if len(_store_names) > 500:
            _store_names.clear()
    return {
        "title": plain_text(title, 60) or "(untitled)",
        "raw_title": raw.get("title"),
        "variant": plain_text(" · ".join(options[:2]), 40) or None,
        "handle": raw.get("handle"),
        "variant_id": raw["variant_id"],
        "shop": host,
        "store": seller or host,
        "price_cents": raw["price_cents"],
        "currency": raw.get("currency"),
        "image_url": raw.get("image_url"),
        "url": raw.get("url"),
    }


def _label_same_titles(rows: list[dict[str, Any]]) -> None:
    """Make ``variant`` tell same-titled rows apart (edition, format, ISBN).

    Computed from the catalogue payload only. Rows whose title is
    unique keep whatever options they carry (or None).
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["title"].casefold(), []).append(row)
    for group in groups.values():
        if len(group) < 2:
            continue
        labels = [r["variant"] or "" for r in group]
        if len(set(labels)) < len(labels):
            for row in group:
                tag = _handle_tag(row.get("handle"), row["title"])
                if tag and tag not in (row["variant"] or ""):
                    row["variant"] = plain_text(
                        " · ".join(p for p in (row["variant"], tag) if p), 40,
                    )
        seen: set[tuple[str, str]] = set()
        for row in group:
            mark = (row["variant"] or "", row["shop"])
            # Same label at the same store: nothing in the payload tells
            # them apart, so fall back to the variant id's tail.
            if mark in seen:
                tail = row["variant_id"][-4:]
                row["variant"] = plain_text(
                    " · ".join(p for p in (row["variant"], f"#{tail}") if p), 40,
                )
            seen.add((row["variant"] or "", row["shop"]))


def _slim(result_id: str, row: dict[str, Any], approved: set[str]) -> dict[str, Any]:
    """The row as the agent sees it. Exactly these keys."""
    return {
        "id": result_id,
        "title": row["title"],
        "variant": row["variant"],
        "price": service._money(row["price_cents"]),
        "store": row["store"],
        "approved": row["shop"] in approved,
        "in_stock": True,
    }


def _usable(raw: list[dict[str, Any]], *, feed: bool = False) -> list[dict[str, Any]]:
    """Offers that can be bought here: in stock, priced, in US dollars."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for offer in raw:
        if not isinstance(offer, dict) or offer.get("available") is not True:
            continue
        cents = offer.get("price_cents")
        if isinstance(cents, bool) or not isinstance(cents, int) or cents <= 0:
            continue
        # A store's own feed does not state a currency (the checkout
        # does); a catalogue offer must say USD.
        if offer.get("currency") != "USD" and not (feed and offer.get("currency") is None):
            continue
        variant = offer.get("variant_id")
        if not isinstance(variant, str) or variant in seen or not offer.get("shop"):
            continue
        seen.add(variant)
        out.append(offer)
    return out


def _remember(session: _Session, rows: list[dict[str, Any]]) -> list[str]:
    """Give ``rows`` the next search letter; return their ids."""
    letter = _letter(session.searches)
    session.searches += 1
    # Letters wrap after several hundred searches; never reuse one that
    # still has an item in the cart.
    for _ in range(40):
        if not any(k[:-1] == letter for k in session.cart):
            break
        letter = _letter(session.searches)
        session.searches += 1
    for stale in [k for k in session.rows if k[:-1] == letter and k not in session.cart]:
        session.rows.pop(stale, None)
    ids = []
    for number, row in enumerate(rows, start=1):
        result_id = f"{letter}{number}"
        session.rows[result_id] = row
        ids.append(result_id)
    for old in list(session.rows):
        if len(session.rows) <= _MAX_ROWS:
            break
        if old not in session.cart and old not in ids:
            session.rows.pop(old, None)
    return ids


# ---------------------------------------------------------------------------
# find / store_products / stores / show
# ---------------------------------------------------------------------------


async def find(
    query: Any, *, max_price_cents: Any = None, approved_only: Any = False,
    show: Any = True, conv: "Conversation | None",
) -> dict[str, Any]:
    """Search the catalogue; return ≤8 slim rows, approved stores first."""
    service._require_enabled()
    approved_only = _flag(approved_only, "approved_only")
    show = True if show is None else _flag(show, "show")
    session = _session(conv)
    raw = await service.search(
        query, max_price_cents=max_price_cents, limit=_CATALOGUE_FETCH,
    )
    offers = _usable(raw)
    if max_price_cents is not None:
        offers = [o for o in offers if o["price_cents"] <= max_price_cents]
    approved = set(await service.approved_shops())
    chosen = [o for o in offers if o["shop"] in approved][:MAX_RESULTS]
    if not approved_only and len(chosen) < MIN_APPROVED:
        others = [o for o in offers if o["shop"] not in approved]
        chosen += others[: MAX_RESULTS - len(chosen)]
    return await _deliver(session, chosen, approved, show and _on_screen(conv))


async def store_products(
    shop: Any, *, query: Any = None, show: Any = True,
    conv: "Conversation | None",
) -> dict[str, Any]:
    """An approved store's own product feed, as the same slim rows."""
    service._require_enabled()
    show = True if show is None else _flag(show, "show")
    session = _session(conv)
    raw = await service.store_products(shop, query=query, limit=MAX_RESULTS * 2)
    approved = set(await service.approved_shops())
    chosen = _usable(raw, feed=True)[:MAX_RESULTS]
    return await _deliver(session, chosen, approved, show and _on_screen(conv))


async def _deliver(
    session: _Session, chosen: list[dict[str, Any]], approved: set[str],
    show: bool,
) -> dict[str, Any]:
    if not chosen:
        return {"results": [], "shown": []}
    rows = [_prepare(o) for o in chosen]
    _label_same_titles(rows)
    ids = _remember(session, rows)
    shown: list[str] = []
    if show:
        shown = ids[:MAX_SHOW]
        if not await _draw_products(session, shown, approved):
            shown = []
    return {
        "results": [_slim(i, session.rows[i], approved) for i in ids],
        "shown": shown,
    }


async def stores() -> list[dict[str, str]]:
    """Approved stores: ``[{"store", "domain"}]``."""
    service._require_enabled()
    return [
        {"store": _store_names.get(host, host), "domain": host}
        for host in await service.approved_shops()
    ]


def _numbers(ids: list[str]) -> list[int]:
    """Card numbers: the id's own digit when those are distinct, else 1..n."""
    digits = [int(i[-1]) for i in ids]
    return digits if len(set(digits)) == len(digits) else list(range(1, len(ids) + 1))


async def show(ids: Any, *, conv: "Conversation | None") -> dict[str, Any]:
    """Put 1–4 cached results on the screen as numbered cards."""
    service._require_enabled()
    wanted = _id_list(ids, "show", least=1, most=MAX_SHOW)
    if len(set(wanted)) != len(wanted):
        raise PurchaseError("show takes each id once")
    session = _session(conv, create=False)
    for result_id in wanted:
        _row(session, result_id)
    approved = set(await service.approved_shops())
    drawn = await _draw_products(session, wanted, approved)
    return {
        "shown": wanted if drawn else [],
        "numbers": {str(n): i for n, i in zip(_numbers(wanted), wanted)},
    }


def _manager() -> Any:
    try:
        from boxbot.displays.manager import get_display_manager

        return get_display_manager()
    except Exception:  # noqa: BLE001
        return None


def _card(number: int, row: dict[str, Any], approved: set[str]) -> dict[str, Any]:
    cached = images.cached_path(row.get("image_url"))
    return {
        "number": number,
        "title": row["title"],
        "variant": row["variant"],
        "price": service._money(row["price_cents"]),
        "store": row["store"],
        "approved": row["shop"] in approved,
        "image": str(cached) if cached is not None else None,
    }


async def _switch(name: str, payload: dict[str, Any]) -> str | None:
    """Publish ``payload`` and switch to display ``name``; the view token."""
    manager = _manager()
    if manager is None:
        return None
    try:
        from boxbot.displays.shop_blocks import publish

        token = publish(payload)
        if await manager.switch(name, args={"view": token}):
            return token
    except Exception:  # noqa: BLE001 — the screen must never fail a shopping call
        logger.exception("shopping: could not draw the %s screen", name)
    return None


async def _draw_products(
    session: _Session, ids: list[str], approved: set[str],
) -> bool:
    rows = [session.rows[i] for i in ids]
    numbers = _numbers(ids)
    hint = "Say a number" if len(ids) > 1 else 'Say "add it", or ask for another'

    def payload() -> dict[str, Any]:
        return {
            "items": [_card(n, r, approved) for n, r in zip(numbers, rows)],
            "hint": hint,
        }

    first = payload()
    token = await _switch("products", first)
    if token is None:
        return False
    missing = [
        r["image_url"] for r, card in zip(rows, first["items"])
        if card["image"] is None and images.allowed_url(r.get("image_url"))
    ]
    if missing:
        _spawn(_fill_images(token, missing, payload))
    return True


async def _fill_images(token: str, urls: list[str], payload: Any) -> None:
    """Fetch the missing photos, then redraw if the cards are still up."""
    try:
        done = await asyncio.wait_for(
            asyncio.gather(*(images.fetch(u) for u in urls)), timeout=_IMAGE_BUDGET,
        )
    except asyncio.TimeoutError:
        done = [images.cached_path(u) for u in urls]
    if not any(done):
        return
    manager = _manager()
    if manager is None or manager.get_active() != "products":
        return
    if (manager.get_active_args() or {}).get("view") != token:
        return
    await _switch("products", payload())


def _spawn(coro: Any) -> None:
    task = asyncio.get_running_loop().create_task(coro, name="shopping-images")
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


# ---------------------------------------------------------------------------
# Cart and quote
# ---------------------------------------------------------------------------


def _lines_key(host: str, lines: list[dict[str, Any]]) -> str:
    blob = host + "|" + ",".join(
        f"{l['variant_id']}x{l['quantity']}"
        for l in sorted(lines, key=lambda l: l["variant_id"])
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _by_store(session: _Session) -> "OrderedDict[str, list[dict[str, Any]]]":
    """Cart lines per store host, in cart order, merged per variant."""
    stores_: "OrderedDict[str, dict[str, int]]" = OrderedDict()
    for result_id, quantity in session.cart.items():
        row = session.rows[result_id]
        per = stores_.setdefault(row["shop"], {})
        per[row["variant_id"]] = per.get(row["variant_id"], 0) + quantity
    return OrderedDict(
        (host, [{"variant_id": v, "quantity": q} for v, q in per.items()])
        for host, per in stores_.items()
    )


def _valid_quote(session: _Session, host: str, key: str) -> _Quote | None:
    quote = session.quotes.get(host)
    if quote is None or quote.key != key or _now() - quote.at > QUOTE_TTL:
        return None
    return quote


def _apply_cart_changes(
    session: _Session, add: Any, remove: Any, replace: Any, clear: bool,
) -> tuple[dict[str, int], list[str]]:
    """The cart after the requested changes, and what changed. No commit."""
    cart = dict(session.cart)
    changed: list[str] = []
    if clear and cart:
        cart = {}
        changed.append("cleared")
    for result_id in _id_list(remove, "remove"):
        if cart.pop(result_id, None) is not None:
            changed.append(f"removed {result_id}")
        else:
            changed.append(f"{result_id} was not in the cart")
    if replace is None:
        replace = {}
    if not isinstance(replace, dict) or len(replace) > _MAX_IDS:
        raise PurchaseError("replace is {old_id: new_id}")
    for old, new in replace.items():
        old, new = _id(old), _id(new)
        _row(session, new)
        if old not in cart:
            raise PurchaseError(f"{old} is not in the cart; cart(add=['{new}']) instead")
        if new in cart and new != old:
            cart.pop(old)
        else:
            # Same position, same quantity.
            cart = {(new if k == old else k): q for k, q in cart.items()}
        changed.append(f"replaced {old} with {new}")
    additions: dict[str, int] = {}
    if isinstance(add, dict):
        if len(add) > _MAX_IDS:
            raise PurchaseError(f"add takes at most {_MAX_IDS} ids")
        for result_id, quantity in add.items():
            if (
                isinstance(quantity, bool) or not isinstance(quantity, int)
                or not 1 <= quantity <= _MAX_QUANTITY
            ):
                raise PurchaseError(f"quantity must be 1–{_MAX_QUANTITY}")
            additions[_id(result_id)] = quantity
    else:
        for result_id in _id_list(add, "add"):
            additions[result_id] = additions.get(result_id, 0) + 1
    for result_id, quantity in additions.items():
        _row(session, result_id)
        total = quantity if isinstance(add, dict) else cart.get(result_id, 0) + quantity
        if total > _MAX_QUANTITY:
            raise PurchaseError(f"quantity must be 1–{_MAX_QUANTITY}")
        changed.append(
            f"added {result_id}" if result_id not in cart
            else f"{result_id} quantity {total}"
        )
        cart[result_id] = total
    if len(cart) > _MAX_LINES:
        raise PurchaseError(f"at most {_MAX_LINES} different items in the cart")
    return cart, changed


async def cart(
    *, add: Any = None, remove: Any = None, replace: Any = None,
    clear: Any = False, quote: Any = False, conv: "Conversation | None",
) -> dict[str, Any]:
    """Change and/or price the conversation's cart; redraw the cart screen.

    Creates no purchase and texts nobody. ``quote`` builds a checkout at
    each approved store to read shipping and tax.
    """
    cfg = service._require_enabled()
    clear = _flag(clear, "clear")
    quote = _flag(quote, "quote")
    session = _session(conv)
    session.cart, changed = _apply_cart_changes(session, add, remove, replace, clear)
    approved = set(await service.approved_shops())
    errors: dict[str, str] = {}
    if quote and session.cart:
        _quote_rate.limit = cfg.max_quotes_per_hour
        for host, lines in _by_store(session).items():
            if host not in approved:
                continue        # never hand an unapproved store a checkout
            try:
                await _quote(session, host, lines, changed)
            except PurchaseError as e:
                errors[host] = str(e)
    view = _cart_view(session, approved, changed, errors)
    if _on_screen(conv):
        await _draw_cart(session, view, approved)
    return view


async def _quote(
    session: _Session, host: str, lines: list[dict[str, Any]], changed: list[str],
) -> _Quote:
    key = _lines_key(host, lines)
    held = _valid_quote(session, host, key)
    if held is not None:
        return held
    if not _quote_rate.take():
        raise PurchaseError("too many quotes this hour; stop and tell the person")
    try:
        checkout = await service.get_shop_client().create_checkout(host, lines)
    except ShopNotConnected as e:
        await service._note_not_connected()
        raise service._shop_failure(e) from e
    except (ShopError, httpx.HTTPError) as e:
        raise service._shop_failure(e) from e
    service.log_checkout_costs("quote", host, checkout)
    facts = summarize_checkout(checkout)
    if not isinstance(checkout.get("id"), str) or not facts["lines"]:
        raise PurchaseError("the store returned an empty checkout")
    if facts["currency"] != "USD" or not isinstance(facts["total_cents"], int):
        raise PurchaseError("the store did not give a total in US dollars")
    # The checkout's unit price is the real one; keep the cache honest.
    confirmed = {
        l["variant_id"]: l["unit_cents"] for l in facts["lines"]
        if isinstance(l.get("unit_cents"), int)
    }
    for result_id in session.cart:
        row = session.rows[result_id]
        now_cents = confirmed.get(row["variant_id"])
        if row["shop"] == host and now_cents and now_cents != row["price_cents"]:
            row["price_cents"] = now_cents
            changed.append(f"{result_id} is {service._money(now_cents)} at checkout")
    quote = session.quotes[host] = _Quote(key, checkout, facts, _now())
    return quote


def _money(cents: Any) -> str | None:
    return service._money(cents) if isinstance(cents, int) else None


def _cart_view(
    session: _Session, approved: set[str], changed: list[str],
    errors: dict[str, str],
) -> dict[str, Any]:
    lines = []
    for result_id, quantity in session.cart.items():
        row = session.rows[result_id]
        lines.append({
            "id": result_id, "title": row["title"], "variant": row["variant"],
            "qty": quantity, "price": service._money(row["price_cents"]),
        })
    stores_ = []
    subtotal_all = 0
    total_all: int | None = 0
    for host, store_lines in _by_store(session).items():
        priced = {
            session.rows[i]["variant_id"]: session.rows[i]["price_cents"]
            for i in session.cart if session.rows[i]["shop"] == host
        }
        subtotal = sum(priced[l["variant_id"]] * l["quantity"] for l in store_lines)
        entry: dict[str, Any] = {
            "store": _store_names.get(host, host),
            "approved": host in approved,
            "subtotal": service._money(subtotal),
            "shipping": None, "tax": None, "total": None, "delivery": None,
        }
        quote = _valid_quote(session, host, _lines_key(host, store_lines))
        if host not in approved:
            entry["needs_approval"] = True
            entry["domain"] = host
            total_all = None
        elif quote is not None:
            facts = quote.facts
            if isinstance(facts["subtotal_cents"], int):
                subtotal = facts["subtotal_cents"]
                entry["subtotal"] = service._money(subtotal)
            entry["shipping"] = _money(facts["shipping_cents"])
            entry["tax"] = _money(facts["tax_cents"])
            entry["total"] = service._money(facts["total_cents"])
            delivery = facts.get("delivery") or {}
            if delivery.get("name"):
                entry["delivery"] = " · ".join(
                    p for p in (delivery["name"], _money(delivery.get("cents"))) if p
                )
            if total_all is not None:
                total_all += facts["total_cents"]
        else:
            total_all = None
        if host in errors:
            entry["quote_error"] = errors[host]
        subtotal_all += subtotal
        stores_.append(entry)
    return {
        "lines": lines,
        "stores": stores_,
        "subtotal": service._money(subtotal_all) if lines else None,
        "total": service._money(total_all) if lines and total_all is not None else None,
        "changed": changed,
    }


def _sum_money(session: _Session, approved: set[str], key: str) -> str:
    """Sum of one quoted figure over the cart's stores, or a dash."""
    total = 0
    for host, lines in _by_store(session).items():
        quote = _valid_quote(session, host, _lines_key(host, lines))
        if host not in approved or quote is None:
            return "—"
        cents = quote.facts.get(key)
        if isinstance(cents, int):
            total += cents
    return service._money(total)


async def _draw_cart(
    session: _Session, view: dict[str, Any], approved: set[str],
) -> None:
    manager = _manager()
    if manager is None:
        return
    if not session.cart:
        # Nothing to show: hand the screen back rather than pin an
        # empty cart on it.
        try:
            if manager.get_active() in ("cart", "products"):
                await manager.unpin()
        except Exception:  # noqa: BLE001
            logger.exception("shopping: could not release the screen")
        return
    from boxbot.displays.shop_blocks import MAX_CART_LINES

    lines = []
    for result_id, quantity in session.cart.items():
        row = session.rows[result_id]
        lines.append({
            "qty": quantity, "title": row["title"], "variant": row["variant"],
            "store": row["store"], "approved": row["shop"] in approved,
            "price": service._money(row["price_cents"] * quantity),
        })
    waiting = [s["store"] for s in view["stores"] if s.get("needs_approval")]
    quoted = view["total"] is not None
    await _switch("cart", {
        "lines": lines[:MAX_CART_LINES],
        "more": max(0, len(lines) - MAX_CART_LINES),
        "figures": [
            ["Subtotal", view["subtotal"]],
            ["Shipping", _sum_money(session, approved, "shipping_cents")],
            ["Tax", _sum_money(session, approved, "tax_cents")],
        ],
        "total_label": "Total" if quoted else "Before shipping",
        "total": view["total"] if quoted else view["subtotal"],
        "note": (
            f"{waiting[0]} needs an admin's OK" if waiting else ""
        ),
        "hint": (
            'Say "send it" to text it for payment, or "cancel"' if quoted
            else 'Ask "what\'s the total?" for shipping and tax'
        ),
    })


# ---------------------------------------------------------------------------
# send / cancel
# ---------------------------------------------------------------------------


def scrub_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    """A propose outcome as the sandbox may see it.

    The checker's verdict stays in the decision log. Handing it back —
    or the list of rules that passed, from which it can be read off —
    would give a compromised agent a per-try signal to tune proposals
    against. Where automatic purchasing is on, a proposal that goes to
    review says only that.
    """
    outcome = dict(outcome)
    outcome.pop("checker", None)
    if service._config().mode == "auto" and outcome.get("decision") == "review":
        outcome["blocked_by"] = ["needs_review"]
    return outcome


def note_sent(conv: "Conversation | None", outcome: dict[str, Any]) -> None:
    """Remember a purchase proposed directly, so cancel() can default to it."""
    try:
        session = _session(conv)
    except PurchaseError:
        return
    purchase_id = (outcome.get("purchase") or {}).get("id")
    if isinstance(purchase_id, str):
        session.batches.append([purchase_id])
        del session.batches[:-10]


async def _admin_names() -> list[str]:
    try:
        from boxbot.communication.auth import get_auth_manager

        auth = get_auth_manager()
        if auth is None:
            return []
        return [
            plain_text(u.name, 40) for u in await auth.list_users()
            if u.role == "admin"
        ][:5]
    except Exception:  # noqa: BLE001
        return []


async def send(reason: Any, *, conv: "Conversation | None") -> dict[str, Any]:
    """Propose the cart: one purchase per approved store.

    Each store goes through :func:`service.propose` — same rules, same
    checker, same review link to the admins. Returns ``{"purchases":
    [<propose outcome + "store">], "needs_approval": [...], "total",
    "admins"}``.
    """
    service._require_enabled()
    if not isinstance(reason, str) or not reason.strip():
        raise PurchaseError("a reason for the purchase is required")
    if len(reason) > 2000:
        raise PurchaseError("reason is one sentence (at most 2000 characters)")
    session = _session(conv, create=False)
    if not session.cart:
        raise PurchaseError("the cart is empty; cart(add=[...]) first")
    if not _send_rate.take():
        raise PurchaseError("too many purchases sent this hour; stop and tell the person")
    approved = set(await service.approved_shops())
    store = get_purchase_store()
    purchases: list[dict[str, Any]] = []
    waiting: list[dict[str, Any]] = []
    batch: list[str] = []
    failures: list[str] = []
    for host, lines in _by_store(session).items():
        name = _store_names.get(host, host)
        if host not in approved:
            asked = await service.request_shop_approval(host, conv)
            waiting.append({
                "store": name, "domain": host,
                "note": service.shop_ask_reply(asked),
            })
            continue
        key = _lines_key(host, lines)
        replaces, replaced_row = await _supersede(session, store, host, key)
        quote = _valid_quote(session, host, key)
        try:
            outcome = await service.propose(
                host, lines, reason, conv=conv,
                adopt=quote.checkout if quote is not None else None,
                replaces=replaces,
            )
        except PurchaseError as e:
            # The earlier purchase is already cancelled and its link is
            # still out there: say so on its own.
            if replaced_row is not None and replaces:
                await service.send_cancel_notice(replaced_row)
            failures.append(str(e))
            purchases.append({"store": name, "error": str(e)})
            continue
        if quote is not None and not outcome.get("already_proposed"):
            session.quotes.pop(host, None)      # one purchase per checkout
        purchase_id = outcome["purchase"]["id"]
        if not outcome.get("already_proposed"):
            session.sent[host] = _Sent(purchase_id, key, _now())
        batch.append(purchase_id)
        purchases.append({"store": name, **scrub_outcome(outcome)})
    if batch:
        session.batches.append(batch)
        del session.batches[:-10]
    if failures and not batch and not waiting:
        raise PurchaseError(failures[0])
    cents = [
        p["purchase"]["total_cents"] for p in purchases
        if "purchase" in p and isinstance(p["purchase"].get("total_cents"), int)
    ]
    told = any(p.get("admins_notified") for p in purchases)
    return {
        "purchases": purchases,
        "needs_approval": waiting,
        "total": service._money(sum(cents)) if cents else None,
        "admins": await _admin_names() if told else [],
    }


async def _supersede(
    session: _Session, store: Any, host: str, key: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """Cancel this conversation's earlier live purchase at ``host``, if the
    cart changed (or the earlier one is past the re-send window).

    Returns ``(id to name as replaced, the cancelled row)``. The id is
    None when there was nothing to replace or the admins were never
    sent the earlier one.
    """
    earlier = session.sent.get(host)
    if earlier is None:
        return None, None
    if earlier.key == key and _now() - earlier.at < _RESEND_SECONDS:
        return None, None       # identical and recent: propose() returns it
    before = await store.get(earlier.purchase_id)
    if before is None or before["status"] not in _LIVE:
        return None, None
    try:
        row, _ = await service.cancel_by_agent(earlier.purchase_id, notify=False)
    except PurchaseError:
        return None, None       # too late to stop; it stands on its own
    known = bool(before.get("link_sent_at")) or before["status"] != "review"
    return (earlier.purchase_id if known else None), row


async def cancel(
    purchase_id: Any = None, *, conv: "Conversation | None",
) -> dict[str, Any]:
    """Cancel one purchase, or (no id) this conversation's latest send.

    Anyone may ask: stopping a purchase needs no authority. Returns
    ``{"cancelled": [<purchase>…], "admins_told": int}``.
    """
    if purchase_id is not None:
        if not isinstance(purchase_id, str) or not re.fullmatch(
            r"[0-9a-fA-F]{6,12}", purchase_id.strip()
        ):
            raise PurchaseError("'id' is a purchase id like 'ab12cd34'")
        row, told = await service.cancel_by_agent(purchase_id.strip().lower())
        return {"cancelled": [service.public_view(row)], "admins_told": told}
    try:
        session = _session(conv, create=False)
    except PurchaseError:
        session = None
    store = get_purchase_store()
    for batch in reversed(session.batches if session is not None else []):
        live = []
        for candidate in batch:
            row = await store.get(candidate)
            if row is not None and row["status"] in _LIVE:
                live.append(candidate)
        if not live:
            continue
        cancelled, told = [], 0
        for candidate in live:
            try:
                row, sent = await service.cancel_by_agent(candidate)
            except PurchaseError:
                continue
            cancelled.append(service.public_view(row))
            told += sent
        if cancelled:
            return {"cancelled": cancelled, "admins_told": told}
    raise PurchaseError(
        "nothing sent from this conversation is waiting to be cancelled. "
        "To cancel another purchase pass its id (list() shows them); to "
        "empty the cart, cart(clear=True)"
    )
