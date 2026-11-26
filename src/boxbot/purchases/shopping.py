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
  (rules, checker, review link or veto window — all unchanged). After
  a send, the only live purchases this conversation's cart has out are
  the ones standing for the cart as it is now: an earlier one whose
  lines changed, or whose store left the cart, is cancelled.
- **Cancel is local, so it is always announced.** The store's checkout
  link stays payable after a cancel here. Every purchase cancelled by
  ``send`` or ``cancel`` whose link reached an admin gets exactly one
  text — a "Replaces purchase …" line in the new purchase's text, or
  the standalone cancel notice — on every exit path (:class:`_Retired`).

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
_MAX_TRACKED = 40           # purchase ids remembered per conversation
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
    # store host → the purchase standing for the cart's lines there
    sent: dict[str, _Sent] = field(default_factory=dict)
    # every purchase id this conversation put out (send or propose), oldest first
    purchases: list[str] = field(default_factory=list)


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
    if not session.cart:
        # An empty cart is not a cancelled purchase. Say so, rather
        # than let "cleared" pass for "nothing is out there".
        out = await _still_out(session)
        if out:
            view["still_out"] = out
            view["note"] = (
                "The cart is empty, but "
                + ", ".join(f"purchase {p['id']} ({p['store']}, {p['total']})" for p in out)
                + (" is" if len(out) == 1 else " are")
                + " still out for payment. Emptying the cart cancels "
                "nothing; cancel() pulls back everything this conversation sent."
            )
    if _on_screen(conv):
        await _draw_cart(session, view, approved)
    return view


async def _still_out(session: _Session) -> list[dict[str, Any]]:
    """Live purchases this conversation has sent: ``[{"id", "store",
    "total", "status"}]``."""
    store = get_purchase_store()
    out = []
    for purchase_id in session.purchases:
        row = await store.get(purchase_id)
        if row is not None and row["status"] in _LIVE:
            out.append({
                "id": purchase_id,
                "store": _store_names.get(row["shop"], row["shop"]),
                "total": service._money(row["total_cents"], row["currency"]),
                "status": row["status"],
            })
    return out


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
    outcome.pop("replaces_announced", None)     # bookkeeping for send()
    if service._config().mode == "auto" and outcome.get("decision") == "review":
        outcome["blocked_by"] = ["needs_review"]
    return outcome


def _track(session: _Session, purchase_id: str) -> None:
    if purchase_id in session.purchases:
        session.purchases.remove(purchase_id)
    session.purchases.append(purchase_id)
    del session.purchases[:-_MAX_TRACKED]


def note_sent(conv: "Conversation | None", outcome: dict[str, Any]) -> None:
    """Remember a purchase proposed directly, so cancel() covers it."""
    try:
        session = _session(conv)
    except PurchaseError:
        return
    purchase_id = (outcome.get("purchase") or {}).get("id")
    if isinstance(purchase_id, str):
        _track(session, purchase_id)


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


class _Retired:
    """Purchases one call cancelled, and the notices still owed for them.

    Cancelling is local — the link already texted stays payable — so the
    rule is: **a cancelled purchase whose link (or veto notice) reached
    an admin gets exactly one text saying so.** Either the new
    purchase's own text names it ("Replaces purchase …"), or the
    standalone cancel notice does.

    To make that hold on every exit path, a cancel here never sends
    anything: it records the notice as owed (``NOTICE_PENDING``). It is
    cleared only by :meth:`announced`, called when a text naming the id
    was actually accepted for an admin. Whatever is still owed when the
    caller's ``finally`` runs :meth:`flush` goes out as the standalone
    notice — after a refused replacement, an ``already_proposed``
    answer, an unexpected exception, or the caller being cancelled.
    The hourly notice ceiling can still stop a text; then the entry
    reads ``not_told`` and the agent is told to say so.
    """

    def __init__(self) -> None:
        self._entries: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

    async def cancel(self, purchase_id: str, *, strict: bool = False) -> bool:
        """Cancel ``purchase_id`` if it is still live. True when it was.

        ``strict`` re-raises the service's refusal (unknown id, too
        late) instead of treating it as "nothing to do".
        """
        if purchase_id in self._entries:
            return True
        try:
            row, notice = await service.cancel_by_agent(purchase_id, notify=False)
        except PurchaseError:
            if strict:
                raise
            return False        # already settled, or too late: it stands
        self._entries[purchase_id] = {"row": row, "notice": notice}
        return True

    def owed(self) -> list[tuple[str, str]]:
        """``[(id, store host)]`` still waiting to be named to the admins."""
        return [
            (pid, e["row"]["shop"]) for pid, e in self._entries.items()
            if e["notice"] == service.NOTICE_PENDING
        ]

    def announced(self, ids: list[str]) -> None:
        """A text naming ``ids`` reached an admin: nothing more is owed."""
        for pid in ids:
            entry = self._entries.get(pid)
            if entry is not None and entry["notice"] == service.NOTICE_PENDING:
                entry["notice"] = service.NOTICE_TOLD

    async def flush(self) -> None:
        """Send the standalone notice for everything still owed.

        Runs in a main-process task the caller's cancellation cannot
        interrupt, like the proposal itself.
        """
        if self.owed():
            await asyncio.shield(
                service._spawn(self._flush(), "purchase-cancel-notices")
            )

    async def _flush(self) -> None:
        for entry in self._entries.values():
            if entry["notice"] != service.NOTICE_PENDING:
                continue
            # Settle the entry before the send can be interrupted twice.
            entry["notice"] = service.NOTICE_NOT_TOLD
            try:
                if await service.send_cancel_notice(entry["row"]):
                    entry["notice"] = service.NOTICE_TOLD
            except Exception:  # noqa: BLE001 — one failure must not stop the rest
                logger.exception(
                    "purchase %s: cancel notice failed", entry["row"]["id"],
                )

    def views(self) -> list[dict[str, Any]]:
        """Each cancelled purchase as the agent sees it, with
        ``admin_notice``: ``told`` | ``none_needed`` | ``not_told``."""
        return [
            {**service.public_view(e["row"]), "admin_notice": e["notice"]}
            for e in self._entries.values()
        ]

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "id": pid,
                "store": _store_names.get(e["row"]["shop"], e["row"]["shop"]),
                "admin_notice": e["notice"],
            }
            for pid, e in self._entries.items()
        ]


async def send(reason: Any, *, conv: "Conversation | None") -> dict[str, Any]:
    """Propose the cart: one purchase per approved store.

    Each store goes through :func:`service.propose` — same rules, same
    checker, same review link to the admins. Earlier purchases from
    this conversation's cart that the cart no longer stands for are
    cancelled, and the admins told (:class:`_Retired`).

    Returns ``{"purchases": [<propose outcome + "store">],
    "needs_approval": [...], "replaced": [{"id", "store",
    "admin_notice"}], "total", "admins"}``.
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
    retired = _Retired()
    purchases: list[dict[str, Any]] = []
    waiting: list[dict[str, Any]] = []
    try:
        await _send_cart(session, reason, conv, retired, purchases, waiting)
    finally:
        # Every exit: whatever was cancelled and not yet named to the
        # admins gets its own notice now.
        await retired.flush()
    replaced = retired.summary()
    failures = [p["error"] for p in purchases if "error" in p]
    if failures and len(failures) == len(purchases) and not waiting:
        raise PurchaseError(failures[0] + _replaced_note(replaced))
    cents = [
        p["purchase"]["total_cents"] for p in purchases
        if "purchase" in p and isinstance(p["purchase"].get("total_cents"), int)
    ]
    told = any(p.get("admins_notified") for p in purchases)
    return {
        "purchases": purchases,
        "needs_approval": waiting,
        "replaced": replaced,
        "total": service._money(sum(cents)) if cents else None,
        "admins": await _admin_names() if told else [],
    }


def _replaced_note(replaced: list[dict[str, Any]]) -> str:
    """For an error message: what a failed send cancelled anyway."""
    if not replaced:
        return ""
    parts = []
    for entry in replaced:
        if entry["admin_notice"] == service.NOTICE_NOT_TOLD:
            parts.append(
                f"{entry['id']} (admins could NOT be told; its earlier "
                "link still works — tell them)"
            )
        else:
            parts.append(entry["id"])
    return ". Earlier purchase cancelled: " + ", ".join(parts)


async def _send_cart(
    session: _Session, reason: str, conv: "Conversation | None",
    retired: _Retired, purchases: list[dict[str, Any]],
    waiting: list[dict[str, Any]],
) -> None:
    approved = set(await service.approved_shops())
    by_store = _by_store(session)

    # 1. Earlier purchases the cart no longer stands for: the store has
    #    left the cart, or (at a store that cannot be sent to right now)
    #    the lines changed. Cancelled first, so the first new text of
    #    this send can name them.
    for host, earlier in list(session.sent.items()):
        lines = by_store.get(host)
        if lines and host in approved:
            continue            # decided below, next to its replacement
        if lines and earlier.key == _lines_key(host, lines):
            continue            # same lines; it stands until the store is approved again
        await retired.cancel(earlier.purchase_id)
        session.sent.pop(host, None)

    # 2. One purchase per approved store.
    for host, lines in by_store.items():
        name = _store_names.get(host, host)
        if host not in approved:
            asked = await service.request_shop_approval(host, conv)
            waiting.append({
                "store": name, "domain": host,
                "note": service.shop_ask_reply(asked),
            })
            continue
        key = _lines_key(host, lines)
        earlier = session.sent.get(host)
        recent_same = (
            earlier is not None and earlier.key == key
            and _now() - earlier.at < _RESEND_SECONDS
        )
        if earlier is not None and not recent_same:
            # Changed, or past the re-send window: a new purchase
            # replaces it. (Identical and recent: propose() returns it.)
            await retired.cancel(earlier.purchase_id)
            session.sent.pop(host, None)
        quote = _valid_quote(session, host, key)
        owed = retired.owed()
        try:
            outcome = await service.propose(
                host, lines, reason, conv=conv,
                adopt=quote.checkout if quote is not None else None,
                replaces=owed or None,
            )
        except PurchaseError as e:
            # Anything cancelled above stays owed; a later store's text
            # or the flush in send() carries it.
            purchases.append({"store": name, "error": str(e)})
            continue
        if outcome.get("replaces_announced"):
            retired.announced([pid for pid, _ in owed])
        if quote is not None and not outcome.get("already_proposed"):
            session.quotes.pop(host, None)      # one purchase per checkout
        purchase_id = outcome["purchase"]["id"]
        same = earlier is not None and earlier.purchase_id == purchase_id
        if earlier is not None and not same:
            # propose() answered with a different purchase than the one
            # on our books for this store (an earlier identical proposal
            # on the same channel, say). Ours must not stay payable.
            await retired.cancel(earlier.purchase_id)
        session.sent[host] = _Sent(purchase_id, key, earlier.at if same else _now())
        _track(session, purchase_id)
        purchases.append({"store": name, **scrub_outcome(outcome)})


async def cancel(
    purchase_id: Any = None, *, conv: "Conversation | None",
) -> dict[str, Any]:
    """Cancel one purchase, or (no id) everything live this conversation sent.

    Anyone may ask: stopping a purchase needs no authority. No id means
    *every* live purchase sent from this conversation — after "cancel",
    nothing this conversation put out stays payable un-flagged.

    Returns ``{"cancelled": [<purchase + "admin_notice">…]}`` where
    ``admin_notice`` is ``"told"`` (an admin was texted to ignore the
    link), ``"none_needed"`` (no link ever reached anyone) or
    ``"not_told"`` (a link is out and the admins could not be told).
    """
    retired = _Retired()
    if purchase_id is not None:
        if not isinstance(purchase_id, str) or not re.fullmatch(
            r"[0-9a-fA-F]{6,12}", purchase_id.strip()
        ):
            raise PurchaseError("'id' is a purchase id like 'ab12cd34'")
        try:
            await retired.cancel(purchase_id.strip().lower(), strict=True)
        finally:
            await retired.flush()
        return {"cancelled": retired.views()}
    try:
        session = _session(conv, create=False)
    except PurchaseError:
        session = None
    try:
        for candidate in reversed(session.purchases if session is not None else []):
            await retired.cancel(candidate)
    finally:
        await retired.flush()
    cancelled = retired.views()
    if not cancelled:
        raise PurchaseError(
            "nothing sent from this conversation is waiting to be cancelled. "
            "To cancel another purchase pass its id (list() shows them); to "
            "empty the cart, cart(clear=True)"
        )
    if session is not None:
        gone = {c["id"] for c in cancelled}
        for host in [h for h, sent in session.sent.items() if sent.purchase_id in gone]:
            session.sent.pop(host, None)
    return {"cancelled": cancelled}
