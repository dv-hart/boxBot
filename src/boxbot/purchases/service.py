"""Purchase orchestration: propose, decide, review link or automatic buy.

The conversation agent can only *propose* (through ``bb.purchases`` →
the ``purchases.*`` action handler). Everything that moves money or
carries a secret happens here, in the main process:

1. **Build the checkout** from store + variant ids + quantities only.
   Buyer, address, and payment come from the Shop account's saved
   defaults — the agent cannot supply them.
2. **Decide.** Deterministic rules (:mod:`rules`) and the isolated
   policy checker (:mod:`checker`) both run on every proposal and are
   logged in full. Automatic purchase needs both to pass.
3. **Act.**

   - *Review* (the default, and the fallback for anything that does not
     qualify): the checkout's review link is texted to the admins by
     this service. The agent never sees or composes that link.
   - *Auto*: the admins are told what will be bought and for how much,
     a veto window passes (``cancel buy <id>`` stops it), then the
     checkout is re-read, re-verified, and completed **once** with the
     Shop spending-budget instrument.

Payment is sent at most once per purchase. Two compare-and-set gates
(``scheduled → verifying``, then ``verifying → paying`` immediately
before the one completion call) each admit a single caller; the call is
never retried; and unless Shop explicitly refused *and* the checkout
reads back unpaid, the outcome is recorded as ``unknown`` for a human
to check — never re-attempted, and still counted against the caps.

What counts as a person's request comes from
``Conversation.exchange_log`` — utterances recorded at the transport and
messages actually delivered — never from the agent-visible thread or
from anything the agent wrote.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import httpx

from boxbot.purchases import checker, rules, shop_auth
from boxbot.purchases.shop_client import (
    URL_RE,
    ShopClient,
    ShopError,
    ShopNotConnected,
    budget_instrument,
    fetch_store_products,
    idempotency_key_for,
    normalize_shop,
    normalize_variant_id,
    plain_text,
    summarize_checkout,
)
from boxbot.purchases.store import get_purchase_store, utc_iso

if TYPE_CHECKING:
    from boxbot.communication.auth import User
    from boxbot.core.conversation import Conversation

logger = logging.getLogger(__name__)

_MAX_LINES = 10
_MAX_QUANTITY = 10
_MAX_REASON_LEN = 200
_MAX_TURNS = 24
_MAX_TURN_CHARS = 2000
_MAX_BOXBOT_CHARS = 300
_PROPOSAL_REUSE_SECONDS = 600
_READBACK_GAP_SECONDS = 3.0
_MAX_RECENT_PURCHASES = 20
_LINK_RESEND_SECONDS = 600
_NOT_CONNECTED_NOTICE_SECONDS = 3600
_SWEEP_SECONDS = 60
_CONNECT_MAX_SECONDS = 1200
_REVIEW_LINK_MAX_AGE_SECONDS = 86400
# Checkout states in which Shop has certainly not taken payment.
_UNPAID_STATES = ("ready_for_complete", "incomplete", "requires_escalation")
_TEXT_CHANNELS = ("signal", "whatsapp")

_CANCEL_RE = re.compile(r"^\s*cancel\s+buy\s+([0-9a-fA-F]{6,12})\s*$", re.IGNORECASE)
_CONNECT_RE = re.compile(r"^\s*shop\s+connect\s*$", re.IGNORECASE)
_ALLOW_SHOP_RE = re.compile(r"^\s*allow\s+shop\s+(\S{4,253})\s*$", re.IGNORECASE)
_BLOCK_SHOP_RE = re.compile(r"^\s*block\s+shop\s+(\S{4,253})\s*$", re.IGNORECASE)


class PurchaseError(Exception):
    """A purchase action was refused. The message is safe to show the agent."""


class _NotBuying(Exception):
    """Pre-payment verification failed; the purchase goes to review."""


class _RateWindow:
    """At most ``limit`` events per ``seconds``, in memory."""

    def __init__(self, limit: int, seconds: float) -> None:
        self.limit = limit
        self.seconds = seconds
        self._events: deque[float] = deque()

    def _trim(self) -> None:
        cutoff = time.monotonic() - self.seconds
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    def count(self) -> int:
        self._trim()
        return len(self._events)

    def take(self) -> bool:
        self._trim()
        if len(self._events) >= self.limit:
            return False
        self._events.append(time.monotonic())
        return True

    def add(self) -> None:
        self._events.append(time.monotonic())


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

_client: ShopClient | None = None
_tasks: set[asyncio.Task] = set()
_armed: set[str] = set()              # purchase ids with a live veto timer
_propose_lock = asyncio.Lock()
_last_not_connected_notice = 0.0
_connect_task: asyncio.Task | None = None
_search_rate = _RateWindow(60, 3600)
_feed_rate = _RateWindow(30, 3600)
_resend_rate = _RateWindow(6, 3600)          # review-link resends, all purchases
_shop_ask_rate = _RateWindow(3, 86400)       # "may boxBot shop at …?" texts
_shop_asked: dict[str, float] = {}           # host → when the admins were last asked
# Every propose() attempt, including ones that fail before a row exists.
_propose_attempts = _RateWindow(10**9, 86400)
# Identical proposals from one conversation share one task for a while,
# so a retry after a script timeout returns the first result instead of
# building a second checkout and texting the admins again.
_recent_proposals: dict[tuple, tuple[asyncio.Task, float]] = {}
_sleep = asyncio.sleep                # indirection so tests can skip the wait
_clock = time.time                    # …and move the clock with it


def get_shop_client() -> ShopClient:
    global _client
    if _client is None:
        _client = ShopClient()
    return _client


def set_shop_client(client: ShopClient | None) -> None:
    """Replace the process-wide Shop client (tests)."""
    global _client
    _client = client


def _config() -> Any:
    from boxbot.core.config import get_config

    return get_config().purchases


def _require_enabled() -> Any:
    cfg = _config()
    if not cfg.enabled:
        raise PurchaseError(
            "purchasing is turned off on this box (purchases.enabled)"
        )
    return cfg


def _spawn(coro: Any, name: str) -> asyncio.Task:
    task = asyncio.get_running_loop().create_task(coro, name=name)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task


async def _notify_admins(text: str) -> int:
    """Text every registered admin; return how many sends were accepted.

    "Accepted" is the transport's word, not proof a person read it —
    WhatsApp in particular can accept and never deliver.
    """
    from boxbot.communication.channels import send_to_admins

    try:
        delivered, _ = await send_to_admins(text)
    except Exception:  # noqa: BLE001 — a notice must never break a purchase path
        logger.exception("purchase notice to admins failed")
        return 0
    return delivered


async def _note_not_connected() -> None:
    """Ask the admins to reconnect Shop, at most once an hour."""
    global _last_not_connected_notice
    now = time.monotonic()
    if _last_not_connected_notice and (
        now - _last_not_connected_notice < _NOT_CONNECTED_NOTICE_SECONDS
    ):
        return
    _last_not_connected_notice = now
    await _notify_admins(
        "🛒 boxBot can't reach the Shop account (sign-in missing or "
        'expired). Reply "shop connect" to sign in again.'
    )


def _shop_failure(e: Exception) -> PurchaseError:
    if isinstance(e, ShopNotConnected):
        return PurchaseError(
            "the Shop account is not connected; the admins have been "
            "asked to sign in again. Tell the person — do not retry."
        )
    if isinstance(e, ShopError):
        return PurchaseError(
            f"Shop refused the request: {plain_text(str(e), 200)}"
        )
    return PurchaseError("could not reach Shop (network error); try again later")


# ---------------------------------------------------------------------------
# Views and formatting
# ---------------------------------------------------------------------------


def public_view(row: dict[str, Any]) -> dict[str, Any]:
    """The purchase record as the agent may see it.

    Omits the review link, the checkout id, and the order reference —
    each is a handle on the checkout or order (and the page behind it
    shows the delivery address) that only an admin should hold.
    ``note`` is always service-written text, shown only for a finished
    purchase.
    """
    return {
        "id": row["id"],
        "shop": row["shop"],
        "items": row["items"],
        "total_cents": row["total_cents"],
        "currency": row["currency"],
        "reason": row["reason"],
        "status": row["status"],
        "decision": row["decision"],
        "complete_after": row["complete_after"],
        # Only once the purchase is finished: an in-flight note can
        # say how the checks went.
        "note": row["note"] if row["status"] in _FINAL_STATUSES else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


_FINAL_STATUSES = ("ordered", "cancelled", "failed", "unknown")


def _money(cents: Any, currency: str | None = "USD") -> str:
    if not isinstance(cents, int):
        return "unknown"
    amount = f"{cents / 100:,.2f}"
    return f"${amount}" if currency == "USD" else f"{amount} {currency or '?'}"


def _describe(row: dict[str, Any], facts: dict[str, Any] | None = None) -> str:
    """Store, items and total — every figure read from Shop, not the agent.

    Item titles are merchant text: single-line, link-stripped, clipped.
    """
    listed = row["shop"] in _config().allowed_shops
    lines = [
        f"Store: {row['shop']} "
        f"({'on' if listed else 'NOT on'} your allow-list)"
    ]
    for item in row["items"]:
        lines.append(
            f"- {item.get('quantity')} × "
            f"{plain_text(item.get('title'), 120)} "
            f"({_money(item.get('unit_cents'), row['currency'])} each)"
        )
    total = f"Total: {_money(row['total_cents'], row['currency'])}"
    shipping = (facts or {}).get("shipping_cents")
    if isinstance(shipping, int) and shipping:
        total += f" (includes {_money(shipping, row['currency'])} shipping)"
    lines.append(total)
    return "\n".join(lines)


def _epoch(iso: str | None) -> float | None:
    """A stored UTC timestamp as epoch seconds."""
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return None


def _minutes_ago(at: float | None, now: float) -> int | None:
    return max(0, round((now - at) / 60)) if at is not None else None


# Wording only a genuine notice from this service carries.
_NOTICE_RE = re.compile(
    r"\bboxbot (?:purchase|bought|will buy this in|was asked to shop)\b"
    r"|\bshop sign-in for boxbot\b"
    r"|\bcan't reach the shop account\b"
    # the approval command spelled out with a store domain
    r"|\ballow shop\s+\S+\.\S"
    # Shop's own sign-in and payment pages
    r"|\baccounts\.shop\.app\b|\bshop\.app/(?:pay|checkouts|device)\b"
)
_NOTICE_LABEL_RE = re.compile(r"\breview\s*(?:and|&|\+|n)\s*pay\s*[:→>\-–—=]")
# With a link in the message: plain "pay here" language, or a sign-in
# request that mentions Shop.
_PAY_BAIT_RE = re.compile(
    r"\b(?:review (?:and|&) pay|pay here|pay now|complete (?:your |the )?payment|"
    r"finish paying|shop pay|device code|verification code)\b"
)
_SIGN_IN_RE = re.compile(r"\b(?:sign[ -]?in|log[ -]?in)\b")
_SHOP_WORD_RE = re.compile(r"\bshop\b")
_MAX_FILTER_CHARS = 8000


def looks_like_service_notice(text: str) -> bool:
    """True when ``text`` imitates a purchase notice or a pay/sign-in link.

    Purchase notices, review links and Shop sign-in links reach people
    only from this service. Agent-originated text to a person (the
    ``message`` tool, ``bb.auth.notify_admins``) is refused when it
    carries a notice's wording, or pairs a link with pay-here or
    Shop-sign-in language, so the agent cannot drop a look-alike into
    the same thread.

    This is a bar-raiser, not a proof: wording can always be varied,
    and it is tuned to leave ordinary messages (recipe links, "you may
    need to log in", "approved") alone. What actually protects the
    money is that paying from a link needs the admin's own Shop
    sign-in, that automatic purchases never depend on a link, and that
    a store is approved only by an admin's own text. Only applied while
    purchasing is enabled.
    """
    import unicodedata

    try:
        if not _config().enabled:
            return False
    except Exception:  # noqa: BLE001 — config not loaded (tests, tools)
        return False
    if len(text or "") > _MAX_FILTER_CHARS:
        # Too long to be an ordinary message; not scanned, not sent.
        return True
    folded = "".join(
        ch for ch in unicodedata.normalize("NFKC", text or "")
        if unicodedata.category(ch) not in ("Cf", "Co", "Cn", "Cs")
    ).lower()
    folded = " ".join(folded.split())
    if _NOTICE_RE.search(folded):
        return True
    if not URL_RE.search(folded):
        return False
    return bool(
        _NOTICE_LABEL_RE.search(folded)
        or _PAY_BAIT_RE.search(folded)
        or (_SIGN_IN_RE.search(folded) and _SHOP_WORD_RE.search(folded))
    )


# ---------------------------------------------------------------------------
# Who asked, and what was said
# ---------------------------------------------------------------------------


async def identify_requester(
    conv: "Conversation | None",
) -> tuple[rules.Requester, "User | None"]:
    """Establish the requester from the transport, never from the agent."""
    if conv is None:
        return rules.Requester(None, False, "no conversation"), None
    channel = getattr(conv, "channel", None)
    key = getattr(conv, "channel_key", "") or ""
    if channel in _TEXT_CHANNELS and ":" in key:
        phone = key.split(":", 1)[1]
        from boxbot.communication.auth import get_auth_manager

        auth = get_auth_manager()
        user = await auth.get_user(phone) if auth is not None else None
        if user is not None:
            # Role first and the display name quoted: the name is
            # chosen by the user and must not read as part of the label.
            role = "admin" if user.role == "admin" else "user"
            return rules.Requester(
                channel,
                user.role == "admin",
                f"{role} '{plain_text(user.name, 40)}' ({channel} text thread)",
            ), user
        return rules.Requester(channel, False, f"unregistered {channel} sender"), None
    if channel == "voice":
        return rules.Requester(channel, False, "voice in the room"), None
    return rules.Requester(channel, False, f"{channel or 'unknown'} conversation"), None


def conversation_turns(
    conv: "Conversation | None", user: "User | None" = None,
) -> tuple[list[dict[str, Any]], float | None]:
    """``(turns, when a person last spoke)`` from the provenance ledger.

    Reads ``Conversation.exchange_log`` only. ``person`` entries are
    utterances the transport attributed to a human. A ``boxbot`` entry
    is a message that was actually delivered *to the people in this
    conversation* — a text to this conversation's own user, or speech
    in the room for a voice conversation — and only the one directly
    before a person's turn is kept (the thing they were answering),
    link-stripped and clipped hard, because it is agent-written text.
    Each turn carries ``minutes_ago``.
    """
    log = list(getattr(conv, "exchange_log", None) or [])
    channel = getattr(conv, "channel", None)
    names = set()
    if user is not None:
        names = {user.name.strip().lower(), user.phone.strip().lower()}
    now = time.time()
    turns: list[dict[str, Any]] = []
    pending_boxbot: dict[str, Any] | None = None
    latest_person_at: float | None = None
    for entry in log:
        text = str(entry.get("text") or "").strip()
        if not text:
            continue
        at = entry.get("at")
        at = float(at) if isinstance(at, (int, float)) else None
        if entry.get("speaker") == "person":
            if pending_boxbot is not None:
                turns.append(pending_boxbot)
                pending_boxbot = None
            name = entry.get("name")
            turns.append({
                "speaker": "person",
                "text": (f"[{plain_text(name, 40)}] " if name else "")
                + text[:_MAX_TURN_CHARS],
                "minutes_ago": _minutes_ago(at, now),
            })
            if at is not None:
                latest_person_at = at
        elif entry.get("speaker") == "boxbot":
            to = str(entry.get("to") or "").strip().lower()
            mine = (
                channel in _TEXT_CHANNELS
                and entry.get("channel") == "text"
                and to in names
            ) or (channel == "voice" and entry.get("channel") == "voice")
            if mine:
                shown = plain_text(text, _MAX_BOXBOT_CHARS)
                pending_boxbot = {
                    "speaker": "boxbot",
                    "text": shown,
                    "minutes_ago": _minutes_ago(at, now),
                }
                if len(text) > _MAX_BOXBOT_CHARS or shown.endswith("…"):
                    pending_boxbot["truncated"] = True
    return turns[-_MAX_TURNS:], latest_person_at


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


async def search(
    query: Any, *, max_price_cents: Any = None, limit: Any = 8,
) -> list[dict[str, Any]]:
    _require_enabled()
    if not isinstance(query, str) or not query.strip():
        raise PurchaseError("a search query is required")
    if max_price_cents is not None and (
        isinstance(max_price_cents, bool)
        or not isinstance(max_price_cents, int)
        or max_price_cents <= 0
    ):
        raise PurchaseError("max_price_cents must be a positive integer")
    limit = _limit(limit, 8, 20)
    if not _search_rate.take():
        raise PurchaseError("too many searches this hour; stop and tell the person")
    try:
        return await get_shop_client().search(
            query.strip()[:200], max_price_cents=max_price_cents, limit=limit,
        )
    except ShopNotConnected as e:
        await _note_not_connected()
        raise _shop_failure(e) from e
    except (ShopError, httpx.HTTPError) as e:
        raise _shop_failure(e) from e


async def store_products(
    shop: Any, *, query: Any = None, limit: Any = 20,
) -> list[dict[str, Any]]:
    cfg = _require_enabled()
    try:
        host = normalize_shop(shop)
    except ValueError as e:
        raise PurchaseError(str(e)) from e
    if host not in cfg.allowed_shops:
        raise PurchaseError(
            f"{host} is not in purchases.allowed_shops; only an admin can "
            "add a store. Use search() for stores Shop lists."
        )
    if query is not None and not isinstance(query, str):
        raise PurchaseError("query must be text")
    limit = _limit(limit, 20, 50)
    if not _feed_rate.take():
        raise PurchaseError("too many store lookups this hour; stop and tell the person")
    try:
        return await fetch_store_products(
            host, query=(query or "")[:200] or None, limit=limit,
        )
    except (ShopError, httpx.HTTPError) as e:
        raise _shop_failure(e) from e


def _limit(value: Any, default: int, ceiling: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(1, min(value, ceiling))


# ---------------------------------------------------------------------------
# Propose
# ---------------------------------------------------------------------------


def _validate_items(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list) or not items:
        raise PurchaseError("items must be a non-empty list")
    if len(items) > _MAX_LINES:
        raise PurchaseError(f"at most {_MAX_LINES} different items per purchase")
    cleaned: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            raise PurchaseError("each item is {'variant_id': …, 'quantity': …}")
        try:
            variant = normalize_variant_id(item.get("variant_id"))
        except ValueError as e:
            raise PurchaseError(str(e)) from e
        quantity = item.get("quantity", 1)
        if (
            isinstance(quantity, bool)
            or not isinstance(quantity, int)
            or not 1 <= quantity <= _MAX_QUANTITY
        ):
            raise PurchaseError(f"quantity must be 1–{_MAX_QUANTITY}")
        cleaned[variant] = cleaned.get(variant, 0) + quantity
        if cleaned[variant] > _MAX_QUANTITY:
            raise PurchaseError(f"quantity must be 1–{_MAX_QUANTITY}")
    return [{"variant_id": v, "quantity": q} for v, q in cleaned.items()]


async def propose(
    shop: Any,
    items: Any,
    reason: Any,
    *,
    conv: "Conversation | None",
) -> dict[str, Any]:
    """Build a checkout, decide, and either schedule it or send it for review.

    Returns ``{"purchase": <public view>, "decision": "auto" |
    "review", "blocked_by": [rule names], "checker": "allow" | "deny" |
    "error", "admins_notified": int, "already_proposed": bool}``. The
    action handler removes ``checker`` (and, in auto mode, the rule
    names) before anything reaches the sandbox.

    The work runs in a main-process task that the caller's cancellation
    cannot interrupt: once a checkout exists, the decision is always
    logged and the admins are always told, even if the sandbox script
    that asked has timed out. An identical proposal from the same
    conversation within ten minutes returns that first result
    (``already_proposed: True``) instead of building another checkout.
    """
    _require_enabled()
    try:
        host = normalize_shop(shop)
    except ValueError as e:
        raise PurchaseError(str(e)) from e
    lines = _validate_items(items)
    if not isinstance(reason, str) or not reason.strip():
        raise PurchaseError("a reason for the purchase is required")
    # Shown to admins, never to the checker. Single line, no links.
    reason = plain_text(reason[:2000], _MAX_REASON_LEN)

    now = time.monotonic()
    for stale in [
        k for k, (_, at) in _recent_proposals.items()
        if now - at > _PROPOSAL_REUSE_SECONDS
    ]:
        _recent_proposals.pop(stale, None)
    key = (
        getattr(conv, "channel_key", None) if conv is not None else None,
        host,
        tuple(sorted((l["variant_id"], l["quantity"]) for l in lines)),
    )
    while True:
        existing = _recent_proposals.get(key)
        if existing is None:
            break
        outcome = dict(await asyncio.shield(existing[0]))
        current = await get_purchase_store().get(outcome["purchase"]["id"])
        if current is not None and current["status"] not in ("cancelled", "failed"):
            # Report the purchase as it stands now, not as it was decided.
            outcome["already_proposed"] = True
            outcome["purchase"] = public_view(current)
            outcome["decision"] = current["decision"] or "review"
            return outcome
        # That proposal is dead (cancelled or refused). Asking again is
        # a new proposal — which the duplicate rule sends to review.
        # Drop the entry only if it is still the one we looked at, then
        # look again: another caller may already have started the
        # replacement, and this one should join it, not start a second.
        if _recent_proposals.get(key) is existing:
            _recent_proposals.pop(key, None)

    task = _spawn(_propose(host, lines, reason, conv), "purchase-propose")
    _recent_proposals[key] = (task, now)

    def _forget_failure(done: asyncio.Task) -> None:
        # Only a proposal that produced a purchase is worth reusing.
        if done.cancelled() or done.exception() is not None:
            if _recent_proposals.get(key, (None,))[0] is done:
                _recent_proposals.pop(key, None)

    task.add_done_callback(_forget_failure)
    outcome = dict(await asyncio.shield(task))
    outcome["already_proposed"] = False
    return outcome


async def _propose(
    host: str,
    lines: list[dict[str, Any]],
    reason: str,
    conv: "Conversation | None",
) -> dict[str, Any]:
    cfg = _config()
    store = get_purchase_store()
    # One proposal at a time: the attempt ceiling, the spend caps and
    # the duplicate check must each see the proposals before them.
    async with _propose_lock:
        requester, user = await identify_requester(conv)
        # Creating a checkout hands the store the household's saved
        # contact and delivery details, so it happens only for a store
        # the household allow-listed or an admin approved by their own
        # text command. Nothing the agent says can stand in for that.
        if host not in cfg.allowed_shops and not await store.shop_approved(host):
            asked = await _ask_shop_approval(host, requester)
            raise PurchaseError(
                f"{host} is not a store this household has approved. "
                f"{_SHOP_ASK_REPLY[asked]} Tell the person — do not retry."
            )

        utc_now = datetime.now(timezone.utc)
        day_ago = utc_iso(utc_now - timedelta(hours=24))
        attempts = max(_propose_attempts.count(), await store.count_since(day_ago))
        if attempts >= cfg.max_proposals_per_day:
            raise PurchaseError(
                "too many purchase proposals in the last 24 hours; stop and "
                "tell the person"
            )
        _propose_attempts.add()

        turns, latest_person_at = conversation_turns(conv, user)
        channel_key = getattr(conv, "channel_key", None) if conv is not None else None

        try:
            checkout = await get_shop_client().create_checkout(host, lines)
        except ShopNotConnected as e:
            await _note_not_connected()
            raise _shop_failure(e) from e
        except (ShopError, httpx.HTTPError) as e:
            raise _shop_failure(e) from e

        facts = summarize_checkout(checkout)
        checkout_id = checkout.get("id")
        if not isinstance(checkout_id, str) or not checkout_id:
            raise PurchaseError("Shop returned no checkout id")
        if not facts["lines"]:
            raise PurchaseError("the store returned an empty checkout")
        continue_url = checkout.get("continue_url")
        if not (isinstance(continue_url, str) and continue_url.startswith("https://")):
            continue_url = None

        now = datetime.now()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = day_start.replace(day=1)
        earlier = await store.since(
            utc_iso(utc_now - timedelta(days=cfg.duplicate_window_days))
        )
        wanted = {str(l.get("variant_id")) for l in facts["lines"]}
        # Any earlier proposal for this item counts, whatever became of
        # it: bought, possibly bought, sent for review (a person may
        # have paid), or cancelled (possibly an admin's veto).
        duplicate_of = next(
            (
                r["id"] for r in earlier
                if r["shop"] == host
                and wanted & {str(i.get("variant_id")) for i in r["items"]}
            ),
            None,
        )
        last_proposal = (
            _epoch(await store.last_proposal_at(channel_key))
            if channel_key else None
        )
        epoch_now = time.time()

        row = await store.create(
            shop=host,
            items=facts["lines"],
            total_cents=facts["total_cents"],
            currency=facts["currency"],
            reason=reason,
            requested_by=requester.label,
            channel_key=channel_key,
            checkout_id=checkout_id,
            continue_url=continue_url,
        )

        try:
            final, results, verdict, notified = await _decide_and_act(
                cfg=cfg, store=store, row=row, host=host, facts=facts,
                requester=requester, turns=turns, earlier=earlier,
                duplicate_of=duplicate_of, latest_person_at=latest_person_at,
                last_proposal=last_proposal, epoch_now=epoch_now,
                day_start=day_start, month_start=month_start,
            )
        except Exception:  # noqa: BLE001
            # A checkout exists; it must not be left with no decision
            # and nobody told. Whatever went wrong, a human reviews it.
            logger.exception("purchase %s: decision step failed", row["id"])
            final, results, verdict, notified = "review", [], None, 0
            note = "decided by fallback: an internal error during the checks"
            try:
                # It may already have been scheduled; nothing is bought
                # automatically after an error.
                if await store.transition(
                    row["id"], "review", expect="scheduled",
                    decision="review", complete_after=None, note=note,
                ) is None:
                    await store.update_fields(row["id"], decision="review", note=note)
            except Exception:  # noqa: BLE001
                logger.exception("purchase %s: could not record fallback", row["id"])

    if final == "review":
        notified = await _send_review_link(row["id"], facts=facts)

    logger.info(
        "Purchase %s proposed: %s, %s item(s) from %s → %s (checker=%s)",
        row["id"], _money(facts["total_cents"], facts["currency"]),
        facts["item_count"], host, final,
        verdict.decision if verdict is not None else "not run",
    )
    current = await store.get(row["id"])
    return {
        "purchase": public_view(current or row),
        "decision": final,
        "blocked_by": rules.failed_names(results) if results else ["internal_error"],
        "checker": verdict.decision if verdict is not None else "error",
        "admins_notified": notified,
    }


async def _decide_and_act(
    *, cfg: Any, store: Any, row: dict[str, Any], host: str,
    facts: dict[str, Any], requester: rules.Requester,
    turns: list[dict[str, Any]], earlier: list[dict[str, Any]],
    duplicate_of: str | None, latest_person_at: float | None,
    last_proposal: float | None, epoch_now: float,
    day_start: datetime, month_start: datetime,
) -> tuple[str, list[rules.RuleResult], checker.CheckerResult, int]:
    """Rules + checker + decision log, then schedule if automatic.

    Returns ``(final, rule results, checker verdict, admins notified)``.
    """
    results = rules.evaluate(
        config=cfg,
        shop=host,
        facts=facts,
        requester=requester,
        spent_today_cents=await store.committed_cents_since(utc_iso(day_start)),
        spent_month_cents=await store.committed_cents_since(utc_iso(month_start)),
        duplicate_of=duplicate_of,
        latest_person_at=latest_person_at,
        last_proposal_at=last_proposal,
    )
    rules_ok = rules.all_pass(results)
    verdict = await checker.check(
        model=cfg.checker_model or _large_model(),
        timeout=cfg.checker_timeout_seconds,
        policy=cfg.policy,
        conversation=turns,
        shop=host,
        facts=facts,
        recent_purchases=[
            {
                "minutes_ago": _minutes_ago(_epoch(r["created_at"]), epoch_now),
                "store": r["shop"],
                "status": r["status"],
                "items": [
                    {"title": i.get("title"), "quantity": i.get("quantity")}
                    for i in r["items"]
                ],
            }
            for r in earlier[:_MAX_RECENT_PURCHASES]
        ],
        purchase_id=row["id"],
    )
    final = "auto" if rules_ok and verdict.allowed else "review"
    await store.record_decision(
        purchase_id=row["id"],
        mode=cfg.mode,
        rules=[r.as_dict() for r in results],
        rules_pass=rules_ok,
        checker_model=verdict.model,
        checker_input=verdict.input_json,
        checker_output=verdict.output_raw,
        checker_decision=verdict.decision,
        checker_latency_ms=verdict.latency_ms,
        final=final,
    )

    notified = 0
    if final == "auto":
        # Claim the timer slot before the row becomes "scheduled", so
        # the sweep never sees a scheduled row it thinks is orphaned
        # while the notice below is still being sent.
        _armed.add(row["id"])
        timer_started = False
        try:
            # Schedule, then announce: if the row was cancelled while
            # the checker ran, nobody is told "will buy".
            due = datetime.now(timezone.utc) + timedelta(
                seconds=cfg.veto_window_seconds
            )
            scheduled = await store.transition(
                row["id"], "scheduled", expect="review",
                decision="auto", complete_after=utc_iso(due),
            )
            if scheduled is None:
                final = "review"        # cancelled meanwhile; stays cancelled
            else:
                notified = await _announce_auto(row, facts, cfg)
                if notified == 0:
                    # Nobody could veto, so nobody was asked. Not automatic.
                    final = "review"
                    await store.transition(
                        row["id"], "review", expect="scheduled",
                        decision="review", complete_after=None,
                    )
                    logger.info(
                        "Purchase %s: no admin reachable for the veto notice",
                        row["id"],
                    )
                else:
                    _start_timer(row["id"], cfg.veto_window_seconds)
                    timer_started = True
        finally:
            if not timer_started:
                _armed.discard(row["id"])
    if final == "review":
        await store.update_fields(row["id"], decision="review")
    return final, results, verdict, notified


async def _ask_shop_approval(host: str, requester: rules.Requester) -> str:
    """Ask the admins whether boxBot may shop at ``host``. Rate-limited.

    Returns ``"asked"`` (a request reached an admin now), ``"already"``
    (they were asked about this store within the last day),
    ``"limit"`` (too many stores asked about today; nobody was asked),
    or ``"unreachable"`` (no admin could be reached). At most one text
    per store per day and three stores per day, so the agent cannot use
    it to spam. The limits are in memory and reset on restart.
    """
    now = time.monotonic()
    last = _shop_asked.get(host)
    if last is not None and now - last < 86400:
        return "already"
    if not _shop_ask_rate.take():
        return "limit"
    delivered = await _notify_admins(
        f"🛒 boxBot was asked to shop at a store that is not on your "
        f"allow-list:\n{host}\n"
        f"Came from: {requester.label}\n"
        "Building a checkout there gives that store your saved name, "
        "contact and delivery address. Nothing has been sent to it.\n"
        "Allowing it lets boxBot send you review links for that store "
        "from now on; it never buys there on its own.\n"
        f"To allow it, reply: allow shop {host}"
    )
    if delivered == 0:
        return "unreachable"
    _shop_asked[host] = now
    return "asked"


_SHOP_ASK_REPLY = {
    "asked": "The admins have been asked; propose again only after one "
             "of them approves it.",
    "already": "The admins were already asked about this store today.",
    "limit": "Too many new stores have been put to the admins today, so "
             "they were NOT asked about this one.",
    "unreachable": "No admin could be reached to ask.",
}


def _large_model() -> str:
    from boxbot.core.config import get_config

    return get_config().models.large


async def _announce_auto(row: dict[str, Any], facts: dict[str, Any], cfg: Any) -> int:
    minutes = max(1, round(cfg.veto_window_seconds / 60))
    return await _notify_admins(
        f"🛒 boxBot will buy this in about {minutes} "
        f"minute{'s' if minutes != 1 else ''} (purchase {row['id']}):\n"
        f"Came from: {row['requested_by']}\n"
        f"{_describe(row, facts)}\n"
        f'Reply "cancel buy {row["id"]}" to stop it.'
    )


async def _send_review_link(
    purchase_id: str,
    *,
    facts: dict[str, Any] | None = None,
    headline: str = "needs your review",
) -> int:
    """Text the checkout's review link to the admins. Returns reach count.

    Layout is deliberate: everything the service established (who
    asked, store, items, total, policy check) comes first; the agent's
    own words are quoted and marked unverified; the real link is the
    last line.
    """
    store = get_purchase_store()
    row = await store.get(purchase_id)
    if row is None or row["status"] != "review":
        return 0
    decisions = await store.decisions_for(purchase_id)
    verdict = decisions[-1]["checker_decision"] if decisions else "not run"
    body = (
        f"🛒 boxBot purchase {row['id']} — {headline}\n"
        f"Came from: {row['requested_by']}\n"
        f"{_describe(row, facts)}\n"
        f"Policy check: {verdict}\n"
        f"boxBot's stated reason (unverified): \"{plain_text(row['reason'], _MAX_REASON_LEN)}\"\n"
    )
    if not row["continue_url"]:
        delivered = await _notify_admins(body + "The store gave no review link.")
    else:
        delivered = await _notify_admins(
            body + f"Review and pay: {row['continue_url']}"
        )
    if delivered:
        await store.update_fields(purchase_id, link_sent_at=utc_iso())
    return delivered


async def resend_review_link(purchase_id: str) -> dict[str, Any]:
    """Agent-callable: send the review link again (rate-limited)."""
    _require_enabled()
    store = get_purchase_store()
    row = await store.get(purchase_id)
    if row is None:
        raise PurchaseError(f"no purchase '{purchase_id}' found")
    if row["status"] != "review":
        raise PurchaseError(
            f"purchase {purchase_id} is {row['status']}; there is no "
            "review link to send"
        )
    created = _epoch(row["created_at"])
    if created is None or time.time() - created > _REVIEW_LINK_MAX_AGE_SECONDS:
        raise PurchaseError(
            f"purchase {purchase_id} is more than a day old; propose it "
            "again if the person still wants it"
        )
    sent = _epoch(row["link_sent_at"])
    if sent is not None and time.time() - sent < _LINK_RESEND_SECONDS:
        raise PurchaseError("the review link was sent within the last 10 minutes")
    if not _resend_rate.take():
        raise PurchaseError(
            "too many review links re-sent this hour; stop and tell the person"
        )
    delivered = await _send_review_link(purchase_id)
    return {"purchase": public_view(row), "admins_notified": delivered}


_TOO_LATE = {
    "paying": "payment is being sent right now and can no longer be stopped here",
    "ordered": "the order has been placed; contact the store to change it",
    "unknown": "payment may already have gone through; check your Shop orders",
    "cancelled": "it was already cancelled",
    "failed": "it already failed; nothing was bought",
}


async def cancel(purchase_id: str, *, by: str) -> dict[str, Any]:
    """Stop a purchase before payment is sent."""
    store = get_purchase_store()
    row = await store.get(purchase_id)
    if row is None:
        raise PurchaseError(f"no purchase '{purchase_id}' found")
    updated = await store.transition(
        purchase_id, "cancelled", expect=("review", "scheduled", "verifying"),
        note=f"cancelled by {plain_text(by, 60)}",
    )
    if updated is None:
        current = await store.get(purchase_id)
        status = (current or row)["status"]
        raise PurchaseError(
            f"purchase {purchase_id} cannot be cancelled: "
            + _TOO_LATE.get(status, f"it is {status}")
        )
    logger.info("Purchase %s cancelled by %s", purchase_id, by)
    return updated


# ---------------------------------------------------------------------------
# Automatic completion
# ---------------------------------------------------------------------------


def _start_timer(purchase_id: str, delay: float) -> None:
    """Start the veto timer. The caller has put the id in ``_armed``;
    the timer task removes it when it ends, however it ends."""
    _spawn(_complete_after(purchase_id, delay), f"purchase-complete-{purchase_id}")


async def _complete_after(purchase_id: str, delay: float) -> None:
    try:
        if delay > 0:
            await _sleep(delay)
        await _complete(purchase_id)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — a crash here must not strand the row
        logger.exception("purchase %s completion task crashed", purchase_id)
        await _settle_stuck(purchase_id, "an internal error")
    finally:
        _armed.discard(purchase_id)


async def _settle_stuck(purchase_id: str, why: str) -> None:
    """Resolve a purchase whose completion task died, by where it stopped.

    Before the ``paying`` gate no payment was sent → human review. At
    or after it, payment may have been sent → ``unknown``.
    """
    try:
        row = await get_purchase_store().get(purchase_id)
        if row is None:
            return
        if row["status"] in ("scheduled", "verifying"):
            await _back_to_review(purchase_id, why, from_status=row["status"])
        elif row["status"] == "paying":
            await _mark_unknown(purchase_id, why)
    except Exception:  # noqa: BLE001
        logger.exception("could not settle stuck purchase %s", purchase_id)


async def _back_to_review(purchase_id: str, why: str, *, from_status: str) -> None:
    """Payment was not sent; hand the purchase to a human instead.

    Only ever called for a row in ``scheduled`` or ``verifying``.
    """
    assert from_status in ("scheduled", "verifying"), from_status
    store = get_purchase_store()
    row = await store.transition(
        purchase_id, "review", expect=from_status,
        decision="review", note=f"not bought automatically: {why}",
        complete_after=None,
    )
    if row is None:
        return
    logger.info("Purchase %s not bought automatically: %s", purchase_id, why)
    await _send_review_link(
        purchase_id, headline=f"was not bought automatically ({why})",
    )


async def _mark_unknown(purchase_id: str, why: str) -> None:
    store = get_purchase_store()
    row = await store.transition(
        purchase_id, "unknown", expect="paying",
        note=f"payment outcome unconfirmed: {why}",
    )
    if row is None:
        return
    logger.error("Purchase %s outcome unknown: %s", purchase_id, why)
    await _notify_admins(
        f"⚠️ boxBot purchase {purchase_id}: a payment request may have "
        f"been sent and the outcome could not be confirmed ({why}). It "
        f"will NOT be retried. Check your Shop orders.\n{_describe(row)}"
    )


async def _mark_ordered(purchase_id: str, order: Any) -> None:
    store = get_purchase_store()
    ref: dict[str, Any] = {}
    if isinstance(order, dict):
        for key in ("id", "permalink_url", "label"):
            if isinstance(order.get(key), str):
                ref[key] = order[key]
    row = await store.transition(
        purchase_id, "ordered", expect="paying",
        order_ref=json.dumps(ref) if ref else None,
    )
    if row is None:
        return
    logger.info("Purchase %s ordered", purchase_id)
    link = ref.get("permalink_url", "")
    await _notify_admins(
        f"✅ boxBot bought (purchase {purchase_id}):\n{_describe(row)}"
        + (f"\nOrder: {link}" if link.startswith("https://") else "")
    )


async def _verify(row: dict[str, Any]) -> dict[str, Any]:
    """Re-read and re-check the checkout; return the instrument to pay with.

    Sends no payment. Raises :class:`_NotBuying` with the reason when
    anything is off.
    """
    cfg = _config()
    if not (cfg.enabled and cfg.mode == "auto"):
        raise _NotBuying("automatic purchasing was switched off")
    try:
        checkout = await get_shop_client().get_checkout(row["shop"], row["checkout_id"])
    except ShopNotConnected as e:
        await _note_not_connected()
        raise _NotBuying("the Shop account is not connected") from e
    except (ShopError, httpx.HTTPError) as e:
        raise _NotBuying("the checkout could not be re-read") from e

    facts = summarize_checkout(checkout)
    if facts["status"] != "ready_for_complete":
        raise _NotBuying(f"checkout is {plain_text(facts['status'], 40)}")
    if facts["total_cents"] != row["total_cents"]:
        raise _NotBuying("the total changed")
    if facts["currency"] != row["currency"]:
        raise _NotBuying("the currency changed")
    if not isinstance(facts["total_cents"], int) or (
        facts["total_cents"] > cfg.max_per_purchase_cents
    ):
        raise _NotBuying("the total is over the per-purchase limit")
    if _line_key(facts["lines"]) != _line_key(row["items"]):
        raise _NotBuying("the items changed")
    if row["shop"] not in cfg.allowed_shops:
        raise _NotBuying("the store is no longer on the allow-list")
    if len(facts["destinations"]) != 1:
        raise _NotBuying("the delivery destination changed")
    if any(m.get("type") in ("error", "warning") for m in facts["messages"]):
        raise _NotBuying("the checkout now carries a warning")
    # The caps again, as of now. Both sums include this purchase.
    store = get_purchase_store()
    now = datetime.now()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if await store.committed_cents_since(utc_iso(day_start)) > cfg.max_per_day_cents:
        raise _NotBuying("the daily limit would be exceeded")
    if await store.committed_cents_since(
        utc_iso(day_start.replace(day=1))
    ) > cfg.max_per_month_cents:
        raise _NotBuying("the monthly limit would be exceeded")
    instrument = budget_instrument(checkout)
    if instrument is None:
        raise _NotBuying("the Shop spending budget is not offered")
    return instrument


async def _complete(purchase_id: str) -> None:
    """Pay for one scheduled purchase. Sends payment at most once."""
    store = get_purchase_store()

    # Never before the announced time, whoever calls: the veto window
    # the admins were promised is not negotiable.
    pending = await store.get(purchase_id)
    if pending is None or pending["status"] != "scheduled":
        return
    due = _epoch(pending["complete_after"])
    if due is None or _clock() < due:
        logger.warning("purchase %s: completion attempted before it was due", purchase_id)
        return

    # Gate 1: one caller per purchase gets to verify.
    row = await store.transition(purchase_id, "verifying", expect="scheduled")
    if row is None:
        return  # cancelled, or already handled

    # No payment has been sent, so any problem here — including a bug —
    # safely falls back to human review.
    try:
        instrument = await _verify(row)
    except _NotBuying as e:
        await _back_to_review(purchase_id, str(e), from_status="verifying")
        return
    except Exception:  # noqa: BLE001
        logger.exception("purchase %s verification crashed", purchase_id)
        await _back_to_review(
            purchase_id, "an internal error before payment",
            from_status="verifying",
        )
        return

    # Gate 2: set immediately before the call, so a row found in
    # "paying" is the only kind that may have had payment sent. A
    # cancel that landed during verification wins here.
    if await store.transition(purchase_id, "paying", expect="verifying") is None:
        return

    # --- point of no return: one completion call, never repeated ---------
    try:
        result = await get_shop_client().complete_checkout(
            row["shop"], row["checkout_id"], instrument,
            idempotency_key=idempotency_key_for(purchase_id),
        )
    except ShopError as e:
        logger.warning("purchase %s completion error: %s", purchase_id, e.code)
        await _resolve_after_payment_attempt(
            purchase_id, row, refused=e.refused, detail=e.code,
        )
        return
    except Exception as e:  # noqa: BLE001 — network and anything else: ambiguous
        logger.warning("purchase %s completion failed: %r", purchase_id, e)
        await _resolve_after_payment_attempt(
            purchase_id, row, refused=False, detail=type(e).__name__,
        )
        return

    if isinstance(result, dict) and result.get("status") == "completed":
        await _mark_ordered(purchase_id, result.get("order"))
        return
    await _resolve_after_payment_attempt(
        purchase_id, row, refused=False, detail="completion did not confirm",
    )


def _line_key(lines: list[dict[str, Any]]) -> list[tuple[Any, Any, Any]]:
    return sorted(
        (str(l.get("variant_id")), l.get("quantity"), l.get("unit_cents"))
        for l in lines
    )


async def _read_status(row: dict[str, Any]) -> tuple[str | None, Any]:
    """``(checkout status, order)`` from one read-back; ``(None, None)`` on failure."""
    try:
        checkout = await get_shop_client().get_checkout(row["shop"], row["checkout_id"])
    except Exception:  # noqa: BLE001
        return None, None
    if not isinstance(checkout, dict):
        return None, None
    status = checkout.get("status")
    return (status if isinstance(status, str) else None), checkout.get("order")


async def _resolve_after_payment_attempt(
    purchase_id: str, row: dict[str, Any], *, refused: bool, detail: str,
) -> None:
    """Read the checkout back after a payment attempt that did not confirm.

    - reads back ``completed`` → ``ordered``;
    - Shop explicitly refused (``refused``) **and** two read-backs a few
      seconds apart both show a state in which no payment was taken →
      ``failed``;
    - anything else — gateway errors, unreadable bodies, timeouts, an
      in-progress or unreadable checkout → ``unknown``.

    Never re-sends the payment.
    """
    store = get_purchase_store()
    status, order = await _read_status(row)
    if status == "completed":
        await _mark_ordered(purchase_id, order)
        return
    if refused and status in _UNPAID_STATES:
        # A refusal that raced a late capture would show up on a second
        # look; "failed" frees the item to be bought again, so be sure.
        await _sleep(_READBACK_GAP_SECONDS)
        status, order = await _read_status(row)
        if status == "completed":
            await _mark_ordered(purchase_id, order)
            return
        if status in _UNPAID_STATES:
            failed = await store.transition(
                purchase_id, "failed", expect="paying",
                note="Shop refused the payment; nothing was bought",
            )
            if failed is not None:
                logger.warning("Purchase %s refused by Shop (%s)", purchase_id, detail)
                await _notify_admins(
                    f"🛒 boxBot purchase {purchase_id} was NOT bought — Shop "
                    f"refused the payment.\n{_describe(failed)}"
                )
            return
    await _mark_unknown(purchase_id, plain_text(detail, 80))


# ---------------------------------------------------------------------------
# Startup recovery and the stuck-row sweep
# ---------------------------------------------------------------------------


def start() -> None:
    """Begin background recovery + sweeping. Call once at startup.

    Returns immediately; recovery does network reads and must not hold
    up the agent coming online.
    """
    _spawn(_run_background(), "purchases-background")


async def _run_background() -> None:
    try:
        await recover()
    except Exception:  # noqa: BLE001
        logger.exception("purchase recovery failed")
    while True:
        await asyncio.sleep(_SWEEP_SECONDS)
        try:
            await _sweep_scheduled()
        except Exception:  # noqa: BLE001
            logger.exception("purchase sweep failed")


async def recover() -> None:
    """Settle purchases a restart interrupted.

    - ``paying``: payment may have been in flight. The checkout is read
      back once and the result recorded — never re-sent.
    - ``verifying``: the process died before payment was sent → review.
    - ``scheduled``: the veto window was interrupted → review (see
      :func:`_sweep_scheduled`). Nothing is bought automatically on the
      strength of a notice sent before a restart.

    Runs whether or not purchasing is currently enabled — switching it
    off must not strand rows — but does nothing if there has never been
    a purchase database.
    """
    store = get_purchase_store()
    if not store.exists():
        return
    for row in await store.list(status="paying", limit=500):
        logger.warning("Purchase %s was mid-payment at shutdown", row["id"])
        try:
            await _resolve_after_payment_attempt(
                row["id"], row, refused=False,
                detail="boxBot restarted while payment was in flight",
            )
        except Exception:  # noqa: BLE001 — one bad row must not block the rest
            logger.exception("recovery of paying purchase %s failed", row["id"])
    for row in await store.list(status="verifying", limit=500):
        try:
            await _back_to_review(
                row["id"], "boxBot restarted before payment was sent",
                from_status="verifying",
            )
        except Exception:  # noqa: BLE001
            logger.exception("recovery of verifying purchase %s failed", row["id"])
    await _sweep_scheduled()


async def _sweep_scheduled() -> None:
    """Send every ``scheduled`` purchase that has no live timer to review.

    A scheduled row without a timer means the process restarted (or the
    timer died) during the veto window. The admins' chance to veto was
    interrupted — a "cancel buy" sent while boxBot was down may not
    have been received — so it is never bought automatically after
    that. A human gets the review link instead.
    """
    store = get_purchase_store()
    if not store.exists():
        return
    for row in await store.list(status="scheduled", limit=500):
        if row["id"] in _armed:
            continue
        try:
            await _back_to_review(
                row["id"], "boxBot restarted during the veto window",
                from_status="scheduled",
            )
        except Exception:  # noqa: BLE001
            logger.exception("sweep of scheduled purchase %s failed", row["id"])


# ---------------------------------------------------------------------------
# Admin text commands (called by MessageRouter, admins only)
# ---------------------------------------------------------------------------


def parse_admin_command(text: str | None) -> tuple[str, str] | None:
    """``("cancel", <id>)`` / ``("connect", "")`` / ``("allow_shop",
    <host>)`` / ``("block_shop", <host>)`` or None."""
    if not text:
        return None
    m = _CANCEL_RE.match(text)
    if m is not None:
        return "cancel", m.group(1).lower()
    if _CONNECT_RE.match(text):
        return "connect", ""
    m = _ALLOW_SHOP_RE.match(text)
    if m is not None:
        return "allow_shop", m.group(1)
    m = _BLOCK_SHOP_RE.match(text)
    if m is not None:
        return "block_shop", m.group(1)
    return None


async def handle_admin_command(verb: str, arg: str, *, admin: "User") -> str:
    """Apply an admin command; return the reply text for that admin.

    The router has already verified ``admin.role == "admin"``.
    """
    if verb == "cancel":
        try:
            row = await cancel(arg, by=f"admin {admin.name}")
        except PurchaseError as e:
            text = str(e)
            return text[:1].upper() + text[1:] + "."
        return (
            f"Cancelled purchase {arg} "
            f"({_money(row['total_cents'], row['currency'])} at {row['shop']}). "
            "boxBot will not buy it. If you already paid from a review "
            "link, that order stands."
        )
    if verb == "connect":
        return await _start_connect(admin)
    if verb == "allow_shop":
        try:
            host = normalize_shop(arg)
        except ValueError:
            return "That is not a store domain. Example: allow shop www.example.com"
        await get_purchase_store().approve_shop(host, approved_by=admin.phone)
        logger.info("Shop %s approved by %s", host, admin.phone)
        reply = (
            f"OK. boxBot may now build checkouts at {host} and send them "
            "to you to review, until you reply: "
            f"block shop {host}\n"
            "It will not buy there automatically; add the store to "
            "purchases.allowed_shops for that."
        )
        # A mistyped domain approves a store nobody asked for and
        # leaves the wanted one blocked, with no error anywhere. Say so
        # when this is not a store boxBot has asked about.
        asked = sorted(h for h in _shop_asked if h != host)
        if host not in _shop_asked and asked:
            reply += (
                "\nNote: boxBot has not asked about that store. It did "
                "ask about: " + ", ".join(asked[:5]) + ". Check the "
                "spelling if you meant one of those."
            )
        return reply
    if verb == "block_shop":
        try:
            host = normalize_shop(arg)
        except ValueError:
            return "That is not a store domain. Example: block shop www.example.com"
        removed = await get_purchase_store().revoke_shop(host)
        logger.info("Shop %s blocked by %s (was approved: %s)",
                    host, admin.phone, removed)
        if host in _config().allowed_shops:
            return (
                f"{host} is on the configured allow-list "
                "(purchases.allowed_shops), which a text cannot change. "
                "Remove it from the config to block it."
            )
        if not removed:
            return f"{host} was not an approved store. Nothing changed."
        return (
            f"OK. boxBot will no longer build checkouts at {host}. "
            "Review links already sent still work."
        )
    return "Unknown purchase command."


async def _start_connect(admin: "User") -> str:
    global _connect_task
    result = await asyncio.to_thread(shop_auth.device_start)
    if not result.get("ok"):
        return (
            "Could not start Shop sign-in: "
            f"{plain_text(result.get('error'), 120)}"
        )
    # One sign-in at a time: a new request replaces the pending one
    # (device_start has already overwritten its device code).
    if _connect_task is not None and not _connect_task.done():
        _connect_task.cancel()
    _connect_task = _spawn(
        _finish_connect(admin, float(result.get("interval") or 5)),
        "shop-connect",
    )
    return (
        "Shop sign-in for boxBot, started by your \"shop connect\" just now:\n"
        f"1. Open {result['link']}\n"
        f"2. Check the code shown is {result['user_code']} and approve.\n"
        f"Valid for {max(1, result['expires_in'] // 60)} minutes. boxBot only "
        "ever sends this in direct reply to \"shop connect\" — ignore any "
        "sign-in link that arrives another way."
    )


async def _finish_connect(admin: "User", interval: float) -> None:
    global _last_not_connected_notice
    state = "timed out"
    deadline = time.monotonic() + _CONNECT_MAX_SECONDS
    try:
        while time.monotonic() < deadline:
            await asyncio.sleep(interval)
            state = await asyncio.to_thread(shop_auth.device_poll_once)
            if state == "pending":
                state = "timed out"
                continue
            if state == "slow_down":
                interval += 5
                state = "timed out"
                continue
            break
    except asyncio.CancelledError:
        raise  # replaced by a newer sign-in; say nothing
    except Exception:  # noqa: BLE001
        logger.exception("Shop sign-in polling failed")
        state = "internal error"
    if state == "approved":
        _last_not_connected_notice = 0.0
        text = (
            "✅ Shop connected. Automatic purchases also need a spending "
            "limit set on this new boxBot connection in your Shop "
            "account settings (Connections)."
        )
    else:
        text = (
            f"Shop sign-in did not finish ({plain_text(state, 40)}). "
            'Send "shop connect" to try again.'
        )
    logger.info("Shop connect finished: %s", state)
    await _send_to_admin(admin, text)


async def _send_to_admin(admin: "User", text: str) -> None:
    from boxbot.communication.channels import Channel, get_outbound_channel

    try:
        out = get_outbound_channel(Channel(admin.channel))
    except ValueError:
        out = None
    if out is None:
        logger.warning("Cannot reach admin %s on %r", admin.phone, admin.channel)
        return
    try:
        await out.send_text(admin.phone, text)
    except Exception:  # noqa: BLE001
        logger.exception("purchase notice to %s failed", admin.phone)
