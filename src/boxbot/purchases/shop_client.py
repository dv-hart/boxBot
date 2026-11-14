"""Shop agent API client (main process only).

Thin async wrapper over ``https://mcp.shop.com/`` — stateless JSON-RPC
``tools/call`` over HTTPS. Holds the Shop access token (via
:mod:`shop_auth`) and refreshes it; nothing here is reachable from the
sandbox except through the ``purchases.*`` action handler, which only
ever returns the sanitised shapes built by the ``summarize_*`` helpers.

Two rules from Shop's transaction docs are enforced structurally:

- ``complete_checkout`` is sent **at most once** per checkout. It is
  never retried here, not even on an auth failure; after an uncertain
  outcome the caller reads ``get_checkout`` instead.
- The payment instrument is copied **exactly** as Shop offered it, with
  only ``selected: true`` added.

Reference: https://shop.com/references/transactions.md
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import unicodedata
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx

from boxbot.purchases import shop_auth

logger = logging.getLogger(__name__)

ENDPOINT = "https://mcp.shop.com/"
PROTOCOL_VERSION = "2026-07-28"
_META = {
    "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
    "io.modelcontextprotocol/clientInfo": {"name": "boxbot", "version": "1.0"},
    "io.modelcontextprotocol/clientCapabilities": {},
}
_TIMEOUT = 40.0

# The only instrument type boxBot will ever complete with: the spending
# budget the account owner delegated to this connection in Shop. A saved
# card is offered too, but paying with it would bypass that budget.
BUDGET_INSTRUMENT_TYPE = "delegated_budget"

_HOST_RE = re.compile(
    r"^(?=.{4,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}$"
)
_VARIANT_GID = "gid://shopify/ProductVariant/"
_HANDLE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,200}$")
_MAX_FEED_BYTES = 8 * 1024 * 1024
_MAX_TEXT = 300


class ShopError(RuntimeError):
    """The Shop API refused or failed a call.

    ``refused`` is True only when Shop answered with an explicit,
    well-formed refusal — an auth rejection, a JSON-RPC error, or a
    tool-level error. A gateway failure, a malformed or truncated body,
    or anything else that leaves "did Shop act on this?" open is
    ``refused=False``. The distinction decides what a failed payment
    request is recorded as, so it defaults to the cautious answer.
    """

    def __init__(
        self, message: str, *, code: str = "shop_error", refused: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.refused = refused


class ShopNotConnected(ShopError):
    """No usable Shop sign-in; a human must run the device flow."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            "Shop account is not connected"
            + (f" ({detail})" if detail else ""),
            code="not_connected",
        )


def normalize_shop(shop: str) -> str:
    """Return a bare lowercase hostname, or raise ``ValueError``.

    Accepts ``example.com`` or a full ``https://example.com/…`` URL.
    Rejects IP literals, ports, userinfo, and anything that is not a
    plain public-looking DNS name.
    """
    if not isinstance(shop, str) or not shop.strip():
        raise ValueError("shop is required")
    raw = shop.strip().lower()
    if "://" in raw:
        parsed = urlparse(raw)
        if parsed.scheme != "https" or parsed.port or parsed.username:
            raise ValueError(f"invalid shop {shop!r}")
        raw = parsed.hostname or ""
    if not _HOST_RE.match(raw):
        raise ValueError(f"invalid shop {shop!r} — expected a store domain")
    return raw


def normalize_variant_id(variant_id: Any) -> str:
    """Return ``gid://shopify/ProductVariant/<digits>`` or raise."""
    text = str(variant_id).strip()
    if text.startswith(_VARIANT_GID):
        text = text[len(_VARIANT_GID):]
    # isdigit() alone admits full-width and superscript digits.
    if not (text.isascii() and text.isdecimal()) or len(text) > 20:
        raise ValueError(f"invalid variant id {variant_id!r}")
    return _VARIANT_GID + text


def _clip(value: Any, limit: int = _MAX_TEXT) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _product_url(url: Any, shop: str) -> str | None:
    """A product page URL, only if it is https on the offer's own store."""
    if not isinstance(url, str) or len(url) > 600:
        return None
    if not url.isascii() or any(ch.isspace() or ord(ch) < 0x21 for ch in url):
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme != "https" or (parsed.hostname or "").lower() != shop:
        return None
    if parsed.port or parsed.username:
        return None
    return url


# Anything a phone would turn into a tap target. \w is Unicode-aware, so
# non-ASCII (IDN) hosts are covered too.
#
# Every alternative is anchored with a lookbehind on its own character
# class, so a long run of such characters offers ONE starting point,
# not one per character — matching stays linear on hostile input.
URL_RE = re.compile(
    r"(?i)"
    # scheme://…
    r"(?<![a-z0-9+.-])[a-z][a-z0-9+.-]*://\S+"
    # www.…
    r"|(?<![\w.-])www\.\S+"
    # tel:/sms:-style links (need a number right after the colon)
    r"|(?<![\w.-])(?:tel|sms|smsto|wa|tg|geo):\+?\d\S*"
    # bare host.tld/path — the last label must be letters, so "4.5/5"
    # and "size 8.5/10" are left alone
    r"|(?<![\w.-])[\w-]+(?:\.[\w-]+)*\.[^\W\d_]{2,}/\S*"
)
# A dot that could still make a messenger auto-link a bare domain or an
# e-mail address ("evil.example", "пример.рф", "a@b.co"). "6.99", "v1.2"
# and "No. 5" are untouched.
_DOT_RE = re.compile(r"(?<=\w)\.(?=[^\W\d_]{2,})")
# Phone-number-shaped runs (nine or more digits), which phones make
# tappable.
_PHONE_RE = re.compile(r"\+?\d(?:[\s().-]*\d){8,}")
# Blank-looking characters that are not whitespace to str.split().
_BLANKS = dict.fromkeys(map(ord, "\u2800\u3164\u115f\u1160\uffa0\u180e"), " ")
_QUOTES = dict.fromkeys(map(ord, "\u201c\u201d\u201e\u201f\u00ab\u00bb\u2033\uff02\""), "'")
# Glyphs the service uses to mark its own notices.
_MARKS = dict.fromkeys(map(ord, "\U0001f6d2\u2705\u26a0\ufe0f\u2714\u2611"), None)


def plain_text(value: Any, limit: int = _MAX_TEXT) -> str:
    """Third-party text made safe to embed in a message or tool reply.

    Used for merchant titles, Shop error text, user display names and
    the agent's stated reason wherever they are shown to a person or
    handed back to the agent. The result:

    - is a single line (every Unicode line break and blank collapses to
      one space, so it cannot start a line of its own);
    - carries no invisible or direction-changing characters;
    - contains nothing a phone will turn into a link (links and phone
      numbers are removed, remaining ``name.tld`` dots are written
      ``[.]``);
    - contains no double quotes (callers quote it) and none of the
      glyphs that mark a genuine boxBot notice;
    - is clipped to ``limit``.
    """
    # Bound the work before any scanning: nothing past this could
    # survive the final clip anyway.
    text = str(value or "")[: max(16 * limit, 2048)]
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_BLANKS).translate(_QUOTES).translate(_MARKS)
    text = "".join(
        " " if unicodedata.category(ch) in ("Zl", "Zp", "Zs", "Cc") else ch
        for ch in text
        if unicodedata.category(ch) not in ("Cf", "Co", "Cn", "Cs")
    )
    text = URL_RE.sub("[link removed]", text)
    text = _DOT_RE.sub("[.]", text)
    text = _PHONE_RE.sub("[number removed]", text)
    return _clip(text, limit)


def _amount(totals: Any, kind: str) -> int | None:
    for entry in totals or []:
        if isinstance(entry, dict) and entry.get("type") == kind:
            amount = entry.get("amount")
            return amount if isinstance(amount, int) else None
    return None


# ---------------------------------------------------------------------------
# Sanitised views — the only shapes that may leave the main process
# ---------------------------------------------------------------------------


def summarize_search(result: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    """Flatten a ``search_catalog`` result to one row per offer.

    Every field is type-checked or sanitised: titles and descriptions
    are merchant-written text, and the rest must be the id, number or
    code it claims to be. Offers that are not well-formed are dropped.
    """
    rows: list[dict[str, Any]] = []
    for product in _dicts(result.get("products")):
        rating = _dict(product.get("rating"))
        description = product.get("description")
        if isinstance(description, dict):
            description = description.get("plain")
        for variant in _dicts(product.get("variants")):
            seller = _dict(variant.get("seller"))
            price = _dict(variant.get("price"))
            try:
                shop = normalize_shop(seller.get("url") or "")
                variant_id = normalize_variant_id(variant.get("id"))
            except ValueError:
                continue
            rows.append({
                "title": plain_text(variant.get("title") or product.get("title")),
                "description": plain_text(description),
                "variant_id": variant_id,
                "shop": shop,
                "seller": plain_text(seller.get("name"), 80),
                "price_cents": _int(price.get("amount")),
                "currency": _currency(price.get("currency")),
                "available": _dict(variant.get("availability")).get("available") is True,
                "url": _product_url(variant.get("url"), shop),
                "rating": _number(rating.get("value")),
                "rating_count": _int(rating.get("count")),
            })
            if len(rows) >= limit:
                return rows
    return rows


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _currency(value: Any) -> str | None:
    if isinstance(value, str) and len(value) == 3 and value.isascii() and value.isalpha():
        return value.upper()
    return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def summarize_checkout(checkout: dict[str, Any]) -> dict[str, Any]:
    """Reduce a checkout to the facts rules, the checker, and the agent use.

    Deliberately omits ``continue_url``, buyer contact details, the
    delivery address, and every payment credential.
    """
    lines = []
    for line in _dicts(checkout.get("line_items")):
        item = line.get("item") if isinstance(line.get("item"), dict) else {}
        lines.append({
            "variant_id": item.get("id") if isinstance(item.get("id"), str) else None,
            "title": plain_text(item.get("title"), 200),
            "unit_cents": _int(item.get("price")),
            "quantity": _int(line.get("quantity")),
        })
    instruments = _dicts(_dict(checkout.get("payment")).get("instruments"))
    selected = [
        i for i in instruments
        if isinstance(i, dict) and i.get("selected") is True
    ]
    destinations = []
    for method in _dicts(_dict(checkout.get("fulfillment")).get("methods")):
        for dest in _dicts(method.get("destinations")):
            destinations.append({
                "city": dest.get("address_locality"),
                "region": dest.get("address_region"),
                "country": dest.get("address_country"),
            })
    totals = checkout.get("totals")
    return {
        "checkout_id": checkout.get("id"),
        "status": checkout.get("status") if isinstance(checkout.get("status"), str) else None,
        "currency": _currency(checkout.get("currency")),
        "total_cents": _amount(totals, "total"),
        "subtotal_cents": _amount(totals, "subtotal"),
        "shipping_cents": _amount(totals, "fulfillment"),
        "tax_cents": _amount(totals, "tax"),
        "lines": lines,
        "item_count": sum(
            l["quantity"] for l in lines if isinstance(l["quantity"], int)
        ),
        "selected_instrument_type": (
            selected[0].get("type") if len(selected) == 1 else None
        ),
        "budget_remaining_cents": (
            _dict(selected[0].get("display")).get("remaining_amount")
            if len(selected) == 1 else None
        ),
        "destinations": destinations,
        "messages": [
            {
                "type": plain_text(m.get("type"), 40),
                "code": plain_text(m.get("code"), 80),
                "severity": plain_text(m.get("severity"), 40),
                "content": plain_text(m.get("content")),
            }
            for m in _dicts(checkout.get("messages"))
        ],
        "expires_at": checkout.get("expires_at"),
    }


def budget_instrument(checkout: dict[str, Any]) -> dict[str, Any] | None:
    """The delegated-budget instrument Shop pre-selected, copied exactly.

    Returns a deep copy with ``selected: true`` (the only permitted
    change), or None when Shop did not pre-select exactly one
    delegated-budget instrument.
    """
    instruments = _dicts(_dict(checkout.get("payment")).get("instruments"))
    selected = [i for i in instruments if i.get("selected") is True]
    if len(selected) != 1 or selected[0].get("type") != BUDGET_INSTRUMENT_TYPE:
        return None
    instrument = copy.deepcopy(selected[0])
    instrument["selected"] = True
    return instrument


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ShopClient:
    """Async client for the Shop agent API."""

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport
        self._refresh_lock = asyncio.Lock()

    async def _ensure_token(self, *, force: bool = False) -> tuple[str, str]:
        """Return a usable Authorization header, refreshing if needed."""
        async with self._refresh_lock:
            if force or not shop_auth.token_is_fresh():
                if not shop_auth.is_connected():
                    raise ShopNotConnected("never signed in")
                ok, detail = await asyncio.to_thread(shop_auth.refresh)
                if not ok:
                    logger.warning("Shop token refresh failed: %s", detail)
                    if detail.startswith("network"):
                        # The sign-in may be fine; Shop was unreachable.
                        raise ShopError(
                            "could not reach Shop to refresh sign-in",
                            code="network",
                        )
                    raise ShopNotConnected(detail)
            header = shop_auth.auth_header()
        if header is None:
            raise ShopNotConnected("no access token on disk")
        return header

    async def _post(self, name: str, arguments: dict[str, Any],
                    header: tuple[str, str] | None) -> httpx.Response:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
            "Mcp-Method": "tools/call",
            "Mcp-Name": name,
        }
        if header is not None:
            headers[header[0]] = header[1]
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments, "_meta": _META},
        }
        async with httpx.AsyncClient(
            timeout=_TIMEOUT, transport=self._transport, follow_redirects=False,
        ) as http:
            return await http.post(ENDPOINT, json=body, headers=headers)

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        auth: bool = True,
        retry_auth: bool = True,
    ) -> dict[str, Any]:
        """One ``tools/call``; returns the tool's structured content.

        Raises :class:`ShopNotConnected` when sign-in is missing or
        dead, :class:`ShopError` for any other refusal. A 401 triggers
        one forced refresh and one resend unless ``retry_auth`` is
        False (a 401 means Shop did not act on the request).
        """
        header = await self._ensure_token() if auth else None
        response = await self._post(name, arguments, header)
        if response.status_code == 401 and auth:
            if not retry_auth:
                raise ShopError(
                    "Shop rejected the access token", code="unauthorized",
                    refused=True,
                )
            header = await self._ensure_token(force=True)
            response = await self._post(name, arguments, header)
            if response.status_code == 401:
                raise ShopNotConnected("token rejected after refresh")
        if response.status_code >= 400:
            raise ShopError(
                f"Shop API HTTP {response.status_code}", code=f"http_{response.status_code}"
            )
        envelope = _parse_envelope(response.text)
        if envelope is None:
            raise ShopError("Shop API returned an unreadable body", code="bad_response")
        if "error" in envelope:
            err = _dict(envelope["error"])
            raise ShopError(
                plain_text(err.get("message") or "Shop API error"),
                code=f"rpc_{err.get('code', 'error')}",
                refused=True,
            )
        result = _dict(envelope.get("result"))
        content = result.get("structuredContent")
        if not isinstance(content, dict):
            raise ShopError("Shop API returned no structured content", code="bad_response")
        if result.get("isError"):
            detail = "; ".join(
                plain_text(m.get("content"), 160)
                for m in _dicts(content.get("messages"))
            )
            raise ShopError(
                detail or f"{name} failed", code="tool_error", refused=True,
            )
        return content

    # -- catalogue ---------------------------------------------------------

    async def search(
        self, query: str, *, max_price_cents: int | None = None, limit: int = 8,
    ) -> list[dict[str, Any]]:
        catalog: dict[str, Any] = {
            "query": query,
            "context": {"address_country": "US", "currency": "USD"},
            "pagination": {"limit": limit},
        }
        if max_price_cents is not None:
            catalog["filters"] = {"price": {"max": max_price_cents}}
        result = await self.call("search_catalog", {"catalog": catalog})
        return summarize_search(result, limit)

    # -- checkout ----------------------------------------------------------

    async def create_checkout(
        self, shop: str, items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a checkout holding exactly ``items``.

        ``items`` are ``{"variant_id": gid, "quantity": int}``. Nothing
        else is sent: Shop fills the buyer, delivery address, and
        payment from the signed-in account's saved defaults, so no
        caller-supplied address or contact detail can reach a checkout.
        """
        return await self.call("create_checkout", {
            "shop": shop,
            "checkout": {
                "line_items": [
                    {"item": {"id": i["variant_id"]}, "quantity": i["quantity"]}
                    for i in items
                ],
            },
        })

    async def get_checkout(self, shop: str, checkout_id: str) -> dict[str, Any]:
        return await self.call("get_checkout", {"shop": shop, "id": checkout_id})

    async def complete_checkout(
        self,
        shop: str,
        checkout_id: str,
        instrument: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Send payment. Call exactly once per checkout; never retried."""
        return await self.call(
            "complete_checkout",
            {
                "shop": shop,
                "id": checkout_id,
                "checkout": {"payment": {"instruments": [instrument]}},
                "meta": {"idempotency-key": idempotency_key},
            },
            retry_auth=False,
        )

    async def get_account(self) -> dict[str, Any]:
        return await self.call("get_account", {})


def _parse_envelope(raw: str) -> dict[str, Any] | None:
    """The JSON-RPC response object from a plain or SSE body, or None.

    An SSE body may carry several ``data:`` events (progress notices
    before the response); the response is the one with a ``result`` or
    ``error`` member.
    """
    text = raw.strip()
    candidates: list[str]
    if text.startswith(("event:", "data:", ":")):
        candidates = []
        current: list[str] = []
        for line in text.splitlines():
            if line.startswith("data:"):
                current.append(line[5:].lstrip())
            elif not line.strip() and current:
                candidates.append("\n".join(current))
                current = []
        if current:
            candidates.append("\n".join(current))
    else:
        candidates = [text]
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict) and ("result" in parsed or "error" in parsed):
            return parsed
    return None


def idempotency_key_for(purchase_id: str) -> str:
    """Stable UUID for one purchase's single completion attempt."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"boxbot:purchase:{purchase_id}"))


# ---------------------------------------------------------------------------
# Store product feeds (for allow-listed stores Shop search does not cover)
# ---------------------------------------------------------------------------


async def fetch_store_products(
    shop: str,
    *,
    query: str | None = None,
    limit: int = 20,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[dict[str, Any]]:
    """Read ``https://<shop>/products.json`` and return matching variants.

    The caller must have checked ``shop`` against the configured
    allow-list first — this fetches a host named by the agent, so the
    allow-list is what keeps it from being pointed at arbitrary
    addresses. Redirects are not followed.
    """
    host = normalize_shop(shop)
    terms = [t for t in (query or "").lower().split() if t]
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        timeout=20.0, transport=transport, follow_redirects=False,
    ) as http:
        for page in range(1, 5):
            body = bytearray()
            async with http.stream(
                "GET", f"https://{host}/products.json",
                params={"limit": 250, "page": page},
            ) as response:
                if response.status_code != 200:
                    raise ShopError(
                        f"{host} product feed HTTP {response.status_code}",
                        code="feed_unavailable",
                    )
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > _MAX_FEED_BYTES:
                        raise ShopError(
                            f"{host} product feed is too large",
                            code="feed_unavailable",
                        )
            try:
                products = json.loads(body).get("products") or []
            except (ValueError, AttributeError) as e:
                raise ShopError(
                    f"{host} product feed is not JSON", code="feed_unavailable"
                ) from e
            if not isinstance(products, list) or not products:
                break
            for product in products:
                if not isinstance(product, dict):
                    continue
                haystack = " ".join(
                    str(product.get(k) or "")
                    for k in ("title", "product_type", "vendor")
                ).lower() + " " + " ".join(
                    str(t) for t in (product.get("tags") or [])
                ).lower()
                if terms and not all(t in haystack for t in terms):
                    continue
                for variant in product.get("variants") or []:
                    if not isinstance(variant, dict) or not variant.get("available"):
                        continue
                    try:
                        cents = int(round(float(variant.get("price")) * 100))
                        variant_id = normalize_variant_id(variant.get("id"))
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if not 0 < cents < 10_000_000:
                        continue
                    title = str(product.get("title") or "")
                    if variant.get("title") not in (None, "", "Default Title"):
                        title = f"{title} — {variant['title']}"
                    rows.append({
                        "title": plain_text(title),
                        "description": plain_text(product.get("product_type"), 80),
                        "variant_id": variant_id,
                        "shop": host,
                        "seller": plain_text(product.get("vendor"), 80),
                        "price_cents": cents,
                        # The feed does not state its currency; the
                        # checkout total is authoritative.
                        "currency": None,
                        "available": True,
                        "url": _product_url(
                            f"https://{host}/products/{product.get('handle')}", host,
                        ) if _HANDLE_RE.match(str(product.get("handle") or "")) else None,
                        "rating": None,
                        "rating_count": None,
                    })
                    if len(rows) >= limit:
                        return rows
    return rows
