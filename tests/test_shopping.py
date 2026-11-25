"""Tests for the find → cart → send shopping flow.

Nothing here touches the network: the Shop client is a fake, product
images come from ``httpx.MockTransport``, the screen is a recording
stub, and outbound messaging is the recording stub from
``test_purchases``.

Covers:
- slim rows, stable ids, approved-first ordering, variant labels
- the per-conversation result cache: isolation, expiry, caps
- the ``sdk_actions`` log no longer echoing response payloads
- product cards (1 / 2 / 4) and the cart screen, from ids only
- the bounded product-image fetch
- one definition of "approved"; ``stores()``
- cart operations; quotes (no record, no text, reused, approved only)
- the cost breakdown persisted and shown
- ``send`` per store, supersede, re-send; ``cancel`` default + notice
- the guidance files name only calls that exist
"""

from __future__ import annotations

import copy
import functools
import io
import json
import logging
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from PIL import Image

from boxbot.purchases import checker, images, service, shopping
from boxbot.purchases.shop_client import (
    checkout_cost_shape,
    selected_delivery,
    summarize_checkout,
    summarize_search,
)
from boxbot.purchases.store import PurchaseStore
from boxbot.tools._sandbox_actions import ActionContext, process_action
from test_purchases import (  # noqa: F401 — fixtures used by name
    ADMIN,
    SHOP,
    FakeShop,
    _isolated_token_dir,
    _no_real_services,
    admins,
    allow,
    conv_from,
    deny,
    env,
    outbound,
    person,
    purchases_cfg,
    shop,
    store,
    texts,
)

OTHER = "toys.example.org"          # approved by an admin's text in some tests
STRANGER = "far.example.net"        # never approved
REPO = Path(__file__).parent.parent


def gid(n):
    return f"gid://shopify/ProductVariant/{n}"


def offer(n, title="Titanic", shop=SHOP, price=699, seller="Example Books", **kw):
    row = {
        "title": title, "product_title": title, "variant_id": gid(n),
        "shop": shop, "seller": seller, "price_cents": price,
        "currency": "USD", "available": True,
        "description": "<b>long html</b> " * 200,
        "url": f"https://{shop}/products/p{n}?variant={n}&utm_source=x&_gsid=abc",
        "image_url": None, "options": [], "handle": None,
        "rating": 4.8, "rating_count": 12,
    }
    row.update(kw)
    return row


class Catalogue(FakeShop):
    """FakeShop with a settable catalogue and a checkout per cart."""

    def __init__(self):
        super().__init__()
        self.offers = [offer(1)]
        self.searches = []
        self.shipping = {SHOP: 599}
        self.tax = 0
        self.delivery = None
        self.prices = {}

    async def search(self, query, *, max_price_cents=None, limit=8):
        self.searches.append((query, limit))
        return copy.deepcopy(self.offers)

    async def create_checkout(self, shop, items):
        if self.create_error:
            raise self.create_error
        self.created.append((shop, items))
        known = {o["variant_id"]: o for o in self.offers}
        lines, subtotal = [], 0
        for item in items:
            price = self.prices.get(
                item["variant_id"],
                known.get(item["variant_id"], {}).get("price_cents", 500),
            )
            subtotal += price * item["quantity"]
            lines.append({
                "id": f"l{len(lines)}",
                "item": {"id": item["variant_id"], "title": "Titanic", "price": price},
                "quantity": item["quantity"],
            })
        shipping = self.shipping.get(shop, 0)
        n = len(self.created)
        fulfillment = {"methods": [{
            "destinations": [{
                "street_address": "1 Secret Lane", "address_locality": "Hillsboro",
                "address_region": "OR", "address_country": "US",
            }],
        }]}
        if self.delivery:
            fulfillment["methods"][0]["groups"] = [self.delivery]
        checkout = copy.deepcopy(self.checkout)
        checkout.update({
            "id": f"chk_{n}", "line_items": lines,
            "totals": [
                {"type": "subtotal", "amount": subtotal},
                {"type": "fulfillment", "amount": shipping},
                {"type": "tax", "amount": self.tax},
                {"type": "total", "amount": subtotal + shipping + self.tax},
            ],
            "fulfillment": fulfillment,
            "continue_url": f"https://{shop}/checkouts/secret-{n}",
        })
        return checkout


class FakeScreen:
    def __init__(self):
        self.switches = []
        self.active = None
        self.args = {}
        self.unpinned = 0

    async def switch(self, name, args=None, pin=True):
        self.switches.append((name, dict(args or {})))
        self.active, self.args = name, dict(args or {})
        return True

    def get_active(self):
        return self.active

    def get_active_args(self):
        return self.args

    async def unpin(self):
        self.unpinned += 1
        self.active = None
        return True


def room(conversation_id="conv-1"):
    """A voice conversation at the box."""
    conv = conv_from(phone=None, channel="voice",
                     log=[person("please get the Titanic book")])
    conv.conversation_id = conversation_id
    return conv


@pytest.fixture
async def world(env, monkeypatch, tmp_path):
    """Purchasing on, a catalogue, a screen, images off the network."""
    from boxbot.displays import manager as display_manager

    catalogue = Catalogue()
    service.set_shop_client(catalogue)
    screen = FakeScreen()
    display_manager.set_display_manager(screen)
    monkeypatch.setattr(images, "cache_dir", lambda: tmp_path / "product-images")
    monkeypatch.setattr(images, "_transport", httpx.MockTransport(
        lambda request: httpx.Response(404)
    ))
    images._failed.clear()
    shopping.reset()
    clock = [1000.0]
    monkeypatch.setattr(shopping, "_now", lambda: clock[0])
    yield SimpleNamespace(
        cfg=env.cfg, store=env.store, out=env.out, shop=catalogue,
        screen=screen, clock=clock, conv=room(),
    )
    for task in list(shopping._tasks):
        task.cancel()
    display_manager.set_display_manager(None)
    shopping.reset()


async def approve(world, host=OTHER):
    await world.store.approve_shop(host, approved_by=ADMIN)


# ---------------------------------------------------------------------------
# A. Slim rows and the result cache
# ---------------------------------------------------------------------------


class TestFind:
    async def test_rows_are_slim_and_exact(self, world):
        world.shop.offers = [offer(1, title="T" * 200)]
        out = await shopping.find("titanic", conv=world.conv)
        (row,) = out["results"]
        assert list(row) == ["id", "title", "variant", "price", "store",
                             "approved", "in_stock"]
        assert row["id"] == "a1" and row["price"] == "$6.99"
        assert row["store"] == "Example Books" and row["approved"] is True
        assert len(row["title"]) <= 60 and row["variant"] is None
        blob = json.dumps(out)
        for leaked in ("gid://", "html", "utm_source", "https://", SHOP):
            assert leaked not in blob
        assert len(blob) < 400

    async def test_ids_are_search_letter_plus_card_number(self, world):
        world.shop.offers = [offer(n, title=f"Book {n}") for n in range(1, 4)]
        first = await shopping.find("one", conv=world.conv)
        second = await shopping.find("two", conv=world.conv)
        assert [r["id"] for r in first["results"]] == ["a1", "a2", "a3"]
        assert [r["id"] for r in second["results"]] == ["b1", "b2", "b3"]
        # Earlier ids still resolve: no need to search again.
        view = await shopping.cart(add=["a2", "b3"], conv=world.conv)
        assert [l["title"] for l in view["lines"]] == ["Book 2", "Book 3"]
        assert len(world.shop.searches) == 2

    async def test_at_most_eight_rows(self, world):
        world.shop.offers = [offer(n, title=f"Book {n}") for n in range(1, 16)]
        out = await shopping.find("books", conv=world.conv)
        assert [r["id"] for r in out["results"]] == [f"a{n}" for n in range(1, 9)]

    async def test_approved_stores_first_and_unapproved_only_as_fill(self, world):
        world.shop.offers = [
            offer(1, shop=STRANGER, seller="Far Away"),
            offer(2, title="A"), offer(3, shop=STRANGER, seller="Far Away"),
            offer(4, title="B"),
        ]
        out = await shopping.find("x", conv=world.conv)
        assert [r["approved"] for r in out["results"]] == [True, True, False, False]
        # Enough approved offers (the documented minimum, 3): no others at all.
        world.shop.offers.append(offer(5, title="C"))
        out = await shopping.find("x", conv=world.conv)
        assert [r["approved"] for r in out["results"]] == [True] * shopping.MIN_APPROVED

    async def test_approved_only(self, world):
        world.shop.offers = [offer(1, shop=STRANGER), offer(2)]
        out = await shopping.find("x", approved_only=True, conv=world.conv)
        assert [r["approved"] for r in out["results"]] == [True]
        world.shop.offers = [offer(1, shop=STRANGER)]
        assert (await shopping.find("x", approved_only=True, conv=world.conv))[
            "results"] == []

    async def test_admin_text_approval_counts_as_approved(self, world):
        world.shop.offers = [offer(1, shop=OTHER, seller="Toy Shop")]
        assert (await shopping.find("x", conv=world.conv))["results"][0][
            "approved"] is False
        await approve(world)
        assert (await shopping.find("x", conv=world.conv))["results"][0][
            "approved"] is True

    async def test_non_usd_unavailable_and_overpriced_are_dropped(self, world):
        world.shop.offers = [
            offer(1, currency="GBP"), offer(2, available=False),
            offer(3, price_cents=None), offer(4, price=5000), offer(5, title="Keep"),
            offer(5, title="Duplicate variant"),
        ]
        out = await shopping.find("x", max_price_cents=2000, conv=world.conv)
        assert [r["title"] for r in out["results"]] == ["Keep"]

    async def test_same_title_rows_get_variant_labels_from_the_payload(self, world):
        world.shop.offers = [
            offer(1, options=["Paperback", "2010"]),
            offer(2, price=1099, options=["Hardcover", "2019"]),
        ]
        rows = (await shopping.find("x", conv=world.conv))["results"]
        assert [r["variant"] for r in rows] == ["Paperback · 2010", "Hardcover · 2019"]

    async def test_same_title_no_options_falls_back_to_isbn_then_variant_id(self, world):
        world.shop.offers = [
            offer(1, handle="9781546150251"), offer(2, price=1099, handle="9780545206945"),
        ]
        rows = (await shopping.find("x", conv=world.conv))["results"]
        assert [r["variant"] for r in rows] == ["ISBN ending 0251", "ISBN ending 6945"]
        world.shop.offers = [offer(46511459696881), offer(46511459691111, price=1099)]
        rows = (await shopping.find("x", conv=world.conv))["results"]
        assert rows[0]["variant"] != rows[1]["variant"]
        assert rows[1]["variant"] == "#1111"

    async def test_same_title_at_two_stores_needs_no_label(self, world):
        await approve(world)
        world.shop.offers = [offer(1), offer(2, shop=OTHER, seller="Toy Shop")]
        rows = (await shopping.find("x", conv=world.conv))["results"]
        assert [r["variant"] for r in rows] == [None, None]

    def test_catalogue_extraction_keeps_what_tells_offers_apart(self):
        rows = summarize_search({"products": [{
            "title": "Titanic", "handle": "titanic-special",
            "variants": [
                {"id": gid(1), "title": "Titanic",
                 "selected_options": [{"name": "Format", "value": "Hardcover"},
                                      {"name": "Title", "value": "Default Title"}],
                 "price": {"amount": 1099, "currency": "USD"},
                 "seller": {"name": "B", "url": "https://books.example.com"}},
                {"id": gid(2), "title": "Paperback",
                 "url": "https://books.example.com/products/9781546150251?variant=2",
                 "price": {"amount": 699, "currency": "USD"},
                 "seller": {"name": "B", "url": "https://books.example.com"}},
            ],
        }]}, 8)
        assert rows[0]["options"] == ["Hardcover"]
        assert rows[0]["handle"] == "titanic-special"
        assert rows[1]["options"] == ["Paperback"]
        assert rows[0]["product_title"] == "Titanic"
        solo = summarize_search({"products": [{"variants": [{
            "id": gid(3), "title": "X" * 80, "options": "nonsense",
            "url": "https://books.example.com/products/9781546150251?variant=3",
            "seller": {"url": "https://books.example.com"},
        }]}]}, 8)
        assert solo[0]["options"] == [] and solo[0]["handle"] == "9781546150251"

    async def test_bad_arguments_are_refused_before_a_search(self, world):
        for bad in (
            dict(query=""), dict(query=7), dict(query="x", max_price_cents=-1),
            dict(query="x", max_price_cents="12"), dict(query="x", approved_only="yes"),
            dict(query="x", show=1),
        ):
            with pytest.raises(service.PurchaseError):
                await shopping.find(conv=world.conv, **bad)
        assert world.shop.searches == []

    async def test_hourly_search_ceiling_still_applies(self, world):
        for _ in range(service._search_rate.limit):
            await shopping.find("x", conv=world.conv)
        with pytest.raises(service.PurchaseError, match="too many searches"):
            await shopping.find("x", conv=world.conv)


class TestCache:
    async def test_ids_do_not_cross_conversations(self, world):
        await shopping.find("x", conv=world.conv)
        other = room("conv-2")
        with pytest.raises(service.PurchaseError):
            await shopping.cart(add=["a1"], conv=other)
        with pytest.raises(service.PurchaseError):
            await shopping.show(["a1"], conv=other)
        # …and the other conversation's own a1 is its own row.
        world.shop.offers = [offer(9, title="Other book")]
        await shopping.find("y", conv=other)
        mine = await shopping.cart(add=["a1"], conv=world.conv)
        theirs = await shopping.cart(add=["a1"], conv=other)
        assert mine["lines"][0]["title"] == "Titanic"
        assert theirs["lines"][0]["title"] == "Other book"

    async def test_the_session_key_never_comes_from_the_payload(self, world):
        from boxbot.tools._tool_context import current_conversation

        await shopping.find("x", conv=world.conv)
        token = current_conversation.set(room("conv-2"))
        try:
            out = await process_action({
                "_sdk": "purchases.cart", "add": ["a1"],
                "conversation_id": "conv-1", "conv": "conv-1",
                "channel_key": "voice:room",
            }, ActionContext())
        finally:
            current_conversation.reset(token)
        assert out["status"] == "error" and "a1" in out["message"]

    async def test_no_conversation_no_session(self, world):
        with pytest.raises(service.PurchaseError, match="needs a conversation"):
            await shopping.find("x", conv=None)

    async def test_idle_sessions_expire(self, world):
        await shopping.find("x", conv=world.conv)
        await shopping.cart(add=["a1"], conv=world.conv)
        world.clock[0] += shopping.SESSION_TTL - 5
        await shopping.cart(conv=world.conv)                 # use keeps it alive
        world.clock[0] += shopping.SESSION_TTL - 5
        assert (await shopping.cart(conv=world.conv))["lines"]
        world.clock[0] += shopping.SESSION_TTL + 1
        assert (await shopping.cart(conv=world.conv))["lines"] == []
        with pytest.raises(service.PurchaseError, match="find\\(\\) again"):
            await shopping.cart(add=["a1"], conv=world.conv)

    async def test_sessions_and_rows_are_capped(self, world):
        for n in range(shopping._MAX_SESSIONS + 5):
            await shopping.find("x", conv=room(f"c{n}"))
        assert len(shopping._sessions) == shopping._MAX_SESSIONS
        service._search_rate._events.clear()
        world.shop.offers = [offer(n, title=f"B{n}") for n in range(1, 9)]
        await shopping.find("x", conv=world.conv)
        await shopping.cart(add=["a1"], conv=world.conv)
        for _ in range(30):
            await shopping.find("x", conv=world.conv)
        session = shopping._sessions["conv-1"]
        assert len(session.rows) <= shopping._MAX_ROWS
        assert "a1" in session.rows                          # in the cart: kept

    async def test_session_is_dropped_when_the_conversation_ends(self, world, event_bus):
        from boxbot.core.events import ConversationEnded

        shopping._subscribed = False
        shopping.start()
        await shopping.find("x", conv=world.conv)
        await event_bus.publish(ConversationEnded(conversation_id="conv-1"))
        assert "conv-1" not in shopping._sessions
        shopping._subscribed = False


# ---------------------------------------------------------------------------
# B. The action log does not repeat response payloads
# ---------------------------------------------------------------------------


class TestActionLog:
    async def dispatch(self, world, action, ctx, **payload):
        from boxbot.tools._tool_context import current_conversation

        token = current_conversation.set(world.conv)
        try:
            return await process_action({"_sdk": action, **payload}, ctx)
        finally:
            current_conversation.reset(token)

    async def test_a_replied_action_is_logged_as_a_summary(self, world):
        world.shop.offers = [offer(n, title=f"Book {n}") for n in range(1, 9)]
        ctx = ActionContext()
        reply = await self.dispatch(
            world, "purchases.find", ctx, query="x", _expects_response=True,
        )
        assert len(reply["results"]) == 8                    # the script got it all
        (entry,) = ctx.action_log
        assert entry == {
            "action": "purchases.find", "status": "ok", "n_results": 8, "n_shown": 4,
        }
        assert "Book" not in json.dumps(ctx.action_log)

    async def test_errors_and_records_keep_what_identifies_them(self, world):
        ctx = ActionContext()
        await self.dispatch(world, "purchases.cart", ctx, add=["zz9"],
                            _expects_response=True)
        assert ctx.action_log[-1]["status"] == "error"
        assert "zz9" in ctx.action_log[-1]["message"]
        with patch.object(checker, "check", side_effect=deny):
            out = await self.dispatch(
                world, "purchases.propose", ctx, shop=SHOP,
                items=[{"variant_id": gid(1), "quantity": 1}], reason="why",
                _expects_response=True,
            )
        entry = ctx.action_log[-1]
        assert entry["purchase_id"] == out["purchase"]["id"]
        assert entry["decision"] == "review" and "purchase" not in entry

    async def test_a_fire_and_forget_action_is_logged_whole(self, world):
        ctx = ActionContext()
        await self.dispatch(world, "purchases.find", ctx, query="x")
        assert ctx.action_log[-1]["results"][0]["id"] == "a1"

    async def test_long_strings_are_dropped_short_scalars_kept(self):
        from boxbot.tools._sandbox_actions import _log_entry

        entry = _log_entry("x.read", {
            "status": "ok", "content": "c" * 5000, "merged_count": 3,
            "updated": True, "path": "notes/" + "p" * 300, "entries": [1, 2],
            "nothing": None,
        }, replied=True)
        assert entry == {
            "action": "x.read", "status": "ok", "merged_count": 3, "updated": True,
            "path": "notes/" + "p" * 300, "n_entries": 2, "nothing": None,
        }

    async def test_workspace_path_still_reaches_the_trigger_receipt(
        self, world, tmp_path, monkeypatch,
    ):
        """The one consumer of ``sdk_actions``: core/agent.py reads the
        workspace paths a run wrote."""
        import boxbot.workspace as workspace_module
        from boxbot.core.agent import _workspace_artifacts_from_thread

        monkeypatch.setattr(
            workspace_module, "Workspace",
            functools.partial(workspace_module.Workspace, root=tmp_path),
        )
        ctx = ActionContext()
        reply = await self.dispatch(
            world, "workspace.write", ctx, path="notes/review.md",
            content="x" * 4000, _expects_response=True,
        )
        assert reply["status"] == "ok"
        body = json.dumps({"status": "success", "sdk_actions": ctx.action_log})
        assert len(body) < 400
        thread = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": body},
        ]}]
        assert _workspace_artifacts_from_thread(thread) == ["notes/review.md"]


# ---------------------------------------------------------------------------
# C. Screens from ids; the bounded image fetch
# ---------------------------------------------------------------------------


def jpeg_bytes(size=(60, 80), exif=False):
    image = Image.new("RGB", size, (200, 30, 30))
    buffer = io.BytesIO()
    extra = {}
    if exif:
        data = Image.Exif()
        data[0x010F] = "SecretCamera"
        extra["exif"] = data.tobytes()
    image.save(buffer, format="JPEG", **extra)
    return buffer.getvalue()


GOOD_IMAGE = "https://cdn.shopify.com/s/files/1/book.jpg"


class TestScreens:
    def payload(self, world, token=None):
        from boxbot.displays.shop_blocks import payload_for

        return payload_for(token or world.screen.args["view"])

    async def test_find_shows_its_top_four_automatically(self, world):
        world.shop.offers = [offer(n, title=f"Book {n}") for n in range(1, 7)]
        out = await shopping.find("x", conv=world.conv)
        assert out["shown"] == ["a1", "a2", "a3", "a4"]
        (name, args) = world.screen.switches[-1]
        assert name == "products" and list(args) == ["view"]
        cards = self.payload(world)["items"]
        assert [c["number"] for c in cards] == [1, 2, 3, 4]
        assert [c["title"] for c in cards] == [f"Book {n}" for n in range(1, 5)]

    async def test_the_real_display_manager_draws_both_screens(self, world):
        from boxbot.displays import manager as display_manager
        from boxbot.displays.themes import get_theme

        real = display_manager.DisplayManager()
        await real.load_displays()
        display_manager.set_display_manager(real)
        world.shop.offers = [offer(n, title=f"Book {n}") for n in range(1, 4)]
        await shopping.find("x", conv=world.conv)
        assert real.get_active() == "products" and real.is_pinned()
        accent = get_theme("boxbot").color_rgb("accent")
        assert accent in set(real.get_current_frame().getdata())
        await shopping.cart(add=["a2"], conv=world.conv)
        assert real.get_active() == "cart"
        assert accent in set(real.get_current_frame().getdata())
        # A model-typed switch to the same display draws no product.
        await real.switch("products", args={"title": "Typed", "price": "$0.01"})
        assert accent not in set(real.get_current_frame().getdata())

    async def test_show_false_and_text_threads_leave_the_screen_alone(self, world):
        await shopping.find("x", show=False, conv=world.conv)
        texting = conv_from()                                # a Signal thread
        out = await shopping.find("x", conv=texting)
        await shopping.cart(add=["a1"], conv=texting)
        assert out["shown"] == [] and world.screen.switches == []

    @pytest.mark.parametrize("count", [1, 2, 4])
    async def test_show_renders_one_card_or_equal_columns(self, world, count):
        from boxbot.displays.builtins import get_builtin_specs
        from boxbot.displays.renderer import render_to_image
        from boxbot.displays.spec import resolve_bindings
        from boxbot.displays.themes import get_theme

        world.shop.offers = [
            offer(n, title=f"I Survived the Sinking of the Titanic {n}",
                  options=["Hardcover"], shop=SHOP if n < 3 else STRANGER)
            for n in range(1, 5)
        ]
        await shopping.find("x", show=False, conv=world.conv)
        ids = [f"a{n}" for n in range(1, count + 1)]
        out = await shopping.show(ids, conv=world.conv)
        assert out["shown"] == ids
        assert out["numbers"] == {str(n): f"a{n}" for n in range(1, count + 1)}
        name, args = world.screen.switches[-1]
        assert name == "products"
        spec = next(s for s in get_builtin_specs() if s.name == "products")
        theme = get_theme("boxbot")
        frame = render_to_image(
            resolve_bindings(spec.root_block, {"args": args}), theme,
        )
        assert frame.size == (1024, 600)
        # One accent-filled number circle per card, laid out in one row:
        # sample the accent colour along the circles' row and count runs.
        accent = theme.color_rgb("accent")
        row = [frame.getpixel((x, 36 if count > 1 else 30)) for x in range(1024)]
        runs = sum(
            1 for x in range(1, 1024) if row[x] == accent and row[x - 1] != accent
        )
        assert runs == count
        # Nothing accent-coloured in the lower half's left edge column
        # band where a second grid row would put its circle.
        if count == 4:
            assert all(
                frame.getpixel((x, 330)) != accent for x in range(30, 110)
            )

    def test_text_sizes_are_readable_from_across_the_room(self):
        from boxbot.displays import shop_blocks

        assert shop_blocks.MIN_TEXT_PX >= 28
        assert shop_blocks.NUMBER_CIRCLE_PX >= 80
        assert shop_blocks.PRICE_PX >= 56 and shop_blocks.TITLE_PX >= 30
        source = Path(shop_blocks.__file__).read_text()
        # Every font goes through _font(), which clamps to the minimum.
        assert "_get_font(" not in source.replace(
            "return _get_font(ctx.theme.fonts.family, max(MIN_TEXT_PX, size), weight)",
            "",
        ).replace("    _get_font,\n", "")

    async def test_colliding_card_digits_are_renumbered(self, world):
        await shopping.find("x", show=False, conv=world.conv)
        await shopping.find("y", show=False, conv=world.conv)
        out = await shopping.show(["a1", "b1"], conv=world.conv)
        assert out["numbers"] == {"1": "a1", "2": "b1"}

    @pytest.mark.parametrize("ids", [
        [], ["a1"] * 2, ["a1", "a2", "a3", "a4", "a5"], ["A1"], ["a9"], ["a1; rm"],
        [7], "nope", {"a1": 1}, ["../../etc/passwd"],
    ])
    async def test_show_validates_ids(self, world, ids):
        world.shop.offers = [offer(n, title=f"B{n}") for n in range(1, 6)]
        await shopping.find("x", show=False, conv=world.conv)
        with pytest.raises(service.PurchaseError):
            await shopping.show(ids, conv=world.conv)
        assert world.screen.switches == []

    async def test_what_is_drawn_never_comes_from_display_args(self, world):
        """switch_display('products', args={...typed by the model...})
        draws nothing: the block reads only a published view token."""
        from boxbot.displays.shop_blocks import payload_for

        assert payload_for("made-up") is None
        assert payload_for({"items": [{"title": "Typed by the model"}]}) is None

    async def test_cart_screen_redraws_and_empty_cart_releases_the_screen(self, world):
        await shopping.find("x", show=False, conv=world.conv)
        await shopping.cart(add=["a1"], conv=world.conv)
        assert world.screen.switches[-1][0] == "cart"
        shown = self.payload(world)
        assert shown["lines"][0]["title"] == "Titanic"
        assert shown["total_label"] == "Before shipping" and shown["total"] == "$6.99"
        await shopping.cart(quote=True, conv=world.conv)
        shown = self.payload(world)
        assert shown["total_label"] == "Total" and shown["total"] == "$12.98"
        assert ["Shipping", "$5.99"] in shown["figures"]
        await shopping.cart(clear=True, conv=world.conv)
        assert world.screen.unpinned == 1

    async def test_cart_screen_renders(self, world):
        from boxbot.displays.shop_blocks import CartBlock, publish
        from boxbot.displays.renderer import render_to_image
        from boxbot.displays.themes import get_theme

        token = publish({
            "lines": [{"qty": 2, "title": "Titanic <b>bold</b>", "variant": "Hardcover",
                       "store": "Far Away", "approved": False, "price": "$21.98"}],
            "figures": [["Subtotal", "$21.98"], ["Shipping", "—"], ["Tax", "—"]],
            "total_label": "Before shipping", "total": "$21.98",
            "note": "Far Away needs an admin's OK", "hint": "hint",
        })
        frame = render_to_image(CartBlock(view=token), get_theme("boxbot"))
        assert frame.size == (1024, 600)
        blank = render_to_image(CartBlock(view="gone"), get_theme("boxbot"))
        assert frame.tobytes() != blank.tobytes()


class TestImages:
    @pytest.fixture(autouse=True)
    def _setup(self, purchases_cfg, tmp_path, monkeypatch):
        self.cfg = purchases_cfg
        self.dir = tmp_path / "product-images"
        monkeypatch.setattr(images, "cache_dir", lambda: self.dir)
        images._failed.clear()
        self.calls = []
        self.monkeypatch = monkeypatch

    def serve(self, handler):
        def recorded(request):
            self.calls.append(str(request.url))
            return handler(request)

        self.monkeypatch.setattr(images, "_transport", httpx.MockTransport(recorded))

    async def test_fetch_reencodes_and_strips_metadata(self):
        raw = jpeg_bytes((1600, 1200), exif=True)
        assert b"SecretCamera" in raw
        self.serve(lambda r: httpx.Response(
            200, content=raw, headers={"content-type": "image/jpeg"}))
        path = await images.fetch(GOOD_IMAGE)
        assert path is not None and path.parent == self.dir
        assert b"SecretCamera" not in path.read_bytes()
        with Image.open(path) as saved:
            assert saved.format == "JPEG" and max(saved.size) <= images.OUT_LONG_EDGE
            assert not saved.getexif()
        assert await images.fetch(GOOD_IMAGE) == path          # cached
        assert len(self.calls) == 1
        assert images.is_cached_image(path) == path.resolve()

    @pytest.mark.parametrize("url", [
        "https://evil.example.com/a.jpg",
        "http://cdn.shopify.com/a.jpg",
        "https://cdn.shopify.com:8443/a.jpg",
        "https://user@cdn.shopify.com/a.jpg",
        "https://cdn.shopify.com.evil.example/a.jpg",
        "https://sub.cdn.shopify.com/a.jpg",
        "https://127.0.0.1/a.jpg",
        "file:///etc/passwd",
        "https://cdn.shopify.com/a b.jpg",
        None, 7, "",
    ])
    async def test_only_allow_listed_https_hosts_are_ever_requested(self, url):
        self.serve(lambda r: httpx.Response(
            200, content=jpeg_bytes(), headers={"content-type": "image/jpeg"}))
        assert await images.fetch(url) is None
        assert self.calls == []

    async def test_redirect_off_the_allow_list_is_not_followed(self):
        self.serve(lambda r: httpx.Response(
            302, headers={"location": "https://169.254.169.254/latest/meta-data"}))
        assert await images.fetch(GOOD_IMAGE) is None
        assert self.calls == [GOOD_IMAGE]                       # one request, no hop

    async def test_redirect_within_the_allow_list_is_followed_a_bounded_number_of_times(self):
        def handler(request):
            if request.url.path == "/final.jpg":
                return httpx.Response(200, content=jpeg_bytes(),
                                      headers={"content-type": "image/jpeg"})
            return httpx.Response(
                301, headers={"location": "https://cdn.shopify.com/final.jpg"})

        self.serve(handler)
        assert await images.fetch(GOOD_IMAGE) is not None
        assert len(self.calls) == 2
        images._failed.clear()
        self.calls.clear()
        self.serve(lambda r: httpx.Response(
            302, headers={"location": "https://cdn.shopify.com/loop.jpg"}))
        assert await images.fetch("https://cdn.shopify.com/loop.jpg") is None
        assert len(self.calls) == images._MAX_REDIRECTS + 1

    async def test_relative_and_downgraded_redirects_are_dropped(self):
        for location in ("/other.jpg", "http://cdn.shopify.com/x.jpg", ""):
            images._failed.clear()
            self.calls.clear()
            self.serve(lambda r, loc=location: httpx.Response(
                302, headers={"location": loc}))
            assert await images.fetch(GOOD_IMAGE) is None
            assert len(self.calls) == 1

    async def test_oversized_bodies_are_abandoned(self):
        big = b"\xff\xd8" + b"0" * (images.MAX_BYTES + 10)
        # Declared too large: refused from the header.
        self.serve(lambda r: httpx.Response(
            200, content=big, headers={"content-type": "image/jpeg"}))
        assert await images.fetch(GOOD_IMAGE) is None
        # Undeclared (chunked): cut off while streaming.
        images._failed.clear()

        def chunks():
            for _ in range(images.MAX_BYTES // 65536 + 3):
                yield b"0" * 65536

        self.serve(lambda r: httpx.Response(
            200, content=chunks(), headers={"content-type": "image/png"}))
        assert await images.fetch(GOOD_IMAGE) is None
        assert not list(self.dir.glob("*")) if self.dir.exists() else True

    @pytest.mark.parametrize("kind", [
        "text/html", "image/svg+xml", "application/octet-stream", "image/gif", "",
    ])
    async def test_wrong_content_type_is_refused(self, kind):
        self.serve(lambda r: httpx.Response(
            200, content=jpeg_bytes(), headers={"content-type": kind}))
        assert await images.fetch(GOOD_IMAGE) is None

    async def test_bytes_that_are_not_the_image_they_claim_are_refused(self):
        self.serve(lambda r: httpx.Response(
            200, content=b"<svg onload=alert(1)>", headers={"content-type": "image/png"}))
        assert await images.fetch(GOOD_IMAGE) is None
        gif = io.BytesIO()
        Image.new("RGB", (4, 4)).save(gif, format="GIF")
        images._failed.clear()
        self.serve(lambda r: httpx.Response(
            200, content=gif.getvalue(), headers={"content-type": "image/png"}))
        assert await images.fetch(GOOD_IMAGE) is None

    async def test_failures_are_not_retried_at_once_and_never_raise(self):
        def boom(request):
            raise httpx.ConnectError("down")

        self.serve(boom)
        assert await images.fetch(GOOD_IMAGE) is None
        assert await images.fetch(GOOD_IMAGE) is None
        assert len(self.calls) == 1

    async def test_switch_and_host_list_come_from_config(self):
        self.serve(lambda r: httpx.Response(
            200, content=jpeg_bytes(), headers={"content-type": "image/jpeg"}))
        self.cfg.fetch_product_images = False
        assert await images.fetch(GOOD_IMAGE) is None and self.calls == []
        self.cfg.fetch_product_images = True
        self.cfg.product_image_hosts = ["img.example.com"]
        assert await images.fetch(GOOD_IMAGE) is None and self.calls == []
        assert await images.fetch("https://img.example.com/a.jpg") is not None

    def test_host_list_rejects_wildcards_and_urls(self):
        from boxbot.core.config import PurchasesConfig

        assert PurchasesConfig().product_image_hosts == ["cdn.shopify.com"]
        assert PurchasesConfig().fetch_product_images is True
        for bad in (["*.shopify.com"], ["cdn.shopify.com/path"], ["10.0.0.1"]):
            with pytest.raises(ValueError):
                PurchasesConfig(product_image_hosts=bad)

    def test_the_card_opens_only_files_in_the_image_cache(self, tmp_path):
        self.dir.mkdir()
        inside = self.dir / "abc.jpg"
        inside.write_bytes(jpeg_bytes())
        outside = tmp_path / "secret.jpg"
        outside.write_bytes(jpeg_bytes())
        assert images.is_cached_image(str(inside)) is not None
        for bad in (str(outside), "/etc/passwd", str(self.dir / ".." / "secret.jpg"),
                    str(self.dir / "missing.jpg"), "", None, 7):
            assert images.is_cached_image(bad) is None

    async def test_cards_show_at_once_then_redraw_when_the_image_arrives(self, world):
        import asyncio

        from boxbot.displays.shop_blocks import payload_for

        gate = asyncio.Event()

        async def slow(url):
            await gate.wait()
            self.dir.mkdir(exist_ok=True)
            path = self.dir / f"{images._key(url)}.jpg"
            path.write_bytes(jpeg_bytes())
            return path

        self.monkeypatch.setattr(images, "fetch", slow)
        world.shop.offers = [offer(1, image_url=GOOD_IMAGE),
                             offer(2, title="No picture",
                                   image_url="https://evil.example.com/x.jpg")]
        await shopping.find("x", conv=world.conv)
        assert len(world.screen.switches) == 1               # drawn immediately
        first = payload_for(world.screen.args["view"])
        assert [c["image"] for c in first["items"]] == [None, None]
        gate.set()
        await asyncio.gather(*list(shopping._tasks))
        assert len(world.screen.switches) == 2               # redrawn once
        second = payload_for(world.screen.args["view"])
        assert second["items"][0]["image"].endswith(".jpg")
        assert second["items"][1]["image"] is None           # off-list: never fetched

    async def test_a_late_image_does_not_steal_the_screen_back(self, world):
        import asyncio

        gate = asyncio.Event()

        async def slow(url):
            await gate.wait()
            self.dir.mkdir(exist_ok=True)
            path = self.dir / f"{images._key(url)}.jpg"
            path.write_bytes(jpeg_bytes())
            return path

        self.monkeypatch.setattr(images, "fetch", slow)
        world.shop.offers = [offer(1, image_url=GOOD_IMAGE)]
        await shopping.find("x", conv=world.conv)
        await world.screen.switch("clock")                   # someone moved on
        gate.set()
        await asyncio.gather(*list(shopping._tasks))
        assert world.screen.active == "clock"


# ---------------------------------------------------------------------------
# D. Stores and one definition of "approved"
# ---------------------------------------------------------------------------


class TestStores:
    async def test_stores_lists_config_and_admin_approved(self, world):
        assert await shopping.stores() == [{"store": SHOP, "domain": SHOP}]
        await approve(world)
        world.shop.offers = [offer(1)]
        await shopping.find("x", conv=world.conv)            # learns the seller name
        assert await shopping.stores() == [
            {"store": "Example Books", "domain": SHOP},
            {"store": OTHER, "domain": OTHER},
        ]

    async def test_one_function_answers_is_this_store_approved(self, world):
        assert await service.shop_is_approved(SHOP)
        assert not await service.shop_is_approved(OTHER)
        await approve(world)
        assert await service.shop_is_approved(OTHER)
        assert await service.approved_shops() == [SHOP, OTHER]
        await world.store.revoke_shop(OTHER)
        assert not await service.shop_is_approved(OTHER)

    async def test_no_path_checks_the_config_list_on_its_own_for_approval(self):
        """The config list alone decides only automatic purchase."""
        source = Path(service.__file__).read_text()
        uses = [
            line.strip() for line in source.splitlines()
            if "allowed_shops" in line and "in " in line and "cfg" in line
        ]
        assert uses == ['if row["shop"] not in cfg.allowed_shops:']   # _verify (auto)
        assert "allowed_shops" not in Path(shopping.__file__).read_text()

    async def test_store_feed_opens_for_any_approved_store_and_no_other(
        self, world, monkeypatch,
    ):
        fetched = []

        async def feed(host, **kw):
            fetched.append(host)
            return [offer(7, title="Goldilocks", shop=host, currency=None,
                          options=["Board book"])]

        monkeypatch.setattr(service, "fetch_store_products", feed)
        with pytest.raises(service.PurchaseError) as e:
            await shopping.store_products(OTHER, conv=world.conv)
        assert "not approved yet" in str(e.value)
        assert "an admin can approve it by text" in str(e.value)
        assert "allowed_shops" not in str(e.value) and "config" not in str(e.value)
        assert fetched == []
        for hostile in ("169.254.169.254", "localhost", "https://x.example.com:8080",
                        "user@x.example.com", ""):
            with pytest.raises(service.PurchaseError):
                await shopping.store_products(hostile, conv=world.conv)
        assert fetched == []
        await approve(world)
        out = await shopping.store_products(OTHER, query="gold", conv=world.conv)
        assert fetched == [OTHER]
        assert out["results"][0] == {
            "id": "a1", "title": "Goldilocks", "variant": "Board book",
            "price": "$6.99", "store": "Example Books", "approved": True,
            "in_stock": True,
        }


# ---------------------------------------------------------------------------
# E. Cart and quote
# ---------------------------------------------------------------------------


class TestCart:
    @pytest.fixture(autouse=True)
    async def _found(self, world):
        world.shop.offers = [
            offer(1, title="Paperback Titanic"),
            offer(2, title="Hardcover Titanic", price=1099),
            offer(3, title="Far book", shop=STRANGER, seller="Far Away", price=400),
        ]
        await shopping.find("x", show=False, conv=world.conv)
        self.w = world

    async def cart(self, **kw):
        return await shopping.cart(conv=self.w.conv, **kw)

    async def test_add_remove_replace_clear(self):
        view = await self.cart(add=["a1"])
        assert view["lines"] == [{"id": "a1", "title": "Paperback Titanic",
                                  "variant": None, "qty": 1, "price": "$6.99"}]
        assert view["changed"] == ["added a1"]
        assert view["stores"] == [{
            "store": "Example Books", "approved": True, "subtotal": "$6.99",
            "shipping": None, "tax": None, "total": None, "delivery": None,
        }]
        assert view["subtotal"] == "$6.99" and view["total"] is None
        view = await self.cart(add=["a1"])
        assert view["lines"][0]["qty"] == 2 and view["subtotal"] == "$13.98"
        view = await self.cart(add={"a1": 1})
        assert view["lines"][0]["qty"] == 1
        view = await self.cart(replace={"a1": "a2"})
        assert [l["id"] for l in view["lines"]] == ["a2"]
        assert view["changed"] == ["replaced a1 with a2"]
        view = await self.cart(remove=["a2", "a1"])
        assert view["lines"] == [] and "a1 was not in the cart" in view["changed"]
        await self.cart(add=["a1", "a2"])
        view = await self.cart(clear=True)
        assert view["lines"] == [] and view["changed"] == ["cleared"]
        assert view["subtotal"] is None

    async def test_nothing_is_persisted_and_nobody_is_texted(self):
        await self.cart(add=["a1", "a2"])
        await self.cart(quote=True)
        assert await self.w.store.list() == []
        assert self.w.out.sent == []

    @pytest.mark.parametrize("bad", [
        dict(add=["a1"] * 11), dict(add=["nope"]), dict(add=[1]), dict(add="a1;"),
        dict(add={"a1": 0}), dict(add={"a1": 11}), dict(add={"a1": True}),
        dict(add={"a1": "2"}), dict(remove={"a1": 1}), dict(replace=["a1", "a2"]),
        dict(replace={"a2": "a1"}), dict(replace={"a1": "zz9"}),
        dict(add=["a1"], clear="yes"), dict(add=["a1"], quote=1),
        dict(add=["z1"]),
    ])
    async def test_bad_arguments_change_nothing(self, bad):
        await self.cart(add=["a1"])
        with pytest.raises(service.PurchaseError):
            await self.cart(**bad)
        assert [l["id"] for l in (await self.cart())["lines"]] == ["a1"]
        assert self.w.shop.created == []

    async def test_quantity_and_line_limits(self):
        for _ in range(10):
            await self.cart(add=["a1"])
        with pytest.raises(service.PurchaseError, match="quantity"):
            await self.cart(add=["a1"])
        self.w.shop.offers = [offer(100 + n, title=f"B{n}") for n in range(8)]
        await shopping.find("y", show=False, conv=self.w.conv)
        await shopping.find("z", show=False, conv=self.w.conv)
        await self.cart(add=[f"b{n}" for n in range(1, 9)])
        with pytest.raises(service.PurchaseError, match="at most 10"):
            await self.cart(add=["c1", "c2"])

    async def test_quote_returns_the_store_breakdown(self):
        self.w.shop.tax = 56
        self.w.shop.delivery = {
            "selected_option_id": "std",
            "options": [
                {"id": "exp", "title": "Express", "totals": [{"type": "total", "amount": 1599}]},
                {"id": "std", "title": "Standard", "totals": [{"type": "total", "amount": 599}]},
            ],
        }
        await self.cart(add=["a2"])
        view = await self.cart(quote=True)
        assert view["stores"] == [{
            "store": "Example Books", "approved": True, "subtotal": "$10.99",
            "shipping": "$5.99", "tax": "$0.56", "total": "$17.54",
            "delivery": "Standard · $5.99",
        }]
        assert view["total"] == "$17.54"
        assert self.w.shop.created == [(SHOP, [{"variant_id": gid(2), "quantity": 1}])]
        blob = json.dumps(view)
        assert "secret" not in blob and "chk_" not in blob and "Hillsboro" not in blob

    async def test_quote_is_reused_until_the_lines_change_or_it_ages_out(self):
        await self.cart(add=["a1"])
        await self.cart(quote=True)
        await self.cart(quote=True)
        again = await self.cart()                            # plain read keeps figures
        assert again["total"] == "$12.98"
        assert len(self.w.shop.created) == 1
        await self.cart(add=["a2"], quote=True)              # lines changed
        assert len(self.w.shop.created) == 2
        self.w.clock[0] += shopping.QUOTE_TTL + 1
        assert (await self.cart())["total"] is None          # stale: not shown
        await self.cart(quote=True)
        assert len(self.w.shop.created) == 3

    async def test_unapproved_store_is_never_quoted(self):
        await self.cart(add=["a1", "a3"])
        view = await self.cart(quote=True)
        assert [c[0] for c in self.w.shop.created] == [SHOP]   # nothing sent to the other
        far = view["stores"][1]
        assert far == {
            "store": "Far Away", "approved": False, "subtotal": "$4.00",
            "shipping": None, "tax": None, "total": None, "delivery": None,
            "needs_approval": True, "domain": STRANGER,
        }
        assert view["stores"][0]["total"] == "$12.98"
        assert view["total"] is None and view["subtotal"] == "$10.99"
        assert self.w.out.sent == []                         # and nobody was texted

    async def test_quote_has_its_own_hourly_ceiling(self):
        self.w.cfg.max_quotes_per_hour = 2
        await self.cart(add=["a1"], quote=True)
        await self.cart(add=["a1"], quote=True)
        view = await self.cart(add=["a1"], quote=True)
        assert "too many quotes" in view["stores"][0]["quote_error"]
        assert view["total"] is None and len(self.w.shop.created) == 2

    async def test_quote_failure_is_reported_per_store(self):
        from boxbot.purchases.shop_client import ShopError

        await self.cart(add=["a1"])
        self.w.shop.create_error = ShopError("out of stock https://evil.example/x")
        view = await self.cart(quote=True)
        assert "Shop refused" in view["stores"][0]["quote_error"]
        assert "evil.example" not in json.dumps(view)
        assert view["lines"] and view["total"] is None

    async def test_checkout_price_corrects_the_catalogue_price(self):
        self.w.shop.prices[gid(1)] = 749
        view = await self.cart(add=["a1"], quote=True)
        assert "a1 is $7.49 at checkout" in view["changed"]
        assert view["lines"][0]["price"] == "$7.49"

    async def test_quote_logs_totals_and_fulfilment_structure_only(self, caplog):
        self.w.shop.delivery = {"selected_option_id": "std", "options": [
            {"id": "std", "title": "Standard", "amount": 699},
        ]}
        await self.cart(add=["a1"])
        with caplog.at_level(logging.INFO, logger="boxbot.purchases.service"):
            await self.cart(quote=True)
        (line,) = [r.getMessage() for r in caplog.records
                   if "Checkout costs (quote" in r.getMessage()]
        assert '"type": "fulfillment"' in line and '"amount": 599' in line
        assert '"destinations": 1' in line and '"amount": 699' in line
        for private in ("Secret Lane", "Hillsboro", "owner@example.com", "+1555",
                        "BUDGETSECRET", "secret-1", "Standard"):
            assert private not in line

    def test_cost_shape_shows_a_doubled_shipping_charge(self):
        checkout = {
            "totals": [{"type": "subtotal", "amount": 699},
                       {"type": "fulfillment", "amount": 699, "display_text": "Ship"},
                       {"type": "fulfillment", "amount": 699},
                       {"type": "total", "amount": 2097}],
            "fulfillment": {"methods": [{"type": "shipping", "destinations": [
                {"street_address": "1 Secret Lane"}], "groups": [
                {"selected_option_id": "a", "options": [
                    {"id": "a", "title": "Flat rate", "totals": [
                        {"type": "total", "amount": 699}]}]},
                {"selected_option_id": "b", "options": [
                    {"id": "b", "title": "Flat rate", "totals": [
                        {"type": "total", "amount": 699}]}]},
            ]}]},
        }
        shape = checkout_cost_shape(checkout)
        assert [t["amount"] for t in shape["totals"] if t["type"] == "fulfillment"] == [
            699, 699]
        assert len(shape["fulfillment"]["methods"][0]["groups"]) == 2
        assert "Secret" not in json.dumps(shape) and "Flat" not in json.dumps(shape)
        # Two selected options: nothing is guessed as "the" delivery.
        assert selected_delivery(checkout) is None

    def test_selected_delivery_reads_only_what_the_store_selected(self):
        one = {"fulfillment": {"methods": [{"options": [
            {"id": "a", "name": "Ground\nhttps://evil.example/x", "selected": True,
             "price": {"amount": 500}},
            {"id": "b", "name": "Air", "price": {"amount": 900}},
        ]}]}}
        got = selected_delivery(one)
        assert got["cents"] == 500 and "evil.example" not in got["name"]
        assert selected_delivery({}) is None
        assert selected_delivery({"fulfillment": {"methods": [{"options": [
            {"id": "a", "name": "Ground"}]}]}}) is None
        assert summarize_checkout(one)["delivery"] == got


# ---------------------------------------------------------------------------
# F. The breakdown on real purchases
# ---------------------------------------------------------------------------


class TestBreakdown:
    async def test_persisted_shown_to_the_agent_and_in_the_admin_text(self, world):
        world.shop.tax = 56
        world.shop.delivery = {"selected_option_id": "s", "options": [
            {"id": "s", "title": "Standard", "amount": 599}]}
        with patch.object(checker, "check", side_effect=deny):
            out = await service.propose(
                SHOP, [{"variant_id": gid(1), "quantity": 1}], "why", conv=conv_from())
        view = out["purchase"]
        assert (view["subtotal_cents"], view["shipping_cents"], view["tax_cents"],
                view["total_cents"], view["delivery"]) == (699, 599, 56, 1354, "Standard")
        row = await world.store.get(view["id"])
        assert row["shipping_cents"] == 599 and row["tax_cents"] == 56
        (text,) = texts(world.out)
        assert "Subtotal: $6.99 · Shipping: $5.99 (Standard) · Tax: $0.56" in text
        assert "Total: $13.54" in text

    async def test_existing_database_gains_the_columns(self, tmp_path):
        path = tmp_path / "old" / "purchases.db"
        path.parent.mkdir()
        db = sqlite3.connect(path)
        db.execute(
            "CREATE TABLE purchases (id TEXT PRIMARY KEY, shop TEXT NOT NULL, "
            "items TEXT NOT NULL, total_cents INTEGER, currency TEXT, "
            "reason TEXT NOT NULL, requested_by TEXT, channel_key TEXT, "
            "status TEXT NOT NULL, decision TEXT, checkout_id TEXT NOT NULL, "
            "continue_url TEXT, complete_after TEXT, link_sent_at TEXT, "
            "order_ref TEXT, note TEXT, created_at TEXT NOT NULL, "
            "updated_at TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO purchases VALUES ('old00001', 'books.example.com', '[]', "
            "1298, 'USD', 'r', 'x', 'k', 'review', 'review', 'chk', NULL, NULL, "
            "NULL, NULL, NULL, '2026-01-01T00:00:00.000000+00:00', "
            "'2026-01-01T00:00:00.000000+00:00')"
        )
        db.commit()
        db.close()
        old = PurchaseStore(db_path=path)
        row = await old.get("old00001")
        assert row["total_cents"] == 1298 and row["shipping_cents"] is None
        assert service.public_view(row)["shipping_cents"] is None
        new = await old.create(
            shop=SHOP, items=[], total_cents=5, currency="USD", reason="r",
            requested_by=None, channel_key=None, checkout_id="c", continue_url=None,
            subtotal_cents=3, shipping_cents=2, tax_cents=0, delivery="Standard",
        )
        assert new["shipping_cents"] == 2 and new["delivery"] == "Standard"
        assert (await old.get("old00001"))["id"] == "old00001"   # reopened, intact


# ---------------------------------------------------------------------------
# G. Send and cancel
# ---------------------------------------------------------------------------


class TestSend:
    @pytest.fixture(autouse=True)
    async def _found(self, world):
        await approve(world)
        world.shop.shipping[OTHER] = 300
        world.shop.offers = [
            offer(1, title="Paperback Titanic"),
            offer(2, title="Hardcover Titanic", price=1099),
            offer(3, title="Toy boat", shop=OTHER, seller="Toy Shop", price=400),
        ]
        await shopping.find("x", show=False, conv=world.conv)
        self.w = world

    async def send(self, reason="asked for the Titanic book", verdict=deny):
        with patch.object(checker, "check", side_effect=verdict):
            return await shopping.send(reason, conv=self.w.conv)

    async def cart(self, **kw):
        return await shopping.cart(conv=self.w.conv, **kw)

    async def test_one_purchase_per_store_and_the_same_shape_as_propose(self):
        await self.cart(add=["a1", "a3"])
        out = await self.send()
        assert [p["store"] for p in out["purchases"]] == ["Example Books", "Toy Shop"]
        for entry in out["purchases"]:
            assert set(entry) == {"store", "purchase", "decision", "blocked_by",
                                  "admins_notified", "already_proposed"}
            assert entry["decision"] == "review" and entry["admins_notified"] == 1
        assert out["total"] == "$19.98" and out["admins"] == ["Admin"]
        assert out["needs_approval"] == []
        rows = await self.w.store.list()
        assert sorted(r["shop"] for r in rows) == [SHOP, OTHER]
        assert len(texts(self.w.out)) == 2
        blob = json.dumps(out)
        assert "secret-" not in blob and "chk_" not in blob and "checker" not in blob

    async def test_requester_is_the_transport_never_a_voice_name(self):
        self.w.conv.exchange_log = [person("buy the Titanic book", name="Jacob")]
        self.w.conv.participants = ["Jacob"]
        await self.cart(add=["a1"])
        await self.send("Jacob asked for it")
        (row,) = await self.w.store.list()
        assert row["requested_by"] == "voice at the box"
        (text,) = texts(self.w.out)
        (came,) = [l for l in text.splitlines() if l.startswith("Came from: ")]
        assert re.fullmatch(r"Came from: voice at the box, \d{1,2}:\d\d [ap]m", came)
        assert "Jacob" not in came

    async def test_send_adopts_a_valid_quote_instead_of_a_second_checkout(self):
        await self.cart(add=["a1"], quote=True)
        out = await self.send()
        assert len(self.w.shop.created) == 1
        row = await self.w.store.get(out["purchases"][0]["purchase"]["id"])
        assert row["checkout_id"] == "chk_1" and row["total_cents"] == 1298
        # The quote is spent: after a cancel, a new purchase gets its own.
        await shopping.cancel(conv=self.w.conv)
        again = await self.send()
        assert again["purchases"][0]["already_proposed"] is False
        assert len(self.w.shop.created) == 2

    async def test_stale_quote_is_not_adopted(self):
        await self.cart(add=["a1"], quote=True)
        self.w.clock[0] += shopping.QUOTE_TTL + 1
        await self.send()
        assert len(self.w.shop.created) == 2

    async def test_identical_cart_again_returns_the_existing_purchase(self):
        await self.cart(add=["a1"])
        first = await self.send()
        second = await self.send()
        assert second["purchases"][0]["already_proposed"] is True
        assert (second["purchases"][0]["purchase"]["id"]
                == first["purchases"][0]["purchase"]["id"])
        assert len(await self.w.store.list()) == 1 and len(texts(self.w.out)) == 1

    async def test_after_a_cancel_the_same_cart_is_a_new_purchase(self):
        await self.cart(add=["a1"])
        first = await self.send()
        await shopping.cancel(conv=self.w.conv)
        second = await self.send()
        assert second["purchases"][0]["already_proposed"] is False
        assert (second["purchases"][0]["purchase"]["id"]
                != first["purchases"][0]["purchase"]["id"])

    async def test_changed_cart_supersedes_with_one_text_marked_as_replacing(self):
        await self.cart(add=["a1"])
        first = (await self.send())["purchases"][0]["purchase"]["id"]
        await self.cart(replace={"a1": "a2"})
        second = (await self.send())["purchases"][0]["purchase"]["id"]
        assert second != first
        assert (await self.w.store.get(first))["status"] == "cancelled"
        assert (await self.w.store.get(second))["status"] == "review"
        sent = texts(self.w.out)
        assert len(sent) == 2                                # one per send, no extra
        assert f"Replaces purchase {first}, now cancelled" in sent[1]
        assert f"ignore the earlier link for {SHOP}" in sent[1]
        assert sent[1].splitlines()[-1].startswith("Review and pay: https://")

    async def test_supersede_only_touches_the_store_that_changed(self):
        await self.cart(add=["a1", "a3"])
        first = {p["store"]: p["purchase"]["id"] for p in (await self.send())["purchases"]}
        await self.cart(replace={"a1": "a2"})
        second = await self.send()
        by_store = {p["store"]: p for p in second["purchases"]}
        assert by_store["Toy Shop"]["already_proposed"] is True
        assert by_store["Toy Shop"]["purchase"]["id"] == first["Toy Shop"]
        assert (await self.w.store.get(first["Toy Shop"]))["status"] == "review"
        assert (await self.w.store.get(first["Example Books"]))["status"] == "cancelled"

    async def test_failed_replacement_still_tells_admins_the_old_link_is_dead(self):
        from boxbot.purchases.shop_client import ShopError

        await self.cart(add=["a1"])
        first = (await self.send())["purchases"][0]["purchase"]["id"]
        await self.cart(replace={"a1": "a2"})
        self.w.shop.create_error = ShopError("sold out")
        with pytest.raises(service.PurchaseError):
            await self.send()
        last = texts(self.w.out)[-1]
        assert f"boxBot purchase {first} cancelled — ignore the earlier link" in last

    async def test_unapproved_store_is_asked_about_not_sent_to(self):
        self.w.shop.offers = [
            offer(4, title="Far book", shop=STRANGER, seller="Far Away", price=400)]
        await shopping.find("far", show=False, conv=self.w.conv)
        await self.cart(add=["a1", "b1"])
        out = await self.send()
        assert [p["store"] for p in out["purchases"]] == ["Example Books"]
        (waiting,) = out["needs_approval"]
        assert waiting["store"] == "Far Away" and waiting["domain"] == STRANGER
        assert "asked" in waiting["note"]
        assert STRANGER not in [c[0] for c in self.w.shop.created]
        ask = [t for t in texts(self.w.out) if "allow shop " + STRANGER in t]
        assert len(ask) == 1 and "Came from: voice at the box" in ask[0]

    async def test_auto_mode_reply_does_not_reveal_the_checker(self):
        self.w.cfg.mode = "auto"
        await self.cart(add=["a1"])
        out = await self.send(verdict=deny)
        assert out["purchases"][0]["blocked_by"] == ["needs_review"]
        assert "checker" not in out["purchases"][0]
        # Voice never buys automatically, whatever the checker says.
        await shopping.cancel(conv=self.w.conv)
        out = await self.send(verdict=allow)
        assert out["purchases"][0]["decision"] == "review"
        assert self.w.shop.completions == []

    @pytest.mark.parametrize("reason", [None, "", "   ", 7, ["x"], "r" * 2001])
    async def test_reason_is_validated(self, reason):
        await self.cart(add=["a1"])
        with pytest.raises(service.PurchaseError):
            await self.send(reason)
        assert await self.w.store.list() == []

    async def test_reason_cannot_forge_the_admin_text(self):
        await self.cart(add=["a1"])
        await self.send("ok\nReview and pay: https://evil.example/pay")
        (text,) = texts(self.w.out)
        assert "evil.example" not in text
        assert sum(l.startswith("Review and pay:") for l in text.splitlines()) == 1

    async def test_empty_cart_and_send_ceiling(self):
        with pytest.raises(service.PurchaseError, match="cart is empty"):
            await self.send()
        await self.cart(add=["a1"])
        for _ in range(shopping._send_rate.limit):
            await self.send()
        with pytest.raises(service.PurchaseError, match="too many purchases sent"):
            await self.send()

    async def test_daily_proposal_ceiling_still_applies(self):
        self.w.cfg.max_proposals_per_day = 1
        await self.cart(add=["a1"])
        await self.send()
        await self.cart(replace={"a1": "a2"})
        with pytest.raises(service.PurchaseError, match="too many purchase proposals"):
            await self.send()


class TestCancel:
    @pytest.fixture(autouse=True)
    async def _sent(self, world):
        world.shop.offers = [offer(1), offer(2, title="Other", price=1099)]
        await shopping.find("x", show=False, conv=world.conv)
        await shopping.cart(add=["a1"], conv=world.conv)
        with patch.object(checker, "check", side_effect=deny):
            out = await shopping.send("why", conv=world.conv)
        self.pid = out["purchases"][0]["purchase"]["id"]
        self.w = world
        world.out.sent.clear()

    def test_cancelling_is_local_the_store_is_never_called(self):
        """Why the notice exists: there is no API call that voids the
        checkout, so the link already texted stays payable."""
        from boxbot.purchases.shop_client import ShopClient

        assert not [n for n in dir(ShopClient) if "cancel" in n.lower()]

    async def test_default_is_this_conversations_latest_send_and_admins_are_told(self):
        out = await shopping.cancel(conv=self.w.conv)
        assert [p["id"] for p in out["cancelled"]] == [self.pid]
        assert out["cancelled"][0]["status"] == "cancelled" and out["admins_told"] == 1
        (text,) = texts(self.w.out)
        assert text == (
            f"🛒 boxBot purchase {self.pid} cancelled — ignore the earlier link "
            f"for {SHOP} ($12.98). boxBot will not buy it."
        )
        assert service.looks_like_service_notice(text)        # agent cannot forge it
        assert self.w.shop.gets == 0 and self.w.shop.completions == []

    async def test_cancel_with_nothing_live_says_what_to_do(self):
        await shopping.cancel(conv=self.w.conv)
        with pytest.raises(service.PurchaseError, match="cart\\(clear=True\\)"):
            await shopping.cancel(conv=self.w.conv)
        with pytest.raises(service.PurchaseError):
            await shopping.cancel(conv=room("someone-else"))
        assert (await self.w.store.get(self.pid))["status"] == "cancelled"

    async def test_another_conversation_does_not_default_to_this_purchase(self):
        with pytest.raises(service.PurchaseError):
            await shopping.cancel(conv=room("conv-2"))
        assert (await self.w.store.get(self.pid))["status"] == "review"
        # By id anyone may: stopping a purchase needs no authority.
        out = await shopping.cancel(self.pid, conv=room("conv-2"))
        assert out["cancelled"][0]["status"] == "cancelled"

    async def test_no_notice_when_no_link_ever_reached_an_admin(self):
        await self.w.store.update_fields(self.pid, link_sent_at=None)
        out = await shopping.cancel(conv=self.w.conv)
        assert out["admins_told"] == 0 and self.w.out.sent == []

    async def test_notice_is_rate_limited_but_the_cancel_still_happens(self):
        for _ in range(service._cancel_notice_rate.limit):
            service._cancel_notice_rate.add()
        out = await shopping.cancel(conv=self.w.conv)
        assert out["cancelled"][0]["status"] == "cancelled"
        assert out["admins_told"] == 0 and self.w.out.sent == []

    @pytest.mark.parametrize("bad", ["", "nope", "ab12cd34; drop", 7, ["ab12cd34"]])
    async def test_id_is_validated(self, bad):
        with pytest.raises(service.PurchaseError):
            await shopping.cancel(bad, conv=self.w.conv)
        assert (await self.w.store.get(self.pid))["status"] == "review"

    async def test_too_late_is_refused_truthfully(self):
        await self.w.store.transition(self.pid, "scheduled", expect="review")
        await self.w.store.transition(self.pid, "verifying", expect="scheduled")
        await self.w.store.transition(self.pid, "paying", expect="verifying")
        with pytest.raises(service.PurchaseError, match="can no longer be stopped"):
            await shopping.cancel(self.pid, conv=self.w.conv)

    async def test_handler_and_sdk_shapes(self):
        from boxbot.tools._tool_context import current_conversation

        token = current_conversation.set(self.w.conv)
        try:
            by_default = await process_action({"_sdk": "purchases.cancel"}, ActionContext())
            with patch.object(checker, "check", side_effect=deny):
                again = await process_action(
                    {"_sdk": "purchases.send", "reason": "again"}, ActionContext())
            by_id = await process_action(
                {"_sdk": "purchases.cancel",
                 "id": again["purchases"][0]["purchase"]["id"]}, ActionContext())
        finally:
            current_conversation.reset(token)
        assert by_default["status"] == "ok" and "purchase" not in by_default
        assert by_default["cancelled"][0]["id"] == self.pid
        assert by_id["purchase"]["status"] == "cancelled"


# ---------------------------------------------------------------------------
# H / I. Guidance and SDK
# ---------------------------------------------------------------------------


class TestGuidance:
    SKILL = REPO / "skills/purchasing/SKILL.md"
    API = REPO / "skills/bb/modules/purchases.md"
    DISPLAY = REPO / "skills/bb/modules/display.md"

    def test_guidance_names_only_calls_that_exist(self):
        import boxbot.sdk as bb

        real = {n for n in dir(bb.purchases) if not n.startswith("_")}
        for doc in (self.SKILL, self.API, self.DISPLAY):
            text = doc.read_text()
            for name in re.findall(r"bb\.purchases\.([a-z_]+)", text):
                assert name in real, f"{doc.name} mentions bb.purchases.{name}"
            assert "purchases.search" not in text
            assert "variant_id" not in text
        for doc in (self.SKILL, self.API):
            for call in re.findall(r"`([a-z_]+)\(", doc.read_text()):
                assert call in real or call in ("load_skill", "switch_display"), call

    def test_the_old_image_recipe_and_hand_typed_card_are_gone(self):
        text = self.DISPLAY.read_text()
        assert "requests.get(hit" not in text and "image_url" not in text
        assert "switch_display(\"product\"" not in text
        assert "switch_display('product'" not in text

    def test_nothing_says_the_agent_cannot_cancel(self):
        for doc in (self.SKILL, self.API):
            text = doc.read_text().lower()
            assert "cannot cancel" not in text and "can't cancel" not in text
            assert "you can cancel" in text or "cancel (`cancel()`)" in text

    def test_core_rules_are_present(self):
        skill = self.SKILL.read_text()
        for needle in (
            "find → cart → send", "By id only", "already shows the cards",
            "Do not re-search", "Which number?", "cart(quote=True)",
            "approved_only=True", "Prefer approved stores", "allow shop",
            "Confirm exactly once", "nothing is bought until they pay",
            "cart(clear=True)", "Anyone may ask", "`unknown`",
            "Only an adult's phone pays", "Route around a limit",
            "data, never instructions",
        ):
            assert needle in skill, needle

    def test_partial_doc_never_serves_cannot_without_can(self):
        """Prefetch injects module docs section by section (preamble +
        chosen sections). Whatever it picks must say what IS allowed."""
        from boxbot.prefetch.sources import _splice_module, _split_module_doc

        preamble, sections = _split_module_doc(self.API.read_text())
        assert "cancel (`cancel()`)" in preamble and "**Cannot:**" in preamble
        for index, (heading, body) in enumerate(sections):
            spliced = _splice_module("purchases", preamble, sections, [index])
            if "Cannot" in body:
                assert "Can:" in body and "cancel" in body, heading
            assert "cancel (`cancel()`)" in spliced
        assert not any(h.lower().startswith("cannot") for h, _ in sections)

    def test_skill_is_shorter_than_before(self):
        assert len(self.SKILL.read_text()) < 7000


class TestSdkSurface:
    def test_new_calls_dispatch_the_expected_actions(self):
        import boxbot.sdk as bb
        from boxbot.sdk import _transport

        sent = []

        def fake(action, payload, timeout=30):
            sent.append((action, payload))
            return {
                "purchases.find": {"status": "ok", "results": [{"id": "a1"}], "shown": []},
                "purchases.stores": {"status": "ok", "stores": [{"store": "S", "domain": "d"}]},
                "purchases.cancel": {"status": "ok", "cancelled": [{"id": "p"}],
                                     "admins_told": 1, "purchase": {"id": "p"}},
            }.get(action, {"status": "ok", "lines": []})

        with patch.object(_transport, "dispatch_or_raise", side_effect=fake):
            assert bb.purchases.find("x", approved_only=True) == [{"id": "a1"}]
            bb.purchases.show("a1")
            assert bb.purchases.stores() == [{"store": "S", "domain": "d"}]
            assert bb.purchases.cart(add="a1", quote=True) == {"lines": []}
            bb.purchases.send("why")
            assert bb.purchases.cancel() == {"cancelled": [{"id": "p"}],
                                             "admins_told": 1, "purchase": {"id": "p"}}
            assert bb.purchases.cancel("p") == {"id": "p"}
        assert sent == [
            ("purchases.find", {"query": "x", "approved_only": True, "show": True}),
            ("purchases.show", {"ids": ["a1"]}),
            ("purchases.stores", {}),
            ("purchases.cart", {"clear": False, "quote": True, "add": ["a1"]}),
            ("purchases.send", {"reason": "why"}),
            ("purchases.cancel", {}),
            ("purchases.cancel", {"id": "p"}),
        ]

    def test_sdk_version_was_bumped(self):
        text = (REPO / "src/boxbot/sdk/pyproject.toml").read_text()
        assert 'version = "0.6.0"' in text

    def test_docstrings_stay_terse(self):
        import boxbot.sdk as bb

        for name in ("find", "show", "stores", "cart", "send", "cancel"):
            assert len(getattr(bb.purchases, name).__doc__) < 1100, name
