"""In-process harness for exercising the display SDK without a real boxBot.

Spins up a real :class:`DisplayManager` on a background asyncio loop,
patches :mod:`boxbot_sdk._transport` so the SDK's synchronous
``request()`` calls land directly in the dispatcher, and returns a
``bb``-shaped namespace that authoring scripts can use.

Typical use from an authoring script::

    from tests.display_test_harness import setup_bb
    bb = setup_bb()

    spec = {
        "name": "demo",
        "theme": "boxbot",
        "data_sources": [{"name": "weather"}],
        "layout": {"type": "text", "content": "{weather.temp}",
                   "size": "title"},
    }
    result = bb.display.preview(spec)
    print(result["path"])         # PNG path you can open / Read
    print(result["warnings"])     # binding issues, if any
    bb.display.save(spec)

The harness is intentionally local-only: no network, no Pi, no
WhatsApp. It exists so you can iterate on display specs and find
SDK/doc rough edges quickly.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any


_HARNESS_STATE: dict[str, Any] = {}


def setup_bb(*, fresh_data_dir: bool = True) -> SimpleNamespace:
    """Wire a live DisplayManager + SDK transport for in-process testing.

    Args:
        fresh_data_dir: If True, allocate a new temp ``BOXBOT_DATA_DIR``
            so each call starts clean. Pass False to reuse the previous
            harness state across calls (e.g. across cells in a notebook).

    Returns:
        A namespace with ``display`` and ``manager`` attributes. The
        ``display`` module behaves exactly like ``boxbot_sdk.display``
        in production; ``manager`` is the live DisplayManager so you
        can introspect state directly when debugging the harness
        itself.
    """
    if fresh_data_dir or "BOXBOT_DATA_DIR" not in os.environ:
        os.environ["BOXBOT_DATA_DIR"] = tempfile.mkdtemp(prefix="bb-harness-")

    repo_src = Path(__file__).resolve().parents[1] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    from boxbot.displays.manager import (  # noqa: PLC0415
        DisplayManager,
        set_display_manager,
    )
    from boxbot.sdk import _transport, display  # noqa: PLC0415
    from boxbot.tools._sandbox_actions import (  # noqa: PLC0415
        ActionContext,
        _handle_display_action,
    )

    # Reuse a single background loop across calls so the manager state
    # is consistent. If a previous run shut down, recreate.
    loop = _HARNESS_STATE.get("loop")
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _HARNESS_STATE["loop"] = loop

        def _run_loop() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_run_loop, daemon=True)
        thread.start()
        _HARNESS_STATE["thread"] = thread

    async def _start_mgr() -> DisplayManager:
        m = DisplayManager()
        await m.start()
        return m

    # If a previous manager exists, stop it first when fresh.
    if fresh_data_dir and "manager" in _HARNESS_STATE:
        old = _HARNESS_STATE["manager"]
        try:
            asyncio.run_coroutine_threadsafe(old.stop(), loop).result(timeout=10)
        except Exception:
            pass
        _HARNESS_STATE.pop("manager", None)

    mgr = _HARNESS_STATE.get("manager")
    if mgr is None:
        mgr = asyncio.run_coroutine_threadsafe(_start_mgr(), loop).result(
            timeout=30,
        )
        _HARNESS_STATE["manager"] = mgr
        set_display_manager(mgr)

    ctx = ActionContext()
    _HARNESS_STATE["ctx"] = ctx

    def fake_request(action_type: str, payload: dict[str, Any], *,
                     timeout: int = 30) -> dict[str, Any]:
        # The dispatcher reads action_type from action["_sdk"]; mirror
        # that wire format here so the handler sees what production
        # would.
        action = {"_sdk": action_type, **payload}

        async def _run() -> dict[str, Any]:
            return await _handle_display_action(action_type, action, ctx)

        future = asyncio.run_coroutine_threadsafe(_run(), loop)
        return future.result(timeout=timeout)

    def fake_emit(action_type: str, payload: dict[str, Any], *,
                  expects_response: bool = False) -> dict[str, Any] | None:
        # The new SDK only uses request(), but keep a working stub so
        # legacy paths don't silently drop.
        if expects_response:
            return fake_request(action_type, payload)
        return None

    _transport.request = fake_request
    _transport.emit_action = fake_emit

    return SimpleNamespace(display=display, manager=mgr, ctx=ctx)


def shutdown() -> None:
    """Best-effort teardown for use in long-lived test sessions."""
    loop = _HARNESS_STATE.get("loop")
    mgr = _HARNESS_STATE.get("manager")
    if mgr is not None and loop is not None and not loop.is_closed():
        try:
            asyncio.run_coroutine_threadsafe(mgr.stop(), loop).result(timeout=10)
        except Exception:
            pass
    if loop is not None and not loop.is_closed():
        loop.call_soon_threadsafe(loop.stop)
    _HARNESS_STATE.clear()
