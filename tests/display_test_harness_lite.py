"""Lite in-process harness for the display SDK.

Same shape as ``tests/display_test_harness.setup_bb()`` — but loads
only the display subsystem via ``importlib.util``, sidestepping
``boxbot/__init__.py``, ``boxbot/displays/__init__.py``, and the heavy
``boxbot/core/__init__.py`` chain (which pulls in numpy, anthropic,
sentence-transformers, the cost subsystem, and the memory store).

Use this when the broader package import is blocked or slow but you
still want to drive the SDK end-to-end against a real DisplayManager.

Usage:

    from tests.display_test_harness_lite import setup_bb
    bb = setup_bb()

    spec = {
        "name": "demo",
        "theme": "boxbot",
        "layout": {"type": "text", "content": "hello"},
    }
    print(bb.display.preview(spec)["path"])
    bb.display.save(spec)

What's wired:
    bb.display          — full SDK module
    bb.manager          — live DisplayManager on a background loop
    bb.ctx              — ActionContext (image_attachments, action_log)

What's NOT wired:
    bb.workspace, bb.camera, bb.photos, bb.memory, bb.tasks, ...
    They'd require the heavy chain we're avoiding.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import tempfile
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any


_HARNESS_STATE: dict[str, Any] = {}
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(
        name, _SRC_ROOT / rel_path,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap_packages() -> None:
    """Install stub package entries so dataclass+importlib see real modules."""
    for pkg in ("boxbot", "boxbot.displays", "boxbot.core", "boxbot.tools",
                "boxbot.sdk"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(_SRC_ROOT / pkg.replace(".", "/"))]
            sys.modules[pkg] = m


def setup_bb(*, fresh_data_dir: bool = True) -> SimpleNamespace:
    """Wire the SDK + DisplayManager for in-process testing.

    Args:
        fresh_data_dir: Allocate a new temp ``BOXBOT_DATA_DIR``.
    """
    if fresh_data_dir or "BOXBOT_DATA_DIR" not in os.environ:
        os.environ["BOXBOT_DATA_DIR"] = tempfile.mkdtemp(prefix="bb-lite-")

    _bootstrap_packages()

    # Load display subsystem in dependency order.
    if "boxbot.core.paths" not in sys.modules:
        _load_module("boxbot.core.paths", "boxbot/core/paths.py")
    themes = sys.modules.get("boxbot.displays.themes") or _load_module(
        "boxbot.displays.themes", "boxbot/displays/themes.py",
    )
    blocks = sys.modules.get("boxbot.displays.blocks") or _load_module(
        "boxbot.displays.blocks", "boxbot/displays/blocks.py",
    )
    spec_mod = sys.modules.get("boxbot.displays.spec") or _load_module(
        "boxbot.displays.spec", "boxbot/displays/spec.py",
    )
    data_sources = sys.modules.get("boxbot.displays.data_sources") or _load_module(
        "boxbot.displays.data_sources", "boxbot/displays/data_sources.py",
    )
    renderer = sys.modules.get("boxbot.displays.renderer") or _load_module(
        "boxbot.displays.renderer", "boxbot/displays/renderer.py",
    )

    # Stub out the programmatic builtins module — manager.start() pulls
    # in the real one, which imports from boxbot.displays. Provide a
    # zero-list stub so start() succeeds without running through the
    # full package init.
    if "boxbot.displays.builtins" not in sys.modules:
        builtins_stub = types.ModuleType("boxbot.displays.builtins")

        def get_builtin_specs():
            return []

        builtins_stub.get_builtin_specs = get_builtin_specs
        sys.modules["boxbot.displays.builtins"] = builtins_stub

    manager_mod = sys.modules.get("boxbot.displays.manager") or _load_module(
        "boxbot.displays.manager", "boxbot/displays/manager.py",
    )

    # Load the SDK transport + display module.
    if "boxbot.sdk._validators" not in sys.modules:
        _load_module("boxbot.sdk._validators", "boxbot/sdk/_validators.py")
    transport = sys.modules.get("boxbot.sdk._transport") or _load_module(
        "boxbot.sdk._transport", "boxbot/sdk/_transport.py",
    )
    # The SDK's display module imports `from . import _transport,
    # _validators as v`. Rewrite it to use the modules we just loaded.
    if "boxbot.sdk.display" not in sys.modules:
        # Patch the package object so relative imports resolve.
        boxbot_sdk_pkg = sys.modules["boxbot.sdk"]
        boxbot_sdk_pkg._transport = transport
        boxbot_sdk_pkg._validators = sys.modules["boxbot.sdk._validators"]
        display = _load_module("boxbot.sdk.display", "boxbot/sdk/display.py")
    else:
        display = sys.modules["boxbot.sdk.display"]

    # Load the dispatcher — it's the real handler the SDK talks to.
    # _sandbox_actions imports boxbot.core.paths (already loaded) and
    # runtime imports for boxbot.displays.* (already loaded).
    sandbox_actions = sys.modules.get("boxbot.tools._sandbox_actions") or (
        _load_module(
            "boxbot.tools._sandbox_actions",
            "boxbot/tools/_sandbox_actions.py",
        )
    )

    # Background asyncio loop for the manager.
    loop = _HARNESS_STATE.get("loop")
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _HARNESS_STATE["loop"] = loop

        def _run_loop() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        threading.Thread(target=_run_loop, daemon=True).start()

    if fresh_data_dir and "manager" in _HARNESS_STATE:
        old = _HARNESS_STATE.pop("manager")
        try:
            asyncio.run_coroutine_threadsafe(old.stop(), loop).result(timeout=10)
        except Exception:
            pass

    if "manager" not in _HARNESS_STATE:
        async def _start_mgr():
            m = manager_mod.DisplayManager()
            await m.start()
            return m

        mgr = asyncio.run_coroutine_threadsafe(
            _start_mgr(), loop,
        ).result(timeout=30)
        _HARNESS_STATE["manager"] = mgr
        manager_mod.set_display_manager(mgr)
    else:
        mgr = _HARNESS_STATE["manager"]

    ctx = sandbox_actions.ActionContext()
    _HARNESS_STATE["ctx"] = ctx

    def fake_request(action_type: str, payload: dict, *,
                     timeout: int = 30) -> dict:
        action = {"_sdk": action_type, **payload}

        async def _run():
            handler = sandbox_actions._handle_display_action
            return await handler(action_type, action, ctx)

        return asyncio.run_coroutine_threadsafe(
            _run(), loop,
        ).result(timeout=timeout)

    def fake_emit(action_type, payload, *, expects_response=False):
        if expects_response:
            return fake_request(action_type, payload)
        return None

    transport.request = fake_request
    transport.emit_action = fake_emit

    return SimpleNamespace(display=display, manager=mgr, ctx=ctx)


def shutdown() -> None:
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
