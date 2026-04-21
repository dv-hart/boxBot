"""Shared fixtures for the boxBot test suite.

Provides reusable pytest fixtures for config, events, database paths,
and memory store instances used across multiple test modules.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Pre-mock unavailable third-party modules so that boxbot can be imported
# without requiring the full dependency tree. We only need to mock modules
# that are imported at the *top level* of production code but are not
# installed in the test environment.
# ---------------------------------------------------------------------------

_MOCK_MODULES = ["anthropic"]

for _mod_name in _MOCK_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

import boxbot.core.config as config_module
import boxbot.core.events as events_module


# ---------------------------------------------------------------------------
# Event loop
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset the config singleton before every test.

    Prevents one test's load_config() from leaking into another.
    """
    original = config_module._config
    config_module._config = None
    yield
    config_module._config = original


@pytest.fixture
def tmp_config(tmp_path):
    """Create a minimal config YAML in tmp_path and return the path.

    The YAML sets essential values for tests without needing env vars.
    """
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        """\
agent:
  name: TestBot
  wake_word: hey test
  max_turns: 5
schedule:
  idle_timeout: 60
  person_trigger_expiry_days: 3
display:
  rotation_interval: 10
  idle_displays:
    - picture
camera:
  resolution: [640, 480]
  scan_fps: 2
audio:
  stt_provider: test
  tts_provider: test
memory:
  max_context_memories: 5
photos:
  storage_path: "{photos_path}"
  max_storage_percent: 10
sandbox:
  timeout: 5
  enforce: false
models:
  large: test-large
  small: test-small
logging:
  level: DEBUG
""".format(photos_path=str(tmp_path / "photos"))
    )
    return config_yaml


@pytest.fixture
def mock_config(tmp_config):
    """Load config from the tmp YAML and return the BoxBotConfig instance.

    Also patches environment variables so _overlay_env does not pull in
    real secrets.
    """
    with patch.dict("os.environ", {}, clear=False):
        # Remove any real boxbot env vars that might interfere
        env_clean = {
            k: v
            for k, v in __import__("os").environ.items()
            if not k.startswith("BOXBOT_")
            and k not in ("ANTHROPIC_API_KEY", "WHATSAPP_ACCESS_TOKEN")
        }
        with patch.dict("os.environ", env_clean, clear=True):
            cfg = config_module.load_config(tmp_config)
    return cfg


# ---------------------------------------------------------------------------
# Event bus fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_event_bus():
    """Reset the event bus singleton before every test."""
    original = events_module._event_bus
    events_module._event_bus = None
    yield
    events_module._event_bus = original


@pytest.fixture
def event_bus():
    """Return a fresh EventBus instance (also set as the global singleton)."""
    bus = events_module.EventBus()
    events_module._event_bus = bus
    return bus


# ---------------------------------------------------------------------------
# Database path fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary database path (does not create the file)."""
    return tmp_path / "test.db"


@pytest.fixture
def tmp_memory_db(tmp_path):
    """Return a temporary path for the memory store database."""
    return tmp_path / "memory.db"


@pytest.fixture
def tmp_photos_db(tmp_path):
    """Return a temporary path for the photo store database."""
    return tmp_path / "photos.db"


@pytest.fixture
def tmp_auth_db(tmp_path):
    """Return a temporary path for the auth database."""
    return tmp_path / "auth.db"


# ---------------------------------------------------------------------------
# Memory store fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def memory_store(tmp_memory_db):
    """Create an initialized MemoryStore using an in-memory-like temp DB.

    Patches the SYSTEM_MEMORY_PATH to a temp location so tests do not
    touch the real data/memory/system.md file.
    """
    from boxbot.memory.store import MemoryStore

    store = MemoryStore(db_path=tmp_memory_db)

    # Patch system memory path to temp
    sys_mem_path = tmp_memory_db.parent / "system.md"
    with patch("boxbot.memory.store.SYSTEM_MEMORY_PATH", sys_mem_path):
        await store.initialize()
        yield store
        await store.close()


# ---------------------------------------------------------------------------
# Photo store fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def photo_store(tmp_photos_db):
    """Create an initialized PhotoStore backed by a temp database."""
    from boxbot.photos.store import PhotoStore

    store = PhotoStore(db_path=tmp_photos_db)
    await store.initialize()
    yield store
    await store.close()


# ---------------------------------------------------------------------------
# Auth manager fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def auth_manager(tmp_auth_db):
    """Create an initialized AuthManager backed by a temp database."""
    from boxbot.communication.auth import AuthManager

    auth = AuthManager(
        db_path=tmp_auth_db,
        code_expiry=600,
        max_attempts_per_window=3,
        attempt_window=60,
        temp_block_duration=120,
        max_temp_blocks=2,
    )
    await auth.init_db()
    return auth
