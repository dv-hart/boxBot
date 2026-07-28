"""Home Assistant events bridge — outbound WebSocket → EntityStateChanged.

Feeds the scheduler's *entity* trigger conditions with real-time state
changes from Home Assistant (e.g. an Alarm.com camera's
``binary_sensor.front_door_person`` flipping ``on``).

Security model: this is an **outbound** connection from the box to the
household HA instance — no new network listener is opened. Credentials
are the same ``HOME_ASSISTANT_URL`` / ``HOME_ASSISTANT_TOKEN`` secrets
the ``home_assistant`` integration already uses; if they aren't stored,
the bridge is a silent no-op.

Flow per connection:

1. Connect to ``ws://<host>/api/websocket`` and complete the
   ``auth_required`` → ``auth`` → ``auth_ok`` handshake.
2. Snapshot the current state of watched entities (``get_states``) and
   publish them with ``snapshot=True`` — the scheduler updates its state
   map but never fires on snapshots, so reconnects can't re-fire
   triggers on long-lived states.
3. Subscribe to ``state_changed`` events and publish an
   :class:`~boxbot.core.events.EntityStateChanged` for each change to a
   watched entity.

The watch set is the set of entity_ids referenced by active entity
triggers (:func:`boxbot.core.scheduler.watched_entities`), cached with a
short TTL so a freshly created trigger starts being watched within
seconds without a per-event DB query.

On disconnect the bridge publishes ``new_state="unknown"`` for every
entity it had reported, so stale states can't keep satisfying compound
conditions while the bridge is down. Reconnects use exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from boxbot.core.events import EntityStateChanged, get_event_bus

logger = logging.getLogger(__name__)

URL_SECRET = "HOME_ASSISTANT_URL"
TOKEN_SECRET = "HOME_ASSISTANT_TOKEN"

_RECONNECT_MIN_S = 5.0
_RECONNECT_MAX_S = 300.0


def _load_connection() -> tuple[str, str] | None:
    """Read HA URL + token from the secret store, or None if unset."""
    from boxbot.secrets import get_secret_store

    store = get_secret_store()
    url = (store.load(URL_SECRET) or "").strip()
    token = (store.load(TOKEN_SECRET) or "").strip()
    if not url or not token:
        return None
    return url.rstrip("/"), token


def _ws_url(base_url: str) -> str:
    """Derive the WebSocket API URL from the HA base URL."""
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://"):] + "/api/websocket"
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://"):] + "/api/websocket"
    return "ws://" + base_url + "/api/websocket"


class HAEventsBridge:
    """Maintains the HA WebSocket subscription and publishes entity events."""

    def __init__(
        self,
        *,
        watch_provider: Callable[[], Awaitable[set[str]]] | None = None,
        watch_ttl_s: float = 15.0,
    ) -> None:
        # Injectable for tests; defaults to the scheduler's active-trigger set.
        if watch_provider is None:
            from boxbot.core.scheduler import watched_entities

            watch_provider = watched_entities
        self._watch_provider = watch_provider
        self._watch_ttl_s = watch_ttl_s
        self._watched: set[str] = set()
        self._watched_at: float = 0.0
        # Entities we've published a real state for this connection —
        # reset to "unknown" on disconnect.
        self._reported: set[str] = set()
        self._task: asyncio.Task[None] | None = None
        self._running = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Start the bridge loop. No-op if HA secrets are not stored."""
        if self._running:
            return
        if _load_connection() is None:
            logger.info(
                "HA events bridge idle — %s/%s not in secret store",
                URL_SECRET, TOKEN_SECRET,
            )
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="ha-events-bridge")
        logger.info("HA events bridge started")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._publish_unknown()
        logger.info("HA events bridge stopped")

    # ------------------------------------------------------------------ #
    # Connection loop
    # ------------------------------------------------------------------ #

    async def _run(self) -> None:
        backoff = _RECONNECT_MIN_S
        while self._running:
            try:
                await self._session_once()
                backoff = _RECONNECT_MIN_S  # clean session → reset backoff
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("HA events bridge connection error: %s", exc)
            await self._publish_unknown()
            if not self._running:
                return
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_S)

    async def _session_once(self) -> None:
        """One full connect → auth → snapshot → subscribe → pump session."""
        import aiohttp

        conn = _load_connection()
        if conn is None:
            raise RuntimeError("HA secrets removed from store")
        base_url, token = conn

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                _ws_url(base_url), heartbeat=30
            ) as ws:
                await self._authenticate(ws, token)
                logger.info("HA events bridge connected to %s", base_url)

                msg_id = 1
                await ws.send_json({"id": msg_id, "type": "get_states"})
                snapshot_id = msg_id
                msg_id += 1
                await ws.send_json({
                    "id": msg_id,
                    "type": "subscribe_events",
                    "event_type": "state_changed",
                })

                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        break
                    data = msg.json()
                    if data.get("type") == "result":
                        if data.get("id") == snapshot_id and data.get("success"):
                            await self._handle_snapshot(data.get("result") or [])
                        elif not data.get("success"):
                            logger.warning(
                                "HA WS command failed: %s", data.get("error")
                            )
                    elif data.get("type") == "event":
                        await self._handle_event(data)
        # Normal socket close falls through — the run loop reconnects.

    @staticmethod
    async def _authenticate(ws: Any, token: str) -> None:
        """Complete HA's auth_required → auth → auth_ok handshake."""
        hello = await ws.receive_json()
        if hello.get("type") != "auth_required":
            raise RuntimeError(f"unexpected HA WS greeting: {hello.get('type')}")
        await ws.send_json({"type": "auth", "access_token": token})
        verdict = await ws.receive_json()
        if verdict.get("type") != "auth_ok":
            raise RuntimeError(
                "HA WebSocket auth failed — check HOME_ASSISTANT_TOKEN"
            )

    # ------------------------------------------------------------------ #
    # Message handling
    # ------------------------------------------------------------------ #

    async def _handle_snapshot(self, states: list[dict[str, Any]]) -> None:
        """Publish snapshot=True events for watched entities' current states."""
        watched = await self._get_watched()
        for state in states:
            entity_id = state.get("entity_id", "")
            if entity_id not in watched:
                continue
            await self._publish(
                entity_id=entity_id,
                new_state=state.get("state", ""),
                old_state="",
                friendly_name=(state.get("attributes") or {}).get(
                    "friendly_name", ""
                ),
                snapshot=True,
            )

    async def _handle_event(self, data: dict[str, Any]) -> None:
        """Publish a live state_changed event if the entity is watched."""
        event = (data.get("event") or {})
        if event.get("event_type") != "state_changed":
            return
        payload = event.get("data") or {}
        entity_id = payload.get("entity_id", "")
        if not entity_id or entity_id not in await self._get_watched():
            return
        new = payload.get("new_state") or {}
        old = payload.get("old_state") or {}
        await self._publish(
            entity_id=entity_id,
            new_state=new.get("state", "unknown"),
            old_state=old.get("state", ""),
            friendly_name=(new.get("attributes") or {}).get("friendly_name", ""),
            snapshot=False,
        )

    async def _get_watched(self) -> set[str]:
        """Watched entity set, refreshed from active triggers on a short TTL."""
        now = asyncio.get_running_loop().time()
        if now - self._watched_at >= self._watch_ttl_s:
            try:
                self._watched = await self._watch_provider()
            except Exception:
                logger.debug("watched-entities refresh failed", exc_info=True)
            self._watched_at = now
        return self._watched

    async def _publish(
        self,
        *,
        entity_id: str,
        new_state: str,
        old_state: str,
        friendly_name: str,
        snapshot: bool,
    ) -> None:
        self._reported.add(entity_id)
        bus = get_event_bus()
        await bus.publish(EntityStateChanged(
            entity_id=entity_id,
            new_state=new_state,
            old_state=old_state,
            friendly_name=friendly_name,
            snapshot=snapshot,
        ))
        if not snapshot:
            logger.info(
                "HA entity %s: %s -> %s", entity_id, old_state, new_state
            )

    async def _publish_unknown(self) -> None:
        """Reset previously reported entities so stale states can't linger."""
        reported, self._reported = self._reported, set()
        bus = get_event_bus()
        for entity_id in reported:
            await bus.publish(EntityStateChanged(
                entity_id=entity_id,
                new_state="unknown",
                old_state="",
                friendly_name="",
                snapshot=True,
            ))
