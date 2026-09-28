"""Dev/eval "local chat" channel that spoofs Signal over plain TCP.

This exists so a developer (or an automated eval) can hold a real
conversation with the agent on a laptop with no signal-cli daemon, no
Signal account, and no phone. It reuses the Signal seam end to end —
``Channel.SIGNAL`` inbound routing and the ``OutboundChannel`` registry
— so the agent, router, auth, and output dispatcher behave exactly as
they do in production. There is deliberately **no new Channel value**.

Wire protocol — line-delimited UTF-8 over a TCP socket:

* Client → BB: each non-blank line is one inbound message, routed as if
  it arrived on Signal from the configured dev user.
* BB → client: every outbound send is broadcast to all connected
  clients as ``BB> <text>\\n`` so an eval client can parse it trivially.

Enabled via ``signal.driver = "local"``; the default ``"cli"`` path is
the real daemon and is untouched. Connect with any line client, e.g.::

    nc 127.0.0.1 8765
"""

from __future__ import annotations

import asyncio
import logging

from boxbot.communication.auth import AuthManager, get_auth_manager
from boxbot.communication.channels import (
    Channel,
    register_outbound_channel,
)
from boxbot.communication.router import MessageRouter

logger = logging.getLogger(__name__)


# Singleton accessor — mirrors signal_client's get/set pattern so the two
# main.py seams (_init_signal_client, _init_signal_inbound) reach the one
# instance: one builds+registers it, the other starts its server.
_local_chat: "LocalChatChannel | None" = None


def get_local_chat_channel() -> "LocalChatChannel | None":
    """Return the process-wide LocalChatChannel, or None if unset."""
    return _local_chat


def set_local_chat_channel(channel: "LocalChatChannel | None") -> None:
    """Register the process-wide LocalChatChannel."""
    global _local_chat
    _local_chat = channel


class LocalChatChannel:
    """A TCP line server that impersonates the Signal transport.

    Satisfies the ``OutboundChannel`` Protocol (``name == "signal"``) so
    the output dispatcher routes agent replies here via ``Channel.SIGNAL``.
    Inbound lines are fed to ``MessageRouter.route_incoming`` under the
    same channel, attributed to the configured dev user.

    Args:
        host: Bind address for the TCP server.
        port: Bind port. ``0`` picks a free port (read back via ``port``).
        dev_phone: Phone the dev user is registered under; inbound lines
            are attributed to it. Must be a registered user or the router
            silently drops the message (unknown-number policy).
        dev_name: Display name for the dev user.
        router: The MessageRouter inbound lines are routed through. May be
            set later via ``set_router`` (main builds the channel before
            the router exists).
        auth: AuthManager used to ensure the dev user exists. Falls back
            to the process-wide manager at ``start`` if not provided.
    """

    # Satisfies OutboundChannel Protocol — deliberately the Signal name.
    name: str = "signal"

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        dev_phone: str = "15550000000",
        dev_name: str = "Dev",
        router: MessageRouter | None = None,
        auth: AuthManager | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._dev_phone = dev_phone
        self._dev_name = dev_name
        self._router = router
        self._auth = auth

        self._server: asyncio.AbstractServer | None = None
        self._clients: set[asyncio.StreamWriter] = set()
        self._stopped = asyncio.Event()

    def set_router(self, router: MessageRouter) -> None:
        """Bind the router used for inbound routing (main wires it late)."""
        self._router = router

    @property
    def port(self) -> int:
        """Actual listening port (resolves ``0`` after the server binds)."""
        if self._server is not None and self._server.sockets:
            return self._server.sockets[0].getsockname()[1]
        return self._port

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def start(self) -> None:
        """Ensure the dev user, start the TCP server, register outbound.

        Idempotent: a second call is a no-op. Without the guard a repeat
        start binds a second server — EADDRINUSE on the configured
        ``signal.local_port``, and on an ephemeral port it silently
        leaks the first one (``self._server`` is overwritten, so
        ``stop()`` only closes the second).
        """
        if self._server is not None:
            logger.debug(
                "local chat channel already listening on %s:%d",
                self._host, self.port,
            )
            return

        self._stopped.clear()
        auth = self._auth or get_auth_manager()
        if auth is not None:
            await self._ensure_dev_user(auth)
        else:
            logger.warning(
                "local_chat: no AuthManager available; dev user not ensured "
                "(inbound lines will be dropped as an unknown number)"
            )

        self._server = await asyncio.start_server(
            self._handle_client, self._host, self._port
        )
        register_outbound_channel(Channel.SIGNAL, self)
        logger.info(
            "local chat channel listening on %s:%d (dev user %s / %s)",
            self._host,
            self.port,
            self._dev_name,
            self._dev_phone,
        )

    async def stop(self) -> None:
        """Close all clients and the server; unregister outbound."""
        self._stopped.set()
        for writer in list(self._clients):
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass
        self._clients.clear()
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # noqa: BLE001
                pass
            self._server = None
        register_outbound_channel(Channel.SIGNAL, None)

    async def _ensure_dev_user(self, auth: AuthManager) -> None:
        """Idempotently register the dev user so route_incoming accepts it.

        The router silently drops unknown numbers, so the dev phone must be
        a genuine registered user. Uses only the public auth flow — mint a
        single-use code and consume it — so authorization is faithful to
        real registration. Guarded by ``get_user`` so it fires at most once
        per user's existence. Dev-only (driver="local").
        """
        if await auth.get_user(self._dev_phone) is not None:
            return
        if not await auth.has_admins():
            # Fresh box: the dev user bootstraps as the first admin.
            code = await auth.generate_bootstrap_code()
        else:
            # An admin already exists; borrow one as the code's creator.
            admin = next(
                u for u in await auth.list_users() if u.role == "admin"
            )
            code = await auth.generate_registration_code(created_by=admin.phone)
        result = await auth.register_user(
            self._dev_phone, self._dev_name, code, channel="signal"
        )
        if result.success:
            logger.info(
                "local_chat: registered dev user %s (%s)",
                self._dev_name,
                self._dev_phone,
            )
        else:
            logger.error(
                "local_chat: could not register dev user: %s", result.error
            )

    # -----------------------------------------------------------------
    # Inbound: client → BB
    # -----------------------------------------------------------------

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Serve one connected client until it disconnects.

        Survives disconnect/reconnect: on EOF we drop the writer and
        return; the server keeps accepting new connections.
        """
        peer = writer.get_extra_info("peername")
        self._clients.add(writer)
        logger.info("local_chat: client connected %s", peer)
        try:
            while not self._stopped.is_set():
                line = await reader.readline()
                if not line:
                    break  # EOF — client disconnected
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                await self._route(text)
        except (ConnectionError, asyncio.CancelledError):
            pass
        except Exception:  # noqa: BLE001
            logger.exception("local_chat: client handler error")
        finally:
            self._clients.discard(writer)
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass
            logger.info("local_chat: client disconnected %s", peer)

    async def _route(self, text: str) -> None:
        """Route one inbound line as a Signal message from the dev user."""
        if self._router is None:
            logger.warning("local_chat: no router bound; dropping inbound line")
            return
        await self._router.route_incoming(
            Channel.SIGNAL,
            self._dev_phone,
            text,
            sender_name=self._dev_name,
        )

    # -----------------------------------------------------------------
    # Outbound: BB → client (OutboundChannel surface)
    # -----------------------------------------------------------------

    async def send_text(self, phone: str, message: str) -> bool:
        """Broadcast an agent reply to every connected client. Always True."""
        await self._broadcast(f"BB> {message}")
        return True

    async def send_attachment(
        self,
        phone: str,
        file_path: str,
        caption: str | None = None,
    ) -> bool:
        """Broadcast a note that an attachment was sent. Always True."""
        note = f"BB> [attachment: {file_path}]"
        if caption:
            note += f" {caption}"
        await self._broadcast(note)
        return True

    async def download_media(self, media_id: str) -> tuple[bytes, str] | None:
        """No inbound media over the local chat transport."""
        return None

    async def _broadcast(self, line: str) -> None:
        """Write a newline-terminated line to all clients; prune dead ones."""
        data = (line + "\n").encode("utf-8")
        dead: list[asyncio.StreamWriter] = []
        for writer in list(self._clients):
            try:
                writer.write(data)
                await writer.drain()
            except Exception:  # noqa: BLE001
                dead.append(writer)
        for writer in dead:
            self._clients.discard(writer)
