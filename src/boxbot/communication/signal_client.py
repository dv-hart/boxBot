"""signal-cli daemon JSON-RPC client over a unix socket.

The Pi runs ``signal-cli daemon --socket /run/signal-cli/socket`` as a
systemd unit (see ``scripts/systemd/signal-cli.service``). This module
maintains a single long-lived connection to that socket and speaks
line-delimited JSON-RPC 2.0:

* Requests (``send``, ``subscribeReceive``, etc.) get a per-request id;
  responses come back on the same connection and are dispatched to the
  awaiting future by id.
* Notifications (server-pushed ``receive`` events) have no id; they're
  forwarded to a registered ``notification_handler``.

The connection auto-reconnects on failure with bounded backoff. In-flight
requests outstanding when the connection drops resolve with
``ConnectionError``; the caller decides whether to retry.

``SignalClient`` satisfies the ``OutboundChannel`` Protocol from
``boxbot.communication.channels`` so the output dispatcher can route to
it the same way it routes to WhatsApp.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Hard cap on inbound attachment bytes we'll read off disk. Defense in
# depth: signal-cli already bounds attachment sizes.
MAX_MEDIA_DOWNLOAD_BYTES = 20 * 1024 * 1024


# Default signal-cli attachment cache. The daemon downloads inbound
# attachments here automatically (we run without --ignore-attachments).
# Override via ``attachments_dir`` to ``SignalClient``.
DEFAULT_ATTACHMENTS_DIR = Path.home() / ".local" / "share" / "signal-cli" / "attachments"


# Singleton accessor — set during startup; read by the output dispatcher
# via the channel registry.
_signal_client: "SignalClient | None" = None


def get_signal_client() -> "SignalClient | None":
    """Return the process-wide SignalClient, or None if unset."""
    return _signal_client


def set_signal_client(client: "SignalClient | None") -> None:
    """Register the process-wide SignalClient.

    Also (un)registers the client in the channel-agnostic OutboundChannel
    registry so the dispatcher can reach it via ``Channel.SIGNAL``.
    """
    global _signal_client
    _signal_client = client
    # Local import — avoids a circular import at module load time.
    from boxbot.communication.channels import Channel, register_outbound_channel
    register_outbound_channel(Channel.SIGNAL, client)


NotificationHandler = Callable[[dict[str, Any]], Awaitable[None]]


class SignalClient:
    """Async JSON-RPC client for signal-cli daemon over a unix socket.

    Args:
        socket_path: Filesystem path of the daemon's unix socket.
        account: The registered Signal account phone (E.164). Sent as the
            ``account`` field on send requests for multi-account daemons;
            optional for single-account daemons but cheap to include.
        attachments_dir: Where signal-cli writes downloaded attachments.
            ``download_media`` reads from this directory.
        connect_timeout: Seconds to wait for the unix socket connect.
        request_timeout: Default seconds to wait for a JSON-RPC response.
    """

    # Satisfies OutboundChannel Protocol.
    name: str = "signal"

    def __init__(
        self,
        *,
        socket_path: str | Path = "/run/signal-cli/socket",
        account: str,
        attachments_dir: str | Path = DEFAULT_ATTACHMENTS_DIR,
        connect_timeout: float = 5.0,
        request_timeout: float = 30.0,
    ) -> None:
        self._socket_path = str(socket_path)
        self._account = account
        self._attachments_dir = Path(attachments_dir)
        self._connect_timeout = connect_timeout
        self._request_timeout = request_timeout

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()

        self._next_id = 1
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._notification_handler: NotificationHandler | None = None

        self._stopped = asyncio.Event()

    @property
    def account(self) -> str:
        return self._account

    def set_notification_handler(self, handler: NotificationHandler | None) -> None:
        """Register the callback invoked on every server-pushed notification.

        The handler receives the full notification dict; the inbound
        layer is responsible for filtering to ``receive`` events,
        extracting the envelope, and routing.
        """
        self._notification_handler = handler

    # -----------------------------------------------------------------
    # Connection lifecycle
    # -----------------------------------------------------------------

    async def connect(self) -> None:
        """Open the unix socket connection and start the reader task.

        Idempotent — no-op if already connected.
        """
        if self._reader_task is not None and not self._reader_task.done():
            return
        self._stopped.clear()
        await self._open()
        self._reader_task = asyncio.create_task(
            self._read_loop(), name="signal-cli-reader"
        )

    async def disconnect(self) -> None:
        """Close the connection and stop the reader task."""
        self._stopped.set()
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None
        if self._reader_task is not None:
            try:
                await asyncio.wait_for(self._reader_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._reader_task.cancel()
            self._reader_task = None
        # Fail any outstanding requests.
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("signal-cli client disconnected"))
        self._pending.clear()

    async def _open(self) -> None:
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self._socket_path),
                timeout=self._connect_timeout,
            )
        except (OSError, asyncio.TimeoutError) as e:
            raise ConnectionError(
                f"signal-cli daemon socket unreachable at "
                f"{self._socket_path}: {e}"
            ) from e
        logger.info("signal-cli daemon connected via %s", self._socket_path)

    async def _read_loop(self) -> None:
        """Read JSON-RPC frames from the daemon, dispatch by id or method.

        Handles connection loss by reconnecting with bounded backoff. Each
        time we reconnect, the notification handler must re-call
        ``subscribeReceive`` — the daemon does NOT remember prior
        subscriptions across socket disconnects.
        """
        backoff = 1.0
        while not self._stopped.is_set():
            if self._reader is None:
                try:
                    await self._open()
                    backoff = 1.0
                except ConnectionError:
                    logger.warning(
                        "signal-cli reconnect failed; retrying in %.1fs", backoff,
                    )
                    try:
                        await asyncio.wait_for(
                            self._stopped.wait(), timeout=backoff
                        )
                    except asyncio.TimeoutError:
                        pass
                    backoff = min(backoff * 2, 30.0)
                    continue

            try:
                line = await self._reader.readline()
            except (asyncio.CancelledError, ConnectionError):
                raise
            except Exception:
                logger.exception("signal-cli reader errored; reconnecting")
                line = b""

            if not line:
                # EOF — daemon closed the connection.
                logger.warning("signal-cli daemon disconnected; will reconnect")
                self._reader = None
                if self._writer is not None:
                    try:
                        self._writer.close()
                    except Exception:
                        pass
                    self._writer = None
                # Fail in-flight requests so callers don't hang forever.
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(
                            ConnectionError("signal-cli daemon dropped the connection")
                        )
                self._pending.clear()
                continue

            try:
                msg = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                logger.warning("signal-cli: unparseable frame: %r", line[:200])
                continue

            await self._dispatch_frame(msg)

    async def _dispatch_frame(self, msg: dict[str, Any]) -> None:
        """Dispatch one parsed JSON-RPC frame."""
        # Response — has an id that maps to a pending future.
        if "id" in msg and msg.get("id") in self._pending:
            fut = self._pending.pop(msg["id"])
            if fut.done():
                return
            if "error" in msg and msg["error"] is not None:
                fut.set_exception(
                    SignalRpcError(
                        msg["error"].get("code", -1),
                        msg["error"].get("message", "unknown error"),
                        msg["error"].get("data"),
                    )
                )
            else:
                fut.set_result(msg.get("result"))
            return

        # Notification — server-pushed, no matching id.
        if "method" in msg and self._notification_handler is not None:
            try:
                await self._notification_handler(msg)
            except Exception:
                logger.exception(
                    "signal-cli notification handler raised on method %s",
                    msg.get("method"),
                )
            return

        # Otherwise ignore — late response to a cancelled request, or
        # a notification with no handler attached.

    # -----------------------------------------------------------------
    # Request / response
    # -----------------------------------------------------------------

    async def _send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Send a JSON-RPC request and await its response.

        Raises ``ConnectionError`` if the socket isn't connected,
        ``asyncio.TimeoutError`` on response timeout, or
        ``SignalRpcError`` if the daemon returns an error.
        """
        if self._writer is None or self._reader is None:
            raise ConnectionError("signal-cli daemon not connected")

        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut

        payload = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        frame = (json.dumps(payload) + "\n").encode("utf-8")

        try:
            async with self._write_lock:
                self._writer.write(frame)
                await self._writer.drain()
        except Exception:
            self._pending.pop(req_id, None)
            raise

        try:
            return await asyncio.wait_for(
                fut, timeout=timeout or self._request_timeout
            )
        finally:
            self._pending.pop(req_id, None)

    # -----------------------------------------------------------------
    # OutboundChannel surface
    # -----------------------------------------------------------------

    async def send_text(self, phone: str, message: str) -> bool:
        """Send a text message to a single recipient. Returns True on success.

        signal-cli requires E.164 recipients (``+15035086292``) but
        boxBot's user records store phones without the leading ``+``
        (WhatsApp webhook convention). Normalise here so callers can
        pass either format.
        """
        params = {
            "account": self._account,
            "recipient": [_to_e164(phone)],
            "message": message,
        }
        try:
            await self._send_request("send", params)
            logger.debug("Signal text sent to %s", phone)
            return True
        except SignalRpcError as e:
            logger.error("Signal send_text(%s) failed: %s", phone, e)
            return False
        except (ConnectionError, asyncio.TimeoutError) as e:
            logger.error("Signal send_text(%s) transport error: %s", phone, e)
            return False

    async def send_attachment(
        self,
        phone: str,
        file_path: str,
        caption: str | None = None,
    ) -> bool:
        """Send a local file as an attachment, optionally with a caption."""
        params: dict[str, Any] = {
            "account": self._account,
            "recipient": [_to_e164(phone)],
            "attachment": [file_path],
        }
        if caption:
            params["message"] = caption
        try:
            await self._send_request("send", params)
            logger.debug("Signal attachment sent to %s (%s)", phone, file_path)
            return True
        except SignalRpcError as e:
            logger.error("Signal send_attachment(%s) failed: %s", phone, e)
            return False
        except (ConnectionError, asyncio.TimeoutError) as e:
            logger.error(
                "Signal send_attachment(%s) transport error: %s", phone, e
            )
            return False

    async def download_media(
        self, media_id: str
    ) -> tuple[bytes, str] | None:
        """Read an inbound attachment that signal-cli already downloaded.

        signal-cli's daemon writes downloaded attachments to its data
        directory (default ``~/.local/share/signal-cli/attachments/``).
        We just read the file off disk and return ``(bytes, mime_type)``.

        ``media_id`` is the ``id`` field from the receive-envelope
        attachment entry. Mime type is inferred from the file extension
        if the inbound notification didn't carry it — callers that have
        the mime in hand should pass it through other channels.
        """
        # media_id might be the bare id or already-prefixed; signal-cli
        # filenames are typically the id with no extension.
        candidates = [
            self._attachments_dir / media_id,
        ]
        # signal-cli historically also writes ``<id>.<ext>`` files for
        # known types — fall back to a directory glob.
        try:
            for path in self._attachments_dir.glob(f"{media_id}*"):
                if path not in candidates:
                    candidates.append(path)
        except OSError:
            pass

        for path in candidates:
            try:
                # Size-cap before reading so a huge attachment can't
                # balloon RAM on the Pi (defense in depth — signal-cli
                # already bounds these).
                if path.stat().st_size > MAX_MEDIA_DOWNLOAD_BYTES:
                    logger.warning(
                        "Signal attachment %s over %d-byte cap; skipping",
                        media_id, MAX_MEDIA_DOWNLOAD_BYTES,
                    )
                    return None
                data = path.read_bytes()
            except OSError:
                continue
            mime = _mime_from_extension(path)
            return data, mime
        logger.warning("Signal attachment %s not found under %s",
                       media_id, self._attachments_dir)
        return None

    # -----------------------------------------------------------------
    # Inbound subscription
    # -----------------------------------------------------------------

    async def subscribe_receive(self) -> int:
        """Register for server-pushed ``receive`` notifications.

        Returns the subscription id from the daemon. Notifications will
        flow to the handler set via ``set_notification_handler``.
        """
        result = await self._send_request(
            "subscribeReceive", {"account": self._account}
        )
        sub_id = int(result) if isinstance(result, (int, str)) else -1
        logger.info("signal-cli subscribeReceive id=%s", sub_id)
        return sub_id


class SignalRpcError(Exception):
    """Raised when signal-cli's JSON-RPC returns an error response."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"signal-cli error {code}: {message}")
        self.code = code
        self.signal_message = message
        self.data = data


_EXT_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".pdf": "application/pdf",
}


def _mime_from_extension(path: Path) -> str:
    ext = path.suffix.lower()
    return _EXT_MIME.get(ext, "application/octet-stream")


def _to_e164(phone: str) -> str:
    """signal-cli wants ``+15551234567``; boxBot stores ``15551234567``."""
    phone = phone.strip()
    return phone if phone.startswith("+") else f"+{phone}"
