"""Minimal Chrome DevTools Protocol client over a pipe pair.

Chromium started with ``--remote-debugging-pipe`` reads commands on fd 3
and writes replies/events on fd 4, each message a JSON object terminated
by a NUL byte. This is the only control channel to the kiosk browser —
there is no debugging TCP port, so nothing listens on the network.

Only what :mod:`boxbot.displays.web_view` needs: send a command, await
its reply, and wait for a named event.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_READ_CHUNK = 65536
# A screenshot of a 1024x600 page is ~1-3 MB of base64; anything far past
# that is a runaway peer, not a message worth buffering.
_MAX_MESSAGE_BYTES = 32 * 1024 * 1024


class CDPError(RuntimeError):
    """The browser rejected a command, or the pipe closed."""


class CDPPipe:
    """One DevTools connection over two already-open file descriptors.

    ``write_fd`` carries commands to the browser; ``read_fd`` carries its
    replies and events. The caller owns both fds and closes them after
    :meth:`close`. ``write_fd`` should be non-blocking.
    """

    def __init__(self, write_fd: int, read_fd: int) -> None:
        self._write_fd = write_fd
        self._read_fd = read_fd
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._event_waiters: list[tuple[str, str | None, asyncio.Future[dict[str, Any]]]] = []
        self._buffer = bytearray()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        """Begin reading replies. Must be called from the event loop."""
        self._loop = asyncio.get_running_loop()
        os.set_blocking(self._read_fd, False)
        self._loop.add_reader(self._read_fd, self._on_readable)

    def close(self) -> None:
        """Stop reading and fail every outstanding call."""
        if self._closed:
            return
        self._closed = True
        if self._loop is not None:
            try:
                self._loop.remove_reader(self._read_fd)
            except (ValueError, OSError):
                pass
        error = CDPError("devtools pipe closed")
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()
        for _, _, future in self._event_waiters:
            if not future.done():
                future.set_exception(error)
        self._event_waiters.clear()

    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Send one command and return its ``result`` object."""
        if self._closed or self._loop is None:
            raise CDPError("devtools pipe is not open")
        self._next_id += 1
        call_id = self._next_id
        message: dict[str, Any] = {"id": call_id, "method": method}
        if params:
            message["params"] = params
        if session_id:
            message["sessionId"] = session_id

        future: asyncio.Future[dict[str, Any]] = self._loop.create_future()
        self._pending[call_id] = future
        data = json.dumps(message).encode() + b"\0"
        try:
            # The fd is non-blocking: a browser that has stopped reading
            # must surface as an error here, never as an event loop
            # stuck in write().
            written = os.write(self._write_fd, data)
        except BlockingIOError as e:
            self._pending.pop(call_id, None)
            raise CDPError("devtools pipe is full: browser not reading") from e
        except OSError as e:
            self._pending.pop(call_id, None)
            raise CDPError(f"devtools write failed: {e}") from e
        if written != len(data):
            # A torn frame would desynchronise every message after it.
            self._pending.pop(call_id, None)
            self.close()
            raise CDPError("devtools write was cut short")

        try:
            reply = await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError as e:
            raise CDPError(f"{method} timed out after {timeout:.0f}s") from e
        finally:
            self._pending.pop(call_id, None)

        if "error" in reply:
            detail = reply["error"]
            text = detail.get("message") if isinstance(detail, dict) else detail
            raise CDPError(f"{method} failed: {text}")
        result = reply.get("result")
        return result if isinstance(result, dict) else {}

    async def wait_event(
        self,
        method: str,
        *,
        session_id: str | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any] | None:
        """Wait for the next ``method`` event. ``None`` on timeout."""
        if self._closed or self._loop is None:
            raise CDPError("devtools pipe is not open")
        future: asyncio.Future[dict[str, Any]] = self._loop.create_future()
        entry = (method, session_id, future)
        self._event_waiters.append(entry)
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            if entry in self._event_waiters:
                self._event_waiters.remove(entry)

    # ------------------------------------------------------------------

    def _on_readable(self) -> None:
        try:
            chunk = os.read(self._read_fd, _READ_CHUNK)
        except BlockingIOError:
            return
        except OSError as e:
            logger.debug("devtools read failed: %s", e)
            self.close()
            return
        if not chunk:
            self.close()
            return

        self._buffer.extend(chunk)
        if len(self._buffer) > _MAX_MESSAGE_BYTES:
            logger.warning("devtools message exceeded %d bytes", _MAX_MESSAGE_BYTES)
            self.close()
            return

        while True:
            end = self._buffer.find(b"\0")
            if end < 0:
                return
            raw = bytes(self._buffer[:end])
            del self._buffer[: end + 1]
            try:
                message = json.loads(raw)
            except ValueError:
                logger.debug("devtools sent unparseable message (%d bytes)", len(raw))
                continue
            if isinstance(message, dict):
                self._dispatch(message)

    def _dispatch(self, message: dict[str, Any]) -> None:
        call_id = message.get("id")
        if isinstance(call_id, int):
            future = self._pending.get(call_id)
            if future is not None and not future.done():
                future.set_result(message)
            return

        method = message.get("method")
        if not isinstance(method, str):
            return
        session_id = message.get("sessionId")
        params = message.get("params")
        for entry in list(self._event_waiters):
            want_method, want_session, future = entry
            if want_method != method or future.done():
                continue
            if want_session is not None and want_session != session_id:
                continue
            future.set_result(params if isinstance(params, dict) else {})
            self._event_waiters.remove(entry)
