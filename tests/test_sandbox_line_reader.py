"""Regression tests for :func:`read_sandbox_line`.

A sandbox script can ``print()`` an arbitrarily large blob on a single
line (e.g. raw image bytes). asyncio's ``StreamReader.readline()`` raises
``ValueError`` (``LimitOverrunError``) once a line exceeds its buffer
limit; unhandled, that crashed the whole ``execute_script`` tool call —
the failure mode seen when a ~944 KB inbound image was echoed to stdout.

These tests pin the contract: lines under the limit pass through intact,
over-limit lines are dropped (not raised), and the stream resyncs so the
following protocol line is still read.
"""

from __future__ import annotations

import asyncio

from boxbot.tools._sandbox_actions import (
    SANDBOX_STREAM_LIMIT,
    read_sandbox_line,
)

DONE_MARKER = b"__BOXBOT_SANDBOX_DONE__:"


def _reader() -> asyncio.StreamReader:
    return asyncio.StreamReader(limit=SANDBOX_STREAM_LIMIT)


async def test_normal_lines_pass_through():
    r = _reader()
    r.feed_data(b"hello\nworld\n")
    r.feed_eof()
    assert await read_sandbox_line(r) == b"hello\n"
    assert await read_sandbox_line(r) == b"world\n"
    assert await read_sandbox_line(r) == b""  # EOF


async def test_large_but_under_limit_line_intact():
    """The reported scenario: a ~944 KB blob on one line is well under the
    8 MiB buffer, so it must come back whole and not disturb later lines."""
    blob = b"X" * (944 * 1024)
    r = _reader()
    r.feed_data(blob + b"\n" + b"after\n")
    r.feed_eof()
    assert await read_sandbox_line(r) == blob + b"\n"
    assert await read_sandbox_line(r) == b"after\n"


async def test_oversized_line_dropped_not_raised():
    """A line past the limit yields a non-empty marker (never raises, never
    fakes EOF) and the next line is still recoverable."""
    huge = b"Y" * (SANDBOX_STREAM_LIMIT + 3 * 1024 * 1024)
    r = _reader()
    r.feed_data(huge + b"\n" + b"recovered\n")
    r.feed_eof()
    dropped = await read_sandbox_line(r)
    assert dropped != b""  # not EOF
    assert b"oversized" in dropped
    assert await read_sandbox_line(r) == b"recovered\n"


async def test_oversized_line_streamed_in_chunks_reaches_done():
    """Mirror the real pipe on the Pi: the oversized line arrives in many
    small chunks. The pump must never crash and must still reach the
    trailing DONE marker that terminates the protocol."""
    r = _reader()

    async def feeder() -> None:
        sent = 0
        total = SANDBOX_STREAM_LIMIT + 3 * 1024 * 1024
        while sent < total:
            r.feed_data(b"Z" * (256 * 1024))
            sent += 256 * 1024
            await asyncio.sleep(0)
        r.feed_data(b"\n" + DONE_MARKER + b'{"status":"ok"}\n')
        r.feed_eof()

    feed_task = asyncio.create_task(feeder())
    saw_done = False
    for _ in range(1000):
        line = await read_sandbox_line(r)
        if not line:
            break
        if line.startswith(DONE_MARKER):
            saw_done = True
            break
    await feed_task
    assert saw_done
