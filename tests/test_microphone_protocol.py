"""Tests for the ReSpeaker pixel_ring USB protocol in microphone.py.

Covers the 2026-04-23 rewrite: LED writes now use the SHOW command
(cmd=6 + wIndex=0x1C + 48-byte RGBA payload) and DOA reads use the
parameter-ID encoding (0x80 | 0x40 | 21, int32 LE response). Both
paths are exercised with a mocked `_usb_device` so tests do not
require hardware.
"""

from __future__ import annotations

import struct
from unittest.mock import MagicMock

import pytest

from boxbot.hardware.microphone import (
    _CMD_SHOW,
    _LED_COUNT,
    _PARAM_DOAANGLE_ID,
    _PIXEL_RING_IFACE,
    _USB_REQUEST,
    _USB_REQUEST_TYPE_READ,
    _USB_REQUEST_TYPE_WRITE,
    Microphone,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mic(usb_device=None, doa_enabled: bool = True) -> Microphone:
    """Build a Microphone with a mocked USB device and no real hardware."""
    mic = Microphone(
        device_name="ReSpeaker",
        sample_rate=16000,
        capture_channels=6,
        output_channel=0,
        chunk_duration_ms=64,
        doa_enabled=doa_enabled,
        led_brightness=0.5,
    )
    mic._usb_device = usb_device
    return mic


# ---------------------------------------------------------------------------
# _set_leds_raw
# ---------------------------------------------------------------------------


class TestSetLedsRaw:
    def test_uses_cmd_show_and_pixel_ring_iface(self) -> None:
        """_set_leds_raw must send a single ctrl_transfer with the
        SHOW command and the pixel_ring wIndex (not per-LED register
        writes as the old protocol did)."""
        usb = MagicMock()
        mic = _make_mic(usb_device=usb)

        colors = [(10 * i, 20, 30) for i in range(_LED_COUNT)]
        mic._set_leds_raw(colors)

        usb.ctrl_transfer.assert_called_once()
        args, kwargs = usb.ctrl_transfer.call_args

        # Positional args per the rewrite:
        #   (bmRequestType, bRequest, wValue, wIndex, data, timeout)
        assert args[0] == _USB_REQUEST_TYPE_WRITE
        assert args[1] == _USB_REQUEST
        assert args[2] == _CMD_SHOW
        assert args[3] == _PIXEL_RING_IFACE

        payload = args[4]
        assert isinstance(payload, (bytes, bytearray))
        assert len(payload) == _LED_COUNT * 4  # 12 LEDs * 4 bytes (RGBA)

        # Timeout is the sixth positional arg, 8000ms per spec.
        assert args[5] == 8000

    def test_payload_is_rgba_with_full_alpha(self) -> None:
        """Each LED should serialize as (R, G, B, 0xFF)."""
        usb = MagicMock()
        mic = _make_mic(usb_device=usb)

        colors = [(1, 2, 3)] * _LED_COUNT
        mic._set_leds_raw(colors)

        payload = bytes(usb.ctrl_transfer.call_args[0][4])
        expected = bytes([1, 2, 3, 0xFF]) * _LED_COUNT
        assert payload == expected

    def test_pads_short_input(self) -> None:
        """Passing fewer than 12 colors should zero-pad the payload to
        the full 48-byte frame so the firmware never sees a short write."""
        usb = MagicMock()
        mic = _make_mic(usb_device=usb)

        mic._set_leds_raw([(255, 0, 0), (0, 255, 0)])

        payload = bytes(usb.ctrl_transfer.call_args[0][4])
        assert len(payload) == _LED_COUNT * 4
        # First two LEDs populated, the rest zeroed.
        assert payload[0:4] == bytes([255, 0, 0, 0xFF])
        assert payload[4:8] == bytes([0, 255, 0, 0xFF])
        assert payload[8:] == bytes(_LED_COUNT * 4 - 8)

    def test_no_op_when_no_usb_device(self) -> None:
        """With _usb_device=None the call must silently return — no
        exception, no ctrl_transfer."""
        mic = _make_mic(usb_device=None)
        # Should not raise.
        mic._set_leds_raw([(1, 2, 3)] * _LED_COUNT)

    def test_write_failure_logs_warning_then_debug(self, caplog) -> None:
        """First USB failure logs WARNING, subsequent failures DEBUG
        (throttling to prevent 30 FPS log spam)."""
        import logging

        usb = MagicMock()
        usb.ctrl_transfer.side_effect = RuntimeError("STALL")
        mic = _make_mic(usb_device=usb)

        with caplog.at_level(logging.DEBUG, logger="boxbot.hardware.microphone"):
            mic._set_leds_raw([(1, 2, 3)] * _LED_COUNT)
            mic._set_leds_raw([(1, 2, 3)] * _LED_COUNT)
            mic._set_leds_raw([(1, 2, 3)] * _LED_COUNT)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert mic._usb_write_failure_count == 3


# ---------------------------------------------------------------------------
# get_doa
# ---------------------------------------------------------------------------


class TestGetDoa:
    def test_uses_param_id_encoding(self) -> None:
        """get_doa must use wValue = 0x80 | 0x40 | DOAANGLE id, not
        the old raw register read."""
        usb = MagicMock()
        # Response: int32 LE angle, int32 LE max.
        usb.ctrl_transfer.return_value = struct.pack("<ii", 180, 360)
        mic = _make_mic(usb_device=usb)

        angle = mic.get_doa()
        assert angle == 180

        usb.ctrl_transfer.assert_called_once()
        args = usb.ctrl_transfer.call_args[0]
        # (bmRequestType, bRequest, wValue, wIndex, wLength, timeout)
        assert args[0] == _USB_REQUEST_TYPE_READ
        assert args[1] == _USB_REQUEST
        assert args[2] == (0x80 | 0x40 | _PARAM_DOAANGLE_ID)
        assert args[3] == 0
        assert args[4] == 8
        assert args[5] == 8000

    def test_parses_int32_le(self) -> None:
        """The first 4 bytes of the response should be decoded as
        little-endian int32."""
        usb = MagicMock()
        # 270 degrees LE, plus 8 bytes of extra trailing data.
        usb.ctrl_transfer.return_value = struct.pack("<ii", 270, 999)
        mic = _make_mic(usb_device=usb)
        assert mic.get_doa() == 270

        usb.ctrl_transfer.return_value = struct.pack("<ii", 0, 0)
        assert mic.get_doa() == 0

        usb.ctrl_transfer.return_value = struct.pack("<ii", 359, 0)
        assert mic.get_doa() == 359

    def test_rejects_out_of_range_angle(self) -> None:
        """Angles outside [0, 360) should be treated as invalid."""
        usb = MagicMock()
        usb.ctrl_transfer.return_value = struct.pack("<ii", 400, 0)
        mic = _make_mic(usb_device=usb)
        assert mic.get_doa() is None

        usb.ctrl_transfer.return_value = struct.pack("<ii", -1, 0)
        assert mic.get_doa() is None

    def test_returns_none_when_doa_disabled(self) -> None:
        mic = _make_mic(usb_device=MagicMock(), doa_enabled=False)
        assert mic.get_doa() is None

    def test_returns_none_when_no_usb_device(self) -> None:
        mic = _make_mic(usb_device=None)
        assert mic.get_doa() is None

    def test_returns_none_on_short_response(self) -> None:
        """If the firmware returns fewer than 4 bytes we can't decode
        an int32 — return None instead of raising."""
        usb = MagicMock()
        usb.ctrl_transfer.return_value = bytes([1, 2])
        mic = _make_mic(usb_device=usb)
        assert mic.get_doa() is None

    def test_returns_none_on_exception(self) -> None:
        """USB I/O errors during DOA reads must not propagate."""
        usb = MagicMock()
        usb.ctrl_transfer.side_effect = RuntimeError("Errno 5 I/O Error")
        mic = _make_mic(usb_device=usb)
        assert mic.get_doa() is None


# ---------------------------------------------------------------------------
# Protocol constants sanity check
# ---------------------------------------------------------------------------


def test_protocol_constants() -> None:
    """Pin the protocol constants so accidental edits get caught."""
    assert _PIXEL_RING_IFACE == 0x1C
    assert _CMD_SHOW == 6
    assert _PARAM_DOAANGLE_ID == 21
    assert _USB_REQUEST_TYPE_WRITE == 0x40
    assert _USB_REQUEST_TYPE_READ == 0xC0
    assert _LED_COUNT == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
