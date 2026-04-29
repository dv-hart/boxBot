"""bb.camera — see what's in front of boxBot.

The camera is a shared hardware resource; these calls go through the
main-process HAL, which serializes with the perception pipeline.
Captures are written to disk and returned as image content blocks
attached to the tool result so the agent actually sees the pixels.

Common patterns:

    # Grab what's in front of boxBot and look at it
    bb.camera.capture()

    # Capture at full sensor resolution (slower, for detailed photos)
    bb.camera.capture(full_res=True)

    # Persist the capture into the workspace for later reference
    bb.camera.capture(save_to="captures/erik_2026-04-24.jpg")

    # Crop to a person's bounding box (e.g., from perception)
    bb.camera.capture_cropped(
        bbox={"x": 420, "y": 80, "w": 180, "h": 240},
        save_to="notes/people/erik/headshot.jpg",
    )

Every capture you make is attached to the tool result — if you call
`capture()` three times in one script, you get three images back. The
cap is 8 images per execute_script call.
"""

from __future__ import annotations

from typing import Any

from . import _transport, _validators as v


_TIMEOUT = 20


class CameraError(Exception):
    """Raised when the camera is unavailable or the capture fails."""


def _check(resp: dict[str, Any]) -> dict[str, Any]:
    if isinstance(resp, dict) and resp.get("status") == "error":
        raise CameraError(resp.get("error", "camera error"))
    return resp


def capture(
    *,
    save_to: str | None = None,
    full_res: bool = False,
) -> dict[str, Any]:
    """Capture a still from the camera.

    Args:
        save_to: Optional workspace-relative path. If provided, the
            capture is persisted there. If omitted, it goes to an
            ephemeral tmp area (still attached to the tool result so
            you can look at it this turn).
        full_res: If True, use the full-sensor still mode (up to 12 MP,
            ~200 ms mode switch). Default False uses the preview stream
            (720 p, <50 ms) — fine for human viewing and perception.

    Returns:
        ``{ref, width, height, saved, path}``. ``path`` is the workspace
        path when ``save_to`` was given, otherwise the internal tmp ref.
    """
    payload: dict[str, Any] = {"full_res": bool(full_res)}
    if save_to is not None:
        v.require_str(save_to, "save_to")
        payload["save_to"] = save_to
    return _check(_transport.request("camera.capture", payload, timeout=_TIMEOUT))


def capture_cropped(
    bbox: dict[str, int],
    *,
    save_to: str | None = None,
    full_res: bool = False,
) -> dict[str, Any]:
    """Capture a still and crop to a bounding box.

    Args:
        bbox: ``{x, y, w, h}`` in the main-stream coordinate space
            (top-left origin). Values are clamped to image bounds.
        save_to: As in :func:`capture`.
        full_res: As in :func:`capture`. If True, the bbox is treated
            as being in full-sensor coordinates.
    """
    v.require_dict(bbox, "bbox")
    for k in ("x", "y", "w", "h"):
        if k not in bbox:
            raise CameraError(f"bbox missing '{k}'")
        v.require_int(bbox[k], f"bbox.{k}", min_val=0)
    payload: dict[str, Any] = {
        "bbox": {k: int(bbox[k]) for k in ("x", "y", "w", "h")},
        "full_res": bool(full_res),
    }
    if save_to is not None:
        v.require_str(save_to, "save_to")
        payload["save_to"] = save_to
    return _check(_transport.request("camera.capture_cropped", payload, timeout=_TIMEOUT))
