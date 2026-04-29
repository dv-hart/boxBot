"""Action dispatcher for sandbox SDK calls.

``execute_script`` streams lines from the sandbox's stdout, identifies SDK
action lines (``__BOXBOT_SDK_ACTION__:{...}``), and routes each to the
appropriate handler here. Handlers return a dict that is:

  (a) written back to the sandbox on stdin as a JSON line, *if* the action
      was tagged with ``_expects_response`` (paired with
      :func:`boxbot_sdk._transport.collect_response` on the other side);
  (b) always collected into the ``sdk_actions`` array of the tool result
      so the agent can see what happened.

Handlers may also mutate :class:`ActionContext` — most importantly to add
absolute file paths to ``ctx.image_attachments``. After the script
completes, ``execute_script`` turns those paths into API image content
blocks interleaved into the tool result, so the agent actually sees the
pixels (a camera capture, a viewed photo, a crop of a speaker, …).

Security:
- Image attachments are restricted to an allowlist of filesystem roots
  (:data:`ATTACH_ROOTS`). The sandbox cannot coerce the main process into
  reading arbitrary files.
- Size cap per image: 4 MB (Anthropic accepts up to 5 MB; we leave slack
  for base64 overhead and metadata).
- Count cap per tool call: 8 images.
"""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Filesystem roots from which images may be attached to tool results.
# Any absolute path outside these roots is refused. Project-anchored
# roots come from ``boxbot.core.paths`` so they resolve to the same
# location regardless of process CWD. The sandbox tmp root resolves
# against ``cfg.sandbox`` so the same code works whether the sandbox
# lives under ``/var/lib/...`` (production) or somewhere else (dev
# override).
def _project_attach_roots() -> tuple[Path, ...]:
    from boxbot.core.paths import (
        PERCEPTION_CROPS_DIR,
        PHOTOS_DIR,
        WORKSPACE_DIR,
    )

    return (WORKSPACE_DIR, PHOTOS_DIR, PERCEPTION_CROPS_DIR)


def _attach_roots() -> tuple[Path, ...]:
    roots: list[Path] = [r.resolve() for r in _project_attach_roots()]
    roots.append(_sandbox_tmp_dir().resolve())
    return tuple(roots)


def _sandbox_tmp_dir() -> Path:
    """Resolve the sandbox tmp dir from config, with a sensible fallback.

    Avoids hardcoding a path so deployers can place the sandbox tree
    wherever their distro prefers (``/var/lib/boxbot-sandbox`` by
    default).
    """
    try:
        from boxbot.core.config import get_config

        return Path(get_config().sandbox.tmp_dir)
    except Exception:
        return Path("/var/lib/boxbot-sandbox/tmp")

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGES_PER_CALL = 8


@dataclass
class ActionContext:
    """State accumulated across a single ``execute_script`` run."""

    image_attachments: list[Path] = field(default_factory=list)
    # Every action processed, mirrored into the final tool result so the
    # agent can observe side effects (e.g. "photos.set_tags: ok").
    action_log: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Image block construction
# ---------------------------------------------------------------------------


def _is_attach_allowed(abs_path: Path) -> bool:
    try:
        resolved = abs_path.resolve()
    except OSError:
        return False
    for root in _attach_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def build_image_block(abs_path: Path) -> dict[str, Any] | None:
    """Return an API ``image`` content block for ``abs_path``, or None.

    Returns None on any of: path outside allowlist, unreadable, too big,
    unknown/unsupported media type.
    """
    if not _is_attach_allowed(abs_path):
        logger.warning("image attach refused (outside allowlist): %s", abs_path)
        return None
    try:
        size = abs_path.stat().st_size
    except OSError as e:
        logger.warning("image attach failed (stat): %s: %s", abs_path, e)
        return None
    if size > MAX_IMAGE_BYTES:
        logger.warning(
            "image attach refused (too large, %d bytes): %s", size, abs_path
        )
        return None
    mt, _ = mimetypes.guess_type(str(abs_path))
    if mt not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
        logger.warning("image attach refused (unsupported type %s): %s", mt, abs_path)
        return None
    try:
        data = abs_path.read_bytes()
    except OSError as e:
        logger.warning("image attach failed (read): %s: %s", abs_path, e)
        return None
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mt,
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


# ---------------------------------------------------------------------------
# Workspace action handler
# ---------------------------------------------------------------------------


def _handle_workspace_action(
    action_type: str,
    payload: dict[str, Any],
    ctx: ActionContext,
) -> dict[str, Any]:
    """Dispatch workspace.* actions to :class:`boxbot.workspace.Workspace`."""
    from boxbot.workspace import Workspace, WorkspaceError

    ws = Workspace()
    sub = action_type.split(".", 1)[1] if "." in action_type else action_type

    try:
        if sub == "write":
            content: str | bytes
            if "b64" in payload:
                content = base64.b64decode(payload["b64"])
            else:
                content = payload.get("content", "")
            return {"status": "ok", **ws.write(payload["path"], content)}

        if sub == "append":
            return {
                "status": "ok",
                **ws.append(payload["path"], payload.get("content", "")),
            }

        if sub == "read":
            result = ws.read(payload["path"], binary=bool(payload.get("binary")))
            return {"status": "ok", **result}

        if sub == "ls":
            entries = ws.ls(payload.get("path"))
            return {"status": "ok", "entries": entries}

        if sub == "exists":
            return {"status": "ok", **ws.exists(payload["path"])}

        if sub == "search":
            hits = ws.search(
                payload["query"],
                path=payload.get("path"),
                limit=int(payload.get("limit", 50)),
                case_insensitive=bool(payload.get("case_insensitive", True)),
            )
            return {"status": "ok", "hits": hits}

        if sub == "view":
            result = ws.view(payload["path"])
            # Image views: pop the absolute path and queue for attachment;
            # do not leak it back to the sandbox.
            if result.get("kind") == "image":
                abs_path = Path(result.pop("absolute_path"))
                if len(ctx.image_attachments) < MAX_IMAGES_PER_CALL:
                    ctx.image_attachments.append(abs_path)
                else:
                    logger.warning(
                        "image attach refused (exceeded %d/call)",
                        MAX_IMAGES_PER_CALL,
                    )
                result["attached"] = True
            return {"status": "ok", **result}

        if sub == "delete":
            return {"status": "ok", **ws.delete(payload["path"])}

        if sub == "csv_write":
            return {
                "status": "ok",
                **ws.csv_write(
                    payload["path"],
                    payload.get("rows", []),
                    fieldnames=payload.get("fieldnames"),
                ),
            }

        if sub == "csv_append":
            return {
                "status": "ok",
                **ws.csv_append(payload["path"], payload.get("row", {})),
            }

        if sub == "csv_read":
            rows = ws.csv_read(payload["path"])
            return {"status": "ok", "rows": rows}

        return {
            "status": "error",
            "error": f"unknown workspace action: {action_type}",
        }

    except WorkspaceError as e:
        return {"status": "error", "error": str(e)}
    except KeyError as e:
        return {"status": "error", "error": f"missing field: {e}"}
    except Exception as e:  # noqa: BLE001
        logger.exception("workspace.%s failed", sub)
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Photos action handler
# ---------------------------------------------------------------------------


async def _handle_photos_action(
    action_type: str,
    payload: dict[str, Any],
    ctx: ActionContext,
) -> dict[str, Any]:
    from boxbot.photos.search import resolve_photo_path, search_photos

    try:
        if action_type == "photos.search":
            result = await search_photos(
                mode="search",
                query=payload.get("query"),
                tags=payload.get("tags"),
                people=payload.get("people"),
                date_from=payload.get("date_from"),
                date_to=payload.get("date_to"),
                source=payload.get("source"),
                in_slideshow=payload.get("in_slideshow"),
                include_deleted=bool(payload.get("include_deleted", False)),
                limit=int(payload.get("limit", 20)),
            )
            if "error" in result:
                return {"status": "error", "error": result["error"]}
            return {"status": "ok", **result}

        if action_type == "photos.get":
            pid = payload.get("id")
            if not pid:
                return {"status": "error", "error": "photo id is required"}
            result = await search_photos(mode="get", photo_id=pid)
            if "error" in result:
                return {"status": "error", "error": result["error"]}
            return {"status": "ok", **result}

        if action_type == "photos.view":
            pid = payload.get("id")
            if not pid:
                return {"status": "error", "error": "photo id is required"}
            abs_path = await resolve_photo_path(pid)
            if abs_path is None or not abs_path.exists():
                return {
                    "status": "error",
                    "error": f"photo {pid} not found on disk",
                }
            if len(ctx.image_attachments) < MAX_IMAGES_PER_CALL:
                ctx.image_attachments.append(abs_path)
                attached = True
            else:
                attached = False
                logger.warning(
                    "photos.view not attached (exceeded %d/call)",
                    MAX_IMAGES_PER_CALL,
                )
            return {
                "status": "ok",
                "id": pid,
                "filename": abs_path.name,
                "kind": "image",
                "attached": attached,
            }

        if action_type == "photos.show_on_screen":
            ids = payload.get("photo_ids") or []
            if not isinstance(ids, list) or not ids:
                return {
                    "status": "error",
                    "error": "photo_ids must be a non-empty list",
                }
            from boxbot.displays.manager import get_display_manager

            mgr = get_display_manager()
            if mgr is None:
                # Display system not wired up yet — log intent and return
                # a best-effort response so the agent can move on.
                logger.info(
                    "photos.show_on_screen queued (no DisplayManager): %s", ids
                )
                return {
                    "status": "ok",
                    "dispatched": False,
                    "reason": "display manager not running",
                    "photo_ids": ids,
                }
            ok = await mgr.switch("picture", args={"image_ids": ids})
            return {
                "status": "ok" if ok else "error",
                "dispatched": bool(ok),
                "photo_ids": ids,
            }

        # Fire-and-forget ops (tags, slideshow, delete, …) — acknowledge
        # without actually mutating, to match current backend completeness.
        return {
            "status": "stub",
            "message": f"{action_type} acknowledged but not yet wired.",
        }

    except KeyError as e:
        return {"status": "error", "error": f"missing field: {e}"}
    except Exception as e:  # noqa: BLE001
        logger.exception("%s failed", action_type)
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Camera action handler
# ---------------------------------------------------------------------------


def _tmp_capture_dir() -> Path:
    """Resolve the sandbox tmp dir at call time (not import time).

    See :func:`_sandbox_tmp_dir` — same function under a different name
    used by the camera capture handler. Kept as a separate accessor in
    case the capture path ever needs to diverge (e.g. a faster scratch
    on tmpfs).
    """
    return _sandbox_tmp_dir()


def _test_pattern_frame(width: int = 640, height: int = 360):
    """A solid-color frame used when no Camera HAL is available.

    Lets the image-attach pipeline be exercised end-to-end on dev
    machines without a Pi. Returns an ``(H, W, 3)`` uint8 numpy array.
    """
    import numpy as np

    frame = np.zeros((height, width, 3), dtype="uint8")
    # Teal with a diagonal stripe so it's visibly "a capture", not garbage
    frame[:, :, 0] = 20
    frame[:, :, 1] = 110
    frame[:, :, 2] = 140
    for i in range(min(height, width)):
        frame[i, i] = (240, 200, 80)
    return frame


def _crop_frame(frame, bbox: dict[str, int]):
    h, w = frame.shape[:2]
    x = max(0, min(int(bbox["x"]), w - 1))
    y = max(0, min(int(bbox["y"]), h - 1))
    bw = max(1, min(int(bbox["w"]), w - x))
    bh = max(1, min(int(bbox["h"]), h - y))
    return frame[y : y + bh, x : x + bw]


def _frame_to_jpeg(frame) -> bytes:
    """Encode an RGB numpy array as JPEG bytes."""
    from PIL import Image

    img = Image.fromarray(frame, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


async def _grab_frame(full_res: bool):
    """Grab a frame from the live camera, or fall back to a test pattern."""
    from boxbot.hardware.camera import get_camera

    cam = get_camera()
    if cam is None:
        logger.warning("camera HAL not available — returning test pattern")
        return _test_pattern_frame(), True  # fallback=True
    frame = await (cam.capture_photo() if full_res else cam.capture_frame())
    return frame, False


async def _handle_camera_action(
    action_type: str,
    payload: dict[str, Any],
    ctx: ActionContext,
) -> dict[str, Any]:
    from boxbot.workspace import Workspace, WorkspaceError

    try:
        full_res = bool(payload.get("full_res", False))
        save_to = payload.get("save_to")

        frame, is_fallback = await _grab_frame(full_res)

        if action_type == "camera.capture_cropped":
            bbox = payload.get("bbox") or {}
            if not all(k in bbox for k in ("x", "y", "w", "h")):
                return {
                    "status": "error",
                    "error": "bbox must include x, y, w, h",
                }
            frame = _crop_frame(frame, bbox)
        elif action_type != "camera.capture":
            return {
                "status": "error",
                "error": f"unknown camera action: {action_type}",
            }

        jpeg = _frame_to_jpeg(frame)
        h, w = frame.shape[:2]

        if save_to:
            # Persist to workspace (path safety + quota handled there)
            ws = Workspace()
            try:
                ws.write(save_to, jpeg)
            except WorkspaceError as e:
                return {"status": "error", "error": str(e)}
            abs_path = (ws.root / save_to).resolve()
            path_for_agent = save_to
            saved = True
        else:
            tmp_dir = _tmp_capture_dir()
            tmp_dir.mkdir(parents=True, exist_ok=True)
            ref = f"camera_{uuid.uuid4().hex[:12]}.jpg"
            abs_path = (tmp_dir / ref).resolve()
            abs_path.write_bytes(jpeg)
            path_for_agent = f"tmp/{ref}"
            saved = False

        if len(ctx.image_attachments) < MAX_IMAGES_PER_CALL:
            ctx.image_attachments.append(abs_path)
        else:
            logger.warning(
                "camera image not attached (exceeded %d/call)",
                MAX_IMAGES_PER_CALL,
            )

        return {
            "status": "ok",
            "ref": abs_path.name,
            "path": path_for_agent,
            "width": int(w),
            "height": int(h),
            "saved": saved,
            "attached": True,
            "fallback": is_fallback,
        }

    except KeyError as e:
        return {"status": "error", "error": f"missing field: {e}"}
    except Exception as e:  # noqa: BLE001
        logger.exception("%s failed", action_type)
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Calendar action handler (preserves existing behaviour)
# ---------------------------------------------------------------------------


async def _handle_calendar_action(
    action_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    from boxbot.integrations import google_calendar as gc

    try:
        if action_type == "calendar.create_event":
            event_id = await gc.create_event(
                summary=payload["summary"],
                start=payload["start"],
                end=payload["end"],
                description=payload.get("description"),
                location=payload.get("location"),
                calendar_id=payload.get("calendar_id"),
                all_day=bool(payload.get("all_day", False)),
            )
            return {"status": "ok", "event_id": event_id}

        if action_type == "calendar.update_event":
            ok = await gc.update_event(
                payload["event_id"],
                summary=payload.get("summary"),
                start=payload.get("start"),
                end=payload.get("end"),
                description=payload.get("description"),
                location=payload.get("location"),
                calendar_id=payload.get("calendar_id"),
                all_day=bool(payload.get("all_day", False)),
            )
            return {"status": "ok" if ok else "error"}

        if action_type == "calendar.delete_event":
            ok = await gc.delete_event(
                payload["event_id"],
                calendar_id=payload.get("calendar_id"),
            )
            return {"status": "ok" if ok else "error"}

        if action_type == "calendar.list_upcoming_events":
            events = await gc.list_upcoming_events(
                max_results=int(payload.get("max_results", 5)),
                calendar_id=payload.get("calendar_id"),
            )
            return {"status": "ok", "events": events}

        return {
            "status": "error",
            "error": f"unknown calendar action: {action_type}",
        }

    except gc.CalendarNotAuthenticated as e:
        return {
            "status": "error",
            "error": str(e),
            "remedy": "Run scripts/calendar_auth.py to grant access.",
        }
    except KeyError as e:
        return {"status": "error", "error": f"missing field: {e}"}
    except Exception as e:  # noqa: BLE001
        logger.exception("%s failed", action_type)
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


async def process_action(
    action: dict[str, Any],
    ctx: ActionContext,
) -> dict[str, Any]:
    """Route an action to its handler and return the response dict.

    The response is written back to the sandbox iff the action was tagged
    ``_expects_response: true``. It is also appended to ``ctx.action_log``
    so the tool result reflects every side effect.
    """
    action_type = action.get("_sdk") or action.get("action") or "unknown"

    logger.info("sandbox action: %s", action_type)

    if action_type.startswith("workspace."):
        result = _handle_workspace_action(action_type, action, ctx)
    elif action_type.startswith("camera."):
        result = await _handle_camera_action(action_type, action, ctx)
    elif action_type.startswith("photos."):
        result = await _handle_photos_action(action_type, action, ctx)
    elif action_type.startswith("calendar."):
        result = await _handle_calendar_action(action_type, action)
    else:
        # Legacy / unimplemented actions: acknowledge so the sandbox
        # doesn't block forever if it called request() instead of
        # emit_action().
        result = {
            "status": "stub",
            "message": f"action '{action_type}' acknowledged but not yet handled.",
        }

    ctx.action_log.append({"action": action_type, **result})
    return result
