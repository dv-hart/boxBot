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
import inspect
import io
import logging
import mimetypes
import uuid
from collections.abc import Awaitable, Callable
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
# Handler registry — handlers self-register via @action_handler(prefix). The
# dispatcher routes by the ``<prefix>.<verb>`` convention used in action
# names (e.g. ``workspace.write``, ``skill.save``). All handlers share the
# signature ``(action_type, payload, ctx) -> dict | Awaitable[dict]``;
# handlers that don't need ``ctx`` accept and ignore it.
# ---------------------------------------------------------------------------

Handler = Callable[[str, dict[str, Any], "ActionContext"], "dict[str, Any] | Awaitable[dict[str, Any]]"]
_HANDLERS: dict[str, Handler] = {}


def action_handler(prefix: str) -> Callable[[Handler], Handler]:
    """Register a handler for action types of the form ``<prefix>.<verb>``."""

    def decorator(fn: Handler) -> Handler:
        if prefix in _HANDLERS:
            raise RuntimeError(f"duplicate handler registered for prefix '{prefix}'")
        _HANDLERS[prefix] = fn
        return fn

    return decorator


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


@action_handler("workspace")
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


@action_handler("photos")
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

        if action_type == "photos.view_path":
            raw = payload.get("path")
            if not raw:
                return {"status": "error", "error": "path is required"}
            try:
                abs_path = Path(raw).resolve()
            except OSError as e:
                return {"status": "error", "error": f"bad path: {e}"}
            if not abs_path.exists() or not abs_path.is_file():
                return {
                    "status": "error",
                    "error": f"file not found: {abs_path}",
                }
            if not _is_attach_allowed(abs_path):
                return {
                    "status": "error",
                    "error": (
                        "path is outside the allowed roots "
                        "(sandbox tmp, workspace, photos, perception crops)"
                    ),
                }
            if len(ctx.image_attachments) < MAX_IMAGES_PER_CALL:
                ctx.image_attachments.append(abs_path)
                attached = True
            else:
                attached = False
                logger.warning(
                    "photos.view_path not attached (exceeded %d/call)",
                    MAX_IMAGES_PER_CALL,
                )
            return {
                "status": "ok",
                "path": str(abs_path),
                "filename": abs_path.name,
                "kind": "image",
                "attached": attached,
            }

        if action_type == "photos.ingest":
            raw = payload.get("path")
            source = payload.get("source")
            if not raw:
                return {"status": "error", "error": "path is required"}
            if not source:
                return {"status": "error", "error": "source is required"}
            try:
                abs_path = Path(raw).resolve()
            except OSError as e:
                return {"status": "error", "error": f"bad path: {e}"}
            if not abs_path.exists() or not abs_path.is_file():
                return {
                    "status": "error",
                    "error": f"file not found: {abs_path}",
                }
            if not _is_attach_allowed(abs_path):
                return {
                    "status": "error",
                    "error": (
                        "path is outside the allowed roots "
                        "(sandbox tmp, workspace, photos, perception crops)"
                    ),
                }
            from boxbot.photos.intake import get_intake_pipeline

            pipeline = get_intake_pipeline()
            if pipeline is None:
                return {
                    "status": "error",
                    "error": "photo intake pipeline is not running",
                }
            # Staged inbound files live under tmp/inbound/ — the pipeline
            # owns them after a successful ingest, so it deletes them
            # once the bytes are copied into the photo store.
            staged = "inbound" in abs_path.parts
            try:
                photo_id = await pipeline.enqueue(
                    abs_path,
                    source=str(source),
                    sender=payload.get("sender"),
                    caption=payload.get("caption"),
                    delete_source=staged,
                )
            except (FileNotFoundError, ValueError) as e:
                return {"status": "error", "error": str(e)}
            return {
                "status": "ok",
                "photo_id": photo_id,
                "source": source,
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


@action_handler("camera")
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


# ---------------------------------------------------------------------------
# Auth action handler
# ---------------------------------------------------------------------------


@action_handler("auth")
async def _handle_auth_action(
    action_type: str,
    payload: dict[str, Any],
    ctx: ActionContext,  # unused; kept for uniform handler signature
) -> dict[str, Any]:
    """Auth-state RPC for the agent's onboarding flows.

    Inviting-admin gating for ``generate_registration_code`` is resolved
    here from the *current conversation* via the tool ContextVar. The
    sandbox cannot pass an arbitrary ``created_by``, so a compromised
    script can't mint codes "from" any admin.
    """
    from boxbot.communication.auth import get_auth_manager
    from boxbot.communication.whatsapp import get_whatsapp_client
    from boxbot.tools._tool_context import get_current_conversation

    auth = get_auth_manager()
    if auth is None:
        return {"status": "error", "error": "auth manager not initialised"}

    try:
        if action_type == "auth.list_users":
            users = await auth.list_users()
            return {
                "status": "ok",
                "users": [
                    {
                        "id": u.id,
                        "name": u.name,
                        "phone": u.phone,
                        "role": u.role,
                        "created_at": u.created_at,
                        "last_seen": u.last_seen,
                    }
                    for u in users
                ],
            }

        if action_type == "auth.generate_bootstrap_code":
            code = await auth.generate_bootstrap_code()
            return {"status": "ok", "code": code}

        if action_type == "auth.generate_registration_code":
            conv = get_current_conversation()
            if conv is None or conv.channel != "whatsapp":
                return {
                    "status": "error",
                    "error": (
                        "generate_registration_code requires a WhatsApp "
                        "conversation context (the inviting admin's reply)"
                    ),
                }
            # channel_key is "whatsapp:+15551234567"
            sender_phone = conv.channel_key.split(":", 1)[1] if ":" in conv.channel_key else ""
            if not sender_phone:
                return {
                    "status": "error",
                    "error": "could not resolve sender phone from conversation",
                }
            user = await auth.get_user(sender_phone)
            if user is None or user.role != "admin":
                return {
                    "status": "error",
                    "error": "only admins can generate registration codes",
                }
            code = await auth.generate_registration_code(created_by=sender_phone)
            return {"status": "ok", "code": code}

        if action_type == "auth.notify_admins":
            text = str(payload.get("text") or "")
            if not text.strip():
                return {"status": "error", "error": "text is required"}
            wa = get_whatsapp_client()
            if wa is None:
                return {"status": "error", "error": "WhatsApp client not configured"}
            admins = [u for u in await auth.list_users() if u.role == "admin"]
            sent = 0
            for admin in admins:
                if await wa.send_text(admin.phone, text):
                    sent += 1
            return {"status": "ok", "delivered": sent, "admins": len(admins)}

        return {"status": "error", "error": f"unknown auth action: {action_type}"}

    except RuntimeError as e:
        # Bootstrap-disabled, rate-limit, or admin-not-found from AuthManager
        return {"status": "error", "error": str(e)}
    except Exception as e:  # noqa: BLE001
        logger.exception("%s failed", action_type)
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Display action handler
# ---------------------------------------------------------------------------


# Field schemas for built-in data sources. Lets the agent discover what
# bindings a source exposes without reading source code.
_BUILTIN_SOURCE_SCHEMAS: dict[str, dict[str, Any]] = {
    "weather": {
        "fields": {
            "temp": "string — current temperature, e.g. '72'",
            "condition": "string — human label, e.g. 'Partly Cloudy'",
            "icon": "string — Lucide icon name, e.g. 'cloud-sun'",
            "humidity": "string — percent, e.g. '65'",
            "wind": "string — wind speed, e.g. '12 mph'",
            "forecast": (
                "array of {day, icon, high, low} — next ~5 days. "
                "Bind individual days as {weather.forecast[0].high}."
            ),
        },
    },
    "calendar": {
        "fields": {
            "events": (
                "array of {time, title, duration, location} — upcoming "
                "events. Bind as {calendar.events[0].title}."
            ),
            "count": "int — total events fetched",
        },
    },
    "tasks": {
        "fields": {
            "items": (
                "array of {id, description, due_date, for_person, "
                "status} — open to-dos. Use a 'repeat' block with "
                "source='{tasks.items}' and bind {.description}."
            ),
            "count": "int — total open to-dos",
        },
    },
    "people": {
        "fields": {
            "present": (
                "array of {name, since} — people currently detected. "
                "Bind as {people.present[0].name}."
            ),
            "count": "int — number of people present",
        },
    },
    "agent_status": {
        "fields": {
            "state": "string — sleeping | listening | thinking | speaking",
            "last_active": "string — humanised timestamp",
            "next_wake": "string — when the next scheduled wake fires",
        },
    },
    "clock": {
        "fields": {
            "hour": "int (0-23)",
            "minute": "int (0-59)",
            "second": "int (0-59)",
            "display": "string — formatted time, e.g. '7:42'",
            "date": "string — long date, e.g. 'May 1, 2026'",
            "day_of_week": "string — e.g. 'Friday'",
        },
    },
}


@action_handler("skill")
def _handle_skill_action(
    action_type: str,
    payload: dict[str, Any],
    ctx: ActionContext,  # unused; kept for uniform handler signature
) -> dict[str, Any]:
    """Dispatch skill.* actions. Currently only ``skill.save``.

    The writer lives in :mod:`boxbot.skills.persist` so the file-layout
    contract is reusable from elsewhere (tests, future ``skill.update``).
    """
    if action_type != "skill.save":
        return {
            "status": "error",
            "message": f"unknown skill action '{action_type}'",
        }

    from boxbot.skills.persist import write_skill

    try:
        return write_skill(payload)
    except Exception as exc:  # noqa: BLE001
        logger.exception("skill.save failed")
        return {"status": "error", "message": str(exc)}


@action_handler("integrations")
async def _handle_integrations_action(
    action_type: str,
    payload: dict[str, Any],
    ctx: ActionContext,  # unused; kept for uniform handler signature
) -> dict[str, Any]:
    """Dispatch ``integrations.*`` — list, get, create, update, delete, logs.

    Read paths (``list``, ``get``, ``logs``) defer to the runner and
    log store. Write paths (``create``, ``update``, ``delete``) defer
    to the persist module. All persist operations re-validate the
    payload main-side; the SDK validators are convenience.
    """
    sub = action_type.split(".", 1)[1] if "." in action_type else action_type

    try:
        if sub == "list":
            from boxbot.integrations.loader import discover_integrations

            metas = discover_integrations()
            return {
                "status": "ok",
                "integrations": [
                    {
                        "name": m.name,
                        "description": m.description,
                        "inputs": m.inputs,
                        "outputs": m.outputs,
                        "secrets": list(m.secrets),
                        "timeout": m.timeout,
                    }
                    for m in metas
                ],
            }

        if sub == "get":
            from boxbot.integrations.runner import IntegrationRunError, run

            name = payload.get("name")
            if not isinstance(name, str) or not name:
                return {"status": "error", "message": "'name' is required"}
            inputs = payload.get("inputs") or {}
            if not isinstance(inputs, dict):
                return {"status": "error", "message": "'inputs' must be a dict"}
            try:
                return await run(name, inputs)
            except IntegrationRunError as exc:
                return {"status": "error", "message": str(exc)}

        if sub == "logs":
            from boxbot.integrations.logs import list_runs

            name = payload.get("name")
            if not isinstance(name, str) or not name:
                return {"status": "error", "message": "'name' is required"}
            limit = payload.get("limit", 20)
            if not isinstance(limit, int) or limit < 1:
                return {"status": "error", "message": "'limit' must be a positive integer"}
            return {"status": "ok", "runs": list_runs(name, limit=limit)}

        if sub == "create":
            from boxbot.integrations.persist import create_integration

            return create_integration(payload)

        if sub == "update":
            from boxbot.integrations.persist import update_integration

            return update_integration(payload)

        if sub == "delete":
            from boxbot.integrations.persist import delete_integration

            return delete_integration(payload.get("name"))

        return {
            "status": "error",
            "message": f"unknown integrations action '{action_type}'",
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s failed", action_type)
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Memory action handler
# ---------------------------------------------------------------------------


# Lazy MemoryStore singleton for the dispatcher. The search_memory tool
# keeps its own singleton in builtins/search_memory.py — using a separate
# one here avoids reaching into a private API of another module. Both
# point at the same SQLite file; aiosqlite handles concurrent writers.
_memory_store: Any = None
_memory_store_lock: Any = None


async def _get_handler_memory_store() -> Any:
    import asyncio

    global _memory_store, _memory_store_lock
    if _memory_store_lock is None:
        _memory_store_lock = asyncio.Lock()
    async with _memory_store_lock:
        if _memory_store is None:
            from boxbot.memory.store import MemoryStore

            store = MemoryStore()
            await store.initialize()
            _memory_store = store
    return _memory_store


def _derive_summary(content: str, max_len: int = 80) -> str:
    """First-sentence/line of ``content``, trimmed to ``max_len``.

    Memory search reranking and conversation-start injection both lean
    on ``summary`` being short and self-contained. Agents are encouraged
    to pass their own; this fallback keeps a missing one from breaking
    the write entirely.
    """
    text = content.strip()
    for sep in (". ", "\n", "; "):
        if sep in text:
            text = text.split(sep, 1)[0].rstrip(".;").strip()
            break
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text or "memory"


@action_handler("memory")
async def _handle_memory_action(
    action_type: str,
    payload: dict[str, Any],
    ctx: ActionContext,  # unused; kept for uniform handler signature
) -> dict[str, Any]:
    """Dispatch ``memory.*`` — save, search, delete.

    Routes to the same MemoryStore + search backend the search_memory
    core tool uses, so a write here is visible to a later tool-side
    lookup and vice-versa. Save derives ``summary`` from ``content``
    when the agent doesn't pass one (memory rows require it).
    """
    sub = action_type.split(".", 1)[1] if "." in action_type else action_type

    try:
        store = await _get_handler_memory_store()

        if sub == "save":
            content = payload.get("content")
            if not isinstance(content, str) or not content.strip():
                return {"status": "error", "message": "'content' is required"}
            mem_type = payload.get("type") or "household"
            people = payload.get("people")
            if people is not None and not isinstance(people, list):
                return {"status": "error", "message": "'people' must be a list"}
            tags = payload.get("tags")
            if tags is not None and not isinstance(tags, list):
                return {"status": "error", "message": "'tags' must be a list"}
            summary = payload.get("summary") or _derive_summary(content)
            person = payload.get("person") or (
                people[0] if people else None
            )

            # Stamp source_conversation when the call comes from inside
            # an active conversation — useful provenance and lets the
            # post-conversation extraction pipeline avoid re-deriving
            # the same fact.
            from boxbot.tools._tool_context import get_current_conversation

            conv = get_current_conversation()
            source_conversation = (
                conv.conversation_id if conv is not None else None
            )

            try:
                memory_id = await store.create_memory(
                    type=mem_type,
                    content=content,
                    summary=summary,
                    person=person,
                    people=people or [],
                    tags=tags or [],
                    source_conversation=source_conversation,
                )
            except ValueError as e:
                return {"status": "error", "message": str(e)}
            return {"status": "ok", "id": memory_id, "summary": summary}

        if sub == "search":
            from boxbot.memory.search import search_memories

            query = payload.get("query")
            if not isinstance(query, str) or not query.strip():
                return {"status": "error", "message": "'query' is required"}
            try:
                limit = int(payload.get("limit", 10))
            except (TypeError, ValueError):
                return {"status": "error", "message": "'limit' must be int"}

            mem_type = payload.get("type")
            types = [mem_type] if isinstance(mem_type, str) else None
            people = payload.get("people")
            person = (
                people[0]
                if isinstance(people, list) and people
                else None
            )

            try:
                result = await search_memories(
                    store,
                    mode="lookup",
                    query=query,
                    types=types,
                    person=person,
                )
            except ValueError as e:
                return {"status": "error", "message": str(e)}

            facts = (result.get("facts") or [])[:limit]
            return {"status": "ok", "results": facts}

        if sub == "delete":
            memory_id = payload.get("id")
            if not isinstance(memory_id, str) or not memory_id:
                return {"status": "error", "message": "'id' is required"}
            from boxbot.tools._tool_context import get_current_conversation

            conv = get_current_conversation()
            invalidated_by = (
                conv.conversation_id if conv is not None else "agent"
            )
            await store.invalidate_memory(
                memory_id, invalidated_by=invalidated_by
            )
            return {"status": "ok", "id": memory_id}

        return {
            "status": "error",
            "message": f"unknown memory action '{action_type}'",
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("%s failed", action_type)
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Tasks action handler
# ---------------------------------------------------------------------------


@action_handler("tasks")
async def _handle_tasks_action(
    action_type: str,
    payload: dict[str, Any],
    ctx: ActionContext,  # unused; kept for uniform handler signature
) -> dict[str, Any]:
    """Dispatch ``tasks.*`` — create_trigger, create_todo, list_*, get,
    complete, cancel.

    Mirrors the manage_tasks core tool's surface, reachable from inside
    sandbox scripts so they can compose task management with other SDK
    calls in one turn.
    """
    sub = action_type.split(".", 1)[1] if "." in action_type else action_type

    try:
        from boxbot.core import scheduler

        if sub == "create_trigger":
            description = payload.get("description")
            instructions = payload.get("instructions")
            if not isinstance(description, str) or not description.strip():
                return {
                    "status": "error",
                    "message": "'description' is required",
                }
            if not isinstance(instructions, str) or not instructions.strip():
                return {
                    "status": "error",
                    "message": "'instructions' is required",
                }
            try:
                trigger_id = await scheduler.create_trigger(
                    description=description,
                    instructions=instructions,
                    fire_at=payload.get("fire_at"),
                    fire_after=payload.get("fire_after"),
                    cron=payload.get("cron"),
                    person=payload.get("person"),
                    for_person=payload.get("for_person"),
                    todo_id=payload.get("todo_id"),
                    source="agent",
                )
            except ValueError as e:
                return {"status": "error", "message": str(e)}
            return {"status": "ok", "id": trigger_id}

        if sub == "create_todo":
            description = payload.get("description")
            if not isinstance(description, str) or not description.strip():
                return {
                    "status": "error",
                    "message": "'description' is required",
                }
            todo_id = await scheduler.create_todo(
                description=description,
                notes=payload.get("notes"),
                for_person=payload.get("for_person"),
                due_date=payload.get("due_date"),
                source="agent",
            )
            return {"status": "ok", "id": todo_id}

        if sub == "list_triggers":
            triggers = await scheduler.list_triggers(
                status=payload.get("status"),
                for_person=payload.get("for_person"),
            )
            return {"status": "ok", "results": triggers}

        if sub == "list_todos":
            todos = await scheduler.list_todos(
                status=payload.get("status"),
                for_person=payload.get("for_person"),
            )
            return {"status": "ok", "results": todos}

        if sub == "get":
            item_id = payload.get("id")
            if not isinstance(item_id, str) or not item_id:
                return {"status": "error", "message": "'id' is required"}
            if item_id.startswith("t_"):
                trig = await scheduler.get_trigger(item_id)
                if trig is None:
                    return {
                        "status": "error",
                        "message": f"trigger {item_id} not found",
                    }
                return {"status": "ok", "item_type": "trigger", **trig}
            if item_id.startswith("d_"):
                td = await scheduler.get_todo(item_id)
                if td is None:
                    return {
                        "status": "error",
                        "message": f"to-do {item_id} not found",
                    }
                return {"status": "ok", "item_type": "todo", **td}
            # Unknown prefix — try both.
            trig = await scheduler.get_trigger(item_id)
            if trig is not None:
                return {"status": "ok", "item_type": "trigger", **trig}
            td = await scheduler.get_todo(item_id)
            if td is not None:
                return {"status": "ok", "item_type": "todo", **td}
            return {
                "status": "error",
                "message": f"item {item_id} not found",
            }

        if sub == "complete":
            item_id = payload.get("id")
            if not isinstance(item_id, str) or not item_id:
                return {"status": "error", "message": "'id' is required"}
            ok = await scheduler.complete_todo(item_id)
            if not ok:
                return {
                    "status": "error",
                    "message": f"to-do {item_id} not found",
                }
            return {"status": "ok", "id": item_id}

        if sub == "cancel":
            item_id = payload.get("id")
            if not isinstance(item_id, str) or not item_id:
                return {"status": "error", "message": "'id' is required"}
            if item_id.startswith("t_"):
                ok = await scheduler.cancel_trigger(item_id)
            elif item_id.startswith("d_"):
                ok = await scheduler.cancel_todo(item_id)
            else:
                ok = await scheduler.cancel_trigger(item_id) or (
                    await scheduler.cancel_todo(item_id)
                )
            if not ok:
                return {
                    "status": "error",
                    "message": f"item {item_id} not found",
                }
            return {"status": "ok", "id": item_id}

        return {
            "status": "error",
            "message": f"unknown tasks action '{action_type}'",
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("%s failed", action_type)
        return {"status": "error", "message": str(exc)}


@action_handler("secrets")
def _handle_secrets_action(
    action_type: str,
    payload: dict[str, Any],
    ctx: ActionContext,  # unused; kept for uniform handler signature
) -> dict[str, Any]:
    """Dispatch ``secrets.*`` — store, delete, list, use.

    The handler never echoes secret values. ``store`` accepts a value
    and persists it; ``use`` confirms a secret exists and returns the
    env-var name the runner will set; ``list`` returns names + stored_at
    only; ``delete`` removes by name.

    Action log redaction: only ``name`` and ``status`` (and the optional
    ``previous`` / ``env_var`` fields) are returned, so the action log
    surfaced to the agent never carries the value.
    """
    sub = action_type.split(".", 1)[1] if "." in action_type else action_type

    try:
        from boxbot.secrets import SecretStoreError, get_secret_store

        store = get_secret_store()

        if sub == "store":
            name = payload.get("name")
            value = payload.get("value")
            try:
                return store.store(name, value)  # type: ignore[arg-type]
            except SecretStoreError as exc:
                return {"status": "error", "message": str(exc)}

        if sub == "delete":
            name = payload.get("name")
            try:
                return store.delete(name)  # type: ignore[arg-type]
            except SecretStoreError as exc:
                return {"status": "error", "message": str(exc)}

        if sub == "list":
            return {"status": "ok", "secrets": store.list_names()}

        if sub == "use":
            name = payload.get("name")
            if not isinstance(name, str) or not name:
                return {"status": "error", "message": "'name' is required"}
            if not store.has(name):
                return {"status": "missing", "name": name}
            return {
                "status": "ok",
                "name": name,
                "env_var": f"BOXBOT_SECRET_{name.upper()}",
            }

        return {
            "status": "error",
            "message": f"unknown secrets action '{action_type}'",
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s failed", action_type)
        return {"status": "error", "message": str(exc)}


def _previews_dir() -> Path:
    from boxbot.core.paths import PREVIEWS_DIR

    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    return PREVIEWS_DIR


def _agent_displays_dir() -> Path:
    from boxbot.core.paths import DISPLAYS_DIR

    DISPLAYS_DIR.mkdir(parents=True, exist_ok=True)
    return DISPLAYS_DIR


def _classify_display_source(name: str, agent_dir: Path) -> str:
    """Return 'agent' if saved through the SDK, else 'builtin' / 'user'."""
    if (agent_dir / f"{name}.json").exists():
        return "agent"
    # Best-effort: anything in the project's displays/ tree is 'user',
    # everything else (programmatic builtins, builtins/) is 'builtin'.
    from boxbot.core.paths import PROJECT_ROOT

    user_root = PROJECT_ROOT / "displays"
    if (user_root / name / "display.json").exists():
        return "user"
    if (user_root / f"{name}.json").exists():
        return "user"
    return "builtin"


def _collect_unresolved_bindings(
    spec_dict: dict[str, Any],
    render_data: dict[str, Any] | None = None,
) -> list[str]:
    """Walk the layout looking for bindings that don't resolve.

    Returns a list of human-readable warnings — typos, missing fields,
    sources that aren't declared.

    ``render_data`` is the dict the renderer actually used, assembled
    by :meth:`DisplayManager.build_preview_data`. Passing it eliminates
    false positives on bindings to live-fetched data and to ``static``
    sources whose declared ``value=`` populated the renderer's view but
    not any standalone placeholder pass.
    """
    from boxbot.displays.data_sources import get_placeholder_data
    from boxbot.displays.spec import _BINDING_PATTERN, _lookup_binding

    declared: set[str] = {"args", "current"}
    for src in spec_dict.get("data_sources", []) or []:
        if isinstance(src, dict) and src.get("name"):
            declared.add(src["name"])

    if render_data is not None:
        sample = dict(render_data)
        sample.setdefault("args", {})
    else:
        # Standalone fallback for callers without a manager (e.g. ad-hoc
        # spec linting). Builtins get plausible placeholders; everything
        # else stays empty and will warn.
        sample = {}
        for name in declared:
            if name in ("args", "current"):
                continue
            sample[name] = get_placeholder_data(name) or {}
        sample["args"] = {}

    from boxbot.displays.renderer import lucide_icon_exists

    warnings: list[str] = []
    seen: set[str] = set()
    seen_icons: set[str] = set()

    def _walk(node: Any) -> None:
        # Icon-name validation: when an icon block specifies a literal
        # name (no binding), confirm the SVG is bundled. Without this,
        # an unknown icon falls back to a circled-letter placeholder
        # silently and the agent has no way to notice.
        if (isinstance(node, dict)
                and node.get("type") == "icon"):
            raw_name = node.get("name", "")
            if isinstance(raw_name, str) and raw_name and raw_name not in seen_icons:
                seen_icons.add(raw_name)
                # Skip pure-binding names — they resolve at render time.
                if not (raw_name.startswith("{") and raw_name.endswith("}")):
                    if not lucide_icon_exists(raw_name):
                        warnings.append(
                            f"icon name '{raw_name}' is not in the bundled "
                            "Lucide set — render will fall back to a "
                            "circled-letter placeholder. Use "
                            "bb.display.schema()['icons'] to see what's "
                            "available."
                        )

        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, str):
            for match in _BINDING_PATTERN.finditer(node):
                path = match.group(1)
                if path in seen:
                    continue
                seen.add(path)
                # Skip repeat-template / rotate-current paths — they're
                # only valid inside their parent block. Those resolve
                # at render time per-item, not at the top level.
                if path.startswith(".") or path.startswith("current."):
                    continue
                source = path.split(".", 1)[0]
                if source not in declared:
                    warnings.append(
                        f"binding '{{{path}}}' references undeclared source "
                        f"'{source}' — call d.data('{source}') or check the "
                        f"source name."
                    )
                    continue
                value = _lookup_binding(path, sample, None, None)
                if value is None:
                    warnings.append(
                        f"binding '{{{path}}}' did not resolve at render "
                        f"time. Check the field name on '{source}'."
                    )

    _walk(spec_dict.get("layout"))
    return warnings


@action_handler("display")
async def _handle_display_action(
    action_type: str,
    payload: dict[str, Any],
    ctx: ActionContext,
) -> dict[str, Any]:
    """Dispatch display.* actions: preview, save, list, load, delete,
    describe_source, update_data.

    All actions are synchronous — the SDK uses :func:`request` so the
    agent sees results in the same script run. Validation errors are
    returned in the response payload, not raised, so the agent can read
    them and fix the spec.
    """
    import json

    from boxbot.displays.manager import get_display_manager
    from boxbot.displays.spec import parse_spec, validate_spec

    sub = action_type.split(".", 1)[1] if "." in action_type else action_type

    try:
        if sub == "preview":
            spec_dict = payload.get("spec") or {}
            try:
                parsed = parse_spec(spec_dict)
            except Exception as e:
                return {
                    "status": "error",
                    "errors": [f"could not parse spec: {e}"],
                }
            errors = validate_spec(parsed)
            if errors:
                return {"status": "error", "errors": errors}

            mgr = get_display_manager()
            if mgr is None:
                return {
                    "status": "error",
                    "errors": ["display manager not running"],
                }

            # Register the spec temporarily so render_preview can find
            # it by name. If a spec by this name was already registered,
            # restore it after rendering so we don't clobber a saved
            # display when the agent is iterating.
            existing = mgr.get_spec(parsed.name)
            mgr.register_spec(parsed)
            try:
                # Compute warnings against the same data the renderer
                # uses, so static-source values and live-fetched data
                # don't get falsely flagged. ``data`` from the payload
                # is an agent-supplied override (useful for testing
                # http_json fields/maps without a real fetch); it
                # layers on top of live + static + placeholder data.
                #
                # For each declared source the override touches, run
                # the source's ``fields`` config over the fixture
                # before handing it to the renderer. That's what the
                # author actually wants: validate "given an API
                # response shaped like this, do my fields+map produce
                # the right binding values?" Without this, the
                # override would have to be in post-fields shape,
                # defeating the purpose.
                override = payload.get("data") or {}
                render_data = mgr.build_preview_data(parsed)
                if isinstance(override, dict):
                    render_data.update(_apply_fields_to_override(
                        override, parsed,
                    ))
                image = mgr.render_preview(parsed.name, data=render_data)
            finally:
                if existing is not None:
                    mgr.register_spec(existing)
                else:
                    mgr.unregister_spec(parsed.name)

            if image is None:
                return {
                    "status": "error",
                    "errors": ["render_preview returned no image"],
                }

            previews = _previews_dir()
            out_path = previews / f"{parsed.name}-{uuid.uuid4().hex[:8]}.png"
            try:
                image.save(out_path, format="PNG")
            except Exception as e:
                return {
                    "status": "error",
                    "errors": [f"could not write preview PNG: {e}"],
                }

            attached = False
            if len(ctx.image_attachments) < MAX_IMAGES_PER_CALL:
                ctx.image_attachments.append(out_path)
                attached = True
            else:
                logger.warning(
                    "display.preview not attached (exceeded %d/call)",
                    MAX_IMAGES_PER_CALL,
                )

            return {
                "status": "ok",
                "path": str(out_path),
                "attached": attached,
                "warnings": _collect_unresolved_bindings(
                    spec_dict, render_data=render_data,
                ),
            }

        if sub == "save":
            spec_dict = payload.get("spec") or {}
            name = payload.get("name") or spec_dict.get("name")
            if not name:
                return {"status": "error", "errors": ["display name required"]}
            try:
                parsed = parse_spec(spec_dict)
            except Exception as e:
                return {
                    "status": "error",
                    "errors": [f"could not parse spec: {e}"],
                }
            errors = validate_spec(parsed)
            if errors:
                return {"status": "error", "errors": errors}

            agent_dir = _agent_displays_dir()
            out_path = agent_dir / f"{parsed.name}.json"
            try:
                out_path.write_text(json.dumps(spec_dict, indent=2))
            except Exception as e:
                return {
                    "status": "error",
                    "errors": [f"could not write spec: {e}"],
                }

            mgr = get_display_manager()
            registered = False
            render_data: dict[str, Any] | None = None
            if mgr is not None:
                # Hot-reload: register on the live manager so
                # switch_display(name) works in this same conversation,
                # no restart needed.
                mgr.register_spec(parsed)
                registered = True
                render_data = mgr.build_preview_data(parsed)

            return {
                "status": "ok",
                "path": str(out_path),
                "registered": registered,
                "warnings": _collect_unresolved_bindings(
                    spec_dict, render_data=render_data,
                ),
            }

        if sub == "list":
            mgr = get_display_manager()
            if mgr is None:
                return {"status": "error", "error": "display manager not running"}
            agent_dir = _agent_displays_dir()
            displays = [
                {"name": n, "source": _classify_display_source(n, agent_dir)}
                for n in mgr.list_available()
            ]
            return {"status": "ok", "displays": displays}

        if sub == "load":
            name = payload.get("name")
            if not name:
                return {"status": "error", "error": "display name required"}
            mgr = get_display_manager()
            if mgr is None:
                return {"status": "error", "error": "display manager not running"}
            spec = mgr.get_spec(name)
            if spec is None:
                return {"status": "error", "error": f"display '{name}' not found"}
            spec_dict = _spec_to_dict(spec)
            return {"status": "ok", "spec": spec_dict}

        if sub == "delete":
            name = payload.get("name")
            if not name:
                return {"status": "error", "error": "display name required"}
            agent_dir = _agent_displays_dir()
            spec_path = agent_dir / f"{name}.json"
            if not spec_path.exists():
                return {
                    "status": "error",
                    "error": (
                        f"display '{name}' is not agent-saved; only "
                        "displays created via the SDK can be deleted."
                    ),
                }
            try:
                spec_path.unlink()
            except Exception as e:
                return {"status": "error", "error": f"could not delete: {e}"}
            mgr = get_display_manager()
            if mgr is not None:
                mgr.unregister_spec(name)
            return {"status": "ok"}

        if sub == "describe_source":
            name = payload.get("name", "")
            if not name:
                return {"status": "error", "error": "name required"}
            from boxbot.displays.data_sources import get_placeholder_data

            schema = _BUILTIN_SOURCE_SCHEMAS.get(name)
            if schema is None:
                return {
                    "status": "error",
                    "error": (
                        f"no schema for source '{name}'. Built-in "
                        f"sources: {sorted(_BUILTIN_SOURCE_SCHEMAS)}"
                    ),
                }
            return {
                "status": "ok",
                "fields": schema["fields"],
                "example": get_placeholder_data(name),
            }

        if sub == "schema":
            return _build_display_schema()

        if sub == "update_data":
            mgr = get_display_manager()
            if mgr is None:
                return {"status": "error", "error": "display manager not running"}
            ok = mgr.update_static_data(
                payload.get("display", ""),
                payload.get("source", ""),
                payload.get("value"),
            )
            if not ok:
                return {
                    "status": "error",
                    "error": (
                        "update_data failed — display must be active and "
                        "source must be of type 'static'."
                    ),
                }
            return {"status": "ok"}

        return {"status": "error", "error": f"unknown display action: {action_type}"}

    except Exception as e:  # noqa: BLE001
        logger.exception("%s failed", action_type)
        return {"status": "error", "error": str(e)}


def _apply_fields_to_override(
    override: dict[str, Any],
    parsed_spec: Any,
) -> dict[str, Any]:
    """Run each source's ``fields`` transform over the matching override.

    The agent passes ``data=`` to ``preview()`` to validate an
    ``http_json`` source's ``fields`` mapping before the source is
    live. Without this hook, the override would have to already be in
    post-fields shape — the override would dodge the very transform
    the agent is trying to verify. We mirror what
    :class:`HttpJsonSource.do_fetch` does: the override entry is
    treated as the raw fetch response, ``_apply_field_transforms``
    layers the named fields on top.

    Sources without a ``fields`` config or sources that aren't
    overridden pass through unchanged.
    """
    from boxbot.displays.data_sources import _apply_field_transforms

    out = dict(override)
    for src_spec in parsed_spec.data_sources:
        if src_spec.name not in out:
            continue
        if not src_spec.fields:
            continue
        raw = out[src_spec.name]
        if not isinstance(raw, dict):
            continue
        out[src_spec.name] = _apply_field_transforms(raw, src_spec.fields)
    return out


def _build_display_schema() -> dict[str, Any]:
    """Return the full block + spec reference as a dict.

    Introspection target for ``bb.display.schema()``. Each block entry
    lists its fields with type/default/valid-values pulled directly
    from the dataclass + the SDK validators, so the doc and the
    runtime can never drift.
    """
    import dataclasses

    from boxbot.displays.blocks import BLOCK_REGISTRY
    from boxbot.sdk import _validators as v

    # Validator-set names → human-readable bullets the agent can match
    # against the field's annotated type.
    enum_sets: dict[str, list[str]] = {
        "VALID_TEXT_SIZES": sorted(v.VALID_TEXT_SIZES),
        "VALID_TEXT_COLORS": sorted(v.VALID_TEXT_COLORS),
        "VALID_TEXT_WEIGHTS": sorted(v.VALID_TEXT_WEIGHTS),
        "VALID_TEXT_ALIGNS": sorted(v.VALID_TEXT_ALIGNS),
        "VALID_TEXT_ANIMATIONS": sorted(v.VALID_TEXT_ANIMATIONS),
        "VALID_CONTAINER_ALIGNS": sorted(v.VALID_CONTAINER_ALIGNS),
        "VALID_ICON_SIZES": sorted(v.VALID_ICON_SIZES),
        "VALID_EMOJI_SIZES": sorted(v.VALID_EMOJI_SIZES),
        "VALID_CLOCK_FORMATS": sorted(v.VALID_CLOCK_FORMATS),
        "VALID_CLOCK_SIZES": sorted(v.VALID_CLOCK_SIZES),
        "VALID_CHART_TYPES": sorted(v.VALID_CHART_TYPES),
        "VALID_LIST_STYLES": sorted(v.VALID_LIST_STYLES),
        "VALID_IMAGE_FITS": sorted(v.VALID_IMAGE_FITS),
        "VALID_METRIC_ANIMATIONS": sorted(v.VALID_METRIC_ANIMATIONS),
        "VALID_DIVIDER_ORIENTATIONS": sorted(v.VALID_DIVIDER_ORIENTATIONS),
    }

    # block-name → field-name → which enum set governs it. Drives the
    # "valid values" column without us hand-listing every block.
    field_enums: dict[str, dict[str, str]] = {
        "row":      {"align": "VALID_CONTAINER_ALIGNS"},
        "column":   {"align": "VALID_CONTAINER_ALIGNS"},
        "stack":    {"align": "VALID_CONTAINER_ALIGNS"},
        "columns":  {},
        "card":     {},
        "spacer":   {},
        "divider":  {"orientation": "VALID_DIVIDER_ORIENTATIONS"},
        "repeat":   {},
        "text":     {"size": "VALID_TEXT_SIZES",
                     "color": "VALID_TEXT_COLORS",
                     "weight": "VALID_TEXT_WEIGHTS",
                     "align": "VALID_TEXT_ALIGNS",
                     "animation": "VALID_TEXT_ANIMATIONS"},
        "metric":   {"animation": "VALID_METRIC_ANIMATIONS"},
        "badge":    {},
        "list":     {"style": "VALID_LIST_STYLES"},
        "table":    {},
        "key_value": {},
        "icon":     {"size": "VALID_ICON_SIZES"},
        "emoji":    {"size": "VALID_EMOJI_SIZES"},
        "image":    {"fit": "VALID_IMAGE_FITS"},
        "chart":    {"type": "VALID_CHART_TYPES"},
        "progress": {},
        "clock":    {"format": "VALID_CLOCK_FORMATS",
                     "size": "VALID_CLOCK_SIZES"},
        "countdown": {},
        "weather_widget": {},
        "calendar_widget": {},
        "rotate":   {},
        "page_dots": {},
    }

    container_types = {"row", "column", "stack", "columns", "card",
                       "spacer", "divider", "repeat"}

    blocks: dict[str, dict[str, Any]] = {}
    for block_type, cls in BLOCK_REGISTRY.items():
        fields: dict[str, dict[str, Any]] = {}
        enums = field_enums.get(block_type, {})
        for f in dataclasses.fields(cls):
            if f.name in ("block_type", "children", "params"):
                continue
            field_info: dict[str, Any] = {
                "type": _annotation_to_str(f.type),
            }
            if f.default is not dataclasses.MISSING:
                field_info["default"] = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                field_info["default_factory"] = (
                    f.default_factory.__name__
                    if hasattr(f.default_factory, "__name__")
                    else str(f.default_factory)
                )
            if f.name in enums:
                field_info["valid_values"] = enum_sets[enums[f.name]]
            fields[f.name] = field_info
        blocks[block_type] = {
            "kind": "container" if block_type in container_types else "content",
            "fields": fields,
            "accepts_children": block_type in container_types,
        }

    from boxbot.displays.renderer import available_lucide_icons

    return {
        "status": "ok",
        "blocks": blocks,
        "themes": sorted(v.VALID_THEMES),
        "data_source_types": ["builtin", *sorted(v.VALID_DATA_SOURCE_TYPES)],
        "transitions": sorted(v.VALID_TRANSITIONS),
        "data_sources": _data_source_schema(),
        "icons": available_lucide_icons(),
        "binding_syntax": {
            "source_field": "{source.field}",
            "array_index": "{source.field[0].sub}",
            "repeat_item": "{.field}  (inside a repeat block)",
            "rotate_item": "{current.field}  (inside a rotate block)",
            "args": "{args.field}  (passed to switch_display)",
        },
    }


def _data_source_schema() -> dict[str, Any]:
    """Per-source-type field reference, mirroring the per-block one.

    Each entry lists required + optional fields the spec accepts, plus
    the built-in source names so the agent can introspect what's
    available without re-reading the doc.
    """
    return {
        "builtin": {
            "kind": "builtin",
            "fields": {
                "name": {"type": "str", "required": True,
                         "valid_values": sorted(_BUILTIN_SOURCE_SCHEMAS)},
            },
            "available_names": sorted(_BUILTIN_SOURCE_SCHEMAS),
            "describe": (
                "Use bb.display.describe_source(name) to see fields a "
                "specific built-in exposes."
            ),
        },
        "http_json": {
            "kind": "custom",
            "fields": {
                "name": {"type": "str", "required": True},
                "url": {"type": "str", "required": True},
                "params": {"type": "dict", "required": False,
                           "describe": "Query string params"},
                "secret": {"type": "str", "required": False,
                           "describe": "Secret name resolved at fetch time; "
                                       "sent as 'Authorization: Bearer <value>'"},
                "refresh": {"type": "int (seconds)", "required": False,
                            "default": 60},
                "fields": {
                    "type": "dict",
                    "required": False,
                    "describe": (
                        "Field extraction/mapping. Each key is the "
                        "output binding name; the value is either a "
                        "dotted-path string ('data.current.price') or "
                        "{from: <path>, map: {raw: out, ...}} for "
                        "value mapping."
                    ),
                },
            },
        },
        "http_text": {
            "kind": "custom",
            "fields": {
                "name": {"type": "str", "required": True},
                "url": {"type": "str", "required": True},
                "refresh": {"type": "int (seconds)", "required": False,
                            "default": 60},
            },
            "describe": "Bind body as {<name>.text}",
        },
        "static": {
            "kind": "custom",
            "fields": {
                "name": {"type": "str", "required": True},
                "value": {
                    "type": "any (typically dict)",
                    "required": True,
                    "describe": (
                        "Initial value. Update at runtime with "
                        "bb.display.update_data(...)."
                    ),
                },
            },
        },
        "memory_query": {
            "kind": "custom",
            "fields": {
                "name": {"type": "str", "required": True},
                "query": {"type": "str", "required": True,
                          "describe": "Memory search query string"},
                "refresh": {"type": "int (seconds)", "required": False,
                            "default": 300},
            },
            "describe": "Bind results as {<name>.results}",
        },
    }


def _annotation_to_str(annotation: Any) -> str:
    """Render a dataclass field annotation as a short human string."""
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", str(annotation))


def _spec_to_dict(spec: Any) -> dict[str, Any]:
    """Serialize a DisplaySpec back to the on-disk JSON shape.

    Inverse of :func:`boxbot.displays.spec.parse_spec`. Used by
    ``display.load`` so the agent can edit existing specs.
    """
    out: dict[str, Any] = {
        "name": spec.name,
        "theme": spec.theme,
        "transition": spec.transition,
    }
    sources: list[dict[str, Any]] = []
    for src in spec.data_sources:
        d: dict[str, Any] = {"name": src.name}
        if src.source_type and src.source_type != "builtin":
            d["type"] = src.source_type
        if src.url:
            d["url"] = src.url
        if src.params:
            d["params"] = dict(src.params)
        if src.secret:
            d["secret"] = src.secret
        if src.refresh is not None:
            d["refresh"] = src.refresh
        if src.fields:
            d["fields"] = dict(src.fields)
        if src.value is not None:
            d["value"] = src.value
        if src.query:
            d["query"] = src.query
        sources.append(d)
    if sources:
        out["data_sources"] = sources
    if spec.root_block is not None:
        # Block.to_dict() already produces the JSON-compatible shape
        # (type + params + recursively-serialized children).
        out["layout"] = spec.root_block.to_dict()
    return out


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

    Handlers register themselves via :func:`action_handler` and are
    looked up by the prefix before the first ``.`` in ``action_type``.
    Unknown prefixes return ``status: error`` so the sandbox sees the
    failure rather than a silent acknowledgement.
    """
    action_type = action.get("_sdk") or action.get("action") or "unknown"

    logger.info("sandbox action: %s", action_type)

    prefix = action_type.split(".", 1)[0]
    handler = _HANDLERS.get(prefix)
    if handler is None:
        result: dict[str, Any] = {
            "status": "error",
            "message": f"unknown action '{action_type}' (no handler registered for prefix '{prefix}')",
        }
    else:
        outcome = handler(action_type, action, ctx)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        result = outcome

    ctx.action_log.append({"action": action_type, **result})
    return result
