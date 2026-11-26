"""Product photos for the screen — the one bounded main-process fetch.

Everywhere else the main process refuses to fetch a URL that came from
a store. Product cards need a picture, and routing it through the
sandbox cost a model round trip per image, so this module fetches it
directly under fixed limits (docs/security.md, "Product images"):

- ``https`` only, default port, no userinfo;
- host must be on ``purchases.product_image_hosts`` (exact match,
  default ``cdn.shopify.com``) — on the first request **and on every
  redirect hop** (at most :data:`_MAX_REDIRECTS`);
- response must declare ``image/jpeg``, ``image/png`` or ``image/webp``;
- body read in chunks and abandoned past :data:`MAX_BYTES`;
- total time bounded by :data:`_TIMEOUT`;
- the bytes are decoded and **re-encoded** as a plain JPEG (pixel-count
  capped, metadata dropped, downscaled) before anything else reads
  them; the raw download is never written to disk;
- no cookies, no credentials, no request body, a fixed User-Agent.

Results are cached under ``data/purchases/product-images/`` by URL
hash — next to the purchase database, mode ``0700``, a directory no
sandbox action can name: it is outside the workspace, the photo
library and every image-attach root, and :func:`is_cached_image`
refuses a cache that is ever configured inside one of them. So a file
in the cache was written by :func:`_reencode` and by nothing else.
The URL is never supplied by the sandbox: callers pass the
``image_url`` held in the main-process result cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

MAX_BYTES = 3 * 1024 * 1024
MAX_PIXELS = 24_000_000
OUT_LONG_EDGE = 800
_TIMEOUT = 8.0
_MAX_REDIRECTS = 2
_TYPES = ("image/jpeg", "image/png", "image/webp")
_FORMATS = ("JPEG", "PNG", "WEBP")
_MAX_CACHE_FILES = 300
_FAIL_SECONDS = 600

_transport: httpx.AsyncBaseTransport | None = None   # tests
_failed: dict[str, float] = {}                       # url hash → when it failed
_inflight: dict[str, asyncio.Task] = {}
_slots: asyncio.Semaphore | None = None


def cache_dir() -> Path:
    """The image cache: main-process-only, beside the purchase database."""
    from boxbot.core import paths

    return paths.PURCHASES_DIR / "product-images"


def _agent_roots() -> list[Path]:
    """Every directory a sandbox action can read or write by path."""
    from boxbot.core import paths

    roots = [paths.WORKSPACE_DIR, paths.PHOTOS_DIR, paths.DISPLAYS_DIR,
             paths.SANDBOX_DIR]
    try:
        from boxbot.tools._sandbox_actions import _attach_roots

        roots.extend(_attach_roots())
    except Exception:  # noqa: BLE001 — the static list still applies
        logger.exception("product image: could not read the attach roots")
    try:
        from boxbot.core.config import get_config

        config = get_config()
        roots.append(Path(config.photos.storage_path))
        roots.append(Path(config.sandbox.runtime_dir))
    except Exception:  # noqa: BLE001 — config not loaded (tests, tools)
        pass
    out = []
    for root in roots:
        try:
            out.append(Path(root).resolve())
        except (OSError, ValueError):
            continue
    return out


def _agent_reachable(path: Path) -> bool:
    """True when ``path`` (resolved) lies in a directory the sandbox can
    address — the workspace, the photo library, an attach root."""
    return any(path == root or root in path.parents for root in _agent_roots())


def _ensure_cache_dir() -> Path | None:
    """Create the cache owner-only; None if it may not be used."""
    try:
        root = cache_dir()
        if _agent_reachable(root.resolve()):
            logger.error("product image cache %s is agent-reachable; not used", root)
            return None
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        return root
    except OSError:
        logger.exception("product image: could not prepare the cache")
        return None


def _config() -> Any:
    from boxbot.core.config import get_config

    return get_config().purchases


def allowed_url(url: Any) -> str | None:
    """``url`` if it is an https URL on an allow-listed image host."""
    if not isinstance(url, str) or not 0 < len(url) <= 600 or not url.isascii():
        return None
    if any(ch.isspace() or ord(ch) < 0x21 for ch in url):
        return None
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme != "https" or port not in (None, 443):
        return None
    if parts.username is not None or parts.password is not None:
        return None
    host = (parts.hostname or "").lower()
    if host not in _config().product_image_hosts:
        return None
    return url


def _key(url: str) -> str:
    return hashlib.sha256(url.encode("ascii")).hexdigest()[:32]


def cached_path(url: Any) -> Path | None:
    """The cached, re-encoded file for ``url`` if it exists."""
    if allowed_url(url) is None:
        return None
    return is_cached_image(cache_dir() / f"{_key(url)}.jpg")


def is_cached_image(path: Any) -> Path | None:
    """``path`` resolved, only if it is a file inside the image cache.

    The product blocks open nothing else: an ``image`` value arriving
    from anywhere but this module (an agent-authored display spec, say)
    cannot make the renderer read an arbitrary file. And never a file
    the sandbox could have written: a path under the workspace or any
    other agent-addressable directory is refused even if the cache were
    pointed there.
    """
    if not isinstance(path, (str, Path)) or not str(path):
        return None
    try:
        resolved = Path(path).resolve()
        root = cache_dir().resolve()
    except (OSError, ValueError):
        return None
    if resolved.parent != root or resolved.suffix != ".jpg" or not resolved.is_file():
        return None
    if _agent_reachable(resolved):
        return None
    return resolved


def open_cached(path: Any) -> Any:
    """The cached image at ``path`` as an RGB ``PIL.Image``, or None.

    The only way the renderer opens a product photo. Belt and braces
    behind :func:`is_cached_image`: the header is read first, and a file
    that is not a JPEG or is over :data:`MAX_PIXELS` is refused before
    any pixel is decoded — whatever is on disk, a render cannot be made
    to allocate an arbitrarily large bitmap.
    """
    from PIL import Image

    safe = is_cached_image(path)
    if safe is None:
        return None
    try:
        with Image.open(safe) as opened:
            if opened.format != "JPEG":
                return None
            width, height = opened.size
            if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
                return None
            return opened.convert("RGB")
    except Exception:  # noqa: BLE001 — a broken file is just "no image"
        return None


async def _download(url: str) -> bytes | None:
    """The image bytes, or None. Enforces host, type and size per hop."""
    current = url
    async with httpx.AsyncClient(
        timeout=_TIMEOUT, transport=_transport, follow_redirects=False,
        headers={"User-Agent": "boxbot-product-image/1", "Accept": "image/*"},
    ) as http:
        for _hop in range(_MAX_REDIRECTS + 1):
            if allowed_url(current) is None:
                logger.info("product image: host not allowed, dropped")
                return None
            async with http.stream("GET", current) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location") or ""
                    # Absolute https URLs only; a relative or downgraded
                    # redirect is not followed.
                    current = location
                    continue
                if response.status_code != 200:
                    return None
                kind = (response.headers.get("content-type") or "").split(";")[0]
                if kind.strip().lower() not in _TYPES:
                    return None
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > MAX_BYTES:
                    return None
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_BYTES:
                        return None
                return bytes(body)
    return None


def _reencode(raw: bytes, dest: Path) -> bool:
    """Decode ``raw`` and write a metadata-free JPEG to ``dest``."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as probe:
            if probe.format not in _FORMATS:
                return False
            width, height = probe.size
            if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
                return False
            probe.draft("RGB", (OUT_LONG_EDGE, OUT_LONG_EDGE))
            image = probe.convert("RGBA")
        # Flatten on white: product shots are cut-outs more often than not.
        flat = Image.new("RGB", image.size, (255, 255, 255))
        flat.paste(image, mask=image.split()[3])
        flat.thumbnail((OUT_LONG_EDGE, OUT_LONG_EDGE), Image.Resampling.LANCZOS)
        if _ensure_cache_dir() is None or dest.parent != cache_dir():
            return False
        partial = dest.with_name(f".{dest.name}.{os.getpid()}.part")
        # A fresh Image carries no EXIF/ICC/XMP from the download.
        flat.save(partial, format="JPEG", quality=85)
        os.replace(partial, dest)
        return True
    except Exception:  # noqa: BLE001 — a hostile or broken image is just "no image"
        logger.info("product image: could not decode")
        return False


def _prune() -> None:
    try:
        files = sorted(
            cache_dir().glob("*.jpg"), key=lambda f: f.stat().st_mtime,
        )
    except OSError:
        return
    for stale in files[:-_MAX_CACHE_FILES]:
        try:
            stale.unlink()
        except OSError:
            pass


async def _fetch(url: str, key: str) -> Path | None:
    global _slots
    if _slots is None:
        _slots = asyncio.Semaphore(4)
    dest = cache_dir() / f"{key}.jpg"
    try:
        async with _slots:
            raw = await asyncio.wait_for(_download(url), timeout=_TIMEOUT + 2)
            ok = raw is not None and await asyncio.to_thread(_reencode, raw, dest)
    except (httpx.HTTPError, asyncio.TimeoutError, OSError):
        ok = False
    if not ok:
        _failed[key] = time.monotonic()
        if len(_failed) > 500:
            _failed.clear()
        return None
    await asyncio.to_thread(_prune)
    return dest


async def fetch(url: Any) -> Path | None:
    """Cached path of the product image at ``url``, fetching if needed.

    None when image fetching is off, the URL is not on an allowed host,
    or anything about the download or the image is out of bounds.
    Never raises.
    """
    try:
        if not _config().fetch_product_images:
            return None
        if allowed_url(url) is None:
            return None
        hit = cached_path(url)
        if hit is not None:
            return hit
        key = _key(url)
        failed_at = _failed.get(key)
        if failed_at is not None and time.monotonic() - failed_at < _FAIL_SECONDS:
            return None
        task = _inflight.get(key)
        if task is None or task.done():
            task = asyncio.get_running_loop().create_task(_fetch(url, key))
            _inflight[key] = task
            task.add_done_callback(lambda _t, k=key: _inflight.pop(k, None))
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("product image fetch failed")
        return None
