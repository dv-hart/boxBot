"""Shared image safety helpers for the photo paths.

Content-based sniffing (magic bytes) rather than trusting a server-claimed
MIME or a filename extension. Used by the inbound staging path, the
multimodal attach path, and the intake pipeline so a mislabeled or
non-image file is rejected at the boundary instead of deep inside PIL.

Sniffing reads only the first few bytes — it never decodes the image, so
it is safe to call on untrusted input (no decompression-bomb exposure).
"""

from __future__ import annotations

from pathlib import Path

# Canonical media types boxBot accepts end-to-end. Kept in lockstep with
# the multimodal attach pipeline (build_image_block) and the inbound
# extension map (_INBOUND_IMAGE_EXTS).
SUPPORTED_IMAGE_MIMES = ("image/jpeg", "image/png", "image/gif", "image/webp")


def sniff_image_mime(data: bytes) -> str | None:
    """Return the image media type from magic bytes, or None if not a
    recognised image. Does not decode the image.
    """
    if len(data) < 12:
        return None
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def sniff_image_file(path: str | Path) -> str | None:
    """Sniff an image media type from the first bytes of a file on disk."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return None
    return sniff_image_mime(head)
