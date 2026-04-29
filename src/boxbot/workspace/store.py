"""Workspace store — validated file operations on the agent notebook.

All paths are relative to the workspace root. Absolute paths, ``..``
segments, and symlinks that escape the root are rejected. Binary files
are read/written as bytes; text files are read/written as UTF-8.

Quota: the store tracks total size and rejects writes that would push
the workspace above its configured cap (default 100 MB). This is a
soft ceiling; the cap protects the SD card, not privacy.
"""

from __future__ import annotations

import csv
import io
import logging
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from boxbot.core.paths import WORKSPACE_DIR

logger = logging.getLogger(__name__)

DEFAULT_ROOT = WORKSPACE_DIR
DEFAULT_QUOTA_BYTES = 100 * 1024 * 1024  # 100 MB

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
TEXT_EXTS = {
    ".md", ".txt", ".csv", ".json", ".yaml", ".yml",
    ".log", ".tsv", ".py", ".html", ".xml",
}


class WorkspaceError(Exception):
    """Raised for invalid operations (bad path, quota, missing file)."""


@dataclass
class FileInfo:
    path: str            # workspace-relative
    size: int
    modified: float      # unix ts
    is_dir: bool
    kind: str            # "text" | "image" | "csv" | "json" | "binary" | "dir"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "modified": self.modified,
            "is_dir": self.is_dir,
            "kind": self.kind,
        }


@dataclass
class SearchHit:
    path: str
    line: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "line": self.line, "text": self.text}


def _classify(p: Path) -> str:
    if p.is_dir():
        return "dir"
    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext == ".csv":
        return "csv"
    if ext == ".json":
        return "json"
    if ext in TEXT_EXTS:
        return "text"
    # Fall back to mimetype sniff
    mt, _ = mimetypes.guess_type(str(p))
    if mt and mt.startswith("text/"):
        return "text"
    return "binary"


class Workspace:
    """Filesystem-backed workspace with path validation and quota.

    Methods return plain dicts / primitives so they serialize cleanly
    across the sandbox boundary.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        quota_bytes: int = DEFAULT_QUOTA_BYTES,
    ) -> None:
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self.quota_bytes = quota_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        # Resolve once for symlink checks later
        self._root_resolved = self.root.resolve()

    # ------------------------------------------------------------------
    # Path validation
    # ------------------------------------------------------------------

    def _safe_path(self, rel: str, *, must_exist: bool = False) -> Path:
        """Validate a user-supplied relative path and return the absolute Path.

        Rejects: empty, absolute paths, ``..`` segments, NUL bytes,
        paths that resolve outside the workspace root (symlink escape).
        """
        if not isinstance(rel, str) or not rel.strip():
            raise WorkspaceError("path must be a non-empty string")
        if "\x00" in rel:
            raise WorkspaceError("path contains null byte")

        # Normalize separators; reject absolute & parent-escapes
        rel_path = Path(rel)
        if rel_path.is_absolute():
            raise WorkspaceError(f"path must be relative, got {rel!r}")
        parts = rel_path.parts
        if any(p == ".." for p in parts):
            raise WorkspaceError(f"path must not contain '..', got {rel!r}")

        abs_path = (self.root / rel_path)
        # Resolve to catch symlink escapes
        try:
            resolved = abs_path.resolve(strict=must_exist)
        except FileNotFoundError:
            # strict=False case handles non-existing; re-resolve permissively
            resolved = abs_path.resolve()

        try:
            resolved.relative_to(self._root_resolved)
        except ValueError:
            raise WorkspaceError(
                f"path escapes workspace root: {rel!r}"
            )

        if must_exist and not abs_path.exists():
            raise WorkspaceError(f"not found: {rel!r}")
        return abs_path

    def _rel_of(self, abs_path: Path) -> str:
        """Return workspace-relative form of an absolute path."""
        try:
            return str(abs_path.resolve().relative_to(self._root_resolved))
        except ValueError:
            return str(abs_path)

    # ------------------------------------------------------------------
    # Quota
    # ------------------------------------------------------------------

    def used_bytes(self) -> int:
        total = 0
        for dirpath, _, filenames in os.walk(self.root):
            for f in filenames:
                try:
                    total += (Path(dirpath) / f).stat().st_size
                except OSError:
                    pass
        return total

    def _check_quota(self, incoming_bytes: int, replace_path: Path | None = None) -> None:
        existing = 0
        if replace_path is not None and replace_path.exists():
            try:
                existing = replace_path.stat().st_size
            except OSError:
                pass
        projected = self.used_bytes() - existing + incoming_bytes
        if projected > self.quota_bytes:
            raise WorkspaceError(
                f"quota exceeded: {projected / 1e6:.1f} MB would exceed "
                f"{self.quota_bytes / 1e6:.1f} MB cap"
            )

    # ------------------------------------------------------------------
    # Core file ops
    # ------------------------------------------------------------------

    def write(self, path: str, content: str | bytes) -> dict[str, Any]:
        abs_path = self._safe_path(path)
        if isinstance(content, str):
            data = content.encode("utf-8")
        elif isinstance(content, (bytes, bytearray)):
            data = bytes(content)
        else:
            raise WorkspaceError("content must be str or bytes")
        self._check_quota(len(data), replace_path=abs_path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(data)
        return {
            "path": self._rel_of(abs_path),
            "size": len(data),
            "kind": _classify(abs_path),
        }

    def append(self, path: str, content: str) -> dict[str, Any]:
        abs_path = self._safe_path(path)
        if not isinstance(content, str):
            raise WorkspaceError("append content must be str")
        data = content.encode("utf-8")
        self._check_quota(len(data))
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(abs_path, "ab") as f:
            f.write(data)
        return {
            "path": self._rel_of(abs_path),
            "size": abs_path.stat().st_size,
            "kind": _classify(abs_path),
        }

    def read(self, path: str, *, binary: bool = False) -> dict[str, Any]:
        abs_path = self._safe_path(path, must_exist=True)
        if abs_path.is_dir():
            raise WorkspaceError(f"is a directory: {path!r}")
        if binary or _classify(abs_path) == "binary":
            data = abs_path.read_bytes()
            # Binary content is returned as bytes length only via this RPC;
            # use view() to surface images, and keep arbitrary binary opaque.
            return {
                "path": self._rel_of(abs_path),
                "size": len(data),
                "kind": _classify(abs_path),
                "binary": True,
            }
        content = abs_path.read_text(encoding="utf-8", errors="replace")
        return {
            "path": self._rel_of(abs_path),
            "size": len(content.encode("utf-8")),
            "kind": _classify(abs_path),
            "content": content,
        }

    def delete(self, path: str) -> dict[str, Any]:
        abs_path = self._safe_path(path, must_exist=True)
        if abs_path.is_dir():
            # Refuse to rm -rf silently; require empty
            try:
                abs_path.rmdir()
            except OSError as e:
                raise WorkspaceError(f"directory not empty: {path!r}") from e
        else:
            abs_path.unlink()
        return {"path": self._rel_of(abs_path), "deleted": True}

    def exists(self, path: str) -> dict[str, Any]:
        try:
            abs_path = self._safe_path(path)
        except WorkspaceError:
            return {"path": path, "exists": False}
        return {"path": self._rel_of(abs_path), "exists": abs_path.exists()}

    def ls(self, path: str | None = None) -> list[dict[str, Any]]:
        if path in (None, "", "."):
            target = self.root
        else:
            target = self._safe_path(path, must_exist=True)
            if not target.is_dir():
                raise WorkspaceError(f"not a directory: {path!r}")
        entries: list[FileInfo] = []
        for child in sorted(target.iterdir()):
            try:
                st = child.stat()
            except OSError:
                continue
            entries.append(FileInfo(
                path=self._rel_of(child),
                size=st.st_size if not child.is_dir() else 0,
                modified=st.st_mtime,
                is_dir=child.is_dir(),
                kind=_classify(child),
            ))
        return [e.to_dict() for e in entries]

    # ------------------------------------------------------------------
    # Search — simple grep across text files
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        path: str | None = None,
        limit: int = 50,
        case_insensitive: bool = True,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query:
            raise WorkspaceError("query must be a non-empty string")
        if path in (None, "", "."):
            target = self.root
        else:
            target = self._safe_path(path, must_exist=True)

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            pattern = re.compile(query, flags)
        except re.error:
            # Fall back to literal search
            pattern = re.compile(re.escape(query), flags)

        hits: list[SearchHit] = []
        if target.is_file():
            files: Iterable[Path] = [target]
        else:
            files = (p for p in target.rglob("*") if p.is_file())

        for f in files:
            if _classify(f) not in {"text", "csv", "json"}:
                continue
            try:
                with open(f, encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh, start=1):
                        if pattern.search(line):
                            hits.append(SearchHit(
                                path=self._rel_of(f),
                                line=i,
                                text=line.rstrip("\n")[:300],
                            ))
                            if len(hits) >= limit:
                                return [h.to_dict() for h in hits]
            except OSError:
                continue
        return [h.to_dict() for h in hits]

    # ------------------------------------------------------------------
    # View — returns content for text, or signals image attachment
    # ------------------------------------------------------------------

    def view(self, path: str) -> dict[str, Any]:
        """Return a view descriptor.

        For text files, returns content directly. For images, returns a
        descriptor with absolute_path so the caller (execute_script) can
        emit an image content block. Binary non-image files return kind
        only.
        """
        abs_path = self._safe_path(path, must_exist=True)
        if abs_path.is_dir():
            raise WorkspaceError(f"cannot view directory: {path!r}")
        kind = _classify(abs_path)
        if kind == "image":
            return {
                "path": self._rel_of(abs_path),
                "kind": "image",
                "absolute_path": str(abs_path),
            }
        if kind in {"text", "csv", "json"}:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
            return {
                "path": self._rel_of(abs_path),
                "kind": kind,
                "content": content,
            }
        return {
            "path": self._rel_of(abs_path),
            "kind": kind,
            "message": "binary file — use read(binary=True) to inspect size",
        }

    # ------------------------------------------------------------------
    # CSV helpers — thin conveniences over write/read
    # ------------------------------------------------------------------

    def csv_write(
        self,
        path: str,
        rows: list[dict[str, Any]],
        *,
        fieldnames: list[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(rows, list):
            raise WorkspaceError("rows must be a list of dicts")
        if fieldnames is None:
            seen: list[str] = []
            for r in rows:
                if not isinstance(r, dict):
                    raise WorkspaceError("each row must be a dict")
                for k in r.keys():
                    if k not in seen:
                        seen.append(k)
            fieldnames = seen
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        return self.write(path, buf.getvalue())

    def csv_append(self, path: str, row: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(row, dict):
            raise WorkspaceError("row must be a dict")
        abs_path = self._safe_path(path)
        if abs_path.exists():
            # Read existing header to preserve field order
            with open(abs_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
            fieldnames = header if header else list(row.keys())
        else:
            fieldnames = list(row.keys())
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            with open(abs_path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(fieldnames)
        buf = io.StringIO()
        csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore").writerow(row)
        line = buf.getvalue()
        self._check_quota(len(line.encode("utf-8")))
        with open(abs_path, "a", encoding="utf-8", newline="") as f:
            f.write(line)
        return {
            "path": self._rel_of(abs_path),
            "size": abs_path.stat().st_size,
            "kind": "csv",
        }

    def csv_read(self, path: str) -> list[dict[str, Any]]:
        abs_path = self._safe_path(path, must_exist=True)
        with open(abs_path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
