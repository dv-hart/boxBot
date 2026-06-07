"""Tests for the agent workspace store and sandbox action handlers."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import pytest

from boxbot.workspace import Workspace, WorkspaceError


# ---------------------------------------------------------------------------
# Workspace store — path safety, quota, file ops
# ---------------------------------------------------------------------------


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(root=tmp_path, quota_bytes=10_000)


class TestWorkspaceBasics:
    def test_write_and_read_text(self, ws: Workspace) -> None:
        r = ws.write("notes/hello.md", "# hi\n")
        assert r["kind"] == "text"
        assert r["size"] == len("# hi\n")
        assert ws.read("notes/hello.md")["content"] == "# hi\n"

    def test_write_accepts_bytes(self, ws: Workspace) -> None:
        ws.write("captures/x.png", b"\x89PNG" + b"\x00" * 10)
        r = ws.read("captures/x.png", binary=True)
        assert r["binary"] is True
        assert r["kind"] == "image"

    def test_append_creates_if_missing(self, ws: Workspace) -> None:
        ws.append("log.txt", "line one\n")
        ws.append("log.txt", "line two\n")
        assert ws.read("log.txt")["content"] == "line one\nline two\n"

    def test_ls_returns_kind(self, ws: Workspace) -> None:
        ws.write("notes/a.md", "x")
        ws.write("data/b.csv", "h\n1\n")
        entries = {e["path"]: e for e in ws.ls()}
        assert "notes" in entries and entries["notes"]["is_dir"]
        assert "data" in entries and entries["data"]["is_dir"]
        # And inside a subdir
        sub = {e["path"]: e for e in ws.ls("data")}
        assert any(p.endswith("b.csv") for p in sub)

    def test_exists(self, ws: Workspace) -> None:
        ws.write("a.md", "x")
        assert ws.exists("a.md")["exists"] is True
        assert ws.exists("nope.md")["exists"] is False

    def test_delete_file(self, ws: Workspace) -> None:
        ws.write("a.md", "x")
        ws.delete("a.md")
        assert ws.exists("a.md")["exists"] is False

    def test_delete_nonempty_dir_refused(self, ws: Workspace) -> None:
        ws.write("d/a.md", "x")
        with pytest.raises(WorkspaceError):
            ws.delete("d")


class TestPathSafety:
    @pytest.mark.parametrize(
        "bad",
        [
            "../escape",
            "/etc/passwd",
            "notes/../../escape",
            "",
            "a\x00b",
        ],
    )
    def test_traversal_rejected(self, ws: Workspace, bad: str) -> None:
        with pytest.raises(WorkspaceError):
            ws.read(bad)
        with pytest.raises(WorkspaceError):
            ws.write(bad, "x")


class TestQuota:
    def test_over_quota_rejected(self, ws: Workspace) -> None:
        with pytest.raises(WorkspaceError, match="quota"):
            ws.write("big.bin", b"x" * 20_000)

    def test_replacing_file_does_not_double_count(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path, quota_bytes=1_000)
        ws.write("a.bin", b"x" * 800)
        # Replacing with similar-sized file should be allowed
        ws.write("a.bin", b"y" * 800)


class TestSearch:
    def test_finds_matches_in_text_files_only(self, ws: Workspace) -> None:
        ws.write("notes/a.md", "the quick brown fox\nTODO: jump\n")
        ws.write("notes/b.md", "no todos here\n")
        ws.write("captures/x.png", b"\x89PNG" + b"\x00" * 10)
        hits = ws.search("TODO", case_insensitive=False)
        assert any(h["path"].endswith("a.md") for h in hits)
        assert not any(h["path"].endswith("b.md") for h in hits)
        # Image file must not appear in hits
        assert not any(h["path"].endswith("x.png") for h in hits)

    def test_regex_falls_back_to_literal(self, ws: Workspace) -> None:
        ws.write("a.md", "the cost is [unknown](tbd)\n")
        # '[unknown' is an invalid regex (unterminated character class); the
        # search must fall back to literal substring matching rather than
        # raising.
        hits = ws.search("[unknown")
        assert any("[unknown" in h["text"] for h in hits)


class TestCSV:
    def test_round_trip(self, ws: Workspace) -> None:
        ws.csv_write(
            "chores.csv",
            [
                {"task": "dishes", "done": False},
                {"task": "trash", "done": True},
            ],
        )
        rows = ws.csv_read("chores.csv")
        assert len(rows) == 2
        assert rows[0]["task"] == "dishes"

    def test_append_preserves_header(self, ws: Workspace) -> None:
        ws.csv_write("chores.csv", [{"task": "dishes", "done": False}])
        ws.csv_append("chores.csv", {"task": "trash", "done": True})
        rows = ws.csv_read("chores.csv")
        assert len(rows) == 2
        # Header order preserved
        assert list(rows[1].keys()) == ["task", "done"]


class TestView:
    def test_text_returns_content(self, ws: Workspace) -> None:
        ws.write("a.md", "hello")
        r = ws.view("a.md")
        assert r["kind"] == "text"
        assert r["content"] == "hello"

    def test_image_returns_absolute_path(self, ws: Workspace) -> None:
        ws.write("captures/x.png", b"\x89PNG" + b"\x00" * 10)
        r = ws.view("captures/x.png")
        assert r["kind"] == "image"
        assert "absolute_path" in r
        assert Path(r["absolute_path"]).exists()


# ---------------------------------------------------------------------------
# Sandbox action handler — workspace.* dispatch
# ---------------------------------------------------------------------------


class TestWorkspaceActions:
    """Dispatcher: verify workspace.* actions round-trip through the handler."""

    def test_write_and_read(self, tmp_path: Path, monkeypatch) -> None:
        from boxbot.tools import _sandbox_actions as sa

        # Point Workspace default root at the tmp dir for this handler call
        monkeypatch.setattr(sa, "_handle_workspace_action", sa._handle_workspace_action)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "workspace").mkdir(parents=True)

        ctx = sa.ActionContext()
        resp = sa._handle_workspace_action(
            "workspace.write",
            {"path": "a.md", "content": "hello"},
            ctx,
        )
        assert resp["status"] == "ok"
        assert resp["kind"] == "text"

        resp = sa._handle_workspace_action(
            "workspace.read", {"path": "a.md"}, ctx
        )
        assert resp["status"] == "ok"
        assert resp["content"] == "hello"

    def test_view_image_queues_attachment(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from boxbot.tools import _sandbox_actions as sa

        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "workspace").mkdir(parents=True)

        ctx = sa.ActionContext()

        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        sa._handle_workspace_action(
            "workspace.write",
            {"path": "x.png", "b64": base64.b64encode(png_bytes).decode("ascii")},
            ctx,
        )
        resp = sa._handle_workspace_action(
            "workspace.view", {"path": "x.png"}, ctx
        )
        assert resp["status"] == "ok"
        assert resp["kind"] == "image"
        assert resp.get("attached") is True
        # Absolute path should not leak back to the sandbox
        assert "absolute_path" not in resp
        # And the attachment should be queued
        assert len(ctx.image_attachments) == 1

    def test_error_response_on_bad_path(self, tmp_path: Path, monkeypatch) -> None:
        from boxbot.tools import _sandbox_actions as sa

        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "workspace").mkdir(parents=True)

        ctx = sa.ActionContext()
        resp = sa._handle_workspace_action(
            "workspace.read", {"path": "../../../etc/passwd"}, ctx
        )
        assert resp["status"] == "error"
        assert ".." in resp["error"]


class TestImageBlockGuards:
    def test_refuses_outside_allowlist(self, tmp_path: Path) -> None:
        from boxbot.tools._sandbox_actions import build_image_block

        img = tmp_path / "wild.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 100)
        assert build_image_block(img) is None

    def test_accepts_inside_allowlist(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # WORKSPACE_DIR is anchored to BOXBOT_DATA_DIR (or the project
        # tree); chdir doesn't move the allowlist. Override the data
        # dir and reload the modules that captured the path constants.
        monkeypatch.setenv("BOXBOT_DATA_DIR", str(tmp_path))
        import importlib
        import boxbot.core.paths as paths
        importlib.reload(paths)
        import boxbot.tools._sandbox_actions as sa
        importlib.reload(sa)

        (tmp_path / "workspace").mkdir(parents=True)
        img = tmp_path / "workspace" / "x.png"
        # Full 8-byte PNG signature — build_image_block now sniffs the
        # actual bytes (not the .png extension), so the magic must be real.
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        block = sa.build_image_block(img)
        assert block is not None
        assert block["type"] == "image"
        assert block["source"]["media_type"] == "image/png"
        assert block["source"]["type"] == "base64"
        assert block["source"]["data"]

    def test_refuses_oversize(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("BOXBOT_DATA_DIR", str(tmp_path))
        import importlib
        import boxbot.core.paths as paths
        importlib.reload(paths)
        import boxbot.tools._sandbox_actions as sa
        importlib.reload(sa)

        (tmp_path / "workspace").mkdir(parents=True)
        img = tmp_path / "workspace" / "big.jpg"
        # Slightly over the cap
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * (sa.MAX_IMAGE_BYTES + 1))
        assert sa.build_image_block(img) is None

    def test_resizes_large_image_under_api_cap(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A high-res photo must resize so encoded payload stays under
        Anthropic's 5 MB cap. Real-world failure: a 3.98 MB raw JPEG
        from WhatsApp blew the cap at ~5.3 MB after base64."""
        import base64 as _b64
        import io
        from PIL import Image

        monkeypatch.setenv("BOXBOT_DATA_DIR", str(tmp_path))
        import importlib
        import boxbot.core.paths as paths
        importlib.reload(paths)
        import boxbot.tools._sandbox_actions as sa
        importlib.reload(sa)

        (tmp_path / "workspace").mkdir(parents=True)
        img_path = tmp_path / "workspace" / "huge.jpg"
        # Write a real 4000x3000 JPEG. Use a noisy pattern so JPEG
        # can't trivially compress it to near-zero — we want a file
        # that's genuinely a few MB, like a real phone photo.
        big = Image.new("RGB", (4000, 3000))
        pixels = big.load()
        for y in range(0, 3000, 4):
            for x in range(0, 4000, 4):
                pixels[x, y] = ((x * 7) % 256, (y * 11) % 256, ((x + y) * 5) % 256)
        big.save(img_path, format="JPEG", quality=95)
        raw_size = img_path.stat().st_size
        assert raw_size > sa.ATTACH_PASSTHROUGH_BYTES, (
            "test fixture didn't produce a large-enough file"
        )

        block = sa.build_image_block(img_path)
        assert block is not None
        assert block["type"] == "image"
        # After resize-on-attach we always re-encode as JPEG
        assert block["source"]["media_type"] == "image/jpeg"

        encoded_bytes = len(_b64.b64decode(block["source"]["data"]))
        # 5 MB is the API cap; we should be comfortably below it
        assert encoded_bytes < 5 * 1024 * 1024
        # And the resized image's long edge must be <= the target
        with Image.open(io.BytesIO(_b64.b64decode(block["source"]["data"]))) as decoded:
            assert max(decoded.size) <= sa.ATTACH_LONG_EDGE_PX

    def test_small_image_passes_through_verbatim(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Files under the pass-through ceiling should not be re-encoded
        — preserves animated GIFs, keeps perception crops crisp."""
        import base64 as _b64
        from PIL import Image

        monkeypatch.setenv("BOXBOT_DATA_DIR", str(tmp_path))
        import importlib
        import boxbot.core.paths as paths
        importlib.reload(paths)
        import boxbot.tools._sandbox_actions as sa
        importlib.reload(sa)

        (tmp_path / "workspace").mkdir(parents=True)
        img_path = tmp_path / "workspace" / "small.png"
        Image.new("RGB", (320, 240), color=(50, 100, 150)).save(img_path)
        assert img_path.stat().st_size < sa.ATTACH_PASSTHROUGH_BYTES

        block = sa.build_image_block(img_path)
        assert block is not None
        # Pass-through path preserves original media type
        assert block["source"]["media_type"] == "image/png"
        # And the raw bytes round-trip exactly
        assert _b64.b64decode(block["source"]["data"]) == img_path.read_bytes()
