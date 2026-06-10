"""Tests for the SDK-wide error rule (WS3 Part A).

The rule: **mutating** SDK calls raise ``bb.ActionError`` (or a module
subclass) when the main process answers with ``status != "ok"``;
**read** calls return raw response dicts with documented shapes.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import boxbot.sdk as bb
from boxbot.sdk._transport import ActionError, dispatch_or_raise


def _patched_request(response):
    return patch("boxbot.sdk._transport.request", return_value=response)


# ---------------------------------------------------------------------------
# The centralized helper
# ---------------------------------------------------------------------------


class TestDispatchOrRaise:
    def test_ok_passes_response_through(self):
        with _patched_request({"status": "ok", "id": "abc"}):
            resp = dispatch_or_raise("memory.save", {"content": "x"})
        assert resp["id"] == "abc"

    def test_error_status_raises_with_message(self):
        with _patched_request({"status": "error", "message": "store full"}):
            with pytest.raises(ActionError, match="store full") as exc_info:
                dispatch_or_raise("secrets.store", {})
        assert exc_info.value.action == "secrets.store"
        assert exc_info.value.response["status"] == "error"

    def test_missing_status_raises(self):
        """'missing' (and any other non-ok status) raises too."""
        with _patched_request({"status": "missing", "name": "X"}):
            with pytest.raises(ActionError, match="missing"):
                dispatch_or_raise("secrets.delete", {"name": "X"})

    def test_errors_list_joined_into_message(self):
        with _patched_request(
            {"status": "error", "errors": ["bad block", "bad theme"]}
        ):
            with pytest.raises(ActionError, match="bad block"):
                dispatch_or_raise("display.save", {})

    def test_action_error_is_runtime_error(self):
        """Backwards compatibility: older `except RuntimeError` still works."""
        assert issubclass(ActionError, RuntimeError)


class TestExportedFromBbNamespace:
    def test_bb_action_error_exists(self):
        assert bb.ActionError is ActionError

    def test_module_exceptions_subclass_action_error(self):
        from boxbot.sdk.photos import PhotosError
        from boxbot.sdk.workspace import WorkspaceError

        assert issubclass(WorkspaceError, ActionError)
        assert issubclass(PhotosError, ActionError)


# ---------------------------------------------------------------------------
# Write paths raise
# ---------------------------------------------------------------------------


class TestWritePathsRaise:
    def test_secrets_store_raises(self):
        with _patched_request({"status": "error", "message": "bad name"}):
            with pytest.raises(bb.ActionError, match="bad name"):
                bb.secrets.store("bad-name", "v")

    def test_secrets_delete_missing_raises(self):
        with _patched_request({"status": "missing", "name": "NOPE"}):
            with pytest.raises(bb.ActionError):
                bb.secrets.delete("NOPE")

    def test_integrations_update_raises_on_missing(self):
        with _patched_request(
            {"status": "missing", "message": "integration 'x' is not registered"}
        ):
            with pytest.raises(bb.ActionError, match="not registered"):
                bb.integrations.update("x", script="# y\n")

    def test_integrations_delete_raises_on_missing(self):
        with _patched_request({"status": "missing", "message": "absent"}):
            with pytest.raises(bb.ActionError, match="absent"):
                bb.integrations.delete("x")

    def test_integration_builder_save_raises(self):
        with _patched_request(
            {"status": "error", "message": "name already exists"}
        ):
            i = bb.integrations.create("dupe")
            i.description = "x"
            i.script = "# s\n"
            with pytest.raises(bb.ActionError, match="already exists"):
                i.save()

    def test_memory_save_raises_action_error(self):
        with _patched_request({"status": "error", "message": "no handler"}):
            with pytest.raises(bb.ActionError, match="no handler"):
                bb.memory.save("fact")

    def test_tasks_create_todo_raises_action_error(self):
        with _patched_request({"status": "error", "message": "db locked"}):
            with pytest.raises(bb.ActionError, match="db locked"):
                bb.tasks.create_todo("do thing")

    def test_skill_save_now_waits_and_raises(self):
        s = bb.skill.create("dupe-skill")
        s.description = "A skill that already exists somewhere."
        s.body = "# body"
        with _patched_request(
            {"status": "error", "message": "skill 'dupe-skill' exists"}
        ):
            with pytest.raises(bb.ActionError, match="exists"):
                s.save()

    def test_skill_save_returns_response_on_ok(self):
        s = bb.skill.create("fresh-skill")
        s.description = "A fresh skill."
        s.body = "# body"
        with _patched_request(
            {"status": "ok", "name": "fresh-skill", "path": "/skills/fresh-skill"}
        ):
            resp = s.save()
        assert resp["status"] == "ok"

    def test_auth_notify_admins_raises(self):
        with _patched_request({"status": "error", "error": "auth manager not initialised"}):
            with pytest.raises(bb.ActionError, match="auth manager"):
                bb.auth.notify_admins("hello admins")


# ---------------------------------------------------------------------------
# Read paths keep raw returns
# ---------------------------------------------------------------------------


class TestReadPathsReturnRaw:
    def test_integrations_get_returns_error_dict_unraised(self):
        """A failing integration *run* is data the agent inspects."""
        with _patched_request({"status": "error", "error": "401 Unauthorized"}):
            resp = bb.integrations.get("solar", date="2026-06-09")
        assert resp["status"] == "error"

    def test_integrations_get_source_missing_returns_dict(self):
        with _patched_request({"status": "missing", "message": "nope"}):
            resp = bb.integrations.get_source("ghost")
        assert resp["status"] == "missing"

    def test_secrets_list_returns_raw(self):
        with _patched_request(
            {"status": "ok", "secrets": [{"name": "K", "stored_at": "t"}]}
        ):
            resp = bb.secrets.list()
        assert resp["secrets"][0]["name"] == "K"

    def test_secrets_has_returns_bool(self):
        with _patched_request({"status": "missing", "name": "K"}):
            assert bb.secrets.has("K") is False
        with _patched_request(
            {"status": "ok", "name": "K", "env_var": "BOXBOT_SECRET_K"}
        ):
            assert bb.secrets.has("K") is True
