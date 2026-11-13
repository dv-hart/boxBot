"""Tests for the full model-call log (telemetry/model_log.py)."""

from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
from types import SimpleNamespace

import pytest

from boxbot.telemetry import model_log


@pytest.fixture
def log_db(tmp_path):
    path = tmp_path / "telemetry" / "model_calls.db"
    model_log.configure(path)
    yield path
    model_log.configure(None)
    # Leave the module in its "ask config" state for other tests.
    model_log._disabled_reason = None


def _rows(path):
    assert model_log.flush()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM model_calls ORDER BY id"
        )]
    finally:
        conn.close()


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, mode="python"):
        return self._payload


def _anthropic_client(create):
    batches = SimpleNamespace()
    messages = SimpleNamespace(create=create, batches=batches)
    return SimpleNamespace(messages=messages)


def _openai_client(create):
    completions = SimpleNamespace(create=create)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class TestInstrument:
    async def test_logs_full_request_and_response(self, log_db):
        long_text = "x" * 50_000

        async def create(**kwargs):
            return _Resp({"content": [{"type": "text", "text": "pong"}]})

        client = model_log.instrument(_anthropic_client(create), "anthropic")
        await client.messages.create(
            model="claude-haiku-4-5",
            system="be brief",
            messages=[{"role": "user", "content": long_text}],
        )
        (row,) = _rows(log_db)
        assert row["provider"] == "anthropic"
        assert row["endpoint"] == "messages.create"
        assert row["model"] == "claude-haiku-4-5"
        assert row["error"] is None
        assert row["caller"].startswith("tests.test_model_log:") or \
            "test_model_log" in row["caller"]
        req = json.loads(row["request"])
        # Untruncated.
        assert req["messages"][0]["content"] == long_text
        assert req["system"] == "be brief"
        assert json.loads(row["response"])["content"][0]["text"] == "pong"

    async def test_request_snapshot_precedes_caller_mutation(self, log_db):
        messages = [{"role": "user", "content": "first"}]

        async def create(**kwargs):
            # The agent loop appends to this same list after the call.
            kwargs["messages"].append({"role": "assistant", "content": "x"})
            return _Resp({"ok": True})

        client = model_log.instrument(_openai_client(create), "openai")
        await client.chat.completions.create(model="m", messages=messages)
        (row,) = _rows(log_db)
        assert len(json.loads(row["request"])["messages"]) == 1

    async def test_error_is_logged_and_reraised(self, log_db):
        async def create(**kwargs):
            raise RuntimeError("boom")

        client = model_log.instrument(_openai_client(create), "openai")
        with pytest.raises(RuntimeError):
            await client.chat.completions.create(model="m", messages=[])
        (row,) = _rows(log_db)
        assert row["error"] == "RuntimeError: boom"
        assert row["response"] is None

    async def test_context_tags(self, log_db):
        async def create(**kwargs):
            return _Resp({})

        client = model_log.instrument(_openai_client(create), "openai")
        with model_log.call_context(
            purpose="prefetch", conversation_id="conv_1", lane="skills"
        ):
            # Tags follow asyncio tasks created inside the block.
            await asyncio.gather(
                client.chat.completions.create(model="m", messages=[])
            )
        await client.chat.completions.create(model="m", messages=[])
        tagged, bare = _rows(log_db)
        assert tagged["purpose"] == "prefetch"
        assert tagged["conversation_id"] == "conv_1"
        assert json.loads(tagged["tags"]) == {"lane": "skills"}
        assert bare["purpose"] is None and bare["tags"] is None

    async def test_idempotent(self, log_db):
        async def create(**kwargs):
            return _Resp({})

        client = _openai_client(create)
        model_log.instrument(client, "openai")
        model_log.instrument(client, "openai")
        await client.chat.completions.create(model="m", messages=[])
        assert len(_rows(log_db)) == 1

    def test_sync_client(self, log_db):
        def create(**kwargs):
            return _Resp({"data": [{"embedding": [0.1, 0.2]}]})

        client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        model_log.instrument(client, "openai")
        client.embeddings.create(model="emb", input=["hello"])
        (row,) = _rows(log_db)
        assert "test_model_log" in row["caller"]
        assert row["endpoint"] == "embeddings.create"
        assert json.loads(row["request"])["input"] == ["hello"]

    async def test_batch_results_iterator_logged_when_drained(self, log_db):
        async def results(batch_id):
            async def gen():
                yield _Resp({"custom_id": "a"})
                yield _Resp({"custom_id": "b"})
            return gen()

        client = _anthropic_client(None)
        client.messages.batches.results = results
        model_log.instrument(client, "anthropic")
        it = await client.messages.batches.results("batch_1")
        seen = [r.model_dump()["custom_id"] async for r in it]
        assert seen == ["a", "b"]
        (row,) = _rows(log_db)
        assert row["endpoint"] == "messages.batches.results"
        assert json.loads(row["request"])["$args"] == ["batch_1"]
        assert [r["custom_id"] for r in json.loads(row["response"])] == [
            "a", "b",
        ]

    async def test_async_method_behind_sync_decorator(self, log_db):
        # anthropic/openai wrap ``create`` in a plain-function decorator,
        # so it is not a coroutine function yet returns an awaitable.
        async def _impl(**kwargs):
            return _Resp({"text": "late"})

        def create(**kwargs):
            return _impl(**kwargs)

        client = model_log.instrument(_openai_client(create), "openai")
        resp = await client.chat.completions.create(model="m", messages=[])
        assert resp.model_dump() == {"text": "late"}
        (row,) = _rows(log_db)
        assert json.loads(row["response"]) == {"text": "late"}
        assert "test_model_log" in row["caller"]

    async def test_real_sdk_clients_are_patched(self, log_db):
        anthropic = pytest.importorskip("anthropic")
        openai = pytest.importorskip("openai")
        import types

        if not (
            isinstance(anthropic, types.ModuleType)
            and isinstance(openai, types.ModuleType)
        ):
            pytest.skip("SDK modules are mocked in this test session")
        a = model_log.instrument(
            anthropic.AsyncAnthropic(api_key="test"), "anthropic"
        )
        o = model_log.instrument(openai.AsyncOpenAI(api_key="test"), "openai")
        for fn in (
            a.messages.create, a.messages.batches.create,
            a.messages.batches.results, o.chat.completions.create,
            o.embeddings.create,
        ):
            assert getattr(fn, "__wrapped__", None) is not None

    async def test_missing_endpoints_tolerated(self, log_db):
        # Test doubles rarely implement every endpoint.
        client = model_log.instrument(SimpleNamespace(), "anthropic")
        assert client is not None


class TestBlobs:
    async def _call_with_image(self, raw: bytes, shape: str):
        b64 = base64.b64encode(raw).decode()

        async def create(**kwargs):
            return _Resp({})

        if shape == "anthropic":
            content = [{"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": b64,
            }}]
            client = model_log.instrument(
                _anthropic_client(create), "anthropic"
            )
            await client.messages.create(
                model="m", messages=[{"role": "user", "content": content}]
            )
        else:
            content = [{"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{b64}",
            }}]
            client = model_log.instrument(_openai_client(create), "openai")
            await client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": content}]
            )

    @pytest.mark.parametrize("shape", ["anthropic", "openai"])
    async def test_image_moved_to_blob_losslessly(self, log_db, shape):
        raw = bytes(range(256)) * 40
        await self._call_with_image(raw, shape)
        (row,) = _rows(log_db)
        assert base64.b64encode(raw).decode()[:200] not in row["request"]
        req = json.loads(row["request"])
        part = req["messages"][0]["content"][0]
        ref = (
            part["source"]["data"] if shape == "anthropic"
            else part["image_url"]["url"]
        )
        assert ref["bytes"] == len(raw)
        blob = log_db.parent / "blobs" / ref["$blob"]
        assert blob.read_bytes() == raw
        assert (blob.stat().st_mode & 0o777) == 0o600

    async def test_store_images_false_keeps_hash_only(self, tmp_path):
        path = tmp_path / "t" / "calls.db"
        model_log.configure(path, store_images=False)
        try:
            await self._call_with_image(b"\x01" * 4096, "anthropic")
            (row,) = _rows(path)
            ref = json.loads(row["request"])[
                "messages"][0]["content"][0]["source"]["data"]
            assert ref["stored"] is False
            assert not list((path.parent / "blobs").iterdir())
        finally:
            model_log.configure(None)
            model_log._disabled_reason = None


class TestSafety:
    async def test_disabled_is_a_noop(self, tmp_path):
        model_log.configure(None)
        try:
            async def create(**kwargs):
                return "ok"

            client = model_log.instrument(_openai_client(create), "openai")
            assert await client.chat.completions.create(model="m") == "ok"
        finally:
            model_log._disabled_reason = None

    def test_db_is_private(self, log_db):
        model_log.log_call(provider="p", endpoint="e", request={"a": 1})
        _rows(log_db)
        assert (log_db.stat().st_mode & 0o777) == 0o600

    def test_unserializable_objects_do_not_raise(self, log_db):
        model_log.log_call(
            provider="p", endpoint="e",
            request={"obj": object(), "b": b"\x00\x01", "s": {1, 2}},
            response=[SimpleNamespace(x=1)],
        )
        (row,) = _rows(log_db)
        assert "$bytes_b64" in row["request"]
