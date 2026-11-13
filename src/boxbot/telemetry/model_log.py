"""Full-fidelity log of every hosted model call.

Every request sent to a hosted model and every response received is
stored, complete and untruncated, in a dedicated SQLite database
(``telemetry.model_log_path``, default ``data/telemetry/model_calls.db``).
This is the ground truth for offline analysis and for building training
sets (prefetch selectors, policy judge).

How calls are captured
----------------------
``instrument(client, provider)`` patches the call methods on one client
instance — Anthropic ``messages.create`` / ``messages.batches.create`` /
``messages.batches.results``; OpenAI ``chat.completions.create`` /
``responses.create`` / ``embeddings.create``. Every client the codebase
constructs is passed through it, so a new call site is logged without
touching this module. The Claude Agent SDK path has no client method to
patch; it logs through ``log_call`` directly.

``call_context(purpose=..., conversation_id=...)`` attaches free-form
tags to every call made inside the ``with`` block (contextvar, so it
follows asyncio tasks). The calling ``module:function:line`` is always
recorded, tags or not.

Design constraints
------------------
- Never raises into, and never blocks, the model call path. The request
  is snapshotted to JSON on the calling thread (the caller may mutate
  its message list afterwards); everything else — blob extraction,
  SQLite writes — happens on one daemon writer thread.
- Lossless. Inline base64 payloads (images) are moved out of the JSON
  into content-addressed files under ``<db dir>/blobs/`` and replaced by
  ``{"$blob": "<sha256>", "bytes": N}``; nothing is truncated.
- Local only. The database and blobs are created mode 0600/0700 and are
  never read by the sandbox or sent anywhere.

Not covered: on-device models (detection, ReID, speaker embedding, VAD,
local text embedding) and ElevenLabs STT/TTS audio.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import inspect
import json
import logging
import os
import queue
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    provider        TEXT NOT NULL,
    endpoint        TEXT NOT NULL,
    model           TEXT,
    purpose         TEXT,
    conversation_id TEXT,
    caller          TEXT,
    tags            TEXT,
    request         TEXT,
    response        TEXT,
    error           TEXT,
    latency_ms      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_model_calls_ts ON model_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_model_calls_purpose
    ON model_calls(purpose, timestamp);
CREATE INDEX IF NOT EXISTS idx_model_calls_conv
    ON model_calls(conversation_id);
"""

_INSERT = """
INSERT INTO model_calls (
    timestamp, provider, endpoint, model, purpose, conversation_id,
    caller, tags, request, response, error, latency_ms
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Strings at least this long are candidates for blob extraction.
_BLOB_MIN_CHARS = 2048

_INSTRUMENTED_FLAG = "_boxbot_model_log_instrumented"

_tags: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "boxbot_model_log_tags", default={}
)


# ---------------------------------------------------------------------------
# Context tags
# ---------------------------------------------------------------------------


@contextmanager
def call_context(**tags: Any) -> Iterator[None]:
    """Tag every model call made inside the block.

    ``purpose`` and ``conversation_id`` land in their own columns; any
    other key is stored in the ``tags`` JSON column. Nested blocks merge,
    inner keys winning. ``None`` values are ignored.
    """
    merged = {**_tags.get(), **{k: v for k, v in tags.items() if v is not None}}
    token = _tags.set(merged)
    try:
        yield
    finally:
        _tags.reset(token)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    """``json.dumps`` default hook: SDK/pydantic/dataclass objects → plain."""
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:
            pass
    if hasattr(obj, "__dataclass_fields__"):
        return {
            name: getattr(obj, name, None) for name in obj.__dataclass_fields__
        }
    if isinstance(obj, (bytes, bytearray)):
        import base64

        return {"$bytes_b64": base64.b64encode(bytes(obj)).decode("ascii")}
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    return repr(obj)


def _dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    try:
        return json.dumps(obj, default=_jsonable, ensure_ascii=False)
    except Exception:
        return json.dumps({"$unserializable": repr(obj)})


def _looks_base64(s: str) -> bool:
    head = s[:256]
    return all(c.isalnum() or c in "+/=\n\r-_" for c in head)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class _Writer:
    """Single daemon thread owning the SQLite connection."""

    def __init__(self, db_path: Path, *, store_images: bool = True) -> None:
        self.db_path = db_path
        self.blob_dir = db_path.parent / "blobs"
        self.store_images = store_images
        self._q: queue.Queue[tuple[Any, ...] | None] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="model-log-writer", daemon=True
        )
        self._thread.start()

    def put(self, row: tuple[Any, ...]) -> None:
        self._q.put(row)

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until queued rows are written (or timeout). For shutdown
        and tests."""
        deadline = time.monotonic() + timeout
        while self._q.unfinished_tasks:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    # -- thread body -------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.blob_dir.mkdir(exist_ok=True, mode=0o700)
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass
        return conn

    def _run(self) -> None:
        conn: sqlite3.Connection | None = None
        while True:
            row = self._q.get()
            try:
                if row is None:
                    return
                if conn is None:
                    conn = self._open()
                (ts, provider, endpoint, model, purpose, conv_id, caller,
                 tags, request, response, error, latency_ms) = row
                conn.execute(_INSERT, (
                    ts, provider, endpoint, model, purpose, conv_id, caller,
                    tags, self._externalize(request),
                    self._externalize(response), error, latency_ms,
                ))
                conn.commit()
            except Exception:
                logger.exception("model log write failed")
                # Drop the connection so the next row reopens cleanly.
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass
                conn = None
            finally:
                self._q.task_done()

    # -- blob extraction ---------------------------------------------------

    def _externalize(self, payload: str | None) -> str | None:
        """Move large inline base64 payloads out to content-addressed files."""
        if payload is None or len(payload) < _BLOB_MIN_CHARS:
            return payload
        if "base64" not in payload:
            return payload
        try:
            tree = json.loads(payload)
        except Exception:
            return payload
        return json.dumps(self._walk(tree), ensure_ascii=False)

    def _walk(self, node: Any) -> Any:
        if isinstance(node, dict):
            # Anthropic: {"type": "base64", "media_type": ..., "data": "..."}
            data = node.get("data")
            if (
                node.get("type") == "base64"
                and isinstance(data, str)
                and len(data) >= _BLOB_MIN_CHARS
            ):
                out = dict(node)
                out["data"] = self._blob(data, node.get("media_type"))
                return out
            return {k: self._walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [self._walk(v) for v in node]
        if isinstance(node, str) and len(node) >= _BLOB_MIN_CHARS:
            # OpenAI: "data:image/jpeg;base64,...."
            if node.startswith("data:") and ";base64," in node[:128]:
                header, _, data = node.partition(";base64,")
                if _looks_base64(data):
                    return self._blob(data, header[5:])
        return node

    def _blob(self, b64: str, media_type: Any) -> dict[str, Any]:
        import base64

        ref: dict[str, Any] = {"$blob": None, "media_type": media_type}
        try:
            raw = base64.b64decode(b64, validate=False)
        except Exception:
            raw = b64.encode("utf-8", "replace")
            ref["encoding"] = "raw-base64-text"
        digest = hashlib.sha256(raw).hexdigest()
        ref["$blob"] = digest
        ref["bytes"] = len(raw)
        if not self.store_images:
            ref["stored"] = False
            return ref
        path = self.blob_dir / digest
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
            os.replace(tmp, path)
        return ref


_writer: _Writer | None = None
_writer_lock = threading.Lock()
_disabled_reason: str | None = None


def configure(
    db_path: str | Path | None, *, store_images: bool = True
) -> None:
    """Point the log at ``db_path`` (``None`` disables it).

    Called implicitly from config on first use; call explicitly from
    standalone scripts and tests.
    """
    global _writer, _disabled_reason
    with _writer_lock:
        if _writer is not None:
            _writer.flush()
            _writer.put(None)
            _writer = None
        if db_path is None:
            _disabled_reason = "disabled"
            return
        _disabled_reason = None
        _writer = _Writer(Path(db_path), store_images=store_images)


def _get_writer() -> _Writer | None:
    global _disabled_reason
    if _writer is not None:
        return _writer
    if _disabled_reason is not None:
        return None
    try:
        from boxbot.core.config import get_config

        cfg = get_config().telemetry
    except Exception:
        # Config not loaded (unit tests, ad-hoc imports): stay off until
        # someone calls configure() or config loads. Not latched, so a
        # later load_config() turns logging on.
        return None
    if not cfg.model_log_enabled:
        _disabled_reason = "telemetry.model_log_enabled is false"
        logger.warning("model call log is OFF (%s)", _disabled_reason)
        return None
    configure(cfg.model_log_path, store_images=cfg.model_log_store_images)
    logger.info("model call log → %s", cfg.model_log_path)
    return _writer


def flush(timeout: float = 5.0) -> bool:
    """Wait for queued rows to hit disk. True if drained."""
    w = _writer
    return True if w is None else w.flush(timeout)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def _caller(depth: int) -> str | None:
    try:
        frame = sys._getframe(depth)
        mod = frame.f_globals.get("__name__", "?")
        return f"{mod}:{frame.f_code.co_name}:{frame.f_lineno}"
    except Exception:
        return None


def log_call(
    *,
    provider: str,
    endpoint: str,
    request: Any,
    response: Any = None,
    error: BaseException | str | None = None,
    latency_ms: int | None = None,
    model: str | None = None,
    caller: str | None = None,
    request_json: str | None = None,
) -> None:
    """Record one model call. Never raises.

    ``request_json`` lets a wrapper pass a snapshot taken *before* the
    call (the caller may mutate its request objects afterwards).
    """
    try:
        writer = _get_writer()
        if writer is None:
            return
        tags = dict(_tags.get())
        purpose = tags.pop("purpose", None)
        conv_id = tags.pop("conversation_id", None)
        if request_json is None:
            request_json = _dumps(request)
        if model is None and isinstance(request, dict):
            m = request.get("model")
            model = m if isinstance(m, str) else None
        if isinstance(error, BaseException):
            error = f"{type(error).__name__}: {error}"
        writer.put((
            datetime.now(timezone.utc).isoformat(),
            provider,
            endpoint,
            model,
            purpose,
            conv_id,
            caller or _caller(2),
            _dumps(tags) if tags else None,
            request_json,
            _dumps(response),
            error,
            latency_ms,
        ))
    except Exception:
        logger.exception("model log record failed")


# ---------------------------------------------------------------------------
# Client instrumentation
# ---------------------------------------------------------------------------

# provider → dotted attribute paths (relative to the client) to wrap.
_ENDPOINTS: dict[str, tuple[str, ...]] = {
    "anthropic": (
        "messages.create",
        "messages.batches.create",
        "messages.batches.results",
    ),
    "openai": (
        "chat.completions.create",
        "responses.create",
        "embeddings.create",
    ),
}


class _TeeAsyncIter:
    """Pass an async iterator through, logging everything it yielded."""

    def __init__(self, inner: Any, done: Any) -> None:
        self._inner = inner
        self._done = done
        self._items: list[Any] = []
        self._logged = False

    def __aiter__(self) -> "_TeeAsyncIter":
        self._inner = self._inner.__aiter__()
        return self

    async def __anext__(self) -> Any:
        try:
            item = await self._inner.__anext__()
        except StopAsyncIteration:
            self._finish(None)
            raise
        except BaseException as e:
            self._finish(e)
            raise
        self._items.append(item)
        return item

    def _finish(self, error: BaseException | None) -> None:
        if not self._logged:
            self._logged = True
            self._done(self._items, error)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _wrap(fn: Any, provider: str, endpoint: str) -> Any:
    def _snapshot(args: tuple, kwargs: dict) -> tuple[dict, str | None]:
        request: dict[str, Any] = dict(kwargs)
        if args:
            request["$args"] = list(args)
        return request, _dumps(request)

    def _record(request, request_json, caller, t0, response, error) -> None:
        log_call(
            provider=provider, endpoint=endpoint, request=request,
            request_json=request_json, response=response, error=error,
            latency_ms=int((time.monotonic() - t0) * 1000), caller=caller,
        )

    def _finish(request, request_json, caller, t0, result) -> Any:
        # An async-iterator result (batch results) is logged once drained;
        # anything else is logged now.
        if hasattr(result, "__aiter__") and not hasattr(result, "model_dump"):
            return _TeeAsyncIter(
                result,
                lambda items, err: _record(
                    request, request_json, caller, t0, items, err
                ),
            )
        _record(request, request_json, caller, t0, result, None)
        return result

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            caller = _caller(2)
            request, request_json = _snapshot(args, kwargs)
            t0 = time.monotonic()
            try:
                result = await fn(*args, **kwargs)
            except BaseException as e:
                _record(request, request_json, caller, t0, None, e)
                raise
            return _finish(request, request_json, caller, t0, result)

        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        caller = _caller(1)
        request, request_json = _snapshot(args, kwargs)
        t0 = time.monotonic()
        try:
            result = fn(*args, **kwargs)
        except BaseException as e:
            _record(request, request_json, caller, t0, None, e)
            raise
        if inspect.isawaitable(result):
            # An async method hidden behind a plain-function decorator
            # (both SDKs do this on ``create``): log once it resolves.
            async def _await_and_log() -> Any:
                try:
                    resolved = await result
                except BaseException as e:
                    _record(request, request_json, caller, t0, None, e)
                    raise
                return _finish(request, request_json, caller, t0, resolved)

            return _await_and_log()
        return _finish(request, request_json, caller, t0, result)

    return sync_wrapper


def instrument(client: Any, provider: str) -> Any:
    """Patch ``client``'s model-call methods to log every call. Returns
    the same client. Idempotent; never raises."""
    try:
        if client is None or getattr(client, _INSTRUMENTED_FLAG, False):
            return client
        for path in _ENDPOINTS.get(provider, ()):
            *parents, leaf = path.split(".")
            owner = client
            try:
                for name in parents:
                    owner = getattr(owner, name)
                fn = getattr(owner, leaf)
            except AttributeError:
                continue
            if not callable(fn):
                continue
            setattr(owner, leaf, _wrap(fn, provider, path))
        setattr(client, _INSTRUMENTED_FLAG, True)
    except Exception:
        logger.exception("model log: could not instrument %s client", provider)
    return client
