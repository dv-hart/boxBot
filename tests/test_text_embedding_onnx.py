"""Tests for the ONNX text-embedding backend (embeddings.py).

The real model is a 23MB artifact that lives on-device, so these tests
exercise the pipeline with fakes: the pooling/normalization math, the
backend routing, and the load-time gating. An opt-in parity test runs
when ``BOXBOT_TEST_EMBEDDING_ONNX`` points at a real export.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from boxbot.memory import embeddings


@pytest.fixture(autouse=True)
def reset_backends():
    """Isolate every test from the module's lazy-loaded singletons."""
    saved = (
        embeddings._model, embeddings._unavailable,
        embeddings._onnx, embeddings._onnx_unavailable,
        embeddings._api_client, embeddings._api_model,
        embeddings._api_unavailable,
    )
    yield
    (
        embeddings._model, embeddings._unavailable,
        embeddings._onnx, embeddings._onnx_unavailable,
        embeddings._api_client, embeddings._api_model,
        embeddings._api_unavailable,
    ) = saved


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = list(ids)
        self.attention_mask = [1] * len(ids)


class _FakeTokenizer:
    """Deterministic 'tokenizer': one id per whitespace token."""

    def encode_batch(self, texts):
        return [
            _FakeEncoding(range(1, len(t.split()) + 1)) for t in texts
        ]


class _FakeSession:
    """Returns a hidden state where token i is one-hot at dim i."""

    def __init__(self, dim=embeddings.EMBEDDING_DIM, expect_inputs=None):
        self.dim = dim
        self.expect_inputs = expect_inputs
        self.seen_feeds = None

    def run(self, _outputs, feeds):
        self.seen_feeds = feeds
        if self.expect_inputs is not None:
            assert set(feeds) == self.expect_inputs
        ids = feeds["input_ids"]
        batch, seq = ids.shape
        hidden = np.zeros((batch, seq, self.dim), dtype=np.float32)
        for b in range(batch):
            for t in range(seq):
                hidden[b, t, t % self.dim] = 1.0
        return [hidden]


def _install_fake_onnx(monkeypatch, input_names={"input_ids", "attention_mask"}):
    session = _FakeSession(expect_inputs=set(input_names))
    monkeypatch.setattr(embeddings, "_model", None)
    monkeypatch.setattr(embeddings, "_unavailable", True)
    monkeypatch.setattr(
        embeddings, "_onnx", (session, _FakeTokenizer(), set(input_names))
    )
    monkeypatch.setattr(embeddings, "_onnx_unavailable", False)
    return session


def test_onnx_embed_mean_pools_and_normalizes(monkeypatch):
    _install_fake_onnx(monkeypatch)
    vec = embeddings.embed("lock the front door")  # 4 tokens
    assert vec is not None
    assert vec.shape == (embeddings.EMBEDDING_DIM,)
    assert vec.dtype == np.float32
    # 4 one-hot tokens mean-pooled -> 0.25 at dims 0-3, then L2-normalized.
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)
    assert np.allclose(vec[:4], vec[0]) and vec[0] > 0
    assert np.allclose(vec[4:], 0.0)


def test_onnx_embed_batch_pads_and_masks(monkeypatch):
    session = _install_fake_onnx(monkeypatch)
    out = embeddings.embed_batch(["one", "one two three"])
    assert len(out) == 2
    # Padding must be masked out of the mean: the 1-token text pools
    # only its own token even though it was padded to length 3.
    assert np.isclose(np.linalg.norm(out[0]), 1.0, atol=1e-5)
    assert np.isclose(out[0][0], 1.0, atol=1e-5)
    mask = session.seen_feeds["attention_mask"]
    assert mask.tolist() == [[1, 0, 0], [1, 1, 1]]


def test_onnx_feeds_token_type_ids_only_when_model_wants_them(monkeypatch):
    session = _install_fake_onnx(
        monkeypatch,
        input_names={"input_ids", "attention_mask", "token_type_ids"},
    )
    embeddings.embed("hello there")
    assert np.all(session.seen_feeds["token_type_ids"] == 0)


def test_active_model_reports_distinct_onnx_name(monkeypatch):
    _install_fake_onnx(monkeypatch)
    assert embeddings.active_model() == f"{embeddings.MODEL_NAME}-onnx"


def test_onnx_disabled_without_config(monkeypatch, mock_config):
    """No models.embedding_onnx -> backend stays off, no exception."""
    monkeypatch.setattr(embeddings, "_onnx", None)
    monkeypatch.setattr(embeddings, "_onnx_unavailable", False)
    embeddings._load_onnx()
    assert embeddings._onnx is None
    assert embeddings._onnx_unavailable is True


def test_onnx_disabled_when_files_missing(monkeypatch, mock_config):
    mock_config.models.embedding_onnx = "/nonexistent/model.onnx"
    monkeypatch.setattr(embeddings, "_onnx", None)
    monkeypatch.setattr(embeddings, "_onnx_unavailable", False)
    embeddings._load_onnx()
    assert embeddings._onnx is None


@pytest.mark.skipif(
    not os.environ.get("BOXBOT_TEST_EMBEDDING_ONNX"),
    reason="set BOXBOT_TEST_EMBEDDING_ONNX=/path/to/model.onnx to run",
)
def test_real_model_parity(monkeypatch, mock_config):
    """Integration: real export loads and produces sane similarities."""
    mock_config.models.embedding_onnx = os.environ[
        "BOXBOT_TEST_EMBEDDING_ONNX"
    ]
    monkeypatch.setattr(embeddings, "_onnx", None)
    monkeypatch.setattr(embeddings, "_onnx_unavailable", False)
    monkeypatch.setattr(embeddings, "_unavailable", True)
    embeddings._load_onnx()
    assert embeddings._onnx is not None
    a = embeddings.embed("lock the front door")
    b = embeddings.embed("please lock the door")
    c = embeddings.embed("what is the weather tomorrow")
    assert embeddings.cosine_similarity(a, b) > 0.7
    assert embeddings.cosine_similarity(a, b) > embeddings.cosine_similarity(a, c)
