"""Text embedding generation: local MiniLM, ONNX, API, or disabled.

Provides embed() and embed_batch() for creating 384-dimensional float32
embeddings used by store.py on memory creation and by search.py for query
embedding at search time.

Backends, in preference order:

1. **Local** — sentence-transformers all-MiniLM-L6-v2, lazy-loaded.
   Free, private, offline. Always wins when importable.
2. **ONNX** — the same all-MiniLM-L6-v2 via ``onnxruntime`` + a
   ``tokenizers`` tokenizer, from ``models.embedding_onnx`` (env
   ``BOXBOT_MODEL_EMBEDDING_ONNX`` = path to the .onnx export;
   ``tokenizer.json`` alongside). For hosts where torch does not fit
   (small aarch64 hosts) but onnxruntime is already loaded for speaker
   embeddings. ~tens of ms per query vs ~0.5s for the API round-trip.
3. **API** — OpenAI/Azure ``text-embedding-3-*`` via ``models.embedding``
   (env ``BOXBOT_MODEL_EMBEDDING``). Requests
   ``dimensions=EMBEDDING_DIM`` so vectors share the store's 384-wide
   layout. A per-call API failure returns None — the row gets a NULL
   vector and can be backfilled later (``scripts/reembed_memories.py``).
4. **Disabled** — embed() returns None, callers store NULL vectors, and
   search falls back to keyword-only ranking. Fabricated vectors are
   worse than none — they outrank exact keyword matches with noise.

The backends produce vectors in *different spaces* (ONNX is the MiniLM
space modulo quantization noise, and reports a distinct model name so
provenance stays honest); the store's ``embedding_model`` marker
(store.py) warns when stored vectors and the active backend disagree.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
MODEL_NAME = "all-MiniLM-L6-v2"

# Max inputs per API embeddings request (OpenAI hard limit is 2048).
_API_BATCH_LIMIT = 512

# MiniLM's max sequence length — sentence-transformers truncates here.
_ONNX_MAX_TOKENS = 256

# Lazy-loaded local model singleton
_model = None
_unavailable = False

# Lazy-loaded ONNX backend: (session, tokenizer, input_names).
_onnx = None
_onnx_unavailable = False

# Lazy-built API backend: (client, model_id) once resolved.
_api_client = None
_api_model: str | None = None
_api_unavailable = False
_api_call_failed = False  # first per-call failure logs at WARNING, rest DEBUG


def _load_model() -> None:
    """Load the sentence-transformers model on first use."""
    global _model, _unavailable

    if _model is not None or _unavailable:
        return

    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s...", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        _unavailable = True
        _load_onnx()
        if _onnx is not None:
            logger.info(
                "sentence-transformers unavailable (%s); using local ONNX "
                "embedder (%s).", e, _onnx_model_name(),
            )
            return
        _load_api()
        if _api_client is not None:
            logger.info(
                "Local embedder unavailable (%s); using API embeddings "
                "(%s, dimensions=%d).",
                e,
                _api_model,
                EMBEDDING_DIM,
            )
        else:
            logger.warning(
                "Vector search disabled: sentence_transformers unavailable (%s) "
                "and no API embedding model configured; using keyword search "
                "only. Install sentence-transformers, set "
                "BOXBOT_MODEL_EMBEDDING_ONNX to a local MiniLM export, or "
                "set BOXBOT_MODEL_EMBEDDING to restore semantic ranking.",
                e,
            )


def _onnx_model_name() -> str:
    """Provenance name for ONNX-produced vectors.

    Same space as MiniLM modulo quantization noise, but reported
    distinctly so the store's provenance marker and the hot-task
    centroid fingerprint invalidate on a backend switch instead of
    silently mixing near-identical spaces.
    """
    return f"{MODEL_NAME}-onnx"


def _load_onnx() -> None:
    """Build (once) the ONNX embedding backend, if configured.

    Needs ``models.embedding_onnx`` (path to the .onnx export) with
    ``tokenizer.json`` in the same directory, plus the ``onnxruntime``
    and ``tokenizers`` packages. Any missing piece leaves the backend
    disabled; the caller logs the combined outcome.
    """
    global _onnx, _onnx_unavailable

    if _onnx is not None or _onnx_unavailable:
        return
    _onnx_unavailable = True  # one attempt; flipped back on success

    try:
        from boxbot.core.config import get_config

        model_path = get_config().models.embedding_onnx
    except Exception:
        return
    if not model_path:
        return

    from pathlib import Path

    onnx_file = Path(model_path)
    tokenizer_file = onnx_file.parent / "tokenizer.json"
    if not onnx_file.is_file() or not tokenizer_file.is_file():
        logger.warning(
            "BOXBOT_MODEL_EMBEDDING_ONNX=%s but the model or its "
            "tokenizer.json is missing; ONNX embeddings disabled.",
            model_path,
        )
        return

    try:
        import onnxruntime
        from tokenizers import Tokenizer
    except ImportError as e:
        logger.warning(
            "models.embedding_onnx is set but %s; ONNX embeddings "
            "disabled.", e,
        )
        return

    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        tokenizer.enable_truncation(max_length=_ONNX_MAX_TOKENS)
        session = onnxruntime.InferenceSession(
            str(onnx_file), providers=["CPUExecutionProvider"]
        )
    except Exception:
        logger.exception("Failed to load ONNX embedding model %s", onnx_file)
        return

    input_names = {i.name for i in session.get_inputs()}
    _onnx = (session, tokenizer, input_names)
    _onnx_unavailable = False
    logger.info("ONNX embedding model loaded from %s", onnx_file)


def _onnx_embed(texts: list[str]) -> list[np.ndarray]:
    """Embed ``texts`` with the ONNX backend.

    Mirrors the sentence-transformers pipeline for all-MiniLM-L6-v2
    exactly: WordPiece tokenize (truncate at 256), transformer, mean
    pooling over the attention mask, L2 normalize.
    """
    session, tokenizer, input_names = _onnx

    encodings = tokenizer.encode_batch(texts)
    max_len = max(len(e.ids) for e in encodings)
    batch = len(encodings)
    input_ids = np.zeros((batch, max_len), dtype=np.int64)
    attention_mask = np.zeros((batch, max_len), dtype=np.int64)
    for i, enc in enumerate(encodings):
        n = len(enc.ids)
        input_ids[i, :n] = enc.ids
        attention_mask[i, :n] = enc.attention_mask

    feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in input_names:
        feeds["token_type_ids"] = np.zeros((batch, max_len), dtype=np.int64)

    # First output is last_hidden_state: (batch, seq, EMBEDDING_DIM).
    hidden = session.run(None, feeds)[0]
    mask = attention_mask[:, :, None].astype(np.float32)
    summed = (hidden.astype(np.float32) * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    pooled = summed / counts
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    pooled = pooled / np.clip(norms, 1e-12, None)
    return [row.astype(np.float32) for row in pooled]


def _load_api() -> None:
    """Build (once) the API embedding client, if configured.

    Requires ``models.embedding`` plus an OpenAI key; mirrors the Azure/
    public routing of the agent's chat client. Any missing piece —
    config not loaded, no key, no ``openai`` package — silently leaves
    the backend disabled (the caller logs the combined outcome).
    """
    global _api_client, _api_model, _api_unavailable

    if _api_client is not None or _api_unavailable:
        return
    _api_unavailable = True  # one attempt; flipped back on success

    try:
        from boxbot.core.config import get_config

        config = get_config()
    except Exception:
        return
    model = config.models.embedding
    api_key = config.api_keys.openai
    if not model or not api_key:
        return

    try:
        import openai
    except ImportError:
        logger.warning(
            "models.embedding=%s is set but the `openai` package is not "
            "installed; API embeddings disabled.",
            model,
        )
        return

    oa = config.openai
    if oa.is_azure:
        if not (oa.api_base and oa.api_version):
            logger.warning(
                "Azure OpenAI embeddings need OPENAI_API_BASE and "
                "OPENAI_API_VERSION; API embeddings disabled."
            )
            return
        client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=oa.api_base,
            api_version=oa.api_version,
            timeout=10.0,
            max_retries=1,
        )
    else:
        client = openai.OpenAI(
            api_key=api_key, base_url=oa.api_base, timeout=10.0, max_retries=1
        )

    _api_client = client
    _api_model = model
    _api_unavailable = False


def _api_embed(texts: list[str]) -> list[np.ndarray | None]:
    """Embed ``texts`` through the API backend; Nones on failure."""
    global _api_call_failed

    results: list[np.ndarray | None] = []
    try:
        for start in range(0, len(texts), _API_BATCH_LIMIT):
            chunk = texts[start : start + _API_BATCH_LIMIT]
            response = _api_client.embeddings.create(
                model=_api_model, input=chunk, dimensions=EMBEDDING_DIM
            )
            results.extend(
                np.asarray(item.embedding, dtype=np.float32)
                for item in response.data
            )
        _api_call_failed = False
        return results
    except Exception as e:
        log = logger.debug if _api_call_failed else logger.warning
        log(
            "API embedding call failed (%s): %s — storing NULL vectors; "
            "backfill with scripts/reembed_memories.py once resolved.",
            _api_model,
            e,
        )
        _api_call_failed = True
        return [None] * len(texts)


def active_model() -> str | None:
    """Name of the model producing embeddings, or None when disabled."""
    _load_model()
    if not _unavailable:
        return MODEL_NAME
    if _onnx is not None:
        return _onnx_model_name()
    return _api_model if _api_client is not None else None


def embed(text: str) -> np.ndarray | None:
    """Generate a 384-dimensional embedding for a text string.

    Args:
        text: The text to embed.

    Returns:
        A float32 numpy array of shape (384,), or None when no embedding
        model is available (or the API call failed).
    """
    _load_model()

    if not _unavailable:
        result = _model.encode(text, normalize_embeddings=True)
        return np.asarray(result, dtype=np.float32)
    if _onnx is not None:
        return _onnx_embed([text])[0]
    if _api_client is not None:
        return _api_embed([text])[0]
    return None


def embed_batch(texts: list[str]) -> list[np.ndarray | None]:
    """Generate embeddings for a batch of texts.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of float32 numpy arrays, each of shape (384,) — or Nones
        when no embedding model is available.
    """
    if not texts:
        return []

    _load_model()

    if not _unavailable:
        results = _model.encode(texts, normalize_embeddings=True)
        return [np.asarray(r, dtype=np.float32) for r in results]
    if _onnx is not None:
        return list(_onnx_embed(texts))
    if _api_client is not None:
        return _api_embed(texts)
    return [None] * len(texts)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1]. Returns 0.0 if either vector is zero.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
