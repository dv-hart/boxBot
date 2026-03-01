"""On-device text embedding generation using MiniLM.

Provides embed() and embed_batch() for creating 384-dimensional float32
embeddings used by store.py on memory creation and by search.py for query
embedding at search time.

Uses sentence-transformers all-MiniLM-L6-v2 with lazy model loading.
Falls back to random embeddings with a warning if sentence-transformers
is unavailable (for development without the model installed).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384

# Lazy-loaded model singleton
_model = None
_use_fallback = False


def _load_model() -> None:
    """Load the sentence-transformers model on first use."""
    global _model, _use_fallback

    if _model is not None or _use_fallback:
        return

    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model all-MiniLM-L6-v2...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded successfully")
    except (ImportError, Exception) as e:
        _use_fallback = True
        warnings.warn(
            f"sentence-transformers unavailable ({e}); using random "
            f"embeddings. Install sentence-transformers for real search.",
            stacklevel=2,
        )
        logger.warning("Falling back to random embeddings: %s", e)


def embed(text: str) -> np.ndarray:
    """Generate a 384-dimensional embedding for a text string.

    Args:
        text: The text to embed.

    Returns:
        A float32 numpy array of shape (384,).
    """
    _load_model()

    if _use_fallback:
        # Deterministic fallback: hash-seeded random for consistency
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
        # Normalize to unit vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    result = _model.encode(text, normalize_embeddings=True)
    return np.asarray(result, dtype=np.float32)


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Generate embeddings for a batch of texts.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of float32 numpy arrays, each of shape (384,).
    """
    if not texts:
        return []

    _load_model()

    if _use_fallback:
        return [embed(t) for t in texts]

    results = _model.encode(texts, normalize_embeddings=True)
    return [np.asarray(r, dtype=np.float32) for r in results]


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
