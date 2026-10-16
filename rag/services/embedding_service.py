"""
Embeddings. Provider is chosen with EMBEDDING_PROVIDER:
  local   Sentence Transformers on CPU (default, free, model loaded ONCE)
  openai  OpenAI-compatible /embeddings endpoint
Vectors are L2-normalised, so inner product == cosine similarity.
"""
import logging
import threading

import numpy as np
from django.conf import settings

from .cache_service import TTLCache
from .exceptions import ConfigError, EmbeddingError

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()
_query_cache = TTLCache(maxsize=1024, ttl=3600)


def get_model():
    """Load the local model once (thread-safe)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s ...", settings.EMBEDDING_MODEL)
                    _model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")
                except Exception as exc:
                    logger.exception("Could not load embedding model")
                    raise EmbeddingError(
                        "Could not load the embedding model. Check your internet connection "
                        "for the first download and that sentence-transformers is installed."
                    ) from exc
    return _model


def _encode_local(texts, batch_size):
    return get_model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)


def _encode_openai(texts):
    if not settings.EMBEDDING_API_KEY:
        raise ConfigError("Missing required environment variable: EMBEDDING_API_KEY.")
    from openai import OpenAI

    client = OpenAI(api_key=settings.EMBEDDING_API_KEY, base_url=settings.EMBEDDING_BASE_URL or None)
    vectors = []
    for start in range(0, len(texts), 100):
        response = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=texts[start:start + 100])
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    matrix = np.asarray(vectors, dtype="float32")
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def embed_texts(texts, batch_size: int = 64) -> np.ndarray:
    """-> float32 array (len(texts), dimension)."""
    texts = list(texts)
    if not texts:
        raise EmbeddingError("There is no text to embed.")
    try:
        if settings.EMBEDDING_PROVIDER == "openai":
            vectors = _encode_openai(texts)
        else:
            vectors = _encode_local(texts, batch_size)
    except (EmbeddingError, ConfigError):
        raise
    except Exception as exc:
        logger.exception("Embedding failed")
        raise EmbeddingError() from exc
    return np.ascontiguousarray(vectors, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    """Embed one question (cached: repeated questions cost nothing)."""
    key = (settings.EMBEDDING_MODEL_ID, text)
    cached = _query_cache.get(key)
    if cached is not None:
        return cached
    vector = embed_texts([text])[0]
    _query_cache.set(key, vector)
    return vector


def clear_query_cache():
    _query_cache.clear()
