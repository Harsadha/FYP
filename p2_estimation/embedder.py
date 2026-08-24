"""
P2 - Embedding generation (Day 1, 11:00-13:00).

embed(chunks) -> List[np.ndarray]

Uses sentence-transformers/all-MiniLM-L6-v2: small, runs on CPU, no API
key or rate limits to worry about -- important for a 3-day sprint.
Model is lazy-loaded and cached at module level so repeated calls
(e.g. across estimator tests) don't reload it each time.

Offline fallback: if the model can't be downloaded (no network / first
run on an offline machine), falls back to a deterministic bag-of-words
vector so the rest of the pipeline can still be exercised. This is a
degraded-quality stand-in, not a substitute for the real model -- a
warning is printed once so it's never mistaken for the real thing in a
demo. Remove this fallback once every dev machine has the model cached
locally (it exists purely to de-risk the sprint, not as permanent
architecture).
"""
from typing import List
import re
import warnings
import numpy as np
import hashlib

_model = None
_model_load_failed = False
_vocab: dict = {}  # only used by the offline fallback


def _get_model():
    global _model, _model_load_failed
    if _model is None and not _model_load_failed:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            _model_load_failed = True
            warnings.warn(
                f"Could not load sentence-transformers model ({e}). "
                "Falling back to a deterministic bag-of-words embedder. "
                "This is degraded quality -- fix network/model access before demoing.",
                RuntimeWarning,
            )
    return _model


def _bow_embed(chunks: List[str], dim: int = 256) -> List[np.ndarray]:
    """Deterministic offline fallback: hashed bag-of-words, L2-normalized."""
    vectors = []
    for text in chunks:
        vec = np.zeros(dim, dtype=np.float32)
        words = re.findall(r"[a-z0-9]+", text.lower())
        for w in words:
            idx = int(
                hashlib.sha256(w.encode()).hexdigest(),
                16
            ) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)
    return vectors


def embed(chunks: List[str]) -> List[np.ndarray]:
    if not chunks:
        return []
    model = _get_model()
    if model is None:
        return _bow_embed(chunks)
    vectors = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    return [v for v in vectors]
