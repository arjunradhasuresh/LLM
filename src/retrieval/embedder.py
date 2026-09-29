"""Sentence-embedding wrapper (Sentence Transformers).

Embeddings are L2-normalised, so the inner product of two vectors equals
their cosine similarity.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray:  # (n, dim) float32, normalised
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError(
                "sentence-transformers is not installed. Run `pip install -r requirements.txt`."
            ) from e
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("encode() called with an empty list of texts")
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=len(texts) > 500, convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def load_embedder(model_name: str, batch_size: int = 64) -> Embedder:
    return SentenceTransformerEmbedder(model_name, batch_size=batch_size)
