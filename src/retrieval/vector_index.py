"""Exact (brute-force) cosine-similarity vector index.

Uses FAISS `IndexFlatIP` when available. With normalised vectors, inner
product == cosine similarity, and a *flat* index compares the query with
every chunk, so results are exact and deterministic (no approximation).
If FAISS is not installed, an equivalent NumPy implementation is used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # optional dependency
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    faiss = None


class VectorIndex:
    def __init__(self, embeddings: np.ndarray, use_faiss: bool = True):
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty 2-D array")
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.backend = "faiss" if (use_faiss and faiss is not None) else "numpy"
        if self.backend == "faiss":
            self._index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self._index.add(self.embeddings)

    @property
    def size(self) -> int:
        return self.embeddings.shape[0]

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices), each of shape (n_queries, k), best first."""
        q = np.ascontiguousarray(np.atleast_2d(query_vectors), dtype=np.float32)
        k = min(k, self.size)
        if self.backend == "faiss":
            scores, idx = self._index.search(q, k)
            return scores, idx
        sims = q @ self.embeddings.T
        # stable sort -> ties broken by lower chunk index (deterministic)
        idx = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(sims, idx, axis=1), idx

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        if self.backend == "faiss":
            faiss.write_index(self._index, str(directory / "index.faiss"))

    @classmethod
    def load(cls, directory: str | Path, use_faiss: bool = True) -> "VectorIndex":
        path = Path(directory) / "embeddings.npy"
        if not path.is_file():
            raise FileNotFoundError(
                f"No vector index at {directory}. Run `python -m src.retrieval.build_index` first."
            )
        return cls(np.load(path), use_faiss=use_faiss)
