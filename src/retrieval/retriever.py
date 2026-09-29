"""Dense retriever: question -> embedding -> top-k most similar corpus chunks."""
from __future__ import annotations

import json
from typing import Any

from src.config import resolve_path
from src.retrieval.build_index import index_settings
from src.retrieval.embedder import Embedder, load_embedder
from src.retrieval.vector_index import VectorIndex
from src.utils import read_jsonl


class RetrievalError(RuntimeError):
    pass


class Retriever:
    def __init__(self, chunks: list[dict[str, Any]], index: VectorIndex,
                 embedder: Embedder, top_k: int):
        if len(chunks) != index.size:
            raise RetrievalError(
                f"Index has {index.size} vectors but there are {len(chunks)} chunks — rebuild the index."
            )
        if top_k <= 0:
            raise RetrievalError("top_k must be positive")
        self.chunks = chunks
        self.index = index
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return the top-k chunks with rank (1 = best) and cosine similarity score."""
        if not question or not question.strip():
            raise RetrievalError("Cannot retrieve for an empty question")
        k = top_k or self.top_k
        query_vec = self.embedder.encode([question])
        scores, idx = self.index.search(query_vec, k)
        results = []
        for rank, (i, s) in enumerate(zip(idx[0], scores[0]), start=1):
            if i < 0:  # FAISS pads with -1 if fewer than k vectors exist
                continue
            c = self.chunks[int(i)]
            results.append({
                "rank": rank,
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "title": c["title"],
                "section": c["section"],
                "score": round(float(s), 6),
                "text": c["text"],
            })
        if not results:
            raise RetrievalError(f"Retrieval returned no results for question: {question!r}")
        return results

    @classmethod
    def from_config(cls, cfg: dict[str, Any], embedder: Embedder | None = None) -> "Retriever":
        r = cfg["retrieval"]
        index_dir = resolve_path(r["index_dir"])
        meta_path = index_dir / "index_meta.json"
        if not meta_path.is_file():
            raise RetrievalError(f"{meta_path} missing — run `python -m src.retrieval.build_index`.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = index_settings(cfg)
        stale = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
        if stale:
            raise RetrievalError(
                f"Index was built with different settings {stale} (index, config). "
                "Re-run `python -m src.retrieval.build_index`."
            )
        chunks = read_jsonl(resolve_path(cfg["corpus"]["chunks_path"]))
        index = VectorIndex.load(index_dir)
        embedder = embedder or load_embedder(r["embedding_model"], r.get("batch_size", 64))
        return cls(chunks, index, embedder, r["top_k"])
