"""Step 3 — chunk the corpus, embed every chunk and build the vector index.

    python -m src.retrieval.build_index [--config config/config.yaml]

Outputs
  data/corpus/chunks.jsonl     all chunks (id, source article, section, text)
  data/index/embeddings.npy    (n_chunks x dim) normalised chunk embeddings
  data/index/index.faiss       FAISS IndexFlatIP over the embeddings
  data/index/index_meta.json   settings used (checked at query time)
"""
from __future__ import annotations

from typing import Any

from src.config import base_arg_parser, load_config, resolve_path
from src.retrieval.chunking import chunk_corpus, embedding_text
from src.retrieval.embedder import Embedder, load_embedder
from src.retrieval.vector_index import VectorIndex
from src.utils import get_logger, read_jsonl, sha256_file, utc_now, write_json, write_jsonl

log = get_logger(__name__)


def index_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Settings that determine the index; a mismatch means the index is stale."""
    r = cfg["retrieval"]
    return {
        "embedding_model": r["embedding_model"],
        "chunk_size": r["chunk_size"],
        "chunk_overlap": r["chunk_overlap"],
        "min_chunk_words": r.get("min_chunk_words", 1),
    }


def build_index(cfg: dict[str, Any], embedder: Embedder | None = None) -> dict[str, Any]:
    r = cfg["retrieval"]
    docs_path = resolve_path(cfg["corpus"]["documents_path"])
    if not docs_path.is_file():
        raise FileNotFoundError(f"{docs_path} not found — run `python -m src.data.build_corpus` first.")
    docs = read_jsonl(docs_path)
    if not docs:
        raise ValueError(f"Corpus {docs_path} is empty.")

    chunks = chunk_corpus(docs, r["chunk_size"], r["chunk_overlap"], r.get("min_chunk_words", 1))
    if not chunks:
        raise ValueError("Chunking produced no chunks — check the corpus and chunk settings.")
    chunks_path = resolve_path(cfg["corpus"]["chunks_path"])
    write_jsonl(chunks_path, chunks)
    log.info("%d documents -> %d chunks (size=%d, overlap=%d)",
             len(docs), len(chunks), r["chunk_size"], r["chunk_overlap"])

    embedder = embedder or load_embedder(r["embedding_model"], r.get("batch_size", 64))
    log.info("Embedding chunks with %s ...", r["embedding_model"])
    vectors = embedder.encode([embedding_text(c) for c in chunks])

    index = VectorIndex(vectors)
    index_dir = resolve_path(r["index_dir"])
    index.save(index_dir)
    meta = {
        **index_settings(cfg),
        "n_documents": len(docs),
        "n_chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "backend": index.backend,
        "corpus_sha256": sha256_file(docs_path),
        "chunks_sha256": sha256_file(chunks_path),
        "created_at": utc_now(),
    }
    write_json(index_dir / "index_meta.json", meta)
    log.info("Index saved to %s (backend=%s, dim=%d)", index_dir, index.backend, meta["dim"])
    return meta


def main() -> None:
    args = base_arg_parser(__doc__).parse_args()
    build_index(load_config(args.config))


if __name__ == "__main__":
    main()
