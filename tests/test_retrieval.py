import numpy as np
import pytest

from src.retrieval import vector_index
from src.retrieval.build_index import build_index
from src.retrieval.retriever import RetrievalError, Retriever
from src.retrieval.vector_index import VectorIndex


def test_build_and_retrieve(tmp_cfg, fake_embedder):
    meta = build_index(tmp_cfg, embedder=fake_embedder)
    assert meta["n_documents"] == 3 and meta["n_chunks"] >= 3
    retriever = Retriever.from_config(tmp_cfg, embedder=fake_embedder)
    res = retriever.retrieve("What is the capital city of Australia?")
    assert len(res) == 2
    assert res[0]["doc_id"] == "wiki_1"
    assert [r["rank"] for r in res] == [1, 2]
    assert res[0]["score"] >= res[1]["score"]
    assert {"chunk_id", "title", "section", "text", "score"} <= res[0].keys()


def test_retriever_rejects_empty_question(tmp_cfg, fake_embedder):
    build_index(tmp_cfg, embedder=fake_embedder)
    retriever = Retriever.from_config(tmp_cfg, embedder=fake_embedder)
    with pytest.raises(RetrievalError):
        retriever.retrieve("   ")


def test_stale_index_detected(tmp_cfg, fake_embedder):
    build_index(tmp_cfg, embedder=fake_embedder)
    tmp_cfg["retrieval"]["chunk_size"] = 99
    with pytest.raises(RetrievalError, match="different settings"):
        Retriever.from_config(tmp_cfg, embedder=fake_embedder)


def test_missing_index(tmp_cfg, fake_embedder):
    with pytest.raises(RetrievalError):
        Retriever.from_config(tmp_cfg, embedder=fake_embedder)


def test_numpy_and_faiss_agree():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(50, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = emb[7] + 0.01
    s_np, i_np = VectorIndex(emb, use_faiss=False).search(q, 5)
    assert i_np[0, 0] == 7
    if vector_index.faiss is not None:
        s_f, i_f = VectorIndex(emb, use_faiss=True).search(q, 5)
        assert (i_f == i_np).all()
        assert np.allclose(s_f, s_np, atol=1e-5)


def test_index_save_load(tmp_path):
    emb = np.eye(4, dtype=np.float32)
    VectorIndex(emb).save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded.size == 4
    with pytest.raises(FileNotFoundError):
        VectorIndex.load(tmp_path / "missing")
