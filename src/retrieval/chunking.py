"""Split corpus documents into overlapping word-window chunks.

Each chunk contains `chunk_size` words; consecutive chunks of the same
document share `chunk_overlap` words so that a fact is not lost when it
falls on a chunk boundary. Chunks never cross document boundaries.
"""
from __future__ import annotations

from typing import Any


def chunk_words(words: list[str], chunk_size: int, chunk_overlap: int,
                min_chunk_words: int = 1) -> list[tuple[int, int]]:
    """Return (start, end) word offsets of sliding windows over `words`.

    >>> chunk_words(list("abcdefghij"), 4, 2)
    [(0, 4), (2, 6), (4, 8), (6, 10)]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
    n = len(words)
    if n == 0:
        return []
    step = chunk_size - chunk_overlap
    spans = []
    start = 0
    while True:
        end = min(start + chunk_size, n)
        spans.append((start, end))
        if end >= n:
            break
        start += step
    # drop a tiny trailing fragment that is fully covered by the overlap anyway
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < min_chunk_words:
        spans.pop()
        spans[-1] = (spans[-1][0], n)
    return spans


def chunk_document(doc: dict[str, Any], chunk_size: int, chunk_overlap: int,
                   min_chunk_words: int = 1) -> list[dict[str, Any]]:
    """Chunk one corpus document; each chunk remembers its source and section."""
    words: list[str] = []
    word_section: list[str] = []
    for block in doc["sections"]:
        block_words = block["text"].split()
        words.extend(block_words)
        word_section.extend([block["section"]] * len(block_words))

    chunks = []
    for i, (start, end) in enumerate(chunk_words(words, chunk_size, chunk_overlap, min_chunk_words)):
        chunks.append({
            "chunk_id": f"{doc['doc_id']}_c{i:04d}",
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "section": word_section[start],
            "url": doc.get("url", ""),
            "revid": doc.get("revid"),
            "word_start": start,
            "word_end": end,
            "text": " ".join(words[start:end]),
        })
    return chunks


def chunk_corpus(docs: list[dict[str, Any]], chunk_size: int, chunk_overlap: int,
                 min_chunk_words: int = 1) -> list[dict[str, Any]]:
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap, min_chunk_words))
    return chunks


def embedding_text(chunk: dict[str, Any]) -> str:
    """Text that is embedded for a chunk: article title gives topical signal."""
    return f"{chunk['title']}: {chunk['text']}"
