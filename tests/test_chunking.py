import pytest

from src.data.build_corpus import html_to_sections
from src.retrieval.chunking import chunk_document, chunk_words, embedding_text


def test_chunk_words_overlap():
    spans = chunk_words(list("abcdefghij"), 4, 2)
    assert spans == [(0, 4), (2, 6), (4, 8), (6, 10)]


def test_chunk_words_covers_everything_without_overlap():
    words = [str(i) for i in range(23)]
    spans = chunk_words(words, 10, 0)
    assert spans == [(0, 10), (10, 20), (20, 23)]


def test_tiny_tail_is_merged():
    spans = chunk_words([str(i) for i in range(21)], 10, 0, min_chunk_words=3)
    assert spans == [(0, 10), (10, 21)]


def test_empty_and_invalid():
    assert chunk_words([], 10, 2) == []
    with pytest.raises(ValueError):
        chunk_words(["a"], 10, 10)
    with pytest.raises(ValueError):
        chunk_words(["a"], 0, 0)


def test_chunk_document_metadata():
    doc = {"doc_id": "wiki_9", "title": "T", "sections": [
        {"section": "Intro", "text": " ".join(f"w{i}" for i in range(15))},
        {"section": "History", "text": " ".join(f"h{i}" for i in range(15))}]}
    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)
    assert chunks[0]["chunk_id"] == "wiki_9_c0000"
    assert chunks[0]["section"] == "Intro"
    assert chunks[-1]["section"] == "History"
    assert all(c["doc_id"] == "wiki_9" for c in chunks)
    # overlap: last 2 words of chunk 0 start chunk 1
    assert chunks[0]["text"].split()[-2:] == chunks[1]["text"].split()[:2]
    assert embedding_text(chunks[0]).startswith("T: ")


def test_html_to_sections_cleans_wikipedia_html():
    html = """<div class="mw-parser-output">
      <table class="infobox"><tr><td>Infobox junk</td></tr></table>
      <p>Canberra is the capital of Australia.<sup class="reference">[1]</sup></p>
      <div class="mw-heading"><h2>History</h2></div>
      <p>Founded in 1913 by decision of parliament.</p>
      <ul><li>First item of a list here</li><li>Second item here too</li></ul>
      <table class="wikitable"><tr><th>Year</th><th>Event</th></tr><tr><td>1913</td><td>Founded</td></tr></table>
      <h2>References</h2><p>Should not appear in the output at all.</p>
    </div>"""
    blocks = html_to_sections(html)
    texts = [b["text"] for b in blocks]
    assert "Canberra is the capital of Australia." in texts
    assert not any("Infobox" in t or "[1]" in t or "Should not appear" in t for t in texts)
    assert {"section": "History", "text": "1913 | Founded"} in blocks
    assert blocks[0]["section"] == "Introduction"
    assert any(b["section"] == "History" and "list" in b["text"] for b in blocks)
