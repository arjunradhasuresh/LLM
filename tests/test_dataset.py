import pandas as pd
import pytest

from src.data.prepare import eligible_questions, sample_questions
from src.data.truthfulqa import (DatasetError, download_dataset, load_truthfulqa, split_answers,
                                 wikipedia_title_from_url)


def _write_csv(path, n=10):
    rows = []
    for i in range(n):
        src = f"https://en.wikipedia.org/wiki/Topic_{i}" if i % 2 == 0 else f"https://example.org/{i}"
        rows.append({"Type": "Adversarial", "Category": "Misconceptions", "Question": f"Question {i}?",
                     "Best Answer": f"True {i}", "Best Incorrect Answer": f"False {i}",
                     "Correct Answers": f"True {i}; Also true {i}", "Incorrect Answers": f"False {i}",
                     "Source": src})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_split_answers():
    assert split_answers("A; B ;C;; ") == ["A", "B", "C"]
    assert split_answers(float("nan")) == []


@pytest.mark.parametrize("url,title", [
    ("https://en.wikipedia.org/wiki/List_of_common_misconceptions#Health", "List of common misconceptions"),
    ("https://en.m.wikipedia.org/wiki/Caf%C3%A9", "Café"),
    ("https://de.wikipedia.org/wiki/Berlin", None),
    ("https://www.webmd.com/x", None),
    ("https://en.wikiquote.org/wiki/Misquotations", None),
    (None, None),
])
def test_wikipedia_title_from_url(url, title):
    assert wikipedia_title_from_url(url) == title


def test_load_truthfulqa(tmp_path):
    p = tmp_path / "tqa.csv"
    _write_csv(p)
    recs = load_truthfulqa(p)
    assert len(recs) == 10
    assert recs[0]["question_id"] == "tqa_000"
    assert recs[0]["correct_answers"] == ["True 0", "Also true 0"]
    assert recs[0]["source_wikipedia_title"] == "Topic 0"
    assert recs[1]["source_wikipedia_title"] is None


def test_load_missing_file(tmp_path):
    with pytest.raises(DatasetError):
        load_truthfulqa(tmp_path / "nope.csv")


def test_load_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"Question": ["q"]}).to_csv(p, index=False)
    with pytest.raises(DatasetError, match="missing expected columns"):
        load_truthfulqa(p)


def test_sampling_is_deterministic_and_filtered(tmp_path):
    p = tmp_path / "tqa.csv"
    _write_csv(p, n=20)
    pool = eligible_questions(load_truthfulqa(p), require_wikipedia=True)
    assert len(pool) == 10 and all(r["source_wikipedia_title"] for r in pool)
    s1 = sample_questions(pool, 5, seed=42)
    s2 = sample_questions(pool, 5, seed=42)
    s3 = sample_questions(pool, 5, seed=7)
    assert [r["question_id"] for r in s1] == [r["question_id"] for r in s2]
    assert [r["question_id"] for r in s1] != [r["question_id"] for r in s3]
    with pytest.raises(DatasetError):
        sample_questions(pool, 11, seed=42)


def test_checksum_mismatch(tmp_path):
    p = tmp_path / "tqa.csv"
    _write_csv(p)
    with pytest.raises(DatasetError, match="Checksum mismatch"):
        download_dataset("http://unused", p, expected_sha256="0" * 64)
