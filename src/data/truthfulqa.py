"""Download, verify and parse the TruthfulQA dataset.

TruthfulQA (Lin, Hilton & Evans, ACL 2022) — https://github.com/sylinrl/TruthfulQA
License: Apache-2.0.

Relevant CSV columns
--------------------
Type                 Adversarial / Non-Adversarial
Category             one of 37 topical categories (Misconceptions, Law, ...)
Question             the question posed to the LLM
Best Answer          the single best correct reference answer
Best Incorrect Answer
Correct Answers      ';'-separated list of acceptable (true) answers
Incorrect Answers    ';'-separated list of false answers (common misconceptions)
Source               URL supporting the correct answer (often Wikipedia)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd
import requests

from src.utils import get_logger, sha256_file

log = get_logger(__name__)

EXPECTED_COLUMNS = [
    "Type", "Category", "Question", "Best Answer", "Best Incorrect Answer",
    "Correct Answers", "Incorrect Answers", "Source",
]
_WIKI_HOST_RE = re.compile(r"^(en\.)(m\.)?wikipedia\.org$")


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be downloaded, verified or parsed."""


def download_dataset(url: str, dest: str | Path, expected_sha256: str | None = None,
                     timeout: int = 60) -> Path:
    """Download the CSV once (skips if present) and verify its SHA-256 checksum."""
    dest = Path(dest)
    if not dest.exists():
        log.info("Downloading TruthfulQA from %s", url)
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise DatasetError(
                f"Could not download the dataset from {url}: {e}. "
                "Check your internet connection or place the CSV manually at "
                f"{dest}."
            ) from e
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
    else:
        log.info("Using existing dataset file %s", dest)

    if expected_sha256:
        actual = sha256_file(dest)
        if actual != expected_sha256:
            raise DatasetError(
                f"Checksum mismatch for {dest}: expected {expected_sha256}, got {actual}. "
                "The file is not the pinned dataset version; delete it and re-run."
            )
        log.info("SHA-256 verified (%s...)", actual[:12])
    return dest


def split_answers(cell: Any) -> list[str]:
    """Split a ';'-separated answer cell into a clean list (TruthfulQA format)."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    return [a.strip() for a in str(cell).split(";") if a.strip()]


def wikipedia_title_from_url(url: Any) -> str | None:
    """Return the English Wikipedia article title for a URL, else None.

    'https://en.wikipedia.org/wiki/List_of_common_misconceptions#Health'
    -> 'List of common misconceptions'
    """
    if not isinstance(url, str) or not url.strip():
        return None
    parsed = urlparse(url.strip())
    if not _WIKI_HOST_RE.match(parsed.netloc.lower()):
        return None
    if not parsed.path.startswith("/wiki/"):
        return None
    title = unquote(parsed.path[len("/wiki/"):]).replace("_", " ").strip()
    return title or None


def load_truthfulqa(csv_path: str | Path) -> list[dict[str, Any]]:
    """Parse the TruthfulQA CSV into a list of question records.

    `question_id` is derived from the row position in the pinned CSV, so the
    same question always has the same ID.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise DatasetError(f"Dataset file not found: {csv_path}. Run `python -m src.data.prepare`.")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:  # pandas raises several exception types
        raise DatasetError(f"Could not parse {csv_path} as CSV: {e}") from e

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(f"Dataset is missing expected columns: {missing}")

    records = []
    for idx, row in df.iterrows():
        source = row["Source"] if isinstance(row["Source"], str) else ""
        records.append({
            "question_id": f"tqa_{idx:03d}",
            "row_index": int(idx),
            "type": row["Type"],
            "category": row["Category"],
            "question": str(row["Question"]).strip(),
            "best_answer": str(row["Best Answer"]).strip(),
            "best_incorrect_answer": str(row["Best Incorrect Answer"]).strip(),
            "correct_answers": split_answers(row["Correct Answers"]),
            "incorrect_answers": split_answers(row["Incorrect Answers"]),
            "source_url": source,
            "source_wikipedia_title": wikipedia_title_from_url(source),
        })
    return records
