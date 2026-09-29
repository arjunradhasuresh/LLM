"""Step 1 — download TruthfulQA and draw the fixed evaluation question set.

    python -m src.data.prepare [--config config/config.yaml]

Outputs
  data/raw/TruthfulQA.csv            original dataset (checksum-verified, unmodified)
  data/processed/questions.jsonl     the sampled questions + reference answers
  data/processed/sampling_report.json  pool size, seed, category counts, IDs

The sample is a simple random sample (no cherry-picking) drawn with
`experiment.random_seed` from all eligible questions. Both experimental
conditions later read exactly this file.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Any

from src.config import base_arg_parser, load_config, resolve_path
from src.data.truthfulqa import DatasetError, download_dataset, load_truthfulqa
from src.utils import get_logger, utc_now, write_json, write_jsonl

log = get_logger(__name__)


def eligible_questions(records: list[dict[str, Any]], require_wikipedia: bool) -> list[dict[str, Any]]:
    """Filter the pool: questions need reference answers (and a Wikipedia source if required)."""
    pool = [r for r in records if r["question"] and r["correct_answers"] and r["incorrect_answers"]]
    if require_wikipedia:
        pool = [r for r in pool if r["source_wikipedia_title"]]
    return pool


def sample_questions(pool: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Seeded simple random sample of `n` questions, returned in ID order."""
    if n > len(pool):
        raise DatasetError(
            f"dataset.number_of_questions={n} but only {len(pool)} eligible questions exist."
        )
    rng = random.Random(seed)
    chosen = rng.sample(pool, n)
    return sorted(chosen, key=lambda r: r["row_index"])


def main() -> None:
    args = base_arg_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    ds = cfg["dataset"]

    csv_path = download_dataset(ds["url"], resolve_path(ds["raw_path"]), ds.get("sha256"))
    records = load_truthfulqa(csv_path)
    pool = eligible_questions(records, ds.get("require_wikipedia_source", True))
    sample = sample_questions(pool, ds["number_of_questions"], cfg["experiment"]["random_seed"])

    out_path = resolve_path(ds["processed_path"])
    write_jsonl(out_path, sample)

    report = {
        "created_at": utc_now(),
        "dataset_url": ds["url"],
        "dataset_sha256": ds.get("sha256"),
        "total_questions_in_dataset": len(records),
        "require_wikipedia_source": ds.get("require_wikipedia_source", True),
        "eligible_pool_size": len(pool),
        "number_of_questions": len(sample),
        "random_seed": cfg["experiment"]["random_seed"],
        "category_counts": dict(Counter(r["category"] for r in sample).most_common()),
        "type_counts": dict(Counter(r["type"] for r in sample)),
        "question_ids": [r["question_id"] for r in sample],
    }
    write_json(out_path.parent / "sampling_report.json", report)
    log.info("Dataset: %d questions, eligible pool: %d, sampled: %d -> %s",
             len(records), len(pool), len(sample), out_path)


if __name__ == "__main__":
    main()
