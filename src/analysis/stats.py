"""Statistics for the paired RAG vs No-RAG comparison.

Why McNemar's test?
  Both conditions answer the *same* questions, so the observations are
  paired binary outcomes (correct / incorrect per question and condition).
  An unpaired test (chi-square on two proportions, two-sample t-test) would
  ignore this pairing and violate its independence assumption. McNemar's
  test uses only the discordant pairs:
      b = questions correct without RAG but incorrect with RAG
      c = questions incorrect without RAG but correct with RAG
  and tests H0: b and c are equally likely (retrieval does not change the
  probability of a correct answer). We use the *exact* (binomial) version,
  which is valid for any number of discordant pairs; the chi-square
  approximation is only reliable when b + c is large (rule of thumb >= 25),
  which is not guaranteed with 50–100 questions.
  Assumptions: binary outcome, paired observations, questions independent of
  each other. All hold by design (random sample, one answer per condition).

Confidence intervals (not tests):
  - Wilson score interval for each condition's accuracy
  - paired bootstrap (resampling questions) percentile CI for the difference
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binomtest


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (in [0, 1])."""
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_table(a: list[int], b: list[int]) -> dict[str, int]:
    """2x2 table for paired binary outcomes a (No-RAG) and b (RAG)."""
    if len(a) != len(b):
        raise ValueError("paired outcome lists must have equal length")
    both = sum(1 for x, y in zip(a, b) if x and y)
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if not x and y)
    neither = sum(1 for x, y in zip(a, b) if not x and not y)
    return {"both_correct": both, "no_rag_only_correct": only_a,
            "rag_only_correct": only_b, "both_incorrect": neither}


def mcnemar_exact(b: int, c: int) -> dict[str, float]:
    """Exact two-sided McNemar test on discordant counts b and c."""
    n = b + c
    if n == 0:
        return {"discordant_pairs": 0, "p_value": 1.0, "statistic_b": b, "statistic_c": c,
                "note": "no discordant pairs: conditions never disagree"}
    p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return {"discordant_pairs": n, "p_value": float(min(1.0, p)), "statistic_b": b,
            "statistic_c": c, "note": ""}


def bootstrap_diff_ci(a: list[int], b: list[int], n_boot: int = 10000, seed: int = 42,
                      confidence: float = 0.95) -> tuple[float, float]:
    """Percentile CI for mean(b) - mean(a), resampling questions (pairs kept)."""
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    if len(a_arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a_arr), size=(n_boot, len(a_arr)))
    diffs = b_arr[idx].mean(axis=1) - a_arr[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100])
    return (float(lo), float(hi))
