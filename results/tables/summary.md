# Results summary

- Generated: 2026-09-29T01:26:42+00:00
- Answer model(s): gpt-4.1-mini-2025-04-14
- Judge model: gpt-4.1-2025-04-14
- Questions: 100 sampled, 100 in the paired analysis

## Factual accuracy

| Metric | n | No-RAG | RAG | Diff (pp) [95% CI] | Rel. change | McNemar b / c | exact p |
|---|---|---|---|---|---|---|---|
| strict_accuracy (PRIMARY) | 100 | 62.0% [52.2, 70.9] | 62.0% [52.2, 70.9] | +0.0 [-11.0, 11.0] | +0.0% | 16 / 16 | 1 |
| lenient_accuracy | 100 | 79.0% [70.0, 85.8] | 77.0% [67.8, 84.2] | -2.0 [-12.0, 8.0] | -2.5% | 15 / 13 | 0.8506 |
| decisive_only_accuracy | 98 | 63.3% [53.4, 72.1] | 63.3% [53.4, 72.1] | +0.0 [-11.2, 11.2] | +0.0% | 16 / 16 | 1 |
| semantic_maxsim_accuracy | 100 | 55.0% [45.2, 64.4] | 58.0% [48.2, 67.2] | +3.0 [-8.0, 14.0] | +5.5% | 13 / 16 | 0.7111 |

b = correct only without RAG, c = correct only with RAG. Only the strict metric is the pre-specified primary outcome; the other rows are sensitivity analyses.

## Retrieval

- Source article retrieved in top-k: 77/97 (79.4%)
- Mean top-1 cosine similarity: 0.570
- Context relevance (judge): {'RELEVANT': 58, 'PARTIALLY_RELEVANT': 22, 'IRRELEVANT': 20}

See `retrieval_vs_correctness.csv` for the cross-tabulation with answer correctness.
