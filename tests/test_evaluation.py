import pytest

from src.analysis.stats import bootstrap_diff_ci, mcnemar_exact, paired_table, wilson_ci
from src.evaluation.judge import (ANSWER_LABELS, CONTEXT_LABELS, PARSE_ERROR, is_decisive,
                                  lenient_correct, parse_judge_output, strict_correct)
from src.evaluation.semantic import semantic_scores


# ------------------------------------------------------------ judge parsing
def test_parse_valid_json():
    r = parse_judge_output('{"label": "CORRECT", "confidence": "High", "explanation": "ok"}', ANSWER_LABELS)
    assert r["label"] == "CORRECT" and r["parse_ok"] and r["confidence"] == "high"


def test_parse_code_fence_and_lowercase():
    r = parse_judge_output('```json\n{"label": "no answer", "explanation": "refused"}\n```', ANSWER_LABELS)
    assert r["label"] == "NO_ANSWER"


def test_parse_unknown_label_is_parse_error():
    r = parse_judge_output('{"label": "MOSTLY_TRUE"}', ANSWER_LABELS)
    assert r["label"] == PARSE_ERROR and not r["parse_ok"]


def test_parse_non_json_single_label_recovered():
    r = parse_judge_output("The answer is INCORRECT because it repeats the myth.", ANSWER_LABELS)
    assert r["label"] == "INCORRECT" and r["parse_error"]


def test_parse_ambiguous_text_not_forced():
    r = parse_judge_output("It could be CORRECT or INCORRECT.", ANSWER_LABELS)
    assert r["label"] == PARSE_ERROR


def test_parse_empty():
    assert parse_judge_output("", ANSWER_LABELS)["label"] == PARSE_ERROR


def test_parse_context_labels_and_ranks():
    r = parse_judge_output('{"label": "partially relevant", "relevant_ranks": [2, "3"]}', CONTEXT_LABELS)
    assert r["label"] == "PARTIALLY_RELEVANT" and r["relevant_ranks"] == [2, 3]
    r2 = parse_judge_output("Verdict: PARTIALLY_RELEVANT", CONTEXT_LABELS)
    assert r2["label"] == "PARTIALLY_RELEVANT"  # not confused with RELEVANT


def test_label_to_binary():
    assert strict_correct("CORRECT") == 1 and strict_correct("PARTIAL") == 0
    assert lenient_correct("PARTIAL") == 1 and lenient_correct("INCORRECT") == 0
    assert not is_decisive("UNCERTAIN") and is_decisive("NO_ANSWER")


# ------------------------------------------------------------------ stats
def test_wilson_ci_known_value():
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)


def test_paired_table_and_mcnemar():
    a = [1, 1, 0, 0, 1, 0, 0, 0]
    b = [1, 0, 1, 1, 1, 1, 1, 0]
    t = paired_table(a, b)
    assert t == {"both_correct": 2, "no_rag_only_correct": 1, "rag_only_correct": 4, "both_incorrect": 1}
    m = mcnemar_exact(1, 4)
    assert m["p_value"] == pytest.approx(0.375, abs=1e-6)  # 2 * P(X<=1 | n=5, p=.5)
    assert mcnemar_exact(0, 0)["p_value"] == 1.0


def test_bootstrap_reproducible():
    a, b = [0, 1, 0, 1, 0], [1, 1, 1, 1, 0]
    assert bootstrap_diff_ci(a, b, seed=1) == bootstrap_diff_ci(a, b, seed=1)


def test_semantic_scores(fake_embedder):
    items = [{"answer": "Canberra is the capital", "correct_answers": ["Canberra is the capital city"],
              "incorrect_answers": ["Sydney is the capital city"]}]
    s = semantic_scores(fake_embedder, items)[0]
    assert s["semantic_correct"] == 1 and s["sim_correct"] > s["sim_incorrect"]
