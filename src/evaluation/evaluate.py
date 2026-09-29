"""Step 6 — evaluate factual correctness of both conditions + retrieval quality.

    python -m src.evaluation.evaluate [--config config/config.yaml]

1. Answer correctness (same blinded LLM judge, same prompt, for both conditions)
2. Semantic max-similarity check (judge-independent, secondary)
3. Retrieval quality for every RAG question:
     - source_hit : was the question's own TruthfulQA source article retrieved?
     - context_label : judge verdict whether the retrieved text contains the answer

Raw judge outputs are cached in results/raw/judge_outputs.jsonl (keyed by the
exact judge input), so re-running never re-bills finished judgements and the
evaluator's verbatim output is preserved.

Outputs
  results/processed/evaluation.jsonl            one row per (condition, question)
  results/processed/retrieval_evaluation.jsonl  one row per RAG question
  results/processed/evaluation_meta.json        judge settings, counts
"""
from __future__ import annotations

import hashlib
import sys
from typing import Any

import pandas as pd

from src.config import base_arg_parser, fingerprint, load_config, resolve_path
from src.evaluation.judge import (ANSWER_LABELS, CONTEXT_LABELS, JUDGE_ERROR, is_decisive,
                                  lenient_correct, parse_judge_output, strict_correct)
from src.experiments.runner import CONDITIONS, latest_records, load_questions, results_path
from src.llm.client import LLMError, FatalLLMError, make_client
from src.prompts import fill, format_answer_list, format_context, load_template
from src.utils import append_jsonl, get_logger, read_jsonl, utc_now, write_json, write_jsonl

log = get_logger(__name__)
JUDGE_SYSTEM = "You are a meticulous, impartial evaluator. You always reply with valid JSON."


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class CachedJudge:
    """Calls the judge model once per unique input and stores the raw output."""

    def __init__(self, cfg: dict[str, Any], client=None):
        ev = cfg["evaluation"]
        self.settings = {k: ev.get(k) for k in ("provider", "model", "base_url", "temperature", "max_tokens")}
        self.settings["seed"] = cfg["experiment"]["random_seed"]
        self.templates = {"answer": load_template(cfg, "judge_answer"),
                          "context": load_template(cfg, "judge_context")}
        self.judge_id = fingerprint({"settings": self.settings, "templates": self.templates,
                                     "system": JUDGE_SYSTEM})
        self.cache_path = resolve_path(cfg["paths"]["results_raw"]) / "judge_outputs.jsonl"
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path.is_file():
            for r in read_jsonl(self.cache_path):
                self.cache[r["cache_key"]] = r
        self._client = client
        self._cfg = cfg
        self.n_calls = 0

    @property
    def client(self):
        if self._client is None:
            self._client = make_client(self._cfg["evaluation"], seed=self.settings["seed"])
        return self._client

    def judge(self, kind: str, condition: str, question_id: str, prompt: str) -> dict[str, Any]:
        allowed = ANSWER_LABELS if kind == "answer" else CONTEXT_LABELS
        key = f"{self.judge_id}|{kind}|{condition}|{question_id}|{_hash(prompt)}"
        if key in self.cache:
            raw = self.cache[key]
        else:
            try:
                resp = self.client.complete(JUDGE_SYSTEM, prompt, json_mode=True)
            except FatalLLMError:
                raise
            except LLMError as e:
                log.error("Judge call failed for %s/%s/%s: %s", kind, condition, question_id, e)
                return {"label": JUDGE_ERROR, "raw_output": None, "parse_ok": False,
                        "parse_error": str(e), "confidence": None, "explanation": None,
                        "relevant_ranks": None, "judge_model_returned": None}
            raw = {"cache_key": key, "judge_id": self.judge_id, "kind": kind,
                   "condition": condition, "question_id": question_id,
                   "judge_prompt": prompt, "raw_output": resp.text,
                   "judge_model_returned": resp.model, "timestamp": utc_now()}
            append_jsonl(self.cache_path, raw)
            self.cache[key] = raw
            self.n_calls += 1
        parsed = parse_judge_output(raw["raw_output"], allowed)
        return {**parsed, "raw_output": raw["raw_output"],
                "judge_model_returned": raw.get("judge_model_returned")}


def gold_doc_ids(cfg: dict[str, Any]) -> dict[str, str]:
    """Map Wikipedia source title (as cited by TruthfulQA) -> corpus doc_id."""
    path = resolve_path(cfg["corpus"].get("manifest_path", "data/corpus/manifest.csv"))
    if not path.is_file():
        log.warning("%s not found — source_hit cannot be computed.", path)
        return {}
    m = pd.read_csv(path)
    m = m[m["status"] == "ok"]
    return dict(zip(m["requested_title"], m["doc_id"]))


def evaluate(cfg: dict[str, Any], judge_client=None, semantic_embedder=None,
             allow_incomplete: bool = False) -> dict[str, Any]:
    questions = {q["question_id"]: q for q in load_questions(cfg)}
    records = {c: latest_records(results_path(cfg, c)) for c in CONDITIONS}
    for c in CONDITIONS:
        if not records[c]:
            raise FileNotFoundError(f"No results for condition '{c}' — run src.experiments.run_{c} first.")
    # Refuse to spend judge calls on a half-finished experiment.
    n_ok = {c: sum(1 for q in questions if records[c].get(q, {}).get("status") == "ok") for c in CONDITIONS}
    if not allow_incomplete and any(n < len(questions) for n in n_ok.values()):
        raise ValueError(
            "The experiment is not complete yet: "
            + ", ".join(f"{c} has {n_ok[c]}/{len(questions)} answers" for c in CONDITIONS)
            + ". Run `python -m src.experiments.run_no_rag` and `python -m src.experiments.run_rag` "
            "(without --limit) until both report all questions done, then evaluate again. "
            "(Use --allow-incomplete to evaluate a partial run anyway.)"
        )

    ok = {c: {qid for qid, r in records[c].items() if r.get("status") == "ok"} for c in CONDITIONS}
    paired = sorted(set(questions) & ok["no_rag"] & ok["rag"])
    excluded = sorted(set(questions) - set(paired))
    if excluded:
        log.warning("%d questions lack a successful answer in both conditions and are excluded "
                    "from the paired comparison: %s", len(excluded), excluded)

    judge = CachedJudge(cfg, client=judge_client)
    ev = cfg["evaluation"]
    rows: list[dict[str, Any]] = []
    for c in CONDITIONS:
        for qid, q in questions.items():
            rec = records[c].get(qid)
            row = {
                "condition": c, "question_id": qid, "question": q["question"],
                "category": q["category"], "type": q["type"], "best_answer": q["best_answer"],
                "correct_answers": q["correct_answers"], "incorrect_answers": q["incorrect_answers"],
                "answer": rec.get("answer") if rec else None,
                "generation_status": rec.get("status") if rec else "missing",
                "in_paired_set": qid in paired,
            }
            if row["generation_status"] == "ok":
                prompt = fill(judge.templates["answer"], question=q["question"],
                              best_answer=q["best_answer"],
                              correct_answers=format_answer_list(q["correct_answers"]),
                              incorrect_answers=format_answer_list(q["incorrect_answers"]),
                              answer=row["answer"])
                j = judge.judge("answer", c, qid, prompt)
                row.update(judge_label=j["label"], judge_confidence=j["confidence"],
                           judge_explanation=j["explanation"], judge_raw_output=j["raw_output"],
                           judge_parse_ok=j["parse_ok"], judge_parse_error=j["parse_error"],
                           judge_model_returned=j["judge_model_returned"])
            else:
                row.update(judge_label="NO_OUTPUT", judge_parse_ok=False)
            row["strict_correct"] = strict_correct(row["judge_label"])
            row["lenient_correct"] = lenient_correct(row["judge_label"])
            row["decisive"] = is_decisive(row["judge_label"]) and row["judge_label"] != "NO_OUTPUT"
            rows.append(row)
        log.info("[%s] judged %d answers", c, len(questions))

    # ---- secondary semantic metric ------------------------------------------
    if ev.get("semantic_check", True):
        if semantic_embedder is None:
            from src.retrieval.embedder import load_embedder
            semantic_embedder = load_embedder(ev.get("semantic_model", cfg["retrieval"]["embedding_model"]))
        from src.evaluation.semantic import semantic_scores
        with_answer = [r for r in rows if r["generation_status"] == "ok"]
        for r, s in zip(with_answer, semantic_scores(semantic_embedder, with_answer)):
            r.update(s)

    # ---- retrieval evaluation -------------------------------------------------
    gold = gold_doc_ids(cfg)
    retrieval_rows = []
    for qid, q in questions.items():
        rec = records["rag"].get(qid)
        if not rec or not rec.get("retrieved"):
            continue
        retrieved = rec["retrieved"]
        gold_id = gold.get(q.get("source_wikipedia_title"))
        gold_ranks = [c["rank"] for c in retrieved if gold_id and c["doc_id"] == gold_id]
        rr = {
            "question_id": qid, "question": q["question"],
            "source_wikipedia_title": q.get("source_wikipedia_title"), "gold_doc_id": gold_id,
            "source_hit": (bool(gold_ranks) if gold_id else None),
            "gold_best_rank": (min(gold_ranks) if gold_ranks else None),
            "top1_score": retrieved[0]["score"],
            "mean_score": round(sum(c["score"] for c in retrieved) / len(retrieved), 6),
            "retrieved_doc_ids": [c["doc_id"] for c in retrieved],
            "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved],
            "retrieved_scores": [c["score"] for c in retrieved],
            "retrieved": retrieved,
        }
        if ev.get("judge_context_relevance", True):
            prompt = fill(judge.templates["context"], question=q["question"],
                          best_answer=q["best_answer"],
                          correct_answers=format_answer_list(q["correct_answers"]),
                          context=format_context(retrieved))
            j = judge.judge("context", "rag", qid, prompt)
            rr.update(context_label=j["label"], context_relevant_ranks=j["relevant_ranks"],
                      context_explanation=j["explanation"], context_raw_output=j["raw_output"],
                      context_parse_ok=j["parse_ok"])
        retrieval_rows.append(rr)

    out_dir = resolve_path(cfg["paths"]["results_processed"])
    write_jsonl(out_dir / "evaluation.jsonl", rows)
    write_jsonl(out_dir / "retrieval_evaluation.jsonl", retrieval_rows)
    n_judge_errors = sum(r["judge_label"] == JUDGE_ERROR for r in rows) + sum(
        r.get("context_label") == JUDGE_ERROR for r in retrieval_rows)
    meta = {
        "evaluated_at": utc_now(), "judge_id": judge.judge_id, "judge_settings": judge.settings,
        "judge_system_prompt": JUDGE_SYSTEM, "new_judge_calls": judge.n_calls,
        "n_questions": len(questions), "n_paired": len(paired), "excluded_question_ids": excluded,
        "n_judge_errors": n_judge_errors,
        "generation_providers": sorted({r.get("provider", "?") for c in CONDITIONS for r in records[c].values()}),
        "generation_models": sorted({r.get("model_requested", "?") for c in CONDITIONS for r in records[c].values()}),
        "semantic_model": ev.get("semantic_model") if ev.get("semantic_check", True) else None,
    }
    write_json(out_dir / "evaluation_meta.json", meta)
    log.info("Evaluation written to %s (%d new judge calls, %d judge errors)",
             out_dir, judge.n_calls, n_judge_errors)
    return meta


def main() -> None:
    parser = base_arg_parser(__doc__)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Evaluate even if some questions have no answer in a condition")
    args = parser.parse_args()
    cfg = load_config(args.config)
    try:
        meta = evaluate(cfg, allow_incomplete=args.allow_incomplete)
    except (LLMError, FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        sys.exit(1)
    if meta["n_judge_errors"]:
        log.warning("Some judge calls failed; re-run this command to retry them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
