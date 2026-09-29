"""Step 7 — compute metrics + statistics and write all tables and figures.

    python -m src.analysis.generate_results [--config config/config.yaml]

Measured quantities (all computed from results/processed/*.jsonl):
  PRIMARY   strict factual accuracy = #CORRECT / #paired questions x 100
            absolute difference (percentage points), relative change (%),
            exact McNemar test, 95% Wilson CIs, paired-bootstrap CI of the difference
  SENSITIVITY  lenient accuracy (CORRECT+PARTIAL), decisive-only accuracy,
            judge-independent semantic max-similarity accuracy
  RETRIEVAL source-article hit rate, context-relevance labels, and their
            cross-tabulation with RAG answer correctness

Outputs: results/tables/*.csv, summary.json, summary.md; results/figures/*.png|pdf
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import plots
from src.analysis.stats import bootstrap_diff_ci, mcnemar_exact, paired_table, wilson_ci
from src.config import base_arg_parser, load_config, resolve_path
from src.evaluation.judge import LABEL_ORDER_FOR_TABLES
from src.experiments.runner import CONDITIONS, latest_records, results_path
from src.utils import get_logger, read_jsonl, utc_now, write_json

log = get_logger(__name__)


def _join(v: Any) -> Any:
    if isinstance(v, list):
        return " || ".join(str(x) for x in v)
    return v


def _flat(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.map(_join)) if len(df) else df


def compare(a: list[int], b: list[int], seed: int) -> dict[str, Any]:
    """Accuracy of both conditions + paired statistics for binary outcomes a, b."""
    n = len(a)
    out: dict[str, Any] = {"n": n}
    for name, vals in (("no_rag", a), ("rag", b)):
        k = int(sum(vals))
        lo, hi = wilson_ci(k, n)
        out[name] = {"n": n, "n_correct": k, "accuracy": k / n if n else float("nan"),
                     "ci_low": lo, "ci_high": hi}
    acc_a, acc_b = out["no_rag"]["accuracy"], out["rag"]["accuracy"]
    out["abs_diff_pp"] = (acc_b - acc_a) * 100
    out["rel_change_pct"] = ((acc_b - acc_a) / acc_a * 100) if acc_a else None
    table = paired_table(a, b)
    out["paired_table"] = table
    out["mcnemar"] = mcnemar_exact(table["no_rag_only_correct"], table["rag_only_correct"])
    lo, hi = bootstrap_diff_ci(a, b, seed=seed)
    out["diff_ci_pp"] = (lo * 100, hi * 100)
    return out


def generate(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = cfg["experiment"]["random_seed"]
    proc = resolve_path(cfg["paths"]["results_processed"])
    tables = resolve_path(cfg["paths"]["results_tables"])
    figs = resolve_path(cfg["paths"]["results_figures"])
    tables.mkdir(parents=True, exist_ok=True)

    eval_path = proc / "evaluation.jsonl"
    if not eval_path.is_file():
        raise SystemExit(f"{eval_path} not found — run `python -m src.evaluation.evaluate` first.")
    ev = pd.DataFrame(read_jsonl(eval_path))
    ret = pd.DataFrame(read_jsonl(proc / "retrieval_evaluation.jsonl")) \
        if (proc / "retrieval_evaluation.jsonl").is_file() else pd.DataFrame()
    meta = json.loads((proc / "evaluation_meta.json").read_text())
    raw = {c: latest_records(results_path(cfg, c)) for c in CONDITIONS}
    mock = ("mock" in meta.get("generation_providers", [])
            or meta.get("judge_settings", {}).get("provider") == "mock")
    if mock:
        log.warning("Results were produced with the MOCK provider — they are NOT real results.")

    # ------------------------------------------------------------------ tables
    for c in CONDITIONS:
        df = pd.DataFrame(list(raw[c].values()))
        if c == "rag" and "retrieved" in df:
            df["retrieved_doc_ids"] = df["retrieved"].map(lambda r: [x["doc_id"] for x in r] if isinstance(r, list) else None)
            df["retrieved_scores"] = df["retrieved"].map(lambda r: [x["score"] for x in r] if isinstance(r, list) else None)
            df = df.drop(columns=["retrieved"])
        cols = [col for col in ["question_id", "question", "answer", "status", "error", "model_requested",
                                "model_returned", "temperature", "max_tokens", "seed", "finish_reason",
                                "retrieved_doc_ids", "retrieved_scores", "user_prompt", "system_prompt",
                                "system_fingerprint", "timestamp", "run_id"] if col in df]
        _flat(df[cols]).to_csv(tables / f"{c}_results.csv", index=False)

    drop = [col for col in ("correct_answers", "incorrect_answers") if col in ev]
    _flat(ev.drop(columns=drop)).to_csv(tables / "evaluation_results.csv", index=False)

    # per-chunk retrieval table
    chunk_rows = []
    for _, r in ret.iterrows():
        rel = set(r.get("context_relevant_ranks") or []) if isinstance(r.get("context_relevant_ranks"), list) else set()
        for c in r["retrieved"]:
            chunk_rows.append({
                "question_id": r["question_id"], "question": r["question"], "rank": c["rank"],
                "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": c["title"],
                "section": c["section"], "score": c["score"],
                "is_source_article": (c["doc_id"] == r["gold_doc_id"]) if r["gold_doc_id"] else None,
                "judged_answer_relevant": (c["rank"] in rel) if "context_label" in r else None,
                "text": c["text"],
            })
    pd.DataFrame(chunk_rows).to_csv(tables / "retrieval_results.csv", index=False)

    # combined wide table (one row per question)
    pv = ev.pivot(index="question_id", columns="condition")
    comb = pd.DataFrame({
        "question_id": pv.index,
        "question": pv[("question", "no_rag")],
        "category": pv[("category", "no_rag")],
        "reference_answer": pv[("best_answer", "no_rag")],
        "no_rag_answer": pv[("answer", "no_rag")],
        "rag_answer": pv[("answer", "rag")],
        "no_rag_label": pv[("judge_label", "no_rag")],
        "rag_label": pv[("judge_label", "rag")],
        "no_rag_correct": pv[("strict_correct", "no_rag")],
        "rag_correct": pv[("strict_correct", "rag")],
        "in_paired_set": pv[("in_paired_set", "no_rag")],
    }).reset_index(drop=True)
    comb["outcome_change"] = np.select(
        [(comb.no_rag_correct == 1) & (comb.rag_correct == 1), (comb.no_rag_correct == 0) & (comb.rag_correct == 1),
         (comb.no_rag_correct == 1) & (comb.rag_correct == 0)],
        ["both_correct", "improved_by_rag", "worsened_by_rag"], default="both_not_correct")
    if len(ret):
        r_small = ret.set_index("question_id")
        comb["source_article_retrieved"] = comb.question_id.map(r_small["source_hit"])
        comb["context_label"] = comb.question_id.map(r_small["context_label"]) if "context_label" in r_small else None
        comb["top1_retrieval_score"] = comb.question_id.map(r_small["top1_score"])
        comb["retrieved_context"] = comb.question_id.map(
            r_small["retrieved"].map(lambda cs: "\n\n".join(f"[{c['rank']}] {c['title']} ({c['score']:.3f}): {c['text']}" for c in cs)))
    comb.to_csv(tables / "combined_results.csv", index=False)

    # ------------------------------------------------------------- statistics
    paired = ev[ev.in_paired_set]
    by = {c: paired[paired.condition == c].sort_values("question_id") for c in CONDITIONS}
    if len(by["no_rag"]) == 0:
        raise SystemExit("No paired questions to analyse.")
    primary = compare(by["no_rag"].strict_correct.tolist(), by["rag"].strict_correct.tolist(), seed)
    lenient = compare(by["no_rag"].lenient_correct.tolist(), by["rag"].lenient_correct.tolist(), seed)
    both_decisive = by["no_rag"].decisive.to_numpy() & by["rag"].decisive.to_numpy()
    decisive = compare(by["no_rag"].strict_correct[both_decisive].tolist(),
                       by["rag"].strict_correct[both_decisive].tolist(), seed)
    semantic = None
    if "semantic_correct" in paired:
        semantic = compare(by["no_rag"].semantic_correct.astype(int).tolist(),
                           by["rag"].semantic_correct.astype(int).tolist(), seed)

    label_counts = (paired.groupby(["judge_label", "condition"]).size().unstack(fill_value=0)
                    .reindex(columns=list(CONDITIONS), fill_value=0))
    label_counts = label_counts.reindex([l for l in LABEL_ORDER_FOR_TABLES if l in label_counts.index])
    label_counts.to_csv(tables / "label_distribution.csv")

    conf = paired.groupby(["condition", "judge_confidence"]).size().unstack(fill_value=0) \
        if "judge_confidence" in paired else pd.DataFrame()
    conf.to_csv(tables / "judge_confidence.csv")

    cat = (paired.groupby(["category", "condition"]).strict_correct.agg(["sum", "count"]).unstack())
    cat_tbl = pd.DataFrame({
        "n": cat[("count", "no_rag")],
        "no_rag_correct": cat[("sum", "no_rag")], "rag_correct": cat[("sum", "rag")],
    })
    cat_tbl["no_rag_accuracy"] = cat_tbl.no_rag_correct / cat_tbl.n
    cat_tbl["rag_accuracy"] = cat_tbl.rag_correct / cat_tbl.n
    cat_tbl.sort_values("n", ascending=False).to_csv(tables / "category_accuracy.csv")

    pt = primary["paired_table"]
    pd.DataFrame([[pt["both_correct"], pt["no_rag_only_correct"]], [pt["rag_only_correct"], pt["both_incorrect"]]],
                 index=["no_rag_correct", "no_rag_not_correct"],
                 columns=["rag_correct", "rag_not_correct"]).to_csv(tables / "paired_contingency_table.csv")

    # ------------------------------------------------------ retrieval analysis
    retrieval_summary: dict[str, Any] = {}
    crosstabs: dict[str, pd.DataFrame] = {}
    if len(ret):
        rp = ret[ret.question_id.isin(by["rag"].question_id)].merge(
            by["rag"][["question_id", "strict_correct"]], on="question_id")
        by_nr = by["no_rag"].set_index("question_id").strict_correct
        rp["no_rag_correct"] = rp.question_id.map(by_nr)
        hits = rp.source_hit.dropna()
        retrieval_summary = {
            "n": int(len(rp)),
            "source_hit_rate": float(hits.astype(bool).mean()) if len(hits) else None,
            "n_source_hit": int(hits.astype(bool).sum()),
            "n_source_known": int(len(hits)),
            "mean_best_rank_of_source_article_when_hit": (
                float(rp.gold_best_rank.dropna().mean()) if rp.gold_best_rank.notna().any() else None),
            "mean_top1_score": float(rp.top1_score.mean()),
            "mean_topk_score": float(rp.mean_score.mean()),
        }
        rp["source_article_retrieved"] = rp.source_hit.map({True: "retrieved", False: "not_retrieved"})
        specs = [("source_article_retrieved", ["retrieved", "not_retrieved"])]
        if "context_label" in rp:
            retrieval_summary["context_label_counts"] = rp.context_label.value_counts().to_dict()
            specs.append(("context_label", ["RELEVANT", "PARTIALLY_RELEVANT", "IRRELEVANT"]))
        rows = []
        for col, order in specs:
            sub = rp.dropna(subset=[col])
            ct = (sub.groupby(col).strict_correct.agg(correct="sum", n="count")
                  .reindex([o for o in order if o in set(sub[col])] + sorted(set(sub[col]) - set(order))))
            ct["not_correct"] = ct.n - ct.correct
            crosstabs[col] = ct
            for lvl, r in ct.iterrows():
                s2 = sub[sub[col] == lvl]
                rows.append({
                    "relevance_measure": col, "relevance_level": lvl, "n": int(r.n),
                    "rag_correct": int(r.correct), "rag_not_correct": int(r.not_correct),
                    "rag_accuracy": r.correct / r.n if r.n else None,
                    "no_rag_accuracy_same_questions": float(s2.no_rag_correct.mean()) if len(s2) else None,
                    "improved_by_rag": int(((s2.no_rag_correct == 0) & (s2.strict_correct == 1)).sum()),
                    "worsened_by_rag": int(((s2.no_rag_correct == 1) & (s2.strict_correct == 0)).sum()),
                })
        pd.DataFrame(rows).to_csv(tables / "retrieval_vs_correctness.csv", index=False)

    # ---------------------------------------------------------------- summary
    summary = {
        "generated_at": utc_now(),
        "MOCK_RESULTS_NOT_REAL": mock,
        "n_questions": int(meta["n_questions"]), "n_paired": int(primary["n"]),
        "excluded_question_ids": meta["excluded_question_ids"],
        "generation_models": meta.get("generation_models"),
        "judge_model": meta["judge_settings"]["model"],
        "primary_strict": primary, "sensitivity_lenient": lenient,
        "sensitivity_decisive_only": decisive, "secondary_semantic": semantic,
        "retrieval": retrieval_summary,
        "config": {k: cfg[k] for k in ("experiment", "dataset", "retrieval", "llm", "evaluation")},
    }
    write_json(tables / "summary.json", summary)

    srows = []
    for name, res, note in [
        ("strict_accuracy (PRIMARY)", primary, "CORRECT only"),
        ("lenient_accuracy", lenient, "CORRECT + PARTIAL"),
        ("decisive_only_accuracy", decisive, "excl. UNCERTAIN/PARSE_ERROR/JUDGE_ERROR"),
        ("semantic_maxsim_accuracy", semantic, "judge-independent, embedding-based"),
    ]:
        if res is None:
            continue
        srows.append({
            "metric": name, "n": res["n"],
            "no_rag_correct": res["no_rag"]["n_correct"], "rag_correct": res["rag"]["n_correct"],
            "no_rag_accuracy_pct": round(res["no_rag"]["accuracy"] * 100, 2),
            "no_rag_ci95_pct": f"[{res['no_rag']['ci_low']*100:.1f}, {res['no_rag']['ci_high']*100:.1f}]",
            "rag_accuracy_pct": round(res["rag"]["accuracy"] * 100, 2),
            "rag_ci95_pct": f"[{res['rag']['ci_low']*100:.1f}, {res['rag']['ci_high']*100:.1f}]",
            "abs_diff_pp": round(res["abs_diff_pp"], 2),
            "diff_ci95_pp_bootstrap": f"[{res['diff_ci_pp'][0]:.1f}, {res['diff_ci_pp'][1]:.1f}]",
            "rel_change_pct": None if res["rel_change_pct"] is None else round(res["rel_change_pct"], 2),
            "mcnemar_b_norag_only": res["mcnemar"]["statistic_b"],
            "mcnemar_c_rag_only": res["mcnemar"]["statistic_c"],
            "mcnemar_exact_p": round(res["mcnemar"]["p_value"], 5),
            "note": note + ("  [MOCK – NOT REAL]" if mock else ""),
        })
    pd.DataFrame(srows).to_csv(tables / "summary.csv", index=False)
    _write_markdown(tables / "summary.md", summary, srows, retrieval_summary, mock)

    # ---------------------------------------------------------------- figures
    fig_summary = {**{c: primary[c] for c in CONDITIONS}, "mcnemar": primary["mcnemar"], "n_paired": primary["n"]}
    plots.plot_accuracy(fig_summary, figs, mock)
    plots.plot_correct_incorrect(fig_summary, figs, mock)
    plots.plot_label_distribution(label_counts, figs, mock)
    plots.plot_paired_outcomes(pt, figs, mock)
    if "context_label" in crosstabs:
        plots.plot_relevance_vs_correctness(
            crosstabs["context_label"], figs, "fig4_context_relevance_vs_correctness",
            "Retrieved context (judge verdict)", "RAG correctness by context relevance", mock)
    if "source_article_retrieved" in crosstabs:
        plots.plot_relevance_vs_correctness(
            crosstabs["source_article_retrieved"], figs, "fig5_source_hit_vs_correctness",
            "TruthfulQA source article among top-k?", "RAG correctness by source-article retrieval", mock)

    log.info("PRIMARY strict accuracy: No-RAG %.1f%%  RAG %.1f%%  (diff %+.1f pp, McNemar p=%.4g, n=%d)%s",
             primary["no_rag"]["accuracy"] * 100, primary["rag"]["accuracy"] * 100,
             primary["abs_diff_pp"], primary["mcnemar"]["p_value"], primary["n"],
             "  [MOCK – NOT REAL]" if mock else "")
    log.info("Tables -> %s ; figures -> %s", tables, figs)
    return summary


def main() -> None:
    args = base_arg_parser(__doc__).parse_args()
    generate(load_config(args.config))


def _write_markdown(path, summary, srows, retr, mock) -> None:
    lines = ["# Results summary", ""]
    if mock:
        lines += ["> **WARNING: produced with the MOCK provider – these numbers are NOT real results.**", ""]
    lines += [f"- Generated: {summary['generated_at']}",
              f"- Answer model(s): {', '.join(summary['generation_models'] or [])}",
              f"- Judge model: {summary['judge_model']}",
              f"- Questions: {summary['n_questions']} sampled, {summary['n_paired']} in the paired analysis", "",
              "## Factual accuracy", "",
              "| Metric | n | No-RAG | RAG | Diff (pp) [95% CI] | Rel. change | McNemar b / c | exact p |",
              "|---|---|---|---|---|---|---|---|"]
    for r in srows:
        rel = "n/a" if r["rel_change_pct"] is None else f"{r['rel_change_pct']:+.1f}%"
        lines.append(f"| {r['metric']} | {r['n']} | {r['no_rag_accuracy_pct']:.1f}% {r['no_rag_ci95_pct']} | "
                     f"{r['rag_accuracy_pct']:.1f}% {r['rag_ci95_pct']} | {r['abs_diff_pp']:+.1f} "
                     f"{r['diff_ci95_pp_bootstrap']} | {rel} | {r['mcnemar_b_norag_only']} / "
                     f"{r['mcnemar_c_rag_only']} | {r['mcnemar_exact_p']:.4g} |")
    lines += ["", "b = correct only without RAG, c = correct only with RAG. Only the strict metric is the "
              "pre-specified primary outcome; the other rows are sensitivity analyses.", ""]
    if retr:
        lines += ["## Retrieval", "",
                  f"- Source article retrieved in top-k: {retr['n_source_hit']}/{retr['n_source_known']}"
                  + (f" ({retr['source_hit_rate']*100:.1f}%)" if retr['source_hit_rate'] is not None else ""),
                  f"- Mean top-1 cosine similarity: {retr['mean_top1_score']:.3f}"]
        if "context_label_counts" in retr:
            lines.append(f"- Context relevance (judge): {retr['context_label_counts']}")
        lines += ["", "See `retrieval_vs_correctness.csv` for the cross-tabulation with answer correctness."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
