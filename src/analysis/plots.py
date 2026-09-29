"""Poster-ready figures (matplotlib). All numbers come from the result tables.

Every figure is saved as 300-dpi PNG and as vector PDF (scales to any
poster size). Colours: No-RAG = blue, RAG = orange everywhere (colour-blind
safe pair); values are always printed as text so colour is never the only cue.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"no_rag": "#2a78d6", "rag": "#eb6834"}
NAMES = {"no_rag": "No-RAG", "rag": "RAG"}
LABEL_ORDER = ["CORRECT", "PARTIAL", "INCORRECT", "NO_ANSWER", "UNCERTAIN", "PARSE_ERROR", "JUDGE_ERROR"]
LABEL_COLORS = {"CORRECT": "#2a78d6", "PARTIAL": "#1baf7a", "INCORRECT": "#eb6834",
                "NO_ANSWER": "#eda100", "UNCERTAIN": "#e87ba4", "PARSE_ERROR": "#8a8984",
                "JUDGE_ERROR": "#52514e"}
SEQ_DARK, SEQ_LIGHT = "#1c5cab", "#86b6ef"   # blue ramp steps 550 / 250
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight",
})


def _finish(fig, ax, out_dir: Path, name: str, mock: bool) -> list[Path]:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if mock:
        fig.text(0.5, 0.5, "MOCK DATA – NOT REAL RESULTS", ha="center", va="center",
                 fontsize=26, color="red", alpha=0.35, rotation=20, weight="bold")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=300)
        paths.append(p)
    plt.close(fig)
    return paths


def plot_accuracy(summary: dict, out_dir: Path, mock: bool = False) -> list[Path]:
    """Fig 1 — strict factual accuracy per condition with 95% Wilson CIs."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    conds = ["no_rag", "rag"]
    acc = [summary[c]["accuracy"] * 100 for c in conds]
    lo = [summary[c]["ci_low"] * 100 for c in conds]
    hi = [summary[c]["ci_high"] * 100 for c in conds]
    err = np.clip(np.array([[a - l for a, l in zip(acc, lo)], [h - a for a, h in zip(acc, hi)]]), 0, None)
    x = np.arange(2)
    ax.bar(x, acc, width=0.55, color=[COLORS[c] for c in conds], edgecolor="white", linewidth=2)
    ax.errorbar(x, acc, yerr=err, fmt="none", ecolor=INK, elinewidth=1.5, capsize=8)
    for xi, a, h, c in zip(x, acc, hi, conds):
        ax.text(xi, h + 2, f"{a:.1f}%\n({summary[c]['n_correct']}/{summary[c]['n']})",
                ha="center", va="bottom", color=INK, fontsize=14, weight="bold")
    ax.set_xticks(x, [NAMES[c] for c in conds])
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Factual accuracy (%)")
    p = summary["mcnemar"]["p_value"]
    ax.set_title(f"Factual accuracy (n = {summary['n_paired']} paired questions)")
    ax.text(0.5, -0.14, f"Error bars: 95% Wilson CI · exact McNemar p = {p:.3g}",
            transform=ax.transAxes, ha="center", color=INK_2, fontsize=11)
    return _finish(fig, ax, out_dir, "fig1_accuracy_comparison", mock)


def plot_correct_incorrect(summary: dict, out_dir: Path, mock: bool = False) -> list[Path]:
    """Fig 2 — number of correct vs. not-correct answers per condition."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    groups = ["Correct", "Not correct"]
    x = np.arange(len(groups))
    w = 0.36
    for i, c in enumerate(["no_rag", "rag"]):
        vals = [summary[c]["n_correct"], summary[c]["n"] - summary[c]["n_correct"]]
        bars = ax.bar(x + (i - 0.5) * (w + 0.02), vals, width=w, color=COLORS[c],
                      label=NAMES[c], edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v), ha="center",
                    va="bottom", color=INK, fontsize=13)
    ax.set_xticks(x, groups)
    ax.set_ylabel("Number of questions")
    ax.set_ylim(0, summary["n_paired"] * 1.12 + 1)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Correct vs. not-correct answers (strict)")
    return _finish(fig, ax, out_dir, "fig2_correct_vs_incorrect", mock)


def plot_label_distribution(label_counts: pd.DataFrame, out_dir: Path, mock: bool = False) -> list[Path]:
    """Fig 2b — full judge-label distribution (shows partial / abstain / uncertain)."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    conds = ["no_rag", "rag"]
    labels = [lab for lab in LABEL_ORDER if lab in label_counts.index and label_counts.loc[lab].sum() > 0]
    left = np.zeros(len(conds))
    y = np.arange(len(conds))
    for lab in labels:
        vals = np.array([label_counts.loc[lab, c] if c in label_counts.columns else 0 for c in conds])
        ax.barh(y, vals, left=left, color=LABEL_COLORS[lab], label=lab.replace("_", " ").title(),
                edgecolor="white", linewidth=2, height=0.6)
        for yi, v, l0 in zip(y, vals, left):
            if v >= 3:
                txt_color = INK if lab in ("PARTIAL", "NO_ANSWER", "UNCERTAIN") else "white"
                ax.text(l0 + v / 2, yi, str(int(v)), ha="center", va="center", color=txt_color,
                        fontsize=12, weight="bold")
        left += vals
    ax.set_yticks(y, [NAMES[c] for c in conds])
    ax.invert_yaxis()
    ax.set_xlabel("Number of questions")
    ax.legend(frameon=False, ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("Judge label distribution")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    return _finish(fig, ax, out_dir, "fig2b_label_distribution", mock)


def plot_paired_outcomes(table: dict, out_dir: Path, mock: bool = False) -> list[Path]:
    """Fig 3 — 2x2 paired outcome matrix (the basis of McNemar's test)."""
    m = np.array([[table["both_correct"], table["no_rag_only_correct"]],
                  [table["rag_only_correct"], table["both_incorrect"]]])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=max(m.max(), 1) * 1.3)
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=22, weight="bold", color=INK)
    ax.set_xticks([0, 1], ["RAG correct", "RAG not correct"])
    ax.set_yticks([0, 1], ["No-RAG\ncorrect", "No-RAG\nnot correct"])
    # rows = No-RAG outcome, columns = RAG outcome
    ax.set_title("Paired outcomes per question")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.5, 0.0, "Off-diagonal cells = discordant pairs used by McNemar's test",
             ha="center", color=INK_2, fontsize=11)
    ax.grid(False)
    if mock:
        fig.text(0.5, 0.5, "MOCK DATA – NOT REAL RESULTS", ha="center", va="center",
                 fontsize=24, color="red", alpha=0.35, rotation=20, weight="bold")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"fig3_paired_outcomes.{ext}"
        fig.savefig(p, dpi=300)
        paths.append(p)
    plt.close(fig)
    return paths


def plot_relevance_vs_correctness(ct: pd.DataFrame, out_dir: Path, name: str, xlabel: str,
                                  title: str, mock: bool = False) -> list[Path]:
    """Fig 4 — RAG answer correctness split by retrieval relevance.

    `ct` index = relevance category, columns = ['correct', 'not_correct'].
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cats = list(ct.index)
    x = np.arange(len(cats))
    corr = ct["correct"].to_numpy()
    inc = ct["not_correct"].to_numpy()
    ax.bar(x, corr, width=0.55, color=SEQ_DARK, label="RAG answer correct",
           edgecolor="white", linewidth=2)
    ax.bar(x, inc, bottom=corr, width=0.55, color=SEQ_LIGHT, label="RAG answer not correct",
           edgecolor="white", linewidth=2)
    for xi, cv, iv in zip(x, corr, inc):
        tot = cv + iv
        if cv:
            ax.text(xi, cv / 2, str(int(cv)), ha="center", va="center", color="white", fontsize=13, weight="bold")
        if iv:
            ax.text(xi, cv + iv / 2, str(int(iv)), ha="center", va="center", color=INK, fontsize=13, weight="bold")
        pct = f"{cv / tot * 100:.0f}% correct" if tot else "n = 0"
        ax.text(xi, tot + 0.6, pct, ha="center", va="bottom", color=INK, fontsize=12)
    ax.set_xticks(x, [str(c).replace("_", " ").title() for c in cats])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of RAG questions")
    ax.set_ylim(0, max((corr + inc).max(), 1) * 1.18)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(title)
    return _finish(fig, ax, out_dir, name, mock)
