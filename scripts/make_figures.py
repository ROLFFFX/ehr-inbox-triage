"""Figures for the report (static PDF, print-safe, three highlighted series)."""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from triage.data import ROOT

TAG = sys.argv[1] if len(sys.argv) > 1 else "primary"
RES = ROOT / "results" / TAG
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Reference categorical slots 1-3 (validated all-pairs), muted gray for context.
HIGHLIGHT = {"tfidf_lr": "#2a78d6", "tfidf_ordinal": "#eb6834", "rules_tiered": "#1baf7a"}
MUTED, INK, INK2, GRID = "#b4b2aa", "#0b0b0b", "#52514e", "#e6e4dc"
LABEL = {
    "kw5110": "5110 keyword list",
    "rules_any": "Rules (any tier-1)",
    "rules_tiered": "Rules (tiered)",
    "rules_lr": "Rules + learned weights",
    "tfidf_lr": "TF-IDF LR",
    "tfidf_ordinal": "TF-IDF ordinal",
    "tfidf+rules_lr": "TF-IDF + rules LR",
    "tfidf_lr+bypass": "TF-IDF LR + keyword bypass",
    "cascade_filter_lr": "Keyword filter → LR cascade",
    "bioclinicalbert_binary": "Bio_ClinicalBERT, binary",
    "bioclinicalbert_ordinal": "Bio_ClinicalBERT, ordinal",
    "roberta_binary": "RoBERTa-base, binary",
    "roberta_ordinal": "RoBERTa-base, ordinal",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)

cls = pd.read_csv(RES / "classification.csv")
methods = list(cls["method"])


def frontier():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axvspan(0, 10, ymin=0.95, ymax=1.0, color="#1baf7a", alpha=0.18, lw=0)
    ax.axhline(0.95, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.axvline(10, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(11, 0.905, "target: recall ≥ 0.95\nat < 10 FA / 100", color=INK2, fontsize=7.5, va="top")
    order = [m for m in methods if m not in HIGHLIGHT] + [m for m in HIGHLIGHT if m in methods]
    for m in order:
        d = pd.read_parquet(RES / "scores" / f"{m}__test.parquet")
        fpr, tpr, _ = roc_curve(d["urgent"], d["score"])
        hl = m in HIGHLIGHT
        ax.step(100 * fpr, tpr, where="post", lw=2 if hl else 1, color=HIGHLIGHT.get(m, MUTED), zorder=3 if hl else 2,
                label=LABEL.get(m, m) if hl else None)
    ax.plot([], [], color=MUTED, lw=1, label="other methods")
    ax.set(xlim=(0, 100), ylim=(0, 1.01), xlabel="False alarms per 100 non-urgent messages",
           ylabel="Recall on urgent (Level 1) messages")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"frontier_{TAG}.pdf")
    plt.close(fig)


def ranking():
    rnk = pd.read_csv(RES / "ranking.csv").set_index("method").loc[methods]
    panels = [
        ("pair_acc_all", "(a) Pairwise accuracy", (0.4, 1.0), 0.5),
        ("sim_t_ndcg@10", "(b) T-NDCG@10 (± SD)", (0.0, 0.7), None),
        ("sim_urgent_worst_pos", "(c) Last urgent position ↓", (0.0, 1.0), None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, (col, title, lim, ref) in zip(axes, panels):
        vals = rnk[col].values
        colors = [HIGHLIGHT.get(m, MUTED) for m in methods]
        ax.hlines(y, lim[0], vals, color=GRID, lw=1)
        ax.scatter(vals, y, s=28, color=colors, zorder=3, edgecolor="white", linewidth=1)
        sd = f"{col}_sd" if f"{col}_sd" in rnk else None
        if sd:
            ax.errorbar(vals, y, xerr=rnk[sd].values, fmt="none", ecolor=INK2, elinewidth=0.6, capsize=0, zorder=2)
        if ref is not None:
            ax.axvline(ref, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.set_xlim(*lim)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6)
    axes[0].set_yticks(y, [LABEL.get(m, m) for m in methods], fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"ranking_{TAG}.pdf")
    plt.close(fig)


frontier()
ranking()
print("figures ->", FIG)
