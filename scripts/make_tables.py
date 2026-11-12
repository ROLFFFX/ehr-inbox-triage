"""Write LaTeX tables for the report from results/<tag>/*.csv."""

from __future__ import annotations

import pandas as pd

from triage.data import REPORT, ROOT

OUT = REPORT / "tables"
OUT.mkdir(parents=True, exist_ok=True)

LABEL = {
    "kw5110": "5110 keyword list",
    "rules_any": "Rules, any tier-1 hit",
    "rules_tiered": "Rules, tiered score",
    "rules_lr": "Rules, learned weights",
    "tfidf_lr": "TF-IDF LR",
    "tfidf_ordinal": "TF-IDF ordinal (ridge)",
    "tfidf+rules_lr": "TF-IDF + rule features LR",
    "tfidf_lr+bypass": "TF-IDF LR + keyword bypass",
    "cascade_filter_lr": "Keyword filter $\\to$ LR cascade",
    "bioclinicalbert_binary": "Bio\\_ClinicalBERT, binary",
    "bioclinicalbert_ordinal": "Bio\\_ClinicalBERT, ordinal",
    "roberta_binary": "RoBERTa-base, binary",
    "roberta_ordinal": "RoBERTa-base, ordinal",
}
FAMILY = {
    "kw5110": "Rules", "rules_any": "Rules", "rules_tiered": "Rules", "rules_lr": "Rules",
    "tfidf_lr": "Classical ML", "tfidf_ordinal": "Classical ML", "tfidf+rules_lr": "Hybrid",
    "tfidf_lr+bypass": "Hybrid", "cascade_filter_lr": "Hybrid",
}


def ci(row, k, fmt="{:.2f}", scale=1.0):
    v, lo, hi = (row[k] * scale, row[f"{k}_lo"] * scale, row[f"{k}_hi"] * scale)
    return f"{fmt.format(v)} {{\\scriptsize[{fmt.format(lo)}, {fmt.format(hi)}]}}"


def nominal(row):
    if pd.isna(row.get("nominal_recall")):
        return "--"
    return f"{row['nominal_recall']:.2f} / {row['nominal_fa_per_100']:.1f}"


def classification(tag="primary"):
    c = pd.read_csv(ROOT / "results" / tag / "classification.csv")
    lines = []
    for _, r in c.iterrows():
        lines.append(
            " & ".join(
                [
                    LABEL.get(r.method, r.method),
                    f"{r.cal_fa_per_100:.1f}",
                    f"{r.recall:.2f}",
                    ci(r, "fa_per_100", "{:.1f}"),
                    nominal(r),
                    ci(r, "auroc"),
                    ci(r, "recall_at_fa10"),
                    ci(r, "fa100_at_recall95", "{:.0f}"),
                ]
            )
            + r" \\"
        )
    (OUT / f"classification_{tag}.tex").write_text("\n".join(lines) + "\n")


def ranking(tag="primary"):
    r = pd.read_csv(ROOT / "results" / tag / "ranking.csv")
    lines = []
    for _, x in r.iterrows():
        lines.append(
            " & ".join(
                [
                    LABEL.get(x.method, x.method),
                    f"{x.pair_acc_easy:.2f}",
                    f"{x.pair_acc_medium:.2f}",
                    f"{x.pair_acc_hard:.2f}",
                    f"{x.pair_acc_all:.2f}",
                    f"{x['sim_t_ndcg@10']:.2f}",
                    f"{x['sim_t_ndcg@30']:.2f}",
                    f"{x.sim_urgent_mean_pos:.2f}",
                    f"{x.sim_urgent_worst_pos:.2f}",
                    f"{x['synth_INBOX_A_t_ndcg@10']:.2f}",
                    f"{x['synth_INBOX_B_t_ndcg@10']:.2f}",
                ]
            )
            + r" \\"
        )
    (OUT / f"ranking_{tag}.tex").write_text("\n".join(lines) + "\n")


def transfer(tag="primary"):
    t = pd.read_csv(ROOT / "results" / tag / "transfer.csv")
    lines = [
        f"{LABEL.get(x.method, x.method)} & {x.cssrs_recall:.2f} & {x.cssrs_fa_per_100:.1f} & {x.cssrs_auroc:.2f} & "
        f"{x.cssrs_auprc:.2f} & {x.cssrs_recall_at_fa10:.2f} \\\\"
        for _, x in t.iterrows()
    ]
    (OUT / f"transfer_{tag}.tex").write_text("\n".join(lines) + "\n")


def precision_floor(tag="primary"):
    """Second operating point (PPV >= 2/3, after NYU) and the gate's workload at 2% prevalence."""
    c = pd.read_csv(ROOT / "results" / tag / "classification.csv")
    lines = []
    for _, r in c.iterrows():
        locked = "no threshold" if r.ppv_threshold == float("inf") else ci(r, "ppvop_recall")
        lines.append(
            f"{LABEL.get(r.method, r.method)} & {ci(r, 'recall_at_ppv67')} & {r.recall_at_ppv67_p2:.2f} & {locked} & "
            f"{r.reads_per_urgent_p2:.0f} \\\\"
        )
    (OUT / f"ppv_{tag}.tex").write_text("\n".join(lines) + "\n")


def sensitivity():
    a = pd.read_csv(ROOT / "results" / "primary" / "classification.csv").set_index("method")
    b = pd.read_csv(ROOT / "results" / "sens_level2" / "classification.csv").set_index("method")
    lines = [
        f"{LABEL.get(m, m)} & {a.loc[m, 'auroc']:.2f} & {b.loc[m, 'auroc']:.2f} & {a.loc[m, 'recall_at_fa10']:.2f} & "
        f"{b.loc[m, 'recall_at_fa10']:.2f} & {a.loc[m, 'fa100_at_recall95']:.0f} & {b.loc[m, 'fa100_at_recall95']:.0f} \\\\"
        for m in a.index
    ]
    (OUT / "sensitivity.tex").write_text("\n".join(lines) + "\n")


def rules():
    t = pd.read_csv(ROOT / "results" / "primary" / "rule_coverage.csv", index_col=0)
    t = t[t["n_hits"] > 0]
    lines = [
        f"{c.replace('_', ' ')} & {int(r.tier)} & {100 * r.urgent_hit_rate:.1f} & {100 * r.nonurgent_hit_rate:.1f} & "
        f"{r.lr_weight:+.2f} \\\\"
        for c, r in t.iterrows()
    ]
    (OUT / "rules.tex").write_text("\n".join(lines) + "\n")


def corpora():
    d = pd.read_csv(ROOT / "results" / "primary" / "corpora.csv")
    role = {
        ("pmr_reddit", "train"): "fit + calibrate (5-fold CV)",
        ("pmr_reddit", "dev"): "fit + calibrate (5-fold CV)",
        ("pmr_reddit", "test"): "held-out test, both tracks",
        ("pmr_synth", "INBOX_A"): "ranking, expert inbox",
        ("pmr_synth", "INBOX_B"): "ranking, expert inbox",
        ("cssrs_suicidewatch", "all"): "psychiatry transfer",
    }
    name = {"pmr_reddit": "PMR-Reddit", "pmr_synth": "PMR-Synth", "cssrs_suicidewatch": "C-SSRS r/SuicideWatch"}
    lines = [
        f"{name[r.source]} & {r.split.replace('_', ' ')} & {r.n} & {r.n_urgent} & {100 * r.prevalence:.1f} & "
        f"{r.median_chars} & {role[(r.source, r.split)]} \\\\"
        for _, r in d.iterrows()
    ]
    (OUT / "corpora.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    classification()
    ranking()
    transfer()
    precision_floor()
    sensitivity()
    rules()
    corpora()
    print("tables ->", OUT)
