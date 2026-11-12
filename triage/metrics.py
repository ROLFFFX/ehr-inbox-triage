"""Evaluation metrics for the two tracks.

Classification track: a method's score is thresholded to flag urgent
messages. The clinical operating point from the BMIN 5110 specification is
recall >= 0.95 on urgent messages with fewer than 10 false alarms per 100
non-urgent messages (FPR < 0.10).

Ranking track: a method's score orders a simulated inbox; we measure how
high urgent messages land and how well the whole ordering tracks the PMR
level.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

TARGET_RECALL = 0.95
FA_BUDGET = 0.10  # false alarms per non-urgent message (10 per 100)

# Second operating point, from the BMIN 5070 review: after a pilot at 95%
# sensitivity (PPV 0.28), NYU Langone's nurse managers moved their threshold
# so that two in three flags were truly urgent (Yang et al. 2024).
TARGET_PPV = 2 / 3
# Prevalences the result is transported to: 2% for urgency broadly defined in
# a psychiatric inbox, and the 0.04% floor for self-harm/suicidal content
# among all portal messages (Duvall et al. 2021).
HF_PREVALENCES = {"p2": 0.02, "p004": 0.0004}
WEEKLY_VOLUME = 300  # messages per clinician-week (Adler-Milstein et al. 2020)


# --------------------------------------------------------------------------
# Classification track
# --------------------------------------------------------------------------

def select_threshold(scores: np.ndarray, y: np.ndarray, target_recall: float = TARGET_RECALL) -> float:
    """Highest threshold t such that flagging score >= t reaches target recall.

    Applied to calibration-set scores only; the threshold is then locked
    before touching the test set.
    """
    pos = np.sort(np.asarray(scores)[np.asarray(y, bool)])[::-1]
    if len(pos) == 0:
        raise ValueError("no positives to calibrate on")
    k = math.ceil(target_recall * len(pos))
    return float(pos[k - 1])


def operating_point(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    scores, y = np.asarray(scores), np.asarray(y, bool)
    flag = scores >= threshold
    tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
    npos, nneg = int(y.sum()), int((~y).sum())
    return {
        "recall": tp / npos if npos else np.nan,
        "fa_per_100": 100 * fp / nneg if nneg else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "flag_rate": flag.mean(),
        "missed": npos - tp,
        "false_alarms": fp,
    }


def recall_at_fa(scores: np.ndarray, y: np.ndarray, budget: float = FA_BUDGET) -> float:
    """Best achievable recall with FPR <= budget (threshold chosen on the same data)."""
    fpr, tpr, _ = roc_curve(y, scores)
    return float(tpr[fpr <= budget + 1e-12].max())


def fa_at_recall(scores: np.ndarray, y: np.ndarray, target: float = TARGET_RECALL) -> float:
    """False alarms per 100 non-urgent at the lowest FPR reaching target recall."""
    fpr, tpr, _ = roc_curve(y, scores)
    return float(100 * fpr[tpr >= target - 1e-12].min())


def ppv_at(recall: float, fpr: float, prevalence: float) -> float:
    """PPV of a flag with the given recall and FPR at another prevalence."""
    tp, fp = recall * prevalence, fpr * (1 - prevalence)
    return tp / (tp + fp) if tp + fp > 0 else np.nan


def recall_at_ppv(scores: np.ndarray, y: np.ndarray, target: float = TARGET_PPV, prevalence: float | None = None) -> float:
    """Best recall whose PPV, at the sample's or a given prevalence, reaches target."""
    y = np.asarray(y, bool)
    fpr, tpr, _ = roc_curve(y, scores)
    p = y.mean() if prevalence is None else prevalence
    ppv = np.array([ppv_at(r, f, p) for r, f in zip(tpr, fpr)])
    ok = (tpr > 0) & (ppv >= target - 1e-12)
    return float(tpr[ok].max()) if ok.any() else 0.0


def select_threshold_ppv(scores: np.ndarray, y: np.ndarray, target: float = TARGET_PPV) -> float:
    """Lowest threshold (highest recall) at which calibration PPV reaches target.

    The NYU-style counterpart of ``select_threshold``; locked the same way.
    Returns +inf when no threshold reaches the target.
    """
    scores, y = np.asarray(scores), np.asarray(y, bool)
    order = np.argsort(-scores, kind="stable")
    s, hit = scores[order], y[order]
    tp = np.cumsum(hit)
    ppv = tp / np.arange(1, len(s) + 1)
    # only cut between distinct scores, since a threshold flags every tie
    last_of_tie = np.r_[s[1:] != s[:-1], True]
    ok = last_of_tie & (ppv >= target - 1e-12) & (tp > 0)
    return float(s[np.flatnonzero(ok).max()]) if ok.any() else float("inf")


def workload(recall: float, fa_per_100: float, prevalence: float, volume: int = WEEKLY_VOLUME) -> dict:
    """What an operating point means to the reader of the flag at a given prevalence."""
    fpr = fa_per_100 / 100
    ppv = ppv_at(recall, fpr, prevalence)
    return {
        "ppv": ppv,
        "flags_per_week": volume * (recall * prevalence + fpr * (1 - prevalence)),
        "reads_per_urgent": 1 / ppv if ppv and ppv > 0 else np.inf,
        "missed_per_week": volume * prevalence * (1 - recall),
    }


def threshold_free(scores: np.ndarray, y: np.ndarray) -> dict:
    return {
        "auroc": roc_auc_score(y, scores),
        "auprc": average_precision_score(y, scores),
        "recall_at_fa10": recall_at_fa(scores, y),
        "fa100_at_recall95": fa_at_recall(scores, y),
        "recall_at_ppv67": recall_at_ppv(scores, y),
        "recall_at_ppv67_p2": recall_at_ppv(scores, y, prevalence=HF_PREVALENCES["p2"]),
    }


def bootstrap(fn, scores: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Stratified bootstrap of a metric function returning a dict.

    Positives and negatives are resampled separately so every replicate keeps
    the observed prevalence (important with ~40 positives).
    """
    rng = np.random.default_rng(seed)
    scores, y = np.asarray(scores), np.asarray(y, bool)
    pi, ni = np.flatnonzero(y), np.flatnonzero(~y)
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pi, len(pi)), rng.choice(ni, len(ni))])
        rows.append(fn(scores[idx], y[idx]))
    return pd.DataFrame(rows)


def summarize_ci(point: dict, boots: pd.DataFrame, alpha: float = 0.05) -> dict:
    out = {}
    for k, v in point.items():
        out[k] = v
        if k in boots:
            lo, hi = np.nanpercentile(boots[k], [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out[f"{k}_lo"], out[f"{k}_hi"] = lo, hi
    return out


def meets_spec(recall: float, fa_per_100: float) -> bool:
    return recall >= TARGET_RECALL and fa_per_100 < 100 * FA_BUDGET


# --------------------------------------------------------------------------
# Ranking track
# --------------------------------------------------------------------------

def relevance(level: np.ndarray) -> np.ndarray:
    """Linear gain: Level 1 -> 5, Level 6 -> 0."""
    return 6.0 - np.asarray(level, float)


def _dcg(rel: np.ndarray, k: int) -> float:
    rel = rel[:k]
    return float((rel / np.log2(np.arange(2, len(rel) + 2))).sum())


def ndcg_at_k(order_rel: np.ndarray, k: int) -> float:
    ideal = _dcg(np.sort(order_rel)[::-1], k)
    return _dcg(order_rel, k) / ideal if ideal > 0 else np.nan


def rank_inbox(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Descending order by score; ties broken uniformly at random."""
    return np.lexsort((rng.random(len(scores)), -np.asarray(scores)))


def inbox_metrics(scores: np.ndarray, level: np.ndarray, urgent: np.ndarray, rng: np.random.Generator) -> dict:
    order = rank_inbox(scores, rng)
    rel = relevance(level)[order]
    urg = np.asarray(urgent, bool)[order]
    n = len(order)
    out = {
        "ndcg@10": ndcg_at_k(rel, 10),
        # T-NDCG (Gatto et al. 2026): NDCG of the ordering minus NDCG of its reverse.
        "t_ndcg@10": ndcg_at_k(rel, 10) - ndcg_at_k(rel[::-1], 10),
        "t_ndcg@30": ndcg_at_k(rel, 30) - ndcg_at_k(rel[::-1], 30),
    }
    if urg.any():
        pos = np.flatnonzero(urg) + 1  # 1-indexed queue positions
        out["urgent_mean_pos"] = pos.mean() / n  # 0..1, lower is better
        out["urgent_worst_pos"] = pos.max() / n  # last urgent message: the "sink" penalty
        out["urgent_in_top5"] = (pos <= 5).mean()
    return out


def simulate_inboxes(
    scores: np.ndarray,
    level: np.ndarray,
    urgent: np.ndarray,
    inbox_size: int = 30,
    n_inboxes: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Draw random inboxes from a message pool and score each ordering.

    The same seed gives every method the identical set of inboxes.
    """
    draw_rng = np.random.default_rng(seed)
    tie_rng = np.random.default_rng(seed + 1)
    scores, level, urgent = map(np.asarray, (scores, level, urgent))
    rows = []
    for _ in range(n_inboxes):
        idx = draw_rng.choice(len(scores), inbox_size, replace=False)
        rows.append(inbox_metrics(scores[idx], level[idx], urgent[idx], tie_rng))
    return pd.DataFrame(rows)


def pairwise_accuracy(score_chosen: np.ndarray, score_rejected: np.ndarray) -> float:
    """Share of official pairs where the more urgent message scores higher (ties = 0.5)."""
    d = np.asarray(score_chosen) - np.asarray(score_rejected)
    return float(((d > 0) + 0.5 * (d == 0)).mean())
