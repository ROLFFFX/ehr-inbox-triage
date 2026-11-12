"""Evaluation protocol shared by every method.

1. Calibration. Methods are fit and calibrated only on the PMR-Reddit
   train + dev pool. Trainable methods get out-of-fold scores from
   stratified 5-fold CV; rule methods score the pool directly. The
   threshold reaching 95% recall on those calibration scores is locked.
2. Classification track. The locked threshold is applied to the 362
   held-out test messages: achieved recall and false alarms per 100
   non-urgent, with stratified bootstrap CIs. Threshold-free frontier
   metrics are reported alongside.
3. Ranking track. The method's score orders 500 simulated 30-message
   inboxes drawn from the test messages, the two expert PMR-Synth inboxes,
   and the 1,502 official test pairs.
4. Psychiatry transfer. The locked threshold is applied unchanged to
   r/SuicideWatch posts with C-SSRS labels.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import data, metrics


@dataclass
class Corpora:
    pool: pd.DataFrame
    test: pd.DataFrame
    pairs: pd.DataFrame
    synth: pd.DataFrame
    cssrs: pd.DataFrame

    @classmethod
    def load(cls, max_level: float = data.URGENT_MAX_LEVEL) -> "Corpora":
        return cls(
            pool=data.load_development(max_level),
            test=data.load_reddit_test(max_level),
            pairs=data.load_reddit_test_pairs(),
            synth=data.load_synth(max_level=max_level),
            cssrs=data.load_cssrs(),
        )


@dataclass
class Result:
    method: str
    threshold: float
    scores: dict[str, pd.DataFrame] = field(default_factory=dict)
    classification: dict = field(default_factory=dict)
    ranking: dict = field(default_factory=dict)
    transfer: dict = field(default_factory=dict)
    inboxes: pd.DataFrame | None = None


class FoldEnsemble:
    """Scores new text with the mean of the CV fold models.

    Refitting one model on the full pool changes the score scale (more data,
    a possibly different C), so a threshold calibrated on out-of-fold scores
    would not transfer. Averaging the very models that produced the
    calibration scores keeps the two on the same scale.
    """

    def __init__(self, models):
        self.models = models

    def score(self, texts) -> np.ndarray:
        return np.mean([m.score(texts) for m in self.models], axis=0)


def out_of_fold(method, pool: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> tuple[np.ndarray, FoldEnsemble]:
    oof = np.zeros(len(pool))
    models = []
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(pool, pool["urgent"]):
        m = copy.deepcopy(method).fit(pool["text"].iloc[tr], pool["level"].iloc[tr], pool["urgent"].iloc[tr])
        oof[va] = m.score(pool["text"].iloc[va])
        models.append(m)
    return oof, FoldEnsemble(models)


def _ci_block(scores, y, n_boot, seed, threshold=None) -> dict:
    if threshold is None:
        fn = metrics.threshold_free
    else:
        fn = lambda s, t: metrics.operating_point(s, t, threshold)  # noqa: E731
    return metrics.summarize_ci(fn(scores, y), metrics.bootstrap(fn, scores, y, n_boot, seed))


def evaluate(method, c: Corpora, seed: int = 0, n_boot: int = 2000, n_inboxes: int = 500) -> Result:
    pool, test = c.pool, c.test

    # 1. calibration
    if method.needs_fit:
        cal, final = out_of_fold(method, pool, seed=seed)
    else:
        cal, final = method.score(pool["text"]), method
    thr = metrics.select_threshold(cal, pool["urgent"])
    res = Result(method.name, thr)

    s_test = final.score(test["text"])
    s_synth = final.score(c.synth["text"])
    s_cssrs = final.score(c.cssrs["text"])
    res.scores = {
        "pool_oof": pool.assign(score=cal),
        "test": test.assign(score=s_test),
        "synth": c.synth.assign(score=s_synth),
        "cssrs": c.cssrs.assign(score=s_cssrs),
    }

    # 2. classification track
    y = test["urgent"].values
    op = _ci_block(s_test, y, n_boot, seed, thr)
    tf = _ci_block(s_test, y, n_boot, seed)
    cal_op = metrics.operating_point(cal, pool["urgent"].values, thr)
    res.classification = {
        "threshold": thr,
        "cal_recall": cal_op["recall"],
        "cal_fa_per_100": cal_op["fa_per_100"],
        **op,
        **tf,
        "meets_spec": metrics.meets_spec(op["recall"], op["fa_per_100"]),
    }
    # The method's own decision rule (e.g. "any rule fires", P > 0.5), shown
    # to separate "cannot reach 95% recall" from "reaches it only by flagging
    # everything".
    nominal = getattr(method, "nominal_threshold", None)
    if nominal is not None:
        nop = metrics.operating_point(s_test, y, nominal)
        res.classification |= {"nominal_recall": nop["recall"], "nominal_fa_per_100": nop["fa_per_100"]}

    # Second operating point (BMIN 5070): a threshold locked on calibration so
    # that two in three flags are urgent, as NYU's nurse managers chose.
    thr_ppv = metrics.select_threshold_ppv(cal, pool["urgent"].values)
    pop = _ci_block(s_test, y, n_boot, seed, thr_ppv)
    res.classification |= {"ppv_threshold": thr_ppv, **{f"ppvop_{k}": v for k, v in pop.items()}}
    # What the locked gate threshold would mean in a psychiatric inbox.
    for tag, p in metrics.HF_PREVALENCES.items():
        w = metrics.workload(op["recall"], op["fa_per_100"], p)
        res.classification |= {f"{k}_{tag}": v for k, v in w.items()}

    # 3. ranking track
    inbox = metrics.simulate_inboxes(s_test, test["level"].values, y, n_inboxes=n_inboxes, seed=seed)
    res.inboxes = inbox
    score_of = dict(zip(test["text"], s_test))
    pa = {}
    for diff, g in c.pairs.groupby("difficulty"):
        pa[f"pair_acc_{diff}"] = metrics.pairwise_accuracy(g["chosen"].map(score_of), g["rejected"].map(score_of))
    pa["pair_acc_all"] = metrics.pairwise_accuracy(c.pairs["chosen"].map(score_of), c.pairs["rejected"].map(score_of))
    synth = {}
    rng = np.random.default_rng(seed)
    for name, g in res.scores["synth"].groupby("split"):
        m = metrics.inbox_metrics(g["score"].values, g["level"].values, g["urgent"].values, rng)
        synth.update({f"synth_{name}_{k}": v for k, v in m.items()})
    res.ranking = {
        **{f"sim_{k}": v for k, v in inbox.mean().items()},
        **{f"sim_{k}_sd": v for k, v in inbox.std().items()},
        **pa,
        **synth,
    }

    # 4. psychiatry transfer (locked threshold, no refit)
    yc = c.cssrs["urgent"].values
    res.transfer = {
        **{f"cssrs_{k}": v for k, v in metrics.operating_point(s_cssrs, yc, thr).items()},
        **{f"cssrs_{k}": v for k, v in metrics.threshold_free(s_cssrs, yc).items()},
    }
    return res
