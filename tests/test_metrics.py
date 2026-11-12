import numpy as np

from triage import metrics
from triage.methods.rules import category_hits


def test_threshold_reaches_target_recall():
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.1
    s = rng.normal(size=500) + 2 * y
    t = metrics.select_threshold(s, y, 0.95)
    assert metrics.operating_point(s, y, t)["recall"] >= 0.95
    # one notch higher must fall below target
    assert metrics.operating_point(s, y, np.nextafter(t, np.inf))["recall"] < 0.95


def test_operating_point_counts():
    s = np.array([0.9, 0.8, 0.3, 0.2])
    y = np.array([1, 0, 1, 0], bool)
    op = metrics.operating_point(s, y, 0.5)
    assert op["recall"] == 0.5 and op["fa_per_100"] == 50 and op["missed"] == 1


def test_perfect_ranking_ndcg():
    rel = np.array([5.0, 3, 1, 0])
    assert metrics.ndcg_at_k(rel, 4) == 1.0
    assert metrics.pairwise_accuracy([2, 1, 1], [1, 2, 1]) == 0.5


def test_negation_suppresses_rule():
    assert category_hits("I have chest pain since this morning")["chest_pain"] == 1
    assert category_hits("No chest pain, just a cough")["chest_pain"] == 0
    assert category_hits("I've been thinking about suicide")["suicidality"] == 1


def test_ppv_threshold_reaches_target():
    rng = np.random.default_rng(1)
    y = rng.random(800) < 0.1
    s = rng.normal(size=800) + 2.5 * y
    t = metrics.select_threshold_ppv(s, y, 2 / 3)
    assert metrics.operating_point(s, y, t)["ppv"] >= 2 / 3
    # recall at the locked point equals the best recall the curve allows at that PPV
    assert np.isclose(metrics.operating_point(s, y, t)["recall"], metrics.recall_at_ppv(s, y, 2 / 3))


def test_workload_matches_gate_table():
    # The 5110 gate at 2% prevalence and 300 messages/week (BMIN 5070, Table 5).
    w = metrics.workload(0.95, 10, 0.02)
    assert round(w["flags_per_week"], 1) == 35.1 and round(w["ppv"], 2) == 0.16
    assert round(metrics.workload(0.95, 10, 0.0004)["reads_per_urgent"]) == 264
