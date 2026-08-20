import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

from src.evaluate import bootstrap_metrics, paired_bootstrap_diff, point_metrics


def _fake(n=400, noise=3.0, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.uniform(15, 80, n)
    return y, y + rng.normal(0, noise, n)


def test_point_metrics_cross_check_sklearn():
    y, p = _fake()
    m = point_metrics(y, p)
    assert np.isclose(m["mae"], mean_absolute_error(y, p))
    assert np.isclose(m["r2"], r2_score(y, p))
    assert np.isclose(m["auc_at_40"], roc_auc_score((y < 40).astype(int), -p))


def test_sens_spec_hand_computed():
    y = np.array([30.0, 35.0, 45.0, 60.0])   # two positives at cut 40
    p = np.array([32.0, 44.0, 43.0, 58.0])   # one caught, one missed
    m = point_metrics(y, p)
    assert m["sens_at_40"] == 0.5
    assert m["spec_at_40"] == 1.0
    assert m["n_pos_at_40"] == 2


def test_band_stratification_counts():
    y, p = _fake(500)
    m = point_metrics(y, p)
    total = sum(m[f"n_band_{lo}_{hi}"] for lo, hi in
                ((0, 30), (30, 40), (40, 55), (55, 101)))
    assert total == 500


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    y, p = _fake(300)
    m1 = bootstrap_metrics(y, p, n_boot=200)
    m2 = bootstrap_metrics(y, p, n_boot=200)
    assert m1["mae_ci_lo"] <= m1["mae"] <= m1["mae_ci_hi"]
    assert m1["mae_ci_lo"] == m2["mae_ci_lo"]  # fixed seed => reproducible CIs


def test_paired_bootstrap_direction():
    y, good = _fake(400, noise=2.0)
    _, bad = _fake(400, noise=8.0, seed=2)
    d = paired_bootstrap_diff(y, good, bad, n_boot=300)
    assert d["mae_diff"] < 0 and d["distinguishable"]
    same = paired_bootstrap_diff(y, good, good, n_boot=300)
    assert not same["distinguishable"]
