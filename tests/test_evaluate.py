import numpy as np

from src.evaluate import calibration_table, classification_metrics, lift_table


def test_metrics_on_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = classification_metrics(y, p)
    assert m["n"] == 4 and m["base_rate"] == 0.5
    assert abs(m["auc"] - 0.75) < 1e-12
    expected_logloss = -np.mean([np.log(0.9), np.log(0.6), np.log(0.35), np.log(0.8)])
    assert abs(m["logloss"] - expected_logloss) < 1e-12


def test_metrics_survive_single_class_and_extreme_probabilities():
    m = classification_metrics([0, 0, 0], [0.0, 1.0, 0.5])
    assert m["auc"] is None and m["pr_auc"] is None
    assert np.isfinite(m["logloss"])


def test_lift_table_captures_all_churners():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    lift = lift_table(y, p)
    assert lift["n"].sum() == 1000
    assert lift["churners"].sum() == y.sum()
    assert abs(lift["cum_capture"].iloc[-1] - 1.0) < 1e-12
    assert lift["lift"].iloc[0] > 1.0 > lift["lift"].iloc[-1]


def test_calibration_table_of_calibrated_predictions_has_small_gaps():
    rng = np.random.default_rng(1)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    table = calibration_table(y, p)
    assert len(table) == 10 and table["n"].sum() == 20000
    assert table["gap"].abs().max() < 0.05
