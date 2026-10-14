"""The model marathon: registry, skipping, failure tolerance, resume, ensemble and the two submissions."""

import json
import os

import numpy as np
import pandas as pd
import pytest
from conftest import quiet_logger

from src import model as training
from src.constants import TARGET_COL
from src.evaluate import match_rate
from src.experiment import ensure_run_subdirs
from src.marathon import MODEL_NAMES, ModelSpec, ensemble_members, registry
from src.storage import features_path, write_frame

LOG = quiet_logger()


# --------------------------------------------------------------------------- toy data and toy models


def _features(n, rate, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(np.log(rate / (1 - rate)) + 1.5 * x)))).astype(int)
    return pd.DataFrame(
        {
            "msno": [f"u{seed}_{i}" for i in range(n)],
            "x": x,
            "z": rng.normal(size=n),
            TARGET_COL: y,
            "in_population_expiry": 1,
            "in_labeller_population": 1,
        }
    )


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    for name, rate, seed in (("feb", 0.06, 1), ("mar", 0.08, 2), ("apr", 0.07, 3)):
        df = _features(3000, rate, seed)
        if name == "apr":
            df[TARGET_COL] = 0
        path = features_path(str(d), name, "csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_frame(df, path)
    return str(d)


@pytest.fixture
def run_dir(tmp_path):
    r = str(tmp_path / "run")
    ensure_run_subdirs(r)
    return r


class ConstantSpec(ModelSpec):
    """Predicts training rate * factor for everyone, so we know exactly how it scores."""

    calls = []

    def fit(self, fit_df, stop_df, cols, params, seed, LOG, cfg):
        ConstantSpec.calls.append(self.name)
        if params.get("explode"):
            raise RuntimeError("boom")
        return {"model": float(fit_df[TARGET_COL].mean()) * params["factor"], "best_iteration": 10}

    def predict(self, model, df, cols):
        return np.full(len(df), model)

    final_calls = []

    def fit_final(self, full_df, cols, params, seed, rounds, LOG, cfg):
        ConstantSpec.final_calls.append((self.name, seed))
        return float(full_df[TARGET_COL].mean()) * params["factor"]

    def save(self, model, stem):
        with open(stem + ".json", "w") as f:
            json.dump({"value": model}, f)
        return stem + ".json"


def _toy_registry(*specs):
    return lambda lgb_overrides=None: list(specs)


def _train(data_dir, run_dir, models, **kw):
    return training.run_training(
        data_dir, run_dir, LOG, models=models, frame_format="csv", plots=False, seeds=(1, 2), **kw
    )


# --------------------------------------------------------------------------- registry


def test_registry_covers_every_model_name():
    specs = registry()
    assert tuple(s.name for s in specs) == MODEL_NAMES
    assert {s.name for s in specs if s.role == "baseline"} == {"prior"}
    for s in specs:
        if s.space:
            assert set(s.space) <= set(s.defaults), s.name  # defaults are always trial 1 when tuning


def test_parse_model_list_rejects_unknown_names():
    assert training.parse_model_list("prior, LightGBM,prior") == ["prior", "lightgbm"]
    with pytest.raises(ValueError):
        training.parse_model_list("prior,randomforest")


def test_ensemble_takes_best_member_of_each_candidate_family():
    results = {
        "prior": {"role": "baseline", "family": "baseline", "valid": {"logloss": 0.10}},
        "lightgbm": {"role": "candidate", "family": "lightgbm", "valid": {"logloss": 0.20}},
        "lightgbm_tuned": {"role": "candidate", "family": "lightgbm", "valid": {"logloss": 0.19}},
        "xgboost": {"role": "candidate", "family": "xgboost", "valid": {"logloss": 0.21}},
        "logistic": {"role": "candidate", "family": "linear", "valid": {"logloss": 0.25}},
        "catboost": {"role": "candidate", "family": "catboost", "valid": {"logloss": 0.30}},
    }
    assert ensemble_members(results, 3) == ["lightgbm_tuned", "xgboost", "logistic"]


def test_match_rate_hits_the_target_mean_and_keeps_the_order():
    p = np.random.default_rng(0).beta(1, 15, 5000)
    q = match_rate(p, 0.0726)
    assert abs(q.mean() - 0.0726) < 1e-6
    assert (np.argsort(p, kind="mergesort") == np.argsort(q, kind="mergesort")).all()


# --------------------------------------------------------------------------- whole marathon with toy models


def test_ensemble_champion_refits_every_member(data_dir, run_dir, monkeypatch):
    high = ConstantSpec("high", "a", defaults={"factor": 2.0})
    low = ConstantSpec("low", "b", defaults={"factor": 0.5})
    monkeypatch.setattr(training, "registry", _toy_registry(high, low))
    summary = _train(data_dir, run_dir, ["high", "low"])

    # The average of an over- and an under-predictor is closer to the real rate than either.
    assert summary["champion"] == "ensemble"
    assert summary["models"]["ensemble"]["members"] == ["low", "high"] or summary["models"]["ensemble"]["members"] == [
        "high",
        "low",
    ]
    for member in ("high", "low"):  # one seed per member by default (--ensemble_seeds 1)
        assert os.path.exists(os.path.join(run_dir, "artifacts", f"{member}_final_seed1.json"))
        assert not os.path.exists(os.path.join(run_dir, "artifacts", f"{member}_final_seed2.json"))

    sub = pd.read_csv(summary["submission"])
    recal = pd.read_csv(summary["submission_recalibrated"])
    assert list(sub["msno"]) == list(recal["msno"])
    assert abs(recal["is_churn"].mean() - summary["calibration_target"]) < 1e-4
    valid_rate = summary["models"]["ensemble"]["valid"]["base_rate"]
    assert summary["calibration_target"] == pytest.approx(valid_rate)

    board = pd.read_csv(os.path.join(run_dir, "metrics", "leaderboard.csv"))
    assert board.loc[0, "model"] == "ensemble" and bool(board.loc[0, "champion"])
    assert {"family", "rate-matched", "rounds", "train time"} <= set(board.columns)


def test_failed_and_missing_models_are_skipped_not_fatal(data_dir, run_dir, monkeypatch):
    good = ConstantSpec("good", "a", defaults={"factor": 1.0})
    bad = ConstantSpec("bad", "b", defaults={"factor": 1.0, "explode": True})
    missing = ConstantSpec("missing", "c", library="surely_not_installed_xyz", defaults={"factor": 1.0})
    monkeypatch.setattr(training, "registry", _toy_registry(good, bad, missing))
    summary = _train(data_dir, run_dir, ["good", "bad", "missing"])

    assert summary["champion"] == "good"
    assert "boom" in summary["failed"]["bad"]
    assert "not installed" in summary["skipped"]["missing"]
    assert "ensemble" not in summary["models"]  # only one family finished


def test_resume_reuses_finished_models(data_dir, run_dir, monkeypatch):
    a = ConstantSpec("a", "a", defaults={"factor": 1.0})
    b = ConstantSpec("b", "b", defaults={"factor": 1.2})
    monkeypatch.setattr(training, "registry", _toy_registry(a, b))
    ConstantSpec.calls.clear()
    first = _train(data_dir, run_dir, ["a", "b"])
    assert ConstantSpec.calls == ["a", "b"]

    ConstantSpec.calls.clear()
    second = _train(data_dir, run_dir, ["a", "b"])
    assert ConstantSpec.calls == []  # nothing refitted for validation
    assert second["models"]["a"]["valid"] == first["models"]["a"]["valid"]

    # Changing a setting that affects the result invalidates the cache.
    ConstantSpec.calls.clear()
    _train(data_dir, run_dir, ["a", "b"], holdout_share=0.3)
    assert ConstantSpec.calls == ["a", "b"]


def test_tuning_adds_a_separate_entry_and_never_loses_the_defaults(data_dir, run_dir, monkeypatch):
    a = ConstantSpec("a", "a", defaults={"factor": 1.0}, space={"factor": ("float", 0.2, 3.0, False)})
    monkeypatch.setattr(training, "registry", _toy_registry(a))
    summary = _train(data_dir, run_dir, ["a"], tune_models=["a"], tune_trials=5, ensemble=False)
    assert summary["tuned"] == ["a_tuned"]
    assert "a_tuned" in summary["models"]


def test_calibration_target_override(data_dir, run_dir, monkeypatch):
    a = ConstantSpec("a", "a", defaults={"factor": 1.0})
    monkeypatch.setattr(training, "registry", _toy_registry(a))
    summary = _train(data_dir, run_dir, ["a"], calibration_target=0.05)
    recal = pd.read_csv(summary["submission_recalibrated"])
    assert abs(recal["is_churn"].mean() - 0.05) < 1e-4
    with pytest.raises(ValueError):
        _train(data_dir, run_dir, ["a"], calibration_target=1.5)


def _fake_results(**scores):
    """name -> (logloss, train_seconds, role)"""
    return {n: {"valid": {"logloss": ll}, "train_seconds": sec, "role": role} for n, (ll, sec, role) in scores.items()}


def test_tie_rule_takes_the_quickest_model_within_the_margin():
    results = _fake_results(
        ensemble=(0.1714, 814, "ensemble"),
        xgboost=(0.1717, 47, "candidate"),
        lightgbm=(0.1718, 69, "candidate"),
        catboost=(0.1722, 698, "candidate"),
        logistic=(0.1814, 7, "candidate"),
        prior=(0.2630, 0, "baseline"),
    )
    assert training.choose_champion(results, 0.001) == ("xgboost", "ensemble")
    assert training.choose_champion(results, 0.0) == ("ensemble", "ensemble")
    assert training.choose_champion(results, 0.02) == ("logistic", "ensemble")
    assert training.choose_champion(_fake_results(prior=(0.26, 0, "baseline")), 0.001) == ("prior", "prior")


def test_final_fit_resumes_finished_seeds(data_dir, run_dir, monkeypatch):
    a = ConstantSpec("a", "a", defaults={"factor": 1.0})
    monkeypatch.setattr(training, "registry", _toy_registry(a))
    ConstantSpec.final_calls.clear()
    first = _train(data_dir, run_dir, ["a"])
    assert ConstantSpec.final_calls == [("a", 1), ("a", 2)]

    # Simulate a disconnect after seed 1: seed 2's saved prediction is lost.
    os.remove(os.path.join(run_dir, "artifacts", "final", "a_seed2_test_pred.npy"))
    ConstantSpec.final_calls.clear()
    second = _train(data_dir, run_dir, ["a"])
    assert ConstantSpec.final_calls == [("a", 2)]
    assert pd.read_csv(second["submission"]).equals(pd.read_csv(first["submission"]))


# --------------------------------------------------------------------------- real libraries, when installed


@pytest.mark.parametrize("name", ["xgboost", "catboost"])
def test_boosting_library_runs_end_to_end(name, data_dir, run_dir):
    pytest.importorskip(name)
    summary = _train(
        data_dir,
        run_dir,
        ["prior", name],
        num_boost_round=200,
        early_stopping_rounds=20,
        eval_every=25,
        catboost_iterations=200,
    )
    res = summary["models"][name]
    assert res["best_iteration"] >= 1
    assert res["valid"]["logloss"] < summary["models"]["prior"]["valid"]["logloss"]
    assert os.path.exists(
        os.path.join(run_dir, "artifacts", f"{name}_final_seed1" + (".json" if name == "xgboost" else ".cbm"))
    )
