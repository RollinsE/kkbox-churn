"""End-to-end runs on synthetic data, driven through main.py exactly as a user would."""

import json
import os

import pandas as pd
import pytest

import main as pipeline


def _run(tmp_path, *extra):
    argv = [
        "--mode",
        "all",
        "--demo",
        "--demo_users",
        "1500",
        "--no_plots",
        "--run_base_dir",
        str(tmp_path / "runs"),
        "--data_dir",
        str(tmp_path / "data"),
        *extra,
    ]
    run_dir = pipeline.main(argv)
    with open(os.path.join(run_dir, "logs", "run.log"), encoding="utf-8") as f:
        log = f.read()
    with open(os.path.join(run_dir, "metrics", "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    return run_dir, log, summary


def _check_common(run_dir, log, summary):
    for marker in (
        "[START] component=Pipeline mode=all",
        "[DONE] component=Ingest",
        "[DONE] component=Labels",
        "[DONE] component=Features:feb",
        "[DONE] component=Features:mar",
        "[DONE] component=Features:apr",
        "[DONE] component=Training",
        "[RESULT] component=Pipeline mode=train",
        "[DONE] component=Pipeline mode=all status=success",
    ):
        assert marker in log, marker
    assert "[FAIL]" not in log

    submission = pd.read_csv(os.path.join(run_dir, "artifacts", "submission.csv"))
    assert list(submission.columns) == ["msno", "is_churn"]
    assert len(submission) == summary["rows"]["test"]
    assert submission["msno"].is_unique
    assert submission["is_churn"].between(0, 1).all()
    assert submission["is_churn"].nunique() > 1

    assert "Model ranking by log loss" in log
    leaderboard = pd.read_csv(os.path.join(run_dir, "metrics", "leaderboard.csv"))
    assert leaderboard["rank"].tolist() == list(range(1, len(leaderboard) + 1))
    assert leaderboard["log loss"].is_monotonic_increasing

    for artifact in ("ingest_manifest.json", "feature_columns.json", "champion.json"):
        assert os.path.exists(os.path.join(run_dir, "artifacts", artifact)), artifact


def test_demo_pipeline_sqlite_baselines(tmp_path):
    run_dir, log, summary = _run(tmp_path, "--engine", "sqlite", "--frame_format", "csv", "--models", "prior,logistic")
    _check_common(run_dir, log, summary)
    prior = summary["models"]["prior"]["valid"]
    logistic = summary["models"]["logistic"]["valid"]
    assert summary["champion"] == "logistic"
    assert logistic["logloss"] < prior["logloss"]
    assert logistic["auc"] > 0.75


def test_demo_pipeline_lightgbm(tmp_path):
    pytest.importorskip("lightgbm")
    run_dir, log, summary = _run(
        tmp_path,
        "--engine",
        "sqlite",
        "--frame_format",
        "csv",
        "--models",
        "prior,lightgbm",
        "--num_boost_round",
        "300",
        "--early_stopping_rounds",
        "30",
        "--eval_every",
        "25",
        "--num_leaves",
        "15",
        "--min_data_in_leaf",
        "20",
        "--seeds",
        "1,2",
    )
    _check_common(run_dir, log, summary)
    assert "[EVAL] component=LightGBM iter=25 train_binary_logloss=" in log
    assert "[CHECKPOINT] component=LightGBM" in log
    assert log.count("[CHECKPOINT] component=FinalFit model=lightgbm") == 2
    lgbm = summary["models"]["lightgbm"]
    assert lgbm["best_iteration"] >= 1
    assert lgbm["valid"]["logloss"] < summary["models"]["prior"]["valid"]["logloss"]
    assert os.path.exists(os.path.join(run_dir, "metrics", "lightgbm_shap_importance.csv"))


def test_demo_pipeline_duckdb_parquet(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    run_dir, log, summary = _run(
        tmp_path, "--engine", "duckdb", "--frame_format", "parquet", "--models", "prior,logistic"
    )
    _check_common(run_dir, log, summary)
    assert os.path.exists(tmp_path / "data" / "parquet" / "user_logs.parquet")
    assert os.path.exists(tmp_path / "data" / "features" / "features_apr.parquet")


def test_train_on_two_months_with_rebuilt_labels(tmp_path):
    run_dir, log, summary = _run(
        tmp_path,
        "--engine",
        "sqlite",
        "--frame_format",
        "csv",
        "--models",
        "prior,logistic",
        "--train_snapshots",
        "jan,feb",
    )
    _check_common(run_dir, log, summary)
    assert summary["train_snapshots"] == ["jan", "feb"] and summary["population"] == "expiry"
    assert "[INFO] component=Training snapshot=jan population=expiry" in log


def test_validation_month_cannot_be_trained_on(tmp_path):
    argv = [
        "--mode",
        "all",
        "--demo",
        "--demo_users",
        "800",
        "--engine",
        "sqlite",
        "--frame_format",
        "csv",
        "--models",
        "prior",
        "--train_snapshots",
        "feb,mar",
        "--run_dir",
        str(tmp_path / "r"),
        "--data_dir",
        str(tmp_path / "d"),
    ]
    with pytest.raises(SystemExit):
        pipeline.main(argv)
    with open(tmp_path / "r" / "logs" / "run.log", encoding="utf-8") as f:
        assert "validation month cannot also be trained on" in f.read()


def test_failure_is_logged_and_exits_nonzero(tmp_path):
    """Training without features must log [FAIL] with a traceback and exit 1."""
    argv = [
        "--mode",
        "train",
        "--frame_format",
        "csv",
        "--models",
        "prior",
        "--run_dir",
        str(tmp_path / "runs" / "run_fail"),
        "--data_dir",
        str(tmp_path / "empty"),
    ]
    with pytest.raises(SystemExit) as exc:
        pipeline.main(argv)
    assert exc.value.code == 1
    with open(tmp_path / "runs" / "run_fail" / "logs" / "run.log", encoding="utf-8") as f:
        log = f.read()
    assert "[FAIL] component=Training" in log
    assert "[FAIL] component=Pipeline" in log
    assert "FileNotFoundError" in log
