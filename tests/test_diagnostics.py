"""Diagnostics: the label rebuild rule on hand-made cases, the rate matcher, and an end-to-end demo run."""

import os

import numpy as np
import pandas as pd

import main as pipeline
from src.diagnostics import match_rate


def test_match_rate_hits_target_mean_and_keeps_order():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.3, 5000)
    q = match_rate(p, 0.09)
    assert abs(q.mean() - 0.09) < 1e-6
    assert (np.argsort(p) == np.argsort(q)).all()


def test_diagnose_mode_end_to_end(tmp_path):
    common = [
        "--demo",
        "--demo_users",
        "1500",
        "--engine",
        "sqlite",
        "--frame_format",
        "csv",
        "--data_dir",
        str(tmp_path / "data"),
        "--run_base_dir",
        str(tmp_path / "runs"),
    ]
    pipeline.main(["--mode", "ingest", *common])
    pipeline.main(["--mode", "labels", *common])
    pipeline.main(["--mode", "features", *common])
    run_dir = pipeline.main(["--mode", "diagnose", "--diag_rows", "1000", *common])

    report = os.path.join(run_dir, "metrics", "diagnostics_report.md")
    assert os.path.exists(report)
    text = open(report, encoding="utf-8").read()
    assert "Check failed" not in text
    for name in (
        "shift",
        "labels",
        "labels_confusion",
        "labels_outcomes",
        "truncation",
        "truncation_late_early",
        "churn_by_expiry_day",
        "segments",
        "segments_split",
        "drift",
        "expiry",
    ):
        assert os.path.exists(os.path.join(run_dir, "metrics", f"diagnostics_{name}.csv")), name

    shift = pd.read_csv(os.path.join(run_dir, "metrics", "diagnostics_shift.csv"))
    assert len(shift) == 4 and shift["auc"].between(0.5, 1).all()
    labels = pd.read_csv(os.path.join(run_dir, "metrics", "diagnostics_labels.csv"))
    feb = labels[labels["month"] == "2017-02"].iloc[0]
    assert 0 <= feb["agreement"] <= 1
    assert 0 <= feb["official churners also rebuilt churn"] <= 1
    assert labels.loc[labels["month"] == "2017-03", "note"].str.startswith("skipped").all()

    trunc = pd.read_csv(os.path.join(run_dir, "metrics", "diagnostics_truncation.csv"))
    assert len(trunc) == 2 and trunc["agreement"].between(0, 1).all()

    log = open(os.path.join(run_dir, "logs", "run.log"), encoding="utf-8").read()
    assert "[DONE] component=Diagnostics " in log and "[FAIL]" not in log
