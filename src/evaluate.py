"""Metrics, calibration and lift for churn probabilities.

Log loss is the competition metric, so it is the headline number. AUC and
PR-AUC say how well the model ranks users; the calibration table says whether
a predicted 20% really means about 20%.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.logger import log_stage

COMPONENT = "Evaluate"
EPS = 1e-15


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def match_rate(p, target_rate: float, iters: int = 60) -> np.ndarray:
    """Shift predictions on the logit scale until their mean equals target_rate.

    The ranking is unchanged (the shift is the same for everyone), so AUC does not
    move; only the overall level of the probabilities does. Scored against a
    month's own churn rate it is an oracle, so on validation it is a diagnostic.
    """
    z = logit(p)
    lo, hi = -10.0, 10.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return 1 / (1 + np.exp(-(z + (lo + hi) / 2)))


def classification_metrics(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), EPS, 1 - EPS)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "base_rate": float(y_true.mean()),
        "mean_pred": float(y_prob.mean()),
        "logloss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "auc": float(roc_auc_score(y_true, y_prob)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both_classes else None,
    }


def calibration_table(y_true, y_prob, bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predicted probability vs the observed churn rate in each."""
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob, dtype=float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=min(bins, len(df)), labels=False)
    out = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).reset_index()
    out["gap"] = out["mean_pred"] - out["observed"]
    return out


def lift_table(y_true, y_prob, bins: int = 10) -> pd.DataFrame:
    """Deciles from highest to lowest risk: how much churn does the top X% capture?"""
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob, dtype=float)})
    df = df.sort_values("p", ascending=False, kind="mergesort").reset_index(drop=True)
    df["decile"] = (np.arange(len(df)) * bins // len(df)) + 1
    base = df["y"].mean()
    out = df.groupby("decile").agg(n=("y", "size"), churners=("y", "sum"), churn_rate=("y", "mean")).reset_index()
    out["lift"] = out["churn_rate"] / base if base > 0 else np.nan
    out["cum_capture"] = out["churners"].cumsum() / max(int(df["y"].sum()), 1)
    return out


def evaluate_predictions(y_true, y_prob, model_tag: str, split: str, run_dir: str, LOG) -> dict:
    """Compute metrics, write the calibration and lift tables into the run directory, and log them.

    Figures are drawn once per run from these predictions by src/figures.py.
    """
    metrics = classification_metrics(y_true, y_prob)
    log_stage(
        LOG,
        "METRICS",
        COMPONENT,
        model=model_tag,
        split=split,
        **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()},
    )

    stem = f"{model_tag}_{split}"
    calib = calibration_table(y_true, y_prob)
    lift = lift_table(y_true, y_prob)
    calib_path = os.path.join(run_dir, "metrics", f"{stem}_calibration.csv")
    lift_path = os.path.join(run_dir, "metrics", f"{stem}_lift.csv")
    calib.to_csv(calib_path, index=False)
    lift.to_csv(lift_path, index=False)
    log_stage(LOG, "ARTIFACT", COMPONENT, model=model_tag, calibration=calib_path, lift=lift_path)
    metrics["top_decile_capture"] = float(lift["cum_capture"].iloc[0])
    log_stage(
        LOG,
        "INFO",
        COMPONENT,
        model=model_tag,
        split=split,
        top_decile_lift=round(float(lift["lift"].iloc[0]), 3),
        top_decile_capture=round(float(lift["cum_capture"].iloc[0]), 4),
        max_calibration_gap=round(float(calib["gap"].abs().max()), 5),
    )

    return metrics


def write_summary(summary: dict, run_dir: str, LOG) -> str:
    path = os.path.join(run_dir, "metrics", "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    log_stage(LOG, "ARTIFACT", COMPONENT, summary=path)
    return path
