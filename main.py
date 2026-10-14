"""Command-line entry point for the KKBox churn pipeline.

    python main.py --mode all --demo                 # whole pipeline on synthetic data
    python main.py --mode train --run_dir runs/run_a # one stage into an existing run folder

Modes: ingest, labels, features, diagnose, train, all. Every mode writes into one run folder
(logs, artifacts, metrics, plots); see README.md and docs/LOGGING.md.
"""

import argparse
import os
import sys

from src.console import SEPARATOR, key_value_block, raw, stage_banner
from src.constants import (
    DEFAULT_POPULATION,
    FEATURE_SNAPSHOTS,
    POPULATIONS,
    REBUILT_SNAPSHOTS,
    SNAPSHOTS,
    TEST_SNAPSHOT,
    TRAIN_SNAPSHOTS,
    VALID_SNAPSHOT,
)
from src.db import ENGINES, open_source
from src.diagnostics import CHECKS, run_diagnostics
from src.experiment import create_run_dir, derive_run_id, ensure_run_subdirs, find_latest_run_dir
from src.features import run_features
from src.ingest import TABLE_ORDER, run_ingest
from src.labeller import run_labels
from src.logger import component_stage, log_stage, setup_logger
from src.model import DEFAULT_LGB_PARAMS, DEFAULT_MODELS, DEFAULT_TUNE_MODELS, parse_model_list, run_training
from src.storage import FRAME_FORMATS

MODES = ["ingest", "labels", "features", "diagnose", "train", "all"]


def parse_args(argv=None):
    p = argparse.ArgumentParser("KKBox Churn Pipeline")

    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    p.add_argument(
        "--run_base_dir",
        default="runs",
        help="Base folder for timestamped runs: logs, artifacts, metrics and plots.",
    )
    p.add_argument(
        "--run_dir",
        default=None,
        help="Use a specific run directory, e.g. runs/run_YYYYMMDD_HHMMSS.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue in the latest run directory under --run_base_dir (appends to its run.log).",
    )

    p.add_argument(
        "--raw_dir",
        default="data/raw",
        help="Kaggle archives and extracted CSVs (about 40 GB; keep on fast local disk).",
    )
    p.add_argument("--data_dir", default="data/processed", help="Parquet tables and feature tables.")
    p.add_argument(
        "--engine",
        default="duckdb",
        choices=list(ENGINES),
        help="duckdb for real data. sqlite needs no install and is meant for --demo and tests.",
    )
    p.add_argument(
        "--frame_format",
        default="parquet",
        choices=sorted(FRAME_FORMATS),
        help="Feature table format. csv avoids the pyarrow dependency.",
    )

    p.add_argument("--demo", action="store_true", help="Use synthetic data instead of downloading from Kaggle.")
    p.add_argument("--demo_users", type=int, default=5000)

    p.add_argument("--skip_download", action="store_true", help="Fail instead of calling the Kaggle CLI.")
    p.add_argument("--delete_csv", action="store_true", help="Delete extracted CSVs once Parquet is written.")
    p.add_argument("--overwrite", action="store_true", help="Rebuild every Parquet table, even if it exists.")
    p.add_argument(
        "--tables",
        default=None,
        help="Ingest: rebuild only these tables (downloads only their raw files): " + ", ".join(TABLE_ORDER),
    )
    p.add_argument("--memory_limit", default="8GB", help="DuckDB memory limit.")
    p.add_argument("--threads", type=int, default=None, help="DuckDB threads (default: all cores).")

    p.add_argument(
        "--snapshots",
        default=",".join(FEATURE_SNAPSHOTS),
        help="Snapshots to build features for. Available: " + ", ".join(SNAPSHOTS),
    )
    p.add_argument(
        "--label_snapshots",
        default=",".join(REBUILT_SNAPSHOTS),
        help="Snapshots whose labels --mode labels rebuilds with the organisers' rule.",
    )
    p.add_argument(
        "--train_snapshots",
        default=",".join(TRAIN_SNAPSHOTS),
        help="Labelled snapshots to train on, e.g. feb or jan,feb.",
    )
    p.add_argument("--valid_snapshot", default=VALID_SNAPSHOT)
    p.add_argument("--test_snapshot", default=TEST_SNAPSHOT)
    p.add_argument(
        "--population",
        default=DEFAULT_POPULATION,
        choices=list(POPULATIONS),
        help="Which label-file users to train and validate on. expiry: last expiry before the cutoff "
        "falls in the label month (closest to the April test users). labeller: the organisers' "
        "script's rule. all: everyone.",
    )
    p.add_argument(
        "--holdout_share", type=float, default=0.2, help="Share of training users held out for LightGBM early stopping."
    )

    p.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated models for the marathon. Available: "
        + ", ".join(DEFAULT_MODELS)
        + ". A model whose library is not installed is skipped with a warning.",
    )
    p.add_argument(
        "--no_ensemble",
        action="store_true",
        help="Do not score the ensemble (average of the best model from each family).",
    )
    p.add_argument(
        "--tune",
        action="store_true",
        help="Also run tuned versions (Optuna, judged on the training holdout) as <model>_tuned.",
    )
    p.add_argument("--tune_models", default=",".join(DEFAULT_TUNE_MODELS), help="Models to tune when --tune is set.")
    p.add_argument("--tune_trials", type=int, default=15)
    p.add_argument("--tune_rows", type=int, default=300_000, help="Training rows sampled for each tuning trial.")
    p.add_argument("--catboost_iterations", type=int, default=3000)
    p.add_argument(
        "--champion_tolerance",
        type=float,
        default=0.001,
        help="Models within this much log loss of the best count as tied; the quickest to train among "
        "them is the champion. 0 = always the lowest log loss.",
    )
    p.add_argument(
        "--ensemble_seeds",
        type=int,
        default=1,
        help="Seeds per member when the ensemble is refitted (a single-model champion uses --seeds).",
    )
    p.add_argument(
        "--calibration_target",
        type=float,
        default=None,
        help="Mean churn rate for submission_recalibrated.csv. Default: the validation month's rate.",
    )
    p.add_argument("--seeds", default="42,43,44", help="Seeds for the final seed-averaged fit.")
    p.add_argument("--num_boost_round", type=int, default=3000)
    p.add_argument("--early_stopping_rounds", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=50, help="Log boosting metrics every N rounds.")
    p.add_argument("--learning_rate", type=float, default=DEFAULT_LGB_PARAMS["learning_rate"])
    p.add_argument("--num_leaves", type=int, default=DEFAULT_LGB_PARAMS["num_leaves"])
    p.add_argument("--min_data_in_leaf", type=int, default=DEFAULT_LGB_PARAMS["min_data_in_leaf"])
    p.add_argument(
        "--round_multiplier", type=float, default=1.1, help="Final fit uses best_iteration * this many rounds."
    )
    p.add_argument("--logistic_max_rows", type=int, default=300_000)
    p.add_argument("--no_plots", action="store_true")

    p.add_argument(
        "--checks", default=",".join(CHECKS), help="Diagnostics to run with --mode diagnose: " + ", ".join(CHECKS)
    )
    p.add_argument(
        "--diag_rows", type=int, default=400_000, help="Rows sampled per month for the within/across-month check."
    )

    return p.parse_args(argv)


def resolve_run_dir(args):
    if args.run_dir:
        run_dir = args.run_dir
        ensure_run_subdirs(run_dir)
        return derive_run_id(run_dir), run_dir

    if args.resume:
        latest = find_latest_run_dir(args.run_base_dir)
        if not latest:
            print(f"[FATAL] --resume set but no run_ folders under {args.run_base_dir}")
            sys.exit(1)
        ensure_run_subdirs(latest)
        return derive_run_id(latest), latest

    return create_run_dir(base=args.run_base_dir)


def main(argv=None):
    args = parse_args(argv)
    run_id, run_dir = resolve_run_dir(args)
    LOG = setup_logger(run_id, run_dir, log_level=args.log_level)
    key_value_block(
        LOG,
        "KKBox churn pipeline",
        {
            "mode": args.mode + (" (demo data)" if args.demo else ""),
            "run": os.path.basename(run_dir.rstrip("/")),
            "run folder": run_dir,
            "data folder": args.data_dir,
            "log file": os.path.join(run_dir, "logs", "run.log"),
        },
    )

    try:
        with component_stage(
            LOG,
            "Pipeline",
            mode=args.mode,
            run_id=run_id,
            run_dir=run_dir,
            resume=args.resume,
            demo=args.demo,
            engine=args.engine,
        ):
            models = parse_model_list(args.models)
            tune_models = parse_model_list(args.tune_models, flag="--tune_models") if args.tune else []
            seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
            snapshots = [s.strip() for s in args.snapshots.split(",") if s.strip()]
            label_snapshots = [s.strip() for s in args.label_snapshots.split(",") if s.strip()]
            train_snapshots = [s.strip() for s in args.train_snapshots.split(",") if s.strip()]
            tables = [t.strip() for t in args.tables.split(",") if t.strip()] if args.tables else None
            unknown = [
                s
                for s in snapshots + label_snapshots + train_snapshots + [args.valid_snapshot, args.test_snapshot]
                if s not in SNAPSHOTS
            ]
            if unknown:
                raise ValueError(f"Unknown snapshots {unknown}. Choose from {list(SNAPSHOTS)}.")

            if args.mode in ("ingest", "all"):
                stage_banner(LOG, "Stage: ingest")
                manifest = run_ingest(
                    raw_dir=args.raw_dir,
                    data_dir=args.data_dir,
                    run_dir=run_dir,
                    LOG=LOG,
                    engine=args.engine,
                    demo=args.demo,
                    demo_users=args.demo_users,
                    seed=seeds[0],
                    download=not args.skip_download,
                    delete_csv=args.delete_csv,
                    memory_limit=args.memory_limit,
                    threads=args.threads,
                    overwrite=args.overwrite,
                    tables=tables,
                )
                log_stage(
                    LOG,
                    "RESULT",
                    "Pipeline",
                    mode="ingest",
                    tables=len(manifest),
                    rows=sum(t["rows"] for t in manifest.values()),
                )

            if args.mode in ("labels", "all"):
                stage_banner(LOG, "Stage: labels")
                con = open_source(args.engine, args.data_dir, LOG, memory_limit=args.memory_limit, threads=args.threads)
                try:
                    run_labels(con, args.data_dir, run_dir, LOG, engine=args.engine, snapshots=label_snapshots)
                finally:
                    con.close()
                log_stage(LOG, "RESULT", "Pipeline", mode="labels", snapshots=",".join(label_snapshots))

            if args.mode in ("features", "all"):
                stage_banner(LOG, "Stage: features")
                con = open_source(args.engine, args.data_dir, LOG, memory_limit=args.memory_limit, threads=args.threads)
                try:
                    paths = run_features(
                        con,
                        args.data_dir,
                        run_dir,
                        LOG,
                        snapshots=snapshots,
                        frame_format=args.frame_format,
                        engine=args.engine,
                        strict=args.snapshots != ",".join(FEATURE_SNAPSHOTS),
                    )
                finally:
                    con.close()
                log_stage(LOG, "RESULT", "Pipeline", mode="features", snapshots=",".join(paths))

            if args.mode == "diagnose":
                stage_banner(LOG, "Stage: diagnostics")
                checks = [c.strip() for c in args.checks.split(",") if c.strip()]
                unknown_checks = [c for c in checks if c not in CHECKS]
                if unknown_checks:
                    raise ValueError(f"Unknown checks {unknown_checks}. Choose from {list(CHECKS)}.")
                con = (
                    open_source(args.engine, args.data_dir, LOG, memory_limit=args.memory_limit, threads=args.threads)
                    if {"labels", "truncation"} & set(checks)
                    else None
                )
                try:
                    results = run_diagnostics(
                        con,
                        args.data_dir,
                        run_dir,
                        LOG,
                        frame_format=args.frame_format,
                        checks=checks,
                        sample_rows=args.diag_rows,
                        seed=seeds[0],
                        population=args.population,
                        plots=not args.no_plots,
                    )
                finally:
                    if con is not None:
                        con.close()
                failed = [k for k in checks if "error" in results.get(k, {})]
                log_stage(
                    LOG,
                    "RESULT",
                    "Pipeline",
                    mode="diagnose",
                    checks_run=len(checks) - len(failed),
                    checks_failed=",".join(failed) or None,
                    report=results["report"],
                )

            if args.mode in ("train", "all"):
                stage_banner(LOG, "Stage: training")
                summary = run_training(
                    data_dir=args.data_dir,
                    run_dir=run_dir,
                    LOG=LOG,
                    models=models,
                    frame_format=args.frame_format,
                    lgb_params={
                        "learning_rate": args.learning_rate,
                        "num_leaves": args.num_leaves,
                        "min_data_in_leaf": args.min_data_in_leaf,
                    },
                    num_boost_round=args.num_boost_round,
                    early_stopping_rounds=args.early_stopping_rounds,
                    eval_every=args.eval_every,
                    seeds=seeds,
                    round_multiplier=args.round_multiplier,
                    logistic_max_rows=args.logistic_max_rows,
                    plots=not args.no_plots,
                    train_snapshots=train_snapshots,
                    valid_snapshot=args.valid_snapshot,
                    test_snapshot=args.test_snapshot,
                    population=args.population,
                    holdout_share=args.holdout_share,
                    catboost_iterations=args.catboost_iterations,
                    tune_models=tune_models,
                    tune_trials=args.tune_trials,
                    tune_rows=args.tune_rows,
                    ensemble=not args.no_ensemble,
                    calibration_target=args.calibration_target,
                    champion_tolerance=args.champion_tolerance,
                    ensemble_seeds=args.ensemble_seeds,
                )
                champion = summary["champion"]
                log_stage(
                    LOG,
                    "RESULT",
                    "Pipeline",
                    mode="train",
                    champion=champion,
                    valid_logloss=round(summary["models"][champion]["valid"]["logloss"], 6),
                    submission=summary["submission"],
                )

            log_stage(LOG, "DONE", "Pipeline", mode=args.mode, status="success")

    except Exception as exc:
        raw(
            LOG,
            f"\n{SEPARATOR}\nPipeline failed: {type(exc).__name__}: {exc}\n"
            f"Full traceback: {os.path.join(run_dir, 'logs', 'run.log')}\n{SEPARATOR}",
        )
        sys.exit(1)

    _print_outputs(LOG, run_dir)
    return run_dir


def _print_outputs(LOG, run_dir):
    """End-of-run summary of what was written, then the completion line."""
    lines = ["", "Outputs written:"]
    key_files = [
        ("results table", os.path.join("metrics", "leaderboard.csv")),
        ("diagnostics report", os.path.join("metrics", "diagnostics_report.md")),
        ("labels summary", os.path.join("metrics", "labels_summary.csv")),
        ("submission", os.path.join("artifacts", "submission.csv")),
        ("submission, recalibrated", os.path.join("artifacts", "submission_recalibrated.csv")),
        ("summary", os.path.join("metrics", "summary.json")),
    ]
    for label, rel in key_files:
        if os.path.exists(os.path.join(run_dir, rel)):
            lines.append(f"  - {label}: {rel}")
    for sub in ("metrics", "plots", "artifacts"):
        folder = os.path.join(run_dir, sub)
        count = len(os.listdir(folder)) if os.path.isdir(folder) else 0
        lines.append(f"  - {sub}/: {count} files")
    lines.append(f"  - run folder: {run_dir}")
    lines += ["", SEPARATOR, "Pipeline completed successfully", SEPARATOR]
    raw(LOG, "\n".join(lines))


if __name__ == "__main__":
    main()
