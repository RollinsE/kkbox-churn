# Logging

One logger per run, structured stage lines, and a context manager that times every component.
The code is in `src/logger.py` and `src/console.py`.

## One run, one folder, one log

`main.py` creates (or reuses) a run directory and calls `setup_logger(run_id, run_dir)` once.

```text
runs/run_YYYYMMDD_HHMMSS/
├── logs/run.log        every line the console shows, appended across stages
├── artifacts/          models, predictions, manifests, champion.json, resume caches
├── metrics/            summary.json, leaderboard, calibration and lift tables, SHAP importance
└── plots/              interactive figures (Plotly HTML)
```

- `--run_dir <path>` writes into a specific run. Use it to run `ingest`, `features` and `train` as
  separate commands that share one `run.log`, or to resume an interrupted run.
- `--resume` continues in the latest run under `--run_base_dir`.
- With neither flag, a new timestamped run is created.

The log file is opened in append mode, so resumed and multi-stage runs keep a single history.

## Console and file

The same records go to two places, formatted differently:

- **`run.log`** keeps the full structured lines (below). Use it for grepping and for
  the complete history, including every file path written.
- **The console** shows a shorter rendering for reading as the run goes:

```text
================================================================================
STAGE: FEATURES
================================================================================
09:16:49  Features:feb: started (cutoff 2017-02-01)
09:20:57  Features: snapshot feb, rows 992,931, features 47, churn rate 6.39%, no logs share 18.66%
09:21:01  Features:feb: done in 4m 12s
```

  The console drops the date, run id and project name. It formats numbers for reading, shortens paths to
  file names, and leaves `ARTIFACT` and `CHECKPOINT` lines in the file only. Stage banners, result tables
  and the end-of-run "Outputs written" list are plain text with no symbols.
- The console handler (`SafeStreamHandler`) survives a closed output stream: if a notebook or remote
  session drops the process's stdout, the run carries on and keeps writing `run.log`.
- When a run fails, the traceback appears once (from the innermost failing stage), followed by a
  `Pipeline failed: ...` block naming the log file.

## Line format

```text
2026-10-02 12:14:03 | INFO | 20261002_111400 | kkbox_churn | [METRICS] component=Evaluate model=logistic split=valid_mar logloss=0.218753 auc=0.926404
└─ time               └─ level └─ run id        └─ project    └─ [STAGE] component=<name> key=value ...
```

Every line carries the run id, so lines from different runs can be told apart after they are
copied, grepped or merged.

## Stages

| Stage | Meaning |
|---|---|
| `START` / `DONE` / `FAIL` | Written by `component_stage`. `DONE` and `FAIL` include `elapsed_sec`. `FAIL` is followed by the full traceback. |
| `INFO` | Something worth knowing: resolved settings, row counts, skipped work. |
| `WARN` | Recoverable problem. |
| `EVAL` | Boosting metrics (train and early-stopping holdout) every `--eval_every` rounds. CatBoost's are written after it finishes. |
| `METRICS` | Final numbers for a snapshot or a model. |
| `CHECKPOINT` | A model file was saved. |
| `ARTIFACT` | Any other file was written, with its path. |
| `RESULT` | Headline outcome of a component or of the pipeline. Per-model results go to `run.log` only; the console shows the ranking table instead. |

## Components

`Pipeline`, `Ingest`, `Ingest:<table>`, `Demo`, `Labels`, `Labels:<snapshot>`, `Features`, `Features:<snapshot>`, `Diagnostics`,
`Diagnostics:<check>`, `Training`, one per model (`Prior`, `Logistic`, `LightGBM`, `XGBoost`, `CatBoost`, and
`<Model>_tuned` with `--tune`), `FinalFit`, `Evaluate`, `Figures`.

A model that fails writes `[FAIL]` with its traceback, then `[WARN] component=Training model=<name>
skipped=...`, and the marathon carries on. A model whose library is not installed gets only the `WARN`.

## How the code uses it

The logger is created once in `main.py` and passed down explicitly as `LOG`. Modules never create
their own logger and never call `print`; a test enforces that.

```python
from src.logger import component_stage, log_stage

def run_features(con, data_dir, run_dir, LOG, ...):
    with component_stage(LOG, "Features", snapshots="feb,mar,apr"):
        ...
        log_stage(LOG, "METRICS", "Features", snapshot="mar", rows=len(df), churn_rate=0.0899)
        log_stage(LOG, "ARTIFACT", "Features", path=out_path)
```

Rules of thumb:

- Wrap each unit of work in `component_stage`, so a crash always leaves a `[FAIL]` line and a
  traceback in `run.log`.
- Use `log_stage` with keyword arguments instead of free text, so lines stay greppable.
  Keyword arguments set to `None` are left out.
- Log a path every time a file is written.

## Reading a log

```bash
grep "\[METRICS\]\|\[RESULT\]" runs/run_*/logs/run.log     # the numbers
grep "\[FAIL\]" -A 30 runs/run_*/logs/run.log               # what broke
grep "elapsed_sec" runs/run_*/logs/run.log                  # where the time went
```

## Details

1. The logger name is `kkbox_churn`.
2. Calling `setup_logger` again closes the old file handler and clears the old run-id filter, so
   re-running a setup cell in a notebook does not duplicate lines or leave files open.
3. `get_logger()` returns the run logger for code that cannot easily receive `LOG`.
4. `WARN` and `FAIL` lines are logged at WARNING and ERROR level; everything else at INFO.
5. `log_stage(..., file_only=True)` writes a line to `run.log` only, for details the console already
   shows in another form.
