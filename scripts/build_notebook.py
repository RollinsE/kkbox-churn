"""Generate notebooks/colab.ipynb, a thin Google Colab driver for main.py.

The notebook only provides storage, credentials and the commands; all logic lives in
main.py and src/. It is generated rather than edited by hand:

    python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


CELLS = [
    md("""
# KKBox churn prediction on Google Colab

Runs the pipeline from `main.py`, one stage per cell. Every stage writes to the same run folder and
`run.log`, and the run folder name is fixed, so re-running a cell after a disconnect resumes it.

Leave `DEMO = True` for a first pass: it runs everything on synthetic data in a few minutes and needs
no Kaggle account.
"""),
    md("## 1. Settings"),
    code("""
DEMO = True  # False = real KKBox data from Kaggle
REPO_URL = "https://github.com/RollinsE/kkbox-churn.git"
CODE_DIR = "/content/kkbox-churn"

# Processed data and runs go to Drive for real runs, so they survive a disconnect. The raw CSVs
# (about 35 GB extracted) stay on the local disk.
PROJECT_DIR = "/content/kkbox_demo" if DEMO else "/content/drive/MyDrive/kkbox_churn"
RAW_DIR = "/content/raw"
DATA_DIR = f"{PROJECT_DIR}/data"
RUN_DIR = f"{PROJECT_DIR}/runs/run_main"  # fixed: re-running a stage resumes this run

COMMON = f"--run_dir {RUN_DIR} --data_dir {DATA_DIR} --raw_dir {RAW_DIR}" + (" --demo" if DEMO else "")
"""),
    md("""
## 2. Code and dependencies

Clones the repository, or unzips `kkbox-churn.zip` if you uploaded one to `/content`. Run this cell
again after every reconnect.
"""),
    code("""
import os

if not DEMO:
    from google.colab import drive

    drive.mount("/content/drive")

if not os.path.isdir(CODE_DIR):
    if os.path.exists("/content/kkbox-churn.zip"):
        !unzip -q -o /content/kkbox-churn.zip -d /content
    else:
        !git clone -q {REPO_URL} {CODE_DIR}
%cd {CODE_DIR}
!pip install -q -r requirements-dev.txt
"""),
    md("""
## 3. Health check and tests
"""),
    code("""
!python scripts/doctor.py
!python -m pytest
"""),
    md("""
## 4. Kaggle credentials (real runs only)

Create an API token on Kaggle (Settings, API), add it as `KAGGLE_API_TOKEN` under the key icon
(Secrets) in the Colab sidebar, and accept the competition rules on the Kaggle site once. The last
line should list the competition files.
"""),
    code("""
if not DEMO:
    from google.colab import userdata

    for var in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        os.environ.pop(var, None)
    os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
    !kaggle competitions files -c kkbox-churn-prediction-challenge
"""),
    md("""
## 5. Ingest

Kaggle download, 7z extraction, CSV to Parquet with DuckDB. Tables that already exist are skipped.
"""),
    code("""
!python main.py --mode ingest {COMMON} --memory_limit 8GB --delete_csv
"""),
    md("""
## 6. Labels

Rebuilds labels with the SQL port of the organisers' labelling script (January 2017, and February 2017
for comparison with `train.csv`). Used by the diagnostics; training uses the official labels.
"""),
    code("""
!python main.py --mode labels {COMMON} --memory_limit 8GB
"""),
    md("""
## 7. Features

One SQL query per snapshot. Each reads only rows dated before that snapshot's cutoff.
"""),
    code("""
!python main.py --mode features {COMMON} --memory_limit 8GB
"""),
    md("""
## 8. Diagnostics (optional)

Five checks on where accuracy is lost between months. Each prints a table and a short note; the full
report is saved as `metrics/diagnostics_report.md`.
"""),
    code("""
!python main.py --mode diagnose {COMMON} --memory_limit 8GB
"""),
    md("""
## 9. Train

Trains every model on February, scores it on March, refits the champion on both months and predicts
April. On Colab's free CPU this takes about 30 minutes. If the session drops, run cells 1 and 2, then
this cell again: finished models and final-fit seeds are reused.
"""),
    code("""
TRAIN_FLAGS = "--num_leaves 15 --min_data_in_leaf 20 --num_boost_round 500 --catboost_iterations 500" if DEMO else ""
!python main.py --mode train {COMMON} {TRAIN_FLAGS}
"""),
    md("""
## 10. Results
"""),
    code("""
from src.notebook_viewer import display_run

display_run(RUN_DIR)
"""),
    md("""
## 11. Figures as PNG (optional)

Light and dark PNGs of every figure, for the README. Kaleido needs a headless Chrome and its system
libraries; the first two lines install them (a minute or two).
"""),
    code("""
!for p in libnss3 libatk-bridge2.0-0t64 libcups2t64 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2t64; do apt-get install -y -qq "$p" > /dev/null 2>&1 || echo "could not install $p"; done
!plotly_get_chrome -y > /dev/null
!python scripts/make_figures.py --run_dir {RUN_DIR} --data_dir {DATA_DIR} --export_dir {PROJECT_DIR}/figures
"""),
]


def build() -> dict:
    return {
        "cells": CELLS,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
