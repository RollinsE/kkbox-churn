"""Build the results figures for a finished run, without retraining.

    python scripts/make_figures.py --run_dir runs/run_marathon_02 --data_dir data/processed \
        --export_dir docs/figures

Writes interactive HTML into <run_dir>/plots/ and, with --export_dir, a light and a
dark PNG of each figure for the README. Needs the same data folder the run used,
because the figures pair the run's saved predictions with the validation labels.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.console import SEPARATOR, raw  # noqa: E402
from src.experiment import derive_run_id, ensure_run_subdirs  # noqa: E402
from src.figures import build_figures  # noqa: E402
from src.logger import setup_logger  # noqa: E402
from src.storage import FRAME_FORMATS  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser("KKBox churn figures")
    p.add_argument("--run_dir", required=True, help="A finished training run.")
    p.add_argument("--data_dir", default="data/processed", help="The data folder the run used.")
    p.add_argument("--frame_format", default="parquet", choices=sorted(FRAME_FORMATS))
    p.add_argument("--export_dir", default=None, help="Also write light and dark PNGs here, e.g. docs/figures.")
    p.add_argument(
        "--label_comparison",
        default=os.path.join(ROOT, "docs", "results", "label_comparison.csv"),
        help="CSV of validation log loss per training-label choice (docs/results/label_comparison.csv).",
    )
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    if not os.path.exists(os.path.join(args.run_dir, "metrics", "summary.json")):
        sys.exit(f"{args.run_dir} has no metrics/summary.json. Point --run_dir at a finished training run.")
    ensure_run_subdirs(args.run_dir)
    LOG = setup_logger(derive_run_id(args.run_dir), args.run_dir)
    written = build_figures(
        args.run_dir,
        args.data_dir,
        LOG,
        frame_format=args.frame_format,
        export_dir=args.export_dir,
        label_comparison=args.label_comparison,
    )
    raw(
        LOG,
        "\nFigures written:\n"
        + "\n".join(f"  - {path}" for path in written.values())
        + f"\n\n{SEPARATOR}\nFigures completed\n{SEPARATOR}",
    )
    return written


if __name__ == "__main__":
    main()
