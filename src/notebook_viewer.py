"""Notebook helper for viewing a finished run.

    from src.notebook_viewer import display_run
    display_run(RUN_DIR)

Shows the model ranking, the champion, the diagnostics report and the plots that
the run saved, without re-running anything.
"""

from __future__ import annotations

import json
import os


def display_run(run_dir: str, show_plots: bool = True, plot_width: int = 520) -> None:
    import pandas as pd
    from IPython.display import Image, Markdown, display

    metrics = os.path.join(run_dir, "metrics")
    shown = False

    leaderboard = os.path.join(metrics, "leaderboard.csv")
    if os.path.exists(leaderboard):
        display(Markdown("## Model ranking (validation month)"))
        display(pd.read_csv(leaderboard))
        shown = True

    summary = os.path.join(metrics, "summary.json")
    if os.path.exists(summary):
        with open(summary, encoding="utf-8") as f:
            s = json.load(f)
        lines = [f"**Champion:** {s.get('champion')}", f"**Submission (as trained):** `{s.get('submission')}`"]
        if s.get("submission_recalibrated"):
            lines.append(
                f"**Submission (recalibrated to {s['calibration_target'] * 100:.2f}%):** "
                f"`{s['submission_recalibrated']}`"
            )
        for kind, why in {**s.get("skipped", {}), **s.get("failed", {})}.items():
            lines.append(f"**Not run: {kind}:** {why}")
        display(Markdown("  \n".join(lines)))

    report = os.path.join(metrics, "diagnostics_report.md")
    if os.path.exists(report):
        with open(report, encoding="utf-8") as f:
            display(Markdown(f.read()))
        shown = True

    plots = os.path.join(run_dir, "plots")
    if show_plots and os.path.isdir(plots):
        from IPython.display import HTML

        for name in sorted(os.listdir(plots)):
            path = os.path.join(plots, name)
            if name.endswith(".html"):  # interactive Plotly figures from src/figures.py
                with open(path, encoding="utf-8") as f:
                    display(HTML(f.read()))
                shown = True
            elif name.endswith(".png"):  # diagnostics charts
                display(Markdown(f"**{name}**"))
                display(Image(path, width=plot_width))
                shown = True

    if not shown:
        display(Markdown(f"Nothing to show yet in `{run_dir}`. Run `--mode train` or `--mode diagnose` first."))
