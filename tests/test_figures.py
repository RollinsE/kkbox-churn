"""Results figures: specs are built from a finished run folder and say what the numbers say."""

import json
import os

import numpy as np
import pandas as pd
import pytest
from conftest import quiet_logger

import main as pipeline
from src import figures

LOG = quiet_logger()
LABELS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "results", "label_comparison.csv")


@pytest.fixture
def run(tmp_path):
    run_dir = pipeline.main(
        [
            "--mode",
            "all",
            "--demo",
            "--demo_users",
            "1500",
            "--no_plots",
            "--engine",
            "sqlite",
            "--frame_format",
            "csv",
            "--models",
            "prior,logistic,lightgbm",
            "--num_boost_round",
            "200",
            "--early_stopping_rounds",
            "20",
            "--num_leaves",
            "15",
            "--min_data_in_leaf",
            "20",
            "--seeds",
            "1",
            "--run_dir",
            str(tmp_path / "run"),
            "--data_dir",
            str(tmp_path / "data"),
        ]
    )
    return run_dir, str(tmp_path / "data")


def _specs(run, mode="light"):
    run_dir, data_dir = run
    inputs = figures._load_inputs(run_dir, data_dir, "csv", LOG)
    return inputs, figures.build_specs(inputs, mode, pd.read_csv(LABELS))


def test_every_figure_is_built_and_serialisable(run):
    pytest.importorskip("lightgbm")
    inputs, specs = _specs(run)
    assert set(specs) == {
        "model_comparison",
        "cumulative_gains",
        "calibration",
        "base_rate_shift",
        "feature_importance",
        "month_end_effect",
        "label_choice",
    }
    for name, spec in specs.items():
        json.dumps(spec)  # plain data, nothing Plotly-specific
        assert spec["layout"]["title"]["text"], name
        assert spec["layout"]["width"] == figures.WIDTH


def test_titles_and_marks_follow_the_numbers(run):
    pytest.importorskip("lightgbm")
    inputs, specs = _specs(run)
    champion = inputs["summary"]["champion"]
    y = inputs["valid"]["is_churn"].to_numpy()

    gains = specs["cumulative_gains"]
    curve = next(tr for tr in gains["data"] if tr.get("name") == champion)
    assert curve["y"][0] == 0 and curve["y"][-1] == pytest.approx(100)
    assert f"{curve['y'][10]:.0f}% of churners" in gains["layout"]["title"]["text"]

    comp = specs["model_comparison"]
    champ_trace = next(tr for tr in comp["data"] if tr["name"] == "champion")
    assert champ_trace["y"] == [champion]
    assert champ_trace["marker"]["color"] == figures.THEMES["light"]["blue"]
    assert "prior" not in sum((tr["y"] for tr in comp["data"]), [])

    base = specs["base_rate_shift"]
    actual = next(tr for tr in base["data"] if tr["name"] == "Actual churn rate")
    assert actual["y"][1] == pytest.approx(y.mean())


def test_dark_mode_uses_its_own_colours(run):
    pytest.importorskip("lightgbm")
    _, light = _specs(run, "light")
    _, dark = _specs(run, "dark")
    assert light["calibration"]["layout"]["paper_bgcolor"] == figures.THEMES["light"]["surface"]
    assert dark["calibration"]["layout"]["paper_bgcolor"] == figures.THEMES["dark"]["surface"]
    assert dark["calibration"]["data"][1]["line"]["color"] == figures.THEMES["dark"]["blue"]


def test_tie_rule_title():
    board = pd.DataFrame(
        {
            "model": ["ensemble", "xgboost", "lightgbm", "logistic", "prior"],
            "log loss": [0.1714, 0.1717, 0.1718, 0.1814, 0.2630],
            "auc": [0.884, 0.883, 0.883, 0.838, 0.5],
        }
    )
    spec = figures.model_comparison(board, "xgboost", "ensemble", 0.001, 0.2630, 862_158, "March")
    assert spec["layout"]["title"]["text"].startswith("<b>3 models are tied within 0.001")
    assert spec["layout"]["shapes"][0]["x1"] == pytest.approx(0.1724)


def test_feature_labels_are_readable():
    assert figures.feature_label("logs_secs_30d") == "Listening time, last 30 days"
    assert figures.feature_label("days_to_expire") == "Days until membership expires"
    assert figures.feature_label("something_new") == "Something new"
    assert figures.feature_source("logs_unq_7d") == "Listening"
    assert figures.feature_source("expire_on_month_end") == "Calendar"
    assert figures.feature_source("last_is_auto_renew") == "Transactions"


def test_long_titles_are_wrapped():
    text, lines = figures._wrap("word " * 80, 19)
    assert lines > 1 and text.count("<br>") == lines - 1


def test_predictions_must_match_the_data(run, tmp_path):
    pytest.importorskip("lightgbm")
    run_dir, data_dir = run
    path = os.path.join(run_dir, "artifacts", "marathon", "logistic_valid_pred.npy")
    np.save(path, np.zeros(5))
    with pytest.raises(ValueError):
        figures._load_inputs(run_dir, data_dir, "csv", LOG)


def test_make_figures_script_writes_html_and_png(run):
    pytest.importorskip("plotly")
    pytest.importorskip("lightgbm")
    import scripts.make_figures as make

    run_dir, data_dir = run
    export = os.path.join(run_dir, "export")
    written = make.main(
        [
            "--run_dir",
            run_dir,
            "--data_dir",
            data_dir,
            "--frame_format",
            "csv",
            "--export_dir",
            export,
            "--label_comparison",
            LABELS,
        ]
    )
    assert os.path.exists(os.path.join(run_dir, "plots", "cumulative_gains.html"))
    assert (
        any(k.endswith("_dark.png") for k in written)
        or "[WARN] component=Figures png=skipped"
        in open(os.path.join(run_dir, "logs", "run.log"), encoding="utf-8").read()
    )
