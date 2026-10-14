import os
import re

import pytest

from src.experiment import create_run_dir, derive_run_id, find_latest_run_dir
from src.logger import component_stage, get_logger, log_stage, setup_logger

LINE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| [A-Z]+ \| (?P<run_id>\S+) \| kkbox_churn \| ")


def _read(run_dir):
    with open(os.path.join(run_dir, "logs", "run.log"), encoding="utf-8") as f:
        return f.read().splitlines()


def test_run_dir_layout(tmp_path):
    run_id, run_dir = create_run_dir(base=str(tmp_path / "runs"))
    for sub in ("logs", "artifacts", "metrics", "plots"):
        assert os.path.isdir(os.path.join(run_dir, sub))
    assert derive_run_id(run_dir) == run_id
    assert find_latest_run_dir(str(tmp_path / "runs")) == run_dir
    assert find_latest_run_dir(str(tmp_path / "missing")) is None


def test_every_line_has_format_and_run_id(tmp_path):
    LOG = setup_logger("r1", str(tmp_path))
    log_stage(LOG, "METRICS", "LightGBM", auc=0.9, skipped=None)
    lines = _read(tmp_path)
    assert lines and all(LINE.match(line) for line in lines)
    assert all(LINE.match(line).group("run_id") == "r1" for line in lines)
    assert lines[-1].endswith("[METRICS] component=LightGBM auc=0.9")  # None kwargs are dropped


def test_component_stage_logs_start_and_done(tmp_path):
    LOG = setup_logger("r1", str(tmp_path))
    with component_stage(LOG, "Features", snapshot="mar"):
        pass
    text = "\n".join(_read(tmp_path))
    assert "[START] component=Features snapshot=mar" in text
    assert re.search(r"\[DONE\] component=Features elapsed_sec=\d+\.\d", text)


def test_component_stage_logs_fail_with_traceback_and_reraises(tmp_path):
    LOG = setup_logger("r1", str(tmp_path))
    with pytest.raises(ZeroDivisionError):
        with component_stage(LOG, "Training"):
            1 / 0  # noqa: B018
    text = "\n".join(_read(tmp_path))
    assert re.search(r"\[FAIL\] component=Training elapsed_sec=", text)
    assert "Exception in component=Training" in text
    assert "ZeroDivisionError" in text
    assert "[DONE] component=Training" not in text


def test_setup_twice_does_not_duplicate_lines(tmp_path):
    setup_logger("r1", str(tmp_path))
    LOG = setup_logger("r2", str(tmp_path))
    assert len(LOG.handlers) == 2 and len(LOG.filters) == 1
    assert get_logger() is LOG
    log_stage(LOG, "INFO", "Pipeline", marker="once")
    lines = [line for line in _read(tmp_path) if "marker=once" in line]
    assert len(lines) == 1
    assert LINE.match(lines[0]).group("run_id") == "r2"
