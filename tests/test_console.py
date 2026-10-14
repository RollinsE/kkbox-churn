"""Console output: readable, symbol-free, and never breaks the run."""

import io
import logging
import sys

import pandas as pd

from src.console import SafeStreamHandler, format_duration, format_table, format_value, stage_banner
from src.logger import component_stage, log_stage, setup_logger

BANNED_SYMBOLS = ("✓", "✗", "▶", "↻", "—")


def _run_with_console(tmp_path, fn):
    """Run fn(LOG) with stdout captured; returns (console text, run.log text)."""
    real_stdout, buf = sys.stdout, io.StringIO()
    sys.stdout = buf
    try:
        LOG = setup_logger("r1", str(tmp_path))
        fn(LOG)
    finally:
        sys.stdout = real_stdout
    with open(tmp_path / "logs" / "run.log", encoding="utf-8") as f:
        return buf.getvalue(), f.read()


def test_value_formatting():
    assert format_value("rows", 992931) == "992,931"
    assert format_value("churn_rate", 0.06392) == "6.39%"
    assert format_value("learning_rate", 0.05) == "0.05"
    assert format_value("logloss", 0.201838) == "0.2018"
    assert format_value("path", "/home/user/projects/kkbox_churn/runs/run_1/metrics/x.csv") == "x.csv"
    assert format_duration(4.2) == "4.2s"
    assert format_duration(252.2) == "4m 12s"
    assert format_duration(3725) == "1h 02m"


def test_console_is_short_and_file_keeps_structured_lines(tmp_path):
    def body(LOG):
        with component_stage(LOG, "Features:feb", cutoff="2017-02-01"):
            log_stage(LOG, "METRICS", "Features", rows=992931, churn_rate=0.06392)
            log_stage(LOG, "ARTIFACT", "Features", path="/very/long/path/to/features_feb.parquet")

    console, filetext = _run_with_console(tmp_path, body)
    assert "Features: rows 992,931, churn rate 6.39%" in console
    assert "Features:feb: started (cutoff 2017-02-01)" in console
    assert "Features:feb: done in" in console
    assert "ARTIFACT" not in console and "features_feb.parquet" not in console
    assert "Logger initialized" not in console and "| kkbox_churn |" not in console
    # run.log is unchanged in style and complete
    assert "| INFO | r1 | kkbox_churn | [METRICS] component=Features rows=992931 churn_rate=0.06392" in filetext
    assert "[ARTIFACT] component=Features path=/very/long/path/to/features_feb.parquet" in filetext
    assert "Logger initialized" in filetext


def test_banners_and_tables_use_plain_separators(tmp_path):
    def body(LOG):
        stage_banner(LOG, "Stage: training")
        from src.console import print_table

        print_table(LOG, pd.DataFrame({"model": ["lightgbm"], "log loss": [0.2018]}), "Model ranking")

    console, filetext = _run_with_console(tmp_path, body)
    assert "STAGE: TRAINING" in console
    assert "Model ranking\nmodel     log loss\n--------  --------\nlightgbm  0.2018" in console
    assert "Model ranking" in filetext
    for symbol in BANNED_SYMBOLS:
        assert symbol not in console


def test_traceback_is_printed_once_for_nested_failures(tmp_path):
    def body(LOG):
        try:
            with component_stage(LOG, "Pipeline"):
                with component_stage(LOG, "Training"):
                    raise ValueError("boom")
        except ValueError:
            pass

    console, filetext = _run_with_console(tmp_path, body)
    assert filetext.count("Traceback (most recent call last)") == 1
    assert console.count("Traceback (most recent call last)") == 1
    assert "ERROR: Training failed after" in console and "ERROR: Pipeline failed after" in console


def test_format_table_handles_missing_values():
    text = format_table(pd.DataFrame({"a": [1, None], "b": ["x", "y"]}))
    assert text.splitlines()[0].startswith("a")
    assert len(text.splitlines()) == 4


class BrokenStream:
    def write(self, _message):
        raise OSError("transport endpoint is not connected")

    def flush(self):
        raise OSError("transport endpoint is not connected")


def test_safe_stream_handler_survives_a_broken_stream():
    logger = logging.getLogger("test_safe_stream_handler")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = SafeStreamHandler(BrokenStream())
    logger.addHandler(handler)
    logger.info("this should not raise or print a logging traceback")
    logger.info("subsequent messages should also be ignored cleanly")
    assert handler._stream_broken is True


def test_messages_containing_paths_are_not_shortened():
    from src.console import format_value

    assert format_value("path", "/home/user/projects/kkbox_churn/runs/run_a/plots/x.html") == "x.html"
    msg = "BrowserFailedError: could not start /root/.local/share/choreographer/deps/chrome-linux64/chrome."
    assert format_value("reason", msg) == msg
