"""Run logger for the KKBox churn pipeline.

One console + file logger per run, structured ``[STAGE] component=... key=value``
lines, and a context manager that times each pipeline stage.

run.log keeps the full structured lines. The console renders the same records
in a shorter, readable form (src/console.py).

Usage
-----
    from src.logger import setup_logger, log_stage, component_stage

    run_id, run_dir = create_run_dir(base="runs")
    LOG = setup_logger(run_id, run_dir)

    with component_stage(LOG, "Features", snapshot="mar", cutoff="2017-03-01"):
        build_features(...)
        log_stage(LOG, "ARTIFACT", "Features", path=out_path, rows=n_rows)

    log_stage(LOG, "METRICS", "Model", split="valid_mar", logloss=0.1034, auc=0.987)
"""

import logging
import os
import time
from contextlib import contextmanager

from src.console import make_console_handler

LOGGER_NAME = "kkbox_churn"


def setup_logger(run_id: str, run_dir: str, log_level: str = "INFO"):
    """
    Console + file logger with consistent formatting.

    Writes to <run_dir>/logs/run.log in append mode, so separate commands and
    resumed runs continue the same file.
    """
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "run.log")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    for handler in list(logger.handlers):
        handler.close()
    logger.handlers.clear()
    logger.filters.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt=f"%(asctime)s | %(levelname)s | %(run_id)s | {LOGGER_NAME} | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class RunIdFilter(logging.Filter):
        def filter(self, record):
            record.run_id = run_id
            return True

    logger.addFilter(RunIdFilter())

    logger.addHandler(make_console_handler(logger.level))

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    file_only = {"kk_file_only": True}
    logger.info("=" * 70, extra=file_only)
    logger.info(f"Logger initialized. log_path={log_path}", extra=file_only)
    logger.info("=" * 70, extra=file_only)
    return logger


def get_logger():
    """Return the run logger from any module without passing it around.

    Call setup_logger() once at the entry point (CLI / notebook) first.
    """
    return logging.getLogger(LOGGER_NAME)


def log_stage(LOG, stage: str, component: str, file_only: bool = False, **kwargs):
    """
    Standard structured log line.

    stage: START | INFO | WARN | DONE | FAIL | EVAL | CHECKPOINT | ARTIFACT | METRICS | RESULT
    component: Pipeline | Ingest | Labels | Features | Diagnostics | Training | <model> | FinalFit | Evaluate
    file_only: write to run.log but keep it off the console (it is already shown another way).
    """
    fields = {k: v for k, v in kwargs.items() if v is not None}
    extras = " ".join(f"{k}={v}" for k, v in fields.items())
    msg = f"[{stage}] component={component}"
    if extras:
        msg += f" {extras}"
    level = logging.WARNING if stage == "WARN" else logging.ERROR if stage == "FAIL" else logging.INFO
    LOG.log(
        level, msg, extra={"kk_stage": stage, "kk_component": component, "kk_fields": fields, "kk_file_only": file_only}
    )


@contextmanager
def component_stage(LOG, component: str, **kwargs):
    """
    Context manager that logs:
      [START] component=...
      [DONE]  component=... elapsed_sec=...
    and on exception:
      [FAIL]  component=... elapsed_sec=...
      plus stack trace.

    The traceback is written once, by the innermost stage that saw the error;
    outer stages only add their own [FAIL] line.
    """
    log_stage(LOG, "START", component, **kwargs)
    t0 = time.time()
    try:
        yield
    except Exception as exc:
        log_stage(LOG, "FAIL", component, elapsed_sec=round(time.time() - t0, 1))
        if not getattr(exc, "_kk_logged", False):
            LOG.exception(f"Exception in component={component}")
            try:
                exc._kk_logged = True
            except Exception:
                pass
        raise
    else:
        log_stage(LOG, "DONE", component, elapsed_sec=round(time.time() - t0, 1))
