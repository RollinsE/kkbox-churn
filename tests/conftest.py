import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def quiet_logger():
    """A logger for unit tests that do not need a run directory."""
    log = logging.getLogger("kkbox_churn_test")
    log.handlers.clear()
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return log
