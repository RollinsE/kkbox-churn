"""Readable console output.

The file log (run.log) keeps the full structured lines. The console gets a
shorter rendering of the same records:

* time only (HH:MM:SS), no date / run id / project name
* "component: key value, key value" with numbers formatted for reading
  (992,931 rows, 6.39%, 4m 12s) and paths shortened to file names
* ARTIFACT and CHECKPOINT lines stay in the file only
* stage banners, plain-text tables and an end-of-run summary, written as raw
  plain text with no symbols

Everything still goes through the logger, so run.log has the same content.
"""

from __future__ import annotations

import logging
import math
import os
import sys

import pandas as pd

SEPARATOR = "=" * 80
RULE = "-" * 80
FILE_ONLY_STAGES = {"ARTIFACT", "CHECKPOINT"}
PERCENT_KEYS = ("churn_rate", "base_rate", "mean_pred", "_share", "capture", "agreement")


class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that survives a closed or detached output stream.

    Notebooks and remote sessions can drop the stdout of a long-running process.
    The run then carries on and keeps writing run.log, instead of printing
    "--- Logging error ---" for every line.
    """

    def __init__(self, stream=None):
        super().__init__(stream)
        self._stream_broken = False

    def emit(self, record):
        if self._stream_broken:
            return
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.flush()
        except (BrokenPipeError, OSError, ValueError):
            self._stream_broken = True
        except Exception:
            self._stream_broken = True

    def flush(self):
        if self._stream_broken:
            return
        try:
            super().flush()
        except (BrokenPipeError, OSError, ValueError):
            self._stream_broken = True


# --------------------------------------------------------------------------- value formatting


def format_duration(seconds) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes, secs = divmod(int(round(seconds)), 60)
        return f"{minutes}m {secs:02d}s"
    hours, rest = divmod(int(round(seconds)), 3600)
    return f"{hours}h {rest // 60:02d}m"


def _is_percent_key(key: str) -> bool:
    return any(part in key for part in PERCENT_KEYS)


def format_value(key: str, value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int) or (hasattr(value, "dtype") and getattr(value.dtype, "kind", "") in "iu"):
        return f"{int(value):,}"
    if isinstance(value, float) or (hasattr(value, "dtype") and getattr(value.dtype, "kind", "") == "f"):
        value = float(value)
        if math.isnan(value):
            return "n/a"
        if _is_percent_key(key) and 0 <= abs(value) <= 1:
            return f"{value * 100:.2f}%"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.4f}".rstrip("0").rstrip(".") if value != int(value) else f"{value:.1f}"
    text = str(value)
    if os.sep in text and len(text) > 30 and " " not in text:  # a long path: show the file name only
        return os.path.basename(text.rstrip(os.sep)) or text
    return text


def format_fields(fields: dict) -> str:
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key.replace('_', ' ')} {format_value(key, value)}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- formatter / filter


class ConsoleFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(datefmt="%H:%M:%S")

    def format(self, record):
        if getattr(record, "kk_raw", False):
            text = record.getMessage()
        else:
            text = self._render(record)
            text = f"{self.formatTime(record, self.datefmt)}  {text}"
        if record.exc_info and not getattr(record, "kk_raw", False):
            text += "\n" + self.formatException(record.exc_info)
        return text

    def _render(self, record) -> str:
        stage = getattr(record, "kk_stage", None)
        if stage is None:
            prefix = "" if record.levelno < logging.WARNING else f"{record.levelname}: "
            return prefix + record.getMessage()

        component = getattr(record, "kk_component", "")
        fields = dict(getattr(record, "kk_fields", {}) or {})
        elapsed = fields.pop("elapsed_sec", None)

        if stage == "START":
            details = format_fields(fields)
            return f"{component}: started" + (f" ({details})" if details else "")
        if stage == "DONE":
            details = format_fields(fields)
            took = f" in {format_duration(elapsed)}" if elapsed is not None else ""
            return f"{component}: done{took}" + (f" ({details})" if details else "")
        if stage == "FAIL":
            took = f" after {format_duration(elapsed)}" if elapsed is not None else ""
            return f"ERROR: {component} failed{took}"
        if stage == "WARN":
            return f"WARNING: {component}: {format_fields(fields)}"
        if stage == "EVAL":
            return f"{component}: round {fields.pop('iter', '?')}: {format_fields(fields)}"
        return f"{component}: {format_fields(fields)}"


class ConsoleFilter(logging.Filter):
    def filter(self, record):
        if getattr(record, "kk_file_only", False):
            return False
        return getattr(record, "kk_stage", None) not in FILE_ONLY_STAGES


def make_console_handler(level=logging.INFO) -> logging.Handler:
    handler = SafeStreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(ConsoleFormatter())
    handler.addFilter(ConsoleFilter())
    return handler


# --------------------------------------------------------------------------- raw blocks (banners, tables)


def raw(LOG, text: str) -> None:
    """Write a block of text as-is to the console, and to run.log."""
    LOG.info(text, extra={"kk_raw": True})


def stage_banner(LOG, title: str) -> None:
    raw(LOG, f"\n{SEPARATOR}\n{title.upper()}\n{SEPARATOR}")


def key_value_block(LOG, title: str, items: dict) -> None:
    width = max((len(k) for k in items), default=0) + 1
    lines = [title, RULE] + [f"  {(k + ':').ljust(width)} {v}" for k, v in items.items() if v is not None]
    raw(LOG, "\n".join(lines))


def format_table(df: pd.DataFrame, title: str | None = None, formats: dict | None = None) -> str:
    """Plain-text table: padded columns, no box-drawing symbols."""
    if df is None or df.empty:
        return f"{title}\n(no rows)" if title else "(no rows)"
    formats = formats or {}
    headers = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            value = row[col]
            if col in formats and value is not None and not (isinstance(value, float) and math.isnan(value)):
                cells.append(formats[col](value))
            elif value is None or (isinstance(value, float) and math.isnan(value)):
                cells.append("")
            else:
                cells.append(format_value(str(col), value) if not isinstance(value, str) else value)
        rows.append(cells)
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(values):
        return "  ".join(v.ljust(w) for v, w in zip(values, widths, strict=True)).rstrip()

    out = [] if title is None else [title]
    out += [line(headers), "  ".join("-" * w for w in widths), *(line(r) for r in rows)]
    return "\n".join(out)


def print_table(LOG, df: pd.DataFrame, title: str | None = None, formats: dict | None = None) -> None:
    raw(LOG, "\n" + format_table(df, title, formats))


def bullet_block(LOG, title: str, lines: list) -> None:
    if not lines:
        return
    raw(LOG, "\n" + title + "\n" + "\n".join(f"  - {line}" for line in lines))


def pct(value) -> str:
    return f"{float(value) * 100:.2f}%"


def num4(value) -> str:
    return f"{float(value):.4f}"


def pp(value) -> str:
    """Percentage-point difference, signed."""
    return f"{float(value) * 100:+.2f} pp"
