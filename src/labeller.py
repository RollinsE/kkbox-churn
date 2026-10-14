"""The organisers' churn labeller (WSDMChurnLabeller.scala), rewritten as portable SQL.

The Scala/Spark script decides, for one label month M:

1. Who is labelled. Take each user's transactions in the month BEFORE M only
   (the "history" window). Sort them with the script's ordering (below) and take
   the last one's membership_expire_date. Users whose last expiry falls inside M
   are the month's candidates.
2. Whether they churned. Walk through every transaction after the history window
   in the same ordering. A cancellation whose expiry is earlier than the current
   expiry pulls the expiry earlier. The first non-cancel transaction is the
   renewal: gap = renewal date - (possibly earlier) expiry. gap < 30 is a renewal,
   anything else is churn. No later transaction at all, or only cancellations,
   is churn.

The script's ordering (its comparator, turned into sort keys):
  * by transaction_date, ascending
  * same day, different plans: by the text price||days||method, DESCENDING
    (string comparison, so "993041" sorts after "1493041")
  * same day, same plan: renewals before cancellations;
    renewals by expiry ascending, cancellations by expiry descending
So on a day with a renewal and a cancellation of the same plan, the cancellation
counts as the last transaction.

All dates are integer days since 1970-01-01 (see src/constants.py), so date
differences are plain subtraction. The SQL runs unchanged on DuckDB and SQLite.
tests/reference_labeller.py is a line-by-line Python transliteration of the
Scala, used to check this SQL on random data.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass

from src.constants import from_days, to_days

RENEWAL_GAP_DAYS = 30

# Plan signature exactly as the script builds it: string concatenation of three CSV fields.
SIGNATURE = (
    "(CAST({p}plan_list_price AS VARCHAR) || CAST({p}payment_plan_days AS VARCHAR) "
    "|| CAST({p}payment_method_id AS VARCHAR))"
)


def _sig(prefix: str = "") -> str:
    return SIGNATURE.format(p=prefix)


def order_ascending(prefix: str = "") -> str:
    """The script's sort order, first transaction first."""
    p = prefix
    return (
        f"{p}transaction_date ASC, {_sig(p)} DESC, {p}is_cancel ASC, "
        f"CASE WHEN {p}is_cancel = 1 THEN -{p}membership_expire_date ELSE {p}membership_expire_date END ASC"
    )


def order_last_first(prefix: str = "") -> str:
    """The exact reverse of order_ascending: row 1 is the script's `orderedList.last`."""
    p = prefix
    return (
        f"{p}transaction_date DESC, {_sig(p)} ASC, {p}is_cancel DESC, "
        f"CASE WHEN {p}is_cancel = 1 THEN {p}membership_expire_date ELSE -{p}membership_expire_date END ASC"
    )


@dataclass(frozen=True)
class LabelMonth:
    """Day numbers for one label month and its history window (the previous month)."""

    year: int
    month: int

    @property
    def start(self) -> int:
        return to_days(f"{self.year}-{self.month:02d}-01")

    @property
    def end(self) -> int:
        return to_days(f"{self.year}-{self.month:02d}-{calendar.monthrange(self.year, self.month)[1]:02d}")

    @property
    def history_start(self) -> int:
        y, m = (self.year - 1, 12) if self.month == 1 else (self.year, self.month - 1)
        return to_days(f"{y}-{m:02d}-01")

    @property
    def history_end(self) -> int:
        return self.start - 1

    @property
    def data_needed_until(self) -> int:
        """Latest day a renewal can still count (gap < 30 from an expiry no later than month end).

        Anything after it is churn whatever happens, so transactions must reach this day.
        """
        return self.end + RENEWAL_GAP_DAYS - 1

    @property
    def key(self) -> str:
        return f"{self.year}-{self.month:02d}"

    @classmethod
    def parse(cls, text: str) -> "LabelMonth":
        year, month = text.split("-")[:2]
        return cls(int(year), int(month))


def candidates_sql(month: LabelMonth, transactions: str = "transactions") -> str:
    """Users the script labels for `month`, with their last expiry. Columns: msno, last_expire."""
    return f"""
SELECT msno, membership_expire_date AS last_expire
FROM (
    SELECT msno, membership_expire_date,
           ROW_NUMBER() OVER (PARTITION BY msno ORDER BY {order_last_first()}) AS rn
    FROM {transactions}
    WHERE transaction_date >= {month.history_start} AND transaction_date <= {month.history_end}
) h
WHERE rn = 1 AND membership_expire_date >= {month.start} AND membership_expire_date <= {month.end}
"""


def expiry_population_sql(month: LabelMonth, transactions: str = "transactions") -> str:
    """Users whose last transaction before the month (all history, the script's ordering) expires in it.

    Not part of the script: a wider version of its candidate rule that also keeps users whose
    last transaction is older than the previous month (longer plans). It matches the April test
    users more closely. Columns: msno, last_expire.
    """
    return f"""
SELECT msno, membership_expire_date AS last_expire
FROM (
    SELECT msno, membership_expire_date,
           ROW_NUMBER() OVER (PARTITION BY msno ORDER BY {order_last_first()}) AS rn
    FROM {transactions}
    WHERE transaction_date < {month.start}
) h
WHERE rn = 1 AND membership_expire_date >= {month.start} AND membership_expire_date <= {month.end}
"""


def labels_sql(month: LabelMonth, transactions: str = "transactions", future_until: int | None = None) -> str:
    """Full label table for `month`.

    future_until: ignore transactions after this day when deciding renewals. Used to test whether
    a label file was made before all renewals were recorded; None uses everything (as the script).

    Columns: msno, last_expire, effective_expire, renewal_date, renewal_gap, outcome, is_churn
    outcome is one of: renewed, late_renewal, cancel_only, no_activity.
    """
    until = "" if future_until is None else f" AND t.transaction_date <= {int(future_until)}"
    return f"""
WITH candidates AS ({candidates_sql(month, transactions)}),
future AS (
    SELECT t.msno, t.transaction_date, t.membership_expire_date, t.is_cancel,
           ROW_NUMBER() OVER (PARTITION BY t.msno ORDER BY {order_ascending("t.")}) AS rn
    FROM {transactions} t
    JOIN candidates c ON c.msno = t.msno
    WHERE t.transaction_date > {month.history_end}{until}
),
activity AS (
    SELECT msno, COUNT(*) AS n FROM future GROUP BY msno
),
first_renewal AS (
    SELECT msno, MIN(rn) AS rn FROM future WHERE is_cancel = 0 GROUP BY msno
),
renewal AS (
    SELECT f.msno, f.transaction_date AS renewal_date, f.rn
    FROM future f JOIN first_renewal r ON r.msno = f.msno AND r.rn = f.rn
),
cancels_before AS (
    -- cancellations before the first renewal (all of them when there is no renewal)
    SELECT f.msno, MIN(f.membership_expire_date) AS min_expire
    FROM future f LEFT JOIN renewal r ON r.msno = f.msno
    WHERE f.is_cancel = 1 AND (r.rn IS NULL OR f.rn < r.rn)
    GROUP BY f.msno
),
resolved AS (
    SELECT c.msno, c.last_expire,
           CASE WHEN cb.min_expire IS NOT NULL AND cb.min_expire < c.last_expire
                THEN cb.min_expire ELSE c.last_expire END AS effective_expire,
           r.renewal_date,
           CASE WHEN a.msno IS NULL THEN 1 ELSE 0 END AS no_activity
    FROM candidates c
    LEFT JOIN activity a ON a.msno = c.msno
    LEFT JOIN renewal r ON r.msno = c.msno
    LEFT JOIN cancels_before cb ON cb.msno = c.msno
)
SELECT msno, last_expire, effective_expire, renewal_date,
       renewal_date - effective_expire AS renewal_gap,
       CASE WHEN no_activity = 1 THEN 'no_activity'
            WHEN renewal_date IS NULL THEN 'cancel_only'
            WHEN renewal_date - effective_expire >= {RENEWAL_GAP_DAYS} THEN 'late_renewal'
            ELSE 'renewed' END AS outcome,
       CASE WHEN no_activity = 1 OR renewal_date IS NULL
                 OR renewal_date - effective_expire >= {RENEWAL_GAP_DAYS} THEN 1 ELSE 0 END AS is_churn
FROM resolved
ORDER BY msno
"""


def can_label(month: LabelMonth, data_start: int, data_end: int) -> tuple[bool, str]:
    """Whether the transactions cover the history window and the renewal window."""
    if data_start > month.history_start:
        return False, f"needs transactions from {from_days(month.history_start)}, data starts {from_days(data_start)}"
    if data_end < month.data_needed_until:
        return False, f"needs transactions up to {from_days(month.data_needed_until)}, data ends {from_days(data_end)}"
    return True, ""


# --------------------------------------------------------------------------- pipeline stage


def run_labels(con, data_dir: str, run_dir: str, LOG, engine: str = "duckdb", snapshots=None) -> dict:
    """Build rebuilt label tables (--mode labels) and print a summary of each month."""
    import json
    import os

    import pandas as pd

    from src.console import bullet_block, pct, print_table
    from src.constants import REBUILT_SNAPSHOTS, SNAPSHOTS
    from src.db import load_frame, parquet_path, query_df, scalar
    from src.logger import component_stage, log_stage

    component = "Labels"
    names = list(snapshots or REBUILT_SNAPSHOTS)
    bad = [n for n in names if SNAPSHOTS[n].source != "rebuilt"]
    if bad:
        raise ValueError(f"{bad} use official labels; --mode labels builds only {REBUILT_SNAPSHOTS}.")

    rows, notes = [], []
    with component_stage(LOG, component, snapshots=",".join(names)):
        data_start = int(scalar(con, "SELECT MIN(transaction_date) FROM transactions"))
        data_end = int(scalar(con, "SELECT MAX(transaction_date) FROM transactions"))
        log_stage(LOG, "INFO", component, transactions_from=from_days(data_start), transactions_to=from_days(data_end))

        for name in names:
            snap = SNAPSHOTS[name]
            month = LabelMonth.parse(snap.label_month)
            ok, why = can_label(month, data_start, data_end)
            if not ok:
                log_stage(LOG, "WARN", component, snapshot=name, skipped=why)
                rows.append({"snapshot": name, "month": month.key, "note": f"skipped: {why}"})
                continue
            with component_stage(LOG, f"{component}:{name}", month=month.key):
                df = query_df(con, labels_sql(month))
                df["msno"] = df["msno"].astype(str)
                df["is_churn"] = df["is_churn"].astype("int64")
                if engine == "sqlite":
                    load_frame(con, snap.label_table, df)
                    where = "sqlite"
                else:
                    path = parquet_path(data_dir, snap.label_table)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    if _has_pyarrow():
                        df.to_parquet(path, index=False)
                    else:
                        _duckdb_write(con, df, path)
                    con.execute(f"CREATE OR REPLACE VIEW {snap.label_table} AS SELECT * FROM read_parquet('{path}')")
                    where = path
                outcome = df["outcome"].value_counts(normalize=True)
                row = {
                    "snapshot": name,
                    "month": month.key,
                    "users": len(df),
                    "churn rate": float(df["is_churn"].mean()),
                }
                for key in ("renewed", "late_renewal", "cancel_only", "no_activity"):
                    row[key.replace("_", " ")] = float(outcome.get(key, 0.0))
                rows.append(row)
                log_stage(
                    LOG,
                    "METRICS",
                    component,
                    snapshot=name,
                    month=month.key,
                    users=len(df),
                    churn_rate=round(row["churn rate"], 5),
                )
                log_stage(LOG, "ARTIFACT", component, table=snap.label_table, path=where)

        table = pd.DataFrame(rows)
        out = os.path.join(run_dir, "metrics", "labels_summary.csv")
        table.to_csv(out, index=False)
        log_stage(LOG, "ARTIFACT", component, summary=out)
        print_table(
            LOG,
            table,
            "Labels rebuilt with the organisers' rule",
            formats={c: pct for c in ["churn rate", "renewed", "late renewal", "cancel only", "no activity"]},
        )
        notes.append(
            "Churn = no activity after the month before, only cancellations, or a first renewal 30+ days after "
            "the (possibly cancelled-earlier) expiry. Same rule and ordering as WSDMChurnLabeller.scala."
        )
        notes.append(
            "To give these users listening features, rebuild user_logs afterwards: --mode ingest --tables user_logs."
        )
        bullet_block(LOG, "Notes:", notes)
        with open(os.path.join(run_dir, "artifacts", "labels_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "snapshots": names,
                    "transactions_from": str(from_days(data_start)),
                    "transactions_to": str(from_days(data_end)),
                },
                f,
                indent=4,
            )
    return {"table": table}


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def _duckdb_write(con, df, path):
    con.register("_labels_tmp", df)
    con.execute(f"COPY (SELECT * FROM _labels_tmp) TO '{path}' (FORMAT PARQUET)")
    con.unregister("_labels_tmp")
