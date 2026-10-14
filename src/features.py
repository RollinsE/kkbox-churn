"""Cutoff-based feature SQL.

One query builds one row per user in a snapshot's label table. Every source
row is filtered with `< cutoff`, so a feature can never see the month it is
predicting. The SQL sticks to the subset shared by DuckDB and SQLite (integer
date arithmetic, CASE, window functions), which is why tests can check it on
SQLite and the real run can use DuckDB.

Portability rules followed below
--------------------------------
* `1.0 * x` before any division or AVG of integers (SQLite does integer division).
* NULLIF(den, 0) on every denominator.
* No engine-specific functions (no GREATEST, DATE_DIFF, FILTER, QUALIFY).
"""

import json
import os

from src.constants import FEATURE_SNAPSHOTS, ID_COL, LOG_WINDOWS, META_COLS, SNAPSHOTS, TARGET_COL, from_days
from src.db import query_df, table_exists
from src.labeller import LabelMonth, candidates_sql, expiry_population_sql, order_last_first
from src.logger import component_stage, log_stage
from src.storage import features_path, write_frame

COMPONENT = "Features"

# Integer-coded categories. Everything else is numeric.
CATEGORICAL_FEATURES = ["city", "gender_code", "registered_via", "last_payment_method_id"]


def _log_window_columns(cutoff: int, windows=LOG_WINDOWS) -> str:
    cols = []
    for w in windows:
        start = cutoff - w
        in_w = f"log_date >= {start}"
        cols += [
            f"COUNT(DISTINCT CASE WHEN {in_w} THEN log_date END) AS logs_active_days_{w}d",
            f"SUM(CASE WHEN {in_w} THEN total_secs ELSE 0 END) AS logs_secs_{w}d",
            f"SUM(CASE WHEN {in_w} THEN num_unq ELSE 0 END) AS logs_unq_{w}d",
            f"SUM(CASE WHEN {in_w} THEN num_100 ELSE 0 END) AS logs_n100_{w}d",
            f"SUM(CASE WHEN {in_w} THEN num_25 + num_50 + num_75 + num_985 + num_100 ELSE 0 END) AS logs_plays_{w}d",
        ]
    return ",\n        ".join(cols)


def build_feature_sql(label_table: str, cutoff: int, windows=LOG_WINDOWS) -> str:
    """Return the feature query for one snapshot. `cutoff` is in days since 1970-01-01."""
    cutoff = int(cutoff)
    max_w = max(windows)
    log_cols = [f"logs_{m}_{w}d" for w in windows for m in ("active_days", "secs", "unq", "n100", "plays")]
    log_select = ",\n    ".join(f"COALESCE(g.{c}, 0) AS {c}" for c in log_cols)
    # Users the organisers' rule would label for this month (uses only the month before the cutoff).
    month = LabelMonth.parse(from_days(cutoff).isoformat())
    population_sql = candidates_sql(month)
    expiry_sql = expiry_population_sql(month)

    return f"""
WITH tx AS (
    SELECT t.msno, t.payment_method_id, t.payment_plan_days, t.plan_list_price, t.actual_amount_paid,
           t.is_auto_renew, t.transaction_date, t.membership_expire_date, t.is_cancel
    FROM transactions t
    JOIN {label_table} l ON l.msno = t.msno
    WHERE t.transaction_date < {cutoff}
),
tx_ranked AS (
    SELECT tx.*,
           -- The organisers' ordering (src/labeller.py): on a day with a renewal and a
           -- cancellation of the same plan, the cancellation is the last transaction.
           ROW_NUMBER() OVER (
               PARTITION BY msno
               ORDER BY {order_last_first()}, actual_amount_paid DESC, is_auto_renew DESC
           ) AS rn
    FROM tx
),
tx_last AS (
    SELECT msno,
           payment_method_id AS last_payment_method_id,
           payment_plan_days AS last_plan_days,
           plan_list_price AS last_list_price,
           actual_amount_paid AS last_amount_paid,
           is_auto_renew AS last_is_auto_renew,
           is_cancel AS last_is_cancel,
           {cutoff} - transaction_date AS days_since_last_tx,
           membership_expire_date - {cutoff} AS days_to_expire
    FROM tx_ranked
    WHERE rn = 1
),
tx_agg AS (
    SELECT msno,
           COUNT(*) AS tx_count,
           SUM(is_cancel) AS tx_cancel_count,
           AVG(1.0 * is_cancel) AS tx_cancel_rate,
           AVG(1.0 * is_auto_renew) AS tx_auto_renew_rate,
           SUM(actual_amount_paid) AS tx_total_paid,
           AVG(1.0 * actual_amount_paid) AS tx_avg_paid,
           AVG(1.0 * payment_plan_days) AS tx_avg_plan_days,
           COUNT(DISTINCT payment_method_id) AS tx_n_payment_methods,
           SUM(CASE WHEN actual_amount_paid < plan_list_price THEN 1 ELSE 0 END) AS tx_discount_count,
           {cutoff} - MIN(transaction_date) AS days_since_first_tx,
           SUM(CASE WHEN transaction_date >= {cutoff - 90} THEN 1 ELSE 0 END) AS tx_count_90d,
           SUM(CASE WHEN transaction_date >= {cutoff - 90} THEN is_cancel ELSE 0 END) AS tx_cancel_count_90d,
           SUM(CASE WHEN transaction_date >= {cutoff - 30} THEN is_cancel ELSE 0 END) AS tx_cancel_count_30d,
           {cutoff} - MAX(CASE WHEN is_cancel = 1 THEN transaction_date END) AS days_since_last_cancel
    FROM tx
    GROUP BY msno
),
log_agg AS (
    SELECT u.msno,
        {cutoff} - MAX(log_date) AS days_since_last_log,
        {_log_window_columns(cutoff, windows)}
    FROM user_logs u
    JOIN {label_table} l ON l.msno = u.msno
    WHERE u.log_date < {cutoff} AND u.log_date >= {cutoff - max_w}
    GROUP BY u.msno
)
SELECT
    l.msno,
    l.is_churn,
    CASE WHEN e.msno IS NULL THEN 0 ELSE 1 END AS in_population_expiry,
    CASE WHEN p.msno IS NULL THEN 0 ELSE 1 END AS in_labeller_population,
    -- members
    CASE WHEN m.msno IS NULL THEN 0 ELSE 1 END AS has_member_record,
    m.city,
    CASE WHEN m.bd BETWEEN 10 AND 80 THEN m.bd END AS age,
    CASE WHEN m.gender = 'male' THEN 1 WHEN m.gender = 'female' THEN 2 ELSE 0 END AS gender_code,
    m.registered_via,
    {cutoff} - m.registration_init_time AS tenure_days,
    -- transactions
    COALESCE(a.tx_count, 0) AS tx_count,
    COALESCE(a.tx_cancel_count, 0) AS tx_cancel_count,
    a.tx_cancel_rate,
    a.tx_auto_renew_rate,
    COALESCE(a.tx_total_paid, 0) AS tx_total_paid,
    a.tx_avg_paid,
    a.tx_avg_plan_days,
    COALESCE(a.tx_n_payment_methods, 0) AS tx_n_payment_methods,
    COALESCE(a.tx_discount_count, 0) AS tx_discount_count,
    a.days_since_first_tx,
    COALESCE(a.tx_count_90d, 0) AS tx_count_90d,
    COALESCE(a.tx_cancel_count_90d, 0) AS tx_cancel_count_90d,
    COALESCE(a.tx_cancel_count_30d, 0) AS tx_cancel_count_30d,
    a.days_since_last_cancel,
    t.last_payment_method_id,
    t.last_plan_days,
    t.last_list_price,
    t.last_amount_paid,
    t.last_list_price - t.last_amount_paid AS last_discount,
    t.last_is_auto_renew,
    t.last_is_cancel,
    t.days_since_last_tx,
    t.days_to_expire,
    -- listening logs
    g.days_since_last_log,
    {log_select},
    1.0 * g.logs_n100_30d / NULLIF(g.logs_plays_30d, 0) AS logs_completion_30d,
    1.0 * g.logs_secs_30d / NULLIF(g.logs_active_days_30d, 0) AS logs_secs_per_active_day_30d,
    (g.logs_secs_7d / 7.0) / NULLIF(g.logs_secs_30d / 30.0, 0) AS logs_secs_trend_7_30,
    (g.logs_secs_30d / 30.0) / NULLIF(g.logs_secs_90d / 90.0, 0) AS logs_secs_trend_30_90
FROM {label_table} l
LEFT JOIN members m ON m.msno = l.msno
LEFT JOIN tx_agg a ON a.msno = l.msno
LEFT JOIN tx_last t ON t.msno = l.msno
LEFT JOIN log_agg g ON g.msno = l.msno
LEFT JOIN ({population_sql}) p ON p.msno = l.msno
LEFT JOIN ({expiry_sql}) e ON e.msno = l.msno
ORDER BY l.msno
"""


CALENDAR_FEATURES = ["expire_day_of_month", "expire_days_in_month", "expire_days_to_month_end", "expire_on_month_end"]


def add_calendar_features(df, cutoff: int):
    """Where in its calendar month the last-known membership expires.

    Most memberships expire on the last day of a month and rarely churn; the few that expire a day
    or three before the end of a 31-day month churn heavily. days_to_expire (counted from the 1st)
    cannot tell February's 28th (month end) from March's 28th (just before it), so these features
    describe the position relative to the month's own end. Computed from days_to_expire, so they
    use nothing after the cutoff.
    """
    import pandas as pd

    expiry_day = cutoff + df["days_to_expire"]
    known = expiry_day.notna()
    dates = pd.to_datetime(expiry_day[known].astype("int64"), unit="D", origin="unix")
    df["expire_day_of_month"] = float("nan")
    df["expire_days_in_month"] = float("nan")
    df.loc[known, "expire_day_of_month"] = dates.dt.day.astype(float).to_numpy()
    df.loc[known, "expire_days_in_month"] = dates.dt.days_in_month.astype(float).to_numpy()
    df["expire_days_to_month_end"] = df["expire_days_in_month"] - df["expire_day_of_month"]
    df["expire_on_month_end"] = (df["expire_days_to_month_end"] == 0).astype(float).where(known)
    return df


def build_features(con, snapshot_name: str, LOG, windows=LOG_WINDOWS):
    """Run the feature query for one snapshot and return a DataFrame."""
    snap = SNAPSHOTS[snapshot_name]
    if not table_exists(con, snap.label_table):
        hint = "Run --mode labels first." if snap.source == "rebuilt" else "Run --mode ingest first."
        raise FileNotFoundError(f"Label table {snap.label_table} for snapshot {snapshot_name} not found. {hint}")
    df = query_df(con, build_feature_sql(snap.label_table, snap.cutoff_day, windows))

    if df[ID_COL].duplicated().any():
        raise ValueError(f"Duplicate {ID_COL} in features for snapshot={snapshot_name}; check the label table.")

    df = add_calendar_features(df, snap.cutoff_day)
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL, *META_COLS)]
    for col in META_COLS:
        df[col] = df[col].astype("int64")
    for col in feature_cols:
        df[col] = df[col].astype("float64")  # one dtype for both engines, NaN = missing
    df[TARGET_COL] = df[TARGET_COL].astype("int64")
    df[ID_COL] = df[ID_COL].astype(str)

    log_stage(
        LOG,
        "METRICS",
        COMPONENT,
        snapshot=snap.name,
        cutoff=snap.cutoff,
        rows=len(df),
        features=len(feature_cols),
        churn_rate=round(float(df[TARGET_COL].mean()), 5) if snap.labelled else None,
        no_tx_share=round(float((df["tx_count"] == 0).mean()), 4),
        no_logs_share=round(float(df["days_since_last_log"].isna().mean()), 4),
        expiry_population_share=round(float(df["in_population_expiry"].mean()), 4),
        labeller_population_share=round(float(df["in_labeller_population"].mean()), 4),
    )
    return df


def run_features(
    con,
    data_dir: str,
    run_dir: str,
    LOG,
    snapshots=None,
    frame_format: str = "parquet",
    engine: str = "duckdb",
    strict: bool = False,
) -> dict:
    """Build and store the feature table for each snapshot. Returns {snapshot: path}.

    A rebuilt-label snapshot that is not ready (labels not built, or user_logs not covering its
    users) is an error when asked for explicitly (strict) and is skipped with a warning when it
    only came from the default list.
    """
    from src.ingest import logs_cover

    snapshots = list(snapshots or FEATURE_SNAPSHOTS)
    ready = []
    for name in snapshots:
        snap = SNAPSHOTS[name]
        problem = None
        if snap.source == "rebuilt":
            if not table_exists(con, snap.label_table):
                problem = f"label table {snap.label_table} not built yet (run --mode labels)"
            elif not logs_cover(data_dir, snap.label_table, engine):
                # Without this, churners (who appear in no later label file) would have no listening
                # history at all, and "no logs" would leak the answer.
                problem = (
                    f"user_logs does not include the users of {snap.label_table} "
                    "(after --mode labels, run --mode ingest --tables user_logs)"
                )
        if problem and strict:
            raise ValueError(f"Snapshot {name}: {problem}.")
        if problem:
            log_stage(LOG, "WARN", COMPONENT, snapshot=name, skipped=problem)
            continue
        ready.append(name)
    snapshots = ready
    paths = {}
    with component_stage(LOG, COMPONENT, snapshots=",".join(snapshots), frame_format=frame_format):
        columns = None
        for name in snapshots:
            with component_stage(LOG, f"{COMPONENT}:{name}", cutoff=SNAPSHOTS[name].cutoff):
                df = build_features(con, name, LOG)
                paths[name] = write_frame(df, features_path(data_dir, name, frame_format))
                log_stage(LOG, "ARTIFACT", COMPONENT, snapshot=name, path=paths[name], rows=len(df))
                columns = [c for c in df.columns if c not in (ID_COL, TARGET_COL, *META_COLS)]

        schema_path = os.path.join(run_dir, "artifacts", "feature_columns.json")
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "features": columns,
                    "categorical": CATEGORICAL_FEATURES,
                    "cutoffs": {n: SNAPSHOTS[n].cutoff for n in snapshots},
                },
                f,
                indent=4,
            )
        log_stage(LOG, "ARTIFACT", COMPONENT, feature_columns=schema_path)
    return paths
