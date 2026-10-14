"""Feature SQL checks: correctness against pandas, no future leakage, engine parity."""

import numpy as np
import pandas as pd
import pytest
from conftest import quiet_logger

from src.constants import OFFICIAL_SNAPSHOTS, SNAPSHOTS, to_days
from src.db import connect, load_frame
from src.demo import generate_demo_frames
from src.features import CATEGORICAL_FEATURES, build_features

LOG = quiet_logger()
N_USERS = 400


def _frames():
    return generate_demo_frames(n_users=N_USERS, seed=7)


def _con(frames, engine="sqlite"):
    con = connect(engine)
    for name, df in frames.items():
        load_frame(con, name, df)
    return con


def _features(frames, snapshot="mar", engine="sqlite"):
    con = _con(frames, engine)
    try:
        return build_features(con, snapshot, LOG)
    finally:
        con.close()


def _last_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """The organisers' ordering in pandas: date, then plan text descending, then renewals before
    cancellations (renewals by expiry ascending, cancellations by expiry descending)."""
    t = tx.copy()
    t["sig"] = (
        t["plan_list_price"].astype(str) + t["payment_plan_days"].astype(str) + t["payment_method_id"].astype(str)
    )
    t["key"] = np.where(t["is_cancel"] == 1, -t["membership_expire_date"], t["membership_expire_date"])
    t = t.sort_values(
        ["transaction_date", "sig", "is_cancel", "key", "actual_amount_paid", "is_auto_renew"],
        ascending=[True, False, True, True, True, True],
        kind="mergesort",
    )
    return t.groupby("msno").tail(1).set_index("msno")


def test_one_row_per_label_user_and_stable_order():
    frames = _frames()
    for name in OFFICIAL_SNAPSHOTS:
        snap = SNAPSHOTS[name]
        df = _features(frames, name)
        labels = frames[snap.label_table]
        assert len(df) == len(labels)
        assert df["msno"].tolist() == sorted(labels["msno"])
        assert set(CATEGORICAL_FEATURES) <= set(df.columns)


def test_matches_pandas_reference():
    """Recompute a spread of features in pandas and compare with the SQL result."""
    frames = _frames()
    snap = SNAPSHOTS["mar"]
    c = snap.cutoff_day
    got = _features(frames, "mar").set_index("msno")
    users = frames[snap.label_table]["msno"]

    tx = frames["transactions"]
    tx = tx[tx["msno"].isin(users) & (tx["transaction_date"] < c)]
    last = _last_transactions(tx)
    grouped = tx.groupby("msno")

    logs = frames["user_logs"]
    logs = logs[logs["msno"].isin(users) & (logs["log_date"] < c) & (logs["log_date"] >= c - 90)]
    logs30 = logs[logs["log_date"] >= c - 30]
    logs7 = logs[logs["log_date"] >= c - 7]
    members = frames["members"].set_index("msno")

    expected = pd.DataFrame(index=got.index)
    expected["tx_count"] = grouped.size()
    expected["tx_cancel_count"] = grouped["is_cancel"].sum()
    expected["tx_avg_paid"] = grouped["actual_amount_paid"].mean()
    expected["tx_count_90d"] = tx[tx["transaction_date"] >= c - 90].groupby("msno").size()
    expected["days_since_last_tx"] = c - last["transaction_date"]
    expected["days_to_expire"] = last["membership_expire_date"] - c
    expected["last_is_cancel"] = last["is_cancel"]
    expected["last_payment_method_id"] = last["payment_method_id"]
    expected["days_since_last_log"] = c - logs.groupby("msno")["log_date"].max()
    expected["logs_active_days_30d"] = logs30.groupby("msno")["log_date"].nunique()
    expected["logs_secs_7d"] = logs7.groupby("msno")["total_secs"].sum()
    expected["logs_secs_90d"] = logs.groupby("msno")["total_secs"].sum()
    expected["logs_n100_30d"] = logs30.groupby("msno")["num_100"].sum()
    expected["tenure_days"] = c - members["registration_init_time"]
    cancels = tx[tx["is_cancel"] == 1]
    expected["tx_cancel_count_30d"] = cancels[cancels["transaction_date"] >= c - 30].groupby("msno").size()
    expected["days_since_last_cancel"] = c - cancels.groupby("msno")["transaction_date"].max()

    zero_filled = [
        "tx_count",
        "tx_cancel_count",
        "tx_count_90d",
        "tx_cancel_count_30d",
        "logs_active_days_30d",
        "logs_secs_7d",
        "logs_secs_90d",
        "logs_n100_30d",
    ]
    expected[zero_filled] = expected[zero_filled].fillna(0)

    for col in expected.columns:
        np.testing.assert_allclose(
            got[col].to_numpy(dtype=float),
            expected[col].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-6,
            equal_nan=True,
            err_msg=col,
        )

    # ratio features: check the definition on users who have the denominator
    has30 = got["logs_secs_30d"] > 0
    np.testing.assert_allclose(
        got.loc[has30, "logs_secs_trend_7_30"],
        (got.loc[has30, "logs_secs_7d"] / 7.0) / (got.loc[has30, "logs_secs_30d"] / 30.0),
        rtol=1e-9,
    )
    assert got.loc[got["logs_plays_30d"] == 0, "logs_completion_30d"].isna().all()


def test_no_future_leakage():
    """Rows dated on or after the cutoff must not change a single feature value."""
    frames = _frames()
    snap = SNAPSHOTS["feb"]
    c = snap.cutoff_day
    before = _features(frames, "feb")

    users = frames[snap.label_table]["msno"].to_numpy()
    future_tx = pd.DataFrame(
        {
            "msno": np.repeat(users, 2),
            "payment_method_id": 99,
            "payment_plan_days": 999,
            "plan_list_price": 9999,
            "actual_amount_paid": 1,
            "is_auto_renew": 0,
            "transaction_date": np.tile([c, c + 20], len(users)),
            "membership_expire_date": c + 500,
            "is_cancel": 1,
        }
    )
    future_logs = pd.DataFrame(
        {
            "msno": np.repeat(users, 2),
            "log_date": np.tile([c, c + 5], len(users)),
            "num_25": 500,
            "num_50": 500,
            "num_75": 500,
            "num_985": 500,
            "num_100": 500,
            "num_unq": 500,
            "total_secs": 80000.0,
        }
    )
    leaky = dict(frames)
    leaky["transactions"] = pd.concat([frames["transactions"], future_tx], ignore_index=True)
    leaky["user_logs"] = pd.concat([frames["user_logs"], future_logs], ignore_index=True)

    after = _features(leaky, "feb")
    pd.testing.assert_frame_equal(before, after)


def test_past_rows_do_change_features():
    """Guards the leakage test: the same kind of row dated before the cutoff must show up."""
    frames = _frames()
    snap = SNAPSHOTS["feb"]
    c = snap.cutoff_day
    user = frames[snap.label_table]["msno"].iloc[0]
    before = _features(frames, "feb").set_index("msno").loc[user]

    past_tx = pd.DataFrame(
        [
            {
                "msno": user,
                "payment_method_id": 99,
                "payment_plan_days": 30,
                "plan_list_price": 149,
                "actual_amount_paid": 149,
                "is_auto_renew": 0,
                "transaction_date": c - 1,
                "membership_expire_date": c + 29,
                "is_cancel": 1,
            }
        ]
    )
    changed = dict(frames)
    changed["transactions"] = pd.concat([frames["transactions"], past_tx], ignore_index=True)
    after = _features(changed, "feb").set_index("msno").loc[user]

    assert after["tx_count"] == before["tx_count"] + 1
    assert after["days_since_last_tx"] == 1
    assert after["last_is_cancel"] == 1
    assert after["last_payment_method_id"] == 99


def test_user_with_no_history_gets_zeros_and_missing():
    frames = _frames()
    ghost = pd.DataFrame({"msno": ["zzz_ghost"], "is_churn": [1]})
    frames["labels_mar"] = pd.concat([frames["labels_mar"], ghost], ignore_index=True)
    row = _features(frames, "mar").set_index("msno").loc["zzz_ghost"]

    assert row["is_churn"] == 1
    assert row["has_member_record"] == 0
    assert row["tx_count"] == 0 and row["logs_active_days_90d"] == 0 and row["gender_code"] == 0
    for col in ("tenure_days", "days_since_last_tx", "days_since_last_log", "tx_avg_paid", "logs_completion_30d"):
        assert np.isnan(row[col]), col


def test_same_day_cancel_is_the_last_transaction():
    """A renewal and a cancellation on the same day: the cancellation is what the features see."""
    frames = _frames()
    user = frames["labels_feb"]["msno"].iloc[0]
    c = SNAPSHOTS["feb"].cutoff_day
    same_day = pd.DataFrame(
        [
            {
                "msno": user,
                "payment_method_id": 41,
                "payment_plan_days": 30,
                "plan_list_price": 149,
                "actual_amount_paid": 149,
                "is_auto_renew": 1,
                "transaction_date": c - 3,
                "membership_expire_date": c + 27,
                "is_cancel": 0,
            },
            {
                "msno": user,
                "payment_method_id": 41,
                "payment_plan_days": 30,
                "plan_list_price": 149,
                "actual_amount_paid": 149,
                "is_auto_renew": 1,
                "transaction_date": c - 3,
                "membership_expire_date": c + 5,
                "is_cancel": 1,
            },
        ]
    )
    frames["transactions"] = pd.concat([frames["transactions"], same_day], ignore_index=True)
    row = _features(frames, "feb").set_index("msno").loc[user]
    assert row["last_is_cancel"] == 1
    assert row["days_to_expire"] == 5
    assert row["days_since_last_cancel"] == 3


def test_calendar_features_describe_position_in_month():
    frames = _frames()
    users = frames["labels_mar"]["msno"].iloc[:3].tolist()
    c = SNAPSHOTS["mar"].cutoff_day
    rows = []
    for user, expire in zip(users, ["2017-03-31", "2017-03-29", "2017-04-30"], strict=True):
        rows.append(
            {
                "msno": user,
                "payment_method_id": 41,
                "payment_plan_days": 30,
                "plan_list_price": 149,
                "actual_amount_paid": 149,
                "is_auto_renew": 1,
                "transaction_date": c - 1,
                "membership_expire_date": to_days(expire),
                "is_cancel": 0,
            }
        )
    frames["transactions"] = pd.concat([frames["transactions"], pd.DataFrame(rows)], ignore_index=True)
    df = _features(frames, "mar").set_index("msno")
    month_end, before_end, april_end = (df.loc[u] for u in users)
    assert (month_end["expire_day_of_month"], month_end["expire_days_in_month"]) == (31, 31)
    assert month_end["expire_days_to_month_end"] == 0 and month_end["expire_on_month_end"] == 1
    assert before_end["expire_days_to_month_end"] == 2 and before_end["expire_on_month_end"] == 0
    assert (april_end["expire_days_in_month"], april_end["expire_on_month_end"]) == (30, 1)


def test_calendar_features_missing_without_transactions():
    frames = _frames()
    frames["labels_mar"] = pd.concat(
        [frames["labels_mar"], pd.DataFrame({"msno": ["nobody"], "is_churn": [0]})], ignore_index=True
    )
    row = _features(frames, "mar").set_index("msno").loc["nobody"]
    for col in ("expire_day_of_month", "expire_days_in_month", "expire_days_to_month_end", "expire_on_month_end"):
        assert np.isnan(row[col]), col


def test_labeller_population_flag():
    frames = _frames()
    labels = frames["labels_feb"]
    keep = labels["msno"].iloc[0]
    frames["labels_feb"] = pd.concat(
        [labels, pd.DataFrame({"msno": ["dec_only", "jan_to_march"], "is_churn": [1, 0]})], ignore_index=True
    )
    extra = pd.DataFrame(
        [
            {
                "msno": "dec_only",
                "payment_method_id": 41,
                "payment_plan_days": 30,
                "plan_list_price": 149,
                "actual_amount_paid": 149,
                "is_auto_renew": 1,
                "transaction_date": to_days("2016-12-20"),
                "membership_expire_date": to_days("2017-02-10"),
                "is_cancel": 0,
            },
            {
                "msno": "jan_to_march",
                "payment_method_id": 41,
                "payment_plan_days": 30,
                "plan_list_price": 149,
                "actual_amount_paid": 149,
                "is_auto_renew": 1,
                "transaction_date": to_days("2017-01-31"),
                "membership_expire_date": to_days("2017-03-02"),
                "is_cancel": 0,
            },
        ]
    )
    frames["transactions"] = pd.concat([frames["transactions"], extra], ignore_index=True)
    df = _features(frames, "feb").set_index("msno")
    assert df.loc["dec_only", "in_labeller_population"] == 0  # no January transaction
    assert df.loc["dec_only", "in_population_expiry"] == 1  # but its last expiry is in February
    assert df.loc["jan_to_march", "in_labeller_population"] == 0  # last January expiry is in March
    assert df.loc["jan_to_march", "in_population_expiry"] == 0
    assert df["in_labeller_population"].isin([0, 1]).all()
    assert "in_labeller_population" in df.columns and df["in_labeller_population"].dtype.kind == "i"
    assert keep in df.index


def test_rebuilt_snapshot_needs_its_label_table():
    frames = _frames()
    with pytest.raises(FileNotFoundError):
        _features(frames, "jan")


def test_duckdb_matches_sqlite():
    pytest.importorskip("duckdb")
    frames = _frames()
    for name in OFFICIAL_SNAPSHOTS:
        a = _features(frames, name, engine="sqlite")
        b = _features(frames, name, engine="duckdb")
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)
