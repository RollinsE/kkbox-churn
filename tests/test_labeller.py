"""The SQL port of WSDMChurnLabeller.scala must make the same decision as the Scala.

Hand-made cases follow the comments in the script; a randomised test compares the
SQL with a line-by-line Python transliteration (tests/reference_labeller.py) on
data full of the awkward cases: same-day renew+cancel, several plans on one day,
consecutive cancellations, gaps of exactly 29/30 days.
"""

import numpy as np
import pandas as pd
from reference_labeller import label_month

from src.constants import from_days, to_days
from src.db import connect, load_frame, query_df
from src.labeller import LabelMonth, can_label, candidates_sql, labels_sql

FEB = LabelMonth(2017, 2)


def tx(msno, tx_date, expire, cancel=0, price=149, days=30, method=41):
    return {
        "msno": msno,
        "payment_method_id": method,
        "payment_plan_days": days,
        "plan_list_price": price,
        "actual_amount_paid": price,
        "is_auto_renew": 1,
        "transaction_date": to_days(tx_date),
        "membership_expire_date": to_days(expire),
        "is_cancel": cancel,
    }


def run_sql(rows, month=FEB):
    con = connect("sqlite")
    load_frame(con, "transactions", pd.DataFrame(rows))
    out = query_df(con, labels_sql(month))
    con.close()
    return out.set_index("msno")


def test_month_windows():
    assert FEB.history_start == to_days("2017-01-01") and FEB.history_end == to_days("2017-01-31")
    assert FEB.start == to_days("2017-02-01") and FEB.end == to_days("2017-02-28")
    jan = LabelMonth(2017, 1)
    assert jan.history_start == to_days("2016-12-01") and jan.history_end == to_days("2016-12-31")
    ok, _ = can_label(FEB, to_days("2015-01-01"), to_days("2017-03-31"))
    assert ok
    ok, why = can_label(LabelMonth(2017, 3), to_days("2015-01-01"), to_days("2017-03-31"))
    assert not ok and "2017-04-29" in why


def test_candidates_use_previous_month_only_and_last_transaction():
    rows = [
        tx("in_feb", "2017-01-10", "2017-02-09"),
        tx("dec_only", "2016-12-31", "2017-02-15"),  # no January transaction: not a candidate
        tx("expires_march", "2017-01-31", "2017-03-02"),  # last January expiry in March: not a candidate
        # last January transaction (by date) decides, not the latest expiry
        tx("last_wins", "2017-01-05", "2017-03-05", days=60),
        tx("last_wins", "2017-01-20", "2017-02-19"),
    ]
    con = connect("sqlite")
    load_frame(con, "transactions", pd.DataFrame(rows))
    cands = query_df(con, candidates_sql(FEB)).set_index("msno")["last_expire"]
    con.close()
    assert set(cands.index) == {"in_feb", "last_wins"}
    assert cands["last_wins"] == to_days("2017-02-19")


def test_same_day_subscribe_then_cancel_cancel_counts_last():
    rows = [
        # same day, same plan: renewal to March 20 and cancellation to Feb 10 -> cancellation is last
        tx("u", "2017-01-20", "2017-03-20"),
        tx("u", "2017-01-20", "2017-02-10", cancel=1),
    ]
    out = run_sql(rows)
    assert out.loc["u", "last_expire"] == to_days("2017-02-10")
    assert out.loc["u", "outcome"] == "no_activity" and out.loc["u", "is_churn"] == 1


def test_consecutive_cancels_and_renewals_ordering():
    rows = [
        # two cancels same day, same plan: the earlier expiry is last
        tx("cancels", "2017-01-15", "2017-02-20", cancel=1),
        tx("cancels", "2017-01-15", "2017-02-05", cancel=1),
        # two renewals same day, same plan: the later expiry is last (expiry keeps extending)
        tx("renewals", "2017-01-15", "2017-02-05"),
        tx("renewals", "2017-01-15", "2017-02-20"),
    ]
    out = run_sql(rows)
    assert out.loc["cancels", "last_expire"] == to_days("2017-02-05")
    assert out.loc["renewals", "last_expire"] == to_days("2017-02-20")


def test_plan_signature_is_compared_as_text():
    # price||days||method: "1493041" vs "993041". As text "1493041" < "993041", so the 149 plan sorts
    # after the 99 plan and is the last transaction of the day.
    rows = [tx("u", "2017-01-20", "2017-02-25", price=149), tx("u", "2017-01-20", "2017-02-10", price=99)]
    out = run_sql(rows)
    assert out.loc["u", "last_expire"] == to_days("2017-02-25")


def test_renewal_gap_rules():
    rows = [
        tx("on_time", "2017-01-10", "2017-02-09"),
        tx("on_time", "2017-02-09", "2017-03-11"),
        tx("early", "2017-01-10", "2017-02-09"),
        tx("early", "2017-02-01", "2017-03-11"),  # gap negative
        tx("gap29", "2017-01-10", "2017-02-09"),
        tx("gap29", "2017-03-10", "2017-04-09"),  # 29 days
        tx("gap30", "2017-01-10", "2017-02-09"),
        tx("gap30", "2017-03-11", "2017-04-10"),  # 30 days
        # a cancellation before renewing pulls the expiry earlier: Feb 25 -> Feb 05, renewal Mar 10 = 33 days
        tx("cancel_first", "2017-01-26", "2017-02-25"),
        tx("cancel_first", "2017-02-05", "2017-02-05", cancel=1),
        tx("cancel_first", "2017-03-10", "2017-04-09"),
        # a cancellation after the renewal does not matter
        tx("cancel_after", "2017-01-26", "2017-02-25"),
        tx("cancel_after", "2017-02-20", "2017-03-27"),
        tx("cancel_after", "2017-03-01", "2017-03-01", cancel=1),
        tx("only_cancel", "2017-01-26", "2017-02-25"),
        tx("only_cancel", "2017-02-05", "2017-02-05", cancel=1),
        tx("gone", "2017-01-26", "2017-02-25"),
    ]
    out = run_sql(rows)
    expected = {
        "on_time": ("renewed", 0),
        "early": ("renewed", 0),
        "gap29": ("renewed", 0),
        "gap30": ("late_renewal", 1),
        "cancel_first": ("late_renewal", 1),
        "cancel_after": ("renewed", 0),
        "only_cancel": ("cancel_only", 1),
        "gone": ("no_activity", 1),
    }
    for msno, (outcome, churn) in expected.items():
        assert (out.loc[msno, "outcome"], out.loc[msno, "is_churn"]) == (outcome, churn), msno
    assert out.loc["cancel_first", "effective_expire"] == to_days("2017-02-05")
    assert out.loc["cancel_first", "renewal_gap"] == 33


def test_future_cut_turns_unseen_renewals_into_churn():
    rows = [tx("late_feb", "2017-01-25", "2017-02-24"), tx("late_feb", "2017-03-03", "2017-04-02")]
    con = connect("sqlite")
    load_frame(con, "transactions", pd.DataFrame(rows))
    full = query_df(con, labels_sql(FEB)).set_index("msno")
    cut = query_df(con, labels_sql(FEB, future_until=to_days("2017-02-28"))).set_index("msno")
    con.close()
    assert full.loc["late_feb", "is_churn"] == 0
    assert cut.loc["late_feb", "is_churn"] == 1 and cut.loc["late_feb", "outcome"] == "no_activity"


def test_expiry_population_uses_all_history():
    from src.labeller import expiry_population_sql

    rows = [
        tx("long_plan", "2016-11-20", "2017-02-18", days=90),
        tx("prev_month", "2017-01-10", "2017-02-09"),
        tx("later", "2017-01-31", "2017-03-02"),
    ]
    con = connect("sqlite")
    load_frame(con, "transactions", pd.DataFrame(rows))
    expiry = set(query_df(con, expiry_population_sql(FEB))["msno"])
    script = set(query_df(con, candidates_sql(FEB))["msno"])
    con.close()
    assert expiry == {"long_plan", "prev_month"}
    assert script == {"prev_month"}


def _ymd(day: int) -> str:
    return from_days(day).strftime("%Y%m%d")


def _random_transactions(n_users=400, seed=0):
    """Messy transactions designed to hit every branch of the script."""
    rng = np.random.default_rng(seed)
    rows = []
    jan1 = to_days("2017-01-01")
    for u in range(n_users):
        msno = f"u{u:04d}"
        day = jan1 - int(rng.integers(0, 45))
        for _ in range(int(rng.integers(1, 7))):
            same_day = rng.random() < 0.35
            for _k in range(2 if same_day else 1):
                cancel = int(rng.random() < 0.3)
                price = int(rng.choice([99, 129, 149, 180]))
                days = int(rng.choice([30, 30, 30, 7, 90]))
                method = int(rng.choice([36, 38, 40, 41]))
                expire = day + (int(rng.integers(-5, 15)) if cancel else days + int(rng.integers(-3, 4)))
                rows.append(
                    {
                        "msno": msno,
                        "payment_method_id": method,
                        "payment_plan_days": days,
                        "plan_list_price": price,
                        "actual_amount_paid": price,
                        "is_auto_renew": 1,
                        "transaction_date": day,
                        "membership_expire_date": expire,
                        "is_cancel": cancel,
                    }
                )
            day += int(rng.choice([1, 5, 20, 28, 29, 30, 31, 35, 60]))
            if day > to_days("2017-03-31"):
                break
    return rows


def test_sql_matches_scala_transliteration_on_random_data():
    rows = _random_transactions()
    sql = run_sql(rows)["is_churn"].to_dict()

    as_text = [
        {k: (str(v) if k not in ("transaction_date", "membership_expire_date") else _ymd(v)) for k, v in r.items()}
        for r in rows
    ]
    ref = label_month(as_text, "20170101", "20170131", "20170201", "20170228")

    assert len(ref) > 100, "random data should produce plenty of February candidates"
    assert set(sql) == set(ref)
    mismatches = {m: (sql[m], ref[m]) for m in ref if sql[m] != ref[m]}
    assert not mismatches, mismatches
    assert 0 < sum(ref.values()) < len(ref)  # both outcomes occur


def test_sql_matches_on_duckdb_too():
    import pytest

    pytest.importorskip("duckdb")
    rows = _random_transactions(seed=3)
    a = run_sql(rows)
    con = connect("duckdb")
    load_frame(con, "transactions", pd.DataFrame(rows))
    b = query_df(con, labels_sql(FEB)).set_index("msno")
    con.close()
    pd.testing.assert_series_equal(a["is_churn"].astype(int), b["is_churn"].astype(int), check_names=False)
