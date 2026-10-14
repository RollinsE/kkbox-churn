"""Line-by-line Python transliteration of WSDMChurnLabeller.scala, for testing only.

Fields are strings, as Spark reads them from the CSV: dates as YYYYMMDD,
numbers as their digits. The ordering is the Scala comparator, applied with a
stable sort like Scala's sortWith.
"""

from datetime import date
from functools import cmp_to_key


def _lt(x, y):
    if x["transaction_date"] != y["transaction_date"]:
        return x["transaction_date"] < y["transaction_date"]
    x_sig = x["plan_list_price"] + x["payment_plan_days"] + x["payment_method_id"]
    y_sig = y["plan_list_price"] + y["payment_plan_days"] + y["payment_method_id"]
    if x_sig != y_sig:
        return x_sig > y_sig
    if x["is_cancel"] == "1" and y["is_cancel"] == "1":
        return x["membership_expire_date"] > y["membership_expire_date"]
    if x["is_cancel"] == "0" and y["is_cancel"] == "0":
        return x["membership_expire_date"] < y["membership_expire_date"]
    return x["is_cancel"] < y["is_cancel"]


def _cmp(x, y):
    if _lt(x, y):
        return -1
    if _lt(y, x):
        return 1
    return 0


def _sorted(rows):
    return sorted(rows, key=cmp_to_key(_cmp))


def calculate_lastday(rows):
    return _sorted(rows)[-1]["membership_expire_date"]


def _parse(text):
    return date(int(text[0:4]), int(text[4:6]), int(text[6:8]))


def calculate_renewal_gap(rows, last_expiration):
    last_expire_date = _parse(last_expiration)
    gap = 9999
    for row in _sorted(rows):
        if gap != 9999:
            break
        trans_date = _parse(row["transaction_date"])
        expire_date = _parse(row["membership_expire_date"])
        if row["is_cancel"] == "1":
            if expire_date < last_expire_date:
                last_expire_date = expire_date
        else:
            gap = (trans_date - last_expire_date).days
    return gap


def label_month(rows, history_start, history_cutoff, month_start, month_end):
    """Returns {msno: is_churn} for one month, exactly as the script would."""
    history, future = {}, {}
    for r in rows:
        if history_start <= r["transaction_date"] <= history_cutoff:
            history.setdefault(r["msno"], []).append(r)
        if r["transaction_date"] > history_cutoff:
            future.setdefault(r["msno"], []).append(r)

    labels = {}
    for msno, hist in history.items():
        last_expire = calculate_lastday(hist)
        if not (month_start <= last_expire <= month_end):
            continue
        if msno not in future:
            labels[msno] = 1
            continue
        gap = calculate_renewal_gap(future[msno], last_expire)
        labels[msno] = 0 if gap < 30 else 1
    return labels
