"""Synthetic KKBox-shaped data, so the whole pipeline runs without Kaggle access.

The tables match the processed schema in src/constants.py (dates already as
integer days). Churn is driven by a hidden engagement score plus auto-renew,
and both leave traces in the transactions and listening logs, so a model has
real signal to find. The numbers mean nothing beyond that.
"""

import numpy as np
import pandas as pd

from src.constants import LOG_START, OFFICIAL_SNAPSHOTS, SNAPSHOTS, to_days
from src.logger import log_stage

COMPONENT = "Demo"
TX_START = "2016-01-01"
DATA_END = "2017-03-31"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_demo_frames(n_users: int = 5000, seed: int = 42, LOG=None) -> dict:
    rng = np.random.default_rng(seed)
    msno = np.array([f"user_{i:06d}" for i in range(n_users)])

    engagement = rng.normal(0.0, 1.0, n_users)
    auto_renew = (rng.random(n_users) < 0.85).astype(int)
    reg_day = to_days("2017-01-01") - rng.integers(40, 2000, n_users)
    bill_offset = rng.integers(0, 28, n_users)  # day of month the user is billed on
    method = np.where(auto_renew == 1, 41, rng.integers(36, 41, n_users))
    price = rng.choice([99, 129, 149, 180], n_users, p=[0.2, 0.1, 0.6, 0.1])

    # ---- labels: one churn draw per snapshot; churners mostly leave the later snapshots
    # January churn happens in the demo too (its labels are rebuilt by --mode labels, not shipped).
    snaps = [SNAPSHOTS["jan"]] + [SNAPSHOTS[n] for n in OFFICIAL_SNAPSHOTS]
    cutoffs = {s.name: s.cutoff_day for s in snaps}
    in_snapshot, churn = {}, {}
    alive = np.ones(n_users, dtype=bool)
    never = to_days("2100-01-01")
    gone_after = np.full(n_users, never)  # day a user leaves for good
    for s in snaps:
        in_snapshot[s.name] = alive.copy()
        p = _sigmoid(-2.6 - 0.8 * engagement + 1.6 * (1 - auto_renew) + rng.normal(0, 0.6, n_users))
        churn[s.name] = (rng.random(n_users) < p) & alive
        left = churn[s.name] & (rng.random(n_users) >= 0.3)  # 30% of churners come back
        gone_after = np.where(left, s.cutoff_day, gone_after)
        alive = alive & ~left

    # ---- members (some users have no record; bd is noisy like the real file)
    has_member = rng.random(n_users) < 0.88
    bd = np.where(rng.random(n_users) < 0.4, 0, np.clip(rng.normal(28, 8, n_users), 5, 95)).astype(int)
    gender = rng.choice(np.array(["male", "female", None], dtype=object), n_users, p=[0.3, 0.3, 0.4])
    members = pd.DataFrame(
        {
            "msno": msno,
            "city": rng.integers(1, 23, n_users),
            "bd": bd,
            "gender": gender,
            "registered_via": rng.choice([3, 4, 7, 9, 13], n_users),
            "registration_init_time": reg_day,
        }
    )[has_member].reset_index(drop=True)

    # ---- transactions: one renewal per month, plus a cancel shortly before some churn cutoffs
    tx_parts = []
    month_starts = pd.date_range(TX_START, DATA_END, freq="MS")
    for m in month_starts:
        day = to_days(m.date()) + bill_offset
        active = (day >= reg_day) & (day < gone_after) & (day <= to_days(DATA_END))
        discount = rng.random(n_users) < 0.05
        tx_parts.append(
            pd.DataFrame(
                {
                    "msno": msno[active],
                    "payment_method_id": method[active],
                    "payment_plan_days": 30,
                    "plan_list_price": price[active],
                    "actual_amount_paid": np.where(discount, 0, price)[active],
                    "is_auto_renew": auto_renew[active],
                    "transaction_date": day[active],
                    "membership_expire_date": day[active] + 30,
                    "is_cancel": 0,
                }
            )
        )
    for s in snaps:
        cancels = churn[s.name] & (rng.random(n_users) < 0.25) & (auto_renew == 1)
        day = cutoffs[s.name] - rng.integers(1, 21, n_users)
        tx_parts.append(
            pd.DataFrame(
                {
                    "msno": msno[cancels],
                    "payment_method_id": method[cancels],
                    "payment_plan_days": 30,
                    "plan_list_price": price[cancels],
                    "actual_amount_paid": price[cancels],
                    "is_auto_renew": 1,
                    "transaction_date": day[cancels],
                    "membership_expire_date": day[cancels],
                    "is_cancel": 1,
                }
            )
        )
    transactions = pd.concat(tx_parts, ignore_index=True).astype(
        {
            c: "int64"
            for c in [
                "payment_method_id",
                "payment_plan_days",
                "plan_list_price",
                "actual_amount_paid",
                "is_auto_renew",
                "transaction_date",
                "membership_expire_date",
                "is_cancel",
            ]
        }
    )

    # ---- listening logs: engaged users listen on more days; churners fade in their last 30 days
    days = np.arange(to_days(LOG_START), to_days(DATA_END) + 1)
    p_active = np.repeat(_sigmoid(0.2 + 1.1 * engagement)[:, None], len(days), axis=1)
    for s in snaps:
        fading = churn[s.name][:, None] & (days[None, :] >= cutoffs[s.name] - 30) & (days[None, :] < cutoffs[s.name])
        p_active = np.where(fading, p_active * 0.7, p_active)
    p_active = np.where(days[None, :] >= gone_after[:, None], 0.03, p_active)
    p_active = np.where(days[None, :] < reg_day[:, None], 0.0, p_active)
    u_idx, d_idx = np.nonzero(rng.random(p_active.shape) < p_active)
    n_rows = len(u_idx)
    intensity = np.exp(0.5 * engagement[u_idx])
    num_100 = rng.poisson(12 * intensity)
    num_25 = rng.poisson(4, n_rows)
    user_logs = pd.DataFrame(
        {
            "msno": msno[u_idx],
            "log_date": days[d_idx],
            "num_25": num_25,
            "num_50": rng.poisson(1, n_rows),
            "num_75": rng.poisson(1, n_rows),
            "num_985": rng.poisson(1, n_rows),
            "num_100": num_100,
            "num_unq": np.maximum(1, (0.8 * (num_100 + num_25)).astype(int)),
            "total_secs": np.clip(num_100 * rng.normal(230, 25, n_rows) + num_25 * 20, 0, 86400).round(1),
        }
    )

    frames = {"members": members, "transactions": transactions, "user_logs": user_logs}
    for s in (s for s in snaps if s.source == "official"):
        keep = in_snapshot[s.name]
        # The test month ships as a zero-filled sample submission, like Kaggle's.
        labels = churn[s.name][keep].astype(int) if s.labelled else np.zeros(int(keep.sum()), dtype=int)
        frames[s.label_table] = pd.DataFrame({"msno": msno[keep], "is_churn": labels})

    if LOG is not None:
        log_stage(
            LOG,
            "INFO",
            COMPONENT,
            n_users=n_users,
            seed=seed,
            **{f"rows_{name}": len(df) for name, df in frames.items()},
        )
        for s in snaps:
            if s.labelled and s.label_table in frames:
                log_stage(
                    LOG,
                    "INFO",
                    COMPONENT,
                    snapshot=s.name,
                    churn_rate=round(float(frames[s.label_table]["is_churn"].mean()), 4),
                )
    return frames
