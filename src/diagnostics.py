"""Diagnostics: where is accuracy lost between months?

Five independent checks, run with --mode diagnose:

1. shift     Within-month against across-month. Train on 80% of a month and score
             the other 20%, then score the other month. A model that does well
             inside a month but badly across months is fighting a month change,
             not a lack of capacity. A rate-matched score (an oracle that is told
             the scored month's churn rate) separates "the overall rate moved"
             from "the relationship between features and churn moved".
2. labels    Rebuild the churn label with the exact port of the organisers'
             script (src/labeller.py) and compare with train.csv, overall and
             on the churners themselves.
3. segments  Churn rate for the same kinds of user in February and March, and a
             split of the overall change into "different users" and "same users,
             different churn rate". Plus the features whose averages moved most.
4. truncation Were the official labels made before all renewals were in the
             data? Rebuild February with renewals cut at the end of February,
             and compare churn by expiry day across label sets.
5. expiry    Does days_to_expire fall inside the label month, as the snapshot
             design assumes?

One check failing does not stop the others. Results go to
metrics/diagnostics_*.csv and metrics/diagnostics_report.md, and are printed as
tables.
"""

from __future__ import annotations

import calendar
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

from src.console import bullet_block, num4, pct, pp, print_table, stage_banner
from src.constants import (
    ID_COL,
    KAGGLE_COMPETITION,
    META_COLS,
    POPULATION_COLUMNS,
    SNAPSHOTS,
    TARGET_COL,
    from_days,
    to_days,
)
from src.db import query_df, scalar
from src.evaluate import EPS, classification_metrics, match_rate
from src.labeller import LabelMonth, can_label, candidates_sql, labels_sql
from src.logger import component_stage, log_stage
from src.storage import features_path, read_frame

COMPONENT = "Diagnostics"
CHECKS = ("shift", "labels", "truncation", "segments", "expiry")
LABELLER_FILE = "WSDMChurnLabeller.scala"


# --------------------------------------------------------------------------- helpers


def prior_logloss(rate: float) -> float:
    rate = min(max(rate, EPS), 1 - EPS)
    return float(-(rate * np.log(rate) + (1 - rate) * np.log(1 - rate)))


def _skill(logloss: float, rate: float) -> float:
    """Share of the constant-rate log loss the model removes (0 = no better than the rate)."""
    base = prior_logloss(rate)
    return float(1 - logloss / base) if base > 0 else float("nan")


class _Model:
    """LightGBM when installed, otherwise scikit-learn's HistGradientBoosting."""

    def __init__(self, cols, seed):
        self.cols, self.seed, self.kind = cols, seed, None

    def fit(self, df: pd.DataFrame):
        rng = np.random.default_rng(self.seed)
        stop = rng.random(len(df)) < 0.1  # early-stopping slice, inside the training part
        fit_part, stop_part = df[~stop], df[stop]
        try:
            import lightgbm as lgb

            from src.features import CATEGORICAL_FEATURES
            from src.model import DEFAULT_LGB_PARAMS

            cats = [c for c in CATEGORICAL_FEATURES if c in self.cols]
            dfit = lgb.Dataset(fit_part[self.cols], fit_part[TARGET_COL], categorical_feature=cats, free_raw_data=False)
            dstop = lgb.Dataset(stop_part[self.cols], stop_part[TARGET_COL], categorical_feature=cats, reference=dfit)
            self.model = lgb.train(
                {**DEFAULT_LGB_PARAMS, "seed": self.seed},
                dfit,
                num_boost_round=2000,
                valid_sets=[dstop],
                valid_names=["stop"],
                callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)],
            )
            self.kind = "lightgbm"
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.model = HistGradientBoostingClassifier(
                max_iter=500, learning_rate=0.05, early_stopping=True, validation_fraction=0.1, random_state=self.seed
            )
            self.model.fit(df[self.cols], df[TARGET_COL])
            self.kind = "hist_gradient_boosting"
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "lightgbm":
            return self.model.predict(df[self.cols], num_iteration=self.model.best_iteration)
        return self.model.predict_proba(df[self.cols])[:, 1]


def _load_features(data_dir, name, frame_format, LOG):
    path = features_path(data_dir, name, frame_format)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run --mode features first.")
    df = read_frame(path)
    log_stage(LOG, "INFO", COMPONENT, action="load_features", snapshot=name, rows=len(df))
    return df


def _save(df: pd.DataFrame, run_dir: str, name: str, LOG) -> str:
    path = os.path.join(run_dir, "metrics", f"diagnostics_{name}.csv")
    df.to_csv(path, index=False)
    log_stage(LOG, "ARTIFACT", COMPONENT, table=name, path=path)
    return path


# --------------------------------------------------------------------------- 1. shift


def check_shift(feb: pd.DataFrame, mar: pd.DataFrame, run_dir: str, LOG, sample_rows=400_000, seed=42) -> dict:
    cols = [c for c in feb.columns if c not in (ID_COL, TARGET_COL, *META_COLS)]
    months = {"feb": feb, "mar": mar}
    rng = np.random.default_rng(seed)
    samples = {
        m: (df.sample(sample_rows, random_state=seed) if len(df) > sample_rows else df).reset_index(drop=True)
        for m, df in months.items()
    }

    rows, model_kind = [], None
    for train_month, other_month in (("feb", "mar"), ("mar", "feb")):
        data = samples[train_month]
        held = rng.random(len(data)) < 0.2
        model = _Model(cols, seed).fit(data[~held])
        model_kind = model.kind
        for label, scored in (("same month (held-out 20%)", data[held]), ("other month", samples[other_month])):
            y = scored[TARGET_COL].to_numpy()
            p = model.predict(scored)
            m = classification_metrics(y, p)
            p_matched = match_rate(p, m["base_rate"])
            ll_matched = classification_metrics(y, p_matched)["logloss"]
            rows.append(
                {
                    "trained on": train_month,
                    "scored on": train_month if label.startswith("same") else other_month,
                    "split": label,
                    "rows": m["n"],
                    "churn rate": m["base_rate"],
                    "mean pred": m["mean_pred"],
                    "log loss": m["logloss"],
                    "log loss rate-matched": ll_matched,
                    "skill": _skill(m["logloss"], m["base_rate"]),
                    "skill rate-matched": _skill(ll_matched, m["base_rate"]),
                    "auc": m["auc"],
                }
            )
            log_stage(
                LOG,
                "METRICS",
                COMPONENT,
                check="shift",
                trained_on=train_month,
                scored_on=rows[-1]["scored on"],
                logloss=round(m["logloss"], 6),
                logloss_rate_matched=round(ll_matched, 6),
                auc=None if m["auc"] is None else round(m["auc"], 6),
            )

    table = pd.DataFrame(rows)
    _save(table, run_dir, "shift", LOG)
    print_table(
        LOG,
        table.drop(columns=["split"]),
        f"Within-month against across-month ({model_kind}, up to {sample_rows:,} rows per month)",
        formats={
            "churn rate": pct,
            "mean pred": pct,
            "log loss": num4,
            "log loss rate-matched": num4,
            "skill": pct,
            "skill rate-matched": pct,
            "auc": num4,
        },
    )

    notes = []
    for train_month in ("feb", "mar"):
        same = table[(table["trained on"] == train_month) & table["split"].str.startswith("same")].iloc[0]
        other = table[(table["trained on"] == train_month) & ~table["split"].str.startswith("same")].iloc[0]
        drop = same["skill"] - other["skill"]
        recovered = other["skill rate-matched"] - other["skill"]
        remaining = same["skill"] - other["skill rate-matched"]
        notes.append(
            f"Trained on {train_month}: skill {pct(same['skill'])} on its own month, {pct(other['skill'])} on "
            f"{other['scored on']} (AUC {num4(same['auc'])} vs {num4(other['auc'])})."
        )
        if drop > 0.05:
            notes.append(
                f"Crossing months costs {pct(drop)} of skill. Matching the churn rate wins back {pct(recovered)}; "
                f"{pct(remaining)} is still lost, which means the link between features and churn differs between months, "
                "not just the overall rate."
                if remaining > 0.03
                else f"Crossing months costs {pct(drop)} of skill, and matching the churn rate wins back {pct(recovered)} of it: "
                "most of the loss is the overall churn rate moving, and the ranking carries over."
            )
        else:
            notes.append(
                "The model keeps most of its skill across months, so the month change is not the main problem."
            )
    bullet_block(LOG, "What this suggests (rate-matched scores use the answer and are for diagnosis only):", notes)
    return {"table": table, "notes": notes, "model": model_kind}


# --------------------------------------------------------------------------- 2. labels


def fetch_labeller(data_dir: str, run_dir: str, LOG) -> str | None:
    """Keep a copy of the organisers' labelling script next to the data and in the run."""
    ref_dir = os.path.join(data_dir, "reference")
    path = os.path.join(ref_dir, LABELLER_FILE)
    if not os.path.exists(path):
        os.makedirs(ref_dir, exist_ok=True)
        cmd = ["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-f", LABELLER_FILE, "-p", ref_dir]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            ok = proc.returncode == 0 and os.path.exists(path)
        except Exception:
            ok = False
        if not ok:
            log_stage(LOG, "WARN", COMPONENT, check="labels", note=f"could not download {LABELLER_FILE}")
            return None
    shutil.copy(path, os.path.join(run_dir, "artifacts", LABELLER_FILE))
    log_stage(LOG, "ARTIFACT", COMPONENT, labeller=path)
    return path


def check_labels(con, run_dir: str, LOG, data_dir: str | None = None) -> dict:
    """Rebuild labels with the exact port of the organisers' script and compare with the official files."""
    if data_dir:
        fetch_labeller(data_dir, run_dir, LOG)
    data_start = int(scalar(con, "SELECT MIN(transaction_date) FROM transactions"))
    data_end = int(scalar(con, "SELECT MAX(transaction_date) FROM transactions"))
    official = {"2017-02": "labels_feb", "2017-03": "labels_mar"}

    rows, confusion, outcomes = [], None, []
    for key in ("2017-01", "2017-02", "2017-03"):
        month = LabelMonth.parse(key)
        ok, why = can_label(month, data_start, data_end)
        if not ok:
            rows.append({"month": key, "note": f"skipped: {why}"})
            continue
        rebuilt = query_df(con, labels_sql(month))
        rebuilt[ID_COL] = rebuilt[ID_COL].astype(str)
        rebuilt["is_churn"] = rebuilt["is_churn"].astype(int)
        row = {"month": key, "rebuilt users": len(rebuilt), "rebuilt churn rate": float(rebuilt["is_churn"].mean())}
        for outcome, share in rebuilt["outcome"].value_counts(normalize=True).items():
            outcomes.append(
                {
                    "month": key,
                    "outcome": outcome,
                    "share": float(share),
                    "users": int((rebuilt["outcome"] == outcome).sum()),
                }
            )

        if key in official:
            labels = query_df(con, f"SELECT msno, is_churn FROM {official[key]}")
            labels[ID_COL] = labels[ID_COL].astype(str)
            both = labels.merge(
                rebuilt[[ID_COL, "is_churn", "outcome"]].rename(columns={"is_churn": "rebuilt"}), on=ID_COL, how="inner"
            )
            churners = both[TARGET_COL] == 1
            rebuilt_churn = both["rebuilt"] == 1
            row.update(
                {
                    "official users": len(labels),
                    "official churn rate": float(labels[TARGET_COL].mean()),
                    "overlap users": len(both),
                    "official only": len(labels) - len(both),
                    "rebuilt only": len(rebuilt) - len(both),
                    "agreement": float((both[TARGET_COL] == both["rebuilt"]).mean()) if len(both) else float("nan"),
                    # with ~5% churn, overall agreement hides a lot: compare on the churners themselves
                    "official churners also rebuilt churn": float(rebuilt_churn[churners].mean())
                    if churners.any()
                    else float("nan"),
                    "rebuilt churners also official churn": float(churners[rebuilt_churn].mean())
                    if rebuilt_churn.any()
                    else float("nan"),
                }
            )
            if len(both):
                confusion = pd.crosstab(both[TARGET_COL].rename("official"), both["rebuilt"].rename("rebuilt"))
                disagree = both[both[TARGET_COL] != both["rebuilt"]]
                if len(disagree):
                    _save(
                        disagree.groupby([TARGET_COL, "outcome"])
                        .size()
                        .reset_index(name="users")
                        .rename(columns={TARGET_COL: "official is_churn", "outcome": "rebuilt outcome"}),
                        run_dir,
                        "labels_disagreements",
                        LOG,
                    )
        rows.append(row)
        log_stage(
            LOG,
            "METRICS",
            COMPONENT,
            check="labels",
            **{k.replace(" ", "_"): (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()},
        )

    table = pd.DataFrame(rows)
    _save(table, run_dir, "labels", LOG)
    columns = [
        c
        for c in [
            "month",
            "official users",
            "rebuilt users",
            "overlap users",
            "official churn rate",
            "rebuilt churn rate",
            "agreement",
            "official churners also rebuilt churn",
            "rebuilt churners also official churn",
            "note",
        ]
        if c in table.columns
    ]
    print_table(
        LOG,
        table[columns],
        "Official labels against the organisers' rule (exact port of the script)",
        formats={
            c: pct
            for c in [
                "official churn rate",
                "rebuilt churn rate",
                "agreement",
                "official churners also rebuilt churn",
                "rebuilt churners also official churn",
            ]
        },
    )
    outcome_table = pd.DataFrame(outcomes)
    if not outcome_table.empty:
        _save(outcome_table, run_dir, "labels_outcomes", LOG)
        print_table(LOG, outcome_table, "How the rebuilt labels were decided", formats={"share": pct})

    notes = []
    if confusion is not None:
        confusion_df = confusion.reset_index().rename(columns={0: "rebuilt: renewed", 1: "rebuilt: churned"})
        confusion_df["official"] = confusion_df["official"].map({0: "official: renewed", 1: "official: churned"})
        _save(confusion_df, run_dir, "labels_confusion", LOG)
        print_table(LOG, confusion_df, "February: official against rebuilt (users in both)")
        feb = table[table["month"] == "2017-02"].iloc[0]
        recall = feb["official churners also rebuilt churn"]
        precision = feb["rebuilt churners also official churn"]
        if recall >= 0.97 and precision >= 0.97:
            notes.append(
                f"The script reproduces February's official labels: {pct(recall)} of official churners are churn "
                f"under the script and {pct(precision)} of the script's churners are official churners. "
                "train.csv follows the published rule, so the jump in churn from February to March is not "
                "a labelling artefact."
            )
        else:
            notes.append(
                f"The script does not reproduce February's official labels: only {pct(recall)} of official churners "
                f"are churn under the script ({pct(precision)} the other way). The published February labels "
                "follow a different rule from the script, so they may not be like for like with March's. "
                "Compare by training on rebuilt labels: --train_snapshots feb_rebuilt."
            )
        notes.append(
            f"Users in train.csv the script does not label for February: {int(feb['official only']):,}; "
            f"labelled by the script but missing from train.csv: {int(feb['rebuilt only']):,}."
        )
    jan = table[table["month"] == "2017-01"]
    if not jan.empty and "rebuilt users" in jan and jan["rebuilt users"].notna().all():
        notes.append(
            f"January 2017: {int(jan['rebuilt users'].iloc[0]):,} users, churn {pct(jan['rebuilt churn rate'].iloc[0])}. "
        )
    notes.append(
        "March cannot be rebuilt: deciding a March expiry needs April transactions, which the data does not include."
    )
    bullet_block(LOG, "What this suggests:", notes)
    return {"table": table, "notes": notes, "outcomes": outcome_table if not outcome_table.empty else None}


# --------------------------------------------------------------------------- 2b. truncation


def _compare(official: pd.DataFrame, rebuilt: pd.DataFrame) -> dict:
    both = official.merge(rebuilt[[ID_COL, "is_churn"]].rename(columns={"is_churn": "rebuilt"}), on=ID_COL)
    churners, rebuilt_churn = both[TARGET_COL] == 1, both["rebuilt"] == 1
    return {
        "overlap users": len(both),
        "official churn rate": float(both[TARGET_COL].mean()),
        "rebuilt churn rate": float(both["rebuilt"].mean()),
        "agreement": float((both[TARGET_COL] == both["rebuilt"]).mean()),
        "official churners also rebuilt churn": float(rebuilt_churn[churners].mean())
        if churners.any()
        else float("nan"),
        "rebuilt churners also official churn": float(churners[rebuilt_churn].mean())
        if rebuilt_churn.any()
        else float("nan"),
        "disagreements": int((both[TARGET_COL] != both["rebuilt"]).sum()),
    }


def check_truncation(con, run_dir: str, LOG, plots: bool = True) -> dict:
    """Were the official labels made before all renewals were in the data?

    (a) Rebuild February with renewals cut at 28 February (the end of the original release) and see
        whether that reproduces train.csv better than the full data does.
    (b) Churn by expiry day for each label set. A label made too early marks late-month expiries as
        churn because their renewal had not happened yet: churn climbs at the end of the month.
        This is the only check possible for March, which cannot be rebuilt.
    """
    feb = LabelMonth(2017, 2)
    official_feb = query_df(con, "SELECT msno, is_churn FROM labels_feb")
    official_feb[ID_COL] = official_feb[ID_COL].astype(str)

    rows = []
    for label, until in (("all data (to 2017-03-31)", None), ("cut at 2017-02-28", to_days("2017-02-28"))):
        rebuilt = query_df(con, labels_sql(feb, future_until=until))
        rebuilt[ID_COL] = rebuilt[ID_COL].astype(str)
        row = {"February rebuilt with": label, **_compare(official_feb, rebuilt)}
        rows.append(row)
        log_stage(
            LOG,
            "METRICS",
            COMPONENT,
            check="truncation",
            rebuild=label,
            **{
                k.replace(" ", "_"): (round(v, 6) if isinstance(v, float) else v)
                for k, v in row.items()
                if k != "February rebuilt with"
            },
        )
    table = pd.DataFrame(rows)
    _save(table, run_dir, "truncation", LOG)
    print_table(
        LOG,
        table,
        "train.csv against the script, with and without March transactions",
        formats={
            c: pct
            for c in [
                "official churn rate",
                "rebuilt churn rate",
                "agreement",
                "official churners also rebuilt churn",
                "rebuilt churners also official churn",
            ]
        },
    )

    # (b) churn by expiry day, on each month's script candidates (so every user has an expiry date)
    series = {}
    sources = [
        ("jan rebuilt", "labels_jan", LabelMonth(2017, 1)),
        ("feb official", "labels_feb", feb),
        ("feb rebuilt", "labels_feb_rebuilt", feb),
        ("mar official", "labels_mar", LabelMonth(2017, 3)),
    ]
    from src.db import table_exists

    for name, table_name, month in sources:
        if not table_exists(con, table_name):
            log_stage(LOG, "WARN", COMPONENT, check="truncation", series=name, skipped=f"{table_name} not found")
            continue
        df = query_df(
            con,
            f"""
            SELECT c.last_expire - {month.start} + 1 AS day, l.is_churn
            FROM ({candidates_sql(month)}) c JOIN {table_name} l ON l.msno = c.msno""",
        )
        series[name] = df.groupby("day")["is_churn"].agg(["mean", "size"])

    by_day = pd.DataFrame({"expiry day": range(1, 32)})
    counts = pd.DataFrame({"expiry day": range(1, 32)})
    for name, g in series.items():
        by_day[name] = by_day["expiry day"].map(g["mean"])
        counts[name] = counts["expiry day"].map(g["size"])
    by_day = by_day.dropna(how="all", subset=list(series)).reset_index(drop=True)
    _save(by_day, run_dir, "churn_by_expiry_day", LOG)
    _save(counts, run_dir, "churn_by_expiry_day_counts", LOG)

    # late (last 7 days of the month) against early (days 1-10)
    ratios = []
    for name, g in series.items():
        last_day = int(g.index.max())
        early = g.loc[g.index <= 10]
        late = g.loc[g.index > last_day - 7]
        early_rate = float((early["mean"] * early["size"]).sum() / early["size"].sum())
        late_rate = float((late["mean"] * late["size"]).sum() / late["size"].sum())
        ratios.append(
            {
                "labels": name,
                "churn days 1-10": early_rate,
                "churn last 7 days": late_rate,
                "late / early": late_rate / early_rate if early_rate else float("nan"),
            }
        )
    ratio_table = pd.DataFrame(ratios)
    _save(ratio_table, run_dir, "truncation_late_early", LOG)
    print_table(
        LOG,
        ratio_table,
        "Churn for memberships expiring early against late in the month",
        formats={"churn days 1-10": pct, "churn last 7 days": pct, "late / early": lambda v: f"{v:.2f}x"},
    )
    if plots and series:
        from src.figures import churn_by_expiry_day, write_html

        path = os.path.join(run_dir, "plots", "diagnostics_churn_by_expiry_day.html")
        if write_html(churn_by_expiry_day(by_day), path, LOG):
            log_stage(LOG, "ARTIFACT", COMPONENT, plot=path)

    notes = []
    full, cut = table.iloc[0], table.iloc[1]
    if cut["official churners also rebuilt churn"] >= 0.97 and cut["rebuilt churners also official churn"] >= 0.97:
        notes.append(
            f"Cutting renewals at 28 February reproduces train.csv: {pct(cut['agreement'])} agreement, "
            f"{pct(cut['official churners also rebuilt churn'])} of official churners matched "
            f"(against {pct(full['official churners also rebuilt churn'])} with all the data). train.csv was "
            "labelled before March renewals existed, so some of its churners renewed."
        )
    elif cut["official churners also rebuilt churn"] > full["official churners also rebuilt churn"] + 0.05:
        notes.append(
            f"Cutting renewals at 28 February moves the rebuild much closer to train.csv "
            f"({pct(full['official churners also rebuilt churn'])} to {pct(cut['official churners also rebuilt churn'])} "
            "of official churners matched), but not all the way: early labelling explains most, not all, of it."
        )
    else:
        notes.append(
            "Cutting renewals at 28 February does not bring the rebuild closer to train.csv, so early labelling "
            "is not the explanation."
        )
    by_name = {r["labels"]: r for r in ratios}
    for name in ("feb official", "mar official"):
        if name in by_name:
            r = by_name[name]
            notes.append(
                f"{name}: churn is {r['late / early']:.2f}x higher for expiries in the last week than in days 1-10 "
                f"({pct(r['churn last 7 days'])} against {pct(r['churn days 1-10'])})."
            )
    rebuilt_ratios = [
        by_name[n]["late / early"]
        for n in ("jan rebuilt", "feb rebuilt")
        if n in by_name and by_name[n]["late / early"] == by_name[n]["late / early"]
    ]  # skip NaN
    if rebuilt_ratios and "mar official" in by_name:
        reference = max(rebuilt_ratios)
        mar = by_name["mar official"]["late / early"]
        notes.append(
            f"Rebuilt labels, which saw every renewal, reach at most {reference:.2f}x. "
            + (
                "March's official labels climb well above that, which looks like the same early labelling."
                if mar > reference * 1.5
                else "March's official labels stay close to that, so March does not show the early-labelling pattern."
            )
        )
    bullet_block(LOG, "What this suggests:", notes)
    return {"table": table, "late_early": ratio_table, "by_day": by_day, "notes": notes}


# --------------------------------------------------------------------------- 3. segments + drift


def _segments(df: pd.DataFrame) -> dict[str, pd.Series]:
    auto = df["last_is_auto_renew"]
    cancel = df["last_is_cancel"]
    renew = np.select(
        [auto.isna(), (auto == 1) & (cancel == 1), (auto == 1) & (cancel != 1), (auto != 1) & (cancel == 1)],
        ["no transaction", "auto-renew, cancelled", "auto-renew", "manual, cancelled"],
        default="manual",
    )
    plan = np.select(
        [df["last_plan_days"].isna(), df["last_plan_days"] == 30], ["no transaction", "30 days"], default="other length"
    )
    expiry = pd.cut(
        df["days_to_expire"],
        bins=[-np.inf, -1, 9, 19, 31, np.inf],
        labels=["already expired", "0-9 days", "10-19 days", "20-31 days", "over 31 days"],
    ).astype(str)
    expiry = expiry.where(df["days_to_expire"].notna(), "no transaction")
    listening = pd.cut(
        df["logs_active_days_30d"], bins=[-1, 0, 9, 31], labels=["0 days", "1-9 days", "10+ days"]
    ).astype(str)
    return {
        "Renewal type (last transaction)": pd.Series(renew, index=df.index),
        "Plan length": pd.Series(plan, index=df.index),
        "Days to expiry at cutoff": expiry,
        "Listening days, last 30": listening,
    }


def check_segments(feb: pd.DataFrame, mar: pd.DataFrame, run_dir: str, LOG) -> dict:
    seg_feb, seg_mar = _segments(feb), _segments(mar)
    overall_change = mar[TARGET_COL].mean() - feb[TARGET_COL].mean()
    rows, split_rows = [], []
    for family in seg_feb:
        a = feb.groupby(seg_feb[family])[TARGET_COL].agg(["size", "mean"])
        b = mar.groupby(seg_mar[family])[TARGET_COL].agg(["size", "mean"])
        both = a.join(b, how="outer", lsuffix="_feb", rsuffix="_mar").fillna({"size_feb": 0, "size_mar": 0})
        both["share_feb"] = both["size_feb"] / len(feb)
        both["share_mar"] = both["size_mar"] / len(mar)
        for segment, r in both.iterrows():
            rows.append(
                {
                    "segment type": family,
                    "segment": segment,
                    "feb share": r["share_feb"],
                    "feb churn": r["mean_feb"],
                    "mar share": r["share_mar"],
                    "mar churn": r["mean_mar"],
                    "churn change": r["mean_mar"] - r["mean_feb"],
                }
            )
        # Split the overall change: same users churning differently vs a different mix of users.
        shared = both.dropna(subset=["mean_feb", "mean_mar"])
        rate_effect = float((shared["share_feb"] * (shared["mean_mar"] - shared["mean_feb"])).sum())
        split_rows.append(
            {
                "segment type": family,
                "overall change": overall_change,
                "from churn rates within segments": rate_effect,
                "from a different mix of users": overall_change - rate_effect,
            }
        )

    table = pd.DataFrame(rows)
    split = pd.DataFrame(split_rows)
    _save(table, run_dir, "segments", LOG)
    _save(split, run_dir, "segments_split", LOG)
    print_table(
        LOG,
        table,
        "Churn by segment, February against March",
        formats={"feb share": pct, "feb churn": pct, "mar share": pct, "mar churn": pct, "churn change": pp},
    )
    print_table(
        LOG,
        split,
        f"Where the overall change of {pp(overall_change)} comes from",
        formats={"overall change": pp, "from churn rates within segments": pp, "from a different mix of users": pp},
    )

    # Drift: standardised mean difference of every feature.
    cols = [c for c in feb.columns if c not in (ID_COL, TARGET_COL, *META_COLS)]
    drift = []
    for c in cols:
        a, b = feb[c].astype(float), mar[c].astype(float)
        sd = np.sqrt((a.var() + b.var()) / 2)
        drift.append(
            {
                "feature": c,
                "feb mean": a.mean(),
                "mar mean": b.mean(),
                "std difference": (b.mean() - a.mean()) / sd if sd and sd == sd else 0.0,
                "feb missing": a.isna().mean(),
                "mar missing": b.isna().mean(),
            }
        )
    drift = pd.DataFrame(drift).sort_values("std difference", key=np.abs, ascending=False).reset_index(drop=True)
    _save(drift, run_dir, "drift", LOG)
    print_table(
        LOG,
        drift.head(10),
        "Features whose average moved most (standardised difference)",
        formats={"feb missing": pct, "mar missing": pct, "std difference": lambda v: f"{v:+.3f}"},
    )

    notes = []
    biggest = table.dropna(subset=["churn change"]).assign(abs_change=lambda d: d["churn change"].abs())
    biggest = biggest[(biggest["feb share"] > 0.01) & (biggest["mar share"] > 0.01)].nlargest(3, "abs_change")
    for _, r in biggest.iterrows():
        notes.append(
            f"{r['segment type']} = {r['segment']}: churn {pct(r['feb churn'])} in February, "
            f"{pct(r['mar churn'])} in March ({pp(r['churn change'])})."
        )
    main_split = split.iloc[0]
    notes.append(
        f"By renewal type, {pp(main_split['from churn rates within segments'])} of the {pp(overall_change)} change comes "
        f"from the same kind of user churning at a different rate, and {pp(main_split['from a different mix of users'])} "
        "from a different mix of users. A large within-segment part points at the labels or real behaviour changing, "
        "not just who is in the file."
    )
    small_drift = float(drift["std difference"].abs().iloc[0]) < 0.1
    notes.append(
        "No feature average moved much (all under 0.1 standard deviations)."
        if small_drift
        else f"Largest feature shift: {drift['feature'].iloc[0]} ({drift['std difference'].iloc[0]:+.2f} standard deviations)."
    )
    bullet_block(LOG, "What this suggests:", notes)
    return {"table": table, "split": split, "drift": drift, "notes": notes}


# --------------------------------------------------------------------------- 4. expiry


def check_expiry(frames: dict, run_dir: str, LOG) -> dict:
    rows = []
    for name, df in frames.items():
        snap = SNAPSHOTS[name]
        start = from_days(snap.cutoff_day)
        month_len = calendar.monthrange(start.year, start.month)[1]
        d = df["days_to_expire"]
        flag = lambda c, df=df: (df[c] == 1) if c in df else pd.Series(True, index=df.index)  # noqa: E731
        rows.append(
            {
                "snapshot": name,
                "users": len(df),
                "in expiry population": float(flag("in_population_expiry").mean()),
                "in labeller population": float(flag("in_labeller_population").mean()),
                "no transaction": float(d.isna().mean()),
                "expired before month": float((d < 0).mean()),
                "expires in month": float(((d >= 0) & (d < month_len)).mean()),
                "expires later": float((d >= month_len).mean()),
                "median days": float(d.median()),
                "1st pct": float(d.quantile(0.01)),
                "99th pct": float(d.quantile(0.99)),
            }
        )
    table = pd.DataFrame(rows)
    _save(table, run_dir, "expiry", LOG)
    print_table(
        LOG,
        table,
        "Where each snapshot's memberships expire, relative to the cutoff",
        formats={
            c: pct
            for c in [
                "no transaction",
                "expired before month",
                "expires in month",
                "expires later",
                "in expiry population",
                "in labeller population",
            ]
        },
    )
    notes = []
    for _, r in table.iterrows():
        notes.append(
            f"{r['snapshot']}: {pct(r['in expiry population'])} of listed users have their last expiry inside "
            f"the month (kept by --population expiry); {pct(r['in labeller population'])} meet the script's own "
            "rule (--population labeller)."
        )
    notes.append(
        "Training and validation should look like the April test users: use the population whose April "
        "share is closest to 100%."
    )
    bullet_block(LOG, "What this suggests:", notes)
    return {"table": table, "notes": notes}


# --------------------------------------------------------------------------- report + runner


def _md_table(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            whole = df[c].dropna().apply(float.is_integer).all()
            df[c] = df[c].map(lambda v, whole=whole: "" if pd.isna(v) else (f"{int(v):,}" if whole else f"{v:.4f}"))
        else:
            df[c] = df[c].map(lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else v)
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.astype(str).values]
    return "\n".join([header, sep, *body])


def write_report(results: dict, run_dir: str, LOG) -> str:
    titles = {
        "shift": "1. Within-month against across-month",
        "labels": "2. Official against rebuilt labels",
        "truncation": "3. Were the official labels made too early?",
        "segments": "4. Churn by segment and feature drift",
        "expiry": "5. Expiry dates against the label month",
    }
    parts = [
        "# Diagnostics report",
        "",
        "Rates are shown as fractions. Rate-matched scores use the scored month's true churn rate and are for "
        "diagnosis only.",
        "",
    ]
    for key in CHECKS:
        if key not in results:
            continue
        parts += [f"## {titles[key]}", ""]
        res = results[key]
        if "error" in res:
            parts += [f"Check failed: `{res['error']}`", ""]
            continue
        parts += [_md_table(res["table"]), ""]
        if key == "truncation":
            parts += [_md_table(res["late_early"]), ""]
        if key == "labels" and res.get("outcomes") is not None:
            parts += [_md_table(res["outcomes"]), ""]
        if key == "segments":
            parts += [_md_table(res["split"]), "", "Top feature shifts:", "", _md_table(res["drift"].head(10)), ""]
        parts += ["**What this suggests**", ""] + [f"- {n}" for n in res["notes"]] + [""]
    path = os.path.join(run_dir, "metrics", "diagnostics_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    log_stage(LOG, "ARTIFACT", COMPONENT, report=path)
    return path


def run_diagnostics(
    con,
    data_dir: str,
    run_dir: str,
    LOG,
    frame_format: str = "parquet",
    checks=CHECKS,
    sample_rows: int = 400_000,
    seed: int = 42,
    population: str = "expiry",
    plots: bool = True,
) -> dict:
    checks = [c for c in checks if c in CHECKS]
    results = {}
    with component_stage(LOG, COMPONENT, checks=",".join(checks), sample_rows=sample_rows):
        need_features = any(c in checks for c in ("shift", "segments", "expiry"))
        frames, compare = {}, {}
        if need_features:
            for name in ("feb", "mar", "apr"):
                frames[name] = _load_features(data_dir, name, frame_format, LOG)
            for name in ("feb", "mar"):
                df = frames[name]
                col = POPULATION_COLUMNS.get(population)
                if col and col in df:
                    df = df[df[col] == 1].reset_index(drop=True)
                compare[name] = df
            log_stage(
                LOG,
                "INFO",
                COMPONENT,
                population=population,
                feb_rows=len(compare["feb"]),
                mar_rows=len(compare["mar"]),
            )

        runners = {
            "shift": lambda: check_shift(compare["feb"], compare["mar"], run_dir, LOG, sample_rows, seed),
            "labels": lambda: check_labels(con, run_dir, LOG, data_dir=data_dir),
            "truncation": lambda: check_truncation(con, run_dir, LOG, plots=plots),
            "segments": lambda: check_segments(compare["feb"], compare["mar"], run_dir, LOG),
            "expiry": lambda: check_expiry(frames, run_dir, LOG),
        }
        names = {
            "shift": "within-month against across-month",
            "labels": "official against rebuilt labels",
            "truncation": "were the official labels made too early?",
            "segments": "churn by segment and feature drift",
            "expiry": "expiry dates against the label month",
        }
        titles = {k: f"Check {i} of {len(CHECKS)}: {names[k]}" for i, k in enumerate(CHECKS, 1)}
        for key in checks:
            stage_banner(LOG, titles[key])
            try:
                with component_stage(LOG, f"{COMPONENT}:{key}"):
                    results[key] = runners[key]()
            except Exception as exc:  # one failing check should not stop the others
                results[key] = {"error": f"{type(exc).__name__}: {exc}"}
                log_stage(LOG, "WARN", COMPONENT, check=key, skipped_after_error=f"{type(exc).__name__}: {exc}")

        results["report"] = write_report(results, run_dir, LOG)
    return results
