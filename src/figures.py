"""Results figures with Plotly: one consistent style, built from a finished run folder.

Each figure is first built as a plain dict (a Plotly figure spec: {"data": [...],
"layout": {...}}), so the data and wording can be tested without Plotly installed.
Plotly is only needed to write the files:

- runs/<run>/plots/<name>.html   interactive, hover for exact values (light theme)
- <export_dir>/<name>_light.png  and <name>_dark.png, for the README (GitHub shows
                                 the one matching the reader's theme)

Style (fixed for every figure)
------------------------------
- Colour has one meaning per figure, and blue is always "the champion model".
  The palette is colour-blind checked; every multi-series chart also has a legend
  or direct labels, so colour is never the only cue.
- Thin lines (2px), 8px+ markers with a ring in the background colour, rounded bar
  ends, hairline grid, no chart border.
- Titles state the finding and are computed from the numbers, so a new run
  rewrites them; the subtitle says what is plotted.

Build them for any finished run with scripts/make_figures.py (no retraining).
"""

from __future__ import annotations

import json
import os
import textwrap

import numpy as np
import pandas as pd

from src.constants import TARGET_COL
from src.logger import component_stage, log_stage

COMPONENT = "Figures"
FONT = 'Inter, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
WIDTH = 900
LEFT = 24  # title and legend start this many pixels from the figure's left edge
NOISE_NOTE = "run-to-run noise"

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "band": "rgba(42,120,214,0.10)",
        "blue": "#2a78d6",
        "orange": "#eb6834",
        "aqua": "#1baf7a",
        "yellow": "#eda100",
        "violet": "#6250d6",
        "neutral": "#a3a19a",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "band": "rgba(57,135,229,0.18)",
        "blue": "#3987e5",
        "orange": "#d95926",
        "aqua": "#199e70",
        "yellow": "#c98500",
        "violet": "#9085e9",
        "neutral": "#6f6e69",
    },
}

# Feature sources, in legend order, with their colour role.
SOURCES = (("Transactions", "blue"), ("Listening", "orange"), ("Calendar", "aqua"), ("Member", "yellow"))
SOURCE_PHRASES = {
    "Transactions": "payment history",
    "Listening": "listening logs",
    "Calendar": "the expiry date",
    "Member": "member details",
}

FEATURE_NAMES = {
    "days_to_expire": "Days until membership expires",
    "days_since_last_tx": "Days since last transaction",
    "days_since_first_tx": "Days since first transaction",
    "days_since_last_cancel": "Days since last cancellation",
    "days_since_last_log": "Days since last listened",
    "last_is_auto_renew": "Last payment: auto-renew",
    "last_is_cancel": "Last transaction: cancellation",
    "last_plan_days": "Last plan length (days)",
    "last_list_price": "Last plan list price",
    "last_amount_paid": "Last amount paid",
    "last_discount": "Last discount",
    "last_payment_method_id": "Last payment method",
    "tx_count": "Transactions, all time",
    "tx_count_90d": "Transactions, last 90 days",
    "tx_cancel_count": "Cancellations, all time",
    "tx_cancel_count_30d": "Cancellations, last 30 days",
    "tx_cancel_count_90d": "Cancellations, last 90 days",
    "tx_cancel_rate": "Share of transactions cancelled",
    "tx_auto_renew_rate": "Share of payments on auto-renew",
    "tx_discount_count": "Discounted payments",
    "tx_avg_paid": "Average amount paid",
    "tx_total_paid": "Total amount paid",
    "tx_avg_plan_days": "Average plan length (days)",
    "tx_n_payment_methods": "Payment methods used",
    "expire_day_of_month": "Expiry day of month",
    "expire_days_in_month": "Days in expiry month",
    "expire_days_to_month_end": "Days from expiry to month end",
    "expire_on_month_end": "Expires on last day of month",
    "logs_completion_30d": "Songs completed, share (30 days)",
    "logs_secs_per_active_day_30d": "Listening time per active day (30 days)",
    "logs_secs_trend_7_30": "Listening trend, 7 vs 30 days",
    "logs_secs_trend_30_90": "Listening trend, 30 vs 90 days",
    "city": "City",
    "bd": "Age",
    "age": "Age",
    "gender_code": "Gender",
    "registered_via": "Sign-up channel",
    "tenure_days": "Days since registration",
    "has_member_record": "Has member record",
}
_LOG_STEMS = {
    "logs_active_days": "Active listening days",
    "logs_secs": "Listening time",
    "logs_unq": "Unique songs",
    "logs_n100": "Songs played to the end",
    "logs_plays": "Songs played",
}


def feature_label(name: str) -> str:
    if name in FEATURE_NAMES:
        return FEATURE_NAMES[name]
    for stem, label in _LOG_STEMS.items():
        if name.startswith(stem + "_") and name.endswith("d") and name[len(stem) + 1 : -1].isdigit():
            return f"{label}, last {name[len(stem) + 1 : -1]} days"
    return name.replace("_", " ").capitalize()


def feature_source(name: str) -> str:
    if name.startswith("expire_"):
        return "Calendar"
    if name.startswith("logs_") or name == "days_since_last_log":
        return "Listening"
    if name.startswith(("tx_", "last_")) or name in (
        "days_to_expire",
        "days_since_last_tx",
        "days_since_first_tx",
        "days_since_last_cancel",
    ):
        return "Transactions"
    return "Member"


# --------------------------------------------------------------------------- shared layout


def _wrap(text: str, size: int, width: int = WIDTH) -> tuple[str, int]:
    """Plotly does not wrap titles; break the text into lines that fit the figure width."""
    chars = max(30, int((width - 120) / (size * 0.52)))
    lines = textwrap.wrap(text, chars) or [""]
    return "<br>".join(lines), len(lines)


def _axis(t: dict, title: str | None = None, **extra) -> dict:
    axis = {
        "showgrid": True,
        "gridcolor": t["grid"],
        "gridwidth": 1,
        "zeroline": False,
        "showline": True,
        "linecolor": t["axis"],
        "linewidth": 1,
        "ticks": "",
        "automargin": True,
        "tickfont": {"color": t["muted"], "size": 12},
    }
    if title:
        axis["title"] = {"text": title, "font": {"color": t["ink2"], "size": 12}, "standoff": 12}
    axis.update(extra)
    return axis


def _layout(t: dict, title: str, subtitle: str, height: int, legend: bool = False, **extra) -> dict:
    title_html, title_lines = _wrap(title, 19)
    sub_html, sub_lines = _wrap(subtitle, 13)
    top = 28 + 26 * title_lines + 19 * sub_lines + (34 if legend else 14)
    layout = {
        "template": "none",
        "width": WIDTH,
        "height": height + top,
        "paper_bgcolor": t["surface"],
        "plot_bgcolor": t["surface"],
        "font": {"family": FONT, "size": 13, "color": t["ink2"]},
        "title": {
            "text": f"<b>{title_html}</b>",
            "font": {"size": 19, "color": t["ink"]},
            "subtitle": {"text": sub_html, "font": {"size": 13, "color": t["ink2"]}},
            "x": 0,
            "xref": "container",
            "xanchor": "left",
            "y": 1,
            "yref": "container",
            "yanchor": "top",
            "pad": {"t": 22, "l": LEFT},
        },
        "margin": {"t": top, "l": 70, "r": 40, "b": 64},
        "showlegend": legend,
        "hoverlabel": {"bgcolor": t["surface"], "bordercolor": t["axis"], "font": {"color": t["ink"], "family": FONT}},
    }
    if legend:
        layout["legend"] = {
            "orientation": "h",
            "x": 0,
            "xref": "paper",
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "bottom",
            "font": {"color": t["ink2"], "size": 12},
            "bgcolor": "rgba(0,0,0,0)",
        }
    layout.update(extra)
    return layout


def _ring(t: dict, color: str, size: int = 10) -> dict:
    return {"color": color, "size": size, "line": {"color": t["surface"], "width": 2}}


# --------------------------------------------------------------------------- figure specs


def model_comparison(
    board: pd.DataFrame,
    champion: str,
    best: str,
    tolerance: float,
    prior_logloss: float | None,
    n_valid: int,
    valid_name: str,
    mode: str = "light",
) -> dict:
    """Dot per model on validation log loss, with the tie margin shaded."""
    t = THEMES[mode]
    df = board[board["model"] != "prior"].sort_values("log loss").reset_index(drop=True)
    best_ll = float(df["log loss"].min())
    span = float(df["log loss"].max()) - best_ll or 0.01
    tied = df[df["log loss"] <= best_ll + tolerance]["model"].tolist() if tolerance else [best]

    data = []
    for is_champ in (False, True):
        part = df[(df["model"] == champion) == is_champ]
        if part.empty:
            continue
        data.append(
            {
                "type": "scatter",
                "mode": "markers+text",
                "name": "champion" if is_champ else "other models",
                "x": part["log loss"].round(6).tolist(),
                "y": part["model"].tolist(),
                "text": [f"{v:.4f}" + ("   champion" if is_champ else "") for v in part["log loss"]],
                "textposition": "middle right",
                "cliponaxis": False,
                "textfont": {"color": t["ink"] if is_champ else t["ink2"], "size": 12},
                "marker": _ring(t, t["blue"] if is_champ else t["neutral"], 13 if is_champ else 11),
                "customdata": part["auc"].round(4).tolist(),
                "hovertemplate": "<b>%{y}</b><br>log loss %{x:.4f}<br>AUC %{customdata:.3f}<extra></extra>",
            }
        )

    if champion != best and len(tied) > 1:
        title = (
            f"{len(tied)} models are tied within {tolerance:g} log loss; {champion} is the champion "
            f"because it trains fastest"
        )
    else:
        title = f"{champion} has the lowest validation log loss"
    subtitle = (
        f"Log loss on {n_valid:,} {valid_name} users (lower is better). The shaded band is the "
        f"{tolerance:g} tie margin, about the size of {NOISE_NOTE}."
    )
    if prior_logloss is not None:
        subtitle += f" For scale, predicting the average rate for everyone scores {prior_logloss:.4f}."

    layout = _layout(
        t,
        title,
        subtitle,
        height=60 + 52 * len(df),
        xaxis=_axis(t, "Validation log loss", range=[best_ll - span * 0.08, best_ll + span * 1.35], tickformat=".3f"),
        yaxis=_axis(
            t,
            None,
            showgrid=False,
            showline=False,
            categoryorder="array",
            categoryarray=df["model"].tolist()[::-1],
            tickfont={"color": t["ink"], "size": 13},
        ),
        shapes=[
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": best_ll,
                "x1": best_ll + tolerance,
                "y0": 0,
                "y1": 1,
                "fillcolor": t["band"],
                "line": {"width": 0},
                "layer": "below",
            }
        ]
        if tolerance
        else [],
    )
    return {"data": data, "layout": layout}


def cumulative_gains(y, preds: dict, champion: str, valid_name: str, mode: str = "light") -> dict:
    """Share of churners reached when users are contacted in order of predicted risk."""
    t = THEMES[mode]
    y = np.asarray(y).astype(int)
    total = max(int(y.sum()), 1)
    pct = np.arange(0, 101)
    colors = [t["blue"], t["orange"]]
    data = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Random targeting",
            "x": [0, 100],
            "y": [0, 100],
            "line": {"color": t["muted"], "width": 1.5},
            "hoverinfo": "skip",
        }
    ]
    capture_at_10 = None
    for (name, p), color in zip(preds.items(), colors, strict=False):
        order = np.argsort(-np.asarray(p), kind="mergesort")
        cum = np.concatenate([[0], np.cumsum(y[order])]) / total
        idx = np.round(pct / 100 * len(y)).astype(int)
        curve = cum[idx] * 100
        if name == champion:
            capture_at_10 = float(curve[10])
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": name,
                "x": pct.tolist(),
                "y": np.round(curve, 2).tolist(),
                "line": {"color": color, "width": 2},
                "hovertemplate": f"<b>{name}</b><br>top %{{x}}% of users<br>%{{y:.1f}}% of churners<extra></extra>",
            }
        )
    annotations = []
    if capture_at_10 is not None:
        data.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": [10],
                "y": [capture_at_10],
                "showlegend": False,
                "marker": _ring(t, t["blue"], 11),
                "hoverinfo": "skip",
            }
        )
        annotations.append(
            {
                "x": 10,
                "y": capture_at_10,
                "xref": "x",
                "yref": "y",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": t["muted"],
                "ax": 70,
                "ay": 40,
                "xanchor": "left",
                "font": {"color": t["ink"], "size": 13},
                "text": f"<b>Top 10% → {capture_at_10:.0f}% of churners</b>",
            }
        )
        title = f"Targeting the top 10% of predicted risk reaches {capture_at_10:.0f}% of churners"
    else:
        title = "Share of churners reached by targeting the highest predicted risk first"
    others = [n for n in preds if n != champion]
    subtitle = (
        f"Cumulative gains on {valid_name}: users sorted by predicted churn risk from {champion}"
        + (f" and {', '.join(others)}" if others else "")
        + ", against random targeting."
    )
    layout = _layout(
        t,
        title,
        subtitle,
        height=470,
        legend=True,
        xaxis=_axis(t, "Users targeted, highest risk first", range=[0, 100], ticksuffix="%", dtick=10),
        yaxis=_axis(t, "Churners reached", range=[0, 101], ticksuffix="%", dtick=20),
        annotations=annotations,
    )
    return {"data": data, "layout": layout}


def calibration(y, preds: dict, valid_name: str, mode: str = "light", bins: int = 10) -> dict:
    """Mean predicted against observed churn, per tenth of predicted risk (log scales)."""
    from src.evaluate import calibration_table

    t = THEMES[mode]
    y = np.asarray(y).astype(int)
    colors = [t["blue"], t["orange"]]
    lo, hi = 1.0, 0.0
    data = []
    for (name, p), color in zip(preds.items(), colors, strict=False):
        table = calibration_table(y, p, bins)
        table = table[(table["mean_pred"] > 0) & (table["observed"] > 0)]
        lo = min(lo, float(table["mean_pred"].min()), float(table["observed"].min()))
        hi = max(hi, float(table["mean_pred"].max()), float(table["observed"].max()))
        data.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": name,
                "x": (table["mean_pred"] * 100).round(3).tolist(),
                "y": (table["observed"] * 100).round(3).tolist(),
                "line": {"color": color, "width": 2},
                "marker": _ring(t, color, 9),
                "customdata": table["n"].tolist(),
                "hovertemplate": f"<b>{name}</b><br>predicted %{{x:.2f}}%<br>observed %{{y:.2f}}%"
                "<br>%{customdata:,} users<extra></extra>",
            }
        )
    lo_pct, hi_pct = lo * 100 * 0.8, min(100.0, hi * 100 * 1.2)
    data.insert(
        0,
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Perfect calibration",
            "x": [lo_pct, hi_pct],
            "y": [lo_pct, hi_pct],
            "line": {"color": t["muted"], "width": 1.5},
            "hoverinfo": "skip",
        },
    )

    first = next(iter(preds.values()))
    mean_pred, base = float(np.mean(first)), float(y.mean())
    if mean_pred < base * 0.95:
        title = f"Predictions run low: {mean_pred:.1%} predicted against {base:.1%} actual churn"
    elif mean_pred > base * 1.05:
        title = f"Predictions run high: {mean_pred:.1%} predicted against {base:.1%} actual churn"
    else:
        title = f"Predicted churn matches what happened ({mean_pred:.1%} against {base:.1%})"
    subtitle = (
        f"Each point is a tenth of {valid_name} users grouped by predicted risk. Points above the line "
        "churned more than predicted."
    )
    ticks = [v for v in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if lo_pct <= v <= hi_pct]
    labels = [f"{v:g}%" for v in ticks]
    rng = [float(np.log10(lo_pct)), float(np.log10(hi_pct))]
    layout = _layout(
        t,
        title,
        subtitle,
        height=520,
        legend=True,
        xaxis=_axis(t, "Mean predicted churn (log scale)", type="log", range=rng, tickvals=ticks, ticktext=labels),
        yaxis=_axis(t, "Observed churn (log scale)", type="log", range=rng, tickvals=ticks, ticktext=labels),
    )
    return {"data": data, "layout": layout}


def base_rate_shift(
    train_rate: float,
    valid_rate: float,
    valid_pred: float,
    champion: str,
    train_name: str,
    valid_name: str,
    test_means: dict | None = None,
    mode: str = "light",
) -> dict:
    """The level problem: training month rate, prediction and actual rate for the validation month."""
    t = THEMES[mode]
    actual_x = [f"{train_name.capitalize()}<br>actual", f"{valid_name.capitalize()}<br>actual"]
    bar = {
        "type": "bar",
        "width": 0.5,
        "textposition": "outside",
        "cliponaxis": False,
        "texttemplate": "%{y:.1%}",
        "textfont": {"color": t["ink"], "size": 13},
    }
    data = [
        {
            **bar,
            "name": "Actual churn rate",
            "x": actual_x,
            "y": [train_rate, valid_rate],
            "marker": {"color": t["neutral"], "cornerradius": 4},
            "hovertemplate": "%{x}: %{y:.2%}<extra></extra>",
        },
        {
            **bar,
            "name": f"Predicted by {champion}",
            "x": [f"{valid_name.capitalize()}<br>predicted"],
            "y": [valid_pred],
            "marker": {"color": t["blue"], "cornerradius": 4},
            "hovertemplate": "%{x}: %{y:.2%}<extra></extra>",
        },
    ]
    gap = (valid_rate - valid_pred) * 100
    shift = (valid_rate - train_rate) * 100
    if gap > 0.2 and shift > gap / 2:
        title = (
            f"Predictions for {valid_name.capitalize()} are {gap:.1f} points too low because "
            f"{train_name.capitalize()} churned less"
        )
    elif gap < -0.2 and shift < gap / 2:
        title = (
            f"Predictions for {valid_name.capitalize()} are {-gap:.1f} points too high because "
            f"{train_name.capitalize()} churned more"
        )
    else:
        title = f"Actual and predicted churn rates, {train_name.capitalize()} and {valid_name.capitalize()}"
    subtitle = (
        "Churn rates move from month to month, and a model learns the level of the month it was "
        "trained on. The ranking of users still transfers."
    )
    if test_means:
        subtitle += (
            f" For April, the submission as trained averages {test_means['as_trained']:.1%}; the "
            f"recalibrated one, {test_means['recalibrated']:.1%}."
        )
    order = [actual_x[0], f"{valid_name.capitalize()}<br>predicted", actual_x[1]]
    layout = _layout(
        t,
        title,
        subtitle,
        height=420,
        legend=True,
        barmode="overlay",
        xaxis=_axis(
            t,
            None,
            showgrid=False,
            categoryorder="array",
            categoryarray=order,
            tickfont={"color": t["ink"], "size": 13},
        ),
        yaxis=_axis(
            t,
            "Churn rate",
            tickformat=".0%",
            rangemode="tozero",
            range=[0, max(train_rate, valid_rate, valid_pred) * 1.25],
        ),
    )
    return {"data": data, "layout": layout}


def feature_importance(table: pd.DataFrame, top: int = 15, mode: str = "light") -> dict:
    """Top features by mean |SHAP|, coloured by where the feature comes from."""
    t = THEMES[mode]
    df = table.sort_values("mean_abs_shap", ascending=False).head(top).copy()
    df["label"] = df["feature"].map(feature_label)
    df["source"] = df["feature"].map(feature_source)
    data = []
    for source, role in SOURCES:
        part = df[df["source"] == source]
        if part.empty:
            continue
        data.append(
            {
                "type": "bar",
                "orientation": "h",
                "name": source,
                "x": part["mean_abs_shap"].round(5).tolist(),
                "y": part["label"].tolist(),
                "marker": {"color": t[role], "cornerradius": 4},
                "texttemplate": "%{x:.3f}",
                "textposition": "outside",
                "cliponaxis": False,
                "textfont": {"color": t["ink2"], "size": 12},
                "customdata": part["feature"].tolist(),
                "hovertemplate": "<b>%{y}</b><br>%{customdata}<br>mean |SHAP| %{x:.4f}<extra></extra>",
            }
        )
    counts = df["source"].value_counts()
    lead = counts.index[0] if len(counts) else "Transactions"
    title = (
        f"{df['label'].iloc[0]} moves predictions most; {counts.iloc[0]} of the top {len(df)} features "
        f"come from {SOURCE_PHRASES[lead]}"
    )
    subtitle = (
        "Average size of each feature's effect on the predicted log-odds of churn (mean |SHAP| value), "
        "LightGBM on a sample of validation users."
    )
    layout = _layout(
        t,
        title,
        subtitle,
        height=40 + 30 * len(df),
        legend=True,
        barmode="overlay",
        bargap=0.35,
        xaxis=_axis(t, "Mean |SHAP| (log-odds)", range=[0, float(df["mean_abs_shap"].max()) * 1.18]),
        yaxis=_axis(
            t,
            None,
            showgrid=False,
            categoryorder="array",
            categoryarray=df["label"].tolist()[::-1],
            tickfont={"color": t["ink"], "size": 12},
        ),
    )
    layout["margin"]["l"] = 40  # automargin makes room for the feature names
    return {"data": data, "layout": layout}


def month_end_effect(frames: dict, mode: str = "light", min_users: int | None = None) -> dict | None:
    """Churn rate by the day of the month the membership expires, one line per month."""
    t = THEMES[mode]
    if min_users is None:  # 200 on the real data; smaller for the synthetic demo
        min_users = min(200, max(20, min(len(df) for df in frames.values()) // 200))
    colors = [t["aqua"], t["violet"]]
    data, annotations, last_day, overall, peak = [], [], None, None, None
    for (name, df), color in zip(frames.items(), colors, strict=False):
        g = df.groupby("expire_day_of_month")[TARGET_COL].agg(["mean", "size"]).reset_index()
        g = g[g["size"] >= min_users]
        if g.empty:
            continue
        data.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": name.capitalize(),
                "x": g["expire_day_of_month"].astype(int).tolist(),
                "y": g["mean"].round(5).tolist(),
                "line": {"color": color, "width": 2},
                "marker": _ring(t, color, 8),
                "customdata": g["size"].tolist(),
                "hovertemplate": f"<b>{name.capitalize()}</b>, day %{{x}}<br>churn %{{y:.1%}}"
                "<br>%{customdata:,} users<extra></extra>",
            }
        )
        end = g.iloc[-1]
        top = g.loc[g["mean"].idxmax()]
        if peak is None or float(top["mean"]) > peak[2]:
            peak = (name.capitalize(), int(top["expire_day_of_month"]), float(top["mean"]))
        days_in_month = int(df["expire_days_in_month"].max()) if "expire_days_in_month" in df else None
        if days_in_month and int(end["expire_day_of_month"]) == days_in_month:
            last_day, overall = float(end["mean"]), float(df[TARGET_COL].mean())
            last_name = name.capitalize()
    if not data:
        return None
    if last_day is not None and last_day < overall * 0.7:
        title = (
            f"Memberships ending on the last day of the month rarely churn: {last_day:.1%} in "
            f"{last_name} against {overall:.1%} overall"
        )
    else:
        title = "Churn rate by the day of the month the membership expires"
    subtitle = (
        f"Users whose membership expires in the label month, by expiry day. The highest rate is on "
        f"{peak[0]} {peak[1]} ({peak[2]:.0%}). Days with fewer than {min_users:,} users are left out."
    )
    layout = _layout(
        t,
        title,
        subtitle,
        height=440,
        legend=True,
        xaxis=_axis(
            t, "Day of the month the membership expires", range=[0.5, 31.5], tickvals=[1, 5, 10, 15, 20, 25, 28, 31]
        ),
        yaxis=_axis(t, "Churn rate", tickformat=".0%", rangemode="tozero"),
        annotations=annotations,
    )
    return {"data": data, "layout": layout}


def label_choice(table: pd.DataFrame, valid_name: str, mode: str = "light") -> dict:
    """Validation log loss for each set of training labels (from docs/results/label_comparison.csv)."""
    t = THEMES[mode]
    df = table.sort_values("logloss").reset_index(drop=True)
    best = df.iloc[0]
    colors = [t["blue"] if i == 0 else t["neutral"] for i in range(len(df))]
    data = [
        {
            "type": "bar",
            "x": df["labels"].tolist(),
            "y": df["logloss"].tolist(),
            "width": 0.5,
            "marker": {"color": colors, "cornerradius": 4},
            "texttemplate": "%{y:.4f}",
            "textposition": "outside",
            "cliponaxis": False,
            "textfont": {"color": t["ink"], "size": 13},
            "customdata": df["auc"].tolist() if "auc" in df else None,
            "hovertemplate": "<b>%{x}</b><br>log loss %{y:.4f}"
            + ("<br>AUC %{customdata:.3f}" if "auc" in df else "")
            + "<extra></extra>",
        }
    ]
    margin = float(df["logloss"].iloc[1] - best["logloss"]) if len(df) > 1 else 0.0
    name = best["labels"][0].lower() + best["labels"][1:]
    title = f"Training on the {name} labels beats every alternative by at least {margin:.3f} log loss"
    subtitle = (
        f"LightGBM with the same features and settings, validated on the official {valid_name} labels. "
        "Labels rebuilt with the organisers' script mark fewer users as churners."
    )
    layout = _layout(
        t,
        title,
        subtitle,
        height=380,
        xaxis=_axis(t, None, showgrid=False, tickfont={"color": t["ink"], "size": 13}),
        yaxis=_axis(
            t,
            "Validation log loss (lower is better)",
            rangemode="tozero",
            range=[0, float(df["logloss"].max()) * 1.18],
            tickformat=".2f",
        ),
    )
    return {"data": data, "layout": layout}


def churn_by_expiry_day(table: pd.DataFrame, mode: str = "light") -> dict:
    """Diagnostics: churn by expiry day for each label set (labels made too early rise at month end)."""
    t = THEMES[mode]
    colors = [t["blue"], t["orange"], t["aqua"], t["yellow"]]
    data = []
    for name, color in zip([c for c in table.columns if c != "expiry day"], colors, strict=False):
        part = table[["expiry day", name]].dropna()
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": name,
                "x": part["expiry day"].astype(int).tolist(),
                "y": part[name].round(5).tolist(),
                "line": {"color": color, "width": 2},
                "hovertemplate": f"<b>{name}</b>, day %{{x}}<br>churn %{{y:.1%}}<extra></extra>",
            }
        )
    title = "Churn by the day of the month the membership expires, for each label set"
    subtitle = (
        "Labels made before all renewals were recorded would show churn climbing at the end of the month, "
        "where renewals arrive last."
    )
    layout = _layout(
        t,
        title,
        subtitle,
        height=420,
        legend=True,
        xaxis=_axis(
            t, "Day of the month the membership expires", range=[0.5, 31.5], tickvals=[1, 5, 10, 15, 20, 25, 28, 31]
        ),
        yaxis=_axis(t, "Churn rate", tickformat=".0%", rangemode="tozero"),
    )
    return {"data": data, "layout": layout}


def write_html(spec: dict, path: str, LOG=None) -> bool:
    """Write one interactive figure. Returns False (with a warning) when Plotly is not installed."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        if LOG is not None:
            log_stage(LOG, "WARN", COMPONENT, skipped="plotly is not installed (pip install plotly)", path=path)
        return False
    name = os.path.splitext(os.path.basename(path))[0]
    go.Figure(spec).write_html(
        path,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            # The camera button saves a sharp PNG even where Kaleido cannot run.
            "toImageButtonOptions": {"format": "png", "filename": name, "scale": 2},
        },
    )
    return True


# --------------------------------------------------------------------------- building from a run folder


def _load_inputs(run_dir: str, data_dir: str, frame_format: str, LOG) -> dict:
    from src.model import apply_population, load_snapshot

    with open(os.path.join(run_dir, "metrics", "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    board = pd.read_csv(os.path.join(run_dir, "metrics", "leaderboard.csv"))
    population = summary.get("population", "expiry")
    valid_name = summary.get("valid_snapshot", "mar")
    train_names = summary.get("train_snapshots", ["feb"])
    valid = apply_population(load_snapshot(data_dir, valid_name, frame_format, LOG), population, valid_name, LOG)
    train = {
        n: apply_population(load_snapshot(data_dir, n, frame_format, LOG), population, n, LOG) for n in train_names
    }

    preds = {}
    folder = os.path.join(run_dir, "artifacts", "marathon")
    for name in board["model"]:
        path = os.path.join(folder, f"{name}_valid_pred.npy")
        if os.path.exists(path):
            preds[name] = np.load(path)
    ens = summary.get("models", {}).get("ensemble", {})
    if "ensemble" not in preds and ens.get("members") and all(m in preds for m in ens["members"]):
        preds["ensemble"] = np.mean([preds[m] for m in ens["members"]], axis=0)
    for name, p in preds.items():
        if len(p) != len(valid):
            raise ValueError(
                f"Saved {name} predictions have {len(p):,} rows but {valid_name} has {len(valid):,}. "
                "Were the features rebuilt after this run? Use the same data folder as the run."
            )

    shap_path = os.path.join(run_dir, "metrics", "lightgbm_shap_importance.csv")
    return {
        "summary": summary,
        "board": board,
        "valid": valid,
        "train": train,
        "preds": preds,
        "shap": pd.read_csv(shap_path) if os.path.exists(shap_path) else None,
        "valid_name": valid_name,
        "train_names": train_names,
    }


MONTH_NAMES = {"jan": "January", "feb": "February", "feb_rebuilt": "February", "mar": "March", "apr": "April"}


def build_specs(inputs: dict, mode: str = "light", label_comparison: pd.DataFrame | None = None) -> dict:
    """Every figure the inputs allow, as {name: spec}."""
    s, board, valid, preds = inputs["summary"], inputs["board"], inputs["valid"], inputs["preds"]
    valid_name = MONTH_NAMES.get(inputs["valid_name"], inputs["valid_name"])
    y = valid[TARGET_COL].to_numpy()
    champion = s["champion"]
    best = s.get("best_logloss_model") or board.sort_values("log loss")["model"].iloc[0]
    tolerance = float(s.get("champion_tolerance", 0.0) or 0.0)
    prior = board.loc[board["model"] == "prior", "log loss"]
    specs = {}

    specs["model_comparison"] = model_comparison(
        board, champion, best, tolerance, float(prior.iloc[0]) if len(prior) else None, len(valid), valid_name, mode
    )
    if champion in preds:
        shown = {champion: preds[champion]}
        if "logistic" in preds and champion != "logistic":
            shown["logistic"] = preds["logistic"]
        specs["cumulative_gains"] = cumulative_gains(y, shown, champion, valid_name, mode)
        specs["calibration"] = calibration(y, shown, valid_name, mode)
        train_all = pd.concat(inputs["train"].values(), ignore_index=True)
        train_label = " + ".join(MONTH_NAMES.get(n, n) for n in inputs["train_names"])
        specs["base_rate_shift"] = base_rate_shift(
            float(train_all[TARGET_COL].mean()),
            float(y.mean()),
            float(np.mean(preds[champion])),
            champion,
            train_label,
            valid_name,
            s.get("test_mean_pred"),
            mode,
        )
    if inputs["shap"] is not None:
        specs["feature_importance"] = feature_importance(inputs["shap"], mode=mode)
    months = {MONTH_NAMES.get(n, n): df for n, df in inputs["train"].items()}
    months[valid_name] = valid
    if all("expire_day_of_month" in df for df in months.values()):
        spec = month_end_effect(months, mode)
        if spec is not None:
            specs["month_end_effect"] = spec
    if label_comparison is not None and len(label_comparison) > 1:
        specs["label_choice"] = label_choice(label_comparison, valid_name, mode)
    return specs


def build_figures(
    run_dir: str,
    data_dir: str,
    LOG,
    frame_format: str = "parquet",
    export_dir: str | None = None,
    label_comparison: str | None = None,
    html: bool = True,
) -> dict:
    """Write interactive HTML into the run's plots/ folder and, with export_dir, light/dark PNGs."""
    import plotly.graph_objects as go

    with component_stage(LOG, COMPONENT, run_dir=run_dir, export_dir=export_dir):
        inputs = _load_inputs(run_dir, data_dir, frame_format, LOG)
        labels = pd.read_csv(label_comparison) if label_comparison and os.path.exists(label_comparison) else None
        written = {}
        light = build_specs(inputs, "light", labels)
        if html:
            for name, spec in light.items():
                path = os.path.join(run_dir, "plots", f"{name}.html")
                write_html(spec, path)
                log_stage(LOG, "ARTIFACT", COMPONENT, figure=name, path=path)
                written[f"{name}.html"] = path
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
            dark = build_specs(inputs, "dark", labels)
            try:
                for mode, specs in (("light", light), ("dark", dark)):
                    for name, spec in specs.items():
                        path = os.path.join(export_dir, f"{name}_{mode}.png")
                        go.Figure(spec).write_image(path, scale=2)
                        log_stage(LOG, "ARTIFACT", COMPONENT, figure=name, mode=mode, path=path)
                        written[f"{name}_{mode}.png"] = path
            except Exception as exc:  # usually Kaleido's Chrome missing, or missing system libraries for it
                reason = " ".join(f"{type(exc).__name__}: {exc}".split())  # one line, so it survives `tail`
                log_stage(
                    LOG,
                    "WARN",
                    COMPONENT,
                    png="skipped",
                    reason=reason[:600],
                    fix="see 'PNG export' in docs/FIGURES.md",
                )
        log_stage(LOG, "RESULT", COMPONENT, figures=len(light), files=len(written))
        return written
