"""Training: the model marathon, out-of-time validation, seed-averaged final fit, two submissions.

Validation design
-----------------
Train on the February snapshot, validate on March. March is a later month than
anything the models trained on, which is the same situation as scoring April,
so the validation score is an honest estimate (a random split would mix months
and flatter the models). Early stopping and tuning use a holdout of the
training users; March is only ever scored.

The marathon
------------
Every model in src/marathon.py is fitted and scored the same way, then ranked
by March log loss. An ensemble (plain average of the best model from each
family) is scored alongside. The lowest log loss wins.

Final fit
---------
The champion is refit on the training months + March. Boosted models use
best_iteration * 1.1 rounds (more data supports slightly more trees) and are
averaged over several seeds. If the ensemble wins, each member is refit.

Two submissions
---------------
submission.csv is the champion as trained. submission_recalibrated.csv shifts
the same predictions on the logit scale so their mean equals a target churn
rate (by default the validation month's), because the churn rate moves from
month to month and log loss punishes a wrong overall level. The ranking, and
so AUC, is the same in both files. Which one scores better on Kaggle is the
experiment.
"""

import json
import os
import time

import numpy as np
import pandas as pd

from src.console import bullet_block, num4, pct, print_table, raw
from src.constants import (
    DEFAULT_POPULATION,
    ID_COL,
    META_COLS,
    POPULATION_COLUMNS,
    POPULATIONS,
    SNAPSHOTS,
    TARGET_COL,
    TEST_SNAPSHOT,
    TRAIN_SNAPSHOTS,
    VALID_SNAPSHOT,
)
from src.evaluate import classification_metrics, evaluate_predictions, match_rate, write_summary
from src.features import CATEGORICAL_FEATURES
from src.logger import component_stage, log_stage
from src.marathon import (
    DEFAULT_LGB_PARAMS,
    ENSEMBLE_NAME,
    MODEL_NAMES,
    ensemble_members,
    load_cached,
    registry,
    run_one,
    save_cached,
    tune,
)
from src.storage import features_path, read_frame

__all__ = ["DEFAULT_LGB_PARAMS", "MODEL_NAMES", "DEFAULT_MODELS", "parse_model_list", "run_training"]

COMPONENT = "Training"
DEFAULT_MODELS = MODEL_NAMES
DEFAULT_TUNE_MODELS = ("lightgbm", "xgboost", "catboost")


def parse_model_list(value: str, allowed=MODEL_NAMES, flag: str = "--models") -> list:
    models = [m.strip().lower() for m in (value or "").split(",") if m.strip()]
    unknown = [m for m in models if m not in allowed]
    if unknown or not models:
        raise ValueError(f"{flag} must be a comma-separated subset of {tuple(allowed)}; got '{value}'.")
    return list(dict.fromkeys(models))


# --------------------------------------------------------------------------- data


def load_snapshot(data_dir: str, name: str, frame_format: str, LOG) -> pd.DataFrame:
    path = features_path(data_dir, name, frame_format)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Build it with: --mode features --snapshots {name}")
    df = read_frame(path)
    log_stage(LOG, "INFO", COMPONENT, action="load_features", snapshot=name, rows=len(df), path=path)
    return df


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in (ID_COL, TARGET_COL, *META_COLS)]


def apply_population(df: pd.DataFrame, population: str, name: str, LOG) -> pd.DataFrame:
    """Keep only users who look like the April test users (see POPULATIONS in src/constants.py)."""
    if population not in POPULATIONS:
        raise ValueError(f"Unknown population '{population}'. Use one of {POPULATIONS}.")
    if population == "all":
        return df
    col = POPULATION_COLUMNS[population]
    if col not in df.columns:
        raise ValueError(f"Feature tables have no {col} column. Rebuild them with --mode features.")
    keep = df[col] == 1
    dropped = df[~keep]
    log_stage(
        LOG,
        "INFO",
        COMPONENT,
        snapshot=name,
        population=population,
        kept=int(keep.sum()),
        dropped=len(dropped),
        kept_churn_rate=round(float(df.loc[keep, TARGET_COL].mean()), 5),
        dropped_churn_rate=None if dropped.empty else round(float(dropped[TARGET_COL].mean()), 5),
    )
    return df[keep].reset_index(drop=True)


def split_holdout(df: pd.DataFrame, share: float, seed: int):
    """Random user holdout from the training months, for early stopping (never the validation month)."""
    rng = np.random.default_rng(seed)
    hold = rng.random(len(df)) < share
    return df[~hold].reset_index(drop=True), df[hold].reset_index(drop=True)


def _check_snapshots(train_snapshots, valid_snapshot):
    for name in [*train_snapshots, valid_snapshot]:
        if name not in SNAPSHOTS or not SNAPSHOTS[name].labelled:
            raise ValueError(
                f"{name} is not a labelled snapshot. Labelled: {[n for n, s in SNAPSHOTS.items() if s.labelled]}."
            )
    months = {SNAPSHOTS[n].cutoff for n in train_snapshots}
    if len(months) != len(train_snapshots) or SNAPSHOTS[valid_snapshot].cutoff in months:
        raise ValueError("Each training month may appear once, and the validation month cannot also be trained on.")


def _jsonable(obj):
    """Round-trip through JSON so fingerprints compare equal to what was saved (tuples become lists)."""
    return json.loads(json.dumps(obj, default=str))


# --------------------------------------------------------------------------- explainability


def explain_lightgbm(
    booster, X: pd.DataFrame, num_iteration: int, run_dir: str, LOG, sample: int = 20_000, seed: int = 42
) -> pd.DataFrame:
    """Global importance from SHAP values (LightGBM's built-in TreeSHAP, no extra dependency).

    Mean |SHAP| answers "how much does this feature move predictions on average",
    which is a fairer ranking than split counts or raw gain.
    """
    if len(X) > sample:
        X = X.sample(sample, random_state=seed)
    contrib = np.asarray(booster.predict(X, num_iteration=num_iteration, pred_contrib=True))
    mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)  # last column is the base value
    table = (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "mean_abs_shap": mean_abs,
                "gain": booster.feature_importance(importance_type="gain", iteration=num_iteration),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    path = os.path.join(run_dir, "metrics", "lightgbm_shap_importance.csv")
    table.to_csv(path, index=False)
    log_stage(LOG, "ARTIFACT", "LightGBM", shap_importance=path, sample_rows=len(X))
    log_stage(LOG, "INFO", "LightGBM", top_features=",".join(table["feature"].head(8)))

    return table


# --------------------------------------------------------------------------- marathon


def _score(result: dict, y_valid, valid_name: str, run_dir: str, LOG) -> dict:
    """Validation metrics plus the oracle rate-matched log loss (what a perfect level correction would give)."""
    pred = result["valid_pred"]
    metrics = evaluate_predictions(y_valid, pred, result["name"], f"valid_{valid_name}", run_dir, LOG)
    matched = classification_metrics(y_valid, match_rate(pred, float(np.mean(y_valid))))
    metrics["rate_matched_logloss"] = matched["logloss"]
    return metrics


def _run_marathon(
    specs: dict,
    requested: list,
    fit_df,
    stop_df,
    valid,
    cols,
    cfg,
    seed,
    run_dir,
    LOG,
    valid_name: str,
    base_fp: dict,
    tune_models: list,
    ensemble: bool,
):
    y_valid = valid[TARGET_COL].to_numpy()
    results, skipped, failed = {}, {}, {}

    def attempt(spec, name, params_fn, fp):
        cached = load_cached(run_dir, name, fp)
        if cached is not None:
            log_stage(
                LOG, "INFO", COMPONENT, model=name, resumed=True, valid_logloss=round(cached["valid"]["logloss"], 6)
            )
            results[name] = cached
            return
        component = spec.label if name == spec.name else f"{spec.label}_tuned"
        try:
            with component_stage(
                LOG,
                component,
                model=name,
                fit_rows=len(fit_df),
                early_stopping_rows=len(stop_df),
                valid_rows=len(valid),
                features=len(cols),
                seed=seed,
            ):
                params = params_fn()
                res = run_one(
                    spec, name, params, fit_df, stop_df, valid, cols, {**cfg, "component": component}, seed, LOG
                )
                if res["best_iteration"] is not None:
                    log_stage(
                        LOG,
                        "INFO",
                        component,
                        best_iteration=res["best_iteration"],
                        stopped_early=res["best_iteration"] < cfg["num_boost_round"],
                    )
                path = spec.save(res["model"], os.path.join(run_dir, "artifacts", f"{name}_valid"))
                log_stage(LOG, "CHECKPOINT", component, path=path, best_iteration=res["best_iteration"])
                res["valid"] = _score(res, y_valid, valid_name, run_dir, LOG)
                if name == "lightgbm":
                    try:
                        explain_lightgbm(res["model"], valid[cols], res["best_iteration"], run_dir, LOG, seed=seed)
                    except Exception as exc:  # explanation is a bonus, never a reason to stop
                        log_stage(LOG, "WARN", "LightGBM", shap="skipped", reason=f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            failed[name] = f"{type(exc).__name__}: {exc}"
            log_stage(
                LOG, "WARN", COMPONENT, model=name, skipped="failed, the marathon carries on", reason=failed[name]
            )
            return
        save_cached(run_dir, name, res, fp)
        res.pop("model", None)
        results[name] = res

    for name in requested:
        spec = specs[name]
        if not spec.available():
            skipped[name] = f"{spec.library} is not installed (pip install {spec.library})"
            log_stage(LOG, "WARN", COMPONENT, model=name, skipped=skipped[name])
            continue
        attempt(
            spec, name, lambda s=spec: dict(s.defaults), _jsonable({**base_fp, "model": name, "params": spec.defaults})
        )

    for name in tune_models:
        spec = specs[name]
        if name not in results or not spec.space:
            continue
        tuned = f"{name}_tuned"
        fp = _jsonable(
            {
                **base_fp,
                "model": tuned,
                "defaults": spec.defaults,
                "tune_trials": cfg["tune_trials"],
                "tune_rows": cfg["tune_rows"],
            }
        )
        attempt(spec, tuned, lambda s=spec: tune(s, fit_df, stop_df, cols, cfg, seed, LOG), fp)

    if ensemble:
        members = ensemble_members(results, 3)
        if len(members) >= 2:
            pred = np.mean([np.asarray(results[m]["valid_pred"]) for m in members], axis=0)
            res = {
                "name": ENSEMBLE_NAME,
                "family": "ensemble",
                "role": "ensemble",
                "spec": None,
                "members": members,
                "params": None,
                "best_iteration": None,
                "valid_pred": pred,
                "train_seconds": round(sum(results[m]["train_seconds"] for m in members), 1),
            }
            log_stage(LOG, "INFO", COMPONENT, model=ENSEMBLE_NAME, members=",".join(members))
            res["valid"] = _score(res, y_valid, valid_name, run_dir, LOG)
            results[ENSEMBLE_NAME] = res
        else:
            log_stage(
                LOG,
                "INFO",
                COMPONENT,
                model=ENSEMBLE_NAME,
                skipped="needs at least two candidate families",
                families=len(members),
            )
    return results, skipped, failed


# --------------------------------------------------------------------------- reporting


def build_leaderboard(results: dict, champion_name: str) -> pd.DataFrame:
    """One row per model, ranked by validation log loss, with skill relative to the prior."""
    prior_ll = results.get("prior", {}).get("valid", {}).get("logloss")
    rows = []
    for name, res in results.items():
        v = res["valid"]
        rows.append(
            {
                "model": name,
                "family": res.get("family"),
                "log loss": v["logloss"],
                "rate-matched": v.get("rate_matched_logloss"),
                "skill vs prior": (1 - v["logloss"] / prior_ll) if prior_ll else None,
                "auc": v.get("auc"),
                "pr auc": v.get("pr_auc"),
                "brier": v.get("brier"),
                "mean pred": v.get("mean_pred"),
                "churn rate": v.get("base_rate"),
                "top 10% capture": v.get("top_decile_capture"),
                "rounds": res.get("best_iteration"),
                "train time": res.get("train_seconds"),
                "champion": name == champion_name,
            }
        )
    table = pd.DataFrame(rows).sort_values("log loss", kind="mergesort").reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def marathon_notes(
    results: dict,
    champion: str,
    skipped: dict,
    failed: dict,
    valid_name: str,
    best: str | None = None,
    tolerance: float = 0.0,
) -> list:
    """Plain-language reading of the leaderboard."""
    best = best or champion
    order = sorted(results, key=lambda n: results[n]["valid"]["logloss"])
    champ = results[champion]["valid"]
    notes = []
    prior = results.get("prior", {}).get("valid")
    if champion == best:
        text = f"{champion} wins with log loss {champ['logloss']:.4f} on {valid_name}"
        if prior and champion != "prior":
            text += f", {(1 - champ['logloss'] / prior['logloss']) * 100:.1f}% below the prior ({prior['logloss']:.4f})"
        notes.append(text + ".")
        if len(order) > 1:
            runner = order[1]
            gap = results[runner]["valid"]["logloss"] - champ["logloss"]
            notes.append(
                f"Runner-up is {runner}, {gap:.4f} behind. Gaps under about 0.001 are within run-to-run noise."
            )
    else:
        gap = champ["logloss"] - results[best]["valid"]["logloss"]
        tied = [
            n
            for n in order
            if results[n]["valid"]["logloss"] <= results[best]["valid"]["logloss"] + tolerance
            and results[n]["role"] != "baseline"
        ]
        notes.append(
            f"{champion} is the champion (log loss {champ['logloss']:.4f}). {best} scored best "
            f"({results[best]['valid']['logloss']:.4f}), only {gap:.4f} lower, which is within the {tolerance:g} "
            f"tie margin; of the tied models ({', '.join(tied)}), {champion} was the quickest to train. "
            f"Use --champion_tolerance 0 to always take the lowest log loss."
        )
    gap = champ["mean_pred"] - champ["base_rate"]
    notes.append(
        f"The champion's average prediction is {champ['mean_pred'] * 100:.2f}% against an actual churn rate of "
        f"{champ['base_rate'] * 100:.2f}% ({'under' if gap < 0 else 'over'}-predicting by "
        f"{abs(gap) * 100:.2f} points). "
        f"Shifting it to the right level would give {champ['rate_matched_logloss']:.4f}; that uses {valid_name}'s own "
        f"rate, so it is a ceiling for the correction, not a score."
    )
    if ENSEMBLE_NAME in results:
        ens = results[ENSEMBLE_NAME]
        members = ens["members"]
        best = min(members, key=lambda m: results[m]["valid"]["logloss"])
        diff = ens["valid"]["logloss"] - results[best]["valid"]["logloss"]
        notes.append(
            f"The ensemble ({', '.join(members)}) is {abs(diff):.4f} {'better' if diff < 0 else 'worse'} "
            f"than its best member, {best}."
        )
    for name in results:
        if name.endswith("_tuned"):
            base = name[: -len("_tuned")]
            if base in results:
                if results[name].get("params") == results[base].get("params"):
                    notes.append(
                        f"Tuning {base} kept its default settings: no trial beat them on the training holdout."
                    )
                else:
                    diff = results[name]["valid"]["logloss"] - results[base]["valid"]["logloss"]
                    notes.append(f"Tuning {base} changed its log loss on {valid_name} by {diff:+.4f}.")
    for name, why in skipped.items():
        notes.append(f"{name} was not run: {why}.")
    for name, why in failed.items():
        notes.append(f"{name} failed and was left out: {why} (full traceback in run.log).")
    return notes


# --------------------------------------------------------------------------- final fit + submissions


def _final_cache(run_dir: str, name: str, seed) -> tuple:
    folder = os.path.join(run_dir, "artifacts", "final")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, f"{name}_seed{seed}")
    return stem + ".json", stem + "_test_pred.npy"


def _final_predict(
    entry: dict, specs: dict, full, test, cols, cfg, seeds, round_multiplier, run_dir, LOG, base_fp: dict
):
    """Refit one model on train + valid for each seed and predict the test month.

    Each seed's test predictions are saved as soon as they exist, so a re-run into the
    same run folder (after a disconnect, say) skips every seed that already finished.
    """
    spec = specs[entry["spec"]]
    best = entry.get("best_iteration")
    rounds = max(1, int(round(best * round_multiplier))) if best else None
    seed_list = list(seeds) if rounds else list(seeds)[:1]  # only boosted models vary with the seed
    log_stage(
        LOG,
        "INFO",
        "FinalFit",
        model=entry["name"],
        rows=len(full),
        rounds=rounds,
        best_iteration=best,
        seeds=",".join(str(s) for s in seed_list),
    )
    preds = []
    for seed in seed_list:
        fp = _jsonable(
            {
                **base_fp,
                "stage": "final",
                "model": entry["name"],
                "params": entry["params"],
                "rounds": rounds,
                "seed": seed,
                "rows_full": len(full),
                "rows_test": len(test),
            }
        )
        meta_path, pred_path = _final_cache(run_dir, entry["name"], seed)
        if os.path.exists(meta_path) and os.path.exists(pred_path):
            with open(meta_path, encoding="utf-8") as f:
                if json.load(f).get("fingerprint") == fp:
                    preds.append(np.load(pred_path))
                    log_stage(LOG, "INFO", "FinalFit", model=entry["name"], seed=seed, resumed=True)
                    continue
        t0 = time.time()
        model = spec.fit_final(full, cols, entry["params"], int(seed), rounds, LOG, cfg)
        path = spec.save(model, os.path.join(run_dir, "artifacts", f"{entry['name']}_final_seed{seed}"))
        pred = np.asarray(spec.predict(model, test, cols), dtype=float)
        np.save(pred_path, pred)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fp, "seconds": round(time.time() - t0, 1)}, f)
        log_stage(LOG, "CHECKPOINT", "FinalFit", model=entry["name"], seed=seed, rounds=rounds, path=path)
        log_stage(LOG, "INFO", "FinalFit", model=entry["name"], seed=seed, seconds=round(time.time() - t0, 1))
        preds.append(pred)
    return np.mean(preds, axis=0)


def choose_champion(results: dict, tolerance: float) -> tuple:
    """Lowest log loss wins, unless a cheaper model is within `tolerance` of it.

    Gaps under about 0.001 are run-to-run noise, so among the models that close to the
    best, the one that was quickest to train is taken: same accuracy, less to refit.
    Returns (champion, best).
    """
    best = min(results, key=lambda n: results[n]["valid"]["logloss"])
    if not tolerance or tolerance <= 0:
        return best, best
    limit = results[best]["valid"]["logloss"] + tolerance
    tied = [n for n in results if results[n]["valid"]["logloss"] <= limit and results[n]["role"] != "baseline"]
    if not tied:
        return best, best
    champion = min(tied, key=lambda n: (results[n].get("train_seconds") or 0.0, results[n]["valid"]["logloss"]))
    return champion, best


def _write_submission(test, pred, path, LOG, kind):
    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: np.clip(pred, 1e-6, 1 - 1e-6)})
    sub.to_csv(path, index=False)
    log_stage(
        LOG,
        "ARTIFACT",
        COMPONENT,
        submission=path,
        kind=kind,
        rows=len(sub),
        mean_pred=round(float(sub[TARGET_COL].mean()), 5),
    )
    return sub


def run_training(
    data_dir: str,
    run_dir: str,
    LOG,
    models=DEFAULT_MODELS,
    frame_format: str = "parquet",
    lgb_params: dict | None = None,
    num_boost_round: int = 3000,
    early_stopping_rounds: int = 100,
    eval_every: int = 50,
    seeds=(42, 43, 44),
    round_multiplier: float = 1.1,
    logistic_max_rows: int = 300_000,
    plots: bool = True,
    train_snapshots=TRAIN_SNAPSHOTS,
    valid_snapshot: str = VALID_SNAPSHOT,
    test_snapshot: str = TEST_SNAPSHOT,
    population: str = DEFAULT_POPULATION,
    holdout_share: float = 0.2,
    catboost_iterations: int = 3000,
    tune_models=(),
    tune_trials: int = 15,
    tune_rows: int = 300_000,
    ensemble: bool = True,
    calibration_target: float | None = None,
    champion_tolerance: float = 0.001,
    ensemble_seeds: int = 1,
) -> dict:
    models = list(models)
    train_snapshots = list(train_snapshots)
    seeds = list(seeds)
    specs = {s.name: s for s in registry(lgb_params)}
    cfg = {
        "eval_every": eval_every,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "logistic_max_rows": logistic_max_rows,
        "catboost_iterations": catboost_iterations,
        "tune_rows": tune_rows,
        "tune_trials": tune_trials,
    }

    with component_stage(
        LOG,
        COMPONENT,
        models=",".join(models),
        train=",".join(train_snapshots),
        valid=valid_snapshot,
        test=test_snapshot,
        population=population,
        tune=",".join(tune_models) or None,
        seeds=",".join(str(s) for s in seeds),
    ):
        _check_snapshots(train_snapshots, valid_snapshot)
        parts = [
            apply_population(load_snapshot(data_dir, n, frame_format, LOG), population, n, LOG) for n in train_snapshots
        ]
        train = pd.concat(parts, ignore_index=True)
        valid = apply_population(
            load_snapshot(data_dir, valid_snapshot, frame_format, LOG), population, valid_snapshot, LOG
        )
        test = load_snapshot(data_dir, test_snapshot, frame_format, LOG)  # scored as given: every listed user
        cols = feature_columns(train)
        if feature_columns(valid) != cols or feature_columns(test) != cols:
            raise ValueError("Feature columns differ between snapshots. Rebuild with --mode features.")
        fit_df, stop_df = split_holdout(train, holdout_share, seeds[0])
        log_stage(
            LOG,
            "INFO",
            COMPONENT,
            features=len(cols),
            train_rows=len(train),
            fit_rows=len(fit_df),
            early_stopping_rows=len(stop_df),
            valid_rows=len(valid),
            train_churn_rate=round(float(train[TARGET_COL].mean()), 5),
            valid_churn_rate=round(float(valid[TARGET_COL].mean()), 5),
        )

        base_fp = {
            "train": train_snapshots,
            "valid": valid_snapshot,
            "population": population,
            "holdout_share": holdout_share,
            "seed": seeds[0],
            "rows": [len(train), len(valid)],
            "features": cols,
            **{k: v for k, v in cfg.items() if k not in ("eval_every", "tune_rows", "tune_trials")},
        }
        results, skipped, failed = _run_marathon(
            specs,
            models,
            fit_df,
            stop_df,
            valid,
            cols,
            cfg,
            seeds[0],
            run_dir,
            LOG,
            valid_snapshot,
            base_fp,
            [m for m in tune_models if m in models],
            ensemble,
        )
        if not results:
            raise RuntimeError("No model finished. See the warnings above and run.log.")

        # Champion = lowest out-of-time log loss (the competition metric), unless a cheaper model ties with it.
        champion, best = choose_champion(results, champion_tolerance)
        for name, res in results.items():
            log_stage(
                LOG,
                "RESULT",
                COMPONENT,
                model=name,
                family=res["family"],
                valid_logloss=round(res["valid"]["logloss"], 6),
                valid_auc=None if res["valid"]["auc"] is None else round(res["valid"]["auc"], 6),
                rate_matched_logloss=round(res["valid"]["rate_matched_logloss"], 6),
                champion=name == champion,
                file_only=True,
            )  # the console shows the table instead

        leaderboard = build_leaderboard(results, champion)
        leaderboard_path = os.path.join(run_dir, "metrics", "leaderboard.csv")
        leaderboard.to_csv(leaderboard_path, index=False)
        log_stage(LOG, "ARTIFACT", COMPONENT, leaderboard=leaderboard_path)
        churn_rate = float(valid[TARGET_COL].mean())
        print_table(
            LOG,
            leaderboard.drop(columns=["champion", "pr auc", "brier", "churn rate"]),
            f"Model ranking by log loss on {valid_snapshot} (churn rate {churn_rate * 100:.2f}%, population "
            f"{population}; lower is better)",
            formats={
                "log loss": num4,
                "rate-matched": num4,
                "skill vs prior": pct,
                "auc": num4,
                "mean pred": pct,
                "top 10% capture": pct,
                "rounds": lambda v: f"{int(v)}",
                "train time": lambda v: f"{v:.0f}s",
            },
        )
        notes = marathon_notes(results, champion, skipped, failed, valid_snapshot, best, champion_tolerance)
        bullet_block(LOG, "What the ranking says", notes)

        champ = results[champion]
        champion_path = os.path.join(run_dir, "artifacts", "champion.json")
        with open(champion_path, "w", encoding="utf-8") as f:
            json.dump(
                _jsonable(
                    {
                        "name": champion,
                        "family": champ["family"],
                        "members": champ.get("members"),
                        "params": champ.get("params"),
                        "best_iteration": champ.get("best_iteration"),
                        "valid": champ["valid"],
                        "features": cols,
                        "categorical": [c for c in CATEGORICAL_FEATURES if c in cols],
                    }
                ),
                f,
                indent=4,
            )
        log_stage(LOG, "ARTIFACT", COMPONENT, champion=champion_path)

        # ---- final fit on train + valid, then the two submissions
        full = pd.concat([train, valid], ignore_index=True)
        if champion == ENSEMBLE_NAME:
            # Averaging different models already smooths out seed noise, so each member gets fewer seeds.
            entries, final_seeds = [results[m] for m in champ["members"]], seeds[: max(1, ensemble_seeds)]
        else:
            entries, final_seeds = [champ], seeds
        with component_stage(LOG, "FinalFit", model=champion, rows=len(full), test_rows=len(test)):
            test_pred = np.mean(
                [
                    _final_predict(
                        e,
                        specs,
                        full,
                        test,
                        cols,
                        {**cfg, "eval_every": 0},
                        final_seeds,
                        round_multiplier,
                        run_dir,
                        LOG,
                        base_fp,
                    )
                    for e in entries
                ],
                axis=0,
            )

        target = churn_rate if calibration_target is None else float(calibration_target)
        if not 0 < target < 1:
            raise ValueError(f"--calibration_target must be between 0 and 1; got {target}.")
        sub_path = os.path.join(run_dir, "artifacts", "submission.csv")
        recal_path = os.path.join(run_dir, "artifacts", "submission_recalibrated.csv")
        sub = _write_submission(test, test_pred, sub_path, LOG, "as trained")
        recal = _write_submission(test, match_rate(test_pred, target), recal_path, LOG, f"recalibrated to {target:.5f}")

        summary = {
            "champion": champion,
            "best_logloss_model": best,
            "champion_tolerance": champion_tolerance,
            "models": {
                name: {
                    "family": res["family"],
                    "role": res["role"],
                    "valid": res["valid"],
                    "best_iteration": res.get("best_iteration"),
                    "train_seconds": res.get("train_seconds"),
                    "params": res.get("params"),
                    "members": res.get("members"),
                }
                for name, res in results.items()
            },
            "skipped": skipped,
            "failed": failed,
            "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
            "train_snapshots": train_snapshots,
            "valid_snapshot": valid_snapshot,
            "test_snapshot": test_snapshot,
            "population": population,
            "tuned": [m for m in results if m.endswith("_tuned")],
            "submission": sub_path,
            "submission_recalibrated": recal_path,
            "calibration_target": target,
            "calibration_target_source": "valid churn rate" if calibration_target is None else "--calibration_target",
            "test_mean_pred": {
                "as_trained": float(sub[TARGET_COL].mean()),
                "recalibrated": float(recal[TARGET_COL].mean()),
            },
        }
        write_summary(_jsonable(summary), run_dir, LOG)
        raw(
            LOG,
            f"\nChampion: {champion} (log loss {champ['valid']['logloss']:.4f} on {valid_snapshot})\n"
            f"Predictions for {test_snapshot} ({len(sub):,} users), two files:\n"
            f"  - submission.csv: as trained, mean prediction {sub[TARGET_COL].mean() * 100:.2f}%\n"
            f"  - submission_recalibrated.csv: shifted to a mean of {target * 100:.2f}%"
            f" ({'the ' + valid_snapshot + ' churn rate' if calibration_target is None else 'your target'}),"
            f" same ranking\n"
            f"Both files use the Kaggle submission format.",
        )

    if plots:
        make_run_figures(run_dir, data_dir, frame_format, LOG)
    return summary


def make_run_figures(run_dir: str, data_dir: str, frame_format: str, LOG) -> None:
    """Interactive results figures in plots/. A missing Plotly or a figure error never fails the run."""
    try:
        from src.figures import build_figures

        build_figures(run_dir, data_dir, LOG, frame_format=frame_format)
    except ImportError:
        log_stage(LOG, "WARN", "Figures", skipped="plotly is not installed (pip install plotly)")
    except Exception as exc:
        log_stage(
            LOG,
            "WARN",
            "Figures",
            skipped="figure build failed, results are unaffected",
            reason=f"{type(exc).__name__}: {exc}",
        )
