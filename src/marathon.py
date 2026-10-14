"""Model marathon: every candidate model trained and scored the same way.

How the models are trained and scored
-------------------------------------
1. One registry. Each model is a ModelSpec: name, family, role (baseline or
   candidate), the library it needs, default settings and a search space.
   A model whose library is not installed is skipped with a warning, and a
   model that fails is skipped without stopping the run.
2. One scoring method. Every model fits on the training months, early-stops (if
   it can) on a holdout of the training users, and is scored on the validation
   month, which it never sees while training.
3. Baselines. The prior (training churn rate for everyone) is scored alongside,
   so each score has something to be compared with.
4. Optional tuning (--tune). Hyperparameters are searched on a sample of the
   training months and judged on the training holdout, never on the validation
   month. The defaults are always one of the trials, so a tuned model cannot be
   chosen that looked worse than its defaults. Tuned models run as separate
   "<model>_tuned" entries.
5. Ensemble. A plain average of the best model from each of the top three
   candidate families, so its members are different kinds of model.
6. Resume. Each finished model's scores and validation predictions are saved in
   the run folder; re-running into the same --run_dir reuses them.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.constants import TARGET_COL
from src.features import CATEGORICAL_FEATURES
from src.logger import log_stage

ENSEMBLE_NAME = "ensemble"

DEFAULT_LGB_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],  # first metric drives early stopping
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
}


# --------------------------------------------------------------------------- helpers


def _eval_line(LOG, component: str, iteration: int, values: dict) -> None:
    log_stage(LOG, "EVAL", component, iter=iteration, **{k: round(float(v), 6) for k, v in values.items()})


def _categorical(cols: list) -> list:
    return [c for c in CATEGORICAL_FEATURES if c in cols]


def _sample(df: pd.DataFrame, rows: int | None, seed: int) -> pd.DataFrame:
    if rows and len(df) > rows:
        return df.sample(rows, random_state=seed).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- model families


@dataclass
class ModelSpec:
    name: str
    family: str
    role: str = "candidate"  # or "baseline"
    library: str | None = None  # import name; None = always available
    defaults: dict[str, Any] = field(default_factory=dict)
    space: dict[str, tuple] | None = None  # name -> ("int"|"float", low, high, log)
    display: str | None = None  # name shown in logs, e.g. "LightGBM"

    @property
    def label(self) -> str:
        return self.display or self.name

    def available(self) -> bool:
        if self.library is None:
            return True
        try:
            __import__(self.library)
            return True
        except ImportError:
            return False

    # Each family implements these three.
    def fit(self, fit_df, stop_df, cols, params, seed, LOG, cfg) -> dict:  # -> {"model", "best_iteration"}
        raise NotImplementedError

    def predict(self, model, df, cols) -> np.ndarray:
        raise NotImplementedError

    def fit_final(self, full_df, cols, params, seed, rounds, LOG, cfg):
        raise NotImplementedError

    def save(self, model, stem: str) -> str:
        """Write a fitted model next to `stem` (no extension); returns the path."""
        import joblib

        path = stem + ".joblib"
        joblib.dump(model, path)
        return path


class PriorSpec(ModelSpec):
    def fit(self, fit_df, stop_df, cols, params, seed, LOG, cfg):
        rate = float(pd.concat([fit_df[TARGET_COL], stop_df[TARGET_COL]]).mean())
        return {"model": rate, "best_iteration": None}

    def predict(self, model, df, cols):
        return np.full(len(df), float(model))

    def fit_final(self, full_df, cols, params, seed, rounds, LOG, cfg):
        return float(full_df[TARGET_COL].mean())

    def save(self, model, stem):
        path = stem + ".json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"churn_rate": float(model)}, f)
        return path


class LogisticSpec(ModelSpec):
    def _pipe(self, params):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=params.get("C", 1.0), max_iter=1000),
        )

    def fit(self, fit_df, stop_df, cols, params, seed, LOG, cfg):
        train = _sample(pd.concat([fit_df, stop_df], ignore_index=True), cfg["logistic_max_rows"], seed)
        pipe = self._pipe(params).fit(train[cols], train[TARGET_COL])
        return {"model": pipe, "best_iteration": None}

    def predict(self, model, df, cols):
        return model.predict_proba(df[cols])[:, 1]

    def fit_final(self, full_df, cols, params, seed, rounds, LOG, cfg):
        train = _sample(full_df, cfg["logistic_max_rows"], seed)
        return self._pipe(params).fit(train[cols], train[TARGET_COL])


class LightGBMSpec(ModelSpec):
    def _dataset(self, lgb, df, cols):
        return lgb.Dataset(df[cols], label=df[TARGET_COL], categorical_feature=_categorical(cols), free_raw_data=False)

    def fit(self, fit_df, stop_df, cols, params, seed, LOG, cfg):
        import lightgbm as lgb

        component, period = cfg.get("component", self.label), cfg["eval_every"]

        def logger(env):
            it = env.iteration + 1
            if period > 0 and it % period == 0:
                _eval_line(LOG, component, it, {f"{r[0]}_{r[1]}": r[2] for r in env.evaluation_result_list})

        logger.order = 10
        p = {**params, "seed": seed}
        dfit, dstop = self._dataset(lgb, fit_df, cols), self._dataset(lgb, stop_df, cols)
        booster = lgb.train(
            p,
            dfit,
            num_boost_round=cfg["num_boost_round"],
            valid_sets=[dfit, dstop],
            valid_names=["train", "holdout"],
            callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], first_metric_only=True, verbose=False), logger],
        )
        best = int(booster.best_iteration or cfg["num_boost_round"])
        return {"model": booster, "best_iteration": best}

    def predict(self, model, df, cols):
        return model.predict(df[cols], num_iteration=getattr(model, "best_iteration", None) or None)

    def fit_final(self, full_df, cols, params, seed, rounds, LOG, cfg):
        import lightgbm as lgb

        booster = lgb.train({**params, "seed": seed}, self._dataset(lgb, full_df, cols), num_boost_round=rounds)
        booster.best_iteration = 0  # use every tree
        return booster

    def save(self, model, stem):
        path = stem + ".txt"
        model.save_model(path)
        return path


class XGBoostSpec(ModelSpec):
    def fit(self, fit_df, stop_df, cols, params, seed, LOG, cfg):
        import xgboost as xgb

        component, period = cfg.get("component", self.label), cfg["eval_every"]

        class Logger(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                it = epoch + 1
                if period > 0 and it % period == 0:
                    _eval_line(
                        LOG, component, it, {f"{d}_{m}": v[-1] for d, ms in evals_log.items() for m, v in ms.items()}
                    )
                return False

        dfit = xgb.DMatrix(fit_df[cols], label=fit_df[TARGET_COL])
        dstop = xgb.DMatrix(stop_df[cols], label=stop_df[TARGET_COL])
        booster = xgb.train(
            {**params, "seed": seed},
            dfit,
            num_boost_round=cfg["num_boost_round"],
            evals=[(dfit, "train"), (dstop, "holdout")],
            early_stopping_rounds=cfg["early_stopping_rounds"],
            verbose_eval=False,
            callbacks=[Logger()],
        )
        return {"model": booster, "best_iteration": int(booster.best_iteration) + 1}

    def predict(self, model, df, cols):
        import xgboost as xgb

        best = model.attr("best_iteration")  # set only by early stopping; the final refit uses every tree
        rng = (0, int(best) + 1) if best is not None else (0, 0)
        return model.predict(xgb.DMatrix(df[cols]), iteration_range=rng)

    def fit_final(self, full_df, cols, params, seed, rounds, LOG, cfg):
        import xgboost as xgb

        return xgb.train(
            {**params, "seed": seed}, xgb.DMatrix(full_df[cols], label=full_df[TARGET_COL]), num_boost_round=rounds
        )

    def save(self, model, stem):
        path = stem + ".json"
        model.save_model(path)
        return path


class CatBoostSpec(ModelSpec):
    @staticmethod
    def _frame(df, cols):
        X = df[cols].copy()
        for c in _categorical(cols):
            X[c] = X[c].fillna(-1).astype("int64")  # CatBoost needs whole-number or text categories
        return X

    def _model(self, params, seed, iterations, early_stopping):
        from catboost import CatBoostClassifier

        kwargs = dict(
            iterations=iterations,
            random_seed=seed,
            loss_function="Logloss",
            eval_metric="Logloss",
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            **params,
        )
        if early_stopping:
            kwargs.update(od_type="Iter", od_wait=early_stopping, use_best_model=True)
        return CatBoostClassifier(**kwargs)

    def fit(self, fit_df, stop_df, cols, params, seed, LOG, cfg):
        cats = _categorical(cols)
        model = self._model(params, seed, cfg["catboost_iterations"], cfg["early_stopping_rounds"])
        model.fit(
            self._frame(fit_df, cols),
            fit_df[TARGET_COL],
            cat_features=cats,
            eval_set=(self._frame(stop_df, cols), stop_df[TARGET_COL]),
        )
        history = model.get_evals_result()
        period = cfg["eval_every"]
        if period > 0:
            curves = {
                f"{'train' if k == 'learn' else 'holdout'}_logloss": v.get("Logloss", []) for k, v in history.items()
            }
            n = max((len(v) for v in curves.values()), default=0)
            for it in range(period, n + 1, period):
                _eval_line(
                    LOG, cfg.get("component", self.label), it, {k: v[it - 1] for k, v in curves.items() if len(v) >= it}
                )
        return {"model": model, "best_iteration": int(model.get_best_iteration() or 0) + 1}

    def predict(self, model, df, cols):
        return model.predict_proba(self._frame(df, cols))[:, 1]

    def fit_final(self, full_df, cols, params, seed, rounds, LOG, cfg):
        model = self._model(params, seed, rounds, None)
        model.fit(self._frame(full_df, cols), full_df[TARGET_COL], cat_features=_categorical(cols))
        return model

    def save(self, model, stem):
        path = stem + ".cbm"
        model.save_model(path)
        return path


def registry(lgb_overrides: dict | None = None) -> list[ModelSpec]:
    """Baselines plus every model family; availability is checked when the marathon runs."""
    return [
        PriorSpec("prior", "baseline", role="baseline", display="Prior"),
        LogisticSpec(
            "logistic", "linear", defaults={"C": 1.0}, space={"C": ("float", 1e-3, 10.0, True)}, display="Logistic"
        ),
        LightGBMSpec(
            "lightgbm",
            "lightgbm",
            library="lightgbm",
            display="LightGBM",
            defaults={**DEFAULT_LGB_PARAMS, **(lgb_overrides or {})},
            space={
                "num_leaves": ("int", 15, 255, True),
                "min_data_in_leaf": ("int", 20, 1000, True),
                "feature_fraction": ("float", 0.5, 1.0, False),
                "lambda_l2": ("float", 1e-3, 10.0, True),
                "learning_rate": ("float", 0.02, 0.1, True),
            },
        ),
        XGBoostSpec(
            "xgboost",
            "xgboost",
            library="xgboost",
            display="XGBoost",
            defaults={
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "eta": 0.05,
                "max_depth": 8,
                "min_child_weight": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "lambda": 1.0,
            },
            space={
                "max_depth": ("int", 4, 10, False),
                "min_child_weight": ("float", 1.0, 100.0, True),
                "subsample": ("float", 0.6, 1.0, False),
                "colsample_bytree": ("float", 0.5, 1.0, False),
                "eta": ("float", 0.02, 0.1, True),
                "lambda": ("float", 1e-3, 10.0, True),
            },
        ),
        CatBoostSpec(
            "catboost",
            "catboost",
            library="catboost",
            display="CatBoost",
            defaults={"learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 3.0},
            space={
                "depth": ("int", 4, 10, False),
                "l2_leaf_reg": ("float", 1.0, 10.0, True),
                "learning_rate": ("float", 0.03, 0.15, True),
            },
        ),
    ]


MODEL_NAMES = ("prior", "logistic", "lightgbm", "xgboost", "catboost")


# --------------------------------------------------------------------------- tuning


def tune(spec: ModelSpec, fit_df, stop_df, cols, cfg, seed, LOG) -> dict:
    """Search the spec's space on a sample of the training months, judged on the training holdout."""
    from sklearn.metrics import log_loss

    sample = _sample(fit_df, cfg["tune_rows"], seed)
    trial_cfg = {**cfg, "eval_every": 0}

    def score(params):
        fitted = spec.fit(sample, stop_df, cols, params, seed, LOG, trial_cfg)
        p = np.clip(spec.predict(fitted["model"], stop_df, cols), 1e-15, 1 - 1e-15)
        return float(log_loss(stop_df[TARGET_COL], p, labels=[0, 1]))

    trials = []

    def run(params):
        loss = score(params)
        trials.append(loss)
        log_stage(
            LOG,
            "INFO",
            f"{spec.label}_tuned",
            trial=len(trials),
            holdout_logloss=round(loss, 6),
            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in params.items() if k in spec.space},
        )
        return loss

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.enqueue_trial({k: spec.defaults[k] for k in spec.space if k in spec.defaults})

        def objective(trial):
            params = dict(spec.defaults)
            for name, (kind, low, high, log) in spec.space.items():
                params[name] = (
                    trial.suggest_int(name, low, high, log=log)
                    if kind == "int"
                    else trial.suggest_float(name, low, high, log=log)
                )
            return run(params)

        study.optimize(objective, n_trials=cfg["tune_trials"])
        best = {**spec.defaults, **study.best_params}
        method = "optuna"
    except ImportError:
        rng = np.random.default_rng(seed)
        candidates = [dict(spec.defaults)]
        for _ in range(max(0, cfg["tune_trials"] - 1)):
            params = dict(spec.defaults)
            for name, (kind, low, high, log) in spec.space.items():
                value = float(np.exp(rng.uniform(np.log(low), np.log(high)))) if log else float(rng.uniform(low, high))
                params[name] = int(round(value)) if kind == "int" else value
            candidates.append(params)
        losses = [run(p) for p in candidates]
        best = candidates[int(np.argmin(losses))]
        method = "random search"
    log_stage(
        LOG,
        "RESULT",
        f"{spec.label}_tuned",
        method=method,
        trials=len(trials),
        best_holdout_logloss=round(min(trials), 6),
        default_holdout_logloss=round(trials[0], 6),
    )
    return best


# --------------------------------------------------------------------------- per-model cache (resume)


def _cache_paths(run_dir: str, name: str):
    folder = os.path.join(run_dir, "artifacts", "marathon")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{name}.json"), os.path.join(folder, f"{name}_valid_pred.npy")


def load_cached(run_dir: str, name: str, fingerprint: dict) -> dict | None:
    meta_path, pred_path = _cache_paths(run_dir, name)
    if not (os.path.exists(meta_path) and os.path.exists(pred_path)):
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("fingerprint") != fingerprint:
        return None
    meta["valid_pred"] = np.load(pred_path)
    return meta


def save_cached(run_dir: str, name: str, result: dict, fingerprint: dict) -> None:
    meta_path, pred_path = _cache_paths(run_dir, name)
    np.save(pred_path, result["valid_pred"])
    meta = {k: v for k, v in result.items() if k not in ("valid_pred", "model")}
    meta["fingerprint"] = fingerprint
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


# --------------------------------------------------------------------------- ensemble


def ensemble_members(results: dict, max_families: int = 3) -> list:
    """Best model from each of the top candidate families (by validation log loss)."""
    best_per_family: dict[str, str] = {}
    for name in sorted(results, key=lambda n: results[n]["valid"]["logloss"]):
        res = results[name]
        if res["role"] != "candidate":
            continue
        best_per_family.setdefault(res["family"], name)
    return list(best_per_family.values())[:max_families]


# --------------------------------------------------------------------------- one model


def run_one(spec: ModelSpec, name: str, params: dict, fit_df, stop_df, valid, cols, cfg, seed, LOG) -> dict:
    t0 = time.time()
    fitted = spec.fit(fit_df, stop_df, cols, params, seed, LOG, cfg)
    pred = spec.predict(fitted["model"], valid, cols)
    return {
        "name": name,
        "family": spec.family,
        "role": spec.role,
        "spec": spec.name,
        "params": params,
        "best_iteration": fitted["best_iteration"],
        "model": fitted["model"],
        "valid_pred": np.asarray(pred, dtype=float),
        "train_seconds": round(time.time() - t0, 1),
    }
