# Model card

## Purpose

Estimate the probability that a KKBox subscriber whose membership expires in a given month will not
renew within 30 days. Built as a portfolio project on public competition data. It is not deployed
and has not been validated for decisions about real customers.

## Models

Every model is trained and scored the same way, then ranked by March log loss (the "marathon").

| Model | Family | Why it is here |
|---|---|---|
| `prior` | baseline | Predicts the training churn rate for everyone. The floor any model must beat. |
| `logistic` | linear | Median imputation with missing flags, standardisation, logistic regression. Shows how much of the signal is simple and additive. |
| `lightgbm` | lightgbm | Gradient-boosted trees, leaf-wise growth, native categorical splits. The strongest model on this data so far. |
| `xgboost` | xgboost | Gradient-boosted trees, depth-wise growth. A second boosting implementation that makes different mistakes. |
| `catboost` | catboost | Gradient-boosted trees with ordered target statistics for the categorical columns. |
| `ensemble` | ensemble | Plain average of the best model from each of the top three candidate families. |

A model whose library is not installed is skipped with a warning; a model that fails is left out and
the rest carry on.

**Choosing the champion.** The lowest March log loss wins, with one exception: models within 0.001 of
the best (`--champion_tolerance`) count as tied, because gaps that small are run-to-run noise, and the
quickest of the tied models to train is chosen. It is as accurate and much cheaper to refit. Set the
tolerance to 0 to always take the lowest log loss.

With `--tune`, LightGBM, XGBoost and CatBoost (by default) also run as `<model>_tuned`: Optuna searches
their settings on a 300,000-row sample of the training months and judges each trial on the training
holdout, never on March. The defaults are always the first trial. Tuning is off by default: it costs
time, and a better score on the February holdout does not guarantee a better score on a later month.

## Training and validation

- **Train:** February 2017 official labels by default (`--train_snapshots`). **Validate:** March 2017.
  **Score:** April 2017.
- Validation is out of time: models are always tested on a later month than they were trained on,
  which mirrors how they would be used. The validation month can never be a training month.
- Training and validation keep only users whose last expiry falls in the label month
  (`--population expiry`), which matches the April test users. See the data card.
- Boosted models early-stop on a random 20% of the *training* users (`--holdout_share`). March is
  used only for scoring, so its score is not flattered by early stopping on it.
- The champion is refit on the training months plus March. Boosted models use
  `best_iteration × 1.1` rounds and are averaged over three seeds. If the ensemble is the champion,
  each member is refit with one seed (`--ensemble_seeds`), since averaging three different models
  already smooths out seed noise, and their predictions are averaged.
- Re-running `--mode train` into the same run folder reuses everything that already finished with the
  same data and settings: validation fits (`artifacts/marathon/`) and each final-fit seed's test
  predictions (`artifacts/final/`). An interrupted run loses at most the piece in progress.

Default LightGBM settings: learning rate 0.05, 63 leaves, at least 100 rows per leaf, 80% feature
and row sampling, L2 of 1.0. XGBoost: learning rate 0.05, depth 8, `hist` trees, 80% sampling.
CatBoost: learning rate 0.1, depth 6, L2 of 3. `city`, `gender_code`, `registered_via` and
`last_payment_method_id` are treated as categorical by LightGBM and CatBoost.

## Two submissions

Churn moves from month to month (5.5% in February, 7.3% in March, unknown for April), and log loss
punishes a wrong overall level even when the ranking is right. Each run therefore writes:

- `submission.csv`: the champion's predictions as trained.
- `submission_recalibrated.csv`: the same predictions shifted on the logit scale so that their mean
  equals March's churn rate (`--calibration_target` to choose another). The order of users, and so
  AUC, is unchanged.

The leaderboard's `rate-matched` column shows what the same shift would have done on March using
March's own rate. That is a ceiling for the correction, not a fair score, so only the Kaggle scores
of the two files can say which level is right for April.

## Features

53 features from three sources, all computed from rows dated before the snapshot cutoff:

- **Transactions:** counts, cancellations (including the last 30 days and days since the last
  cancellation), auto-renew share, amounts paid, discounts, and the details of the most recent
  transaction, including days until the membership expires. "Most recent" uses the organisers'
  ordering, so a same-day cancellation is not hidden behind a renewal.
- **Calendar:** where the last-known expiry falls in its month (day, days in the month, days to the
  month end, a month-end flag). Month-end expiries rarely churn, while the few expiring just before
  the end of a 31-day month churn heavily; counting days from the 1st cannot tell these apart
  across months of different lengths.
- **Listening logs:** active days, seconds, unique songs and completed plays over the last 7, 30
  and 90 days, plus completion rate and short-versus-long trend ratios.
- **Members:** city, age, gender, registration channel and tenure.

## Metrics

Log loss is the competition metric and the headline number. AUC and PR-AUC describe ranking,
the Brier score and calibration table describe how trustworthy the probabilities are, and the lift
table shows how much churn the highest-risk deciles capture.

## Results

Trained on February 2017 official labels, validated on March 2017 (expiry population, 7.26% churn).

| Model | Valid log loss | Rate-matched (ceiling) | AUC | PR-AUC | Brier | Mean prediction | Top 10% capture | Kaggle private LB |
|---|---|---|---|---|---|---|---|---|
| ensemble | **0.1714** | 0.1673 | **0.8838** | 0.541 | 0.0466 | 5.45% | 63.9% | |
| xgboost | 0.1717 | 0.1677 | 0.8829 | 0.539 | 0.0467 | 5.44% | 63.9% | |
| lightgbm | 0.1718 | 0.1676 | 0.8830 | 0.539 | 0.0467 | 5.43% | 63.9% | |
| catboost | 0.1722 | 0.1684 | 0.8821 | 0.537 | 0.0468 | 5.49% | 63.6% | |
| logistic | 0.1814 | 0.1781 | 0.8379 | 0.509 | 0.0479 | 5.55% | 63.3% | |
| prior | 0.2630 | 0.2604 | 0.5000 | 0.073 | 0.0676 | 5.54% | 9.8% | |

What this shows:

- **The model is not the bottleneck.** The boosted models and their ensemble are within 0.0008 of
  each other. The ensemble's 0.0004 gain over XGBoost is noise, so XGBoost is the champion under the
  tie rule.
- **The level of the predictions is.** All models predict about 5.4% churn against an actual 7.26%.
  Matching March's rate would gain about 0.004 (the rate-matched column), roughly ten times any
  difference between models.
- **The trees find real interactions.** Logistic regression is 0.010 worse and ranks less well
  (AUC 0.838 against 0.883).

On a 2-core cloud CPU (Google Colab's free tier), validation fits took about 1 minute (XGBoost), 2.5–4.5 minutes (LightGBM) and
12–24 minutes (CatBoost).

Which labels to train on was tested first, with the same features and settings:

| Training labels | LightGBM log loss | AUC | Mean prediction |
|---|---|---|---|
| February, official | **0.1718** | **0.883** | 5.43% |
| February, rebuilt with the organisers' script | 0.2138 | 0.853 | 3.56% |
| January + February, rebuilt | 0.2123 | 0.857 | 3.34% |

The rebuilt labels mark fewer users as churners than the official March labels do, so models trained
on them predict too little churn. Official February labels are the default.

## Explainability

`metrics/lightgbm_shap_importance.csv` ranks features by mean absolute SHAP value, computed with
LightGBM's built-in TreeSHAP on a sample of the validation month. The `feature_importance` figure
(`docs/figures/`, or interactive in the run's `plots/` folder) shows the top 15, coloured by where
each feature comes from: payment history, listening logs, the expiry date or member details.

## Limitations

- One validation month. Seasonal effects cannot be measured.
- The base churn rate moves between months, so probabilities can be systematically high or low on
  a new month even when the ranking holds. The two submissions test this; neither fixes it.
- Logistic regression trained on the rebuilt labels collapses (log loss above 0.6) while LightGBM on
  the same data does not. This has not been explained; rebuilt labels are not used for training.
- Age and gender are sparse and noisy in the source data.
- Fairness across demographic groups has not been assessed.
