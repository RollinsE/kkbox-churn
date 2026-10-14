# Data card

## Source

Kaggle, *WSDM – KKBox's Churn Prediction Challenge* (`kkbox-churn-prediction-challenge`).
KKBox is a music streaming subscription service. The data is released under the competition rules:
it must be downloaded with your own Kaggle account and **must not be committed or redistributed**.
`.gitignore` excludes every data and model file.

## Raw files used

| File | Content | Used for |
|---|---|---|
| `train.csv` | `msno`, `is_churn` for memberships expiring in February 2017 | `feb` snapshot (training) |
| `train_v2.csv` | Same for March 2017 | `mar` snapshot (validation) |
| `sample_submission_v2.csv` | Users to score for April 2017 | `apr` snapshot (test) |
| `transactions.csv`, `transactions_v2.csv` | Payments, plans, auto-renew and cancellations | Transaction features |
| `user_logs.csv`, `user_logs_v2.csv` | Daily listening counts and seconds | Listening features |
| `members_v3.csv` | City, age, gender, registration channel and date | Member features |
| `WSDMChurnLabeller.scala` | The organisers' labelling script (Spark) | Ported to SQL in `src/labeller.py` |

Churn means the user did not renew within 30 days of their membership expiring.

## How labels are made (`src/labeller.py`)

`src/labeller.py` is an exact SQL port of `WSDMChurnLabeller.scala`. `tests/test_labeller.py`
checks it against a line-by-line Python transliteration of the Scala. For label month M:

1. **Who is labelled:** users with at least one transaction in the month before M whose *last*
   such transaction has an expiry date inside M. "Last" uses the script's ordering: by date; on the
   same day, by the text `price || plan days || payment method`, descending; for the same plan,
   renewals before cancellations. So a same-day cancellation counts as the last transaction.
2. **Churn:** walk through the later transactions in the same order. Each cancellation can pull the
   expiry date earlier. The first non-cancel transaction is the renewal, and a gap of under 30 days
   from the (possibly earlier) expiry is a renewal. Anything else is churn: a late renewal, only
   cancellations, or no activity at all.

Rebuilt months (`--mode labels`):

| Snapshot | Label month | Notes |
|---|---|---|
| `jan` | January 2017 | No official labels exist; an extra training month. |
| `feb_rebuilt` | February 2017 | To compare with, or replace, the official `train.csv`. |

March cannot be rebuilt, because the data has no April transactions to decide renewals.

## Processing (`src/ingest.py`)

- Dates are converted from `YYYYMMDD` integers to integer days since 1970-01-01. Unparseable dates
  become missing.
- The two transaction files are unioned and exact duplicate rows removed. Every user is kept, because
  the label rebuild needs users who appear in no label file (for example, January churners).
- `user_logs` keeps only users in the label tables that exist when it is built (official files, plus
  any rebuilt ones). `parquet/user_logs_coverage.json` records which tables, and features refuse to
  build a rebuilt month whose users have no listening history: churners would otherwise stand out by
  having no logs at all.
- `user_logs` keeps only rows from 2016-10-01 onwards, which covers the 90 days before the earliest
  cutoff. `date` is renamed `log_date`.
- `total_secs` is clamped to between 0 and 86,400. The raw file contains negative and impossibly
  large values.

Member fields are cleaned in the feature SQL: ages outside 10 to 80 become missing, and gender is
coded 1 (male), 2 (female) or 0 (not given).

## Snapshots and cutoffs

| Snapshot | Cutoff | Labels | Features may use rows dated |
|---|---|---|---|
| `jan` | 2017-01-01 | rebuilt | before 2017-01-01 |
| `feb` | 2017-02-01 | official (`train.csv`) | before 2017-02-01 |
| `feb_rebuilt` | 2017-02-01 | rebuilt | before 2017-02-01 |
| `mar` | 2017-03-01 | official (`train_v2.csv`) | before 2017-03-01 |
| `apr` | 2017-04-01 | none (test) | before 2017-04-01 |

**Population.** Roughly 11 to 14% of the users in `train.csv` and `train_v2.csv` have a last
expiry outside the label month, and they churn far more (about 18% in February, 32% in March).
The April test file has almost none of them. Each feature table carries two flags, and training
and validation keep only flagged users (`--population`):

| `--population` | Keeps users whose... | Feb | Mar | Apr |
|---|---|---|---|---|
| `expiry` (default) | last transaction before the cutoff (all history, organisers' ordering) expires in the month | ~89% | ~89% | ~99.7% |
| `labeller` | script rule: last transaction in the previous month expires in the month | 86.2% | 85.7% | 97.0% |
| `all` | anything | 100% | 100% | 100% |

`expiry` matches the April users most closely, so it is the default. Exact shares are printed by
`--mode diagnose` (expiry check).

**Were the official labels made too early?** `--mode diagnose` (truncation check) rebuilds
February with renewals cut at 28 February, the end of the original data release, and compares
churn by expiry day across label sets. Labels made before renewals were recorded show churn
climbing for memberships that expire late in the month.

`tests/test_features.py` adds rows dated on and after the cutoff and checks that no feature changes.

## Known limitations

- Only two labelled months exist, so validation rests on a single out-of-time month.
- The churn rate differs between the labelled months, and April's is unknown. Predicted
  probabilities may need recalibrating for a month with a different base rate.
- Some labelled users have no member record. A `has_member_record` flag is included.
- Listening logs start in October 2016, so January 2017 is the earliest month with a full 90-day
  history. Earlier months would need older logs.

## Synthetic demo data

`--demo` generates fake tables with the same schema (`src/demo.py`). It exists to exercise the code.
Scores on it say nothing about performance on KKBox data.
