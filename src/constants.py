"""Snapshot definitions, table schemas and shared constants.

Dates
-----
Every date in the processed tables is an integer: days since 1970-01-01.
That keeps the feature SQL to plain integer arithmetic, which is what makes it
run unchanged on DuckDB (real data) and SQLite (tests / demo).
"""

from dataclasses import dataclass
from datetime import date, timedelta

KAGGLE_COMPETITION = "kkbox-churn-prediction-challenge"
EPOCH = date(1970, 1, 1)


def to_days(value) -> int:
    """'2017-02-01' or date -> days since 1970-01-01."""
    if isinstance(value, str):
        value = date.fromisoformat(value)
    return (value - EPOCH).days


def from_days(days: int) -> date:
    return EPOCH + timedelta(days=int(days))


@dataclass(frozen=True)
class Snapshot:
    """One labelled (or to-be-scored) month.

    cutoff: first day of the month the membership expires in. Features may only
            use rows dated strictly before it.
    source: "official" labels come from a Kaggle file (label_file); "rebuilt"
            labels are made by src/labeller.py (the organisers' script, ported).
    """

    name: str
    cutoff: str
    label_table: str
    labelled: bool
    source: str = "official"
    label_file: str | None = None

    @property
    def cutoff_day(self) -> int:
        return to_days(self.cutoff)

    @property
    def label_month(self) -> str:
        return self.cutoff[:7]


SNAPSHOTS = {
    "jan": Snapshot("jan", "2017-01-01", "labels_jan", True, "rebuilt"),
    "feb": Snapshot("feb", "2017-02-01", "labels_feb", True, "official", "train.csv"),
    "feb_rebuilt": Snapshot("feb_rebuilt", "2017-02-01", "labels_feb_rebuilt", True, "rebuilt"),
    "mar": Snapshot("mar", "2017-03-01", "labels_mar", True, "official", "train_v2.csv"),
    "apr": Snapshot("apr", "2017-04-01", "labels_apr", False, "official", "sample_submission_v2.csv"),
}
OFFICIAL_SNAPSHOTS = [n for n, s in SNAPSHOTS.items() if s.source == "official"]
REBUILT_SNAPSHOTS = [n for n, s in SNAPSHOTS.items() if s.source == "rebuilt"]

# Defaults. Training can stack several months (--train_snapshots jan,feb).
TRAIN_SNAPSHOTS = ("feb",)
VALID_SNAPSHOT = "mar"
TEST_SNAPSHOT = "apr"
FEATURE_SNAPSHOTS = ("jan", "feb", "feb_rebuilt", "mar", "apr")

# Which users of a label file training and validation keep (--population):
#   expiry    last transaction before the cutoff (all history, organisers' ordering) expires
#             inside the label month. Matches the April test users most closely (default).
#   labeller  the organisers' script's own rule: a transaction in the previous month whose last
#             expiry falls in the label month. See src/labeller.py.
#   all       every user in the label file.
POPULATIONS = ("expiry", "labeller", "all")
POPULATION_COLUMNS = {"expiry": "in_population_expiry", "labeller": "in_labeller_population"}
DEFAULT_POPULATION = "expiry"

# user_logs is ~30 GB raw. Only the 90 days before the earliest cutoff (jan) are needed.
LOG_START = "2016-10-01"
LOG_WINDOWS = (7, 30, 90)

# Processed table schemas (after ingest). The synthetic demo emits exactly these.
TABLE_COLUMNS = {
    "members": ["msno", "city", "bd", "gender", "registered_via", "registration_init_time"],
    "transactions": [
        "msno",
        "payment_method_id",
        "payment_plan_days",
        "plan_list_price",
        "actual_amount_paid",
        "is_auto_renew",
        "transaction_date",
        "membership_expire_date",
        "is_cancel",
    ],
    "user_logs": ["msno", "log_date", "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"],
    "labels_feb": ["msno", "is_churn"],
    "labels_mar": ["msno", "is_churn"],
    "labels_apr": ["msno", "is_churn"],
}

ID_COL = "msno"
TARGET_COL = "is_churn"
# Columns in feature tables that are not model features.
META_COLS = ("in_population_expiry", "in_labeller_population")
