"""Raw CSV -> Parquet conversion on a tiny hand-written dataset (needs DuckDB)."""

import os

import pytest
from conftest import quiet_logger

from src.constants import TABLE_COLUMNS, to_days
from src.ingest import (
    LOGS_COVERAGE_FILE,
    RAW_CSVS,
    TABLE_ORDER,
    convert_to_parquet,
    find_csv,
    logs_cover,
    table_sources,
    tables_to_build,
)

LOG = quiet_logger()


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _make_raw(raw_dir):
    """u1 is labelled, u_unlabelled is not. Files sit in nested folders like the Kaggle archives."""
    _write(f"{raw_dir}/train.csv", "msno,is_churn\nu1,1\n")
    _write(f"{raw_dir}/data/churn_comp_refresh/train_v2.csv", "msno,is_churn\nu1,0\nu2,1\n")
    _write(f"{raw_dir}/data/churn_comp_refresh/sample_submission_v2.csv", "msno,is_churn\nu2,0\n")
    _write(
        f"{raw_dir}/members_v3.csv",
        "msno,city,bd,gender,registered_via,registration_init_time\nu1,5,28,male,7,20150101\nu2,1,0,,9,20161231\n",
    )
    tx_header = (
        "msno,payment_method_id,payment_plan_days,plan_list_price,actual_amount_paid,"
        "is_auto_renew,transaction_date,membership_expire_date,is_cancel\n"
    )
    _write(
        f"{raw_dir}/transactions.csv",
        tx_header + "u1,41,30,149,149,1,20170105,20170204,0\nu_unlabelled,41,30,149,149,1,20170105,20170204,0\n",
    )
    _write(
        f"{raw_dir}/data/churn_comp_refresh/transactions_v2.csv",
        tx_header + "u1,41,30,149,149,1,20170105,20170204,0\nu2,38,30,99,99,0,20170310,20170409,0\n",
    )
    log_header = "msno,date,num_25,num_50,num_75,num_985,num_100,num_unq,total_secs\n"
    _write(
        f"{raw_dir}/user_logs.csv",
        log_header
        + "u1,20160930,1,1,1,1,1,5,100.0\n"  # before LOG_START: dropped
        + "u1,20161001,1,1,1,1,1,5,-50.0\n"  # negative seconds: clamped to 0
        + "u1,20170101,1,1,1,1,1,5,9999999999.0\n"  # absurd seconds: clamped to one day
        + "u_unlabelled,20170101,1,1,1,1,1,5,100.0\n",
    )  # no label: dropped
    _write(f"{raw_dir}/data/churn_comp_refresh/user_logs_v2.csv", log_header + "u2,20170315,2,0,0,0,9,10,2000.5\n")


def test_find_csv_searches_nested_folders(tmp_path):
    _make_raw(str(tmp_path))
    assert find_csv(str(tmp_path), "train_v2.csv").endswith(os.path.join("churn_comp_refresh", "train_v2.csv"))
    assert find_csv(str(tmp_path), "nope.csv") is None


def test_tables_to_build_and_sources(tmp_path):
    assert tables_to_build(str(tmp_path)) == TABLE_ORDER  # nothing exists yet
    os.makedirs(tmp_path / "parquet")
    for t in TABLE_ORDER:
        (tmp_path / "parquet" / f"{t}.parquet").write_text("x")
    assert tables_to_build(str(tmp_path)) == []
    assert tables_to_build(str(tmp_path), ["transactions"]) == ["transactions"]
    assert table_sources("transactions") == ["transactions.csv", "transactions_v2.csv"]
    assert table_sources("labels_mar") == ["train_v2.csv"]


def test_logs_cover_without_coverage_file_means_official_only(tmp_path):
    assert logs_cover(str(tmp_path), "labels_feb")
    assert not logs_cover(str(tmp_path), "labels_jan")
    assert logs_cover(str(tmp_path), "labels_jan", engine="sqlite")  # the demo keeps every user


def test_convert_to_parquet(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    raw_dir, data_dir = str(tmp_path / "raw"), str(tmp_path / "data")
    _make_raw(raw_dir)
    raw = {name: find_csv(raw_dir, name) for name in RAW_CSVS}

    manifest = convert_to_parquet(raw, data_dir, LOG, memory_limit="1GB")
    assert set(manifest) == set(TABLE_COLUMNS)

    con = duckdb.connect()

    def read(table):
        return con.execute(f"SELECT * FROM read_parquet('{manifest[table]['path']}') ORDER BY ALL").df()

    for table, columns in TABLE_COLUMNS.items():
        assert list(read(table).columns) == columns, table

    members = read("members").set_index("msno")
    assert members.loc["u1", "registration_init_time"] == to_days("2015-01-01")

    tx = read("transactions")
    assert sorted(tx["msno"]) == ["u1", "u2", "u_unlabelled"]  # duplicate removed, every user kept
    assert tx.set_index("msno").loc["u2", "transaction_date"] == to_days("2017-03-10")
    assert tx.set_index("msno").loc["u2", "membership_expire_date"] == to_days("2017-04-09")

    logs = read("user_logs")
    assert sorted(logs["msno"]) == ["u1", "u1", "u2"]  # listening logs are still trimmed to labelled users
    assert os.path.exists(os.path.join(data_dir, "parquet", LOGS_COVERAGE_FILE))
    assert logs_cover(data_dir, "labels_feb") and not logs_cover(data_dir, "labels_jan")
    assert logs["log_date"].min() == to_days("2016-10-01")
    assert logs["total_secs"].min() == 0 and logs["total_secs"].max() == 86400

    # A second call rebuilds nothing unless asked to.
    again = convert_to_parquet(raw, data_dir, LOG, memory_limit="1GB")
    assert {t: m["rows"] for t, m in again.items()} == {t: m["rows"] for t, m in manifest.items()}
