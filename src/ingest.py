"""Kaggle -> 7z -> CSV -> Parquet, using DuckDB so nothing has to fit in RAM.

What happens to the raw files
-----------------------------
* YYYYMMDD integer dates become integer days since 1970-01-01.
* transactions.csv + transactions_v2.csv are unioned and de-duplicated. All
  users are kept (~23M rows): the label rebuild (src/labeller.py) needs users who
  never appear in a label file, e.g. January churners.
* user_logs is cut down to rows on/after LOG_START for users that appear in a
  label file. That takes ~30 GB of CSV down to a few GB of Parquet.
* user_logs.date is renamed log_date and total_secs is clamped to one day
  (the raw file contains negative and astronomically large values).
"""

import glob
import json
import os
import shutil
import subprocess

from src.constants import KAGGLE_COMPETITION, LOG_START, SNAPSHOTS, TABLE_COLUMNS
from src.db import SQLITE_FILENAME, connect, default_temp_dir, load_frame, parquet_path, scalar
from src.logger import component_stage, log_stage

COMPONENT = "Ingest"

RAW_CSVS = [
    "train.csv",
    "train_v2.csv",
    "sample_submission_v2.csv",
    "members_v3.csv",
    "transactions.csv",
    "transactions_v2.csv",
    "user_logs.csv",
    "user_logs_v2.csv",
]


def _days(col: str) -> str:
    """DuckDB expression: YYYYMMDD integer column -> days since 1970-01-01 (NULL if unparseable)."""
    return (
        "CAST(datediff('day', DATE '1970-01-01', "
        f"CAST(try_strptime(CAST({col} AS VARCHAR), '%Y%m%d') AS DATE)) AS INTEGER)"
    )


def find_csv(raw_dir: str, name: str) -> str | None:
    """The Kaggle archives extract into nested folders, so search recursively."""
    hits = sorted(glob.glob(os.path.join(raw_dir, "**", name), recursive=True))
    return hits[0] if hits else None


def download_from_kaggle(raw_dir: str, name: str, LOG) -> str:
    """Download <name>.7z with the Kaggle CLI (needs KAGGLE_USERNAME / KAGGLE_KEY and accepted rules)."""
    archive = os.path.join(raw_dir, f"{name}.7z")
    if os.path.exists(archive):
        log_stage(LOG, "INFO", COMPONENT, action="download_skip", reason="archive_exists", archive=archive)
        return archive

    cmd = ["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-f", f"{name}.7z", "-p", raw_dir]
    log_stage(LOG, "INFO", COMPONENT, action="download", file=f"{name}.7z")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(archive):
        raise RuntimeError(
            f"Kaggle download failed for {name}.7z: {proc.stderr.strip() or proc.stdout.strip()} "
            "(401: the token is wrong, use KAGGLE_API_TOKEN for new-style tokens; "
            "403: accept the competition rules on Kaggle)."
        )
    log_stage(LOG, "ARTIFACT", COMPONENT, archive=archive, size_mb=round(os.path.getsize(archive) / 1e6, 1))
    return archive


def extract_7z(archive: str, out_dir: str, LOG) -> None:
    """Extract with the 7z binary when it is installed; fall back to py7zr."""
    binary = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
    log_stage(LOG, "INFO", COMPONENT, action="extract", archive=archive, tool=binary or "py7zr")
    if binary:
        proc = subprocess.run([binary, "x", "-y", f"-o{out_dir}", archive], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"7z extraction failed for {archive}: {proc.stderr.strip()[-500:]}")
        return

    import py7zr

    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=out_dir)


TABLE_ORDER = ["labels_feb", "labels_mar", "labels_apr", "members", "transactions", "user_logs"]


def table_sources(table: str) -> list:
    """Raw CSVs each processed table is built from."""
    if table.startswith("labels_"):
        return [next(s for s in SNAPSHOTS.values() if s.label_table == table).label_file]
    return {
        "members": ["members_v3.csv"],
        "transactions": ["transactions.csv", "transactions_v2.csv"],
        "user_logs": ["user_logs.csv", "user_logs_v2.csv"],
    }[table]


def acquire_raw(raw_dir: str, LOG, download: bool = True, keep_archives: bool = False, names=None) -> dict:
    """Make sure the raw CSVs in `names` (default: all) are under raw_dir. Returns {name: path}."""
    os.makedirs(raw_dir, exist_ok=True)
    paths = {}
    for name in names or RAW_CSVS:
        path = find_csv(raw_dir, name)
        if path is None:
            if not download:
                raise FileNotFoundError(f"{name} not found under {raw_dir} and --skip_download is set.")
            archive = download_from_kaggle(raw_dir, name, LOG)
            extract_7z(archive, raw_dir, LOG)
            if not keep_archives:
                os.remove(archive)
            path = find_csv(raw_dir, name)
            if path is None:
                raise FileNotFoundError(f"{name} not found under {raw_dir} after extraction.")
        paths[name] = path
        log_stage(LOG, "INFO", COMPONENT, raw_file=name, path=path, size_mb=round(os.path.getsize(path) / 1e6, 1))
    return paths


def _copy(con, select_sql: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    con.execute(f"COPY ({select_sql}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    os.replace(tmp, out_path)  # never leave a half-written Parquet behind


def _table_sql(table: str, raw: dict) -> str:
    """SELECT statement that reads the raw CSV(s) for one processed table."""
    if table.startswith("labels_"):
        snap = next(s for s in SNAPSHOTS.values() if s.label_table == table)
        return (
            "SELECT msno, CAST(is_churn AS TINYINT) AS is_churn "
            f"FROM read_csv('{raw[snap.label_file]}', header=true, "
            "columns={'msno': 'VARCHAR', 'is_churn': 'INTEGER'})"
        )

    if table == "members":
        return (
            "SELECT msno, city, bd, gender, registered_via, "
            f"{_days('registration_init_time')} AS registration_init_time "
            f"FROM read_csv('{raw['members_v3.csv']}', header=true, columns={{"
            "'msno': 'VARCHAR', 'city': 'INTEGER', 'bd': 'INTEGER', 'gender': 'VARCHAR', "
            "'registered_via': 'INTEGER', 'registration_init_time': 'INTEGER'})"
        )

    if table == "transactions":
        files = f"['{raw['transactions.csv']}', '{raw['transactions_v2.csv']}']"
        return (
            "SELECT DISTINCT t.msno, payment_method_id, payment_plan_days, plan_list_price, "
            "actual_amount_paid, is_auto_renew, "
            f"{_days('t.transaction_date')} AS transaction_date, "
            f"{_days('t.membership_expire_date')} AS membership_expire_date, is_cancel "
            f"FROM read_csv({files}, header=true, columns={{"
            "'msno': 'VARCHAR', 'payment_method_id': 'INTEGER', 'payment_plan_days': 'INTEGER', "
            "'plan_list_price': 'INTEGER', 'actual_amount_paid': 'INTEGER', 'is_auto_renew': 'INTEGER', "
            "'transaction_date': 'INTEGER', 'membership_expire_date': 'INTEGER', 'is_cancel': 'INTEGER'}) t "
            "WHERE try_strptime(CAST(t.transaction_date AS VARCHAR), '%Y%m%d') IS NOT NULL"
        )

    if table == "user_logs":
        files = f"['{raw['user_logs.csv']}', '{raw['user_logs_v2.csv']}']"
        log_start = LOG_START.replace("-", "")
        date_col = 'u."date"'  # quoted: date is also a type name
        return (
            f"SELECT u.msno, {_days(date_col)} AS log_date, num_25, num_50, num_75, num_985, num_100, num_unq, "
            "LEAST(GREATEST(total_secs, 0), 86400) AS total_secs "
            f"FROM read_csv({files}, header=true, columns={{"
            "'msno': 'VARCHAR', 'date': 'INTEGER', 'num_25': 'INTEGER', 'num_50': 'INTEGER', "
            "'num_75': 'INTEGER', 'num_985': 'INTEGER', 'num_100': 'INTEGER', 'num_unq': 'INTEGER', "
            "'total_secs': 'DOUBLE'}) u "
            f"WHERE {date_col} >= {log_start} AND u.msno IN (SELECT msno FROM label_users)"
        )

    raise ValueError(f"Unknown table '{table}'")


LOGS_COVERAGE_FILE = "user_logs_coverage.json"


def write_logs_coverage(data_dir: str, label_tables: list) -> None:
    """Record which label tables' users the trimmed user_logs covers."""
    with open(os.path.join(data_dir, "parquet", LOGS_COVERAGE_FILE), "w", encoding="utf-8") as f:
        json.dump({"label_tables": label_tables}, f, indent=2)


def logs_cover(data_dir: str, label_table: str, engine: str = "duckdb") -> bool:
    """Whether user_logs kept the listening history of this label table's users.

    The demo keeps every user. A real user_logs built before this file existed
    was trimmed to the official label files only.
    """
    if engine == "sqlite":
        return True
    path = os.path.join(data_dir, "parquet", LOGS_COVERAGE_FILE)
    if not os.path.exists(path):
        return label_table in {s.label_table for s in SNAPSHOTS.values() if s.source == "official"}
    with open(path, encoding="utf-8") as f:
        return label_table in json.load(f)["label_tables"]


def tables_to_build(data_dir: str, rebuild=None) -> list:
    """Tables that are missing, plus any named in `rebuild`, in dependency order."""
    rebuild = set(rebuild or [])
    unknown = rebuild - set(TABLE_ORDER)
    if unknown:
        raise ValueError(f"Unknown tables {sorted(unknown)}. Choose from {TABLE_ORDER}.")
    return [t for t in TABLE_ORDER if t in rebuild or not os.path.exists(parquet_path(data_dir, t))]


def convert_to_parquet(
    raw: dict, data_dir: str, LOG, memory_limit: str = "8GB", threads: int | None = None, build=None
) -> dict:
    """Convert raw CSVs into one Parquet file per processed table.

    `build` lists the tables to (re)write; the others are left as they are.
    Default: only the tables that do not exist yet.
    """
    build = tables_to_build(data_dir) if build is None else list(build)
    con = connect("duckdb", memory_limit=memory_limit, threads=threads, temp_dir=default_temp_dir())
    manifest = {}
    try:
        for table in TABLE_ORDER:
            out_path = parquet_path(data_dir, table)
            if table == "user_logs" and table in build:
                # The label user list is what trims the 30 GB listening log.
                # Official label files plus any rebuilt label tables (run --mode labels first to include them).
                label_tables = [
                    s.label_table for s in SNAPSHOTS.values() if os.path.exists(parquet_path(data_dir, s.label_table))
                ]
                con.execute(
                    "CREATE OR REPLACE TABLE label_users AS "
                    + " UNION ".join(
                        f"SELECT msno FROM read_parquet('{parquet_path(data_dir, t)}')" for t in label_tables
                    )
                )
                log_stage(
                    LOG,
                    "INFO",
                    COMPONENT,
                    label_users=scalar(con, "SELECT COUNT(*) FROM label_users"),
                    label_tables=",".join(label_tables),
                )

            if table in build:
                with component_stage(LOG, f"{COMPONENT}:{table}", out_path=out_path):
                    _copy(con, _table_sql(table, raw), out_path)
                if table == "user_logs":
                    write_logs_coverage(data_dir, label_tables)
            elif os.path.exists(out_path):
                log_stage(LOG, "INFO", COMPONENT, action="convert_skip", table=table, reason="parquet_exists")
            else:
                continue

            rows = scalar(con, f"SELECT COUNT(*) FROM read_parquet('{out_path}')")
            manifest[table] = {
                "path": out_path,
                "rows": int(rows),
                "size_mb": round(os.path.getsize(out_path) / 1e6, 1),
                "rebuilt": table in build,
            }
            log_stage(
                LOG, "ARTIFACT", COMPONENT, table=table, rows=rows, path=out_path, size_mb=manifest[table]["size_mb"]
            )
    finally:
        con.close()
    return manifest


def write_demo(frames: dict, engine: str, data_dir: str, LOG) -> dict:
    """Store synthetic frames exactly where real ingested data would live."""
    manifest = {}
    if engine == "sqlite":
        os.makedirs(data_dir, exist_ok=True)
        db_path = os.path.join(data_dir, SQLITE_FILENAME)
        con = connect("sqlite", db_path)
        try:
            for table in TABLE_COLUMNS:
                load_frame(con, table, frames[table])
                manifest[table] = {"path": db_path, "rows": int(len(frames[table]))}
                log_stage(LOG, "ARTIFACT", COMPONENT, table=table, rows=len(frames[table]), path=db_path)
        finally:
            con.close()
        return manifest

    con = connect("duckdb")
    try:
        for table in TABLE_COLUMNS:
            out_path = parquet_path(data_dir, table)
            load_frame(con, table, frames[table])
            _copy(con, f"SELECT * FROM {table}", out_path)
            manifest[table] = {"path": out_path, "rows": int(len(frames[table]))}
            log_stage(LOG, "ARTIFACT", COMPONENT, table=table, rows=len(frames[table]), path=out_path)
    finally:
        con.close()
    return manifest


def run_ingest(
    raw_dir: str,
    data_dir: str,
    run_dir: str,
    LOG,
    engine: str = "duckdb",
    demo: bool = False,
    demo_users: int = 5000,
    seed: int = 42,
    download: bool = True,
    delete_csv: bool = False,
    memory_limit: str = "8GB",
    threads: int | None = None,
    overwrite: bool = False,
    tables=None,
) -> dict:
    with component_stage(
        LOG, COMPONENT, demo=demo, engine=engine, raw_dir=None if demo else raw_dir, data_dir=data_dir
    ):
        if demo:
            from src.demo import generate_demo_frames

            frames = generate_demo_frames(n_users=demo_users, seed=seed, LOG=LOG)
            manifest = write_demo(frames, engine, data_dir, LOG)
        else:
            if engine != "duckdb":
                raise ValueError("Real KKBox data needs --engine duckdb (sqlite is for the demo only).")
            rebuild = list(tables or []) or (TABLE_ORDER if overwrite else [])
            build = tables_to_build(data_dir, rebuild)
            if not build:
                # Nothing to convert, so do not re-download ~8 GB of archives.
                log_stage(LOG, "INFO", COMPONENT, action="acquire_skip", reason="all_parquet_exist")
                raw = {}
            else:
                needed = list(dict.fromkeys(name for t in build for name in table_sources(t)))
                log_stage(LOG, "INFO", COMPONENT, tables_to_build=",".join(build), raw_files=",".join(needed))
                raw = acquire_raw(raw_dir, LOG, download=download, names=needed)
            manifest = convert_to_parquet(raw, data_dir, LOG, memory_limit=memory_limit, threads=threads, build=build)
            if delete_csv and raw:
                for path in raw.values():
                    os.remove(path)
                log_stage(LOG, "INFO", COMPONENT, action="delete_raw_csv", files=len(raw))

        manifest_path = os.path.join(run_dir, "artifacts", "ingest_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"demo": demo, "engine": engine, "tables": manifest}, f, indent=4)
        log_stage(LOG, "ARTIFACT", COMPONENT, ingest_manifest=manifest_path)
        return manifest
