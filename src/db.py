"""Thin database layer so the same feature SQL runs on DuckDB and SQLite.

DuckDB is the real engine (out-of-core, reads Parquet). SQLite ships with
Python, so tests and the synthetic demo can run anywhere with no extra install.
"""

import os
import sqlite3
import tempfile

import pandas as pd

from src.constants import SNAPSHOTS, TABLE_COLUMNS
from src.logger import log_stage

ENGINES = ("duckdb", "sqlite")
REBUILT_LABEL_TABLES = [s.label_table for s in SNAPSHOTS.values() if s.source == "rebuilt"]
SQLITE_FILENAME = "kkbox.sqlite"


def default_temp_dir() -> str:
    """Where DuckDB spills when a query outgrows memory.

    Always the local temp folder, never data_dir: data_dir may be a network or cloud-synced folder,
    which is far too slow for spilling.
    """
    return os.path.join(tempfile.gettempdir(), "kkbox_duckdb_tmp")


def engine_of(con) -> str:
    return "sqlite" if isinstance(con, sqlite3.Connection) else "duckdb"


def connect(
    engine: str = "duckdb",
    database: str = ":memory:",
    memory_limit: str | None = None,
    temp_dir: str | None = None,
    threads: int | None = None,
):
    if engine == "sqlite":
        return sqlite3.connect(database)
    if engine != "duckdb":
        raise ValueError(f"Unknown engine '{engine}'. Use one of {ENGINES}.")

    import duckdb

    con = duckdb.connect(database)
    if memory_limit:
        con.execute(f"SET memory_limit='{memory_limit}'")
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        con.execute(f"SET temp_directory='{temp_dir}'")
    if threads:
        con.execute(f"SET threads={int(threads)}")
    # Lets DuckDB stream large CSV -> Parquet conversions instead of buffering them.
    con.execute("SET preserve_insertion_order=false")
    return con


def table_exists(con, name: str) -> bool:
    if engine_of(con) == "sqlite":
        return (
            con.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?", (name,)).fetchone()
            is not None
        )
    return con.execute("SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]).fetchone() is not None


def query_df(con, sql: str) -> pd.DataFrame:
    if engine_of(con) == "sqlite":
        return pd.read_sql_query(sql, con)
    return con.execute(sql).df()


def scalar(con, sql: str):
    return con.execute(sql).fetchone()[0]


def load_frame(con, name: str, df: pd.DataFrame) -> None:
    """Create (or replace) table `name` from a DataFrame."""
    if engine_of(con) == "sqlite":
        df.to_sql(name, con, if_exists="replace", index=False)
        con.commit()
        return
    con.register("_frame_tmp", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _frame_tmp")
    con.unregister("_frame_tmp")


def parquet_path(data_dir: str, table: str) -> str:
    return os.path.join(data_dir, "parquet", f"{table}.parquet")


def open_source(engine: str, data_dir: str, LOG, memory_limit: str | None = None, threads: int | None = None):
    """Open the processed tables for querying.

    duckdb: in-memory connection with one view per Parquet file in <data_dir>/parquet.
    sqlite: the <data_dir>/kkbox.sqlite file written by the demo.
    """
    if engine == "sqlite":
        path = os.path.join(data_dir, SQLITE_FILENAME)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run --mode ingest --demo --engine sqlite first.")
        log_stage(LOG, "INFO", "Pipeline", action="open_source", engine=engine, database=path)
        return connect("sqlite", path)

    con = connect("duckdb", memory_limit=memory_limit, threads=threads, temp_dir=default_temp_dir())
    for table in TABLE_COLUMNS:
        path = parquet_path(data_dir, table)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run --mode ingest first.")
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{path}')")
    for table in REBUILT_LABEL_TABLES:  # written by --mode labels, if it has run
        path = parquet_path(data_dir, table)
        if os.path.exists(path):
            con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{path}')")
    log_stage(
        LOG,
        "INFO",
        "Pipeline",
        action="open_source",
        engine=engine,
        parquet_dir=os.path.join(data_dir, "parquet"),
        tables=len(TABLE_COLUMNS),
    )
    return con
