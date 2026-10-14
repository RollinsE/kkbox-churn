"""Read/write feature frames. Parquet by default; CSV where pyarrow is unavailable."""

import os

import pandas as pd

from src.constants import ID_COL

FRAME_FORMATS = {"parquet": ".parquet", "csv": ".csv.gz"}


def features_path(data_dir: str, snapshot: str, frame_format: str = "parquet") -> str:
    if frame_format not in FRAME_FORMATS:
        raise ValueError(f"Unknown frame_format '{frame_format}'. Use one of {sorted(FRAME_FORMATS)}.")
    return os.path.join(data_dir, "features", f"features_{snapshot}{FRAME_FORMATS[frame_format]}")


def write_frame(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def read_frame(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={ID_COL: str})
