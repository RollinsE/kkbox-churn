"""Lightweight repository and environment health checks for the KKBox churn pipeline."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = ["numpy", "pandas", "sklearn"]
REAL_DATA = ["duckdb", "pyarrow", "lightgbm"]  # needed for real runs; the sqlite demo works without them
MARATHON = ["xgboost", "catboost", "optuna", "plotly", "kaleido"]  # optional: skipped with a warning if missing


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def version_of(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", "installed")
    except Exception:
        return None


def main() -> int:
    checks = {
        "main.py": exists("main.py"),
        "src package": exists("src/__init__.py"),
        "logger": exists("src/logger.py"),
        "requirements": exists("requirements.txt"),
        "README": exists("README.md"),
        "gitignore": exists(".gitignore"),
    }

    print("KKBox Churn doctor")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project root: {ROOT}")

    for label, ok in checks.items():
        print(f"{label}: {'OK' if ok else 'MISSING'}")

    for module in REQUIRED:
        version = version_of(module)
        checks[f"import {module}"] = version is not None
        print(f"{module}: {version or 'MISSING'}")

    real_ready = True
    for module in REAL_DATA:
        version = version_of(module)
        real_ready = real_ready and version is not None
        print(f"{module}: {version or 'not installed (needed for real data)'}")

    for module in MARATHON:
        print(f"{module}: {version_of(module) or 'not installed (optional; that part is skipped)'}")

    has_7z = any(shutil.which(b) for b in ("7z", "7za", "7zr")) or version_of("py7zr") is not None
    has_kaggle_cli = shutil.which("kaggle") is not None
    has_kaggle_key = (
        bool(os.environ.get("KAGGLE_API_TOKEN"))
        or bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
        or (Path.home() / ".kaggle" / "kaggle.json").exists()
        or (Path.home() / ".kaggle" / "access_token").exists()
        or (Path.home() / ".config" / "kaggle" / "kaggle.json").exists()
    )
    print(f"7z extractor: {'present' if has_7z else 'not found'}")
    print(f"Kaggle CLI: {'present' if has_kaggle_cli else 'not found'}")
    print(f"Kaggle credentials: {'present' if has_kaggle_key else 'not set'}")
    print(f"Ready for demo: {'yes' if all(checks.values()) else 'no'}")
    print(f"Ready for real data: {'yes' if real_ready and has_7z and has_kaggle_cli and has_kaggle_key else 'no'}")

    missing = [label for label, ok in checks.items() if not ok]
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print("Doctor: FAIL")
        return 1

    print("Doctor: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
