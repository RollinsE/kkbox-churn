import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_entrypoints_exist():
    assert (ROOT / "main.py").exists()
    for module in (
        "logger",
        "console",
        "experiment",
        "ingest",
        "demo",
        "features",
        "model",
        "evaluate",
        "diagnostics",
        "notebook_viewer",
        "labeller",
    ):
        assert (ROOT / "src" / f"{module}.py").exists(), module


def test_documentation_files_exist():
    assert (ROOT / "README.md").exists()
    assert (ROOT / "docs" / "LOGGING.md").exists()
    assert (ROOT / "docs" / "MODEL_CARD.md").exists()
    assert (ROOT / "docs" / "DATA_CARD.md").exists()


def test_pipeline_code_logs_through_the_run_logger():
    """No bare print() in src/: everything must reach run.log."""
    offenders = []
    for path in sorted((ROOT / "src").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print":
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders


def test_notebook_is_in_sync_with_builder():
    import json
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import build_notebook

    on_disk = json.loads((ROOT / "notebooks" / "colab.ipynb").read_text(encoding="utf-8"))
    assert on_disk == build_notebook.build()
