#!/usr/bin/env python3
"""Export all marimo .py notebooks in content/ to .ipynb with rendered outputs.

Usage:
    python scripts/export_notebooks.py              # export all
    python scripts/export_notebooks.py content/GLM.py  # export one
"""

import subprocess
import sys
from pathlib import Path

CONTENT_DIR = Path(__file__).parent.parent / "content"


def is_marimo_notebook(path: Path) -> bool:
    """Check if a .py file is a marimo notebook (contains 'import marimo')."""
    try:
        with open(path) as f:
            content = f.read(1024)  # read first 1KB
        return "import marimo" in content
    except Exception:
        return False


def export_notebook(py_path: Path) -> bool:
    """Export a single marimo .py to .ipynb with outputs."""
    ipynb_path = py_path.with_suffix(".ipynb")
    print(f"Exporting {py_path.name} -> {ipynb_path.name}")

    result = subprocess.run(
        [
            sys.executable, "-m", "marimo", "export", "ipynb",
            str(py_path),
            "-o", str(ipynb_path),
            "--include-outputs",
            "--sort", "topological",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        # "cells failed to execute" still produces a valid ipynb with error outputs
        if ipynb_path.exists() and "some cells failed to execute" in stderr:
            print(f"  OK (some cells had execution errors — expected if data not available)")
            return True
        print(f"  FAILED: {stderr[-200:]}")
        return False

    print(f"  OK")
    return True


def main():
    if len(sys.argv) > 1:
        # Export specific files
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        # Export all marimo notebooks in content/
        targets = sorted(CONTENT_DIR.glob("*.py"))

    targets = [t for t in targets if is_marimo_notebook(t)]

    if not targets:
        print("No marimo notebooks found to export.")
        return

    print(f"Found {len(targets)} marimo notebook(s) to export.\n")

    results = {True: 0, False: 0}
    for path in targets:
        ok = export_notebook(path)
        results[ok] += 1

    print(f"\nDone: {results[True]} succeeded, {results[False]} failed.")
    if results[False] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
