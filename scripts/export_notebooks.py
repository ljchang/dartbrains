#!/usr/bin/env python3
"""Export all marimo .py notebooks in content/ to .ipynb with rendered outputs.

Usage:
    python scripts/export_notebooks.py              # export all
    python scripts/export_notebooks.py content/GLM.py  # export one
"""

import json
import re
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

    # Post-process the exported ipynb for JB2 compatibility
    postprocess_ipynb(ipynb_path)
    return True


def postprocess_ipynb(ipynb_path: Path):
    """Fix exported ipynb for Jupyter Book 2 rendering.

    - Translate marimo hide_code metadata to JB2 'hide-input' tags
    - Remove duplicate title/author from first markdown cell (JB2 shows these in frontmatter)
    - Convert molab blockquote link to a badge-style HTML
    """
    with open(ipynb_path) as f:
        nb = json.load(f)

    for i, cell in enumerate(nb["cells"]):
        meta = cell.get("metadata", {})
        marimo_config = meta.get("marimo", {}).get("config", {})
        tags = set(meta.get("tags", []))

        # Translate marimo hide_code to JB2 tags
        if marimo_config.get("hide_code"):
            if cell["cell_type"] == "code":
                # First code cell (imports) should be fully hidden
                if i == 0:
                    tags.add("remove-cell")
                else:
                    tags.add("hide-input")

        if tags:
            cell.setdefault("metadata", {})["tags"] = sorted(tags)

    # Remove duplicate title from first markdown cell
    # JB2 extracts the # heading for frontmatter, so having it in the body duplicates it
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            # Strip leading # Title and *Written by Author* lines
            lines = source.split("\n")
            new_lines = []
            skip_until_content = True
            for line in lines:
                if skip_until_content:
                    # Skip the # Title line
                    if re.match(r"^#\s+", line):
                        continue
                    # Skip empty lines after title
                    if line.strip() == "":
                        continue
                    # Skip *Written by ...* author attribution
                    if re.match(r"^\*Written by .+\*$", line.strip()):
                        continue
                    skip_until_content = False
                new_lines.append(line)
            if new_lines != lines:
                cell["source"] = "\n".join(new_lines)
            break  # Only process the first markdown cell

    # Convert molab blockquote to a compact badge-style link
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            if "Open this notebook in molab" in source:
                # Extract the URL
                match = re.search(r'\[Open this notebook in molab\]\((https://[^)]+)\)', source)
                if match:
                    url = match.group(1)
                    cell["source"] = (
                        f'<a href="{url}" target="_blank" '
                        f'style="display:inline-flex;align-items:center;gap:6px;'
                        f'padding:6px 14px;border-radius:6px;background:#f0f0f0;'
                        f'color:#333;text-decoration:none;font-size:14px;'
                        f'border:1px solid #ddd;margin:8px 0;">'
                        f'\U0001f680 <strong>Open in molab</strong> — '
                        f'run code &amp; interact with widgets</a>'
                    )

    with open(ipynb_path, "w") as f:
        json.dump(nb, f, indent=1)


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
