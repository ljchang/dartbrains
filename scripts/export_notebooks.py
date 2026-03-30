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

# Widget class name → JS file mapping for {anywidget} directive
WIDGET_JS_MAP = {
    "CompassWidget": "compass_widget.js",
    "NetMagnetizationWidget": "net_magnetization_widget.js",
    "PrecessionWidget": "precession_widget.js",
    "SpinEnsembleWidget": "spin_ensemble_widget.js",
    "KSpaceWidget": "kspace_widget.js",
    "EncodingWidget": "encoding_widget.js",
    "ConvolutionWidget": "convolution_widget.js",
    "TransformCubeWidget": "transform_cube_widget.js",
    "CostFunctionWidget": "cost_function_widget.js",
    "SmoothingWidget": "smoothing_widget.js",
}

WIDGET_DEFAULTS = {
    "CompassWidget": {"b0": 3.0},
    "NetMagnetizationWidget": {"n_protons": 100, "b0_on": False},
    "PrecessionWidget": {"b0": 3.0, "flip_angle": 90.0, "t1": 0.0, "t2": 0.0, "show_relaxation": False, "paused": False},
    "SpinEnsembleWidget": {"sequence_type": "spin_echo", "speed": 1.0},
    "KSpaceWidget": {"mask_type": "full", "radius_fraction": 1.0, "speed": 1.0},
    "EncodingWidget": {"speed": 1.0},
    "ConvolutionWidget": {"pattern": "block", "speed": 1.0},
    "TransformCubeWidget": {"tx": 0, "ty": 0, "tz": 0, "rot_x": 0, "rot_y": 0, "rot_z": 0, "scale_x": 1.0, "scale_y": 1.0, "scale_z": 1.0},
    "CostFunctionWidget": {"trans_x": 0, "trans_y": 0},
    "SmoothingWidget": {"fwhm": 0},
}

CALLOUT_KIND_MAP = {
    "info": "note",
    "note": "note",
    "success": "tip",
    "tip": "tip",
    "warn": "warning",
    "warning": "warning",
    "danger": "danger",
    "error": "danger",
}


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


def transform_anywidget_cells(cells):
    """Replace code cells containing mo.ui.anywidget(Widget(...)) with {anywidget} markdown directives."""
    new_cells = []
    for cell in cells:
        if cell["cell_type"] != "code":
            new_cells.append(cell)
            continue

        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])

        if "mo.ui.anywidget" not in src:
            new_cells.append(cell)
            continue

        # Find the widget class being instantiated
        widget_match = None
        for widget_name, js_file in WIDGET_JS_MAP.items():
            if widget_name in src:
                widget_match = (widget_name, js_file)
                break

        if not widget_match:
            new_cells.append(cell)
            continue

        widget_name, js_file = widget_match
        model = dict(WIDGET_DEFAULTS.get(widget_name, {}))

        # Try to extract explicit literal kwargs from constructor
        constructor_pattern = re.compile(rf'{widget_name}\s*\((.*?)\)', re.DOTALL)
        constructor_match = constructor_pattern.search(src)
        if constructor_match:
            args_str = constructor_match.group(1)
            for kv_match in re.finditer(r'(\w+)\s*=\s*([^,\)]+)', args_str):
                key = kv_match.group(1)
                val_str = kv_match.group(2).strip()
                try:
                    if val_str in ('True', 'False'):
                        model[key] = val_str == 'True'
                    elif val_str.replace('.', '').replace('-', '').isdigit():
                        model[key] = float(val_str) if '.' in val_str else int(val_str)
                    elif val_str.startswith('"') or val_str.startswith("'"):
                        model[key] = val_str.strip("\"'")
                except (ValueError, SyntaxError):
                    pass

        model_json = json.dumps(model, indent=2)
        directive_md = f':::{{anywidget}} /Code/js/{js_file}\n{model_json}\n:::'

        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": directive_md,
        })

    return new_cells


def transform_accordion_cells(cells):
    """Replace code cells containing mo.accordion({...}) with MyST dropdown admonitions."""
    new_cells = []
    for cell in cells:
        if cell["cell_type"] != "code":
            new_cells.append(cell)
            continue

        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])

        if "mo.accordion" not in src:
            new_cells.append(cell)
            continue

        title_match = re.search(r'["\']([^"\']+)["\']\s*:\s*mo\.md\s*\(', src)
        if not title_match:
            new_cells.append(cell)
            continue

        title = title_match.group(1)

        content_match = re.search(r'mo\.md\s*\(\s*r?"""(.*?)"""', src, re.DOTALL)
        if not content_match:
            content_match = re.search(r"mo\.md\s*\(\s*r?'''(.*?)'''", src, re.DOTALL)
        if not content_match:
            new_cells.append(cell)
            continue

        content = content_match.group(1).strip()
        lines = content.split('\n')
        if lines:
            indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
            min_indent = min(indents) if indents else 0
            lines = [line[min_indent:] if len(line) >= min_indent else line for line in lines]
            content = '\n'.join(lines)

        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": f':::{{admonition}} {title}\n:class: dropdown\n\n{content}\n:::',
        })

    return new_cells


def _extract_callout_md(callout_block):
    """Extract a MyST admonition from a mo.callout(...) block string.

    Returns a markdown string like ':::{note}\ncontent\n:::' or None if parsing fails.
    """
    # Extract kind
    kind_match = re.search(r'kind\s*=\s*["\'](\w+)["\']', callout_block)
    kind = kind_match.group(1) if kind_match else "note"
    myst_kind = CALLOUT_KIND_MAP.get(kind, "note")

    # Try triple-quoted content first: mo.md(r"""...""") or mo.md("""...""")
    content_match = re.search(r'mo\.md\s*\(\s*r?"""(.*?)"""', callout_block, re.DOTALL)
    if not content_match:
        content_match = re.search(r"mo\.md\s*\(\s*r?'''(.*?)'''", callout_block, re.DOTALL)

    if content_match:
        content = content_match.group(1).strip()
    else:
        # Try concatenated strings: mo.md("line1" "line2" f"line3")
        # Find all string literals inside the mo.md(...) call
        md_start = callout_block.find("mo.md(")
        if md_start == -1:
            return None
        # Extract all quoted strings after mo.md(
        strings = re.findall(r'[f]?"([^"]*)"', callout_block[md_start:])
        if not strings:
            strings = re.findall(r"[f]?'([^']*)'", callout_block[md_start:])
        if not strings:
            return None
        content = "".join(strings)

    # Dedent
    lines = content.split('\n')
    if lines:
        indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        min_indent = min(indents) if indents else 0
        lines = [line[min_indent:] if len(line) >= min_indent else line for line in lines]
        content = '\n'.join(lines)

    # Clean up f-string artifacts (e.g., {_b0:.1f} → use defaults)
    content = re.sub(r'\{[^}]+\}', '...', content) if '{_' in content or '{_' in content else content

    return f':::{{{myst_kind}}}\n{content}\n:::'


def transform_callout_cells(cells):
    """Extract mo.callout(...) from code cells and emit as MyST admonitions.

    Handles both:
    - Standalone callout cells (callout is the only expression)
    - Embedded callouts inside mo.vstack/mo.hstack layouts
    """
    new_cells = []
    for cell in cells:
        if cell["cell_type"] != "code":
            new_cells.append(cell)
            continue

        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])

        if "mo.callout" not in src:
            new_cells.append(cell)
            continue

        # Find all mo.callout(...) blocks in the cell
        # Use a bracket-matching approach to extract each callout
        callout_mds = []
        idx = 0
        while True:
            start = src.find("mo.callout(", idx)
            if start == -1:
                break
            # Find matching closing paren by counting brackets
            depth = 0
            end = start
            for j in range(start, len(src)):
                if src[j] == '(':
                    depth += 1
                elif src[j] == ')':
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            callout_block = src[start:end]
            md = _extract_callout_md(callout_block)
            if md:
                callout_mds.append(md)
            idx = end

        if not callout_mds:
            new_cells.append(cell)
            continue

        # Check if cell is ONLY a callout (no other logic)
        is_standalone = "mo.vstack" not in src and "mo.hstack" not in src

        if is_standalone:
            # Replace the entire cell with admonition(s)
            for md in callout_mds:
                new_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": md,
                })
        else:
            # Keep the original cell (it has other content like widgets/plots)
            new_cells.append(cell)
            # Emit callouts as separate markdown cells after it
            for md in callout_mds:
                new_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": md,
                })

    return new_cells


def inject_plotly_renderer(cells):
    """If any cell imports plotly, inject the notebook_connected renderer setting."""
    has_plotly = any(
        "import plotly" in (c["source"] if isinstance(c["source"], str) else "".join(c["source"]))
        for c in cells if c["cell_type"] == "code"
    )
    if not has_plotly:
        return cells

    for cell in cells:
        if cell["cell_type"] == "code":
            src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            if "import plotly" in src:
                renderer_line = '\nimport plotly.io as _pio\n_pio.renderers.default = "notebook_connected+plotly_mimetype"\n'
                cell["source"] = src + renderer_line
                break

    return cells


def tag_slider_only_cells(cells):
    """Tag cells that only define mo.ui.slider/switch/dropdown with remove-cell
    if their associated widget was converted to an {anywidget} directive."""
    has_converted_widgets = any(
        "{anywidget}" in (c["source"] if isinstance(c["source"], str) else "".join(c["source"]))
        for c in cells if c["cell_type"] == "markdown"
    )
    if not has_converted_widgets:
        return cells

    slider_patterns = ['mo.ui.slider', 'mo.ui.switch', 'mo.ui.dropdown']

    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])

        if not any(p in src for p in slider_patterns):
            continue

        # Check if cell ONLY defines UI controls (no other substantial logic)
        lines = [l.strip() for l in src.strip().split('\n') if l.strip() and not l.strip().startswith('#')]
        is_ui_only = all(
            any(p in line for p in slider_patterns + [
                ')', '(', 'label=', 'start=', 'stop=', 'step=', 'value=',
                'full_width=', 'options=', '=', '_slider', '_toggle', '_select',
                '_dropdown', '_switch',
            ])
            for line in lines
        )

        if is_ui_only:
            tags = set(cell.get("metadata", {}).get("tags", []))
            tags.add("remove-cell")
            cell.setdefault("metadata", {})["tags"] = sorted(tags)

    return cells


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

    # Remove *Written by Author* from first markdown cell
    # JB2 shows author in frontmatter, so the attribution line duplicates it
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            lines = source.split("\n")
            new_lines = []
            for line in lines:
                # Skip *Written by ...* author lines
                if re.match(r"^\*Written by .+\*$", line.strip()):
                    continue
                # Skip *Written By ...* (case variant)
                if re.match(r"^\*Written By.+\*$", line.strip()):
                    continue
                new_lines.append(line)
            # Remove any resulting double blank lines
            cleaned = "\n".join(new_lines)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            if cleaned != source:
                cell["source"] = cleaned
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

    # Apply marimo→JB2 transforms
    # Callouts first — extract from vstack cells BEFORE anywidget replaces them
    nb["cells"] = transform_callout_cells(nb["cells"])
    nb["cells"] = transform_anywidget_cells(nb["cells"])
    nb["cells"] = transform_accordion_cells(nb["cells"])
    nb["cells"] = inject_plotly_renderer(nb["cells"])
    nb["cells"] = tag_slider_only_cells(nb["cells"])

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
