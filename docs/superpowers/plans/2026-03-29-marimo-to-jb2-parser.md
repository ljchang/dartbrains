# Marimo→JB2 Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the ipynb post-processor to transform marimo-specific patterns (anywidgets, accordions, callouts, layout) into JB2-compatible equivalents so the static site renders interactive widgets, collapsible sections, and styled admonitions.

**Architecture:** The parser extends `postprocess_ipynb()` in `scripts/export_notebooks.py` with a series of cell-level transforms. Each transform detects a pattern in a code cell's source, then either replaces the cell with a markdown cell or modifies its tags. Transforms run sequentially on the cell list.

**Tech Stack:** Python, json, re (stdlib only — no external parsing deps)

**Spec:** `docs/superpowers/specs/2026-03-29-marimo-to-jb2-parser-design.md`

---

## File Map

| File | Change |
|------|--------|
| `scripts/export_notebooks.py` | Add transform functions to `postprocess_ipynb()` |

This is a single-file change. All transforms are functions called from the existing `postprocess_ipynb()`.

---

### Task 1: Anywidget cell → `{anywidget}` directive

**Files:**
- Modify: `scripts/export_notebooks.py`

- [ ] **Step 1: Add the widget class → JS file mapping**

Add this constant near the top of the file:

```python
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

# Default model values for each widget (used when slider.value refs can't be resolved)
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
```

- [ ] **Step 2: Add the `transform_anywidget_cells` function**

```python
def transform_anywidget_cells(cells: list) -> list:
    """Replace code cells containing mo.ui.anywidget(Widget(...)) with {anywidget} markdown directives."""
    new_cells = []
    for cell in cells:
        if cell["cell_type"] != "code":
            new_cells.append(cell)
            continue

        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])

        # Check if this cell contains mo.ui.anywidget
        if "mo.ui.anywidget" not in src:
            new_cells.append(cell)
            continue

        # Find the widget class being instantiated
        # Pattern: _widget = WidgetClass(...) or WidgetClass(...)
        widget_match = None
        for widget_name, js_file in WIDGET_JS_MAP.items():
            if widget_name in src:
                widget_match = (widget_name, js_file)
                break

        if not widget_match:
            new_cells.append(cell)
            continue

        widget_name, js_file = widget_match

        # Extract kwargs from the constructor call
        # Try to parse literal values; fall back to defaults
        model = dict(WIDGET_DEFAULTS.get(widget_name, {}))

        # Try to extract explicit literal kwargs: key=3.0, key=True, key="string"
        constructor_pattern = re.compile(
            rf'{widget_name}\s*\((.*?)\)',
            re.DOTALL
        )
        constructor_match = constructor_pattern.search(src)
        if constructor_match:
            args_str = constructor_match.group(1)
            # Parse key=value pairs with literal values
            for kv_match in re.finditer(r'(\w+)\s*=\s*([^,\)]+)', args_str):
                key = kv_match.group(1)
                val_str = kv_match.group(2).strip()
                # Try to evaluate simple literals
                try:
                    if val_str in ('True', 'False'):
                        model[key] = val_str == 'True'
                    elif val_str.replace('.', '').replace('-', '').isdigit():
                        model[key] = float(val_str) if '.' in val_str else int(val_str)
                    elif val_str.startswith('"') or val_str.startswith("'"):
                        model[key] = val_str.strip("\"'")
                    # Skip expressions like float(slider.value) — use default
                except (ValueError, SyntaxError):
                    pass

        # Build the {anywidget} directive markdown cell
        model_json = json.dumps(model, indent=2)
        directive_md = f':::{"{"}anywidget{"}"} /Code/js/{js_file}\n{model_json}\n:::'

        md_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": directive_md,
        }
        new_cells.append(md_cell)
        continue

    return new_cells
```

- [ ] **Step 3: Wire it into `postprocess_ipynb`**

In the `postprocess_ipynb` function, add after the existing transforms:

```python
    # Transform anywidget cells into {anywidget} directives
    nb["cells"] = transform_anywidget_cells(nb["cells"])
```

- [ ] **Step 4: Test with MR Physics 1**

```bash
uv run python scripts/export_notebooks.py content/MR_Physics_1_Magnetism_and_Resonance.py
python3 -c "
import json
with open('content/MR_Physics_1_Magnetism_and_Resonance.ipynb') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    if 'anywidget' in src.lower():
        print(f'Cell {i}: type={cell[\"cell_type\"]}')
        print(src[:200])
        print()
"
```

Expected: Code cells with `mo.ui.anywidget` are replaced with markdown cells containing `:::{anywidget}` directives.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_notebooks.py
git commit -m "Parser: transform anywidget cells to MyST {anywidget} directives"
```

---

### Task 2: Accordion cells → MyST dropdown admonitions

**Files:**
- Modify: `scripts/export_notebooks.py`

- [ ] **Step 1: Add the `transform_accordion_cells` function**

```python
def transform_accordion_cells(cells: list) -> list:
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

        # Extract the title from the dictionary key
        # Pattern: "Title": mo.md( or 'Title': mo.md(
        title_match = re.search(r'["\']([^"\']+)["\']\s*:\s*mo\.md\s*\(', src)
        if not title_match:
            new_cells.append(cell)
            continue

        title = title_match.group(1)

        # Extract the markdown content from inside mo.md(r""" ... """)
        # Look for triple-quoted string after mo.md(
        content_match = re.search(
            r'mo\.md\s*\(\s*r?"""(.*?)"""',
            src,
            re.DOTALL
        )
        if not content_match:
            content_match = re.search(
                r"mo\.md\s*\(\s*r?'''(.*?)'''",
                src,
                re.DOTALL
            )

        if not content_match:
            new_cells.append(cell)
            continue

        content = content_match.group(1).strip()
        # Remove leading whitespace that comes from Python indentation
        lines = content.split('\n')
        if lines:
            # Find minimum indentation (ignoring empty lines)
            indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
            min_indent = min(indents) if indents else 0
            lines = [line[min_indent:] if len(line) >= min_indent else line for line in lines]
            content = '\n'.join(lines)

        md_source = f':::{{admonition}} {title}\n:class: dropdown\n\n{content}\n:::'

        md_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": md_source,
        }
        new_cells.append(md_cell)

    return new_cells
```

- [ ] **Step 2: Wire into `postprocess_ipynb`**

```python
    nb["cells"] = transform_accordion_cells(nb["cells"])
```

- [ ] **Step 3: Test**

```bash
uv run python scripts/export_notebooks.py content/MR_Physics_1_Magnetism_and_Resonance.py
python3 -c "
import json
with open('content/MR_Physics_1_Magnetism_and_Resonance.ipynb') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    if 'admonition' in src or 'dropdown' in src:
        print(f'Cell {i}: type={cell[\"cell_type\"]}')
        print(src[:300])
        print()
"
```

Expected: `mo.accordion` code cells are replaced with markdown cells containing `:::{admonition}` with `:class: dropdown`.

- [ ] **Step 4: Commit**

```bash
git add scripts/export_notebooks.py
git commit -m "Parser: transform mo.accordion to MyST dropdown admonitions"
```

---

### Task 3: Callout cells → MyST admonitions

**Files:**
- Modify: `scripts/export_notebooks.py`

- [ ] **Step 1: Add the `transform_callout_cells` function**

```python
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


def transform_callout_cells(cells: list) -> list:
    """Replace code cells that are purely mo.callout(...) with MyST admonitions."""
    new_cells = []
    for cell in cells:
        if cell["cell_type"] != "code":
            new_cells.append(cell)
            continue

        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])

        # Only transform cells where mo.callout is the sole/primary expression
        # Skip cells that mix callout with other logic (vstack, etc.)
        if "mo.callout" not in src or "mo.vstack" in src:
            new_cells.append(cell)
            continue

        # Extract kind
        kind_match = re.search(r'kind\s*=\s*["\'](\w+)["\']', src)
        kind = kind_match.group(1) if kind_match else "note"
        myst_kind = CALLOUT_KIND_MAP.get(kind, "note")

        # Extract markdown content
        content_match = re.search(
            r'mo\.md\s*\(\s*r?"""(.*?)"""',
            src,
            re.DOTALL
        )
        if not content_match:
            content_match = re.search(
                r'mo\.md\s*\(\s*r?["\'](.+?)["\']',
                src,
                re.DOTALL
            )

        if not content_match:
            new_cells.append(cell)
            continue

        content = content_match.group(1).strip()
        # Dedent
        lines = content.split('\n')
        if lines:
            indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
            min_indent = min(indents) if indents else 0
            lines = [line[min_indent:] if len(line) >= min_indent else line for line in lines]
            content = '\n'.join(lines)

        md_source = f':::{{{myst_kind}}}\n{content}\n:::'
        md_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": md_source,
        }
        new_cells.append(md_cell)

    return new_cells
```

- [ ] **Step 2: Wire into `postprocess_ipynb`**

```python
    nb["cells"] = transform_callout_cells(nb["cells"])
```

- [ ] **Step 3: Commit**

```bash
git add scripts/export_notebooks.py
git commit -m "Parser: transform mo.callout to MyST admonitions"
```

---

### Task 4: Plotly renderer injection

**Files:**
- Modify: `scripts/export_notebooks.py`

- [ ] **Step 1: Add the `inject_plotly_renderer` function**

```python
def inject_plotly_renderer(cells: list) -> list:
    """If any cell imports plotly, inject the notebook_connected renderer setting."""
    has_plotly = any(
        "import plotly" in (c["source"] if isinstance(c["source"], str) else "".join(c["source"]))
        for c in cells
        if c["cell_type"] == "code"
    )

    if not has_plotly:
        return cells

    # Find the first code cell and prepend the renderer config
    for cell in cells:
        if cell["cell_type"] == "code":
            src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            if "import plotly" in src:
                renderer_line = '\nimport plotly.io as _pio\n_pio.renderers.default = "notebook_connected+plotly_mimetype"\n'
                cell["source"] = src + renderer_line
                break

    return cells
```

- [ ] **Step 2: Wire into `postprocess_ipynb`**

```python
    nb["cells"] = inject_plotly_renderer(nb["cells"])
```

- [ ] **Step 3: Commit**

```bash
git add scripts/export_notebooks.py
git commit -m "Parser: inject Plotly notebook_connected renderer for interactive plots"
```

---

### Task 5: Slider-only cells → remove-cell

**Files:**
- Modify: `scripts/export_notebooks.py`

- [ ] **Step 1: Add the `tag_slider_only_cells` function**

```python
def tag_slider_only_cells(cells: list) -> list:
    """Tag cells that only define mo.ui.slider/switch/dropdown with remove-cell if their widget was already converted to an {anywidget} directive."""
    # First, check which widget classes were converted (have {anywidget} directives)
    converted_widgets = set()
    for cell in cells:
        if cell["cell_type"] == "markdown":
            src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            if "{anywidget}" in src:
                for widget_name in WIDGET_JS_MAP:
                    js_file = WIDGET_JS_MAP[widget_name]
                    if js_file in src:
                        converted_widgets.add(widget_name)

    if not converted_widgets:
        return cells

    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])

        # Check if cell only defines sliders/switches/dropdowns (no other logic)
        lines = [l.strip() for l in src.strip().split('\n') if l.strip() and not l.strip().startswith('#')]
        is_slider_only = all(
            any(p in line for p in ['mo.ui.slider', 'mo.ui.switch', 'mo.ui.dropdown', ')', 'label=', 'start=', 'stop=', 'step=', 'value=', 'full_width=', 'options='])
            for line in lines
        )

        if is_slider_only and any(p in src for p in ['mo.ui.slider', 'mo.ui.switch', 'mo.ui.dropdown']):
            tags = set(cell.get("metadata", {}).get("tags", []))
            tags.add("remove-cell")
            cell.setdefault("metadata", {})["tags"] = sorted(tags)

    return cells
```

- [ ] **Step 2: Wire into `postprocess_ipynb`**

```python
    nb["cells"] = tag_slider_only_cells(nb["cells"])
```

- [ ] **Step 3: Commit**

```bash
git add scripts/export_notebooks.py
git commit -m "Parser: hide slider-only cells when their widgets are converted to directives"
```

---

### Task 6: Full integration test

- [ ] **Step 1: Re-export all notebooks**

```bash
uv run python scripts/export_notebooks.py
```

Expected: 23 succeeded, 0 failed.

- [ ] **Step 2: Build with mystmd 1.8.2**

```bash
uv run npx --yes mystmd@latest build --site
```

Expected: 35+ pages built, no errors.

- [ ] **Step 3: Start dev server and visually verify**

```bash
uv run npx --yes mystmd@latest start
```

Check:
- MR Physics 1: compass, net magnetization, precession widgets render interactively
- MR Physics 1: "Deep Dive" sections are collapsible dropdowns
- MR Physics 2: precession widget with relaxation renders
- Plotly plots (if any) show as interactive HTML
- No raw `mo.accordion`, `mo.callout`, or `mo.ui.anywidget` code visible

- [ ] **Step 4: Commit any fixes**

```bash
git add scripts/export_notebooks.py
git commit -m "Parser integration fixes"
```
