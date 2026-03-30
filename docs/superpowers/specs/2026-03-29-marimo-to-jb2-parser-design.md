# Marimo→JB2 Parser — Design Spec

## Context

Marimo notebooks (.py) are exported to .ipynb for Jupyter Book 2 rendering. Many marimo-specific patterns (widgets, layout, UI controls) don't render in JB2. This parser post-processes the exported .ipynb files to transform marimo patterns into JB2-compatible equivalents.

## Architecture

The parser is a set of transforms applied in `postprocess_ipynb()` in `scripts/export_notebooks.py`. Each transform reads and modifies the notebook's cell list. Transforms run sequentially on the cell list.

## Transforms

### 1. Anywidget cells → `{anywidget}` directive

**Detects:** Code cells containing `mo.ui.anywidget(WidgetClass(...))`

**Extracts:**
- Widget class name → maps to JS file path via lookup table
- Constructor kwargs → JSON model data

**Replaces with:** A markdown cell containing:
```markdown
:::{anywidget} /Code/js/{widget_file}.js
{"param1": value1, "param2": value2}
:::
```

**Widget class → JS file mapping:**
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
```

**Extraction approach:** Use regex to find `WidgetClass(key=value, ...)` patterns and parse kwargs. For simple literals (numbers, booleans, strings), convert directly to JSON. For expressions referencing slider values (e.g., `float(slider.value)`), use the slider's default value.

### 2. Slider/switch/dropdown cells feeding anywidgets → remove

**Detects:** Code cells that only define `mo.ui.slider(...)`, `mo.ui.switch(...)`, or `mo.ui.dropdown(...)` variables that are consumed by an anywidget cell.

**Action:** Add `remove-cell` tag. The widget's built-in JS controls replace these.

### 3. mo.accordion → MyST dropdown admonition

**Detects:** Code cells containing `mo.accordion({"Title": mo.md("""content""")})`

**Replaces with:** A markdown cell:
```markdown
:::{admonition} Title
:class: dropdown
content
:::
```

**Extraction:** Parse the dictionary key as the title, extract the markdown content from the `mo.md()` call.

### 4. mo.callout → MyST admonition

**Detects:** Code cells containing `mo.callout(mo.md("""content"""), kind="info")`

**Replaces with:** A markdown cell:
```markdown
:::{note}
content
:::
```

**Kind mapping:** `info` → `note`, `success` → `tip`, `warn`/`warning` → `warning`, `danger` → `danger`

### 5. mo.vstack/mo.hstack layout wrappers → unwrap

**Detects:** Code cells where `mo.vstack([...])` or `mo.hstack([...])` is the primary output.

**Action:** These are layout containers. In many cases the cell also contains other logic (widget creation, plot generation). The parser should NOT try to decompose these complex cells — instead, add `hide-input` tag and let the output (if any) render. For cells that are purely layout with no computation, add `remove-cell`.

### 6. Plotly renderer injection

**Detects:** Any notebook that imports `plotly`.

**Action:** Add to the first code cell (or insert a new hidden cell):
```python
import plotly.io as pio
pio.renderers.default = "notebook_connected+plotly_mimetype"
```

This ensures Plotly outputs contain `text/html` that JB2 renders as interactive plots.

### 7. Existing transforms (already implemented)

- Import cells → `remove-cell` tag
- `*Written by Author*` → strip from first markdown cell
- Molab link → compact HTML badge
- `hide_code` metadata → JB2 `hide-input` tag

## Parser Strategy

The parser processes cells in order. For each code cell:

1. Check if it's an anywidget instantiation → Transform 1
2. Check if it's a slider/switch/dropdown only → Transform 2
3. Check if it contains `mo.accordion` → Transform 3
4. Check if it contains `mo.callout` as the sole output → Transform 4
5. Check if it contains `mo.vstack`/`mo.hstack` as layout → Transform 5
6. Otherwise → apply `hide-input` if marimo `hide_code` is set

Transforms 1-4 replace code cells with markdown cells. Transform 5 tags cells. Transform 6 is the fallback.

## Limitations

- **Complex cells** mixing widgets + Plotly + layout can't be fully decomposed. The parser handles common patterns, not arbitrary Python.
- **Slider→Plotly reactivity** is lost. The static site shows the plot with default slider values.
- **`mo.md()` inside `mo.vstack`** — markdown rendered by Python won't appear as markdown in JB2. Only standalone `mo.md()` cells (which marimo exports as markdown cells) work.

## Testing

After parsing, verify:
1. `uv run npx --yes mystmd@latest build --site` builds without errors
2. Anywidget pages render widgets interactively
3. Plotly plots are interactive (zoom/pan/hover)
4. Accordion sections are collapsible
5. Callouts render as styled admonitions
6. No raw `mo.` Python code visible on any page
