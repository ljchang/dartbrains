# CLAUDE.md

## Git

- Do not add Co-Authored-By lines or any Claude attribution in commit messages.

## Marimo notebooks — embedding images

Marimo's `mo.md()` does **not** resolve relative filesystem paths in markdown image syntax (`![alt](../images/foo/bar.png)`). The browser tries to fetch the path as a URL relative to the page origin, producing broken-image icons in the live editor. The same syntax works in the Jupyter Book 2 static build (MyST resolves relative to the source file), so code that looks fine in the rendered site can still be broken in `marimo edit`.

**Rule:** use `mo.image()` for local files, not markdown image syntax.

**Pattern** — add an `IMG_DIR` constant to the notebook's imports cell so paths are portable (independent of CWD when marimo launches):

```python
@app.cell(hide_code=True)
def _():
    import marimo as mo
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parent.parent
    IMG_DIR = _ROOT / "images" / "<section>"   # e.g. "preprocessing", "rsa", "glm"
    return IMG_DIR, mo
```

For prose-with-image flow, split the `mo.md` cell into an `mo.vstack`:

```python
@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.vstack([
        mo.md(r"""
        Prose before the figure...
        """),
        mo.image(str(IMG_DIR / "foo.png")),
        mo.md(r"""
        Prose after the figure.
        """),
    ])
    return
```

`mo.image()` works for PNG, JPG, GIF, and static SVGs. The marimo-book build pipeline runs `marimo export ipynb --include-outputs` per notebook before rendering to Markdown, so `mo.image()` outputs are baked into the static site.

### Animated SVGs (fmriprep QC reports, etc.)

SVGs with embedded `@keyframes` or `animation` CSS **do not animate** when rendered via `mo.image()` because the browser loads them through an `<img>` tag, which sandboxes CSS and scripts. To preserve animations, inline the SVG markup so it lands directly in the DOM:

```python
@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.Html(f'<div style="max-width:100%">{(IMG_DIR / "foo.svg").read_text()}</div>')
    return
```

**Important:** large inlined SVGs (>5 MB — fmriprep QC SVGs contain base64-embedded brain slices) will bust marimo's default 10 MB per-cell output limit when combined with other content in a vstack. Put each large animated SVG in its own cell to stay under the limit.
