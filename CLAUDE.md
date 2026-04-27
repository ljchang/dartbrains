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

## Marimo notebooks — matplotlib/seaborn figures

Marimo's matplotlib integration monkey-patches `matplotlib.artist.Artist._mime_` so a cell renders a figure **only if its last expression returns an `Artist`** (Figure, Axes, Text, Line2D, Legend, BarContainer, AxesImage, …). It is *not* an "auto-capture pyplot at end of cell" mechanism — pyplot side effects alone don't produce output. This rule is enforced by `marimo export ipynb --include-outputs`, which the marimo-book pipeline runs per notebook, so a notebook that looks fine in `marimo edit` (live kernel re-runs reactively) can still produce empty figure cells in the static build.

**Renders (last expr returns an Artist):** `plt.gcf()`, `plt.title("…")`, `ax.set_title("…")`, `ax.set_xlabel("…")`, `ax.legend([…])`, `ax.axvline(...)`, `sns.heatmap(...)`, `plt.imshow(...)`, `plt.bar(...)`, `plt.scatter(...)`.

**Silently empty (last expr returns `None`):** `plt.tight_layout()`, `plt.show()`, `plt.savefig(...)`, `plt.subplots_adjust(...)`, `plt.close(...)`, and any `with` block (the block itself returns `None`, regardless of what's inside — this includes `with sns.plotting_context(...)` and `with mo.persistent_cache(...)`).

**Idiom:** end the cell with `plt.gcf()` to return the current Figure. This is the safest fix when the natural last line is a None-returning op:

```python
@app.cell
def _(plt):
    _f, _a = plt.subplots(...)
    # … plotting …
    plt.tight_layout()
    plt.gcf()                  # ← required for the figure to render
    return
```

For cells whose plotting happens inside a styling context manager:

```python
@app.cell
def _(plt, sns):
    with sns.plotting_context(context='paper', font_scale=1.5):
        _f, _a = plt.subplots(...)
        # … plotting …
    plt.gcf()                  # ← outside the `with`; renders the figure created inside
    return
```

**`mo.persistent_cache` is incompatible with figure rendering.** It memoizes Python variables but not matplotlib side effects, so on a cache hit the plotting code does not re-run and `plt.gcf()` returns an empty figure. Either drop the cache and accept the cold-build cost (the marimo-book BuildCache will skip the notebook on subsequent builds anyway as long as the source is unchanged), or split the cell so the cache wraps only the data computation and plotting happens in a downstream cell from cached arrays.
