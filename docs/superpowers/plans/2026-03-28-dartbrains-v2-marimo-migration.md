# DartBrains v2: Marimo Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate DartBrains from Jupyter Book 1 + ipynb notebooks to marimo notebooks + Jupyter Book 2, with molab links for interactive access and a static site as the primary reading experience.

**Architecture:** Marimo `.py` files are the source of truth in `content/`. A build script exports them to `.ipynb` with rendered outputs, then Jupyter Book 2 builds the static site. Each page includes an "Open in molab" button linking to `molab.marimo.io/github/...` for interactive access. Large fMRI data is hosted on HuggingFace and downloaded on demand.

**Tech Stack:** marimo, anywidget, Jupyter Book 2 (MyST-MD), uv, Python 3.13, Plotly, Three.js, GitHub Actions, GitHub Pages

**Spec:** `docs/superpowers/specs/2026-03-28-dartbrains-v2-marimo-migration-design.md`

---

## File Map

### New files to create

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Python project config & dependencies (uv) |
| `.python-version` | Pin Python 3.13 |
| `myst.yml` | Jupyter Book 2 config (replaces `_config.yml` + `_toc.yml`) |
| `scripts/export_notebooks.py` | Batch export marimo `.py` → `.ipynb --include-outputs` |
| `.github/workflows/deploy-book.yml` | Updated CI: export → JB2 build → deploy |
| `Code/mr_simulations.py` | Shared Bloch equation solver, tissue constants, Plotly helpers |
| `Code/mr_widgets.py` | anywidget wrapper classes for JS widgets |
| `Code/js/*.js` | 10 Three.js/Canvas 2D widget implementations |
| `content/MR_Physics_1_Magnetism_and_Resonance.py` | New marimo notebook |
| `content/MR_Physics_2_Signal_and_Contrast.py` | New marimo notebook |
| `content/MR_Physics_3_Imaging_and_fMRI.py` | New marimo notebook |
| `content/Preprocessing.py` | New marimo notebook (replaces `.ipynb`) |
| `content/*.py` (16 more) | Converted from existing `.ipynb` |

### Files to modify

| File | Change |
|------|--------|
| `.gitignore` | Add `_build/`, `.venv/`, `__pycache__/`, `content/*.ipynb` (generated), `uv.lock` entries |
| `README.md` | Update for new tech stack and contribution workflow |

### Files to remove (after migration complete)

| File | Reason |
|------|--------|
| `_config.yml` | Replaced by `myst.yml` |
| `_toc.yml` | Replaced by `myst.yml` |
| `requirements.txt` | Replaced by `pyproject.toml` |
| `content/Signal_Measurement.ipynb` | Replaced by MR Physics 1-3 |
| `content/Introduction_to_JupyterHub.ipynb` | No longer relevant (molab replaces JupyterHub) |
| `content/Introduction_to_Discovery.ipynb` | Evaluate — may not be relevant to general audience |

---

## Phase 1: Infrastructure & Proof of Concept

The goal of Phase 1 is to prove the full pipeline end-to-end: marimo `.py` → export to `.ipynb` → Jupyter Book 2 build → rendered site. We do this with a single simple notebook before migrating everything.

### Task 1: Set up Python project with uv

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Modify: `.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "dartbrains"
version = "2.0.0"
description = "An open-access introduction to functional neuroimaging analysis in Python"
readme = "README.md"
requires-python = ">=3.13"
license = "CC-BY-SA-4.0"
authors = [
    { name = "Luke Chang", email = "luke.j.chang@dartmouth.edu" },
]
dependencies = [
    "anywidget>=0.9.21",
    "datalad>=1.3.4",
    "datasets>=4.8.4",
    "huggingface-hub[cli,hf-xet]>=1.8.0",
    "marimo>=0.21.1",
    "matplotlib>=3.10.8",
    "nbformat",
    "nibabel>=5.4.2",
    "nilearn",
    "nltools",
    "numpy>=2.4.3",
    "pandas",
    "plotly>=6.6.0",
    "scikit-learn",
    "scipy>=1.17.1",
    "seaborn",
    "networkx",
]

[project.optional-dependencies]
build = [
    "jupyter-book",
]
```

- [ ] **Step 2: Create `.python-version`**

```
3.13
```

- [ ] **Step 3: Update `.gitignore`**

Add the following to the end of the existing `.gitignore`:

```gitignore

# Build artifacts #
###################
_build/

# Python environment #
######################
.venv/
__pycache__/
*.pyc

# Generated ipynb from marimo export #
#######################################
content/*.ipynb

# marimo cache #
################
__marimo__/
```

- [ ] **Step 4: Initialize uv and generate lock file**

Run: `uv lock`

Expected: `uv.lock` file created with resolved dependencies.

- [ ] **Step 5: Verify Python environment works**

Run: `uv run python -c "import marimo; print(marimo.__version__)"`

Expected: Prints marimo version (>=0.21.1).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version .gitignore
git commit -m "Set up uv project with Python 3.13 and dependencies"
```

Note: Do NOT commit `uv.lock` yet — it's large and we may want to decide on that later. Add it to `.gitignore` if desired.

---

### Task 2: Install Jupyter Book 2 and migrate config

**Files:**
- Create: `myst.yml`
- Keep (for now): `_config.yml`, `_toc.yml` (removed in final cleanup)

- [ ] **Step 1: Install Jupyter Book 2**

Run: `uv add jupyter-book --group build`

This installs the JB2 CLI (which uses the MyST-MD engine).

- [ ] **Step 2: Run the JB2 auto-migration tool**

Run: `uv run jupyter book` (run from repo root)

The JB2 CLI detects `_config.yml` + `_toc.yml` and offers to migrate them to `myst.yml`. Accept the migration. It will:
- Generate `myst.yml` from your existing config
- Back up originals as `_config.yml.myst.bak` and `_toc.yml.myst.bak`

- [ ] **Step 3: Review and refine `myst.yml`**

The auto-generated `myst.yml` will need manual adjustments. Ensure it includes:

```yaml
version: 1

project:
  title: DartBrains
  authors:
    - name: Luke Chang
  copyright: '2024'
  description: An open-access introduction to functional neuroimaging analysis in Python
  github: https://github.com/ljchang/dartbrains
  license: CC-BY-SA-4.0
  toc:
    - file: content/intro.md
    - title: Course Overview
      children:
        - file: content/Instructors.md
        - file: content/Syllabus.md
        - file: content/Schedule.md
    - title: Getting Started
      children:
        - file: content/Introduction_to_Programming.ipynb
        - file: content/Introduction_to_Pandas.ipynb
        - file: content/Introduction_to_Plotting.ipynb
        - file: content/Download_Data.ipynb
        - file: content/Glossary.ipynb
    - title: MR Physics & Imaging
      children:
        - file: content/MR_Physics_1_Magnetism_and_Resonance.ipynb
        - file: content/MR_Physics_2_Signal_and_Contrast.ipynb
        - file: content/MR_Physics_3_Imaging_and_fMRI.ipynb
    - title: Neuroimaging Analysis
      children:
        - file: content/Intro_to_Neuroimaging.md
        - file: content/Introduction_to_Neuroimaging_Data.ipynb
        - file: content/Signal_Processing.ipynb
        - file: content/Preprocessing.ipynb
        - file: content/GLM.ipynb
        - file: content/GLM_Single_Subject_Model.ipynb
        - file: content/Group_Analysis.ipynb
        - file: content/Thresholding_Group_Analyses.ipynb
    - title: Advanced Methods
      children:
        - file: content/Connectivity.ipynb
        - file: content/Introduction_to_ICA.ipynb
        - file: content/Multivariate_Prediction.ipynb
        - file: content/RSA.ipynb
        - file: content/Parcellations.ipynb
        - file: content/Resampling_Statistics.ipynb
    - title: Additional Resources
      children:
        - file: content/fmriprep_on_discovery.md
        - url: http://naturalistic-data.org/
          title: Naturalistic Data Analysis
    - title: Project Gallery
      children:
        - file: content/2019_Spring.md
        - file: content/2020_Spring.md
        - file: content/2020_Fall.md
        - file: content/2021_Fall.md
        - file: content/2022_Fall.md
    - title: Contributing
      children:
        - file: content/Contributing.md
        - url: https://github.com/ljchang/dartbrains
          title: GitHub Repository

site:
  template: book-theme
  options:
    logo: images/logo/dartbrains_logo_square_transparent.png
    favicon: images/logo/favicon.ico
```

**Important:** The TOC references `.ipynb` files — these will be the *exported* files generated from the marimo `.py` sources. The `.py` source files are NOT referenced in the TOC.

- [ ] **Step 4: Verify JB2 builds with existing content**

Run: `uv run jupyter book build --site`

Expected: Site builds successfully in `_build/`. There may be warnings about missing files (MR Physics notebooks don't exist yet) — that's fine for now. The existing `.ipynb` and `.md` files should render.

- [ ] **Step 5: Preview the site**

Run: `uv run jupyter book start`

Expected: Local dev server starts. Browse to verify the site renders correctly with existing content.

- [ ] **Step 6: Commit**

```bash
git add myst.yml
git commit -m "Add Jupyter Book 2 config (myst.yml)"
```

---

### Task 3: Create the notebook export script

**Files:**
- Create: `scripts/export_notebooks.py`

- [ ] **Step 1: Create `scripts/export_notebooks.py`**

```python
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
    """Check if a .py file is a marimo notebook (starts with 'import marimo')."""
    try:
        with open(path) as f:
            first_line = f.readline().strip()
        return first_line == "import marimo"
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
        print(f"  FAILED: {result.stderr.strip()}")
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
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/export_notebooks.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/export_notebooks.py
git commit -m "Add marimo-to-ipynb export script"
```

---

### Task 4: End-to-end proof of concept with one notebook

**Files:**
- Create: `content/Introduction_to_Programming.py` (converted from `.ipynb`)

This task proves the full pipeline works before we migrate everything.

- [ ] **Step 1: Convert one existing notebook to marimo format**

Pick the simplest notebook — `Introduction_to_Programming.ipynb` — since it has no fMRI dependencies.

Run: `uv run marimo convert content/Introduction_to_Programming.ipynb -o content/Introduction_to_Programming.py`

- [ ] **Step 2: Inspect the converted notebook**

Run: `uv run marimo edit content/Introduction_to_Programming.py`

Open in browser, verify cells are present and code looks correct. Close when satisfied.

- [ ] **Step 3: Export back to ipynb with outputs**

Run: `uv run python scripts/export_notebooks.py content/Introduction_to_Programming.py`

Expected: `content/Introduction_to_Programming.ipynb` is regenerated with outputs.

- [ ] **Step 4: Build the site with JB2**

Run: `uv run jupyter book build --site`

Expected: Site builds. The `Introduction_to_Programming` page should show rendered outputs.

- [ ] **Step 5: Preview and verify**

Run: `uv run jupyter book start`

Navigate to the Introduction to Programming page. Verify:
- Code cells render with syntax highlighting
- Outputs (text, plots) appear as static content
- Navigation works
- Page is readable

- [ ] **Step 6: Commit**

```bash
git add content/Introduction_to_Programming.py
git commit -m "POC: convert Introduction_to_Programming to marimo"
```

---

## Phase 2: Copy New Content from Dropbox

### Task 5: Copy shared Code modules and widgets

**Files:**
- Create: `Code/mr_simulations.py`
- Create: `Code/mr_widgets.py`
- Create: `Code/js/` (10 files)

- [ ] **Step 1: Copy the Code directory from Dropbox**

```bash
cp /Users/lukechang/Dropbox/Dartbrains/Code/mr_simulations.py Code/mr_simulations.py
cp /Users/lukechang/Dropbox/Dartbrains/Code/mr_widgets.py Code/mr_widgets.py
mkdir -p Code/js
cp /Users/lukechang/Dropbox/Dartbrains/Code/js/*.js Code/js/
```

- [ ] **Step 2: Verify the module imports work**

Run: `uv run python -c "from Code.mr_simulations import TISSUE_PROPERTIES; print(list(TISSUE_PROPERTIES.keys()))"`

Expected: `['1.5T', '3T']`

Run: `uv run python -c "from Code.mr_widgets import PrecessionWidget; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add Code/
git commit -m "Add shared MR simulation modules and anywidget JS widgets"
```

---

### Task 6: Copy MR Physics notebooks

**Files:**
- Create: `content/MR_Physics_1_Magnetism_and_Resonance.py`
- Create: `content/MR_Physics_2_Signal_and_Contrast.py`
- Create: `content/MR_Physics_3_Imaging_and_fMRI.py`

- [ ] **Step 1: Copy the three MR Physics notebooks from Dropbox**

```bash
cp /Users/lukechang/Dropbox/Dartbrains/JupyterHub_Notebooks/MR_Physics_1_Magnetism_and_Resonance.py content/
cp /Users/lukechang/Dropbox/Dartbrains/JupyterHub_Notebooks/MR_Physics_2_Signal_and_Contrast.py content/
cp /Users/lukechang/Dropbox/Dartbrains/JupyterHub_Notebooks/MR_Physics_3_Imaging_and_fMRI.py content/
```

- [ ] **Step 2: Verify each notebook runs in marimo**

Run each in turn and verify it opens without import errors:

```bash
uv run marimo edit content/MR_Physics_1_Magnetism_and_Resonance.py
uv run marimo edit content/MR_Physics_2_Signal_and_Contrast.py
uv run marimo edit content/MR_Physics_3_Imaging_and_fMRI.py
```

Check: Do the imports resolve? Do the widgets render? Fix any path issues (e.g., the notebooks may reference `../Code/` — update imports to match the repo layout).

- [ ] **Step 3: Fix import paths if needed**

The Dropbox notebooks may use relative imports like `sys.path.append("../Code")`. Update these to work from the repo root. The pattern should be:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("__file__").resolve().parent.parent))
from Code.mr_simulations import ...
from Code.mr_widgets import ...
```

Or if marimo supports it, use the `Code/` directory as a package.

- [ ] **Step 4: Export to ipynb and test build**

```bash
uv run python scripts/export_notebooks.py content/MR_Physics_1_Magnetism_and_Resonance.py
uv run jupyter book build --site
```

Verify the MR Physics 1 page renders in the site with static plot outputs.

- [ ] **Step 5: Commit**

```bash
git add content/MR_Physics_*.py
git commit -m "Add MR Physics 1-3 interactive notebooks from Dropbox draft"
```

---

### Task 7: Copy new Preprocessing notebook

**Files:**
- Create: `content/Preprocessing.py` (replaces existing `Preprocessing.ipynb`)

- [ ] **Step 1: Back up the existing Preprocessing notebook**

The existing `content/Preprocessing.ipynb` will be replaced. It's already in git history, so no explicit backup needed — but verify with `git log content/Preprocessing.ipynb` that it's tracked.

- [ ] **Step 2: Copy from Dropbox**

```bash
cp /Users/lukechang/Dropbox/Dartbrains/JupyterHub_Notebooks/Preprocessing.py content/Preprocessing.py
```

- [ ] **Step 3: Fix import paths and verify**

Same pattern as Task 6 Step 3. Then:

Run: `uv run marimo edit content/Preprocessing.py`

Verify: Opens, imports resolve, widgets (TransformCubeWidget, CostFunctionWidget, SmoothingWidget) render.

- [ ] **Step 4: Export and test build**

```bash
uv run python scripts/export_notebooks.py content/Preprocessing.py
uv run jupyter book build --site
```

- [ ] **Step 5: Commit**

```bash
git add content/Preprocessing.py
git commit -m "Add new interactive Preprocessing notebook with anywidgets"
```

---

## Phase 3: Convert Existing Notebooks

Each existing `.ipynb` needs to be converted to marimo `.py` format. The process is the same for each: convert → inspect → fix issues → verify export → commit.

### Task 8: Batch convert remaining notebooks to marimo

**Files:**
- Create: `content/*.py` for each existing `.ipynb` (15 notebooks)

The notebooks to convert (excluding `Introduction_to_Programming.py` done in Task 4 and `Preprocessing.py` done in Task 7):

1. `Introduction_to_Pandas.ipynb`
2. `Introduction_to_Plotting.ipynb`
3. `Introduction_to_Neuroimaging_Data.ipynb`
4. `Signal_Processing.ipynb`
5. `GLM.ipynb`
6. `GLM_Single_Subject_Model.ipynb`
7. `Group_Analysis.ipynb`
8. `Thresholding_Group_Analyses.ipynb`
9. `Connectivity.ipynb`
10. `Introduction_to_ICA.ipynb`
11. `ICA.ipynb`
12. `Multivariate_Prediction.ipynb`
13. `RSA.ipynb`
14. `Parcellations.ipynb`
15. `Resampling_Statistics.ipynb`
16. `Download_Data.ipynb`
17. `Glossary.ipynb`

- [ ] **Step 1: Batch convert all notebooks**

```bash
cd /Users/lukechang/Github/dartbrains
for nb in content/Introduction_to_Pandas.ipynb \
          content/Introduction_to_Plotting.ipynb \
          content/Introduction_to_Neuroimaging_Data.ipynb \
          content/Signal_Processing.ipynb \
          content/GLM.ipynb \
          content/GLM_Single_Subject_Model.ipynb \
          content/Group_Analysis.ipynb \
          content/Thresholding_Group_Analyses.ipynb \
          content/Connectivity.ipynb \
          content/Introduction_to_ICA.ipynb \
          content/ICA.ipynb \
          content/Multivariate_Prediction.ipynb \
          content/RSA.ipynb \
          content/Parcellations.ipynb \
          content/Resampling_Statistics.ipynb \
          content/Download_Data.ipynb \
          content/Glossary.ipynb; do
    outfile="${nb%.ipynb}.py"
    echo "Converting $nb -> $outfile"
    uv run marimo convert "$nb" -o "$outfile"
done
```

- [ ] **Step 2: Spot-check a few converted notebooks**

Open 3-4 of the converted notebooks in marimo to verify they look correct:

```bash
uv run marimo edit content/GLM.py
uv run marimo edit content/Introduction_to_Pandas.py
uv run marimo edit content/RSA.py
```

Check:
- Are all cells present?
- Is markdown content preserved?
- Are code cells intact?
- Do imports look correct?

Common issues to fix:
- IPython magics (`%matplotlib inline`, `!pip install`) will be commented out — remove or replace them
- `display()` calls may need updating for marimo
- `%time` / `%%timeit` magics need replacement

- [ ] **Step 3: Commit all conversions**

```bash
git add content/*.py
git commit -m "Convert all existing notebooks from ipynb to marimo format"
```

- [ ] **Step 4: Test batch export**

```bash
uv run python scripts/export_notebooks.py
```

Expected: All notebooks export successfully. Some may fail due to missing data files or packages — note which ones fail and investigate.

- [ ] **Step 5: Build full site**

```bash
uv run jupyter book build --site
```

Expected: Full site builds. Review in browser with `uv run jupyter book start`.

- [ ] **Step 6: Fix any export/build failures**

For notebooks that fail to export (likely the fMRI-heavy ones that need data downloads), either:
- Add data download cells at the top of the notebook
- Use `marimo export ipynb` without `--include-outputs` for those specific notebooks (they'll show code but no outputs)
- Fix import/dependency issues

Commit fixes:

```bash
git add content/*.py
git commit -m "Fix marimo conversion issues in migrated notebooks"
```

---

## Phase 4: Site Polish & Deployment

### Task 9: Add "Open in molab" buttons to each notebook page

**Files:**
- Modify: each `content/*.py` notebook

- [ ] **Step 1: Add a molab link cell to each marimo notebook**

At the top of each marimo `.py` notebook (after the title cell), add a cell with a molab badge/link. The pattern for each notebook:

```python
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Interactive version:** Run this notebook with live code and interactive widgets on
    > [molab](https://molab.marimo.io/github/ljchang/dartbrains/blob/v2-marimo-migration/content/{NOTEBOOK_NAME}.py)
    """)
    return
```

Replace `{NOTEBOOK_NAME}` with each notebook's filename (without extension). Once the branch merges to master, update the URL path from `v2-marimo-migration` to `master`.

- [ ] **Step 2: Verify links work**

Pick one notebook and test the molab URL in a browser. Verify:
- The notebook loads on molab
- Imports resolve
- Code cells are editable

- [ ] **Step 3: Commit**

```bash
git add content/*.py
git commit -m "Add 'Open in molab' links to all notebooks"
```

---

### Task 10: Update myst.yml TOC for final content structure

**Files:**
- Modify: `myst.yml`

- [ ] **Step 1: Update the TOC in `myst.yml`**

Now that all content is in place, update the TOC to reflect the final structure. Key changes:
- Add MR Physics section
- Remove `Signal_Measurement.ipynb` (replaced by MR Physics series)
- Remove `Introduction_to_JupyterHub.ipynb` (no longer relevant)
- Evaluate `Introduction_to_Discovery.ipynb` — remove if not relevant to general audience
- All notebook entries reference `.ipynb` (the exported files, not the `.py` sources)

See the TOC structure in `myst.yml` from Task 2 Step 3 for the target layout.

- [ ] **Step 2: Build and verify navigation**

```bash
uv run python scripts/export_notebooks.py
uv run jupyter book build --site
uv run jupyter book start
```

Walk through every page in the sidebar. Verify:
- All pages load
- Navigation order is correct
- Section groupings make sense
- No broken links

- [ ] **Step 3: Commit**

```bash
git add myst.yml
git commit -m "Update TOC for v2 content structure"
```

---

### Task 11: Update CI/CD workflow

**Files:**
- Modify: `.github/workflows/deploy-book.yml`

- [ ] **Step 1: Rewrite the deploy workflow**

Replace the contents of `.github/workflows/deploy-book.yml`:

```yaml
name: deploy-book

on:
  push:
    branches:
      - master

jobs:
  deploy-book:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        run: uv python install 3.13

      - name: Install dependencies
        run: uv sync --all-groups

      - name: Export marimo notebooks to ipynb
        run: uv run python scripts/export_notebooks.py

      - name: Build the book
        run: uv run jupyter book build --site

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./_build/html
          cname: dartbrains.org
```

**Note:** The `--include-outputs` flag in `export_notebooks.py` means the export step will actually *execute* each notebook. This requires all dependencies to be available. For notebooks that need fMRI data, the data download cells must be self-contained (downloading from HuggingFace).

If some notebooks are too resource-intensive for CI, modify `export_notebooks.py` to skip `--include-outputs` for those specific files (they'll show code without outputs on the static site).

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/deploy-book.yml
git commit -m "Update CI workflow for marimo + JB2 pipeline"
```

---

### Task 12: Clean up legacy files

**Files:**
- Remove: `_config.yml`, `_toc.yml`, `requirements.txt`
- Remove: `content/Signal_Measurement.ipynb`, `content/Introduction_to_JupyterHub.ipynb`
- Modify: `README.md`

- [ ] **Step 1: Remove legacy config files**

```bash
git rm _config.yml _toc.yml requirements.txt
git rm _config.yml.myst.bak _toc.yml.myst.bak 2>/dev/null || true
```

- [ ] **Step 2: Remove replaced notebooks**

```bash
git rm content/Signal_Measurement.ipynb
git rm content/Introduction_to_JupyterHub.ipynb
```

Decide whether to keep `Introduction_to_Discovery.ipynb`:
- If keeping: convert to marimo and add to TOC
- If removing: `git rm content/Introduction_to_Discovery.ipynb`

- [ ] **Step 3: Remove original .ipynb files that now have .py replacements**

Since `.gitignore` now ignores `content/*.ipynb` (they're generated), and the `.py` files are the source of truth:

```bash
git rm content/Introduction_to_Programming.ipynb
git rm content/Introduction_to_Pandas.ipynb
git rm content/Introduction_to_Plotting.ipynb
git rm content/Introduction_to_Neuroimaging_Data.ipynb
git rm content/Signal_Processing.ipynb
git rm content/Preprocessing.ipynb
git rm content/GLM.ipynb
git rm content/GLM_Single_Subject_Model.ipynb
git rm content/Group_Analysis.ipynb
git rm content/Thresholding_Group_Analyses.ipynb
git rm content/Connectivity.ipynb
git rm content/Introduction_to_ICA.ipynb
git rm content/ICA.ipynb
git rm content/Multivariate_Prediction.ipynb
git rm content/RSA.ipynb
git rm content/Parcellations.ipynb
git rm content/Resampling_Statistics.ipynb
git rm content/Download_Data.ipynb
git rm content/Glossary.ipynb
```

- [ ] **Step 4: Update README.md**

Update to reflect the new tech stack:

```markdown
[![DOI](https://zenodo.org/badge/171529794.svg#left)](https://zenodo.org/badge/latestdoi/171529794)
# DartBrains

An open-access introduction to functional neuroimaging analysis in Python.

**Website:** [dartbrains.org](https://dartbrains.org)

## About

DartBrains provides interactive tutorials covering fMRI data analysis, from MR physics fundamentals through advanced analysis techniques. All content is authored as [marimo](https://marimo.io/) notebooks with interactive visualizations.

## Reading the tutorials

**Browse online:** Visit [dartbrains.org](https://dartbrains.org) to read all tutorials with rendered outputs.

**Run interactively:** Each tutorial page has an "Open in molab" link to launch the notebook with live code, interactive widgets, and editable cells.

## Local development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

\`\`\`bash
# Clone the repo
git clone https://github.com/ljchang/dartbrains.git
cd dartbrains

# Install dependencies
uv sync

# Edit a notebook
uv run marimo edit content/MR_Physics_1_Magnetism_and_Resonance.py

# Build the static site
uv run python scripts/export_notebooks.py
uv run jupyter book build --site
\`\`\`

## License

All content is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Acknowledgements

Created by [Luke Chang](http://www.lukejchang.com/) and supported by NSF CAREER Award 1848370 and the [Dartmouth Center for the Advancement of Learning](https://dcal.dartmouth.edu/about/impact/experiential-learning).
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Remove legacy JB1 files and update README for v2"
```

---

## Phase 5: Verification

### Task 13: End-to-end verification

- [ ] **Step 1: Full export**

```bash
uv run python scripts/export_notebooks.py
```

All notebooks should export. Note any failures.

- [ ] **Step 2: Full build**

```bash
uv run jupyter book build --site
```

Build should complete without errors.

- [ ] **Step 3: Visual review**

```bash
uv run jupyter book start
```

Walk through every page on the site:
- [ ] All pages load without errors
- [ ] Code cells render with syntax highlighting
- [ ] Static outputs (plots, figures, tables) appear correctly
- [ ] Navigation/TOC is correct
- [ ] "Open in molab" links are present on each notebook page
- [ ] Markdown pages (Syllabus, Schedule, etc.) render correctly
- [ ] Images/logos display

- [ ] **Step 4: Test molab integration**

Test 3 representative molab links:
- One simple notebook (Introduction_to_Programming)
- One with widgets (MR_Physics_1)
- One with fMRI data (GLM or Preprocessing)

Verify each loads and runs on molab.

- [ ] **Step 5: Commit any final fixes**

```bash
git add -A
git commit -m "Final verification fixes"
```
