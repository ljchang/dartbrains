# Replace pybids with HuggingFace Dataset Access — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace pybids-based data access with HuggingFace Hub downloads via a shared `Code/data.py` helper module across all DartBrains notebooks.

**Architecture:** A thin wrapper module (`Code/data.py`) maps BIDS-style queries to `hf_hub_download` calls against the `dartbrains/localizer` dataset. Each notebook replaces `from bids import BIDSLayout` + `layout.get(...)` with imports from `Code.data`. The Download_Data notebook gets a new HuggingFace section, and Introduction_to_Programming gets uv install instructions.

**Tech Stack:** huggingface_hub, pandas, nibabel (existing deps)

**Spec:** `docs/superpowers/specs/2026-03-28-replace-pybids-with-huggingface-design.md`

---

## File Map

### New files

| File | Responsibility |
|------|---------------|
| `Code/data.py` | HuggingFace dataset access helper — wraps `hf_hub_download` with BIDS path construction |

### Files to modify

| File | Change |
|------|--------|
| `content/Introduction_to_Neuroimaging_Data.py` | Major rewrite: keep BIDS education, replace pybids with Code.data |
| `content/GLM_Single_Subject_Model.py` | Replace pybids with Code.data |
| `content/Group_Analysis.py` | Update code block in markdown to use Code.data |
| `content/Connectivity.py` | Replace pybids with Code.data (heaviest user) |
| `content/Multivariate_Prediction.py` | Remove unused pybids import, use Code.data for beta access |
| `content/RSA.py` | Replace pybids with Code.data |
| `content/Download_Data.py` | Add HuggingFace section above DataLad |
| `content/Introduction_to_Programming.py` | Add uv install section above conda |
| `pyproject.toml` | Remove pybids dependency |

---

## Verified HuggingFace File Paths

These are the exact paths in the `dartbrains/localizer` dataset:

| Type | HF Path |
|------|---------|
| Events TSV | `sub-{s}/func/sub-{s}_task-localizer_events.tsv` |
| Preprocessed BOLD | `derivatives/fmriprep/sub-{s}/func/sub-{s}_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz` |
| T1w (MNI space) | `derivatives/fmriprep/sub-{s}/anat/sub-{s}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz` |
| T1w (native) | `derivatives/fmriprep/sub-{s}/anat/sub-{s}_desc-preproc_T1w.nii.gz` |
| Confounds | `derivatives/fmriprep/sub-{s}/func/sub-{s}_task-localizer_desc-confounds_regressors.tsv` |
| Brain mask | `derivatives/fmriprep/sub-{s}/func/sub-{s}_task-localizer_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz` |
| Bold ref | `derivatives/fmriprep/sub-{s}/func/sub-{s}_task-localizer_space-MNI152NLin2009cAsym_boldref.nii.gz` |
| All betas | `derivatives/betas/{s}_betas.nii.gz` |
| Condition beta | `derivatives/betas/{s}_beta_{condition}.nii.gz` |

**Note:** No raw BOLD files exist in the HF repo. Notebooks that used raw bold (for n_tr, etc.) should use preprocessed bold instead.

---

### Task 1: Create `Code/data.py`

**Files:**
- Create: `Code/data.py`

- [ ] **Step 1: Create `Code/data.py`**

```python
"""
DartBrains Localizer Dataset Access
====================================

Helper functions to download and access the Pinel Localizer dataset
from HuggingFace Hub (dartbrains/localizer).

Files are downloaded on first access and cached locally by huggingface_hub.
"""

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "dartbrains/localizer"

SUBJECTS = [f"S{i:02d}" for i in range(1, 21)]

TR = 2.4  # seconds, from task-localizer_bold.json

CONDITIONS = [
    "audio_computation",
    "audio_left_hand",
    "audio_right_hand",
    "audio_sentence",
    "horizontal_checkerboard",
    "vertical_checkerboard",
    "video_computation",
    "video_left_hand",
    "video_right_hand",
    "video_sentence",
]


def _download(filename: str) -> str:
    """Download a file from the dartbrains/localizer dataset. Returns local cached path."""
    return hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")


def get_subjects() -> list[str]:
    """Return list of subject IDs (S01-S20)."""
    return list(SUBJECTS)


def get_tr() -> float:
    """Return the repetition time in seconds."""
    return TR


def get_file(subject: str, scope: str, suffix: str, extension: str = ".nii.gz") -> str:
    """Download and return the local path to a dataset file.

    Args:
        subject: Subject ID, e.g. "S01"
        scope: One of "raw", "derivatives", or "betas"
        suffix: BIDS suffix — "bold", "T1w", "events", "confounds", "mask",
                or a condition name for betas (e.g. "audio_computation"),
                or "all" for the stacked betas file
        extension: File extension including dot, e.g. ".nii.gz", ".tsv"

    Returns:
        Local filesystem path to the cached file.
    """
    s = subject  # e.g. "S01"
    sub = f"sub-{s}"  # e.g. "sub-S01"

    if scope == "betas":
        if suffix == "all":
            filename = f"derivatives/betas/{s}_betas{extension}"
        else:
            filename = f"derivatives/betas/{s}_beta_{suffix}{extension}"

    elif scope == "raw":
        if suffix == "events":
            filename = f"{sub}/func/{sub}_task-localizer_events.tsv"
        elif suffix == "bold":
            # No raw bold on HF — fall back to preprocessed
            filename = f"derivatives/fmriprep/{sub}/func/{sub}_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold{extension}"
        else:
            raise ValueError(f"Unknown raw suffix: {suffix}")

    elif scope == "derivatives":
        if suffix == "bold":
            filename = f"derivatives/fmriprep/{sub}/func/{sub}_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold{extension}"
        elif suffix == "T1w":
            filename = f"derivatives/fmriprep/{sub}/anat/{sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w{extension}"
        elif suffix == "confounds":
            filename = f"derivatives/fmriprep/{sub}/func/{sub}_task-localizer_desc-confounds_regressors.tsv"
        elif suffix == "mask":
            filename = f"derivatives/fmriprep/{sub}/func/{sub}_task-localizer_space-MNI152NLin2009cAsym_desc-brain_mask{extension}"
        else:
            raise ValueError(f"Unknown derivatives suffix: {suffix}")
    else:
        raise ValueError(f"Unknown scope: {scope}. Use 'raw', 'derivatives', or 'betas'.")

    return _download(filename)


def load_events(subject: str) -> pd.DataFrame:
    """Download and load the events TSV for a subject as a DataFrame."""
    path = get_file(subject, scope="raw", suffix="events", extension=".tsv")
    return pd.read_csv(path, sep="\t")


def load_confounds(subject: str) -> pd.DataFrame:
    """Download and load the fmriprep confounds TSV for a subject."""
    path = get_file(subject, scope="derivatives", suffix="confounds")
    return pd.read_csv(path, sep="\t")
```

- [ ] **Step 2: Verify module imports**

Run: `uv run python -c "from Code.data import get_subjects, get_tr, REPO_ID; print(get_subjects()[:3], get_tr())"`

Expected: `['S01', 'S02', 'S03'] 2.4`

- [ ] **Step 3: Verify download works**

Run: `uv run python -c "from Code.data import get_file; print(get_file('S01', 'betas', 'all'))"`

Expected: prints a local cache path like `~/.cache/huggingface/hub/datasets--dartbrains--localizer/.../S01_betas.nii.gz`

- [ ] **Step 4: Commit**

```bash
git add Code/data.py
git commit -m "Add HuggingFace dataset access helper (Code/data.py)"
```

---

### Task 2: Update GLM_Single_Subject_Model.py

**Files:**
- Modify: `content/GLM_Single_Subject_Model.py`

- [ ] **Step 1: Read the file and locate pybids usage**

The imports cell (~line 60-91) has:
```python
from bids import BIDSLayout, BIDSValidator
data_dir = '../data/localizer'
layout = BIDSLayout(data_dir, derivatives=True)
```

The `load_bids_events` function (~line 102-114) uses:
- `layout.get_tr()`
- `layout.get(subject=subject, scope='raw', suffix='bold')[0].path`
- `layout.get(subject=subject, suffix='events')[0].path`

The data loading cell (~line 382-385) uses:
- `layout.get(subject=sub, task='localizer', scope='derivatives', suffix='bold', extension='nii.gz', return_type='file')`

- [ ] **Step 2: Replace the imports cell**

Remove:
```python
from bids import BIDSLayout, BIDSValidator
data_dir = '../data/localizer'
layout = BIDSLayout(data_dir, derivatives=True)
```

Add:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("__file__").resolve().parent.parent))
from Code.data import get_file, get_tr, load_events, get_subjects
```

Remove `layout` from the cell's return tuple. Add `get_file`, `get_tr`, `load_events`, `get_subjects` to return tuple.

- [ ] **Step 3: Replace `load_bids_events` function**

Replace:
```python
def load_bids_events(layout, subject):
    '''Create a design_matrix instance from BIDS event file'''
    tr = layout.get_tr()
    n_tr = nib.load(layout.get(subject=subject, scope='raw', suffix='bold')[0].path).shape[-1]
    onsets = pd.read_csv(layout.get(subject=subject, suffix='events')[0].path, sep='\t')
    onsets.columns = ['Onset', 'Duration', 'Stim']
    return onsets_to_dm(onsets, sampling_freq=1/tr, run_length=n_tr)

dm = load_bids_events(layout, 'S01')
```

With:
```python
def load_bids_events(subject):
    '''Create a design_matrix instance from BIDS event file'''
    tr = get_tr()
    n_tr = nib.load(get_file(subject, 'derivatives', 'bold')).shape[-1]
    onsets = load_events(subject)
    onsets.columns = ['Onset', 'Duration', 'Stim']
    return onsets_to_dm(onsets, sampling_freq=1/tr, run_length=n_tr)

dm = load_bids_events('S01')
```

Note: function signature changes from `(layout, subject)` to `(subject)`. Search for ALL calls to this function in the file and update them.

- [ ] **Step 4: Replace data loading**

Replace:
```python
data = Brain_Data(layout.get(subject=sub, task='localizer', scope='derivatives', suffix='bold', extension='nii.gz', return_type='file'))
```

With:
```python
data = Brain_Data(get_file(sub, 'derivatives', 'bold'))
```

- [ ] **Step 5: Search for any remaining `layout` references and replace them**

Grep for `layout` in the file. Replace any remaining references.

- [ ] **Step 6: Commit**

```bash
git add content/GLM_Single_Subject_Model.py
git commit -m "Replace pybids with HuggingFace access in GLM_Single_Subject_Model"
```

---

### Task 3: Update Connectivity.py (heaviest user)

**Files:**
- Modify: `content/Connectivity.py`

- [ ] **Step 1: Read the file and locate all pybids usage**

Imports cell (~line 84-123):
```python
from bids import BIDSLayout, BIDSValidator
base_dir = '..'
data_dir = os.path.join(base_dir, 'data', 'localizer')
layout = BIDSLayout(data_dir, derivatives=True)
```

Usage locations (from grep):
- Line 145: `layout.get(subject=sub, task='localizer', scope='derivatives', suffix='bold', extension='nii.gz', return_type='file')[0]`
- Line 243: `layout.get_tr()`
- Line 257: `layout.get(subject=sub, scope='derivatives', extension='.tsv')[0].path`
- Line 298: `layout.get(subject='S01', scope='raw', suffix='bold')[0].path`
- Line 317: `layout.get_tr()`
- Line 318: `layout.get(subject=subject, scope='raw', suffix='bold')[0].path`
- Line 319: `layout.get(subject=subject, suffix='events')[0].path`
- Line 509: `layout.get(subject=sub, scope='derivatives', extension='.tsv')[0].path`
- Line 546: `layout.get_tr()`

- [ ] **Step 2: Replace imports cell**

Remove `from bids import BIDSLayout, BIDSValidator` and `layout = BIDSLayout(...)`.
Add:
```python
from Code.data import get_file, get_tr, load_events, load_confounds, get_subjects
```

Remove `layout` from return tuple, add the new imports.

- [ ] **Step 3: Replace each `layout` call**

For each occurrence, apply these replacements:

| Old | New |
|-----|-----|
| `layout.get(subject=sub, ..., scope='derivatives', suffix='bold', extension='nii.gz', return_type='file')[0]` | `get_file(sub, 'derivatives', 'bold')` |
| `layout.get_tr()` | `get_tr()` |
| `layout.get(subject=sub, scope='derivatives', extension='.tsv')[0].path` | `load_confounds(sub)` or `get_file(sub, 'derivatives', 'confounds')` depending on whether a path or DataFrame is needed |
| `layout.get(subject='S01', scope='raw', suffix='bold')[0].path` | `get_file('S01', 'derivatives', 'bold')` (no raw bold on HF) |
| `layout.get(subject=subject, suffix='events')[0].path` | `get_file(subject, 'raw', 'events', '.tsv')` |

For confounds: where the code does `pd.read_csv(layout.get(...)[0].path, sep='\t')`, replace with `load_confounds(sub)`.

Where the code does `nib.load(layout.get(...)[0].path)`, use `nib.load(get_file(...))`.

- [ ] **Step 4: Update `load_bids_events` function** (same pattern as Task 2)

- [ ] **Step 5: Verify no remaining `layout` references**

- [ ] **Step 6: Commit**

```bash
git add content/Connectivity.py
git commit -m "Replace pybids with HuggingFace access in Connectivity"
```

---

### Task 4: Update RSA.py and Multivariate_Prediction.py

**Files:**
- Modify: `content/RSA.py`
- Modify: `content/Multivariate_Prediction.py`

- [ ] **Step 1: Update RSA.py**

Replace imports (~line 108-146):
- Remove `from bids import BIDSLayout, BIDSValidator` and `layout = BIDSLayout(data_dir, derivatives=True)`
- Add `from Code.data import get_subjects, get_file`
- Remove `layout` from return tuple

Replace `layout.get_subjects(scope='derivatives')` (~line 381) with `get_subjects()`.

Where beta files are accessed via `glob.glob(os.path.join(data_dir, 'derivatives', 'betas', f'{sub}*'))`, replace with explicit `get_file` calls:
```python
# Old: file_list = glob.glob(os.path.join(data_dir, 'derivatives', 'betas', f'{sub}*'))
# New:
from Code.data import CONDITIONS
file_list = [get_file(sub, 'betas', cond) for cond in CONDITIONS]
```

- [ ] **Step 2: Update Multivariate_Prediction.py**

Replace imports (~line 119-143):
- Remove `from bids import BIDSLayout, BIDSValidator` and `layout = BIDSLayout(data_dir, derivatives=True)`
- Add `from Code.data import get_subjects, get_file, CONDITIONS`
- Remove `layout` from return tuple

Replace any `glob.glob(os.path.join(data_dir, 'derivatives', 'betas', ...))` with `get_file` calls.

- [ ] **Step 3: Commit**

```bash
git add content/RSA.py content/Multivariate_Prediction.py
git commit -m "Replace pybids with HuggingFace access in RSA and Multivariate_Prediction"
```

---

### Task 5: Update Group_Analysis.py

**Files:**
- Modify: `content/Group_Analysis.py`

- [ ] **Step 1: Read the file and locate pybids usage**

The pybids code in this notebook is inside a **markdown code block** (~line 335-394) showing how betas were pre-computed. It is NOT executable code — it's documentation.

The notebook then loads betas via direct paths like:
```python
f'../data/localizer/derivatives/betas/{sub}_betas.nii.gz'
```

- [ ] **Step 2: Update the markdown code block**

Replace the code block content to use the new API. Since this is markdown (documentation of how betas were generated), update it to show the new approach:

Replace references to `BIDSLayout` and `layout.get(...)` with `from Code.data import ...` equivalents.

- [ ] **Step 3: Replace direct path references**

Search for `../data/localizer/derivatives/betas/` and replace with `get_file(sub, 'betas', ...)` calls. Add the Code.data import to the appropriate cell.

- [ ] **Step 4: Commit**

```bash
git add content/Group_Analysis.py
git commit -m "Replace pybids with HuggingFace access in Group_Analysis"
```

---

### Task 6: Update Introduction_to_Neuroimaging_Data.py

**Files:**
- Modify: `content/Introduction_to_Neuroimaging_Data.py`

This is the biggest rewrite. The notebook currently teaches BIDS concepts using pybids to explore the dataset.

- [ ] **Step 1: Read the full notebook to understand the current flow**

Key sections to identify:
- BIDS explanation (keep and expand)
- pybids BIDSLayout tutorial (replace with HuggingFace)
- Loading NIfTI files (keep, update paths)
- Exploring metadata (keep, update access method)

- [ ] **Step 2: Rewrite the BIDS introduction section**

Keep and expand the educational content about BIDS:
- What BIDS is and why it matters
- Directory structure: `sub-XX/anat/`, `sub-XX/func/`, `derivatives/`
- Naming convention: `sub-S01_task-localizer_bold.nii.gz`
- Key files: `_events.tsv`, `_bold.json`, `dataset_description.json`, `participants.tsv`
- Show the actual dartbrains/localizer structure on HuggingFace as example

- [ ] **Step 3: Replace BIDSLayout section with HuggingFace access**

Remove the cell that does:
```python
from bids import BIDSLayout, BIDSValidator
layout = BIDSLayout(data_dir, derivatives=True)
```

Replace with:
```python
from Code.data import get_file, get_subjects, get_tr, load_events, load_confounds, REPO_ID
from huggingface_hub import hf_hub_download
```

Replace `layout.get(...)` queries with `get_file(...)` calls. Show how `hf_hub_download` works underneath for educational purposes.

- [ ] **Step 4: Replace all remaining `layout` references throughout the notebook**

Apply same replacement patterns as Tasks 2-4. Use `get_file('S01', 'derivatives', 'T1w')` for the T1 loading section, etc.

- [ ] **Step 5: Commit**

```bash
git add content/Introduction_to_Neuroimaging_Data.py
git commit -m "Rewrite Neuroimaging Data tutorial: BIDS education + HuggingFace access"
```

---

### Task 7: Add HuggingFace section to Download_Data.py

**Files:**
- Modify: `content/Download_Data.py`

- [ ] **Step 1: Read the current notebook**

The notebook currently teaches DataLad-based downloading from `gin.g-node.org/ljchang/Localizer`.

- [ ] **Step 2: Add a new HuggingFace section ABOVE the DataLad section**

After the introduction/title cells, add new cells covering:

**Cell: Introduction to the dataset on HuggingFace**
```python
mo.md(r"""
## Downloading Data from HuggingFace (Recommended)

The Pinel Localizer dataset is hosted on [HuggingFace](https://huggingface.co/datasets/dartbrains/localizer). This is the recommended way to access the data — files are downloaded automatically and cached locally.

### Quick Start

The `Code.data` module provides helper functions that handle downloads for you:
""")
```

**Cell: Show Code.data usage**
```python
from Code.data import get_file, get_subjects, load_events

# Download a preprocessed BOLD file (cached after first download)
bold_path = get_file('S01', 'derivatives', 'bold')
print(f"BOLD file: {bold_path}")

# Load events for a subject
events = load_events('S01')
events.head()
```

**Cell: Show direct hf_hub_download**
```python
mo.md(r"""
### Direct Download

You can also download files directly using `hf_hub_download`:
""")
```

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="dartbrains/localizer",
    filename="derivatives/betas/S01_betas.nii.gz",
    repo_type="dataset",
)
print(path)
```

**Cell: Show datasets library for bulk access**
```python
mo.md(r"""
### Bulk Loading with the datasets Library

For loading all beta maps or events at once:
""")
```

```python
from datasets import load_dataset

# Load all beta maps
ds = load_dataset("dartbrains/localizer", "betas")
print(f"Loaded {len(ds['train'])} beta maps")
print(ds['train'][0])
```

- [ ] **Step 3: Add a heading before the DataLad section**

Add a markdown cell:
```python
mo.md(r"""
## Downloading Data with DataLad (Legacy)

The dataset is also available via DataLad from the GIN repository. This is the original download method and still works.
""")
```

- [ ] **Step 4: Commit**

```bash
git add content/Download_Data.py
git commit -m "Add HuggingFace download section to Download_Data notebook"
```

---

### Task 8: Add uv install section to Introduction_to_Programming.py

**Files:**
- Modify: `content/Introduction_to_Programming.py`

- [ ] **Step 1: Read the current notebook to find the installation/setup section**

Look for cells covering Python installation, conda, or environment setup.

- [ ] **Step 2: Add a new section about installing Python with uv ABOVE any conda content**

Add cells covering:

**Cell: Markdown intro**
```python
mo.md(r"""
## Installing Python with uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that makes it easy to install Python and manage project dependencies.

### Install uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Set up the DartBrains project

```bash
# Clone the repository
git clone https://github.com/ljchang/dartbrains.git
cd dartbrains

# Install Python and all dependencies
uv sync

# Run a notebook
uv run marimo edit content/Introduction_to_Programming.py
```

`uv sync` reads the `pyproject.toml` file and installs the correct Python version and all required packages automatically.
""")
```

- [ ] **Step 3: Add a heading before existing conda/pip content**

Add a markdown cell:
```python
mo.md(r"""
## Installing Python with Conda (Legacy)

If you prefer using Conda for package management, the following instructions still work.
""")
```

- [ ] **Step 4: Commit**

```bash
git add content/Introduction_to_Programming.py
git commit -m "Add uv install instructions to Introduction_to_Programming"
```

---

### Task 9: Remove pybids from dependencies and re-export

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Remove pybids from pyproject.toml**

The `pybids` package is listed as the `bids` import. Check if it's listed in dependencies. It may not be explicitly listed (could be a transitive dependency of nltools). Search for `pybids` or `bids` in `pyproject.toml`.

If not in pyproject.toml, check if any notebook still imports `from bids import`. Grep across all `.py` files:
```bash
grep -r "from bids import" content/
```

There should be zero matches after Tasks 2-6. If any remain, fix them.

- [ ] **Step 2: Re-export all notebooks**

```bash
uv run python scripts/export_notebooks.py
```

All 22 should export successfully.

- [ ] **Step 3: Rebuild the site**

```bash
uv run jupyter book build --site
```

Should build with zero warnings (or only the minor ones from before).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "Remove pybids dependency"
```
