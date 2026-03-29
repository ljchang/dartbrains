# Replace pybids with HuggingFace Dataset Access -- Design Spec

## Context

The DartBrains notebooks currently use pybids (`BIDSLayout`) to query the Pinel Localizer dataset stored in a local `../data/localizer` directory. This requires students to first download the data via DataLad, then use pybids to discover file paths. We want to replace pybids with direct HuggingFace Hub downloads from `dartbrains/localizer`, which simplifies setup (no DataLad install, no local BIDS directory needed) and works seamlessly on molab.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data access library | `huggingface_hub.hf_hub_download` | Downloads individual files by BIDS path with automatic caching. Maps 1:1 to pybids `layout.get()` patterns. |
| Helper module | `Code/data.py` | Centralizes BIDS path construction so notebooks stay clean. Thin wrapper — no custom caching. |
| TR handling | Hardcoded constant (2.4s) | Single-task dataset. Avoids downloading JSON for one number. |
| Subject list | Hardcoded list S01-S20 | Known fixed dataset. Avoids querying HF API for directory listing. |
| DataLad | Keep in Download_Data notebook | Legacy alternative still works. Add HuggingFace section above it. |
| pybids | Remove from dependencies | No longer needed after migration. |

## Architecture

### `Code/data.py`

```python
REPO_ID = "dartbrains/localizer"
SUBJECTS = [f"S{i:02d}" for i in range(1, 21)]
TR = 2.4
CONDITIONS = [
    "audio_computation", "audio_left_hand", "audio_right_hand", "audio_sentence",
    "horizontal_checkerboard", "vertical_checkerboard",
    "video_computation", "video_left_hand", "video_right_hand", "video_sentence",
]

def get_subjects() -> list[str]:
    return SUBJECTS

def get_tr() -> float:
    return TR

def get_file(subject: str, scope: str, suffix: str, extension: str = ".nii.gz") -> str:
    """Download a single file from the dartbrains/localizer dataset.

    Args:
        subject: e.g. "S01"
        scope: "raw", "derivatives", or "betas"
        suffix: BIDS suffix — "bold", "T1w", "events", "confounds"
        extension: file extension, default ".nii.gz"

    Returns:
        Local path to the cached file.
    """
    # Constructs the BIDS filename and calls hf_hub_download

def load_events(subject: str) -> pd.DataFrame:
    """Download and load the events TSV for a subject."""
    path = get_file(subject, scope="raw", suffix="events", extension=".tsv")
    return pd.read_csv(path, sep="\t")

def load_confounds(subject: str) -> pd.DataFrame:
    """Download and load the fmriprep confounds TSV for a subject."""
    path = get_file(subject, scope="derivatives", suffix="confounds", extension=".tsv")
    return pd.read_csv(path, sep="\t")
```

### BIDS Path Construction in `get_file`

The `get_file` function maps (subject, scope, suffix, extension) to the HF repo path:

| scope | suffix | HF repo path |
|-------|--------|-------------|
| `raw` | `bold` | `sub-{s}/func/sub-{s}_task-localizer_bold.nii.gz` |
| `raw` | `events` | `sub-{s}/func/sub-{s}_task-localizer_events.tsv` |
| `derivatives` | `bold` | `derivatives/fmriprep/sub-{s}/func/sub-{s}_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz` |
| `derivatives` | `T1w` | `derivatives/fmriprep/sub-{s}/anat/sub-{s}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz` |
| `derivatives` | `confounds` | `derivatives/fmriprep/sub-{s}/func/sub-{s}_task-localizer_desc-confounds_timeseries.tsv` |
| `betas` | `all` | `derivatives/betas/{s}_betas.nii.gz` |
| `betas` | `{condition}` | `derivatives/betas/{s}_beta_{condition}.nii.gz` |

Note: The exact filenames need to be verified against the actual HuggingFace repo contents. The paths above are based on standard fmriprep output naming conventions.

## Notebook Changes

### Multivariate_Prediction.py
- **Remove** pybids import and `layout = BIDSLayout(...)` — `layout` is never actually used in this notebook. Files are accessed via `glob` and `data_dir`.
- **Replace** `data_dir = '../data/localizer'` with `from Code.data import REPO_ID` and use `hf_hub_download` for the few direct file accesses, OR keep using `data_dir` if the files are already downloaded.

Actually, this notebook accesses beta files via `glob.glob(os.path.join(data_dir, 'derivatives', 'betas', ...))`. Replace with `get_file(subject, scope='betas', suffix='all')` or similar.

### RSA.py
- **Remove** pybids import and `layout = BIDSLayout(...)`
- **Replace** `layout.get_subjects(scope='derivatives')` with `get_subjects()`
- Beta files accessed via `glob` — replace with `get_file(subject, scope='betas', ...)`

### GLM_Single_Subject_Model.py
- **Replace** pybids imports with `from Code.data import get_file, get_tr, load_events, get_subjects`
- **Replace** `load_bids_events(layout, subject)` function:
  - `layout.get_tr()` → `get_tr()`
  - `layout.get(subject=subject, scope='raw', suffix='bold')[0].path` → `get_file(subject, 'raw', 'bold')`
  - `layout.get(subject=subject, suffix='events')[0].path` → `load_events(subject)` or `get_file(subject, 'raw', 'events', '.tsv')`
- **Replace** `layout.get(subject=sub, task='localizer', scope='derivatives', suffix='bold', ...)` → `get_file(sub, 'derivatives', 'bold')`

### Group_Analysis.py
- The pybids code is in a **markdown code block** (not executable) showing how betas were pre-computed. Update the code block to use the new API but it doesn't need to actually run.
- Beta files are loaded via paths like `f'../data/localizer/derivatives/betas/{sub}_betas.nii.gz'` — replace with `get_file(sub, 'betas', 'all')`

### Connectivity.py (heaviest user)
- **Replace** all `layout.get(...)` calls:
  - `layout.get(subject=sub, ..., scope='derivatives', suffix='bold', ...)` → `get_file(sub, 'derivatives', 'bold')`
  - `layout.get_tr()` → `get_tr()`
  - `layout.get(subject=sub, scope='derivatives', extension='.tsv')[0].path` → `get_file(sub, 'derivatives', 'confounds', '.tsv')`
  - `layout.get(subject='S01', scope='raw', suffix='bold')[0].path` → `get_file('S01', 'raw', 'bold')`
  - `layout.get(subject=subject, suffix='events')[0].path` → `get_file(subject, 'raw', 'events', '.tsv')`

### Introduction_to_Neuroimaging_Data.py
- This is the tutorial that teaches BIDS concepts using pybids. **Major rewrite needed:**
  - Remove BIDSLayout/pybids teaching section
  - Replace with teaching HuggingFace dataset access
  - Show `hf_hub_download` for individual files
  - Show `load_dataset("dartbrains/localizer", "betas")` for bulk access
  - Keep BIDS structure explanation (it's educational)

### Download_Data.py
- Add new HuggingFace section at the top (before DataLad)
- Show three loading methods from the HF dataset card
- Keep DataLad section as legacy alternative

### Introduction_to_Programming.py
- Add a new section at the top showing how to install Python with `uv` (above the existing conda tutorial)
- Cover: installing uv, creating a project, `uv sync`, `uv run`
- Keep existing conda section as legacy alternative

### ICA.py
- Text-only reference to pybids — no code change needed (already fixed to use markdown links)

## Dependencies

- **Remove:** `pybids` (was `bids` package) from `pyproject.toml`
- **Keep:** `datalad`, `huggingface-hub`, `datasets`
- **Add:** nothing new (huggingface-hub already in deps)
